"""미국 시장: Yahoo Finance 재무·시세 + 나스닥 종목 목록 → site/data/us.json (국내 mf.json과 같은 구조)

SEC EDGAR는 GitHub Actions 서버(클라우드 IP)의 접속을 막아서 쓸 수 없음 → Yahoo 재무(fundamentals-timeseries) 사용
- 대상: 나스닥 종목 목록 API의 NYSE·나스닥 상장사 중 금융·부동산·유틸리티 섹터 제외, 재무가 달러 표시인 회사
- 재무: 종목당 요청 1번으로 연간 4년 + 최근 5분기. 손익·현금흐름은 최근 4분기 합(TTM), 4분기가 안 모이면 직전 연간 값
  NetIncomeCommonStockholders = 지배주주(보통주) 순이익, StockholdersEquity = 지배주주 자본
- 공시 시점(추정): 분기 말 + 45일, 회계연도 말 + 80일부터 사용 (미국 10-Q·10-K 제출 기한 기준)
- 그래서 과거 재무는 4년 전 회계연도부터 → 미국 백테스트는 2023년 무렵부터 가능
- 시가총액: 재무의 발행주식수 × 종가 (주식분할 보정)
- 금액 단위는 백만 달러, 시세는 달러
"""
import os, re, sys, json, time, gzip, pickle, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE_DIR = os.path.join(ROOT, "cache")
OUT = os.path.join(ROOT, "site", "data", "us.json")
HIST_DAYS = 1300
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Accept": "application/json, text/plain, */*"}
_s = requests.Session()
_s.headers.update(UA)
EXCL_SECTOR = {"Finance", "Real Estate", "Utilities"}
LAG_Q, LAG_A = 45, 80
FVER = 2                     # 2: 매출액 추가


def _load(name, default):
    p = os.path.join(CACHE_DIR, name)
    if os.path.exists(p):
        try:
            with gzip.open(p, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"  {name} 캐시 읽기 실패: {e}")
    return default


def _save(name, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with gzip.open(os.path.join(CACHE_DIR, name), "wb") as f:
        pickle.dump(obj, f)


def _num(x):
    try:
        return float(str(x).replace("$", "").replace(",", ""))
    except Exception:
        return np.nan


# ---------------- 종목 목록 ----------------
def universe(min_cap=100e6):
    rows = []
    for ex, mk in (("nasdaq", "Q"), ("nyse", "K")):
        for a in range(3):
            try:
                r = _s.get(f"https://api.nasdaq.com/api/screener/stocks?tableonly=true&download=true&exchange={ex}", timeout=60)
                data = r.json()["data"]["rows"]
                break
            except Exception as e:
                print(f"  나스닥 목록 {ex} 재시도: {e}")
                time.sleep(5)
                data = []
        for x in data:
            rows.append({"ticker": x.get("symbol", "").strip(), "name": x.get("name", ""), "mkt": mk,
                         "sector": x.get("sector", ""), "industry": x.get("industry", ""), "cap": _num(x.get("marketCap"))})
    df = pd.DataFrame(rows)
    if df.empty:
        cached = _load("us_univ.pkl.gz", None)
        if cached is None:
            raise RuntimeError("미국 종목 목록을 받지 못했습니다")
        print("  미국 종목 목록: 지난번 목록 사용")
        return cached
    n0 = len(df)
    df = df[df["ticker"].str.fullmatch(r"[A-Z]{1,5}(/[A-Z])?")]               # 우선주·워런트·유닛 제외
    df = df[~df["name"].str.contains(r"Warrant|Right|Unit|Preferred|Depositary Shares? Representing|Notes? due", case=False, regex=True)]
    df = df[~df["sector"].isin(EXCL_SECTOR) & df["sector"].ne("")]
    df = df[df["cap"] >= min_cap]
    df["ticker"] = df["ticker"].str.replace("/", "-")                        # BRK/B → BRK-B (Yahoo 표기)
    df["name"] = df["name"].str.replace(r"\s+(Common Stock|Class [A-Z] Common Stock|Ordinary Shares|Common Shares).*$", "", regex=True)
    df = df.drop_duplicates("ticker").reset_index(drop=True)
    print(f"미국 종목: 전체 {n0} → 금융·부동산·유틸리티·소형(시총 {min_cap / 1e6:.0f}M 미만) 제외 {len(df)}")
    _save("us_univ.pkl.gz", df)
    return df


# ---------------- 재무 (Yahoo fundamentals-timeseries) ----------------
TYPES = ["TotalRevenue", "OperatingIncome", "NetIncomeCommonStockholders", "NetIncome", "NetIncomeIncludingNoncontrollingInterests",
         "TotalAssets", "CurrentAssets", "CurrentLiabilities", "TotalLiabilitiesNetMinorityInterest",
         "StockholdersEquity", "TotalEquityGrossMinorityInterest", "OperatingCashFlow", "CapitalExpenditure",
         "FreeCashFlow", "OrdinarySharesNumber", "ShareIssued"]
FLOWS = {"rev": ["TotalRevenue"], "ebit": ["OperatingIncome"], "ni": ["NetIncomeIncludingNoncontrollingInterests", "NetIncome"],
         "nip": ["NetIncomeCommonStockholders", "NetIncome"], "ocf": ["OperatingCashFlow"],
         "capex": ["CapitalExpenditure"], "fcfy": ["FreeCashFlow"]}
STOCKS = {"ta": ["TotalAssets"], "tl": ["TotalLiabilitiesNetMinorityInterest"], "ca": ["CurrentAssets"],
          "cl": ["CurrentLiabilities"], "eqp": ["StockholdersEquity"], "eq": ["TotalEquityGrossMinorityInterest", "StockholdersEquity"],
          "sh": ["OrdinarySharesNumber", "ShareIssued"]}


def fetch_ts(t):
    tp = ",".join(p + x for p in ("annual", "quarterly") for x in TYPES)
    url = (f"https://query2.finance.yahoo.com/ws/fundamentals-timeseries/v1/finance/timeseries/{t}"
           f"?symbol={t}&type={tp}&period1=1420070400&period2={int(time.time())}")
    for a in range(4):
        try:
            r = _s.get(url, timeout=40)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(20 * (a + 1))
                continue
        except Exception:
            pass
        time.sleep(2 + a * 3)
    return "err"


def parse_ts(j):
    """→ DataFrame(index=사용 시작일, ql, ebit, ni, nip, fcf, ta, tl, ca, cl, eq, eqp, sh) 금액 백만 달러. 달러가 아니면 None"""
    res = (((j or {}).get("timeseries") or {}).get("result")) or []
    A, Q = {}, {}                                # 개념 → {날짜: 값}
    for x in res:
        k = (x.get("meta", {}).get("type") or [""])[0]
        vals = x.get(k) or []
        freq, name = ("annual", k[6:]) if k.startswith("annual") else ("quarterly", k[9:])
        for v in vals:
            if not v:
                continue
            cur = v.get("currencyCode")
            if cur and cur != "USD" and name != "OrdinarySharesNumber" and name != "ShareIssued":
                return None
            raw = (v.get("reportedValue") or {}).get("raw")
            if raw is None:
                continue
            (A if freq == "annual" else Q).setdefault(name, {})[v["asOfDate"]] = float(raw)

    def pick(src, names, d):
        for n in names:
            if d in src.get(n, {}):
                return src[n][d]
        return np.nan

    a_dates = sorted({d for n in ("TotalAssets", "OperatingIncome") for d in A.get(n, {})})
    q_dates = sorted({d for n in ("TotalAssets", "OperatingIncome") for d in Q.get(n, {})})
    rows = []
    for d in a_dates:
        r = {"end": d, "avail": pd.Timestamp(d) + pd.Timedelta(days=LAG_A), "ql": f"{d[2:4]}.FY"}
        for k, ns in FLOWS.items():
            r[k] = pick(A, ns, d)
        for k, ns in STOCKS.items():
            r[k] = pick(A, ns, d)
        rows.append(r)
    for i, d in enumerate(q_dates):
        if d in a_dates:                         # 회계연도 말 분기는 연간 보고서(+80일)로 대신
            continue
        D_ = pd.Timestamp(d)
        r = {"end": d, "avail": D_ + pd.Timedelta(days=LAG_Q), "ql": f"{d[2:4]}.{(D_.month - 1) // 3 + 1}Q"}
        win = [x for x in q_dates if pd.Timestamp(x) <= D_ and (D_ - pd.Timestamp(x)).days < 350]
        prev_a = [x for x in a_dates if x <= d]
        for k, ns in FLOWS.items():
            v = [pick(Q, ns, x) for x in win]
            if len(win) == 4 and all(np.isfinite(v)):
                r[k] = sum(v)                    # 최근 4분기 합
            elif prev_a and (D_ - pd.Timestamp(prev_a[-1])).days < 370:
                r[k] = pick(A, ns, prev_a[-1])   # 4분기가 안 모이면 직전 연간 값
            else:
                r[k] = np.nan
        for k, ns in STOCKS.items():
            r[k] = pick(Q, ns, d)
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows)
    for k in ("rev", "ebit", "ni", "nip", "ocf", "capex", "fcfy", "ta", "tl", "ca", "cl", "eq", "eqp"):
        df[k] = df[k] / 1e6
    df["fcf"] = np.where(np.isfinite(df["fcfy"]), df["fcfy"],
                         df["ocf"] + np.where(np.isfinite(df["capex"]), df["capex"], 0))
    df = df.sort_values(["avail", "end"])
    out, top = [], ""
    for r in df.to_dict("records"):
        if r["end"] < top:
            continue
        top = r["end"]
        out.append(r)
    df = pd.DataFrame(out).drop_duplicates("avail", keep="last").set_index("avail")
    return df[["ql", "rev", "ebit", "ni", "nip", "fcf", "ta", "tl", "ca", "cl", "eq", "eqp", "sh"]]


def load_fund(univ, minutes):
    fac = _load("us_fund.pkl.gz", {})            # 티커 → {df, at}
    now = time.time()
    tick = list(univ["ticker"])
    todo = sorted([t for t in tick if t not in fac or now - fac[t]["at"] > 3 * 86400 or fac[t].get("v", 1) < FVER],
                  key=lambda t: fac.get(t, {}).get("at", 0))
    print(f"미국 재무 요청 {len(todo)}건 (3일마다 새로), 최대 {minutes}분")

    def job(t):
        time.sleep(0.3)
        j = fetch_ts(t)
        if j == "err":
            return t, "err"
        try:
            return t, parse_ts(j)
        except Exception as e:
            return t, f"err {e}"

    t0, done, errs, chunk = time.time(), 0, 0, 100
    with ThreadPoolExecutor(4) as ex:
        for i in range(0, len(todo), chunk):
            if time.time() - t0 > minutes * 60:
                print(f"  시간 한도 — 나머지 {len(todo) - i}건은 다음 실행 때")
                break
            for t, df in ex.map(job, todo[i:i + chunk]):
                if isinstance(df, str):
                    errs += 1
                    continue                      # 실패는 다음 실행 때 다시
                fac[t] = {"df": df, "at": now, "v": FVER}
                done += 1
            if (i // chunk) % 5 == 4:
                print(f"  재무 {i + chunk}/{len(todo)} ({time.time() - t0:.0f}초, 실패 {errs})")
                _save("us_fund.pkl.gz", fac)
    _save("us_fund.pkl.gz", fac)
    ok = {t: fac[t]["df"] for t in tick if t in fac and fac[t]["df"] is not None and len(fac[t]["df"])}
    print(f"미국 재무 이번 {done}건 (실패 {errs}), 사용 가능 {len(ok)}개 ({time.time() - t0:.0f}초)")
    return ok


# ---------------- 시세 ----------------
def _yf_download(tickers, period):
    import yfinance as yf
    out = {}
    for i in range(0, len(tickers), 100):
        part = tickers[i:i + 100]
        for a in range(3):
            try:
                df = yf.download(part, period=period, interval="1d", auto_adjust=False, actions=True,
                                 group_by="ticker", threads=True, progress=False)
                break
            except Exception as e:
                print(f"  yfinance 재시도 {a + 1}: {e}")
                time.sleep(10 * (a + 1))
                df = None
        if df is None or df.empty:
            continue
        for t in part:
            try:
                g = df[t] if isinstance(df.columns, pd.MultiIndex) else df
            except KeyError:
                continue
            g = g.dropna(subset=["Close"])
            if len(g):
                cols = [c for c in ("Close", "Volume", "Stock Splits") if c in g]
                out[t] = g[cols].copy()
        time.sleep(1)
    return out


def load_prices(tickers):
    cache = _load("us_px.pkl.gz", {})
    today = pd.Timestamp.now(tz="America/New_York").tz_localize(None).normalize()
    fresh = [t for t in tickers if t in cache and len(cache[t]) > 100 and (today - cache[t].index[-1]).days <= 25]
    full = [t for t in tickers if t not in fresh]
    t0 = time.time()
    print(f"미국 시세: 이어받기 {len(fresh)}, 전체 받기 {len(full)}")
    if full:
        got = _yf_download(full, f"{int(HIST_DAYS * 1.5)}d")
        cache.update(got)
        print(f"  전체 받기 성공 {len(got)}/{len(full)} ({time.time() - t0:.0f}초)")
    if fresh:
        got = _yf_download(fresh, "1mo")
        for t, g in got.items():
            old = cache[t]
            cache[t] = pd.concat([old[old.index < g.index[0]], g]).iloc[-int(HIST_DAYS * 1.1):]
        print(f"  이어받기 성공 {len(got)}/{len(fresh)} ({time.time() - t0:.0f}초)")
    _save("us_px.pkl.gz", cache)
    return {t: cache[t] for t in tickers if t in cache}


def index_prices():
    got = _yf_download(["^GSPC", "^IXIC"], f"{int(HIST_DAYS * 1.5)}d")
    return {"K": got.get("^GSPC"), "Q": got.get("^IXIC")}


# ---------------- 앱용 JSON ----------------
def rnd(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return int(round(x)) if n == 0 else round(x, n)


def main():
    t0 = time.time()
    now = pd.Timestamp.now(tz="Asia/Seoul")
    print(f"미국 실행 {now:%Y-%m-%d %H:%M} KST")
    univ = universe()
    fund = load_fund(univ, float(os.environ.get("US_MINUTES", "40")))
    meta = univ.set_index("ticker")
    tick = {t: t for t, df in fund.items() if len(df["ta"].dropna()) and df["ta"].dropna().iloc[-1] >= 50}
    hist = load_prices(sorted(tick))
    idx = index_prices()
    if idx.get("K") is None or len(hist) < 300:
        raise SystemExit(f"미국 시세가 부족합니다 (종목 {len(hist)}, 지수 {'있음' if idx.get('K') is not None else '없음'})")
    cal = pd.DatetimeIndex(pd.to_datetime(idx["K"].index).normalize()).unique().sort_values()
    cal = cal[cal >= cal.max() - pd.Timedelta(days=int(HIST_DAYS * 1.45))]
    s = pd.Series(cal, index=cal)
    months = list(s.groupby([s.index.year, s.index.month]).max())
    M = pd.DatetimeIndex(months)
    ix = {}
    for k, h in idx.items():
        if h is None:
            continue
        c = h["Close"].copy()
        c.index = pd.to_datetime(c.index).normalize()
        ix[k] = [rnd(x, 2) for x in c.reindex(M, method="ffill")]

    stocks = []
    for t, h in hist.items():
        c = tick[t]
        f = fund[c]
        h = h.copy()
        h.index = pd.to_datetime(h.index).normalize()
        h = h[~h.index.duplicated(keep="last")].sort_index()
        if len(h) < 30:
            continue
        amt20 = (h["Close"] * h["Volume"] / 1e6).rolling(20, min_periods=5).mean()
        s0 = int(np.searchsorted(M, h.index[0]))
        if s0 >= len(M):
            continue
        Ms = M[s0:]
        px = h["Close"].reindex(Ms, method="ffill")
        px[Ms > h.index[-1] + pd.Timedelta(days=10)] = np.nan
        am = amt20.reindex(Ms, method="ffill")
        # 주식분할 보정: 공시 주식 수 × (그 뒤 분할 배수) → 분할 반영된 주가와 맞춤
        spl = h["Stock Splits"] if "Stock Splits" in h else pd.Series(dtype=float)
        spl = spl[spl > 0]
        def adj(d):
            return float(np.prod(spl[spl.index > d].values)) if len(spl) else 1.0
        fi = f.index.searchsorted(M, side="right") - 1
        fl, prev = [], None
        for mi, r in enumerate(fi):
            if r < 0 or r == prev:
                continue
            prev = r
            row = f.iloc[r]
            sh = row["sh"] * adj(f.index[r]) if np.isfinite(row["sh"]) else None
            fl.append([mi] + [rnd(row[k], 0) for k in ("ebit", "ni", "ta", "tl", "ca", "cl", "fcf", "nip", "eqp")]
                      + [rnd(sh, 0), rnd(row["rev"], 0)])
        if not fl:
            continue
        SH_I = 10                                      # [mi, ebit, ni, ta, tl, ca, cl, fcf, nip, eqp, sh, rev]
        last_sh = next((x[SH_I] for x in reversed(fl) if x[SH_I]), None)
        if not last_sh:
            continue
        stocks.append([t, str(meta.loc[t, "name"]), meta.loc[t, "mkt"], int(last_sh), s0,
                       [rnd(x, 2) for x in px], [rnd(x, 1) for x in am], fl, f["ql"].iloc[-1],
                       (str(meta.loc[t, "industry"]) if "industry" in meta.columns else "") or None])
    out = {
        "region": "US", "cap_div": 1e6, "currency": "USD",
        "generated_at": f"{now:%Y-%m-%d %H:%M}",
        "trade_date": f"{months[-1]:%Y-%m-%d}",
        "fund": True,
        "months": [f"{d:%Y-%m-%d}" for d in months],
        "idx": ix,
        "fields": ["code", "name", "mkt", "shares", "s0", "px", "amt20", "f", "ql", "ind"],
        "f_fields": ["mi", "ebit", "ni", "ta", "tl", "ca", "cl", "fcf", "nip", "eqp", "sh", "rev"],
        "note": "미국(Yahoo 재무): 금액 백만 달러, 시세 달러, mkt K=NYSE Q=나스닥, idx K=S&P500 Q=나스닥종합, sh=분할 보정 발행주식수, 재무 사용 시점 추정(분기+45일, 연간+80일)",
        "stocks": stocks,
    }
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))
    n_fcf = sum(1 for x in stocks if x[7][-1][7] is not None)
    print(f"미국 저장: {len(stocks)}종목 (FCF {n_fcf}), {len(months)}개월, "
          f"{os.path.getsize(OUT) / 1e6:.1f}MB ({time.time() - t0:.0f}초)")


if __name__ == "__main__":
    main()
