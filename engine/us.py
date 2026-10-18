"""미국 시장: SEC EDGAR 재무(XBRL companyfacts) + yfinance 시세 → site/data/us.json (국내 mf.json과 같은 구조)

- 대상: NYSE·나스닥 상장사 중 금융(SIC 6000~6799, 리츠·스팩 포함)·유틸리티(SIC 4900~4949) 제외
- 재무: 회사별 companyfacts 1번 요청으로 전 기간을 받음. 손익·현금흐름은 연초부터 누적 값으로
  최근 4분기 합(TTM) = 직전 연간 + 올해 누적 − 작년 같은 기간 누적 (연간 보고서면 그대로)
  NetIncomeLoss = 지배주주 순이익, StockholdersEquity = 지배주주 자본 (미국 회계 기준 정의)
- 공시 시점: 그 분기 값이 처음 제출된 날(filed) 다음 날부터 사용 → 미래 정보가 섞이지 않음
- 시가총액: 공시의 발행주식수 × 종가. 주식분할은 yfinance 분할 기록으로 맞춤
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
UA = os.environ.get("SEC_UA", "").strip() or "magic-screener kyuiklee@users.noreply.github.com"
YEARS = 7
HIST_DAYS = 1300

_s = requests.Session()
_s.headers.update({"User-Agent": UA, "Accept-Encoding": "gzip, deflate"})


class Rate:
    """SEC 권장: 초당 10회 이하"""
    def __init__(self, per_sec=8):
        self.gap, self.t, self.lock = 1.0 / per_sec, 0.0, threading.Lock()

    def wait(self):
        with self.lock:
            now = time.time()
            if now < self.t:
                time.sleep(self.t - now)
            self.t = max(now, self.t) + self.gap


RATE = Rate()


def sec_get(url, tries=4):
    for a in range(tries):
        RATE.wait()
        try:
            r = _s.get(url, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 404:
                return None
            if r.status_code in (403, 429):
                time.sleep(5 + a * 10)
                continue
        except Exception:
            time.sleep(2 + a * 3)
    return None


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


# ---------------- 종목 목록 ----------------
def universe():
    j = sec_get("https://www.sec.gov/files/company_tickers_exchange.json")
    if not j:
        raise RuntimeError("SEC 종목 목록을 받지 못했습니다 (User-Agent 차단 여부 확인)")
    f = j["fields"]
    df = pd.DataFrame(j["data"], columns=f)
    df = df[df["exchange"].isin(["NYSE", "Nasdaq"]) & df["ticker"].notna()]
    df = df[~df["ticker"].str.contains(r"[\^/]", regex=True)]
    df = df.drop_duplicates("cik", keep="first")          # 한 회사에 여러 종목이면 첫 번째(대표)
    df["mkt"] = np.where(df["exchange"] == "Nasdaq", "Q", "K")
    return df[["cik", "ticker", "name", "mkt"]].reset_index(drop=True)


def excluded_sic(sic):
    try:
        s = int(sic)
    except (TypeError, ValueError):
        return False
    return 6000 <= s <= 6799 or 4900 <= s <= 4949


# ---------------- 재무 파싱 ----------------
FLOW = {   # 키: (분류, 개념 후보)
    "ebit": ["OperatingIncomeLoss"],
    "ni": ["ProfitLoss", "NetIncomeLoss"],
    "nip": ["NetIncomeLoss", "NetIncomeLossAvailableToCommonStockholdersBasic"],
    "ocf": ["NetCashProvidedByUsedInOperatingActivities", "NetCashProvidedByUsedInOperatingActivitiesContinuingOperations"],
    "capex": ["PaymentsToAcquirePropertyPlantAndEquipment", "PaymentsToAcquireProductiveAssets"],
    "capi": ["PaymentsToAcquireIntangibleAssets"],
}
INST = {
    "ta": ["Assets"],
    "tl": ["Liabilities"],
    "ca": ["AssetsCurrent"],
    "cl": ["LiabilitiesCurrent"],
    "eqp": ["StockholdersEquity"],
    "eq": ["StockholdersEquityIncludingPortionAttributableToNoncontrollingInterest", "StockholdersEquity"],
}
FORMS = ("10-K", "10-Q", "10-K/A", "10-Q/A", "10-KT")
D = pd.Timedelta


def _facts(j, name, tax="us-gaap", unit="USD"):
    try:
        return j["facts"][tax][name]["units"][unit]
    except (KeyError, TypeError):
        return []


def _flow_map(j, names):
    """(start, end) → (값, 처음 제출일). 첫 번째로 데이터가 있는 개념을 씀"""
    for nm in names:
        m = {}
        for x in _facts(j, nm):
            if x.get("form") not in FORMS or "start" not in x:
                continue
            k = (x["start"], x["end"])
            if k not in m or x["filed"] < m[k][1]:
                m[k] = (float(x["val"]), x["filed"])
        if m:
            return m
    return {}


def _inst_map(j, names, unit="USD", tax="us-gaap"):
    for nm in names:
        m = {}
        for x in _facts(j, nm, tax, unit):
            if x.get("form") not in FORMS or "start" in x:
                continue
            k = x["end"]
            if k not in m or x["filed"] < m[k][1]:
                m[k] = (float(x["val"]), x["filed"])
        if m:
            return m
    return {}


def _ttm(m, end):
    """연초부터 누적 값들로 end 시점의 최근 12개월 합. 없으면 None"""
    if not m:
        return None
    E = pd.Timestamp(end)
    best = None
    for (s, e), (v, f) in m.items():
        if e != end:
            continue
        days = (E - pd.Timestamp(s)).days
        if 350 <= days <= 380:
            return v                                     # 연간 값 그대로
        if 80 <= days <= 300 and (best is None or days > best[0]):
            best = (days, s, v)                          # 가장 긴 누적(YTD)
    if best is None:
        return None
    days, s, ytd = best
    S = pd.Timestamp(s)
    ann = prev = None
    for (s2, e2), (v2, f2) in m.items():
        S2, E2 = pd.Timestamp(s2), pd.Timestamp(e2)
        if ann is None and abs((E2 - (S - D(days=1))).days) <= 7 and 350 <= (E2 - S2).days <= 380:
            ann = v2
        if prev is None and abs((E2 - (E - D(days=365))).days) <= 7 and abs((E2 - S2).days - days) <= 10:
            prev = v2
    if ann is None or prev is None:
        return None
    return ann + ytd - prev


def parse(j):
    """companyfacts → 분기별 행 [{avail, ql, ebit, ni, nip, fcf, ta, tl, ca, cl, eq, eqp, sh}] (백만 달러, 주식 수는 그대로)"""
    fl = {k: _flow_map(j, v) for k, v in FLOW.items()}
    it = {k: _inst_map(j, v) for k, v in INST.items()}
    sh_dei = _inst_map(j, ["EntityCommonStockSharesOutstanding"], unit="shares", tax="dei")
    sh_w = _flow_map_shares(j)
    # 분기 말(end) 목록과 그 분기 보고서가 처음 제출된 날
    ends = {}
    cut = (pd.Timestamp.now() - pd.DateOffset(years=YEARS)).strftime("%Y-%m-%d")
    meta = {}
    for nm in ("Assets", "OperatingIncomeLoss", "NetIncomeLoss"):
        for x in _facts(j, nm):
            if x.get("form") not in FORMS or x["end"] < cut:
                continue
            e = x["end"]
            if e not in ends or x["filed"] < ends[e]:
                ends[e] = x["filed"]
                meta[e] = (x.get("fy"), x.get("fp"), x.get("accn"))
    rows = []
    for e in sorted(ends):
        r = {"end": e, "avail": pd.Timestamp(ends[e]) + D(days=1)}
        fy, fp, accn = meta[e]
        r["ql"] = f"{int(fy) % 100:02d}.{fp}" if fy and fp else e[2:7]
        for k in ("ebit", "ni", "nip", "ocf", "capex", "capi"):
            v = _ttm(fl[k], e)
            r[k] = v / 1e6 if v is not None else np.nan
        for k in it:
            v = it[k].get(e)
            r[k] = v[0] / 1e6 if v else np.nan
        if not np.isfinite(r["tl"]) and np.isfinite(r["ta"]):        # 부채총계를 안 적는 회사가 많음
            base = r["eq"] if np.isfinite(r["eq"]) else r["eqp"]
            r["tl"] = r["ta"] - base if np.isfinite(base) else np.nan
        cap = (r["capex"] if np.isfinite(r["capex"]) else 0) + (r["capi"] if np.isfinite(r["capi"]) else 0)
        r["fcf"] = r["ocf"] - cap if np.isfinite(r["ocf"]) else np.nan
        # 발행주식수: 그 공시 표지의 주식 수 → 없으면 희석 가중평균
        sd = sorted((f, v) for (ee, (v, f)) in sh_dei.items() if f >= ends[e] and 0 <= (pd.Timestamp(ee) - pd.Timestamp(e)).days < 150)
        sw = sh_w.get(e)
        sh = sd[0][1] if sd and sd[0][0] <= (pd.Timestamp(ends[e]) + D(days=10)).strftime("%Y-%m-%d") else np.nan
        if sw and (not np.isfinite(sh) or sh < sw * 0.5):          # 여러 종류 주식 중 한 종류만 적힌 경우
            sh = sw
        r["sh"] = sh
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values(["avail", "end"])
    out, top = [], ""
    for r in df.to_dict("records"):
        if r["end"] < top:                            # 늦게 낸 옛 분기 보고서는 무시
            continue
        top = r["end"]
        out.append(r)
    df = pd.DataFrame(out).drop_duplicates("avail", keep="last").set_index("avail")
    return df[["ql", "ebit", "ni", "nip", "fcf", "ta", "tl", "ca", "cl", "eq", "eqp", "sh"]]


def _flow_map_shares(j):
    """희석 가중평균 주식 수: end → 값 (가장 짧은 기간 = 그 분기)"""
    m = {}
    for x in _facts(j, "WeightedAverageNumberOfDilutedSharesOutstanding", unit="shares"):
        if x.get("form") not in FORMS or "start" not in x:
            continue
        d = (pd.Timestamp(x["end"]) - pd.Timestamp(x["start"])).days
        e = x["end"]
        if e not in m or d < m[e][1]:
            m[e] = (float(x["val"]), d)
    return {e: v for e, (v, d) in m.items()}


# ---------------- SEC 수집 (캐시) ----------------
def load_fund(univ, minutes):
    sub = _load("us_sub.pkl.gz", {})          # cik → {sic, at}
    fac = _load("us_facts.pkl.gz", {})        # cik → {df, at}
    now = time.time()
    ciks = list(univ["cik"])
    need_sub = [c for c in ciks if c not in sub or now - sub[c]["at"] > 60 * 86400]
    t0 = time.time()
    print(f"SEC 업종 조회 {len(need_sub)}건")

    def get_sub(c):
        j = sec_get(f"https://data.sec.gov/submissions/CIK{int(c):010d}.json")
        return c, (j or {}).get("sic")

    with ThreadPoolExecutor(8) as ex:
        for i, (c, sic) in enumerate(ex.map(get_sub, need_sub), 1):
            sub[c] = {"sic": sic, "at": now}
            if i % 1000 == 0:
                print(f"  업종 {i}/{len(need_sub)} ({time.time() - t0:.0f}초)")
                _save("us_sub.pkl.gz", sub)
    _save("us_sub.pkl.gz", sub)
    keep = [c for c in ciks if not excluded_sic(sub.get(c, {}).get("sic"))]
    print(f"금융·유틸리티 제외 후 {len(keep)}/{len(ciks)}개 ({time.time() - t0:.0f}초)")

    # 재무: 오래된 것부터 정해진 시간만큼 새로 받음 (처음엔 전부, 이후엔 하루 한 번 정도)
    todo = sorted([c for c in keep if c not in fac or now - fac[c]["at"] > 20 * 3600],
                  key=lambda c: fac.get(c, {}).get("at", 0))
    print(f"SEC 재무 요청 {len(todo)}건, 최대 {minutes}분")

    def get_fac(c):
        j = sec_get(f"https://data.sec.gov/api/xbrl/companyfacts/CIK{int(c):010d}.json")
        if j is None:
            return c, None
        try:
            return c, parse(j)
        except Exception as e:
            return c, f"err {e}"

    t1, done, errs = time.time(), 0, 0
    chunk = 80
    with ThreadPoolExecutor(8) as ex:
        for i in range(0, len(todo), chunk):
            if time.time() - t1 > minutes * 60:
                print(f"  시간 한도 — 나머지 {len(todo) - i}건은 다음 실행 때")
                break
            for c, df in ex.map(get_fac, todo[i:i + chunk]):
                if isinstance(df, str):
                    errs += 1
                    df = None
                fac[c] = {"df": df, "at": now}
                done += 1
            if (i // chunk) % 10 == 9:
                print(f"  재무 {done}/{len(todo)} ({time.time() - t1:.0f}초)")
                _save("us_facts.pkl.gz", fac)
    _save("us_facts.pkl.gz", fac)
    ok = {c: fac[c]["df"] for c in keep if c in fac and fac[c]["df"] is not None and len(fac[c]["df"])}
    print(f"SEC 재무 이번 {done}건 (파싱 오류 {errs}), 사용 가능 {len(ok)}개 ({time.time() - t1:.0f}초)")
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
    print(f"미국 실행 {now:%Y-%m-%d %H:%M} KST, SEC User-Agent: {UA}")
    univ = universe()
    print(f"NYSE·나스닥 {len(univ)}개 회사")
    fund = load_fund(univ, float(os.environ.get("US_MINUTES", "40")))
    meta = univ.set_index("cik")
    # 시세는 재무가 있고 자산 5천만 달러 이상인 회사만
    tick = {}
    for c, df in fund.items():
        ta = df["ta"].dropna()
        if len(ta) and ta.iloc[-1] >= 50:
            tick[meta.loc[c, "ticker"]] = c
    hist = load_prices(sorted(tick))
    idx = index_prices()
    if idx.get("K") is None or len(hist) < 500:
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
                      + [rnd(sh, 0)])
        if not fl:
            continue
        last_sh = next((x[-1] for x in reversed(fl) if x[-1]), None)
        if not last_sh:
            continue
        stocks.append([t, str(meta.loc[c, "name"]), meta.loc[c, "mkt"], int(last_sh), s0,
                       [rnd(x, 2) for x in px], [rnd(x, 1) for x in am], fl, f["ql"].iloc[-1]])
    out = {
        "region": "US", "cap_div": 1e6, "currency": "USD",
        "generated_at": f"{now:%Y-%m-%d %H:%M}",
        "trade_date": f"{months[-1]:%Y-%m-%d}",
        "fund": True,
        "months": [f"{d:%Y-%m-%d}" for d in months],
        "idx": ix,
        "fields": ["code", "name", "mkt", "shares", "s0", "px", "amt20", "f", "ql"],
        "f_fields": ["mi", "ebit", "ni", "ta", "tl", "ca", "cl", "fcf", "nip", "eqp", "sh"],
        "note": "미국: 금액 백만 달러, 시세 달러, mkt K=NYSE Q=나스닥, idx K=S&P500 Q=나스닥종합, sh=분할 보정 발행주식수",
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
