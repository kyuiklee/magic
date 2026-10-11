"""매일 자동 실행: 시세·실적 수집 → 월말(+오늘) 스냅샷 → 앱용 JSON 저장

앱이 이 데이터로 직접 순위를 매깁니다 (조건을 바꿔도 다시 실행할 필요 없음).
  - 매달 마지막 거래일 종가, 20일 평균 거래대금
  - 그 날짜에 이미 공시된 최신 재무 (영업이익·순이익 최근 4분기 합, 자산·부채)
  - 맨 마지막 칸은 가장 최근 거래일 (이번 달 진행 중)
"""
import os, re, sys, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import data as D
import fund as FD

ROOT = D.ROOT
OUT = os.path.join(ROOT, "site", "data")
# 그린블라트가 제외한 금융·유틸리티 (KRX 업종명 기준, 업종이 없으면 종목명으로 대신 판단)
EXCL_IND = r"금융|보험|은행|신탁|증권|여신|투자업|전기업|가스"
EXCL_NAME = r"금융$|금융지주|신한지주|은행|보험|증권|캐피탈|화재|생명|카드|자산운용|저축|파이낸셜|한국전력|가스공사|도시가스"


def kst_now():
    return pd.Timestamp.now(tz="Asia/Seoul")


def rnd(x, n=1):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return int(round(x)) if n == 0 else round(x, n)


def month_dates(cal):
    """거래일 목록 → 매달 마지막 거래일 (이번 달은 가장 최근 거래일)"""
    s = pd.Series(cal, index=cal)
    return list(s.groupby([s.index.year, s.index.month]).max())


def excluded(name, ind):
    """금융·유틸리티면 True. 일반 지주회사(LG·SK 등)는 KRX 업종이 '기타 금융업'이라도 사업회사 연결실적이므로 남김"""
    if re.search(EXCL_NAME, name):
        return True
    if not ind:
        return False
    if ind.strip() == "기타 금융업":
        return False
    return bool(re.search(EXCL_IND, ind))


def main():
    t0 = time.time()
    now = kst_now()
    print(f"실행 {now:%Y-%m-%d %H:%M} KST")

    univ = D.load_universe()
    hist = D.load_histories(list(univ["code"]))
    if len(hist) < 1000:
        raise SystemExit(f"일봉 수집 종목이 너무 적습니다 ({len(hist)}) — 데이터 소스 문제")
    print(f"일봉 완료 ({time.time() - t0:.0f}초)")
    fund = FD.load(list(hist)) or {}
    print(f"재무 완료 ({time.time() - t0:.0f}초)")

    # ----- 달력: 코스피 지수 거래일 (없으면 종목 일봉 합집합) -----
    idx = D.index_daily()
    if "KOSPI" in idx:
        cal = pd.DatetimeIndex(pd.to_datetime(idx["KOSPI"].index).normalize()).unique().sort_values()
    else:
        cnt = pd.Series(np.concatenate([pd.to_datetime(h.index).normalize().values for h in hist.values()])).value_counts()
        cal = pd.DatetimeIndex(cnt[cnt >= len(hist) * 0.5].index).sort_values()
    months = month_dates(cal)
    trade_day = months[-1]
    print(f"월 스냅샷 {len(months)}개: {months[0].date()} ~ {trade_day.date()}")
    M = pd.DatetimeIndex(months)

    ix = {}
    for k, s in idx.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index).normalize()
        s = s[~s.index.duplicated(keep="last")].sort_index()
        v = s.reindex(M, method="ffill")
        ix[{"KOSPI": "K", "KOSDAQ": "Q"}[k]] = [rnd(x, 2) for x in v]

    stocks, n_ex, n_nof = [], 0, 0
    meta = univ.set_index("code")
    for code, h in hist.items():
        if code not in meta.index:
            continue
        m = meta.loc[code]
        name, ind = str(m["name"]), str(m.get("ind", "") or "")
        if excluded(name, ind):
            n_ex += 1
            continue
        f = fund.get(code)
        if f is None or not len(f):
            n_nof += 1
            continue
        sh = m.get("shares")
        if not (isinstance(sh, (int, float, np.integer, np.floating)) and np.isfinite(sh) and sh > 0):
            continue
        h = h[["Close", "Volume"]].astype(float).copy()
        h.index = pd.to_datetime(h.index).normalize()
        h = h[~h.index.duplicated(keep="last")].sort_index()
        if len(h) < 30:
            continue
        amt20 = (h["Close"] * h["Volume"] / 1e8).rolling(20, min_periods=5).mean()
        first = h.index[0]
        s0 = int(np.searchsorted(M, first))                 # 상장 후 첫 월말
        if s0 >= len(M):
            continue
        Ms = M[s0:]
        px = h["Close"].reindex(Ms, method="ffill")
        am = amt20.reindex(Ms, method="ffill")
        # 거래가 끊긴 지 오래면(데이터 끝 이후) 값 없음
        last = h.index[-1]
        px[Ms > last + pd.Timedelta(days=10)] = np.nan
        # 재무: 월말 시점에 이미 공시된 최신 값 → 바뀔 때만 기록
        fi = f.index.searchsorted(M, side="right") - 1        # 각 월말의 최신 재무 행 (-1 = 없음)
        fl, prev = [], None
        for mi, r in enumerate(fi):
            if r < 0 or r == prev:
                continue
            prev = r
            row = f.iloc[r]
            fl.append([mi] + [rnd(row[k], 0) for k in ("ebit", "ni", "ta", "tl", "ca", "cl")])
        if not fl:
            n_nof += 1
            continue
        lr = f.iloc[-1]
        stocks.append([code, name, "Q" if m["market"] == "KOSDAQ" else "K", int(sh), s0,
                       [rnd(x, 0) for x in px], [rnd(x, 0) for x in am], fl,
                       lr["ql"] if isinstance(lr["ql"], str) else None])

    out = {
        "generated_at": f"{now:%Y-%m-%d %H:%M}",
        "trade_date": f"{trade_day:%Y-%m-%d}",
        "fund": bool(fund),
        "months": [f"{d:%Y-%m-%d}" for d in months],
        "idx": ix,
        "fields": ["code", "name", "mkt", "shares", "s0", "px", "amt20", "f", "ql"],
        "f_fields": ["mi", "ebit", "ni", "ta", "tl", "ca", "cl"],
        "note": "px·amt20은 months[s0]부터. f는 [월 번호, 영업이익TTM, 순이익TTM, 자산총계, 부채총계, 유동자산, 유동부채] (억 원), "
                "그 월말부터 다음 항목 전까지 적용. 금융·유틸리티 제외",
        "stocks": stocks,
    }
    os.makedirs(OUT, exist_ok=True)
    with open(os.path.join(OUT, "mf.json"), "w", encoding="utf-8") as fp:
        json.dump(out, fp, ensure_ascii=False, separators=(",", ":"))
    size = os.path.getsize(os.path.join(OUT, "mf.json")) / 1e6
    print(f"저장: {len(stocks)}종목 (금융·유틸 제외 {n_ex}, 재무 없음 {n_nof}), {len(months)}개월, {size:.1f}MB "
          f"({time.time() - t0:.0f}초)")
    if not fund:
        print("경고: 재무 데이터가 없어 앱에 순위가 나오지 않습니다 — Secrets에 DART_API_KEY 등록 필요")


if __name__ == "__main__":
    main()
