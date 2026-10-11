"""시세 수집 (종가매매·오닐 앱과 같은 방식): 종목 목록(+업종) + 일봉 (네이버 차트 → FinanceDataReader → pykrx 순으로 시도)"""
import os, re, time, gzip, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TICKERS_CSV = os.path.join(ROOT, "engine", "tickers.csv")
CACHE = os.path.join(ROOT, "cache", "hist.pkl.gz")
UA = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124 Safari/537.36",
      "Referer": "https://finance.naver.com/"}
_session = requests.Session()
_session.headers.update(UA)


# ---------------- 종목 목록 ----------------
def _clean(df):
    df = df.copy()
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df[df["market"].astype(str).str.contains("KOSPI|KOSDAQ") & ~df["market"].astype(str).str.contains("KONEX")]
    df = df[df["code"].str.fullmatch(r"[0-9A-Z]{5}0")]                    # 우선주 제외 (신규 영문 혼합 코드 포함)
    df = df[~df["name"].astype(str).str.contains(r"스팩|\d+호|리츠|ETN|ETF")]  # 스팩·리츠 제외
    df["market"] = np.where(df["market"].astype(str).str.contains("KOSDAQ"), "KOSDAQ", "KOSPI")
    if "ind" not in df:
        df["ind"] = ""
    df["ind"] = df["ind"].fillna("").astype(str)
    return df[["code", "name", "market", "shares", "ind"]].drop_duplicates("code").reset_index(drop=True)


def _industry():
    """KRX 업종(한국표준산업분류). 실패하거나 90초 넘게 걸리면 빈 표 (종목명으로 금융 판단)"""
    try:
        import FinanceDataReader as fdr
        import threading
        box = {}

        def _run():
            try:
                box["d"] = fdr.StockListing("KRX-DESC")
            except Exception as e:
                box["e"] = e
        th = threading.Thread(target=_run, daemon=True)   # 데몬: 멈춰도 프로그램 종료를 막지 않음
        th.start()
        th.join(90)
        if "d" not in box:
            raise RuntimeError(box.get("e") or "90초 초과")
        d = box["d"]
        d = d.rename(columns={"Code": "code", "Industry": "ind"})
        d["code"] = d["code"].astype(str).str.zfill(6)
        return d[["code", "ind"]].drop_duplicates("code")
    except Exception as e:
        print(f"  업종 목록 실패: {e}")
        return pd.DataFrame(columns=["code", "ind"])


def _list_fdr():
    import FinanceDataReader as fdr
    df = fdr.StockListing("KRX")
    df = df.rename(columns={"Code": "code", "Name": "name", "Market": "market", "Stocks": "shares"})
    if "shares" not in df:
        df["shares"] = np.nan
    df["code"] = df["code"].astype(str).str.zfill(6)
    df = df.merge(_industry(), on="code", how="left")
    return _clean(df)


def _list_pykrx():
    from pykrx import stock
    d = pd.Timestamp.now(tz="Asia/Seoul")
    for _ in range(10):
        ds = d.strftime("%Y%m%d")
        rows = []
        for mk in ("KOSPI", "KOSDAQ"):
            cap = stock.get_market_cap(ds, market=mk)
            if cap is None or cap.empty or cap["거래량"].sum() == 0:
                rows = []
                break
            for code, r in cap.iterrows():
                rows.append({"code": code, "name": stock.get_market_ticker_name(code), "market": mk, "shares": r["상장주식수"]})
        if rows:
            return _clean(pd.DataFrame(rows).merge(_industry(), on="code", how="left"))
        d -= pd.Timedelta(days=1)
    raise RuntimeError("pykrx 목록 없음")


def load_universe():
    for f in (_list_fdr, _list_pykrx):
        try:
            df = f()
            if len(df) > 1500:
                df.to_csv(TICKERS_CSV, index=False, encoding="utf-8-sig")
                print(f"종목 목록: {f.__name__} {len(df)}개")
                return df
        except Exception as e:
            print(f"  {f.__name__} 실패: {e}")
    if os.path.exists(TICKERS_CSV):
        df = pd.read_csv(TICKERS_CSV, dtype={"code": str})
        df["code"] = df["code"].str.zfill(6)
        df["ind"] = df.get("ind", pd.Series("", index=df.index)).fillna("").astype(str)
        print(f"종목 목록: 저장된 목록 사용 {len(df)}개")
        return df
    raise RuntimeError("종목 목록을 가져오지 못했습니다")


# ---------------- 일봉 ----------------
def naver_daily(symbol, count):
    url = f"https://fchart.stock.naver.com/sise.nhn?symbol={symbol}&timeframe=day&count={count}&requestType=0"
    for attempt in range(3):
        try:
            r = _session.get(url, timeout=10)
            items = re.findall(r'data="([^"]+)"', r.text)
            if not items:
                return None
            rows = [x.split("|") for x in items]
            df = pd.DataFrame(rows, columns=["Date", "Open", "High", "Low", "Close", "Volume"])
            df["Date"] = pd.to_datetime(df["Date"], format="%Y%m%d")
            df = df.set_index("Date").apply(pd.to_numeric, errors="coerce")
            df = df[(df["Close"] > 0)]
            # 거래정지일 등 시가 0 보정
            for c in ("Open", "High", "Low"):
                df[c] = df[c].where(df[c] > 0, df["Close"])
            return df
        except Exception:
            time.sleep(1 + attempt)
    return None


def fdr_daily(code, days):
    try:
        import FinanceDataReader as fdr
        start = (pd.Timestamp.now() - pd.Timedelta(days=int(days * 1.5))).strftime("%Y-%m-%d")
        h = fdr.DataReader(code, start)
        if h is not None and len(h):
            return h[["Open", "High", "Low", "Close", "Volume"]]
    except Exception:
        pass
    return None


def daily(code, count):
    h = naver_daily(code, count)
    if h is None or len(h) == 0:
        h = fdr_daily(code, count)
    return h


def load_histories(codes, full_count=1300, inc_count=20, workers=12):
    """캐시가 있으면 최근 일봉만 받아서 이어붙임"""
    cache = {}
    if os.path.exists(CACHE):
        try:
            with gzip.open(CACHE, "rb") as f:
                cache = pickle.load(f)
            print(f"캐시 {len(cache)}종목 불러옴")
        except Exception as e:
            print(f"캐시 읽기 실패: {e}")
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None).normalize()

    def job(code):
        old = cache.get(code)
        if old is not None and len(old) > 100 and (today - old.index[-1]).days <= inc_count:
            new = daily(code, inc_count)
            if new is None:
                return code, old
            h = pd.concat([old[old.index < new.index[0]], new])
            return code, h.iloc[-full_count:]
        return code, daily(code, full_count)

    t0 = time.time()
    out, fail = {}, 0
    print(f"일봉 수집 시작 {len(codes)}종목 (캐시 {len(cache)})")
    with ThreadPoolExecutor(workers) as ex:
        for n, (code, h) in enumerate(ex.map(job, codes), 1):
            if h is not None and len(h):
                out[code] = h
            else:
                fail += 1
            if n % 500 == 0:
                print(f"  일봉 {n}/{len(codes)} ({time.time() - t0:.0f}초)")
    print(f"일봉 {len(out)}종목 (실패 {fail}) {time.time() - t0:.0f}초")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with gzip.open(CACHE, "wb") as f:
        pickle.dump(out, f)
    return out


def index_daily(count=1300):
    """코스피·코스닥 지수"""
    res = {}
    for name in ("KOSPI", "KOSDAQ"):
        h = naver_daily(name, count)
        if h is None:
            try:
                import FinanceDataReader as fdr
                h = fdr.DataReader({"KOSPI": "KS11", "KOSDAQ": "KQ11"}[name],
                                   (pd.Timestamp.now() - pd.Timedelta(days=count * 1.5)).strftime("%Y-%m-%d"))
            except Exception:
                h = None
        if h is not None and len(h):
            res[name] = h["Close"]
    return res
