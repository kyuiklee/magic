"""OpenDART '배당에 관한 사항'(alotMatter) → 보통주 주당 현금배당금(DPS) 연도별

- 사업보고서 1건에 당기·전기·전전기 3년치 DPS가 들어 있어 회사당 2번(최근 연도, 3년 전)이면 6년 이력
- 사용 시작일: 그 사업보고서 접수일 다음 날 (전기·전전기 값은 1년·2년 앞 보고서 시점으로 추정)
- 예상 배당수익률 = 가장 최근 연도 DPS ÷ 현재 주가 (직전 연도 배당이 유지된다고 가정)
- 배당이 없으면 DPS 0으로 기록 (모르는 것과 구분)
"""
import os, time, gzip, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "dv.pkl.gz")
API = "https://opendart.fss.or.kr/api/alotMatter.json"


class Limit(Exception):
    pass


def _num(s):
    s = str(s).replace(",", "").strip()
    if s in ("", "-", "--"):
        return 0.0
    try:
        return float(s)
    except Exception:
        return np.nan


def _get(key, corp, year):
    params = {"crtfc_key": key, "corp_code": corp, "bsns_year": str(year), "reprt_code": "11011"}
    for a in range(3):
        try:
            j = requests.get(API, params=params, timeout=40).json()
            st = j.get("status")
            if st == "000":
                return "ok", j.get("list", [])
            if st == "013":
                return "none", []
            if st == "020":
                raise Limit(j.get("message"))
            return "err", None
        except Limit:
            raise
        except Exception:
            time.sleep(1 + a * 2)
    return "err", None


def _parse(rows, year):
    """→ {회계연도: (DPS, 사용 시작일 'YYYY-MM-DD')}"""
    if not rows:
        return {}
    rc = str(rows[0].get("rcept_no", ""))[:8]
    try:
        avail = pd.Timestamp(rc) + pd.Timedelta(days=1)
    except Exception:
        avail = pd.Timestamp(year + 1, 4, 1)
    dps = None
    for r in rows:
        se = str(r.get("se", "")).replace(" ", "")
        kind = str(r.get("stock_knd", "")).replace(" ", "")
        if "주당현금배당금" in se and (kind.startswith("보통") or kind == ""):
            dps = (_num(r.get("thstrm")), _num(r.get("frmtrm")), _num(r.get("lwfr")))
            break
    if dps is None:                      # 배당 표는 있는데 현금배당 줄이 없음 → 무배당
        dps = (0.0, np.nan, np.nan)
    out = {}
    for k, v in enumerate(dps):
        if np.isfinite(v):
            out[year - k] = (v, (avail - pd.DateOffset(years=k)).strftime("%Y-%m-%d"))
    return out


def load(codes, key, corp, minutes=15, workers=8):
    """{종목코드: {회계연도: (DPS, 사용 시작일)}}"""
    cache = _load()
    now = time.time()
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    y1 = today.year - 1 if today.month >= 4 else today.year - 2       # 가장 최근 사업보고서 연도
    jobs = []
    for c in codes:
        if c not in corp:
            continue
        d = cache.setdefault(c, {"y": {}, "got": {}})
        for y in (y1, y1 - 3):
            g = d["got"].get(y)
            # 최근 연도는 사업보고서가 늦게 나올 수 있어 없으면 7일마다 다시 확인
            if g is None or (g == "none" and y == y1 and now - d.get("at", 0) > 7 * 86400):
                jobs.append((0 if y == y1 else 1, c, y))
    jobs.sort()
    print(f"배당 요청 {len(jobs)}건 (최근 {sum(1 for j in jobs if j[0] == 0)}), 최대 {minutes}분")

    def job(j):
        _, c, y = j
        st, rows = _get(key, corp[c], y)
        return c, y, st, rows

    t0, done, chunk = time.time(), 0, workers * 25
    try:
        with ThreadPoolExecutor(workers) as ex:
            for i in range(0, len(jobs), chunk):
                if time.time() - t0 > minutes * 60:
                    print(f"  배당 시간 한도 — 나머지 {len(jobs) - i}건은 다음 실행 때")
                    break
                for c, y, st, rows in ex.map(job, jobs[i:i + chunk]):
                    if st == "err":
                        continue
                    d = cache[c]
                    d["got"][y] = st
                    d["at"] = now
                    for fy, v in _parse(rows, y).items():
                        if fy not in d["y"] or fy == y:          # 그 해 보고서 값이 우선
                            d["y"][fy] = v
                    done += 1
                _save(cache)
    except Limit as e:
        print(f"  DART 하루 요청 한도 초과 — 받은 데까지만 사용: {e}")
    _save(cache)
    n = sum(1 for c in codes if cache.get(c, {}).get("y"))
    print(f"배당 이번 {done}건, 배당 기록 있는 종목 {n} ({time.time() - t0:.0f}초)")
    return {c: cache[c]["y"] for c in codes if cache.get(c, {}).get("y")}


def _load():
    if os.path.exists(CACHE):
        try:
            with gzip.open(CACHE, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            print(f"배당 캐시 읽기 실패: {e}")
    return {}


def _save(cache):
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with gzip.open(CACHE, "wb") as f:
        pickle.dump(cache, f)
