"""OpenDART 분기·연간 실적 → 마법공식용 지표 시계열 (공시 접수일 다음 날부터 사용해 미래 정보가 섞이지 않게 함)

- 다중회사 주요계정 API(fnlttMultiAcnt)로 한 번에 100개 회사씩 받음 (오닐 앱과 같은 방식)
- 연결재무제표가 있으면 연결, 없으면 별도
- 손익: 영업이익(EBIT 대용)·당기순이익을 분기값으로 바꾼 뒤 최근 4분기 합(TTM)
- 재무상태표: 자산총계·부채총계·유동자산·유동부채 (그 보고서 기준 시점 값)
- FCF: cashflow.py (단일회사 전체 재무제표의 현금흐름표)
금액 단위는 억 원.
"""
import os, io, time, gzip, pickle, zipfile
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "fund.pkl.gz")
API = "https://opendart.fss.or.kr/api"
REPRTS = ("11013", "11012", "11014", "11011")          # 1분기, 반기, 3분기, 사업보고서
Q_END = {"11013": (3, 31), "11012": (6, 30), "11014": (9, 30), "11011": (12, 31)}
FLOW = ("op", "ni")                                    # 손익 (기간 값)
STOCK = ("ta", "tl", "ca", "cl", "eq")                 # 재무상태표 (시점 값)
YEARS_BACK = 6
CORP = {}


def _num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan


def _kind(nm):
    n = str(nm).replace(" ", "")
    if n.startswith(("당기순이익", "분기순이익", "반기순이익", "연결당기순이익")):
        return "ni"
    if n.startswith("영업이익"):
        return "op"
    if n.startswith("자산총계"):
        return "ta"
    if n.startswith("부채총계"):
        return "tl"
    if n.startswith("유동자산"):
        return "ca"
    if n.startswith("유동부채"):
        return "cl"
    if n.startswith("자본총계"):
        return "eq"
    return None


def corp_codes(key):
    r = requests.get(f"{API}/corpCode.xml", params={"crtfc_key": key}, timeout=90)
    z = zipfile.ZipFile(io.BytesIO(r.content))
    root = ET.fromstring(z.read(z.namelist()[0]))
    m = {}
    for el in root.iter("list"):
        sc = (el.findtext("stock_code") or "").strip()
        if sc:
            m[sc] = el.findtext("corp_code").strip()
    return m


class Limit(Exception):
    pass


def _fetch(key, corps, year, reprt):
    params = {"crtfc_key": key, "corp_code": ",".join(corps), "bsns_year": str(year), "reprt_code": reprt}
    for attempt in range(4):
        try:
            j = requests.get(f"{API}/fnlttMultiAcnt.json", params=params, timeout=90).json()
            st = j.get("status")
            if st == "000":
                return j.get("list", [])
            if st == "013":                      # 데이터 없음
                return []
            if st == "020":                      # 요청 제한 초과
                raise Limit(j.get("message"))
            print(f"  DART {year}/{reprt}: {st} {j.get('message')}")
            return None
        except Limit:
            raise
        except Exception:
            time.sleep(2 + attempt * 2)
    return None


def _parse(rows):
    """API 응답 → {종목코드: {rcept, op, op_cum, ni, ni_cum, ta, tl, ca, cl, eq}} (원 단위)"""
    out = {}
    if not rows:
        return out
    df = pd.DataFrame(rows)
    df["kind"] = df["account_nm"].map(_kind)
    df = df[df["kind"].notna() & df["stock_code"].astype(str).str.strip().ne("")]
    for code, g in df.groupby(df["stock_code"].astype(str).str.strip()):
        fs = "CFS" if (g["fs_div"] == "CFS").any() else "OFS"
        g = g[g["fs_div"] == fs]
        if not len(g):
            continue
        rec = {"rcept": str(g["rcept_no"].iloc[0])[:8]}
        for k, gg in g.groupby("kind"):
            r = gg.iloc[0]
            rec[k] = _num(r.get("thstrm_amount"))
            if k in FLOW:
                rec[k + "_cum"] = _num(r.get("thstrm_add_amount"))
        out[code] = rec
    return out


def load_raw(codes, key, refresh_hours=20):
    """(연도, 보고서) 단위로 받아서 캐시. 최근 2개 연도는 하루 한 번 새로 받음"""
    cache = {"raw": {}, "at": {}, "corp": None, "corp_at": 0, "v": 2}
    if os.path.exists(CACHE):
        try:
            with gzip.open(CACHE, "rb") as f:
                c = pickle.load(f)
            if c.get("v") == 2:
                cache = c
        except Exception as e:
            print(f"실적 캐시 읽기 실패: {e}")
    now = time.time()
    if not cache.get("corp") or now - cache.get("corp_at", 0) > 7 * 86400:
        try:
            cache["corp"], cache["corp_at"] = corp_codes(key), now
        except Exception as e:
            print(f"DART 회사코드 실패: {e}")
            if not cache.get("corp"):
                return None
    corp = cache["corp"]
    global CORP
    CORP = corp
    corps = [corp[c] for c in codes if c in corp]
    print(f"DART 회사코드 매칭 {len(corps)}/{len(codes)}")
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    y1 = today.year
    todo = []
    for y in range(y1 - YEARS_BACK, y1 + 1):
        for rp in REPRTS:
            mo, dd = Q_END[rp]
            if pd.Timestamp(y, mo, dd) + pd.Timedelta(days=14) > today:
                continue
            k = (y, rp)
            stale = y >= y1 - 1 and now - cache["at"].get(k, 0) > refresh_hours * 3600
            if k not in cache["raw"] or stale:
                todo.append(k)
    batches = [corps[i:i + 100] for i in range(0, len(corps), 100)]
    print(f"DART 요청 {len(todo)}개 보고서 × {len(batches)}묶음")
    t0 = time.time()

    def save():
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with gzip.open(CACHE, "wb") as f:
            pickle.dump(cache, f)

    for n, k in enumerate(todo, 1):
        try:
            with ThreadPoolExecutor(8) as ex:
                res = list(ex.map(lambda b: _fetch(key, b, k[0], k[1]), batches))
        except Limit as e:
            print(f"DART 하루 요청 한도 초과 — 받은 데까지만 사용: {e}")
            break
        if any(r is None for r in res):
            print(f"  {k} 일부 실패 — 다음 실행 때 다시 받음")
            if k in cache["raw"]:
                continue
        rows = [x for r in res if r for x in r]
        cache["raw"][k] = _parse(rows)
        if not any(r is None for r in res):
            cache["at"][k] = now
        print(f"  DART {k[0]}/{k[1]} {len(cache['raw'][k])}종목 ({n}/{len(todo)}, {time.time() - t0:.0f}초)")
        save()                               # 중간 저장: 시간 초과로 끊겨도 다음 실행은 이어서
    print(f"DART 수집 {time.time() - t0:.0f}초")
    save()
    return cache["raw"]


def _cum_ttm(cf, y, qn, k):
    """연초부터 누적 값 → 최근 4분기 합 (원). 직전 연간 + 올해 누적 − 작년 같은 기간 누적"""
    if not cf:
        return np.nan
    g = lambda yy, qq: cf.get((yy, qq), {}).get(k, np.nan)
    if qn == 4:
        return g(y, 4)
    return g(y - 1, 4) + g(y, qn) - g(y - 1, qn)


def series(raw, code, cf=None):
    """종목 하나의 지표를 공시일(다음 날) 기준 시계열로.
    열: ebit, ni, fcf, nip(지배주주 순이익) (최근 4분기 합), ta, tl, ca, cl, eq, eqp(지배주주 자본) (억 원), ql"""
    Q = {}
    years = sorted({y for y, _ in raw})
    for y in years:
        cum = {}
        for qn, rp in enumerate(REPRTS, 1):
            rec = raw.get((y, rp), {}).get(code)
            if rec is None:
                continue
            avail = pd.Timestamp(rec["rcept"]) + pd.Timedelta(days=1)
            vals = {}
            for k in FLOW:
                amt, add = rec.get(k, np.nan), rec.get(k + "_cum", np.nan)
                prev = cum.get((qn - 1, k), np.nan)
                if qn == 1:
                    c, qv = amt, amt
                elif qn < 4:
                    c = add if np.isfinite(add) else prev + amt
                    qv = add - prev if (np.isfinite(add) and amt == add) else amt
                else:                                # 사업보고서 = 연간 값
                    c, qv = amt, amt - cum.get((3, k), np.nan)
                cum[(qn, k)] = c
                vals[k] = qv
            Q[y * 4 + qn - 1] = {**vals, **{k: rec.get(k, np.nan) for k in STOCK},
                                 "avail": avail, "ql": f"{y % 100:02d}.{qn}Q"}
    rows, top = [], -1
    for q in sorted(Q, key=lambda q: (Q[q]["avail"], q)):
        if q < top:                                  # 늦게 낸 옛 분기 보고서는 무시 (최신 분기를 덮지 않게)
            continue
        top = q
        four = [Q.get(q - i) for i in range(4)]
        r = {"avail": Q[q]["avail"], "ql": Q[q]["ql"]}
        for k, out in (("op", "ebit"), ("ni", "ni")):
            v = [x[k] for x in four if x is not None]
            r[out] = sum(v) / 1e8 if len(v) == 4 and all(np.isfinite(v)) else np.nan
        for k in STOCK:
            v = Q[q][k]
            r[k] = v / 1e8 if np.isfinite(v) else np.nan
        y, qn = q // 4, q % 4 + 1
        for k in ("fcf", "nip"):
            v = _cum_ttm(cf, y, qn, k)
            r[k] = v / 1e8 if np.isfinite(v) else np.nan
        # 안전장치: 지배주주 순이익이 연결 순이익보다 30% 넘게 크면 잘못 읽은 값일 가능성 → 연결 값 사용
        if np.isfinite(r["nip"]) and np.isfinite(r["ni"]) and r["ni"] > 0 and r["nip"] > r["ni"] * 1.3:
            r["nip"] = np.nan
        v = (cf or {}).get((y, qn), {}).get("eqp", np.nan)
        r["eqp"] = v / 1e8 if np.isfinite(v) else np.nan
        rows.append(r)
    if not rows:
        return None
    df = pd.DataFrame(rows).sort_values("avail")
    return df.drop_duplicates("avail", keep="last").set_index("avail")


def load(codes):
    """{종목코드: 지표 시계열}. 키가 없거나 실패하면 None"""
    key = os.environ.get("DART_API_KEY", "").strip()
    if not key:
        print("DART_API_KEY 없음 — 마법공식은 재무 데이터가 꼭 필요합니다")
        return None
    print(f"DART 키 확인: 길이 {len(key)}자 (정상은 40자)")
    try:
        raw = load_raw(codes, key)
    except Exception as e:
        print(f"DART 실패: {e}")
        return None
    if not raw:
        return None
    cfd = {}
    mins = float(os.environ.get("CF_MINUTES", "45"))
    if mins > 0:
        try:
            import cashflow
            cfd = cashflow.load(codes, key, CORP, raw, minutes=mins)
        except Exception as e:
            print(f"현금흐름 실패 — FCF 없이 진행: {e}")
    out = {}
    for c in codes:
        s = series(raw, c, cfd.get(c))
        if s is not None:
            out[c] = s
    print(f"재무 시계열 {len(out)}종목")
    return out
