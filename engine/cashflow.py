"""OpenDART 단일회사 전체 재무제표(fnlttSinglAcntAll)에서 FCF와 지배주주 지표 계산

FCF = 영업활동현금흐름 − (유형자산 취득 + 무형자산 취득)
- 현금흐름표는 분기·반기 보고서도 연초부터 누적 값이라, 최근 4분기 합(TTM)은
  직전 사업보고서 + 올해 누적 − 작년 같은 기간 누적 으로 계산
- 회사·보고서마다 요청 1번이 필요해 전체를 한 번에 받을 수 없음 (5년치 약 6만 번, 하루 한도 2만 번)
  → 오늘 TTM에 필요한 보고서부터 받고, 나머지는 실행 때마다 정해진 시간만큼 최신→과거 순으로 채움
- 주요계정 API에 그 보고서가 있는 종목만 요청 (보고서를 안 낸 종목에 헛요청 안 함)
- 같은 응답에서 지배주주 순이익(누적)·지배주주 자본도 뽑음 → PER·PBR을 비지배지분 없이 계산
  (증권사·계열사를 연결로 잡는 지주회사는 연결 순이익·자본이 실제 주주 몫보다 훨씬 큼)
"""
import os, time, gzip, pickle
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "cache", "cf.pkl.gz")
API = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.json"
REPRTS = ("11013", "11012", "11014", "11011")
QN = {rp: i for i, rp in enumerate(REPRTS, 1)}
Q_END = {1: (3, 31), 2: (6, 30), 3: (9, 30), 4: (12, 31)}
OCF_ID = ("ifrs-full_CashFlowsFromUsedInOperatingActivities", "ifrs_CashFlowsFromUsedInOperatingActivities")
PPE_ID = ("ifrs-full_PurchaseOfPropertyPlantAndEquipment", "ifrs_PurchaseOfPropertyPlantAndEquipment")
INT_ID = ("ifrs-full_PurchaseOfIntangibleAssets", "ifrs_PurchaseOfIntangibleAssets")
NIP_ID = ("ifrs-full_ProfitLossAttributableToOwnersOfParent", "ifrs_ProfitLossAttributableToOwnersOfParent")
EQP_ID = ("ifrs-full_EquityAttributableToOwnersOfParent", "ifrs_EquityAttributableToOwnersOfParent")
NI_ID = ("ifrs-full_ProfitLoss", "ifrs_ProfitLoss")
EQ_ID = ("ifrs-full_Equity", "ifrs_Equity")
VER = 3                                   # 2: 지배주주 순이익·자본 포함, 3: 총포괄이익 줄 오인 수정


class Limit(Exception):
    pass


def _num(s):
    try:
        return float(str(s).replace(",", "").strip())
    except Exception:
        return np.nan


def _pick(rows, qn=4):
    """전체 재무제표 행들 → {ocf, capex, nip, eqp} (원). 계정 ID 우선, 없으면 계정명
    nip = 지배주주 순이익 (연초부터 누적), eqp = 지배주주 자본. 별도재무제표면 순이익·자본 전체"""
    ocf = ppe = itg = nip = eqp = ni = eq = np.nan
    nip_nm = np.nan                       # 계정 ID 없이 이름으로만 찾은 값 (ID로 못 찾을 때만 씀)
    in_ci = {}                            # 손익계산서별: 총포괄이익 구간에 들어섰는지
    for r in rows:
        sj = r.get("sj_div")
        aid = str(r.get("account_id", "")).strip()
        nm = str(r.get("account_nm", "")).replace(" ", "")
        if sj in ("IS", "CIS"):
            if "포괄" in nm or "ComprehensiveIncome" in aid or "OtherComprehensive" in aid:
                in_ci[sj] = True          # 이후 '지배기업 소유주지분'은 총포괄이익 귀속 → 순이익 아님
            v = _num(r.get("thstrm_add_amount")) if qn in (2, 3) else _num(r.get("thstrm_amount"))
            if qn in (2, 3) and not np.isfinite(v):
                continue
            if aid in NIP_ID:
                if not np.isfinite(nip):
                    nip = v
            elif (not np.isfinite(nip_nm) and not in_ci.get(sj) and "Comprehensive" not in aid
                  and nm.startswith(("지배기업의소유주", "지배기업소유주", "지배주주"))):
                nip_nm = v
            elif not np.isfinite(ni) and (aid in NI_ID or nm.startswith(("당기순이익", "분기순이익", "반기순이익"))):
                ni = v
            continue
        if sj == "BS":
            v = _num(r.get("thstrm_amount"))
            if not np.isfinite(eqp) and (aid in EQP_ID or nm.startswith(("지배기업의소유주", "지배기업소유주", "지배주주"))):
                eqp = v
            elif not np.isfinite(eq) and (aid in EQ_ID or nm == "자본총계"):
                eq = v
            continue
        if sj != "CF":
            continue
        v = _num(r.get("thstrm_amount"))
        if not np.isfinite(v):
            continue
        if not np.isfinite(ocf) and (aid in OCF_ID or (nm.startswith("영업활동") and "현금흐름" in nm)):
            ocf = v
        elif not np.isfinite(ppe) and (aid in PPE_ID or nm.startswith("유형자산의취득")):
            ppe = abs(v)
        elif not np.isfinite(itg) and (aid in INT_ID or nm.startswith("무형자산의취득")):
            itg = abs(v)
    capex = (0 if not np.isfinite(ppe) else ppe) + (0 if not np.isfinite(itg) else itg)
    if not np.isfinite(nip):
        nip = nip_nm
    return {"ocf": ocf, "capex": capex, "v": VER,
            "nip": nip if np.isfinite(nip) else ni, "eqp": eqp if np.isfinite(eqp) else eq}


def _get(key, corp, y, rp, fs):
    params = {"crtfc_key": key, "corp_code": corp, "bsns_year": str(y), "reprt_code": rp, "fs_div": fs}
    for attempt in range(3):
        try:
            j = requests.get(API, params=params, timeout=60).json()
            st = j.get("status")
            if st == "000":
                rows = j.get("list", [])
                rec = _pick(rows, QN[rp])
                rec["rcept"] = str(rows[0].get("rcept_no", ""))[:8] if rows else ""
                return "ok", rec
            if st == "013":
                return "none", None
            if st == "020":
                raise Limit(j.get("message"))
            return "err", None
        except Limit:
            raise
        except Exception:
            time.sleep(1 + attempt * 2)
    return "err", None


def load(codes, key, corp, multi_raw, minutes=45, workers=12):
    """{종목코드: {(연도, 분기번호): {fcf 누적, nip 누적, eqp 시점}}} (원)"""
    cache = {"d": {}, "fs": {}, "v": 1}
    if os.path.exists(CACHE):
        try:
            with gzip.open(CACHE, "rb") as f:
                cache = pickle.load(f)
        except Exception as e:
            print(f"현금흐름 캐시 읽기 실패: {e}")
    d, fsm = cache["d"], cache["fs"]
    now = time.time()
    today = pd.Timestamp.now(tz="Asia/Seoul").tz_localize(None)
    cset = set(codes)

    # 할 일: 주요계정 API에 보고서가 있는 (종목, 연도, 보고서)
    reps = sorted(multi_raw, key=lambda k: (k[0], QN[k[1]]), reverse=True)
    big = [k for k in reps if len(multi_raw[k]) >= 500]
    prio = set()
    if big:
        ly, lq = big[0][0], QN[big[0][1]]
        prio = {(ly, lq), (ly - 1, 4), (ly - 1, lq)}
    jobs = []
    for y, rp in reps:
        qn = QN[rp]
        mo, dd = Q_END[qn]
        recent = (today - pd.Timestamp(y, mo, dd)).days < 150
        for code in multi_raw[(y, rp)]:
            if code not in cset or code not in corp:
                continue
            k = (code, y, qn)
            v = d.get(k)
            old = v is not None and "ocf" in v and v.get("v", 1) < VER      # 예전 형식 → 다시 받음
            if v is None or old or ("miss" in v and recent and now - v["miss"] > 20 * 3600):
                jobs.append((0 if (y, qn) in prio else 1, -(y * 4 + qn), k, rp))
    jobs.sort()
    print(f"현금흐름 남은 요청 {len(jobs)}건 (오늘 기준 FCF용 {sum(1 for j in jobs if j[0] == 0)}건), 최대 {minutes}분")

    def job(j):
        code, y, qn = j[2]
        first = fsm.get(code, "CFS")
        for fs in (first, "OFS" if first == "CFS" else "CFS"):
            st, rec = _get(key, corp[code], y, j[3], fs)
            if st == "ok":
                return j[2], fs, rec
            if st == "err":
                return j[2], None, None
        return j[2], None, {"miss": now}

    def save():
        os.makedirs(os.path.dirname(CACHE), exist_ok=True)
        with gzip.open(CACHE, "wb") as f:
            pickle.dump(cache, f)

    t0, done, chunk = time.time(), 0, workers * 20
    try:
        with ThreadPoolExecutor(workers) as ex:
            for i in range(0, len(jobs), chunk):
                if time.time() - t0 > minutes * 60:
                    print(f"  현금흐름 시간 한도 — 나머지 {len(jobs) - i}건은 다음 실행 때")
                    break
                for k, fs, rec in ex.map(job, jobs[i:i + chunk]):
                    if rec is None:
                        continue
                    d[k] = rec
                    if fs:
                        fsm[k[0]] = fs
                    done += 1
                save()
                if (i // chunk) % 5 == 4:
                    print(f"  현금흐름 {i + chunk}/{len(jobs)} ({time.time() - t0:.0f}초)")
    except Limit as e:
        print(f"  DART 하루 요청 한도 초과 — 받은 데까지만 사용: {e}")
    save()
    n2 = sum(1 for v in d.values() if v.get("v", 1) >= VER)
    print(f"현금흐름 이번 실행 {done}건, 누적 {sum(1 for v in d.values() if 'ocf' in v)}건 (지배주주 포함 {n2}건) ({time.time() - t0:.0f}초)")

    out = {}
    for (code, y, qn), v in d.items():
        if "ocf" not in v:
            continue
        out.setdefault(code, {})[(y, qn)] = {
            "fcf": v["ocf"] - v["capex"] if np.isfinite(v["ocf"]) else np.nan,
            "nip": v.get("nip", np.nan), "eqp": v.get("eqp", np.nan)}
    return out
