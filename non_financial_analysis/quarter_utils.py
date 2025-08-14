import re
from datetime import date
from typing import Optional, Tuple

# 월 -> 분기 계산 "YYYYQn" 반환
def _ym_to_quarter(y, m):
    q = (m - 1) // 3 + 1
    return f"{y}Q{q}"

# 제목에서 20xx 연도 추출
def _extract_year(nm):
    m = re.search(r"(20\d{2})", nm or "")
    return int(m.group(1)) if m else None

# 제목에서 여러 후보로 추출
def _extract_ym_candidates(nm):
    s = (nm or "")
    pats = [
        r"(20\d{2})\s*[.\-\/]\s*(0?[1-9]|1[0-2])", 
        r"(20\d{2})\s*년\s*(0?[1-9]|1[0-2])\s*월", 
        r"\(\s*(20\d{2})\s*[.\-\/]\s*(0?[1-9]|1[0-2])\s*\)",
    ]
    out = []
    for p in pats:
        for m in re.finditer(p, s):
            y = int(m.group(1)); mm = int(m.group(2))
            out.append((y, mm))
    
    # 중복 제거
    seen = set()
    uniq = []
    for y, mm in out:
        if (y, mm) not in seen:
            seen.add((y, mm)); uniq.append((y, mm))
    return uniq

# 후보 중 분기말 월(3/6/9/12) 우선, 없으면 첫 후보
def _pick_best_ym(nm):
    cands = _extract_ym_candidates(nm)
    if not cands: return None
    for y, m in cands:
        if m in (3, 6, 9, 12):
            return (y, m)
    return cands[0]

# DART의 보고서 타입(A001/2/3)과 제목/접수일을 조합해 귀속 분기 판정
def infer_quarter(report_nm, detail_ty, rcept_dt):
    """
    A001(사업): 제목에 월 있으면 그 분기, 연도만이면 그 해 Q4, 없으면 접수연도-1 Q4
    A002(반기): 제목에 월 있으면 그 분기, 연도만이면 그 해 Q2, 없으면 접수연도 Q2
    A003(분기): “제1/2/3/4분기” 키워드가 최우선(연도 없으면 접수연도), 그 외 월 표기 있으면 그 분기, 없다면 접수 분기-1
    """
    nm = (report_nm or "")
    nm_compact = nm.replace(" ", "")
    detail = (detail_ty or "").upper()
    y_rcpt = int((rcept_dt or "19000101")[:4] or 1900)
    m_rcpt = int((rcept_dt or "19000101")[4:6] or 1)

    best = _pick_best_ym(nm)

    if detail != "A003" and best:
        return _ym_to_quarter(best[0], best[1])

    if detail == "A001":
        y = _extract_year(nm) or (best[0] if best else None) or (y_rcpt - 1)
        if best: 
            return _ym_to_quarter(best[0], best[1])
        return f"{y}Q4"

    if detail == "A002": 
        if best:
            return _ym_to_quarter(best[0], best[1])
        y = _extract_year(nm) or y_rcpt
        return f"{y}Q2"
 
    if detail == "A003": 
        y_in_title = _extract_year(nm) or (best[0] if best else None)
        if "제1분기" in nm_compact or "1분기" in nm_compact:
            return f"{(y_in_title or y_rcpt)}Q1"
        if "제2분기" in nm_compact or "2분기" in nm_compact:
            return f"{(y_in_title or y_rcpt)}Q2"
        if "제3분기" in nm_compact or "3분기" in nm_compact:
            return f"{(y_in_title or y_rcpt)}Q3"
        if "제4분기" in nm_compact or "4분기" in nm_compact:
            return f"{(y_in_title or y_rcpt)}Q4"
        if best:
            return _ym_to_quarter(best[0], best[1])
        
        # 접수 분기-1
        q_rcpt = (m_rcpt - 1)//3 + 1
        q = q_rcpt - 1
        y = y_rcpt

        if q == 0:
            q, y = 4, y - 1

        return f"{y}Q{q}"
    return _ym_to_quarter(y_rcpt, m_rcpt)

# 기준일 이전의 완료된 최근 n개 분기를 역순으로 반환
def last_n_complete_quarters(n=4, asof=None):
    today = asof or date.today()
    y, m = today.year, today.month
    q = (m - 1)//3 + 1

    # 완료된 직전, 가장 최근 분기부터
    q -= 1
    if q == 0:
        y, q = y - 1, 4
    out = []
    for _ in range(n):
        out.append(f"{y}Q{q}")
        q -= 1
        if q == 0:
            y, q = y - 1, 4
    return out
