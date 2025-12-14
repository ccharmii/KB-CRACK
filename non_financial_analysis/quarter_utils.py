# /KB-CRACK/non_financial_analysis/quarter_utils.py
# 보고서 제목 및 접수일 기반 분기 추론 유틸

import re
from datetime import date
from typing import Optional, Tuple


def _ym_to_quarter(y: int, m: int) -> str:
    """연월을 분기 문자열로 변환 수행"""
    q = (m - 1) // 3 + 1
    return f"{y}Q{q}"


def _extract_year(nm: str) -> Optional[int]:
    """문자열에서 연도 추출 수행"""
    m = re.search(r"(20\d{2})", nm or "")
    return int(m.group(1)) if m else None


def _extract_ym_candidates(nm: str) -> list[Tuple[int, int]]:
    """문자열에서 연월 후보 목록 추출 수행"""
    s = nm or ""
    pats = [
        r"(20\d{2})\s*[.\-\/]\s*(0?[1-9]|1[0-2])",
        r"(20\d{2})\s*년\s*(0?[1-9]|1[0-2])\s*월",
        r"\(\s*(20\d{2})\s*[.\-\/]\s*(0?[1-9]|1[0-2])\s*\)",
    ]

    out: list[Tuple[int, int]] = []
    for p in pats:
        for m in re.finditer(p, s):
            y = int(m.group(1))
            mm = int(m.group(2))
            out.append((y, mm))

    seen = set()
    uniq: list[Tuple[int, int]] = []
    for y, mm in out:
        if (y, mm) in seen:
            continue
        seen.add((y, mm))
        uniq.append((y, mm))

    return uniq


def _pick_best_ym(nm: str) -> Optional[Tuple[int, int]]:
    """연월 후보 중 분기말 월 우선으로 최적 연월 선택 수행"""
    cands = _extract_ym_candidates(nm)
    if not cands:
        return None

    for y, m in cands:
        if m in (3, 6, 9, 12):
            return y, m

    return cands[0]


def infer_quarter(report_nm: str, detail_ty: str, rcept_dt: str) -> str:
    """
    보고서 타입과 제목 및 접수일 기반 귀속 분기 추론 수행

    Args:
        report_nm: 보고서 제목 문자열
        detail_ty: DART 보고서 타입 코드
        rcept_dt: 접수일 문자열 YYYYMMDD

    Returns:
        귀속 분기 문자열 반환
    """
    nm = report_nm or ""
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

        q_rcpt = (m_rcpt - 1) // 3 + 1
        q = q_rcpt - 1
        y = y_rcpt

        if q == 0:
            q, y = 4, y - 1

        return f"{y}Q{q}"

    return _ym_to_quarter(y_rcpt, m_rcpt)


def last_n_complete_quarters(n: int = 4, asof: Optional[date] = None) -> list[str]:
    """기준일 이전 완료된 최근 n개 분기 목록 생성 수행"""
    today = asof or date.today()
    y, m = today.year, today.month
    q = (m - 1) // 3 + 1

    q -= 1
    if q == 0:
        y, q = y - 1, 4

    out: list[str] = []
    for _ in range(n):
        out.append(f"{y}Q{q}")
        q -= 1
        if q == 0:
            y, q = y - 1, 4

    return out
