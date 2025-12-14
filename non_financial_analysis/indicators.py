# /KB-CRACK/non_financial_analysis/indicators.py
# 비재무 평가지표 정의 로드 로직

from typing import Any, Dict, List

from .config import INDICATORS


def load_indicators() -> List[Dict[str, Any]]:
    """
    INDICATORS 설정을 표준 딕셔너리 리스트로 변환 반환
    Returns:
        pillar, id, name, desc, cues를 포함한 지표 목록 반환
    """
    out: List[Dict[str, Any]] = []
    for item in INDICATORS:
        if not isinstance(item, dict):
            continue

        out.append(
            {
                "pillar": item.get("pillar", ""),
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "desc": item.get("desc", ""),
                "cues": list(item.get("cues") or []),
            }
        )

    return out
