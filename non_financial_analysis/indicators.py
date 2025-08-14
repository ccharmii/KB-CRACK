from typing import List, Dict, Any
from .config import INDICATORS

def load_indicators():
    """
    목적: 
    1. 검색 쿼리 생성을 돕기 위해
    2. 프롬프트에 무엇을 평가할지를 명확히 주입하기 위해

    INDICATORS -> dict:  {"pillar","id","name","desc","cues":[...]}  
    """
    out: List[Dict[str, Any]] = []
    for item in INDICATORS:
        if isinstance(item, dict):
            out.append({
                "pillar": item.get("pillar", ""),
                "id": item.get("id", ""),
                "name": item.get("name", ""),
                "desc": item.get("desc", ""),
                "cues": list(item.get("cues")),
            })
    return out
