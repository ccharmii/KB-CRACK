# /KB-CRACK/non_financial_analysis/json_out.py
# 비재무 분석 분기별 결과 JSON 출력 로직

import json
import os
from datetime import datetime


def _ensure_dir(p: str) -> None:
    """디렉토리 미존재 시 생성 수행"""
    os.makedirs(p, exist_ok=True)


def write_quarter_json(
    corp_root: str,
    corp_code: str,
    corp_name: str,
    quarter: str,
    indicators,
):
    """
    분기별 비재무 분석 결과를 JSON 파일로 저장
    Args:
        corp_root: 기업 루트 디렉토리 경로
        corp_code: 기업 코드
        corp_name: 기업명
        quarter: 분기 문자열
        indicators: 분기별 비재무 지표 결과 목록
    Returns:
        저장된 파일 경로와 JSON 객체 반환
    """
    _ensure_dir(os.path.join(corp_root, "results"))

    obj = {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "quarter": quarter,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "indicators": indicators,
    }

    path = os.path.join(corp_root, "results", f"{quarter}_nfr_scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    return path, obj


def append_global_jsonl(result_dir: str, quarter_obj: dict) -> str:
    """
    분기별 결과 객체를 전역 JSONL 파일에 한 줄로 추가
    Args:
        result_dir: 전역 결과 저장 디렉토리 경로
        quarter_obj: 분기별 결과 JSON 객체
    Returns:
        JSONL 파일 경로 반환
    """
    _ensure_dir(result_dir)

    path = os.path.join(result_dir, "nfr_scores_all.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(quarter_obj, ensure_ascii=False) + "\n")

    return path
