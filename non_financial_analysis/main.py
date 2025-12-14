# /KB-CRACK/non_financial_analysis/main.py
# 비재무지표 분기별 평가 및 결과 저장 파이프라인

import json
import os
import time
import uuid
from datetime import date, datetime
from typing import Dict, List

from .config import DATA_DIR, DB_FILE
from .dart_api import ensure_report_files, list_regular_reports
from .db import insert_chunks, insert_scores_json, open_db, upsert_filings
from .evaluator import evaluate_quarter
from .indexer import build_or_load_faiss, get_retriever, ingest_texts
from .indicators import load_indicators
from .json_out import write_quarter_json


def corp_root_path(code: str) -> str:
    """기업별 데이터 디렉토리 경로 생성"""
    return os.path.join(DATA_DIR, code)


def _qkey(q: str) -> tuple[int, int]:
    """분기 문자열 정렬 키 생성"""
    return int(q[:4]), int(q[-1])


def _result_path(corp_root: str, quarter: str) -> str:
    """분기별 결과 JSON 파일 경로 생성"""
    res_dir = os.path.join(corp_root, "results")
    os.makedirs(res_dir, exist_ok=True)
    return os.path.join(res_dir, f"{quarter}_nfr_scores.json")


def _result_exists(corp_root: str, quarter: str) -> bool:
    """분기별 결과 파일 존재 여부 판단"""
    p = _result_path(corp_root, quarter)
    return os.path.exists(p) and os.path.getsize(p) > 10


def run_for_corp(corp_code: str, asof: date | None = None, force: bool = False) -> Dict[str, object]:
    """
    기업별 비재무지표 분석 파이프라인 수행
    Args:
        corp_code: DART 기업코드 8자리
        asof: 기준일
        force: 기존 결과 무시 및 재실행 여부
    Returns:
        비재무분석 결과 딕셔너리 반환
    """
    if asof is None:
        asof = date.today()

    print(f"📋 비재무지표 분석 시작: {corp_code} (기준일: {asof.isoformat()})")
    start_time = time.perf_counter()

    try:
        print("  - 정기보고서 수집 중...")
        all_filings = list_regular_reports(corp_code, asof=asof)

        if not all_filings:
            return {
                "success": False,
                "error": "조회 결과가 없습니다",
                "corp_code": corp_code,
                "analysis_date": datetime.now().isoformat(),
            }

        corp_name = all_filings[0].get("corp_name", "")

        by_q: Dict[str, List[Dict]] = {}
        for f in all_filings:
            by_q.setdefault(f["quarter"], []).append(f)

        quarters_avail = sorted(by_q.keys(), key=_qkey, reverse=True)
        targets = quarters_avail[:4]

        print(f"  - 대상 분기: {targets}")

        corp_root = corp_root_path(corp_code)
        os.makedirs(corp_root, exist_ok=True)

        target_filings = [f for q in targets for f in by_q[q]]
        saved = ensure_report_files(corp_root, target_filings)

        for s in saved:
            s["meta"]["path"] = s["path"]

        has_new = any(s.get("is_new") for s in saved)
        print(f"  - 파일 저장: {len(saved)}개 (신규: {sum(1 for s in saved if s.get('is_new'))}개)")

        db = open_db(os.path.join(corp_root, DB_FILE))
        upsert_filings(db, [s["meta"] for s in saved])

        print("  - 텍스트 인덱싱...")
        vs = build_or_load_faiss(corp_root)

        if vs is None or has_new:
            vs, texts, metas = ingest_texts(corp_root, saved, corp_code)
            rows = [
                {
                    "id": m["chunk_id"],
                    "rcept_no": m["rcept_no"],
                    "quarter": m["quarter"],
                    "corp_code": corp_code,
                    "start": 0,
                    "end": len(t),
                    "content": t,
                }
                for t, m in zip(texts, metas)
            ]
            if rows:
                insert_chunks(db, rows)

        indicators = load_indicators()
        print(f"  - 비재무지표 평가 ({len(indicators)}개 지표)...")

        evaluation_results: Dict[str, List[Dict]] = {}

        for q in targets:
            if _result_exists(corp_root, q) and not force:
                print(f"    • {q}: 기존 결과 재사용")
                try:
                    with open(_result_path(corp_root, q), "r", encoding="utf-8") as f:
                        result_data = json.load(f)
                    evaluation_results[q] = result_data.get("indicators", [])
                except Exception:
                    evaluation_results[q] = []
                continue

            if q not in by_q or not by_q[q]:
                print(f"    • {q}: 문서 없음")
                continue

            print(f"    • {q} 평가 중...")
            retriever = get_retriever(vs, quarter=q)
            res = evaluate_quarter(retriever, q, indicators)

            if not res:
                continue

            evaluation_results[q] = res

            rows = [
                {
                    "id": str(uuid.uuid4()),
                    "corp_code": corp_code,
                    "quarter": q,
                    "indicator_id": item["indicator_id"],
                    "indicator_name": item["indicator_name"],
                    "pillar": item["pillar"],
                    "score": float(item["score"]),
                    "confidence": float(item.get("confidence", 0.5)),
                    "rationale": item.get("rationale", "")[:2000],
                    "evidence_json": json.dumps(item.get("evidence", []), ensure_ascii=False),
                    "created_at": datetime.now().isoformat(timespec="seconds"),
                }
                for item in res
            ]
            insert_scores_json(db, rows)
            write_quarter_json(corp_root, corp_code, corp_name, q, res)

        end_time = time.perf_counter()

        latest_quarter = targets[0] if targets else None
        latest_results = evaluation_results.get(latest_quarter, []) if latest_quarter else []
        risk_summary = _calculate_risk_summary(latest_results)

        final_result = {
            "success": True,
            "corp_code": corp_code,
            "corp_name": corp_name,
            "analysis_date": datetime.now().isoformat(),
            "analysis_duration": round(end_time - start_time, 2),
            "analyzed_quarters": list(evaluation_results.keys()),
            "latest_quarter": latest_quarter,
            "latest_quarter_results": latest_results,
            "risk_summary": risk_summary,
            "total_documents": len(saved),
            "total_indicators": len(indicators),
            "evaluation_results_by_quarter": evaluation_results,
        }

        print(f"✅ 비재무지표 분석 완료 ({final_result['analysis_duration']}초)")
        print(f"  - 분석 분기: {len(evaluation_results)}개")
        print(f"  - 위험 수준: {risk_summary.get('overall_risk_level', 'Unknown')}")

        return final_result

    except Exception as e:
        print(f"❌ 비재무분석 중 오류: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "corp_code": corp_code,
            "analysis_date": datetime.now().isoformat(),
        }


def _calculate_risk_summary(quarter_results: List[Dict]) -> Dict[str, object]:
    """
    분기별 지표 결과로 위험도 요약 계산 수행
    Args:
        quarter_results: 최신 분기 평가 결과 리스트
    Returns:
        위험도 요약 딕셔너리 반환
    """
    if not quarter_results:
        return {
            "overall_risk_level": "데이터 없음",
            "average_score": 0,
            "risk_indicators": [],
        }

    scores = [item.get("score", 2) for item in quarter_results]
    avg_score = sum(scores) / len(scores) if scores else 2

    risk_indicators = [
        {
            "indicator": item.get("indicator_name", ""),
            "pillar": item.get("pillar", ""),
            "score": item.get("score", 2),
            "grade": item.get("grade_label", ""),
            "confidence": item.get("confidence", 0),
        }
        for item in quarter_results
        if item.get("score", 2) <= 2
    ]

    if avg_score >= 3.5:
        overall_risk = "낮음"
    elif avg_score >= 2.5:
        overall_risk = "보통"
    elif avg_score >= 1.5:
        overall_risk = "주의"
    else:
        overall_risk = "높음"

    return {
        "overall_risk_level": overall_risk,
        "average_score": round(avg_score, 2),
        "total_indicators": len(quarter_results),
        "risk_indicators_count": len(risk_indicators),
        "risk_indicators": risk_indicators[:5],
        "score_distribution": {
            "excellent": len([s for s in scores if s >= 4]),
            "good": len([s for s in scores if 3 <= s < 4]),
            "neutral": len([s for s in scores if 2 <= s < 3]),
            "poor": len([s for s in scores if 1 <= s < 2]),
            "critical": len([s for s in scores if s < 1]),
        },
    }


if __name__ == "__main__":
    test_corp_code = "00126380"
    result = run_for_corp(corp_code=test_corp_code, asof=date.today(), force=False)

    if result.get("success"):
        print("\n비재무분석 성공:")
        print(f"- 기업: {result.get('corp_name')}")
        print(f"- 분석 분기: {len(result.get('analyzed_quarters', []))}개")
        print(f"- 위험 수준: {result['risk_summary']['overall_risk_level']}")
    else:
        print(f"비재무분석 실패: {result.get('error')}")
