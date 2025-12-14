# /financial_analysis/main.py
# 기업 재무지표 수집과 비교 분석을 수행, 이상치 목록을 추출해 저장


import json

from .calc_metrics import FinancialAnalyzer
from .finance_metric import get_company_financial_indicators, get_industry_average_indicators
from .load_corpinfo import CorpInfo


def analyze_corporation(
    output_dir,
    corp_name: str,
    file_path: str,
    n_years: int = 2,
    max_peers: int = 5,
) -> dict:
    """
    기업 재무지표 분석 파이프라인 실행 수행
    Args:
        output_dir: 결과 저장 디렉토리 Path 객체
        corp_name: 분석 대상 기업명
        file_path: KRX 업종분류현황 CSV 파일 경로
        n_years: 분석 기간 년수
        max_peers: 비교 동종업계 기업 수
    Returns:
        재무분석 결과 딕셔너리 반환
    """
    print(f"📊 {corp_name} 재무지표 분석 시작")

    try:
        corp_info = CorpInfo(corp_name)

        print("  - 개별 기업 재무데이터 수집 중...")
        individual_df = get_company_financial_indicators(corp_name, n_years=n_years)

        if individual_df.empty:
            error_msg = f"{corp_name}의 재무 데이터를 수집할 수 없습니다."
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        print("  - 동종업계 평균 데이터 수집 중...")
        industry_average_df = get_industry_average_indicators(
            file_path=file_path,
            corp_name=corp_name,
            max_companies=max_peers,
            n_years=n_years,
        )

        analyzer = FinancialAnalyzer(corp_name, individual_df, industry_average_df)

        print("  - 재무지표 분석 실행 중...")
        corp_info_dict = corp_info.get_corpinfo_json()
        metrics_ts_json = analyzer.analyze_current_situation()
        metrics_by_category_json = analyzer.evaluate_by_category()
        peer_anomalies_json = analyzer.detect_peer_anomalies()
        ts_anomalies_json = analyzer.detect_timeseries_anomalies()

        results = {
            "기업_정보": corp_info_dict,
            "주요지표_시계열_분석": json.loads(metrics_ts_json),
            "지표_분류별_정량평가": json.loads(metrics_by_category_json),
            "동종업계_비교_이상치_탐지": json.loads(peer_anomalies_json),
            "과거_데이터_비교_이상치_탐지": json.loads(ts_anomalies_json),
            "개별기업_재무지표_전체": json.loads(individual_df.to_json(orient="records", indent=4)),
            "분석_메타데이터": {
                "분석_기간": f"{n_years}년",
                "비교_기업수": max_peers,
                "분석_일시": metrics_ts_json,
            },
        }

        print(f"✅ {corp_name} 재무지표 분석 완료")
        return results

    except Exception as e:
        error_msg = f"재무분석 중 오류 발생: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


def extract_financial_anomalies(analysis_result: dict, output_dir) -> dict:
    """
    재무분석 결과에서 이상치 목록 추출 수행
    Args:
        analysis_result: analyze_corporation 결과 딕셔너리
        output_dir: 결과 저장 디렉토리 Path 객체
    Returns:
        이상치 딕셔너리 반환
    """
    anomalies: dict = {}

    if "error" in analysis_result:
        return anomalies

    peer_anomalies = analysis_result.get("동종업계_비교_이상치_탐지", {})
    for metric, description in peer_anomalies.items():
        anomalies[metric] = {
            "type": "peer_comparison",
            "metric_name": metric,
            "description": description,
            "severity": "High" if any(word in description for word in ["크게", "급격히", "현저히"]) else "Medium",
            "quarter": "Latest",
            "source": "peer_analysis",
        }

    ts_anomalies = analysis_result.get("과거_데이터_비교_이상치_탐지", {})
    for metric, description in ts_anomalies.items():
        key = metric if metric not in anomalies else f"{metric}_timeseries"
        anomalies[key] = {
            "type": "time_series",
            "metric_name": metric,
            "description": description,
            "severity": "High" if any(word in description for word in ["급격히", "크게", "급증", "급감"]) else "Medium",
            "quarter": "Latest",
            "source": "timeseries_analysis",
        }

    with open(output_dir / "financial_anomalies.json", "w", encoding="utf-8") as f:
        json.dump(anomalies, f, ensure_ascii=False, indent=4)

    return anomalies


def main() -> None:
    from pathlib import Path

    target_corp_name = "삼성전자"
    krx_data_file_path = "../업종분류현황_250809.csv"
    output_dir = Path("./analysis_results_tmp")
    output_dir.mkdir(parents=True, exist_ok=True)

    result = analyze_corporation(
        output_dir=output_dir,
        corp_name=target_corp_name,
        file_path=krx_data_file_path,
        n_years=2,
        max_peers=5,
    )

    if "error" in result:
        print(f"분석 실패: {result['error']}")
        return

    print("\n재무분석 결과 키 목록")
    for key in result.keys():
        print(f"- {key}")

    anomalies = extract_financial_anomalies(result, output_dir)
    print(f"\n탐지된 이상치 수: {len(anomalies)}개")
    for _, anomaly in anomalies.items():
        print(f"  - {anomaly.get('metric_name')}: {anomaly.get('severity')}")


if __name__ == "__main__":
    main()
