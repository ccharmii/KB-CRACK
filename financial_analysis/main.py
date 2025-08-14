# -*- coding: utf-8 -*-
"""
재무지표 이상치 탐지 메인 모듈
"""

import json
from .load_corpinfo import CorpInfo
from .finance_metric import get_company_financial_indicators, get_industry_average_indicators
from .calc_metrics import FinancialAnalyzer


def analyze_corporation(output_dir, corp_name: str, file_path: str, n_years: int = 2, max_peers: int = 5) -> dict:
    """
    기업 재무분석 전체 프로세스 실행
    
    Args:
        corp_name (str): 분석할 기업명
        file_path (str): KRX 업종분류현황 CSV 파일 경로
        n_years (int): 분석할 기간 (년)
        max_peers (int): 비교할 동종업계 기업 수
        
    Returns:
        dict: 재무분석 결과 딕셔너리
    """
    print(f"📊 {corp_name} 재무지표 분석 시작")
    
    try:
        # 기업 정보 수집
        corp_info = CorpInfo(corp_name)
        
        # 개별 기업 재무 데이터 수집
        print("  - 개별 기업 재무데이터 수집 중...")
        individual_df = get_company_financial_indicators(corp_name, n_years=n_years)
        
        # 개별 기업 데이터가 없는 경우 분석 중단
        if individual_df.empty:
            error_msg = f"{corp_name}의 재무 데이터를 수집할 수 없습니다."
            print(f"❌ {error_msg}")
            return {"error": error_msg}

        # 동종업계 지표 평균 데이터 수집
        print("  - 동종업계 평균 데이터 수집 중...")
        industry_average_df = get_industry_average_indicators(
            file_path=file_path,
            corp_name=corp_name,
            max_companies=max_peers,
            n_years=n_years
        )
        
        # 재무분석기 생성
        analyzer = FinancialAnalyzer(corp_name, individual_df, industry_average_df)

        # 분석 결과 생성
        print("  - 재무지표 분석 실행 중...")
        
        # 1. 기업 정보
        corp_info_dict = corp_info.get_corpinfo_json()

        # 2. 주요 지표 시계열
        metrics_ts = json.loads(analyzer.analyze_current_situation())
        
        # 3. 지표 분류별 정량 평가
        metrics_by_category = json.loads(analyzer.evaluate_by_category())

        # 4. 동종업계 비교 이상치 탐지
        peer_anomalies = json.loads(analyzer.detect_peer_anomalies())
        
        # 5. 시계열 이상치 탐지
        ts_anomalies = json.loads(analyzer.detect_timeseries_anomalies())

        # 6. 개별 기업 재무지표 전체
        target_corp_metrics = json.loads(individual_df.to_json(orient='records', indent=4))
        
        # 최종 결과 구성
        results = {
            "기업_정보": corp_info_dict,
            "주요지표_시계열_분석": metrics_ts,
            "지표_분류별_정량평가": metrics_by_category,
            "동종업계_비교_이상치_탐지": peer_anomalies,
            "과거_데이터_비교_이상치_탐지": ts_anomalies,
            "개별기업_재무지표_전체": target_corp_metrics,
            "분석_메타데이터": {
                "분석_기간": f"{n_years}년",
                "비교_기업수": max_peers,
                "분석_일시": analyzer.analyze_current_situation()  # 실제로는 timestamp
            }
        }
        
        print(f"✅ {corp_name} 재무지표 분석 완료")

        return results
        
    except Exception as e:
        error_msg = f"재무분석 중 오류 발생: {str(e)}"
        print(f"❌ {error_msg}")
        return {"error": error_msg}


def extract_financial_anomalies(analysis_result: dict, output_dir) -> list:
    """
    재무분석 결과에서 이상치 목록 추출
    
    Args:
        analysis_result: analyze_corporation 함수의 결과
        
    Returns:
        list: 이상치 목록 (근거분석용 포맷)
    """
    anomalies = {}
    
    if "error" in analysis_result:
        return anomalies
    
    # 동종업계 비교 이상치
    peer_anomalies = analysis_result.get("동종업계_비교_이상치_탐지", {})
    for metric, description in peer_anomalies.items():
        anomalies[metric] = {
            "type": "peer_comparison",
            "metric_name": metric,
            "description": description,
            "severity": "High" if any(word in description for word in ["크게", "급격히", "현저히"]) else "Medium",
            "quarter": "Latest",
            "source": "peer_analysis"
        }
    
    # 시계열 이상치
    ts_anomalies = analysis_result.get("과거_데이터_비교_이상치_탐지", {})
    for metric, description in ts_anomalies.items():
        # 같은 metric이 이미 있으면 덮어쓰지 않고 새로운 키 생성
        key = metric if metric not in anomalies else f"{metric}_timeseries"
        anomalies[key] = {
            "type": "time_series",
            "metric_name": metric,
            "description": description,
            "severity": "High" if any(word in description for word in ["급격히", "크게", "급증", "급감"]) else "Medium",
            "quarter": "Latest",
            "source": "timeseries_analysis"
        }
    
    # json 파일로 저장
    with open(output_dir / 'financial_anomalies.json', 'w', encoding='utf-8') as f:
        json.dump(anomalies, f, ensure_ascii=False, indent=4)

    return anomalies


if __name__ == "__main__":
    # 테스트 실행
    TARGET_CORP_NAME = "삼성전자"
    KRX_DATA_FILE_PATH = "../업종분류현황_250809.csv"
    
    result = analyze_corporation(
        corp_name=TARGET_CORP_NAME,
        file_path=KRX_DATA_FILE_PATH,
        n_years=2,
        max_peers=5
    )
    
    if "error" not in result:
        print("\n재무분석 결과:")
        for key in result.keys():
            print(f"- {key}")
        
        # 이상치 추출 테스트
        anomalies = extract_financial_anomalies(result)
        print(f"\n탐지된 이상치: {len(anomalies)}개")
        for anomaly in anomalies:
            print(f"  - {anomaly['metric_name']}: {anomaly['severity']}")
    else:
        print(f"분석 실패: {result['error']}")