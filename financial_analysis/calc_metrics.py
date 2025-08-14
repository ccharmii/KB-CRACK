## calc_metrics.py

import pandas as pd
import json

from .finance_metric import get_company_financial_indicators, get_industry_average_indicators


class FinancialAnalyzer:
    """
    개별 기업과 동종업계 평균 데이터를 기반으로 심층 재무 분석을 수행하는 클래스
    1. 현재 기업 상황 분석 (주요 지표 시계열)
    2. 지표 분류별 정량 평가 (실제 값 비교)
    3. 동종업계 비교 이상치 탐지
    4. 과거 데이터 비교 시계열 이상치 탐지
    """
    def __init__(self, corp_name, individual_df, industry_average_df):
        """
        입력: 기업명, 개별 기업 DF, 동종업계 평균 DF
        """
        self.corp_name = corp_name
        self.individual_df = self._prepare_dataframe(individual_df)
        self.industry_average_df = self._prepare_dataframe(industry_average_df)


    def _prepare_dataframe(self, df):
        """데이터프레임 전처리: 연도와 분기 보고서 기준으로 정렬"""
        if df.empty:
            return df
        report_order = {'1분기보고서': 1, '반기보고서': 2, '3분기보고서': 3, '사업보고서': 4}
        df['report_order'] = df['보고서'].map(report_order)
        df_sorted = df.sort_values(by=['연도', 'report_order'], ascending=[True, True]).reset_index(drop=True)
        return df_sorted.drop(columns=['report_order'])


    def analyze_current_situation(self):
        """
        1. 현재 기업 상황 분석
        - 연도별/분기별 주요 지표를 추출하여 JSON으로 반환
        """
        if self.individual_df.empty:
            return json.dumps({"error": "개별 기업 데이터가 없습니다."}, ensure_ascii=False, indent=4)
        df = self.individual_df.copy()
        latest_row = df.iloc[-1]
        latest_year = latest_row['연도']
        latest_report = latest_row['보고서']
        key_metrics = {'금융비용대부채비율': '금융비용대부채비율',
                       '총자본회전율': '총자본회전율',
                       '영업이익대총자산비율': '총자산영업이익률'}
        
        results = {}
        quarterly_df = df.tail(4)
        for _, row in quarterly_df.iterrows():
            period_key = f"{int(row['연도'])}년 {row['보고서']}"
            results[period_key] = {metric: round(row[col], 2) if pd.notna(row.get(col)) else None
                                   for metric, col in key_metrics.items() if col in row}

        for i in [1, 2]:
            target_year = latest_year - i
            yearly_df = df[(df['연도'] == target_year) & (df['보고서'] == latest_report)]
            if not yearly_df.empty:
                row = yearly_df.iloc[0]
                period_key = f"{int(row['연도'])}년 {row['보고서']}"
                results[period_key] = {metric: round(row[col], 2) if pd.notna(row.get(col)) else None
                                       for metric, col in key_metrics.items() if col in row}
        # 보고서 정렬을 위한 순서 정의
        report_order = {'1분기보고서': 1, '반기보고서': 2, '3분기보고서': 3, '사업보고서': 4}
        sorted_keys = sorted(results.keys(), key=lambda k: (int(k.split('년')[0]), report_order.get(k.split(' ')[1], 0)))
        sorted_results = {key: results[key] for key in sorted_keys}

        return json.dumps(sorted_results, ensure_ascii=False, indent=4)


    def evaluate_by_category(self):
        """
        2. 각 지표 분류 별로 실제 값 비교 (최신 분기 기준)
        - 점수 대신 실제 지표 값을 JSON 형태로 반환하여 직접 비교할 수 있도록 함.
        """
        if self.individual_df.empty or self.industry_average_df.empty:
            return json.dumps({})  ## 비교 데이터가 부족해서 값을 생성할 수 없음

        ind_latest = self.individual_df.iloc[-1]
        avg_latest = self.industry_average_df.iloc[-1]
        evaluation_map = {'안정성': ['부채비율', '유동비율'],
                          '수익성': ['ROE', '매출총이익률'],
                          '성장성': ['매출액증가율(YoY)', '총자산증가율', '영업이익증가율(YoY)'],
                          '활동성': ['총자산회전율', '재고자산회전율'],
                          '주식가치': ['PER', 'PBR', 'EPS']}
        
        results = {self.corp_name: {}, "동종업계 평균": {}}
        for category, metrics in evaluation_map.items():
            ind_metric_values = {}
            avg_metric_values = {}
            for metric in metrics:
                ind_val = ind_latest.get(metric)
                avg_val = avg_latest.get(metric)
                if pd.notna(ind_val):
                    ind_metric_values[metric] = round(ind_val, 2)
                if pd.notna(avg_val):
                    avg_metric_values[metric] = round(avg_val, 2)
            if ind_metric_values:
                results[self.corp_name][category] = ind_metric_values
            if avg_metric_values:
                results["동종업계 평균"][category] = avg_metric_values
        return json.dumps(results, ensure_ascii=False, indent=4)


    def detect_peer_anomalies(self, threshold_pct=30.0):
        """
        3. 임계값을 활용한 상대평가 (최신 분기 기준)
        - 동종업계 평균 대비 편차가 임계값을 초과하는 지표 탐지
        """
        if self.individual_df.empty or self.industry_average_df.empty:
            return json.dumps({})  ## 비교 데이터가 부족해서 값을 생성할 수 없음

        ind_latest = self.individual_df.iloc[-1]
        avg_latest = self.industry_average_df.iloc[-1]
        metrics_to_compare = ['부채비율', 'ROE', '매출액증가율(YoY)', 'PER']
        anomalies = {}

        for metric in metrics_to_compare:
            ind_val = ind_latest.get(metric)
            avg_val = avg_latest.get(metric)
            if pd.notna(ind_val) and pd.notna(avg_val) and avg_val != 0:
                deviation = ((ind_val - avg_val) / abs(avg_val)) * 100
                if abs(deviation) > threshold_pct:
                    direction = "상회" if deviation > 0 else "하회"
                    anomalies[metric] = (f"{round(ind_val, 2)}로 동종업계 평균인 {round(avg_val, 2)}를 "
                                         f"{round(abs(deviation), 1)}%p {direction}하여 이상치로 탐지됨")
        return json.dumps(anomalies, ensure_ascii=False, indent=4)


    def detect_timeseries_anomalies(self, std_multiplier=2.0, change_threshold=0.2):
        """
        4. 시계열 이상치 탐지 (전년 동분기 대비)
        - 최신 분기 지표를 1년, 2년 전 동일 분기와 비교
        - (A) 2년 전 데이터가 있는 경우: 최신 분기 vs (1, 2년 전 동분기 값의) 평균/표준편차 비교
        - (B) 1년 전 데이터만 있는 경우: 최신 분기 vs 1년 전 동분기 값의 변화율 비교
        - (C) 비교 데이터가 불충분한 경우: 분석을 수행하지 않음
        """
        num_rows = len(self.individual_df)
        anomalies = {}
        if num_rows < 5:
            return json.dumps({})  ## 비교 데이터가 부족해서 값을 생성할 수 없음

        metrics_to_check = {'영업이익률': '총자산영업이익률',
                            '부채비율': '부채비율',
                            '매출액증가율': '매출액증가율(YoY)',
                            '재고자산회전율': '재고자산회전율'}
        latest_data = self.individual_df.iloc[-1]
        has_2y_data = num_rows >= 9

        if has_2y_data:
            # 1, 2년 전 데이터와 비교
            comparison_desc = "1, 2년 전 동분기"
            comparison_data = self.individual_df.iloc[[-9, -5]] # 2년 전, 1년 전 데이터

            for metric_name, col_name in metrics_to_check.items():
                # 값 준비 및 유효성 검사
                if col_name not in comparison_data.columns: continue
                latest_val = latest_data.get(col_name)
                series = comparison_data[col_name].dropna()
                if len(series) != 2 or pd.isna(latest_val): continue
                
                # 분석 및 이상치 탐지
                mean, std = series.mean(), series.std()
                if std > 1e-6: # 표준편차 기반 탐지
                    if not (mean - std_multiplier * std <= latest_val <= mean + std_multiplier * std):
                        direction = "상회" if latest_val > mean else "하회"
                        anomalies[metric_name] = f"최신 분기({latest_val:.2f})가 {comparison_desc} 평균({mean:.2f})을 유의미하게 {direction} (표준편차 {std_multiplier}배)"
                elif abs(latest_val - mean) > 1e-6: # 단순 값 비교 (std가 0일 때)
                    direction = "상승" if latest_val > mean else "하락"
                    anomalies[metric_name] = f"최신 분기({latest_val:.2f})가 과거 동분기 값({mean:.2f}) 대비 {direction}"
        else:
            # 1년 전 데이터와 비교
            comparison_desc = "1년 전 동분기"
            
            for metric_name, col_name in metrics_to_check.items():
                if col_name not in self.individual_df.columns: continue
                latest_val = latest_data.get(col_name)
                historical_val = self.individual_df.iloc[-5].get(col_name)
                if pd.isna(latest_val) or pd.isna(historical_val) or abs(historical_val) < 1e-6: continue

                # 분석 및 이상치 탐지 (변화율 기준)
                change = (latest_val - historical_val) / historical_val
                if abs(change) > change_threshold:
                    direction = "증가" if change > 0 else "감소"
                    anomalies[metric_name] = f"최신 분기({latest_val:.2f})가 {comparison_desc} 값({historical_val:.2f}) 대비 {abs(change):.1%} {direction}"
        if not anomalies:
            return json.dumps({})
            
        return json.dumps(anomalies, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    # ================= 설정값 =================
    TARGET_CORP_NAME = "삼성전자"
    KRX_DATA_FILE_PATH = "../업종분류현황_250809.csv" # finance_metric.py와 동일한 위치에 있어야 합니다.
    N_YEARS = 2
    MAX_PEERS = 5
    # ==========================================

    print(f"[{TARGET_CORP_NAME}] 재무 분석을 시작합니다.")

    print("\n[1/2] 개별 기업 데이터 수집 중...")
    individual_df = get_company_financial_indicators(TARGET_CORP_NAME, n_years=N_YEARS)
    
    print("\n[2/2] 동종업계 평균 데이터 수집 중...")
    industry_average_df = get_industry_average_indicators(
        file_path=KRX_DATA_FILE_PATH,
        corp_name=TARGET_CORP_NAME,
        max_companies=MAX_PEERS,
        n_years=N_YEARS
    )

    if individual_df.empty:
        print(f"\n분석 실패: {TARGET_CORP_NAME}의 재무 데이터를 수집할 수 없습니다.")
    else:
        analyzer = FinancialAnalyzer(TARGET_CORP_NAME, individual_df, industry_average_df)

        print("\n\n" + "="*70)
        print("분석 완료. 아래는 최종 결과입니다.")
        print("="*70)
        
        print("\n\n[분석 1] 현재 기업 상황 분석 (주요 지표 시계열)")
        print("-" * 50)
        print(analyzer.analyze_current_situation())

        print("\n\n[분석 2] 지표 분류별 실제 값 비교")
        print("-" * 50)
        print(analyzer.evaluate_by_category())

        print("\n\n[분석 3] 동종업계 비교 이상치 탐지 (Peer Group Anomaly)")
        print("-" * 50)
        print(analyzer.detect_peer_anomalies())
        
        print("\n\n[분석 4] 시계열 이상치 탐지 (Time-Series Anomaly)")
        print("-" * 50)
        print(analyzer.detect_timeseries_anomalies())