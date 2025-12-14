# /financial_analysis/calc_metrics.py
# 재무 지표 기반 기업 분석 및 이상치 탐지를 수행

import json
import pandas as pd

from .finance_metric import get_company_financial_indicators, get_industry_average_indicators


class FinancialAnalyzer:
    """
    개별 기업 데이터와 동종업계 평균 데이터를 기반으로 재무 분석 및 이상치 탐지 수행 클래스
    분석 항목
    - 주요 지표 시계열 요약 생성
    - 분류별 지표 값 비교 생성
    - 동종업계 대비 편차 기반 이상치 탐지
    - 전년 및 전전년 동분기 비교 기반 시계열 이상치 탐지
    """

    def __init__(self, corp_name: str, individual_df: pd.DataFrame, industry_average_df: pd.DataFrame):
        """
        분석 대상 데이터와 기업명 설정 수행
        Args:
            corp_name: 기업명 문자열
            individual_df: 개별 기업 재무 지표 데이터프레임
            industry_average_df: 동종업계 평균 재무 지표 데이터프레임
        """
        self.corp_name = corp_name
        self.individual_df = self._prepare_dataframe(individual_df)
        self.industry_average_df = self._prepare_dataframe(industry_average_df)

    def _prepare_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """연도 및 보고서 기준 정렬을 위한 데이터프레임 전처리 수행"""
        if df.empty:
            return df

        report_order = {"1분기보고서": 1, "반기보고서": 2, "3분기보고서": 3, "사업보고서": 4}
        df = df.copy()
        df["report_order"] = df["보고서"].map(report_order)

        df_sorted = df.sort_values(by=["연도", "report_order"], ascending=[True, True]).reset_index(drop=True)
        return df_sorted.drop(columns=["report_order"])

    def analyze_current_situation(self) -> str:
        """
        최신 분기 기준 주요 지표 시계열 요약 JSON 생성 수행
        Returns:
            주요 지표 시계열 요약 JSON 문자열 반환
        """
        if self.individual_df.empty:
            return json.dumps({"error": "개별 기업 데이터 부재"}, ensure_ascii=False, indent=4)

        df = self.individual_df.copy()
        latest_row = df.iloc[-1]
        latest_year = latest_row["연도"]
        latest_report = latest_row["보고서"]

        key_metrics = {
            "금융비용대부채비율": "금융비용대부채비율",
            "총자본회전율": "총자본회전율",
            "영업이익대총자산비율": "총자산영업이익률",
        }

        results: dict[str, dict[str, float | None]] = {}

        quarterly_df = df.tail(4)
        for _, row in quarterly_df.iterrows():
            period_key = f"{int(row['연도'])}년 {row['보고서']}"
            results[period_key] = {
                metric: round(row[col], 2) if pd.notna(row.get(col)) else None
                for metric, col in key_metrics.items()
                if col in row
            }

        for i in (1, 2):
            target_year = latest_year - i
            yearly_df = df[(df["연도"] == target_year) & (df["보고서"] == latest_report)]
            if yearly_df.empty:
                continue

            row = yearly_df.iloc[0]
            period_key = f"{int(row['연도'])}년 {row['보고서']}"
            results[period_key] = {
                metric: round(row[col], 2) if pd.notna(row.get(col)) else None
                for metric, col in key_metrics.items()
                if col in row
            }

        report_order = {"1분기보고서": 1, "반기보고서": 2, "3분기보고서": 3, "사업보고서": 4}
        sorted_keys = sorted(
            results.keys(),
            key=lambda k: (int(k.split("년")[0]), report_order.get(k.split(" ")[1], 0)),
        )
        sorted_results = {key: results[key] for key in sorted_keys}

        return json.dumps(sorted_results, ensure_ascii=False, indent=4)

    def evaluate_by_category(self) -> str:
        """
        최신 분기 기준 분류별 지표 값 비교 JSON 생성 수행
        Returns:
            기업 및 동종업계 평균 분류별 지표 값 JSON 문자열 반환
        """
        if self.individual_df.empty or self.industry_average_df.empty:
            return json.dumps({}, ensure_ascii=False, indent=4)

        ind_latest = self.individual_df.iloc[-1]
        avg_latest = self.industry_average_df.iloc[-1]

        evaluation_map = {
            "안정성": ["부채비율", "유동비율"],
            "수익성": ["ROE", "매출총이익률"],
            "성장성": ["매출액증가율(YoY)", "총자산증가율", "영업이익증가율(YoY)"],
            "활동성": ["총자산회전율", "재고자산회전율"],
            "주식가치": ["PER", "PBR", "EPS"],
        }

        results: dict[str, dict[str, dict[str, float]]] = {self.corp_name: {}, "동종업계 평균": {}}

        for category, metrics in evaluation_map.items():
            ind_metric_values: dict[str, float] = {}
            avg_metric_values: dict[str, float] = {}

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

    def detect_peer_anomalies(self, threshold_pct: float = 30.0) -> str:
        """
        동종업계 평균 대비 편차 기반 이상치 탐지 JSON 생성 수행
        Args:
            threshold_pct: 편차 임계값 퍼센트 값
        Returns:
            탐지된 이상치 설명 JSON 문자열 반환
        """
        if self.individual_df.empty or self.industry_average_df.empty:
            return json.dumps({}, ensure_ascii=False, indent=4)

        ind_latest = self.individual_df.iloc[-1]
        avg_latest = self.industry_average_df.iloc[-1]

        metrics_to_compare = ["부채비율", "ROE", "매출액증가율(YoY)", "PER"]
        anomalies: dict[str, str] = {}

        for metric in metrics_to_compare:
            ind_val = ind_latest.get(metric)
            avg_val = avg_latest.get(metric)

            if not (pd.notna(ind_val) and pd.notna(avg_val)) or abs(float(avg_val)) < 1e-12:
                continue

            deviation = ((float(ind_val) - float(avg_val)) / abs(float(avg_val))) * 100
            if abs(deviation) <= threshold_pct:
                continue

            direction = "상회" if deviation > 0 else "하회"
            anomalies[metric] = (
                f"{round(float(ind_val), 2)}로 동종업계 평균인 {round(float(avg_val), 2)}를 "
                f"{round(abs(deviation), 1)}%p {direction}하여 이상치로 탐지됨"
            )

        return json.dumps(anomalies, ensure_ascii=False, indent=4)

    def detect_timeseries_anomalies(self, std_multiplier: float = 2.0, change_threshold: float = 0.2) -> str:
        """
        전년 및 전전년 동분기 비교 기반 시계열 이상치 탐지 JSON 생성 수행
        Args:
            std_multiplier: 표준편차 기반 탐지 임계 배수
            change_threshold: 변화율 기반 탐지 임계값
        Returns:
            탐지된 이상치 설명 JSON 문자열 반환
        """
        num_rows = len(self.individual_df)
        if num_rows < 5:
            return json.dumps({}, ensure_ascii=False, indent=4)

        anomalies: dict[str, str] = {}
        latest_data = self.individual_df.iloc[-1]

        metrics_to_check = {
            "영업이익률": "총자산영업이익률",
            "부채비율": "부채비율",
            "매출액증가율": "매출액증가율(YoY)",
            "재고자산회전율": "재고자산회전율",
        }

        has_2y_data = num_rows >= 9

        if has_2y_data:
            comparison_desc = "1, 2년 전 동분기"
            comparison_data = self.individual_df.iloc[[-9, -5]]

            for metric_name, col_name in metrics_to_check.items():
                if col_name not in comparison_data.columns:
                    continue

                latest_val = latest_data.get(col_name)
                series = comparison_data[col_name].dropna()
                if len(series) != 2 or pd.isna(latest_val):
                    continue

                mean = float(series.mean())
                std = float(series.std())
                latest_val_f = float(latest_val)

                if std > 1e-6:
                    in_band = (mean - std_multiplier * std) <= latest_val_f <= (mean + std_multiplier * std)
                    if not in_band:
                        direction = "상회" if latest_val_f > mean else "하회"
                        anomalies[metric_name] = (
                            f"최신 분기({latest_val_f:.2f})가 {comparison_desc} 평균({mean:.2f})을 유의미하게 {direction} "
                            f"(표준편차 {std_multiplier}배)"
                        )
                elif abs(latest_val_f - mean) > 1e-6:
                    direction = "상승" if latest_val_f > mean else "하락"
                    anomalies[metric_name] = f"최신 분기({latest_val_f:.2f})가 과거 동분기 값({mean:.2f}) 대비 {direction}"
        else:
            comparison_desc = "1년 전 동분기"

            for metric_name, col_name in metrics_to_check.items():
                if col_name not in self.individual_df.columns:
                    continue

                latest_val = latest_data.get(col_name)
                historical_val = self.individual_df.iloc[-5].get(col_name)

                if pd.isna(latest_val) or pd.isna(historical_val) or abs(float(historical_val)) < 1e-12:
                    continue

                latest_val_f = float(latest_val)
                historical_val_f = float(historical_val)

                change = (latest_val_f - historical_val_f) / historical_val_f
                if abs(change) <= change_threshold:
                    continue

                direction = "증가" if change > 0 else "감소"
                anomalies[metric_name] = (
                    f"최신 분기({latest_val_f:.2f})가 {comparison_desc} 값({historical_val_f:.2f}) 대비 {abs(change):.1%} {direction}"
                )

        if not anomalies:
            return json.dumps({}, ensure_ascii=False, indent=4)

        return json.dumps(anomalies, ensure_ascii=False, indent=4)


def main() -> None:
    """재무 지표 수집 및 분석 클래스 실행 예시"""
    target_corp_name = "삼성전자"
    krx_data_file_path = "../업종분류현황_250809.csv"
    n_years = 2
    max_peers = 5

    individual_df = get_company_financial_indicators(target_corp_name, n_years=n_years)
    industry_average_df = get_industry_average_indicators(
        file_path=krx_data_file_path,
        corp_name=target_corp_name,
        max_companies=max_peers,
        n_years=n_years,
    )

    if individual_df.empty:
        return

    analyzer = FinancialAnalyzer(target_corp_name, individual_df, industry_average_df)

    analyzer.analyze_current_situation()
    analyzer.evaluate_by_category()
    analyzer.detect_peer_anomalies()
    analyzer.detect_timeseries_anomalies()


if __name__ == "__main__":
    main()