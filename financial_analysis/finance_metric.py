# /financial_analysis/finance_metric.py
# 기업 재무지표와 동종업계 평균 지표를 수집, 통합 데이터프레임 반환


import os
from functools import reduce

import pandas as pd
from dotenv import load_dotenv

from .load_corpinfo import CorpInfo
from .load_finance_a import (
    MetricsCalculator,
    clean_columns,
    fetch_latest_n_years_reports as fetch_a_reports,
    prepare_finreports,
)
from .load_finance_b import fetch_latest_n_years_reports, integrate_stock_data
from .load_samecorpmean_fromcsv import analyze_industry_benchmark

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")


def get_company_financial_indicators(corp_name: str, n_years: int = 2) -> pd.DataFrame:
    """
    개별 기업 재무지표 수집 및 통합 데이터프레임 생성 수행
    Args:
        corp_name: 기업명 문자열
        n_years: 조회 연수 정수
    Returns:
        재무지표 통합 데이터프레임 반환
    """
    try:
        corp_info = CorpInfo(corp_name)
        corp_code = corp_info.corp_code
        stock_code = corp_info.corp_info.iloc[0].get("stock_code")

        financial_data_b = fetch_latest_n_years_reports(corp_code, n_years)
        if not financial_data_b:
            return pd.DataFrame()

        if pd.notna(stock_code):
            financial_data_b = integrate_stock_data(financial_data_b, stock_code)

        df_a_metrics = pd.DataFrame()
        try:
            financial_df_a = fetch_a_reports(DART_API_KEY, corp_code=corp_code, n_years=n_years)
            if not financial_df_a.empty:
                financial_df_a = prepare_finreports(financial_df_a)
                financial_df_a = clean_columns(financial_df_a)

                calculator = MetricsCalculator(financial_df_a, stock_df=None)
                metrics_a = calculator.calculate_metrics()

                df_a_metrics = (
                    pd.DataFrame(
                        {
                            "금융비용대부채비율": metrics_a.get("금융비용대부채비율"),
                            "총자본회전율": metrics_a.get("총자본회전율"),
                        }
                    )
                    .dropna(how="all")
                )

                if not df_a_metrics.empty:
                    df_a_metrics = df_a_metrics.reset_index().rename(columns={"index": "분기"})
                    df_a_metrics["연도"] = df_a_metrics["분기"].str.split("_").str[0].astype(int)

                    quarter_map_inv = {"Q1": "1분기보고서", "Q2": "반기보고서", "Q3": "3분기보고서", "Q4": "사업보고서"}
                    df_a_metrics["보고서"] = df_a_metrics["분기"].str.split("_").str[1].map(quarter_map_inv)
                    df_a_metrics = df_a_metrics.drop(columns=["분기"])
        except Exception:
            pass

        all_dfs = [df for df in financial_data_b.values() if not df.empty]
        if not df_a_metrics.empty:
            all_dfs.append(df_a_metrics)

        if not all_dfs:
            return pd.DataFrame()

        cleaned_dfs = []
        for df in all_dfs:
            if "지표분류" in df.columns:
                cleaned_dfs.append(df.drop(columns=["지표분류"]))
            else:
                cleaned_dfs.append(df)

        if len(cleaned_dfs) == 1:
            final_df = cleaned_dfs[0]
        else:
            final_df = reduce(
                lambda left, right: pd.merge(left, right, on=["연도", "보고서"], how="outer"),
                cleaned_dfs,
            )
            final_df = final_df.loc[:, ~final_df.columns.str.endswith(("_x", "_y"))]

        if not final_df.empty and {"연도", "보고서"}.issubset(final_df.columns):
            final_df = final_df.copy()
            final_df["연도"] = pd.to_numeric(final_df["연도"], errors="coerce")

            is_after_2023_q3 = (final_df["연도"] > 2023) | (
                (final_df["연도"] == 2023) & (final_df["보고서"].isin(["3분기보고서", "사업보고서"]))
            )
            final_df = final_df[is_after_2023_q3].copy()

        return final_df

    except Exception:
        return pd.DataFrame()


def get_industry_average_indicators(
    file_path: str,
    corp_name: str,
    max_companies: int = 5,
    n_years: int = 1,
) -> pd.DataFrame:
    """
    동종업계 평균 재무지표 수집 및 통합 데이터프레임 생성 수행
    Args:
        file_path: KRX 업종 분류 CSV 파일 경로 문자열
        corp_name: 대상 기업명 문자열
        max_companies: 비교 기업 최대 수 정수
        n_years: 조회 연수 정수
    Returns:
        동종업계 평균 재무지표 데이터프레임 반환
    """
    try:
        _, industry_avg_data = analyze_industry_benchmark(
            target_corp_name=corp_name,
            file_path=file_path,
            max_companies=max_companies,
            n_years=n_years,
            latest_only=False,
        )

        if not industry_avg_data or not isinstance(industry_avg_data, dict):
            return pd.DataFrame()

        combined_avg_data = [df for df in industry_avg_data.values() if not df.empty]
        if not combined_avg_data:
            return pd.DataFrame()

        cleaned_dfs = []
        for df in combined_avg_data:
            if "지표분류" in df.columns:
                cleaned_dfs.append(df.drop(columns=["지표분류"]))
            else:
                cleaned_dfs.append(df)

        if len(cleaned_dfs) == 1:
            return cleaned_dfs[0]

        merged_df = reduce(
            lambda left, right: pd.merge(left, right, on=["연도", "보고서"], how="outer"),
            cleaned_dfs,
        )
        return merged_df.loc[:, ~merged_df.columns.str.endswith(("_x", "_y"))]

    except Exception:
        return pd.DataFrame()


def main() -> None:
    """기업 및 동종업계 평균 지표 수집 실행 예시"""
    target_corp_name = "삼성전자"
    krx_data_file_path = "../업종분류현황_250809.csv"
    n_years = 2
    max_peers = 5

    get_company_financial_indicators(target_corp_name, n_years=n_years)
    get_industry_average_indicators(
        file_path=krx_data_file_path,
        corp_name=target_corp_name,
        max_companies=max_peers,
        n_years=n_years,
    )


if __name__ == "__main__":
    main()
