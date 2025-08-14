## finance_metric.py

import pandas as pd
from functools import reduce

from .load_finance_b import fetch_latest_n_years_reports, integrate_stock_data
from .load_finance_a import (fetch_latest_n_years_reports as fetch_a_reports, # 충돌 방지를 위한 별칭 사용
                            prepare_finreports,
                            clean_columns,
                            MetricsCalculator)
from .load_samecorpmean_fromcsv import analyze_industry_benchmark
from .load_corpinfo import CorpInfo
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

DART_API_KEY = os.getenv('DART_API_KEY')

def get_company_financial_indicators(corp_name, n_years=2):
    """
    개별 기업의 재무지표를 통합 DataFrame으로 반환
    (주요 지표 API에서 제공하지 않는 일부 중요 지표를 재무제표에서 추가)
    입력: 기업명, 조회할 연수
    출력: 모든 지표분류가 통합된 재무지표 DF (실패시 빈 DF)
    """
    try:
        corp_info = CorpInfo(corp_name)
        corp_code = corp_info.corp_code
        stock_code = corp_info.corp_info.iloc[0]['stock_code']

        # 주요 재무지표 수집 (from load_finance_b)
        financial_data_b = fetch_latest_n_years_reports(corp_code, n_years)
        if not financial_data_b:
            print(f"[Warning] {corp_name}의 재무 데이터를 찾을 수 없습니다.")
            return pd.DataFrame()
        if not pd.isna(stock_code):
            financial_data_b = integrate_stock_data(financial_data_b, stock_code)

        # 재무제표에서 추가 지표 계산 (from load_finance_a)
        df_a_metrics = pd.DataFrame()
        try:
            # a 모듈 방식으로 재무제표 원본 데이터 수집 (API_KEY 전달)
            financial_df_a = fetch_a_reports(DART_API_KEY, corp_code=corp_code, n_years=n_years)
            if not financial_df_a.empty:
                financial_df_a = prepare_finreports(financial_df_a)
                financial_df_a = clean_columns(financial_df_a)
                calculator = MetricsCalculator(financial_df_a, stock_df=None)
                metrics_a = calculator.calculate_metrics()
                df_a_metrics = pd.DataFrame({'금융비용대부채비율': metrics_a.get('금융비용대부채비율'),
                                             '총자본회전율': metrics_a.get('총자본회전율')}).dropna(how='all')
                if not df_a_metrics.empty:
                    df_a_metrics = df_a_metrics.reset_index().rename(columns={'index': '분기'})
                    df_a_metrics['연도'] = df_a_metrics['분기'].str.split('_').str[0].astype(int)
                    quarter_map_inv = {'Q1': '1분기보고서', 'Q2': '반기보고서', 'Q3': '3분기보고서', 'Q4': '사업보고서'}
                    df_a_metrics['보고서'] = df_a_metrics['분기'].str.split('_').str[1].map(quarter_map_inv)
                    df_a_metrics = df_a_metrics.drop(columns=['분기'])
        except Exception as e:
            print(f"[Info] a모듈 기반 추가 지표 수집 중 오류 발생: {e}")

        # DF 통합
        all_dfs = [df for df in financial_data_b.values() if not df.empty]
        if not df_a_metrics.empty:
            all_dfs.append(df_a_metrics)
        if not all_dfs:
            return pd.DataFrame()
        for i in range(len(all_dfs)):
            if '지표분류' in all_dfs[i].columns:
                all_dfs[i] = all_dfs[i].drop(columns=['지표분류'])
        if len(all_dfs) == 1:
            final_df = all_dfs[0]
        else:
            final_df = reduce(lambda left, right: pd.merge(left, right, on=['연도', '보고서'], how='outer'), all_dfs)
            final_df = final_df.loc[:, ~final_df.columns.str.endswith(('_x', '_y'))]
        # 지표 연동을 위해 2023년 3분기 이후 데이터만 필터링
        if not final_df.empty and '연도' in final_df.columns and '보고서' in final_df.columns:
            final_df['연도'] = pd.to_numeric(final_df['연도'], errors='coerce')
            is_after_2023_q3 = (final_df['연도'] > 2023) | \
                               ((final_df['연도'] == 2023) & (final_df['보고서'].isin(['3분기보고서', '사업보고서'])))
            final_df = final_df[is_after_2023_q3].copy()
        return final_df

    except Exception as e:
        print(f"[Warning] {corp_name} 재무지표 수집 실패: {e}")
        return pd.DataFrame()


def get_industry_average_indicators(file_path, corp_name, max_companies=5, n_years=1):
    """
    동종업계 평균 재무지표를 통합 DF 반환
    입력: KRX CSV 파일 경로, 타겟 기업명, 최대 비교기업 수, 조회할 연수
    출력: 동종업계 평균이 계산된 통합 DF (실패시 빈 DF)
    """
    try:
        _ , industry_avg_data = analyze_industry_benchmark(target_corp_name=corp_name,
                                                         file_path=file_path,
                                                         max_companies=max_companies,
                                                         n_years=n_years,
                                                         latest_only=False)
        if not industry_avg_data or not isinstance(industry_avg_data, dict):
            print("[Warning] 동종업계 평균 데이터를 생성할 수 없습니다.")
            return pd.DataFrame()
        
        combined_avg_data = [df for df in industry_avg_data.values() if not df.empty]
        if not combined_avg_data:
            return pd.DataFrame()
        for i in range(len(combined_avg_data)):
            if '지표분류' in combined_avg_data[i].columns:
                combined_avg_data[i] = combined_avg_data[i].drop(columns=['지표분류'])
        if len(combined_avg_data) == 1:
            return combined_avg_data[0]
        else:
            merged_df = reduce(lambda left, right: pd.merge(left, right, on=['연도', '보고서'], how='outer'), combined_avg_data)
            return merged_df.loc[:, ~merged_df.columns.str.endswith(('_x', '_y'))]

    except Exception as e:
        print(f"[에러] 동종업계 평균 지표 수집 실패: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    
    TARGET_CORP_NAME = "삼성전자"
    KRX_DATA_FILE_PATH = "../업종분류현황_250809.csv"
    N_YEARS = 2 # 조회할 기간 (년)
    MAX_PEERS = 5 # 동종업계 기업 수

    
    # 1. 개별 기업 데이터프레임 불러오기
    print("개별 기업 데이터 수집 중...")
    individual_df = get_company_financial_indicators(TARGET_CORP_NAME, n_years=N_YEARS)
    
    # 2. 동종업계 평균 데이터프레임 불러오기
    print("\n동종업계 평균 데이터 수집 중...")
    industry_average_df = get_industry_average_indicators(file_path=KRX_DATA_FILE_PATH, 
                                                         corp_name=TARGET_CORP_NAME, 
                                                         max_companies=MAX_PEERS, 
                                                         n_years=N_YEARS)

    print("\n\n" + "="*50)
    print("분석 완료")
    print("="*50)

    print("\n개별 기업 최종 DataFrame:")
    if not individual_df.empty:
        print(individual_df.head())
        print(f"\n-> 최종 컬럼 수: {len(individual_df.columns)}")
        if '금융비용대부채비율' in individual_df.columns and '총자본회전율' in individual_df.columns:
            print("-> '금융비용대부채비율', '총자본회전율' 컬럼이 성공적으로 추가되었습니다.")
    else:
        print("-> 데이터를 불러오지 못했습니다.")

    print("\n동종업계 평균 최종 DataFrame:")
    if not industry_average_df.empty:
        print(industry_average_df.head())
        print(f"\n-> 최종 컬럼 수: {len(industry_average_df.columns)}")
    else:
        print("-> 데이터를 불러오지 못했습니다.")
