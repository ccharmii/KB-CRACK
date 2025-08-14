## load_finance_a.py


from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta
import re
import pandas as pd
import numpy as np
import json
import requests
import xml.etree.ElementTree as ET


from .load_corpinfo import CorpInfo


import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

DART_API_KEY = os.getenv('DART_API_KEY')


## KRX 주식 데이터 불러와 정보 반환 ===================== 
def generate_dates_from_financial_df(financial_df):
    """
    재무데이터 분기 정보를 기반으로 주식 데이터 날짜 생성
    입력: 여러 개의 분기 보고서의 재무 지표 데이터로 이루어져있는 재무 지표 데이터 프레임
    출력: 재무 데이터 기반 생성된 분기별 주식 검색용 날짜
    -> 주식 관련 지표 (PER, PBR 등)을 받아오기 위해선 종가가 필요하기 때문에 
       일반화된 분기별 날짜로 매핑시켜 종가, 상장주식 수를 가져오기 위해 필요한 함수
    """
    dates = []
    for quarter in financial_df.index:
        year, q = quarter.split('_')
        # 분기별 날짜 매핑
        quarter_dates = {'Q1': f"{year}0331",
                         'Q2': f"{year}0630", 
                         'Q3': f"{year}0930",
                         'Q4': f"{year}1231"}
        dates.append(quarter_dates[q])
    # print(f"재무데이터 기반 생성된 날짜들: {dates}")
    return dates


class StockInfo:
    def __init__(self, stock_code, dates, market="ALL"):
        self.stock_code = stock_code
        self.dates = dates
        self.market = market
        self._data = self._load_data()
    

    def _get_trading_day(self, date_str):
        """
        특정 날짜가 휴일이라면 증권 시장이 열리지 않기 때문에 휴일인 경우 가장 가까운 trading day를 반환하는 함수
        입력: 분기별 재무 데이터 기반으로 생성된 날짜
        출력: 증권 시장이 열리는 가장 가까운 날 (해당 날짜 이전으로 검색)
        """
        dt = datetime.strptime(date_str, "%Y%m%d")
        for _ in range(10):  # 최대 10일 전까지 검색
            try:
                datestr = dt.strftime("%Y%m%d")
                df = stock.get_market_ohlcv(datestr, market=self.market)
                if not df.empty and df["종가"].sum() != 0:
                    return datestr
            except:
                pass
            dt -= timedelta(days=1)
        return date_str  # 찾지 못하면 원래 날짜 반환
    

    def _load_data(self):
        """
        주어진 날짜들에 대해 주식 시장 데이터를 수집하여 통합 DataFrame 생성
        입력: None
        출력: 날짜별 시가총액, 종가, PER, PBR, EPS 정보가 포함된 DataFrame
        -> 각 날짜에 대해 거래일을 확인하고 해당 종목의 시장 데이터를 수집
        데이터 수집 실패 시 경고 메시지와 함께 해당 날짜 건너뜀
        """
        rows = []
        for orig_date in self.dates:
            try:
                date = self._get_trading_day(orig_date)
                cap_df = stock.get_market_cap(date, market=self.market)
                ohlcv_df = stock.get_market_ohlcv(date, market=self.market)
                fund_df = stock.get_market_fundamental(date, market=self.market)
                if (self.stock_code in cap_df.index and self.stock_code in ohlcv_df.index and self.stock_code in fund_df.index):
                    rows.append({"orig_date": orig_date,
                                 "market_cap": float(cap_df.loc[self.stock_code, "시가총액"]),
                                 "close_price": float(ohlcv_df.loc[self.stock_code, "종가"]),
                                 "PER": float(fund_df.loc[self.stock_code, "PER"]),
                                 "PBR": float(fund_df.loc[self.stock_code, "PBR"]),
                                 "EPS": float(fund_df.loc[self.stock_code, "EPS"])})
            except Exception as e:
                print(f"[경고] {orig_date} 데이터 수집 실패: {e}")
                continue
        df = pd.DataFrame(rows)
        return df.set_index('orig_date') if not df.empty else pd.DataFrame()



## DART Open API 사용해서 재무제표를 불러온 후 지표 계산 ===================== 

REPORT_CODES = {'11011': '사업보고서', '11012': '반기보고서', '11013': '1분기보고서', '11014': '3분기보고서'}
FS_DIV = {'OFS': '재무제표', 'CFS': '연결재무제표'}


def fetch_financial_statements(DART_API_KEY, corp_code, bsns_year, reprt_code, fs_div):
    """
    DART Open API를 통해 특정 기업의 재무제표 데이터 조회
    입력: API키, 기업코드, 사업연도, 보고서코드, 재무제표구분
    출력: 재무제표 항목별 데이터 리스트 (조회 실패시 빈 리스트)
    -> XML 응답을 파싱하여 계정명, 당기금액 등의 재무정보 추출
    """
    ## DART 단일회사 전체 재무제표 불러오는 API url (참고 개발가이드: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2019020)
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.xml"
    params = {'crtfc_key': DART_API_KEY, 'corp_code': corp_code, 'bsns_year': bsns_year, 'reprt_code': reprt_code, 'fs_div': fs_div}

    response = requests.get(url, params=params)
    if response.status_code != 200:
        return []
    root = ET.fromstring(response.content)
    status = root.findtext("status")
    if status != '000':
        if status == '013':
            print(f"[정보] {bsns_year}년 {REPORT_CODES[reprt_code]}: 조회된 데이터 없음.")
        return []
    
    results = []
    for item in root.findall("list"):
        results.append({'bsns_year': bsns_year,
                        'report_type': REPORT_CODES[reprt_code],
                        'account_nm': item.findtext('account_nm'),
                        'fs_div': fs_div,
                        'sj_div': item.findtext('sj_div'),
                        'thstrm_amount': item.findtext('thstrm_amount')})
    return results


def fetch_latest_quarter_report(DART_API_KEY, corp_code, fs_div='CFS'): 
    """최신 분기 보고서 하나만 가져오기"""    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # 현재 시점에서 예상되는 최신 분기 (일반적인 분기 보고서 업로드 주기)
    if current_month <= 3:
        expected_latest_quarter = 'Q4'
        expected_latest_year = current_year - 1
    elif current_month <= 6:
        expected_latest_quarter = 'Q1'
        expected_latest_year = current_year
    elif current_month <= 9:
        expected_latest_quarter = 'Q2'
        expected_latest_year = current_year
    else:
        expected_latest_quarter = 'Q3'
        expected_latest_year = current_year
    ## 디버깅
    # print(f"예상 최신 분기: {expected_latest_year}년 {expected_latest_quarter}")
    
    # 보고서 코드 매핑
    quarter_to_report_code = {'Q1': '11013', 'Q2': '11012', 'Q3': '11014', 'Q4': '11011'}
    # 실제 존재하는 최신 분기 찾기
    year = expected_latest_year
    quarter_nums = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    current_q_num = quarter_nums[expected_latest_quarter]

    # 최대 8분기까지 거슬러 올라가면서 실제 존재하는 보고서 찾기
    for _ in range(8):
        quarter = f'Q{current_q_num}'
        code = quarter_to_report_code[quarter]
        ## 디버깅
        # print(f"[확인 중] {year}년 {quarter} 보고서 존재 여부")
        result = fetch_financial_statements(DART_API_KEY, corp_code, str(year), code, fs_div)
        if result:  # 데이터가 존재하면
            print(f"[최신 분기보고서] {year}년 {quarter} 보고서")
            return pd.DataFrame(result)
        
        # 이전 분기로 이동
        current_q_num -= 1
        if current_q_num == 0:
            current_q_num = 4
            year -= 1
    
    # 찾지 못한 경우 빈 DataFrame 반환
    print("[경고] 최신 분기 보고서를 찾을 수 없습니다.")
    return pd.DataFrame()


def fetch_latest_n_years_reports(DART_API_KEY, corp_code, n_years=2, fs_div='CFS'):
    """최근부터 N년의 분기별 데이터 수집"""    
    current_year = datetime.now().year
    current_month = datetime.now().month
    
    # 현재 시점에서 예상되는 최신 분기
    if current_month <= 3:
        expected_latest_quarter = 'Q4'
        expected_latest_year = current_year - 1
    elif current_month <= 6:
        expected_latest_quarter = 'Q1'
        expected_latest_year = current_year
    elif current_month <= 9:
        expected_latest_quarter = 'Q2'
        expected_latest_year = current_year
    else:
        expected_latest_quarter = 'Q3'
        expected_latest_year = current_year
    # 보고서 코드 매핑
    quarter_to_report_code = {'Q1': '11013', 'Q2': '11012', 'Q3': '11014', 'Q4': '11011'}
    
    # 실제 존재하는 최신 분기 찾기
    def find_actual_latest_quarter():
        year = expected_latest_year
        quarter_nums = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
        current_q_num = quarter_nums[expected_latest_quarter]
        # 최대 8분기까지 거슬러 올라가면서 실제 존재하는 보고서 찾기
        for _ in range(8):
            quarter = f'Q{current_q_num}'
            code = quarter_to_report_code[quarter]
            result = fetch_financial_statements(DART_API_KEY, corp_code, str(year), code, fs_div)
            if result:  # 데이터가 존재하면
                return year, quarter
            
            # 이전 분기로 이동
            current_q_num -= 1
            if current_q_num == 0:
                current_q_num = 4
                year -= 1
        
        # 찾지 못한 경우 기본값 반환
        return expected_latest_year, expected_latest_quarter
    
    actual_latest_year, actual_latest_quarter = find_actual_latest_quarter()  ## 가장 최근의 분기 보고서 연도, 분기
    
    # 실제 최신 분기부터 3년치 + 증가율 계산용 1분기 수집
    quarters_to_collect = []
    year = actual_latest_year
    quarter_nums = {'Q1': 1, 'Q2': 2, 'Q3': 3, 'Q4': 4}
    current_q_num = quarter_nums[actual_latest_quarter]
    
    for _ in range(n_years * 4 + 2):
        quarters_to_collect.append((year, f'Q{current_q_num}'))
        # 이전 분기로 이동
        current_q_num -= 1
        if current_q_num == 0:
            current_q_num = 4
            year -= 1
    ## 디버깅
    # print(f"수집할 분기들: {[(y, q) for y, q in quarters_to_collect]}")
    
    # 데이터 수집
    all_report = []
    successful_quarters = []
    
    for year, quarter in quarters_to_collect:
        code = quarter_to_report_code[quarter]
        try:
            print(f"[수집 중] {year}년 {REPORT_CODES[code]} ({quarter})")
            result = fetch_financial_statements(DART_API_KEY, corp_code, str(year), code, fs_div)
            if result:  # 데이터가 실제로 있는 경우만
                all_report.extend(result)
                successful_quarters.append(f"{year}_{quarter}")
            else:
                print(f"[건너뜀] {year}년 {quarter} - 데이터가 존재하지 않습니다.")
        except Exception as e:
            print(f"[오류 발생] {year}년 {quarter}: {e}")
    
    print(f"실제 수집된 분기들: {successful_quarters}")
    return pd.DataFrame(all_report)


def prepare_finreports(all_reports):
    """
    수집된 재무보고서 데이터를 피벗 테이블 형태로 변환
    입력: 전체 재무보고서 데이터 DF
    출력: 분기별(행) x 계정명(열) 형태의 피벗 테이블
    -> 연결재무제표(CFS)의 BS, IS, CF 데이터만 필터링
       보고서 유형을 분기(Q1~Q4)로 매핑하여 year_quarter 컬럼 생성
    """
    reports = all_reports[(all_reports['report_type'].isin(['사업보고서', '1분기보고서', '반기보고서', '3분기보고서'])) &
                          (all_reports['fs_div'] == 'CFS') &
                          (all_reports['sj_div'].isin(['BS','IS','CF']))].copy()
    quarter_map = {'사업보고서': 'Q4', '1분기보고서': 'Q1', '반기보고서': 'Q2', '3분기보고서': 'Q3'}
    reports['quarter'] = reports['report_type'].map(quarter_map)
    reports['year_quarter'] = reports['bsns_year'] + '_' + reports['quarter']
    return reports.pivot_table(index='year_quarter', columns='account_nm', values='thstrm_amount', aggfunc='first').rename_axis(columns=None)


def clean_columns(df):
    """
    반환된 DF 컬럼명 정리 및 중복 컬럼 병합 처리
    입력: 원본 DataFrame
    출력: 컬럼명이 정리되고 중복이 제거된 DataFrame
    -> 실제 분기별 재무제표에서 컬럼명이 다른 것을 발견
    -> 컬럼명에서 괄호 내용 제거, 공백 제거 등 전처리 수행
       동일한 이름의 중복 컬럼들을 병합하여 데이터 손실 최소화
    -> 추후 더 디벨롭 가능
    """
    df_clean = df.copy()
    
    # 칼럼명 전처리
    new_columns = {}
    for col in df_clean.columns:
        if pd.isna(col): 
            continue
        clean_col = str(col)
        clean_col = re.sub(r'\([^)]*\)', '', clean_col)
        clean_col = re.sub(r'\s+', '', clean_col)
        new_columns[col] = clean_col
    df_clean = df_clean.rename(columns=new_columns)
    
    # 중복 칼럼 병합
    duplicate_cols = df_clean.columns[df_clean.columns.duplicated()].unique()
    if len(duplicate_cols) > 0:
        for col in duplicate_cols:
            same_name_indices = df_clean.columns.get_indexer_for([col])
            base_data = df_clean.iloc[:, same_name_indices[0]].copy()
            for i in range(1, len(same_name_indices)):
                other_data = df_clean.iloc[:, same_name_indices[i]]
                mask = base_data.isna() & other_data.notna()
                base_data.loc[mask] = other_data.loc[mask]
            df_clean.iloc[:, same_name_indices[0]] = base_data
        df_clean = df_clean.loc[:, ~df_clean.columns.duplicated()]
    
    return df_clean.dropna(axis=1)



## 재무제표 불러와서 기본적인 주요 재무 지표 계산  ===================== 

class MetricsCalculator:
    def __init__(self, financial_df, stock_df=None):
        self.full_financial_df = financial_df.copy()  # 전체 데이터
        self.stock_df = stock_df.copy() if stock_df is not None else None
        
        all_quarters = sorted(financial_df.index.astype(str))
        if len(all_quarters) > 1:
            self.analysis_quarters = all_quarters[1:]  # 첫 번째 분기 제외 (because 첫 번째 분기는 증가율 계산용)
        else:
            self.analysis_quarters = []
        ## 디버깅
        # print(f"수집된 전체 분기: {all_quarters}")
        # print(f"실제 분석 대상 분기: {self.analysis_quarters}")

        self._prepare_data()
        self.combined_df = self._combine_data()

    def _prepare_data(self):
        """
        재무데이터 전처리 및 파생 지표 계산
        입력: None
        출력: None (인스턴스 변수 업데이트)
        -> 매출액이 없는 경우 매출총이익 + 매출원가로 계산
        -> 차입금 관련 항목들을 모두 합산하여 총차입금 생성
        """
        df = self.full_financial_df.copy()
        
        # 매출액 계산
        revenue_candidates = ['매출액', '매출', '수익총계', '영업수익']
        if not any(col in df.columns for col in revenue_candidates):
            if all(col in df.columns for col in ['매출총이익', '매출원가']):
                df['매출액'] = pd.to_numeric(df['매출총이익'], errors='coerce') + pd.to_numeric(df['매출원가'], errors='coerce')
        
        # 총차입금 계산
        debt_components = ['단기차입금', '장기차입금', '사채', '차입금']
        available_debt_components = [col for col in debt_components if col in df.columns]
        if available_debt_components:
            df['총차입금'] = df[available_debt_components].apply(pd.to_numeric, errors='coerce').sum(axis=1)
        
        # CAPEX 계산
        capex_candidates = ['유형자산의취득', '유형자산취득', '설비투자', '자본적지출']
        for candidate in capex_candidates:
            if candidate in df.columns:
                df['CAPEX'] = pd.to_numeric(df[candidate], errors='coerce')
                break
        self.full_financial_df = df


    def _combine_data(self):
        """
        분석 대상 분기의 재무데이터와 주식데이터를 결합
        입력: None
        출력: 재무데이터와 주식데이터가 결합된 DF
        -> 분석 대상 분기만 추출해 재무데이터 필터링
        -> 분기별 날짜 매핑을 통해 주식데이터와 조인
        -> if 주식데이터가 없는 경우 재무데이터만 반환
        """
        # 분석 대상 분기만 추출
        analysis_financial = self.full_financial_df.loc[self.full_financial_df.index.astype(str).isin(self.analysis_quarters)]
        
        if self.stock_df is None or self.stock_df.empty:
            return analysis_financial
        
        # 주식데이터 분기별 매핑
        stock_quarter_data = []
        for quarter in self.analysis_quarters:
            year, quarter_num = quarter.split('_')
            quarter_dates = {'Q1': f"{year}0331", 'Q2': f"{year}0630", 'Q3': f"{year}0930", 'Q4': f"{year}1231"}
            target_date = quarter_dates[quarter_num]
            if target_date in self.stock_df.index:
                stock_data = self.stock_df.loc[target_date].copy()
                stock_data.name = quarter
                stock_quarter_data.append(stock_data)
        if stock_quarter_data:
            stock_df_quarters = pd.DataFrame(stock_quarter_data)
            return analysis_financial.join(stock_df_quarters, how='left')
        else:
            return analysis_financial


    def _get_col(self, col_name):
        """
        컬럼명의 안전한 조회 및 대안 검색 수행
        """
        target_df = self.combined_df
        
        # 정확한 매치 먼저 탐색
        if col_name in target_df.columns:
            return pd.to_numeric(target_df[col_name], errors='coerce')
        # 부분 매치 탐색
        similar_cols = [col for col in target_df.columns if col_name in str(col) or str(col) in col_name]
        if similar_cols:
            return pd.to_numeric(target_df[similar_cols[0]], errors='coerce')
        # 없으면 NaN 시리즈 반환
        return pd.Series([np.nan] * len(self.analysis_quarters), index=self.analysis_quarters)


    def _safe_calc(self, num1, num2, operation='divide', multiply=1):
        """
        안전한 수치 계산 수행 (0으로 나누기, NaN 처리 포함)
        입력: 피연산자1, 피연산자2, 연산종류, 곱셈계수 (퍼센테이지 계산 시 100 사용)
        출력: 계산 결과 Series (오류시 NaN 처리)
        -> divide, subtract, add 연산 지원
        -> 분모가 0이거나 NaN인 경우 안전하게 NaN 반환
        """
        if operation == 'divide':
            result = np.where((num2 != 0) & ~pd.isna(num2) & ~pd.isna(num1), (num1 / num2) * multiply, np.nan)
        elif operation == 'subtract':
            result = np.where(~pd.isna(num1) & ~pd.isna(num2), num1 - num2, np.nan)
        elif operation == 'add':
            result = np.where(~pd.isna(num1) & ~pd.isna(num2), num1 + num2, np.nan)
        else:
            result = np.full(len(self.analysis_quarters), np.nan)
        return pd.Series(result, index=self.analysis_quarters)


    def _growth_rate_with_prev_quarter(self, col_name):
        """
        전 분기 대비 증가율 계산
        입력: 증가율을 계산할 컬럼명
        출력: 분기별 전 분기 대비 증가율(%) Series
        -> Q1의 전 분기는 전 년도 Q4로 설정 (전 분기 값이 0이거나 없는 경우 NaN 반환)
        """
        growth_rates = {}
        for quarter in self.analysis_quarters:
            current_year, current_q = quarter.split('_')
            # 전분기 계산
            if current_q == 'Q1':
                prev_quarter = f"{int(current_year)-1}_Q4"
            else:
                q_num = int(current_q[1]) - 1
                prev_quarter = f"{current_year}_Q{q_num}"
            if (prev_quarter in self.full_financial_df.index and 
                col_name in self.full_financial_df.columns and
                quarter in self.combined_df.index and
                col_name in self.combined_df.columns):

                prev_val = pd.to_numeric(self.full_financial_df.loc[prev_quarter, col_name], errors='coerce')
                curr_val = pd.to_numeric(self.combined_df.loc[quarter, col_name], errors='coerce')
                
                if not pd.isna(prev_val) and not pd.isna(curr_val) and prev_val != 0:
                    growth_rate = ((curr_val - prev_val) / prev_val) * 100
                    growth_rates[quarter] = growth_rate
                else:
                    growth_rates[quarter] = np.nan
            else:
                growth_rates[quarter] = np.nan
        
        return pd.Series(growth_rates)


    def calculate_metrics(self):
        """
        모든 재무 지표 계산 수행
        입력: None
        출력: 지표명을 키로 하는 Series들의 딕셔너리
        -> 성장성, 수익성, 안정성, 활동성, 주식 지표 등 포괄적 계산
        """
        metrics = {}
        
        ## 성장성 지표
        growth_items = {'매출액증가율': '매출액',
                        '총자산증가율': '자산총계',
                        '유형자산증가율': '유형자산',
                        '영업이익증가율': '영업이익',
                        '당기순이익증가율': '당기순이익',
                        '자기자본증가율': '자본총계',
                        '부채총계증가율': '부채총계'}
        
        for metric_name, col_name in growth_items.items():
            metrics[metric_name] = self._growth_rate_with_prev_quarter(col_name)
        
        ## 기본 지표
        metrics.update({'CAPEX': self._get_col('CAPEX'),
                        'FCF': self._safe_calc(self._get_col('영업활동현금흐름'), self._get_col('CAPEX'), 'subtract')})
        
        ## 수익성 지표 
        metrics.update({'매출액영업이익률': self._safe_calc(self._get_col('영업이익'), self._get_col('매출액'), 'divide', 100),
                        '매출액세전순이익률': self._safe_calc(self._get_col('법인세비용차감전순이익'), self._get_col('매출액'), 'divide', 100),
                        '당기순이익률': self._safe_calc(self._get_col('당기순이익'), self._get_col('매출액'), 'divide', 100),
                        '순이익률': self._safe_calc(self._get_col('당기순이익'), self._get_col('매출액'), 'divide', 100),
                        'ROE': self._safe_calc(self._get_col('당기순이익'), self._get_col('자본총계'), 'divide', 100),
                        'ROA': self._safe_calc(self._get_col('당기순이익'), self._get_col('자산총계'), 'divide', 100),
                        'OP마진': self._safe_calc(self._get_col('영업이익'), self._get_col('매출액'), 'divide', 100),
                        '매출총이익률': self._safe_calc(self._get_col('매출총이익'), self._get_col('매출액'), 'divide', 100),
                        '세전계속사업이익': self._get_col('법인세비용차감전순이익'),
                        '당기순이익': self._get_col('당기순이익'),
                        '지배주주순이익': self._get_col('지배기업소유주지분')})
        
        ## 안정성 지표
        metrics.update({'부채비율': self._safe_calc(self._get_col('부채총계'), self._get_col('자본총계'), 'divide', 100),
                        '차입금의존도': self._safe_calc(self._get_col('총차입금'), self._get_col('자산총계'), 'divide', 100),
                        '유동비율': self._safe_calc(self._get_col('유동자산'), self._get_col('유동부채'), 'divide', 100),
                        '자기자본비율': self._safe_calc(self._get_col('자본총계'), self._get_col('자산총계'), 'divide', 100),
                        '금융비용대부채비율': self._safe_calc(self._get_col('금융비용'), self._get_col('부채총계'), 'divide', 100),
                        '현금및유가증권비율': self._safe_calc(self._get_col('현금및현금성자산'), self._get_col('자산총계'), 'divide', 100),
                        '자산총계': self._get_col('자산총계'),
                        '부채총계': self._get_col('부채총계'),
                        '자본총계': self._get_col('자본총계'),
                        '총차입금': self._get_col('총차입금')})
        
        ## 활동성 지표
        metrics.update({'총자본회전율': self._safe_calc(self._get_col('매출액'), self._get_col('자본총계'), 'divide'),
                        '총자산회전율': self._safe_calc(self._get_col('매출액'), self._get_col('자산총계'), 'divide'),
                        '재고자산회전율': self._safe_calc(self._get_col('매출원가'), self._get_col('재고자산'), 'divide'),
                        '유형자산회전율': self._safe_calc(self._get_col('매출액'), self._get_col('유형자산'), 'divide'),
                        '자기자본회전율': self._safe_calc(self._get_col('매출액'), self._get_col('자본총계'), 'divide')})
        
        # 주요 지표 
        metrics.update({'매출액': self._get_col('매출액'),
                        '영업이익': self._get_col('영업이익'),
                        '자본금': self._get_col('자본금'),
                        '투자활동현금흐름': self._get_col('투자활동현금흐름'),
                        '재무활동현금흐름': self._get_col('재무활동현금흐름'),
                        '영업활동현금흐름': self._get_col('영업활동현금흐름')})
        
        ## 주식 관련 지표
        market_cap_series = self._get_col('market_cap')
        if not market_cap_series.isna().all():
            metrics.update({'PER': self._get_col('PER'),
                            'PBR': self._get_col('PBR'),
                            'EPS': self._get_col('EPS'),
                            'BPS': self._get_col('BPS'),
                            '발행주식수': self._get_col('listed_shares'),
                            '현금DPS': self._get_col('DPS'),
                            '배당수익률': self._get_col('DIV'),
                            '시가총액': self._get_col('market_cap'),
                            '주가': self._get_col('close_price'),
                            '현금배당수익률': self._safe_calc(self._get_col('DPS'), self._get_col('close_price'), 'divide', 100)})
        
        return metrics


    def get_results_json(self):
        """계산된 모든 지표를 JSON 형태로 변환하여 반환"""
        metrics = self.calculate_metrics()
        json_data = []
        for quarter in self.analysis_quarters:
            row = {'분기': quarter}
            for name, series in metrics.items():
                val = series.get(quarter, None) if hasattr(series, 'get') else None
                row[name] = float(val) if pd.notna(val) and not np.isinf(val) else None
            json_data.append(row)
        return json.dumps(json_data, ensure_ascii=False, indent=2)

    @property
    def analysis_period(self):
        return self.analysis_quarters
    




if __name__ == '__main__': 
    corp = CorpInfo("삼성전자")
    corp_code = corp.corp_code
    ## if 최근으로부터 과거 n년의 보고서 지표를 뽑아내고 싶다면 
    financial_df = fetch_latest_n_years_reports(DART_API_KEY, corp_code, n_years=3)
    financial_df = prepare_finreports(financial_df)
    financial_df = clean_columns(financial_df)

    quarterly_dates = generate_dates_from_financial_df(financial_df)
    stock_info = StockInfo("005930", quarterly_dates)
    stock_df = stock_info._data

    calculator = MetricsCalculator(financial_df, stock_df)
    json_result = calculator.get_results_json()
    df = pd.DataFrame(json.loads(json_result)).set_index('분기')
    df  
