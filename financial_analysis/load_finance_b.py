## load_finance_b.py


from pykrx import stock
import pandas as pd
from datetime import datetime, timedelta
import requests
import xml.etree.ElementTree as ET
from time import sleep

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



## DART Open API 사용해서 주요재무지표를 불러옴 ===================== 

REPORT_CODES = {'11011': '사업보고서', '11012': '반기보고서', '11013': '1분기보고서', '11014': '3분기보고서'}
METRIC_CODES = {'M210000': '수익성지표', 'M220000': '안정성지표', 'M230000': '성장성지표', 'M240000': '활동성지표'}

def api_call(corp_code, year, reprt_code, idx_cl_code):
    """
    DART API 호출하여 특정 기업의 재무지표 데이터 요청
    입력: 기업코드, 연도, 보고서코드, 지표분류코드
    출력: XML root 객체
    """
    params = {'crtfc_key': DART_API_KEY, 'corp_code': corp_code, 'bsns_year': str(year), 'reprt_code': reprt_code, 'idx_cl_code': idx_cl_code}
    ## DART 단일회사 주요 재무지표 불러오는 API url (참고 개발가이드: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2022001)
    response = requests.get("https://opendart.fss.or.kr/api/fnlttSinglIndx.xml", params=params)
    if response.status_code != 200:
        return None    
    root = ET.fromstring(response.content)
    # status가 '000'이면 성공, 아니면 데이터 존재 X
    return root if root.findtext("status") == '000' else None


def get_quarter_candidates():
    """
    최신부터 과거까지 모든 분기 후보 리스트 생성
    입력: None
    출력: (연도, 보고서코드) 튜플의 리스트
    -> 현재연도부터 2022년까지 역순으로 분기 생성
       각 연도별로 1분기 → 반기 → 3분기 → 사업보고서 순서로 배치
       DART API가 2023년 3분기부터 제공되므로 2022년부터 수집 설정
    """
    year = datetime.now().year
    candidates = []
    for y in range(year, 2021, -1): ## 2022년부터 -> 왜냐면 해당 api는 2023년 3분기부터 제공하는데 코드 간결화를 위해 2022년부터 수집 가능하도록 함
        candidates.extend([(y, code) for code in ['11013', '11012', '11014', '11011']])
        # 1분기 → 반기 → 3분기 → 사업보고서
    return candidates


def find_available_quarters(corp_code, n_years=1):
    """
    실제로 데이터가 존재하는 최신 n년간의 분기를 찾아서 반환
    입력: 기업코드, 조회할 연 수
    출력: 데이터가 존재하는 (연도, 보고서코드) 튜플 리스트
    -> 수익성지표를 테스트용으로 사용하여 분기별 데이터 존재 여부 확인
       목표 분기 수(n_years * 4)에 도달하면 검색 종료
       API 호출 제한 방지를 위해 0.2초 간격 적용
    """
    quarters = []
    target_quarters = n_years * 4  # n년 * 4분기
    for year, reprt_code in get_quarter_candidates():
        if len(quarters) >= target_quarters:  # 목표 분기 수에 도달하면 종료
            break
        # 수익성지표로 테스트해서 해당 분기 데이터 존재 여부 확인
        if api_call(corp_code, year, reprt_code, 'M210000'):
            quarters.append((year, reprt_code))
        sleep(0.2)  # API 호출 제한 방지
    return quarters


def get_metrics(corp_code, year, reprt_code, idx_cl_code):
    """
    특정 분기의 특정 지표 카테고리 데이터를 딕셔너리로 추출
    입력: 기업코드, 연도, 보고서코드, 지표분류코드
    출력: 지표명을 키로 하는 지표값 딕셔너리 (실패시 빈 딕셔너리)
    -> XML 응답에서 지표명(idx_nm)과 지표값(idx_val) 추출
       숫자 변환 가능한 값은 float로, 불가능한 값은 문자열로 저장
       API 호출 실패시 빈 딕셔너리 반환
    """
    root = api_call(corp_code, year, reprt_code, idx_cl_code)
    if not root:
        return {}
    metrics = {}
    # XML에서 지표명과 지표값을 추출하여 딕셔너리로 변환
    for item in root.findall("list"):
        name, value = item.findtext('idx_nm'), item.findtext('idx_val')
        if name and value:
            try: metrics[name] = float(value)  # 숫자면 float로 변환
            except: metrics[name] = value  # 변환 안되면 문자열 그대로
    return metrics


def fetch_latest_n_years_reports(corp_code, n_years=1):
    """
    기업의 최신 n년간 모든 재무지표를 수집하여 지표별 DataFrame으로 반환
    입력: 기업코드, 조회할 연수
    출력: 지표분류별 DataFrame 딕셔너리
    -> 수익성, 안정성, 성장성, 활동성 지표를 카테고리별로 수집 (해당 카테고리로 나뉘어져 있음)
       각 분기별로 4가지 지표 카테고리 데이터 순차 수집
       수집된 데이터를 지표분류별 DataFrame으로 구성하여 반환
    """
    quarters = find_available_quarters(corp_code, n_years)  # n_years 매개변수 전달
    result = {category: [] for category in METRIC_CODES.values()}  # 지표별 결과 저장
    
    print(f"수집할 분기 수: {len(quarters)}개 (목표: {n_years}년)")
    
    # 각 분기별로 4가지 지표 카테고리 데이터 수집
    for year, reprt_code in quarters:
        ## 디버깅
        # print(f"[수집 중] {year}년 {REPORT_CODES[reprt_code]}")
        for idx_cl_code, category in METRIC_CODES.items():
            metrics = get_metrics(corp_code, year, reprt_code, idx_cl_code)
            if metrics:  # 데이터가 있으면 결과에 추가
                result[category].append({'연도': year, '보고서': REPORT_CODES[reprt_code], **metrics})
            sleep(0.1)  # API 호출 간격 조절
    final_result = {}
    for category, data_list in result.items():
        if data_list:
            df = pd.DataFrame(data_list)
            # 지표분류 컬럼 추가
            if '지표분류' not in df.columns: df['지표분류'] = category
            final_result[category] = df
    return final_result


def integrate_stock_data(financial_data, stock_code):
    """
    재무지표 데이터에 주식 데이터를 통합하여 반환
    입력: 지표별 DF 딕셔너리, 6자리 주식 종목코드
    출력: 주식지표가 추가된 지표별 DF 딕셔너리
    -> 재무데이터에서 분기 정보를 추출하여 주식 데이터 수집 기간 결정
       StockInfo 클래스를 통해 해당 분기별 주식 데이터 수집
       날짜를 보고서명으로 변환하여 재무지표와 동일한 형태로 통일
       기존 재무지표 딕셔너리에 '주식지표' 카테고리 추가하여 반환
    """
    
    # 재무데이터에서 분기 정보 추출
    quarters = []
    for category, df_cat in financial_data.items():
        if not df_cat.empty:
            for _, row in df_cat.iterrows():
                year, report = row['연도'], row['보고서']
                quarter_map = {'1분기보고서': 'Q1', 
                               '반기보고서': 'Q2', 
                               '3분기보고서': 'Q3', 
                               '사업보고서': 'Q4'}
                if report in quarter_map:
                    quarters.append(f"{year}_{quarter_map[report]}")
    # 중복 제거 및 정렬
    quarters = sorted(list(set(quarters)))
    
    if not quarters:
        print("[주의] 재무데이터에서 분기 정보를 찾을 수 없습니다.")
        return financial_data
    
    # 주식 데이터 수집
    try:
        # 임시 DF로 날짜 생성
        temp_df = pd.DataFrame(index=quarters)
        quarterly_dates = generate_dates_from_financial_df(temp_df)
        # 주식 데이터 수집
        stock_info = StockInfo(stock_code, quarterly_dates)
        stock_df = stock_info._data
        if stock_df.empty:
            print(f"[주의] 주식코드 {stock_code}의 {quarterly_dates}의 주식 데이터를 수집할 수 없습니다.")
            return financial_data
        # 주식 데이터를 재무지표와 동일한 형태로 변환
        stock_formatted = stock_df.reset_index()
        stock_formatted['연도'] = stock_formatted['orig_date'].str[:4].astype(int)
        stock_formatted['지표분류'] = '주식지표'


        def date_to_report(date):
            """날짜를 보고서명으로 변환"""
            month = date[4:6]
            if month == '03': return '1분기보고서'
            elif month == '06': return '반기보고서'
            elif month == '09': return '3분기보고서'
            else: return '사업보고서'
        

        stock_formatted['보고서'] = stock_formatted['orig_date'].apply(date_to_report)
        # 불필요한 'orig_date' 컬럼 제거
        stock_formatted = stock_formatted.drop('orig_date', axis=1)
        
        # 기존 재무지표에 주식지표 칼럼 추가
        financial_data_with_stock = financial_data.copy()
        financial_data_with_stock['주식지표'] = stock_formatted
        return financial_data_with_stock
        
    except Exception as e:
        print(f"[에러] 주식 데이터 통합 실패: {e}")
        return financial_data



if __name__ == "__main__":
    corp = CorpInfo("삼성전자")
    corp_code = corp.corp_code
    stock_code = corp.corp_info.iloc[0]['stock_code']  # 주식코드 추출
    
    # 재무지표 수집
    data = fetch_latest_n_years_reports(corp_code, n_years=2)
    
    # 주식 데이터 통합
    data_with_stock = integrate_stock_data(data, stock_code)
    
    # DF 통합
    financial_combined = []
    for category, df_cat in data_with_stock.items():
        if not df_cat.empty:
            financial_combined.append(df_cat)
    
    if financial_combined:
        final_df = pd.concat(financial_combined, ignore_index=True)
        print("=== 재무지표 + 주식지표 통합 DataFrame ===")
        print(f"총 행 수: {len(final_df)}")
        print(f"지표분류: {final_df['지표분류'].unique().tolist()}")
        print(final_df.head())
