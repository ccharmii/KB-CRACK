## load_samecorpmean_fromcsv.py


import pandas as pd
import requests
import xml.etree.ElementTree as ET
from time import sleep
from datetime import datetime

from .load_corpinfo import CorpInfo
from .load_finance_a import fetch_latest_quarter_report, prepare_finreports, clean_columns
from .load_finance_b import StockInfo, generate_dates_from_financial_df
import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

DART_API_KEY = os.getenv('DART_API_KEY')

# 보고서 코드와 지표 코드
REPORT_CODES = {'11011': '사업보고서', '11012': '반기보고서', '11013': '1분기보고서', '11014': '3분기보고서'}
MULTI_METRIC_CODES = {'M210000': '수익성지표',' M220000': '안정성지표', 'M230000': '성장성지표', 'M240000': '활동성지표'}

def load_krx_data(file_path):
    """KRX CSV 데이터 로드 및 전처리"""
    try:
        df = pd.read_csv(file_path, encoding='cp949')
        df.columns = df.columns.str.strip().str.replace(' ', '')
        print(f"정리된 컬럼명: {list(df.columns)}")
        if '종목코드' in df.columns:
            df['종목코드'] = df['종목코드'].astype(str).str.zfill(6)
        ## 디버깅
        print(f"KRX 데이터 로드 완료: {len(df)}개 기업")
        return df
    except Exception as e:
        print(f"CSV 파일 로드 실패: {e}")
        return pd.DataFrame()


def classify_asset_size(total_assets_billion):
    """
    자산총액 기준으로 기업 규모 분류 
    입력: 자산총액 (단위: 억원)
    출력: 기업 규모 분류 문자열 (대기업/중소기업/중견기업/미분류)
    -> 참고한 기준: 중소기업 현황정보시스템, 중견기업 정보마당, 기업집단포털 등 
    -> 5000억 미만은 중소기업, 5000억 이상 5조 미만은 중견기업, 5조 이상은 대기업으로 분류
       NaN이나 0 이하 값, 그 외의 값은 미분류로 처리
    """
    if pd.isna(total_assets_billion) or total_assets_billion <= 0:
        return "미분류"
    if 0 < total_assets_billion < 5000: ## 5000억 미만
        return "중소기업"
    elif 5000 <= total_assets_billion < 50000: 
        return "중견기업"
    elif 50000 <= total_assets_billion:
        return "대기업"
    else:
        return "미분류"


def classify_company_age(est_date):
    """
    설립일 기준으로 기업 업력(성장단계) 분류
    입력: 설립일 (문자열 YYYYMMDD 형태 또는 datetime 객체)
    출력: 업력 분류 문자열 (도입기/도약기/성장기/성장후기·성숙기/미분류)
    -> 현재 시점 기준으로 업력 계산하여 4단계로 분류
       3년 이하: 도입기, 7년 이하: 도약기, 15년 이하: 성장기
       15년 초과: 성장후기·성숙기, 날짜 변환 실패시: 미분류
    -> 참고한 기준: 논문 '기업 업력구분에 따른 신용보증 지원효과 분석' 
    """
    try:
        if pd.isna(est_date) or not est_date:
            return "미분류"
        if isinstance(est_date, str):
            est_date = datetime.strptime(est_date, '%Y%m%d')
        years = (datetime.now() - est_date).days / 365
        if years <= 3:
            return "도입기"
        elif years <= 7:
            return "도약기"
        elif years <= 15:
            return "성장기"
        else:
            return "성장후기·성숙기"
    except:
        return "미분류"


def get_company_financial_info(corp_code):
    """
    기업의 최신 분기 재무정보에서 자산총계 추출
    입력: 기업 DART 코드
    출력: 자산총계 (단위: 억원, 실패시 None)
    -> 최신 분기 재무제표에서 자산총계 항목 추출
    """
    try:
        financial_df = fetch_latest_quarter_report(DART_API_KEY, corp_code)
        if not financial_df.empty:
            financial_df = prepare_finreports(financial_df)
            financial_df = clean_columns(financial_df)
            if not financial_df.empty and '자산총계' in financial_df.columns:
                latest_quarter = financial_df.index[-1]
                total_assets = pd.to_numeric(financial_df.loc[latest_quarter, '자산총계'], errors='coerce')
                if not pd.isna(total_assets) and total_assets > 0:
                    return total_assets / 100000000
    except Exception as e:
        print(f"[WARNING] {corp_code} 재무데이터 수집 실패: {e}")
    return None


def validate_and_filter_companies(industry_companies, target_count=5):
    """
    우선주 제외하고 유효한 기업만 target_count개 확보
    입력: 동종업계 기업 리스트, 목표 기업 수
    출력: corp_code가 유효한 기업들의 리스트
    -> 우선주, 리츠 등 특수주식을 제외하고 일반 기업만 필터링 (dart에서 기업 정보를 제공하는 기업만)
       CorpInfo를 통해 corp_code 변환 가능한 기업만 선별
    """
    valid_companies = []
    excluded_companies = []
    for company in industry_companies:
        corp_name = company['corp_name']
        # 우선주 및 특수주식 제외
        if any(keyword in corp_name for keyword in ['우', '우선주', '리츠']):
            ## 디버깅
            # print(f"{corp_name}: 특수주식이므로 제외")
            excluded_companies.append(corp_name)
            continue
        try:
            # 일반 기업만 처리
            corp_info_obj = CorpInfo(corp_name)
            correct_corp_code = corp_info_obj.corp_code
            valid_companies.append({'corp_name': corp_name, 'corp_code': correct_corp_code})
        except Exception as e:
            excluded_companies.append(corp_name)
            continue
        # 목표 개수에 도달하면 중단
        if len(valid_companies) >= target_count:
            break
    ## 디버깅
    print(f"\n검증 완료: 유효 {len(valid_companies)}개, 제외 {len(excluded_companies)}개")
    # if excluded_companies:
    #     print(f"제외된 기업: {excluded_companies}")
    return valid_companies[:target_count]


class IndustryAnalyzer:
    """CSV 기반 동종업계 분석기"""
    
    def __init__(self, file_path):
        self.krx_data = load_krx_data(file_path)
        self.target_info = None
        
    def get_target_company_info(self, target_corp_name):
        """
        타겟 기업의 상세 정보 수집 및 분류 정보 생성
        입력: 타겟 기업명
        출력: 타겟 기업의 종합 정보 딕셔너리 (실패시 None)
        -> KRX 데이터와 DART 정보를 매칭해 업종, 시장구분, 자산규모, 업력 등의 분류 정보 추가
        """
        try:
            corp_info_obj = CorpInfo(target_corp_name)
            target_corp_info = corp_info_obj.corp_info.iloc[0].to_dict()
            target_corp_code = corp_info_obj.corp_code
            krx_match = self.krx_data[self.krx_data['종목명'] == target_corp_name]
            if krx_match.empty:
                print(f"[ERROR] {target_corp_name}을 KRX 데이터에서 찾을 수 없습니다.")
                return None
            match_row = krx_match.iloc[0]
            ## 디버깅
            print(f"[매칭 성공] {match_row['종목명']} ({match_row['종목코드']})")
            # 자산총액
            target_assets = get_company_financial_info(target_corp_code)
            self.target_info = {'corp_name': target_corp_name,
                                'corp_code': target_corp_code,
                                'industry': match_row['업종명'],
                                'market': match_row['시장구분'],
                                'market_cap': match_row['시가총액'],
                                'asset_size': classify_asset_size(target_assets),
                                'company_age': classify_company_age(target_corp_info.get('est_dt')),
                                'total_assets': target_assets}
            
            print(f"\n=== {target_corp_name} 분류 정보 ===")
            print(f"업종: {self.target_info['industry']}")
            print(f"시장구분: {self.target_info['market']}")
            print(f"자산규모: {self.target_info['asset_size']}")
            print(f"업력: {self.target_info['company_age']}")
            print(f"시가총액: {self.target_info['market_cap']:,}백만원")
            print("="*50)
            
            return self.target_info
            
        except Exception as e:
            print(f"타겟 기업 정보 수집 실패: {e}")
            return None
    


    def find_similar_companies(self, max_companies=15):
        """
        동종업계 유사 기업 검색 (타겟 기업 제외)
        입력: 최대 수집 기업 수
        출력: 시가총액 차이 기준으로 정렬된 유사 기업 리스트
        필터링 순서: 업종 → 시장구분 
        -> 자산규모 반영을 위해 시가총액 차이를 계산하여 오름차순 정렬 후 반환
        """
        if not self.target_info:
            return []
        
        # 업종 필터링
        industry_filtered = self.krx_data[self.krx_data['업종명'] == self.target_info['industry']].copy()
        ## 디버깅
        print(f"1단계 업종 필터링: {len(industry_filtered)}개")
        if industry_filtered.empty:
            return []
        
        # 시장구분 필터링
        market_filtered = industry_filtered[industry_filtered['시장구분'] == self.target_info['market']].copy()
        ## 디버깅
        print(f"2단계 시장구분 필터링: {len(market_filtered)}개")
        if market_filtered.empty:
            market_filtered = industry_filtered
            print("시장구분 매칭 실패 - 업종만 고려")
        
        # 타겟 기업 제외 
        target_corp_name = self.target_info['corp_name']
        market_filtered = market_filtered[market_filtered['종목명'] != target_corp_name].copy()
                
        # 시가총액 차이 계산 및 정렬
        similar_companies = []
        target_market_cap = self.target_info['market_cap']
        for _, row in market_filtered.iterrows():
            try:
                # 시가총액 차이 계산
                market_cap_diff = abs(target_market_cap - row['시가총액'])
                company_data = {'corp_name': row['종목명'],
                                'corp_code': row['종목코드'],
                                'market_cap': row['시가총액'],
                                'market_cap_diff': market_cap_diff}
                similar_companies.append(company_data)
            except Exception as e:
                continue
        # 시가총액 차이 기준으로 정렬
        similar_companies.sort(key=lambda x: x['market_cap_diff'])
        ## 디버깅
        print(f"최종 수집: {len(similar_companies)}개 기업 (타겟 기업 제외)")
        return similar_companies[:max_companies]


def get_quarter_candidates():
    """
    최신부터 과거까지 모든 분기 후보 리스트 생성 (다중회사용)
    입력: None
    출력: (연도, 보고서코드) 튜플의 리스트
    -> 현재연도부터 2022년까지 역순으로 분기 생성
       각 연도별로 1분기 → 반기 → 3분기 → 사업보고서 순서로 배치
    """
    year = datetime.now().year
    candidates = []
    for y in range(year, 2021, -1):
        candidates.extend([(y, code) for code in ['11013', '11012', '11014', '11011']])
    return candidates


def get_available_quarters_for_multi_companies(corp_codes, n_years=1):
    """
    다중회사 API로 여러 기업의 공통 분기 확인
    입력: 기업코드 리스트, 조회할 연수
    출력: 공통으로 데이터가 존재하는 분기 리스트
    -> 다중회사 재무지표 API로 각 분기별 데이터 존재 여부 확인
       최소 기업 수 이상의 데이터가 있는 분기만 선별
       목표 분기 수에 도달하면 검색 종료
    """
    corp_codes_str = ",".join(corp_codes)
    available_quarters = []
    target_quarters = n_years * 4
    
    for year, reprt_code in get_quarter_candidates():
        if len(available_quarters) >= target_quarters:
            break
        
        # 다중회사 재무지표 API로 테스트 (수익성지표) 
        ## DART 다중회사 주요 재무지표 불러오는 API url (참고 개발가이드: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2022002)
        url = "https://opendart.fss.or.kr/api/fnlttCmpnyIndx.xml"
        params = {'crtfc_key': DART_API_KEY,
                  'corp_code': corp_codes_str,
                  'bsns_year': str(year),
                  'reprt_code': reprt_code,
                  'idx_cl_code': 'M210000'}
        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                root = ET.fromstring(response.content)
                status = root.findtext("status")
                if status == '000':
                    # 실제 데이터가 있는지 확인
                    companies_with_data = set()
                    for item in root.findall("list"):
                        companies_with_data.add(item.findtext('corp_code'))
                    # 최소 기업 수 확인
                    if len(companies_with_data) >= min(3, len(corp_codes) // 2):
                        available_quarters.append((year, reprt_code))
        except Exception as e:
            print(f"API 호출 오류: {e}")
            continue
        sleep(0.3)
    ## 디버깅
    # print(f"사용 가능한 공통 분기: {len(available_quarters)}개")
    return available_quarters


def fix_corp_names_in_metrics(metrics_df, valid_companies):
    """
    corp_code를 기반으로 DF의 기업명 매핑
    입력: 재무지표 DF, 기업 정보 리스트
    출력: 기업명이 올바르게 매핑된 DataFrame
    """
    if metrics_df.empty:
        return metrics_df
    corp_code_to_name = {comp['corp_code']: comp['corp_name'] for comp in valid_companies}
    # corp_code를 기반 기업명 매핑
    metrics_df['corp_name'] = metrics_df['corp_code'].map(corp_code_to_name)
    # None 값이 있는 경우 확인
    none_count = metrics_df['corp_name'].isna().sum()
    if none_count > 0:
        ## 디버깅
        print(f"[주의] {none_count}개 행의 기업명이 매핑되지 않았습니다.")
    # None 값 제거
    metrics_df = metrics_df.dropna(subset=['corp_name'])
    ## 디버깅
    # print(f"기업명 매핑 완료: {len(metrics_df)}개 행")
    return metrics_df


def fetch_multi_company_by_target_quarters(corp_codes, target_quarters, valid_companies):
    """
    타겟 기업 분기 기준으로 다중회사 재무지표 수집
    입력: 기업코드 리스트, 대상 분기 리스트, 유효 기업 정보
    출력: 기업명이 매핑된 재무지표 DF
    -> 다중회사 재무지표 API로 모든 지표 카테고리 데이터 수집
    """
    corp_codes_str = ",".join(corp_codes)
    all_metrics = []
        
    for year, reprt_code in target_quarters:
        ## 디버깅
        # print(f"수집 중: {year}년 {REPORT_CODES[reprt_code]}")
        for idx_cl_code, category_name in MULTI_METRIC_CODES.items():
            ## DART 다중회사 주요 재무지표 불러오는 API url (참고 개발가이드: https://opendart.fss.or.kr/guide/detail.do?apiGrpCd=DS003&apiId=2022002)
            url = "https://opendart.fss.or.kr/api/fnlttCmpnyIndx.xml"
            params = {'crtfc_key': DART_API_KEY,
                      'corp_code': corp_codes_str,
                      'bsns_year': str(year),
                      'reprt_code': reprt_code,
                      'idx_cl_code': idx_cl_code}
            try:
                response = requests.get(url, params=params)
                if response.status_code == 200:
                    root = ET.fromstring(response.content)
                    status = root.findtext("status")
                    if status == '000':
                        companies_found = set()
                        for item in root.findall("list"):
                            corp_name = item.findtext('corp_name')
                            corp_code = item.findtext('corp_code')
                            companies_found.add(corp_code)  # corp_code로 추적
                            metric_data = {'corp_name': corp_name,  
                                           'corp_code': corp_code,
                                           'bsns_year': item.findtext('bsns_year'),
                                           'stock_code': item.findtext('stock_code'),
                                           'report_name': REPORT_CODES.get(item.findtext('reprt_code'), ''),
                                           'category': category_name,
                                           'idx_nm': item.findtext('idx_nm'),
                                           'idx_val': item.findtext('idx_val')}
                            all_metrics.append(metric_data)
                        ## 디버깅
                        print(f"{category_name}: {len(companies_found)}개 기업 데이터 수집")
                    else:
                        print(f"[Warning] {year}년 {REPORT_CODES[reprt_code]} {category_name}: status {status}")
                sleep(0.1)
            except Exception as e:
                print(f"[오류] API 호출 실패: {e}")
                continue
        sleep(0.2)
    # DF 생성 후 기업명 매핑
    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        metrics_df = fix_corp_names_in_metrics(metrics_df, valid_companies)
    return metrics_df


def create_industry_average_by_category(metrics_df):
    """
    수집된 재무지표를 카테고리별 평균으로 변환 (load_finance_b 활용)
    입력: 다중회사 재무지표 DF
    출력: 지표분류별 평균 DF 딕셔너리
    """
    if metrics_df.empty:
        return {}
    # 숫자로 변환
    metrics_df['idx_val_numeric'] = pd.to_numeric(metrics_df['idx_val'], errors='coerce')
    # 카테고리별로 분리하여 처리
    result = {}
    categories = metrics_df['category'].unique()
    for category in categories:
        category_data = metrics_df[metrics_df['category'] == category].copy()
        if category_data.empty:
            continue
        # 분기별 평균 계산
        quarters_data = []
        quarters = category_data.groupby(['bsns_year', 'report_name']).size().index.tolist()
        quarters = sorted(quarters, key=lambda x: (x[0], x[1]))
        for year, report_name in quarters:
            quarter_data = category_data[(category_data['bsns_year'] == year) & 
                                         (category_data['report_name'] == report_name)]
            if quarter_data.empty:
                continue
            # 해당 분기의 지표별 평균 계산
            quarter_avg = quarter_data.groupby('idx_nm')['idx_val_numeric'].mean()
            # 행 데이터 생성 (load_finance_b와 동일한 형태)
            quarter_row = {'연도': int(year),
                           '보고서': report_name,
                           '지표분류': category}
            # 각 지표를 개별 컬럼으로 추가
            for idx_name, avg_value in quarter_avg.items():
                if not pd.isna(avg_value):
                    quarter_row[idx_name] = float(avg_value)
            quarters_data.append(quarter_row)
        
        # 카테고리별 DF 생성
        if quarters_data:
            category_df = pd.DataFrame(quarters_data)
            result[category] = category_df
            ## 디버깅
            print(f"{category}: {len(category_df)}개 분기 데이터")
    return result


def get_industry_companies(target_corp_name, file_path, max_companies=10):
    """
    전체 프로세스 통합 동종업계 기업 검색 
    입력: 타겟 기업명, KRX CSV 파일 경로, 최대 기업 수
    출력: 동종업계 기업 정보 리스트
    -> IndustryAnalyzer를 통한 타겟 기업 정보 수집
       여유분 포함하여 max_companies * 2개 수집 후 필터링용 데이터 제공
    """
    analyzer = IndustryAnalyzer(file_path)
    target_info = analyzer.get_target_company_info(target_corp_name)
    if not target_info:
        return []
    # 여유분 포함해서 더 많이 수집 (max_companies * 2)
    similar_companies = analyzer.find_similar_companies(max_companies * 2)

    result = []
    for company in similar_companies:
        result.append({'corp_name': company['corp_name'],
                       'corp_code': company['corp_code']})
    ## 디버깅
    print(f"\n최종 결과: {len(result)}개 동종업계 기업 발견")    
    return result


def collect_industry_stock_data(valid_companies, available_quarters):
    """
    동종업계 기업들의 주식 데이터 수집 및 통합
    입력: 기업 리스트, 사용 가능한 분기 리스트
    출력: 모든 기업의 주식 데이터가 통합된 DF
    -> 각 기업별로 StockInfo를 통한 주식 데이터 수집
    """    
    all_stock_data = []
    for company in valid_companies:
        try:
            corp_name = company['corp_name']
            corp_code = company['corp_code']
            # CorpInfo로 주식코드 확보
            corp_info = CorpInfo(corp_name)
            stock_code = corp_info.corp_info.iloc[0]['stock_code']
            if pd.isna(stock_code) or not stock_code:
                print(f"{corp_name}: 주식코드가 존재하지 않습니다.")
                continue
            # 분기 정보를 날짜로 변환
            quarters_for_stock = []
            for year, reprt_code in available_quarters:
                quarter_map = {'11011': 'Q4', '11012': 'Q2', '11013': 'Q1', '11014': 'Q3'}
                quarter = quarter_map.get(reprt_code, 'Q1')
                quarters_for_stock.append(f"{year}_{quarter}")
            
            # 임시 DF로 날짜 생성
            temp_df = pd.DataFrame(index=quarters_for_stock)
            quarterly_dates = generate_dates_from_financial_df(temp_df)
            
            # 주식 데이터 수집
            stock_info = StockInfo(stock_code, quarterly_dates)
            stock_df = stock_info._data
            if not stock_df.empty:
                # 주식 데이터를 재무지표와 동일한 형태로 변환
                stock_formatted = stock_df.reset_index()
                stock_formatted['corp_name'] = corp_name
                stock_formatted['corp_code'] = corp_code
                stock_formatted['연도'] = stock_formatted['orig_date'].str[:4].astype(int)
                stock_formatted['지표분류'] = '주식지표'

                def date_to_report(date):
                    month = date[4:6]
                    if month == '03': return '1분기보고서'
                    elif month == '06': return '반기보고서'
                    elif month == '09': return '3분기보고서'
                    else: return '사업보고서'
                
                stock_formatted['보고서'] = stock_formatted['orig_date'].apply(date_to_report)
                stock_formatted = stock_formatted.drop('orig_date', axis=1)
                all_stock_data.append(stock_formatted)
                ## 디버깅
                print(f"{corp_name}: 주식 데이터 수집 완료 ({len(stock_formatted)}개 분기)")
            else:
                ## 디버깅
                print(f"{corp_name}: 주식 데이터 수집 실패")
                
        except Exception as e:
            print(f"{corp_name}: 주식 데이터 오류 - {e}")
            continue
    
    if all_stock_data:
        combined_stock_df = pd.concat(all_stock_data, ignore_index=True)
        print(f"동종업계 주식 데이터 통합 완료: {len(combined_stock_df)}개 행")
        return combined_stock_df
    else:
        print("[Warning] 수집된 주식 데이터가 없습니다.")
        return pd.DataFrame()



def calculate_stock_averages(stock_df):
    """
    주식 데이터의 분기별 평균 계산
    입력: 통합된 주식 데이터 DF
    출력: 분기별 주식지표 평균 DF
    -> 시가총액, 종가, PER, PBR, EPS 등 주식 관련 지표의 평균 계산
       분기별로 그룹화하여 평균값 산출
       load_finance_b와 동일한 형태로 결과 DataFrame 구성
    """
    if stock_df.empty:
        return pd.DataFrame()    
    # 주식 관련 컬럼들
    stock_columns = ['market_cap', 'close_price', 'PER', 'PBR', 'EPS']
    # 분기별 평균 계산
    quarters_data = []
    quarters = stock_df.groupby(['연도', '보고서']).size().index.tolist()
    quarters = sorted(quarters, key=lambda x: (x[0], x[1]))
    
    for year, report_name in quarters:
        quarter_data = stock_df[(stock_df['연도'] == year) & (stock_df['보고서'] == report_name)]
        if quarter_data.empty:
            continue
        quarter_row = {'연도': int(year),
                       '보고서': report_name,
                       '지표분류': '주식지표'}
        # 각 주식 지표의 평균 계산
        for col in stock_columns:
            if col in quarter_data.columns:
                avg_value = quarter_data[col].mean()
                if not pd.isna(avg_value):
                    quarter_row[col] = float(avg_value)
        quarters_data.append(quarter_row)
    if quarters_data:
        stock_avg_df = pd.DataFrame(quarters_data)
        print(f"주식지표 평균 계산 완료: {len(stock_avg_df)}개 분기")
        return stock_avg_df
    else:
        return pd.DataFrame()



def analyze_industry_benchmark(target_corp_name, file_path, max_companies=5, n_years=1, latest_only=False):
    """
    동종업계 DF 생성 (메인 함수)
    입력: 타겟 기업명, CSV 경로, 최대 기업수, 분석 연수, 최신만 여부
    출력: (유효 기업 리스트, 지표별 평균 DF 딕셔너리)
    -> 1. 동종업계 기업 검색, 2. 유효성 검증, 3. 공통 분기 확인
       4. 재무지표 수집 및 평균 계산, 5. 주식지표 수집 및 평균 계산
    """
    
    # 동종업계 기업 찾기 (여유분 포함)
    industry_companies = get_industry_companies(target_corp_name, file_path, max_companies)
    if not industry_companies:
        return [], {}
    
    # 우선주 제외하고 유효한 corp_code 확보 (목표 개수까지)
    valid_companies = validate_and_filter_companies(industry_companies, max_companies)
    if len(valid_companies) < max_companies:
        print(f"[Warning] 목표 {max_companies}개 중 {len(valid_companies)}개만 확보됨")
    if not valid_companies:
        print("[Warning] 유효한 corp_code를 가진 기업이 없습니다.")
        return industry_companies, {}
    
    # 다중회사 API로 공통 분기 확인
    corp_codes = [company['corp_code'] for company in valid_companies]
    if latest_only:
        available_quarters = get_available_quarters_for_multi_companies(corp_codes, n_years=1)[:1]
    else:
        available_quarters = get_available_quarters_for_multi_companies(corp_codes, n_years)
    if not available_quarters:
        print("[Warning] 공통으로 사용 가능한 분기가 없습니다.")
        return valid_companies, {}
    
    # 다중회사 API로 재무지표 수집-
    metrics_df = fetch_multi_company_by_target_quarters(corp_codes, available_quarters, valid_companies)
    if metrics_df.empty:
        print("[Warning] 재무지표 데이터를 수집할 수 없습니다.")
        return valid_companies, {}
    
    # 재무지표 평균 생성
    industry_avg_by_category = create_industry_average_by_category(metrics_df)
    
    # 주식지표 수집 및 평균 계산
    stock_df = collect_industry_stock_data(valid_companies, available_quarters)
    
    if not stock_df.empty:
        stock_avg_df = calculate_stock_averages(stock_df)
        if not stock_avg_df.empty:
            # 주식지표를 기존 결과에 추가
            industry_avg_by_category['주식지표'] = stock_avg_df
            ## 디버깅
            # print("주식지표가 동종업계 평균에 추가되었습니다.")
    
    return valid_companies, industry_avg_by_category



if __name__ == "__main__":
    # 사용 예시
    file_path = "../업종분류현황_250809.csv"
    
    companies, avg_data = analyze_industry_benchmark(
        target_corp_name="삼성전자",
        file_path=file_path,
        max_companies=5,
        n_years=1
    )
    
    # 전체 통합 DataFrame만 출력
    if avg_data:
        all_data = []
        for category, df in avg_data.items():
            if not df.empty:
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            print("=== 동종업계 평균 지표 전체 DataFrame ===")
            print(combined_df)
    else:
        print("데이터가 없습니다.")
