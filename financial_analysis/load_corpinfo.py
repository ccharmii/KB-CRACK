## load_corpinfo.py


import requests
import pandas as pd
from bs4 import BeautifulSoup
import dart_fss

import os
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 로드

DART_API_KEY = os.getenv('DART_API_KEY')


class CorpInfo:
    def __init__(self, corp_name: str):
        '''
        기업명으로 객체를 생성해 corp_code나 기업에 대한 정보를 반환
        '''
        self.api_key = DART_API_KEY
        self.corp_name = corp_name
        self.corp_code = self.get_corpcode()
        self.corp_info = self.get_corpinfo()
        self.corp_info_json = self.get_corpinfo_json()

    
    def get_corpcode(self):
        """
        기업명으로 검색의 기준이 되는 기업 코드 반환
        입력: 기업명
        출력: 공시대상회사 고유번호 (8자리)	
        """
        corp_lst = dart_fss.corp.get_corp_list()
        corp = corp_lst.find_by_corp_name(self.corp_name, exactly=True)[0]
        self.corp = corp
        return corp.corp_code


    def get_sector_product(self):
        """
        기업명으로 해당 기업의 업종과 생산하는 제품군 정보를 반환
        입력: 기업명
        출력: 업종, 제품군 정보
        """
        try:
            # sector와 product 속성이 있는지 확인 후 반환
            sector = getattr(self.corp, 'sector', '')
            product = getattr(self.corp, 'product', '')
            return sector, product
        except Exception as e:
            # 에러 발생시 빈 문자열 반환 -> 만약 없을 경우를 대비 / 필수적으로 필요한 정보가 아님
            return '', ''  
        
          
    def get_corpinfo(self):
        """
        DART에서 제공하는 기업 정보 DF 반환
        입력: 기업명
        출력: 기업 정보 DataFrame
        """
        url = "https://opendart.fss.or.kr/api/company.xml"
        params = {"crtfc_key": self.api_key, "corp_code": self.corp_code}
        response = requests.get(url, params=params)
        soup = BeautifulSoup(response.text, 'xml')
        fields = ['corp_name',       # 회사명 (국문)
                'corp_name_eng',   # 회사명 (영문)
                'stock_code',      # 종목코드 (상장사인 경우)
                'ceo_nm',          # 대표자명
                'corp_cls',        # 법인구분 (Y: 유가, K: 코스닥, N: 비상장 등)
                'jurir_no',        # 법인등록번호
                'bizr_no',         # 사업자등록번호
                'adres',           # 주소
                'induty_code',     # 업종코드
                'est_dt',          # 설립일
                'acc_mt'           # 결산월
                ]
        # 데이터 추출
        data = {field: soup.find(field).text if soup.find(field) else '' for field in fields}
        # 업종과 제품군 정보 추가
        sector, product = self.get_sector_product()
        data['sector'] = sector
        data['product'] = product
        return pd.DataFrame([data])


    def get_corpinfo_json(self) -> dict:
        """
        기업 정보 DataFrame을 한국어로 매핑된 기업 정보 (JSON 형식) 로 변환
        입력: 기업 정보 DataFrame
        출력: 한국어로 매핑된 JSON 기업 정보
        """
        fields = {'corp_name': '기업명',
                  'corp_name_eng': '영문기업명',
                  'stock_code': '종목코드',
                  'ceo_nm': '대표자명',
                  'corp_cls': '법인구분',
                  'jurir_no': '법인등록번호',
                  'bizr_no': '사업자등록번호',
                  'adres': '주소',
                  'induty_code': '업종코드',
                  'est_dt': '설립일',
                  'acc_mt': '결산월', 
                  'sector': '업종',
                  'product': '제품군'}

        raw_json = self.corp_info.to_dict(orient='records')[0]
        translated_json = {fields.get(k, k): v for k, v in raw_json.items()}
        return translated_json
    

if __name__ == "__main__":
    corp = CorpInfo("삼성전자")
    corp_code = corp.corp_code
    corp_info = corp.corp_info
    corp_info_json = corp.get_corpinfo_json()