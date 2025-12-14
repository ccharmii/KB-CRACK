# /financial_analysis/load_corpinfo.py
# DART 기반 기업 기본 정보, 고유번호를 조회 -> 데이터프레임 및 JSON으로 제공


import os

import dart_fss
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")


class CorpInfo:
    """기업명 기반 기업코드 및 기업정보 조회 기능 제공 클래스"""

    def __init__(self, corp_name: str):
        """
        기업명 기반 기업코드 및 기업정보 초기화 수행
        Args:
            corp_name: 기업명 문자열
        """
        self.api_key = DART_API_KEY
        self.corp_name = corp_name
        self.corp_code = self.get_corpcode()
        self.corp_info = self.get_corpinfo()
        self.corp_info_json = self.get_corpinfo_json()

    def get_corpcode(self) -> str:
        """
        기업명 기반 공시대상회사 고유번호 조회 수행
        Returns:
            기업 고유번호 문자열 반환
        """
        corp_lst = dart_fss.corp.get_corp_list()
        corp = corp_lst.find_by_corp_name(self.corp_name, exactly=True)[0]
        self.corp = corp
        return corp.corp_code

    def get_sector_product(self) -> tuple[str, str]:
        """
        기업 업종 및 제품군 정보 조회 수행
        Returns:
            (업종, 제품군) 튜플 반환
        """
        try:
            sector = getattr(self.corp, "sector", "")
            product = getattr(self.corp, "product", "")
            return sector, product
        except Exception:
            return "", ""

    def get_corpinfo(self) -> pd.DataFrame:
        """
        DART 기업 기본 정보 조회 및 데이터프레임 변환 수행
        Returns:
            기업 정보 데이터프레임 반환
        """
        url = "https://opendart.fss.or.kr/api/company.xml"
        params = {"crtfc_key": self.api_key, "corp_code": self.corp_code}

        response = requests.get(url, params=params, timeout=30)
        soup = BeautifulSoup(response.text, "xml")

        fields = [
            "corp_name",
            "corp_name_eng",
            "stock_code",
            "ceo_nm",
            "corp_cls",
            "jurir_no",
            "bizr_no",
            "adres",
            "induty_code",
            "est_dt",
            "acc_mt",
        ]

        data = {field: (soup.find(field).text if soup.find(field) else "") for field in fields}

        sector, product = self.get_sector_product()
        data["sector"] = sector
        data["product"] = product

        return pd.DataFrame([data])

    def get_corpinfo_json(self) -> dict:
        """
        기업 정보 데이터프레임을 한국어 키 기반 JSON으로 변환 수행
        Returns:
            한국어 키 매핑 기업 정보 딕셔너리 반환
        """
        fields = {
            "corp_name": "기업명",
            "corp_name_eng": "영문기업명",
            "stock_code": "종목코드",
            "ceo_nm": "대표자명",
            "corp_cls": "법인구분",
            "jurir_no": "법인등록번호",
            "bizr_no": "사업자등록번호",
            "adres": "주소",
            "induty_code": "업종코드",
            "est_dt": "설립일",
            "acc_mt": "결산월",
            "sector": "업종",
            "product": "제품군",
        }

        raw_json = self.corp_info.to_dict(orient="records")[0]
        translated_json = {fields.get(k, k): v for k, v in raw_json.items()}
        return translated_json


def main() -> None:
    corp = CorpInfo("삼성전자")
    _ = corp.corp_code
    _ = corp.corp_info
    _ = corp.get_corpinfo_json()


if __name__ == "__main__":
    main()
