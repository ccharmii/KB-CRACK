# /financial_analysis/load_finance_b.py
# DART 주요재무지표 API와 KRX 주식 데이터를 수집, 지표분류별 데이터프레임 제공


import os
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
from time import sleep

import pandas as pd
import requests
from dotenv import load_dotenv
from pykrx import stock

from .load_corpinfo import CorpInfo

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")

REPORT_CODES = {"11011": "사업보고서", "11012": "반기보고서", "11013": "1분기보고서", "11014": "3분기보고서"}
METRIC_CODES = {"M210000": "수익성지표", "M220000": "안정성지표", "M230000": "성장성지표", "M240000": "활동성지표"}


def generate_dates_from_financial_df(financial_df: pd.DataFrame) -> list[str]:
    """
    분기 인덱스 기반 주식 데이터 조회용 날짜 목록 생성 수행
    Args:
        financial_df: year_quarter 형태 인덱스를 가진 데이터프레임
    Returns:
        분기별 기준일 문자열 리스트 반환
    """
    dates: list[str] = []
    quarter_dates_tpl = {"Q1": "0331", "Q2": "0630", "Q3": "0930", "Q4": "1231"}

    for quarter in financial_df.index:
        year, q = str(quarter).split("_")
        dates.append(f"{year}{quarter_dates_tpl[q]}")

    return dates


class StockInfo:
    """분기 기준일 목록 기반 주식 시장 데이터를 수집하는 클래스"""
    def __init__(self, stock_code: str, dates: list[str], market: str = "ALL"):
        """
        종목코드와 날짜 목록 기반 주식 데이터 수집 설정 수행
        Args:
            stock_code: 종목코드 문자열
            dates: 기준일 문자열 리스트
            market: 조회 시장 구분 문자열
        """
        self.stock_code = stock_code
        self.dates = dates
        self.market = market
        self._data = self._load_data()

    def _get_trading_day(self, date_str: str) -> str:
        """
        휴일 보정 기반 최근 거래일 탐색 수행
        Args:
            date_str: 기준일 문자열
        Returns:
            거래일 문자열 반환
        """
        dt = datetime.strptime(date_str, "%Y%m%d")
        for _ in range(10):
            try:
                datestr = dt.strftime("%Y%m%d")
                df = stock.get_market_ohlcv(datestr, market=self.market)
                if not df.empty and df["종가"].sum() != 0:
                    return datestr
            except Exception:
                pass
            dt -= timedelta(days=1)
        return date_str

    def _load_data(self) -> pd.DataFrame:
        """
        날짜별 시장 데이터 수집 및 통합 데이터프레임 생성 수행
        Returns:
            날짜 인덱스 기반 주식 지표 데이터프레임 반환
        """
        rows: list[dict[str, float | str]] = []

        for orig_date in self.dates:
            try:
                date = self._get_trading_day(orig_date)
                cap_df = stock.get_market_cap(date, market=self.market)
                ohlcv_df = stock.get_market_ohlcv(date, market=self.market)
                fund_df = stock.get_market_fundamental(date, market=self.market)

                if (
                    self.stock_code in cap_df.index
                    and self.stock_code in ohlcv_df.index
                    and self.stock_code in fund_df.index
                ):
                    rows.append(
                        {
                            "orig_date": orig_date,
                            "market_cap": float(cap_df.loc[self.stock_code, "시가총액"]),
                            "close_price": float(ohlcv_df.loc[self.stock_code, "종가"]),
                            "PER": float(fund_df.loc[self.stock_code, "PER"]),
                            "PBR": float(fund_df.loc[self.stock_code, "PBR"]),
                            "EPS": float(fund_df.loc[self.stock_code, "EPS"]),
                        }
                    )
            except Exception:
                continue

        df = pd.DataFrame(rows)
        return df.set_index("orig_date") if not df.empty else pd.DataFrame()


def api_call(corp_code: str, year: int, reprt_code: str, idx_cl_code: str) -> ET.Element | None:
    """
    DART 주요재무지표 API 호출 수행
    Args:
        corp_code: 기업 고유번호 문자열
        year: 사업연도 정수
        reprt_code: 보고서 코드 문자열
        idx_cl_code: 지표분류 코드 문자열
    Returns:
        성공 시 XML 루트 엘리먼트 반환
    """
    params = {
        "crtfc_key": DART_API_KEY,
        "corp_code": corp_code,
        "bsns_year": str(year),
        "reprt_code": reprt_code,
        "idx_cl_code": idx_cl_code,
    }

    response = requests.get("https://opendart.fss.or.kr/api/fnlttSinglIndx.xml", params=params, timeout=30)
    if response.status_code != 200:
        return None

    root = ET.fromstring(response.content)
    return root if root.findtext("status") == "000" else None


def get_quarter_candidates() -> list[tuple[int, str]]:
    """
    최신부터 과거까지 분기 후보 리스트 생성 수행
    Returns:
        (연도, 보고서코드) 튜플 리스트 반환
    """
    year = datetime.now().year
    candidates: list[tuple[int, str]] = []
    for y in range(year, 2021, -1):
        candidates.extend([(y, code) for code in ["11013", "11012", "11014", "11011"]])
    return candidates


def find_available_quarters(corp_code: str, n_years: int = 1) -> list[tuple[int, str]]:
    """
    데이터가 존재하는 최신 n년 분기 목록 탐색 수행
    Args:
        corp_code: 기업 고유번호 문자열
        n_years: 조회 연수 정수
    Returns:
        데이터 존재 분기 튜플 리스트 반환
    """
    quarters: list[tuple[int, str]] = []
    target_quarters = n_years * 4

    for year, reprt_code in get_quarter_candidates():
        if len(quarters) >= target_quarters:
            break

        if api_call(corp_code, year, reprt_code, "M210000"):
            quarters.append((year, reprt_code))

        sleep(0.2)

    return quarters


def get_metrics(corp_code: str, year: int, reprt_code: str, idx_cl_code: str) -> dict[str, float | str]:
    """
    분기별 지표분류 데이터 추출 수행
    Args:
        corp_code: 기업 고유번호 문자열
        year: 사업연도 정수
        reprt_code: 보고서 코드 문자열
        idx_cl_code: 지표분류 코드 문자열
    Returns:
        지표명 기준 값 딕셔너리 반환
    """
    root = api_call(corp_code, year, reprt_code, idx_cl_code)
    if root is None:
        return {}

    metrics: dict[str, float | str] = {}
    for item in root.findall("list"):
        name, value = item.findtext("idx_nm"), item.findtext("idx_val")
        if not name or value is None:
            continue
        try:
            metrics[name] = float(value)
        except Exception:
            metrics[name] = value

    return metrics


def fetch_latest_n_years_reports(corp_code: str, n_years: int = 1) -> dict[str, pd.DataFrame]:
    """
    최신 n년 주요재무지표를 지표분류별 데이터프레임으로 수집 수행
    Args:
        corp_code: 기업 고유번호 문자열
        n_years: 조회 연수 정수
    Returns:
        지표분류별 데이터프레임 딕셔너리 반환
    """
    quarters = find_available_quarters(corp_code, n_years)
    result: dict[str, list[dict]] = {category: [] for category in METRIC_CODES.values()}

    for year, reprt_code in quarters:
        for idx_cl_code, category in METRIC_CODES.items():
            metrics = get_metrics(corp_code, year, reprt_code, idx_cl_code)
            if metrics:
                result[category].append({"연도": year, "보고서": REPORT_CODES[reprt_code], **metrics})
            sleep(0.1)

    final_result: dict[str, pd.DataFrame] = {}
    for category, data_list in result.items():
        if not data_list:
            continue
        df = pd.DataFrame(data_list)
        if "지표분류" not in df.columns:
            df["지표분류"] = category
        final_result[category] = df

    return final_result


def integrate_stock_data(financial_data: dict[str, pd.DataFrame], stock_code: str) -> dict[str, pd.DataFrame]:
    """
    재무지표 데이터에 주식 데이터를 결합하여 주식지표 카테고리 추가 수행
    Args:
        financial_data: 지표분류별 데이터프레임 딕셔너리
        stock_code: 6자리 주식 종목코드 문자열
    Returns:
        주식지표가 추가된 지표분류별 데이터프레임 딕셔너리 반환
    """
    quarters: list[str] = []
    quarter_map = {"1분기보고서": "Q1", "반기보고서": "Q2", "3분기보고서": "Q3", "사업보고서": "Q4"}

    for df_cat in financial_data.values():
        if df_cat is None or df_cat.empty:
            continue
        for _, row in df_cat.iterrows():
            year, report = row.get("연도"), row.get("보고서")
            if report in quarter_map and pd.notna(year):
                quarters.append(f"{int(year)}_{quarter_map[report]}")

    quarters = sorted(list(set(quarters)))
    if not quarters:
        return financial_data

    try:
        temp_df = pd.DataFrame(index=quarters)
        quarterly_dates = generate_dates_from_financial_df(temp_df)

        stock_info = StockInfo(stock_code, quarterly_dates)
        stock_df = stock_info._data
        if stock_df.empty:
            return financial_data

        stock_formatted = stock_df.reset_index()
        stock_formatted["연도"] = stock_formatted["orig_date"].str[:4].astype(int)
        stock_formatted["지표분류"] = "주식지표"

        def date_to_report(date: str) -> str:
            """기준일 기반 보고서명 매핑 수행"""
            month = date[4:6]
            if month == "03":
                return "1분기보고서"
            if month == "06":
                return "반기보고서"
            if month == "09":
                return "3분기보고서"
            return "사업보고서"

        stock_formatted["보고서"] = stock_formatted["orig_date"].apply(date_to_report)
        stock_formatted = stock_formatted.drop(columns=["orig_date"])

        financial_data_with_stock = dict(financial_data)
        financial_data_with_stock["주식지표"] = stock_formatted
        return financial_data_with_stock

    except Exception:
        return financial_data


def main() -> None:
    """주요재무지표 수집 및 주식지표 결합 실행 예시"""
    corp = CorpInfo("삼성전자")
    corp_code = corp.corp_code
    stock_code = corp.corp_info.iloc[0].get("stock_code")

    data = fetch_latest_n_years_reports(corp_code, n_years=2)

    if pd.notna(stock_code):
        _ = integrate_stock_data(data, str(stock_code))


if __name__ == "__main__":
    main()
