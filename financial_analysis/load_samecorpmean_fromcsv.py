# /financial_analysis/load_samecorpmean_fromcsv.py
# KRX CSV 기반으로 동종업계 유사 기업을 선정하고 다중회사 DART 지표, 주식지표 평균 산출

# -*- coding: utf-8 -*-

import os
import xml.etree.ElementTree as ET
from datetime import datetime
from time import sleep

import pandas as pd
import requests
from dotenv import load_dotenv

from .load_corpinfo import CorpInfo
from .load_finance_a import fetch_latest_quarter_report, prepare_finreports, clean_columns
from .load_finance_b import StockInfo, generate_dates_from_financial_df

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")

REPORT_CODES = {"11011": "사업보고서", "11012": "반기보고서", "11013": "1분기보고서", "11014": "3분기보고서"}
MULTI_METRIC_CODES = {"M210000": "수익성지표", "M220000": "안정성지표", "M230000": "성장성지표", "M240000": "활동성지표"}


def load_krx_data(file_path: str) -> pd.DataFrame:
    """KRX CSV 로드 및 기본 정제 수행"""
    try:
        df = pd.read_csv(file_path, encoding="cp949")
        df.columns = df.columns.str.strip().str.replace(" ", "")
        if "종목코드" in df.columns:
            df["종목코드"] = df["종목코드"].astype(str).str.zfill(6)
        return df
    except Exception:
        return pd.DataFrame()


def classify_asset_size(total_assets_billion: float | None) -> str:
    """
    자산총액 기준 기업 규모 분류 수행
    Args:
        total_assets_billion: 자산총액 억원 단위 값
    Returns:
        기업 규모 분류 문자열 반환
    """
    if pd.isna(total_assets_billion) or total_assets_billion is None or total_assets_billion <= 0:
        return "미분류"
    if 0 < total_assets_billion < 5000:
        return "중소기업"
    if 5000 <= total_assets_billion < 50000:
        return "중견기업"
    if 50000 <= total_assets_billion:
        return "대기업"
    return "미분류"


def classify_company_age(est_date) -> str:
    """
    설립일 기반 업력 단계 분류 수행
    Args:
        est_date: YYYYMMDD 문자열 또는 datetime 객체
    Returns:
        업력 단계 분류 문자열 반환
    """
    try:
        if pd.isna(est_date) or not est_date:
            return "미분류"
        if isinstance(est_date, str):
            est_date = datetime.strptime(est_date, "%Y%m%d")
        years = (datetime.now() - est_date).days / 365
        if years <= 3:
            return "도입기"
        if years <= 7:
            return "도약기"
        if years <= 15:
            return "성장기"
        return "성장후기·성숙기"
    except Exception:
        return "미분류"


def get_company_financial_info(corp_code: str) -> float | None:
    """
    최신 분기 재무제표 기반 자산총계 추출 수행
    Args:
        corp_code: 기업 DART 고유번호 문자열
    Returns:
        자산총계 억원 단위 값 반환
    """
    try:
        financial_df = fetch_latest_quarter_report(DART_API_KEY, corp_code)
        if financial_df.empty:
            return None
        financial_df = prepare_finreports(financial_df)
        financial_df = clean_columns(financial_df)
        if financial_df.empty or "자산총계" not in financial_df.columns:
            return None
        latest_quarter = financial_df.index[-1]
        total_assets = pd.to_numeric(financial_df.loc[latest_quarter, "자산총계"], errors="coerce")
        if not pd.isna(total_assets) and total_assets > 0:
            return float(total_assets) / 100000000
        return None
    except Exception:
        return None


def validate_and_filter_companies(industry_companies: list[dict], target_count: int = 5) -> list[dict]:
    """
    우선주 등 제외 후 유효 기업 리스트 확보 수행
    Args:
        industry_companies: 후보 기업 정보 리스트
        target_count: 목표 기업 수
    Returns:
        corp_code가 확보된 기업 리스트 반환
    """
    valid_companies: list[dict] = []
    for company in industry_companies:
        corp_name = company.get("corp_name", "")
        if any(keyword in corp_name for keyword in ["우", "우선주", "리츠"]):
            continue
        try:
            corp_info_obj = CorpInfo(corp_name)
            valid_companies.append({"corp_name": corp_name, "corp_code": corp_info_obj.corp_code})
        except Exception:
            continue
        if len(valid_companies) >= target_count:
            break
    return valid_companies[:target_count]


class IndustryAnalyzer:
    """KRX CSV 기반 타겟 기업과 유사 기업을 추출하는 클래스"""

    def __init__(self, file_path: str):
        """
        CSV 로드 기반 분석기 초기화 수행
        Args:
            file_path: KRX CSV 파일 경로
        """
        self.krx_data = load_krx_data(file_path)
        self.target_info: dict | None = None

    def get_target_company_info(self, target_corp_name: str) -> dict | None:
        """
        타겟 기업 상세 정보 수집 및 분류 정보 생성 수행
        Args:
            target_corp_name: 타겟 기업명
        Returns:
            타겟 기업 정보 딕셔너리 반환
        """
        try:
            corp_info_obj = CorpInfo(target_corp_name)
            target_corp_info = corp_info_obj.corp_info.iloc[0].to_dict()
            target_corp_code = corp_info_obj.corp_code

            if self.krx_data.empty or "종목명" not in self.krx_data.columns:
                return None

            krx_match = self.krx_data[self.krx_data["종목명"] == target_corp_name]
            if krx_match.empty:
                return None

            match_row = krx_match.iloc[0]
            target_assets = get_company_financial_info(target_corp_code)

            self.target_info = {
                "corp_name": target_corp_name,
                "corp_code": target_corp_code,
                "industry": match_row.get("업종명"),
                "market": match_row.get("시장구분"),
                "market_cap": match_row.get("시가총액"),
                "asset_size": classify_asset_size(target_assets),
                "company_age": classify_company_age(target_corp_info.get("est_dt")),
                "total_assets": target_assets,
            }
            return self.target_info
        except Exception:
            return None

    def find_similar_companies(self, max_companies: int = 15) -> list[dict]:
        """
        업종 및 시장구분 기반 유사 기업 후보 추출 수행
        Args:
            max_companies: 최대 반환 기업 수
        Returns:
            시가총액 차이 기준 정렬된 기업 리스트 반환
        """
        if not self.target_info or self.krx_data.empty:
            return []

        industry_filtered = self.krx_data[self.krx_data["업종명"] == self.target_info["industry"]].copy()
        if industry_filtered.empty:
            return []

        market_filtered = industry_filtered[industry_filtered["시장구분"] == self.target_info["market"]].copy()
        if market_filtered.empty:
            market_filtered = industry_filtered

        market_filtered = market_filtered[market_filtered["종목명"] != self.target_info["corp_name"]].copy()

        similar_companies: list[dict] = []
        target_market_cap = self.target_info.get("market_cap")

        for _, row in market_filtered.iterrows():
            try:
                market_cap = row.get("시가총액")
                market_cap_diff = abs(target_market_cap - market_cap)
                similar_companies.append(
                    {
                        "corp_name": row.get("종목명"),
                        "corp_code": row.get("종목코드"),
                        "market_cap": market_cap,
                        "market_cap_diff": market_cap_diff,
                    }
                )
            except Exception:
                continue

        similar_companies.sort(key=lambda x: x.get("market_cap_diff", float("inf")))
        return similar_companies[:max_companies]


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


def get_available_quarters_for_multi_companies(corp_codes: list[str], n_years: int = 1) -> list[tuple[int, str]]:
    """
    다중회사 API 기반 공통 분기 목록 탐색 수행
    Args:
        corp_codes: 기업코드 리스트
        n_years: 조회 연수
    Returns:
        공통 분기 튜플 리스트 반환
    """
    corp_codes_str = ",".join(corp_codes)
    available_quarters: list[tuple[int, str]] = []
    target_quarters = n_years * 4

    for year, reprt_code in get_quarter_candidates():
        if len(available_quarters) >= target_quarters:
            break

        url = "https://opendart.fss.or.kr/api/fnlttCmpnyIndx.xml"
        params = {
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_codes_str,
            "bsns_year": str(year),
            "reprt_code": reprt_code,
            "idx_cl_code": "M210000",
        }

        try:
            response = requests.get(url, params=params, timeout=30)
            if response.status_code != 200:
                sleep(0.3)
                continue

            root = ET.fromstring(response.content)
            if root.findtext("status") != "000":
                sleep(0.3)
                continue

            companies_with_data = {item.findtext("corp_code") for item in root.findall("list")}
            if len(companies_with_data) >= min(3, len(corp_codes) // 2):
                available_quarters.append((year, reprt_code))
        except Exception:
            pass

        sleep(0.3)

    return available_quarters


def fix_corp_names_in_metrics(metrics_df: pd.DataFrame, valid_companies: list[dict]) -> pd.DataFrame:
    """
    corp_code 기반 기업명 매핑 정합화 수행
    Args:
        metrics_df: 다중회사 지표 원천 데이터프레임
        valid_companies: 유효 기업 리스트
    Returns:
        기업명 컬럼이 보정된 데이터프레임 반환
    """
    if metrics_df.empty:
        return metrics_df

    corp_code_to_name = {comp["corp_code"]: comp["corp_name"] for comp in valid_companies}
    metrics_df["corp_name"] = metrics_df["corp_code"].map(corp_code_to_name)
    metrics_df = metrics_df.dropna(subset=["corp_name"])
    return metrics_df


def fetch_multi_company_by_target_quarters(
    corp_codes: list[str], target_quarters: list[tuple[int, str]], valid_companies: list[dict]
) -> pd.DataFrame:
    """
    지정 분기 목록 기준 다중회사 재무지표 수집 수행
    Args:
        corp_codes: 기업코드 리스트
        target_quarters: 대상 분기 리스트
        valid_companies: 유효 기업 리스트
    Returns:
        기업명이 매핑된 재무지표 데이터프레임 반환
    """
    corp_codes_str = ",".join(corp_codes)
    all_metrics: list[dict] = []

    for year, reprt_code in target_quarters:
        for idx_cl_code, category_name in MULTI_METRIC_CODES.items():
            url = "https://opendart.fss.or.kr/api/fnlttCmpnyIndx.xml"
            params = {
                "crtfc_key": DART_API_KEY,
                "corp_code": corp_codes_str,
                "bsns_year": str(year),
                "reprt_code": reprt_code,
                "idx_cl_code": idx_cl_code,
            }

            try:
                response = requests.get(url, params=params, timeout=30)
                if response.status_code != 200:
                    sleep(0.1)
                    continue

                root = ET.fromstring(response.content)
                if root.findtext("status") != "000":
                    sleep(0.1)
                    continue

                for item in root.findall("list"):
                    all_metrics.append(
                        {
                            "corp_name": item.findtext("corp_name"),
                            "corp_code": item.findtext("corp_code"),
                            "bsns_year": item.findtext("bsns_year"),
                            "stock_code": item.findtext("stock_code"),
                            "report_name": REPORT_CODES.get(item.findtext("reprt_code"), ""),
                            "category": category_name,
                            "idx_nm": item.findtext("idx_nm"),
                            "idx_val": item.findtext("idx_val"),
                        }
                    )
                sleep(0.1)
            except Exception:
                continue

        sleep(0.2)

    metrics_df = pd.DataFrame(all_metrics)
    if not metrics_df.empty:
        metrics_df = fix_corp_names_in_metrics(metrics_df, valid_companies)

    return metrics_df


def create_industry_average_by_category(metrics_df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    지표분류별 분기 평균 데이터프레임 생성 수행
    Args:
        metrics_df: 다중회사 재무지표 원천 데이터프레임
    Returns:
        지표분류별 평균 데이터프레임 딕셔너리 반환
    """
    if metrics_df.empty:
        return {}

    metrics_df = metrics_df.copy()
    metrics_df["idx_val_numeric"] = pd.to_numeric(metrics_df["idx_val"], errors="coerce")

    result: dict[str, pd.DataFrame] = {}
    for category in metrics_df["category"].dropna().unique():
        category_data = metrics_df[metrics_df["category"] == category].copy()
        if category_data.empty:
            continue

        quarters_index = category_data.groupby(["bsns_year", "report_name"]).size().index.tolist()
        quarters_index = sorted(quarters_index, key=lambda x: (x[0], x[1]))

        quarters_rows: list[dict] = []
        for year, report_name in quarters_index:
            quarter_data = category_data[(category_data["bsns_year"] == year) & (category_data["report_name"] == report_name)]
            if quarter_data.empty:
                continue

            quarter_avg = quarter_data.groupby("idx_nm")["idx_val_numeric"].mean()
            quarter_row = {"연도": int(year), "보고서": report_name, "지표분류": category}

            for idx_name, avg_value in quarter_avg.items():
                if not pd.isna(avg_value):
                    quarter_row[idx_name] = float(avg_value)

            quarters_rows.append(quarter_row)

        if quarters_rows:
            result[category] = pd.DataFrame(quarters_rows)

    return result


def get_industry_companies(target_corp_name: str, file_path: str, max_companies: int = 10) -> list[dict]:
    """
    타겟 기업 기준 동종업계 유사 기업 후보 목록 생성 수행
    Args:
        target_corp_name: 타겟 기업명
        file_path: KRX CSV 파일 경로
        max_companies: 최대 기업 수
    Returns:
        기업명 및 종목코드 리스트 반환
    """
    analyzer = IndustryAnalyzer(file_path)
    target_info = analyzer.get_target_company_info(target_corp_name)
    if not target_info:
        return []

    similar_companies = analyzer.find_similar_companies(max_companies * 2)
    return [{"corp_name": c.get("corp_name"), "corp_code": c.get("corp_code")} for c in similar_companies if c.get("corp_name")]


def collect_industry_stock_data(valid_companies: list[dict], available_quarters: list[tuple[int, str]]) -> pd.DataFrame:
    """
    유효 기업 목록 기반 주식 데이터 수집 및 통합 수행
    Args:
        valid_companies: 유효 기업 리스트
        available_quarters: 공통 분기 리스트
    Returns:
        기업별 주식 데이터 통합 데이터프레임 반환
    """
    all_stock_data: list[pd.DataFrame] = []
    quarter_map = {"11011": "Q4", "11012": "Q2", "11013": "Q1", "11014": "Q3"}

    for company in valid_companies:
        try:
            corp_name = company["corp_name"]
            corp_code = company["corp_code"]

            corp_info = CorpInfo(corp_name)
            stock_code = corp_info.corp_info.iloc[0].get("stock_code")
            if pd.isna(stock_code) or not stock_code:
                continue

            quarters_for_stock = [f"{year}_{quarter_map.get(reprt_code, 'Q1')}" for year, reprt_code in available_quarters]
            temp_df = pd.DataFrame(index=quarters_for_stock)
            quarterly_dates = generate_dates_from_financial_df(temp_df)

            stock_info = StockInfo(str(stock_code), quarterly_dates)
            stock_df = stock_info._data
            if stock_df.empty:
                continue

            stock_formatted = stock_df.reset_index()
            stock_formatted["corp_name"] = corp_name
            stock_formatted["corp_code"] = corp_code
            stock_formatted["연도"] = stock_formatted["orig_date"].str[:4].astype(int)
            stock_formatted["지표분류"] = "주식지표"

            def date_to_report(date: str) -> str:
                """기준일 기반 보고서명 변환 수행"""
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

            all_stock_data.append(stock_formatted)
        except Exception:
            continue

    if not all_stock_data:
        return pd.DataFrame()

    return pd.concat(all_stock_data, ignore_index=True)


def calculate_stock_averages(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    주식 지표 분기별 평균 데이터프레임 생성 수행
    Args:
        stock_df: 기업별 주식 데이터 통합 데이터프레임
    Returns:
        분기별 주식지표 평균 데이터프레임 반환
    """
    if stock_df.empty:
        return pd.DataFrame()

    stock_columns = ["market_cap", "close_price", "PER", "PBR", "EPS"]

    quarters_index = stock_df.groupby(["연도", "보고서"]).size().index.tolist()
    quarters_index = sorted(quarters_index, key=lambda x: (x[0], x[1]))

    rows: list[dict] = []
    for year, report_name in quarters_index:
        quarter_data = stock_df[(stock_df["연도"] == year) & (stock_df["보고서"] == report_name)]
        if quarter_data.empty:
            continue

        row = {"연도": int(year), "보고서": report_name, "지표분류": "주식지표"}
        for col in stock_columns:
            if col in quarter_data.columns:
                avg_value = quarter_data[col].mean()
                if not pd.isna(avg_value):
                    row[col] = float(avg_value)
        rows.append(row)

    return pd.DataFrame(rows)


def analyze_industry_benchmark(
    target_corp_name: str,
    file_path: str,
    max_companies: int = 5,
    n_years: int = 1,
    latest_only: bool = False,
) -> tuple[list[dict], dict[str, pd.DataFrame]]:
    """
    동종업계 평균 지표 생성 파이프라인 실행 수행
    Args:
        target_corp_name: 타겟 기업명
        file_path: KRX CSV 파일 경로
        max_companies: 비교 기업 수
        n_years: 조회 연수
        latest_only: 최신 1개 분기만 사용 여부
    Returns:
        (유효 기업 리스트, 지표분류별 평균 데이터프레임 딕셔너리) 반환
    """
    industry_companies = get_industry_companies(target_corp_name, file_path, max_companies)
    if not industry_companies:
        return [], {}

    valid_companies = validate_and_filter_companies(industry_companies, max_companies)
    if not valid_companies:
        return industry_companies, {}

    corp_codes = [company["corp_code"] for company in valid_companies]
    available_quarters = get_available_quarters_for_multi_companies(corp_codes, n_years=1 if latest_only else n_years)
    if latest_only:
        available_quarters = available_quarters[:1]
    if not available_quarters:
        return valid_companies, {}

    metrics_df = fetch_multi_company_by_target_quarters(corp_codes, available_quarters, valid_companies)
    if metrics_df.empty:
        return valid_companies, {}

    industry_avg_by_category = create_industry_average_by_category(metrics_df)

    stock_df = collect_industry_stock_data(valid_companies, available_quarters)
    if not stock_df.empty:
        stock_avg_df = calculate_stock_averages(stock_df)
        if not stock_avg_df.empty:
            industry_avg_by_category["주식지표"] = stock_avg_df

    return valid_companies, industry_avg_by_category


def main() -> None:
    file_path = "../업종분류현황_250809.csv"
    _ = analyze_industry_benchmark(
        target_corp_name="삼성전자",
        file_path=file_path,
        max_companies=5,
        n_years=1,
    )


if __name__ == "__main__":
    main()
