# /financial_analysis/load_finance_a.py
# DART 재무제표와 KRX 주식 데이터를 결합해 분기별 재무 지표를 계산


import json
import os
import re
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from pykrx import stock

from .load_corpinfo import CorpInfo

load_dotenv()

DART_API_KEY = os.getenv("DART_API_KEY")

REPORT_CODES = {"11011": "사업보고서", "11012": "반기보고서", "11013": "1분기보고서", "11014": "3분기보고서"}
FS_DIV = {"OFS": "재무제표", "CFS": "연결재무제표"}


def generate_dates_from_financial_df(financial_df: pd.DataFrame) -> list[str]:
    """
    분기 인덱스 기반 주식 데이터 조회용 날짜 목록 생성 수행
    Args:
        financial_df: year_quarter 형태 인덱스를 가진 재무 데이터프레임
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


def fetch_financial_statements(
    dart_api_key: str,
    corp_code: str,
    bsns_year: str,
    reprt_code: str,
    fs_div: str,
) -> list[dict[str, str]]:
    """
    DART OpenAPI 기반 단일회사 전체 재무제표 조회 수행
    Args:
        dart_api_key: DART 인증키 문자열
        corp_code: 기업 고유번호 문자열
        bsns_year: 사업연도 문자열
        reprt_code: 보고서 코드 문자열
        fs_div: 재무제표 구분 문자열
    Returns:
        재무제표 항목 리스트 반환
    """
    url = "https://opendart.fss.or.kr/api/fnlttSinglAcntAll.xml"
    params = {
        "crtfc_key": dart_api_key,
        "corp_code": corp_code,
        "bsns_year": bsns_year,
        "reprt_code": reprt_code,
        "fs_div": fs_div,
    }

    response = requests.get(url, params=params, timeout=30)
    if response.status_code != 200:
        return []

    root = ET.fromstring(response.content)
    status = root.findtext("status")
    if status != "000":
        return []

    results: list[dict[str, str]] = []
    for item in root.findall("list"):
        results.append(
            {
                "bsns_year": bsns_year,
                "report_type": REPORT_CODES.get(reprt_code, reprt_code),
                "account_nm": item.findtext("account_nm"),
                "fs_div": fs_div,
                "sj_div": item.findtext("sj_div"),
                "thstrm_amount": item.findtext("thstrm_amount"),
            }
        )

    return results


def fetch_latest_quarter_report(dart_api_key: str, corp_code: str, fs_div: str = "CFS") -> pd.DataFrame:
    """실제 존재하는 최신 분기 보고서 1개 조회 수행"""
    current_year = datetime.now().year
    current_month = datetime.now().month

    if current_month <= 3:
        expected_latest_quarter, expected_latest_year = "Q4", current_year - 1
    elif current_month <= 6:
        expected_latest_quarter, expected_latest_year = "Q1", current_year
    elif current_month <= 9:
        expected_latest_quarter, expected_latest_year = "Q2", current_year
    else:
        expected_latest_quarter, expected_latest_year = "Q3", current_year

    quarter_to_report_code = {"Q1": "11013", "Q2": "11012", "Q3": "11014", "Q4": "11011"}
    quarter_nums = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

    year = expected_latest_year
    current_q_num = quarter_nums[expected_latest_quarter]

    for _ in range(8):
        quarter = f"Q{current_q_num}"
        code = quarter_to_report_code[quarter]
        result = fetch_financial_statements(dart_api_key, corp_code, str(year), code, fs_div)
        if result:
            return pd.DataFrame(result)

        current_q_num -= 1
        if current_q_num == 0:
            current_q_num = 4
            year -= 1

    return pd.DataFrame()


def fetch_latest_n_years_reports(
    dart_api_key: str,
    corp_code: str,
    n_years: int = 2,
    fs_div: str = "CFS",
) -> pd.DataFrame:
    """실제 존재하는 최신 분기부터 n년치 분기별 재무제표 수집 수행"""
    current_year = datetime.now().year
    current_month = datetime.now().month

    if current_month <= 3:
        expected_latest_quarter, expected_latest_year = "Q4", current_year - 1
    elif current_month <= 6:
        expected_latest_quarter, expected_latest_year = "Q1", current_year
    elif current_month <= 9:
        expected_latest_quarter, expected_latest_year = "Q2", current_year
    else:
        expected_latest_quarter, expected_latest_year = "Q3", current_year

    quarter_to_report_code = {"Q1": "11013", "Q2": "11012", "Q3": "11014", "Q4": "11011"}
    quarter_nums = {"Q1": 1, "Q2": 2, "Q3": 3, "Q4": 4}

    def find_actual_latest_quarter() -> tuple[int, str]:
        year = expected_latest_year
        current_q_num = quarter_nums[expected_latest_quarter]

        for _ in range(8):
            quarter = f"Q{current_q_num}"
            code = quarter_to_report_code[quarter]
            result = fetch_financial_statements(dart_api_key, corp_code, str(year), code, fs_div)
            if result:
                return year, quarter

            current_q_num -= 1
            if current_q_num == 0:
                current_q_num = 4
                year -= 1

        return expected_latest_year, expected_latest_quarter

    actual_latest_year, actual_latest_quarter = find_actual_latest_quarter()

    quarters_to_collect: list[tuple[int, str]] = []
    year = actual_latest_year
    current_q_num = quarter_nums[actual_latest_quarter]

    for _ in range(n_years * 4 + 2):
        quarters_to_collect.append((year, f"Q{current_q_num}"))
        current_q_num -= 1
        if current_q_num == 0:
            current_q_num = 4
            year -= 1

    all_report: list[dict[str, str]] = []
    for y, quarter in quarters_to_collect:
        code = quarter_to_report_code[quarter]
        result = fetch_financial_statements(dart_api_key, corp_code, str(y), code, fs_div)
        if result:
            all_report.extend(result)

    return pd.DataFrame(all_report)


def prepare_finreports(all_reports: pd.DataFrame) -> pd.DataFrame:
    """
    원시 재무제표 데이터를 분기별 피벗 테이블로 변환 수행
    Args:
        all_reports: 원시 재무제표 데이터프레임
    Returns:
        분기 인덱스 기반 피벗 데이터프레임 반환
    """
    reports = all_reports[
        (all_reports["report_type"].isin(["사업보고서", "1분기보고서", "반기보고서", "3분기보고서"]))
        & (all_reports["fs_div"] == "CFS")
        & (all_reports["sj_div"].isin(["BS", "IS", "CF"]))
    ].copy()

    quarter_map = {"사업보고서": "Q4", "1분기보고서": "Q1", "반기보고서": "Q2", "3분기보고서": "Q3"}
    reports["quarter"] = reports["report_type"].map(quarter_map)
    reports["year_quarter"] = reports["bsns_year"] + "_" + reports["quarter"]

    return (
        reports.pivot_table(
            index="year_quarter",
            columns="account_nm",
            values="thstrm_amount",
            aggfunc="first",
        )
        .rename_axis(columns=None)
    )


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    컬럼명 정리 및 중복 컬럼 병합 수행
    Args:
        df: 원본 데이터프레임
    Returns:
        정리된 데이터프레임 반환
    """
    df_clean = df.copy()

    new_columns: dict[str, str] = {}
    for col in df_clean.columns:
        if pd.isna(col):
            continue
        clean_col = str(col)
        clean_col = re.sub(r"\([^)]*\)", "", clean_col)
        clean_col = re.sub(r"\s+", "", clean_col)
        new_columns[col] = clean_col

    df_clean = df_clean.rename(columns=new_columns)

    duplicate_cols = df_clean.columns[df_clean.columns.duplicated()].unique()
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


class MetricsCalculator:
    """분기별 재무 데이터 기반 주요 재무 지표 계산 클래스"""

    def __init__(self, financial_df: pd.DataFrame, stock_df: pd.DataFrame | None = None):
        """
        재무 및 주식 데이터 기반 지표 계산기 초기화 수행
        Args:
            financial_df: 분기 인덱스 기반 재무제표 데이터프레임
            stock_df: 기준일 인덱스 기반 주식 데이터프레임
        """
        self.full_financial_df = financial_df.copy()
        self.stock_df = stock_df.copy() if stock_df is not None else None

        all_quarters = sorted(financial_df.index.astype(str))
        self.analysis_quarters = all_quarters[1:] if len(all_quarters) > 1 else []

        self._prepare_data()
        self.combined_df = self._combine_data()

    def _prepare_data(self) -> None:
        """재무데이터 전처리 및 파생 지표 컬럼 생성 수행"""
        df = self.full_financial_df.copy()

        revenue_candidates = ["매출액", "매출", "수익총계", "영업수익"]
        if not any(col in df.columns for col in revenue_candidates):
            if all(col in df.columns for col in ["매출총이익", "매출원가"]):
                df["매출액"] = pd.to_numeric(df["매출총이익"], errors="coerce") + pd.to_numeric(df["매출원가"], errors="coerce")

        debt_components = ["단기차입금", "장기차입금", "사채", "차입금"]
        available_debt_components = [col for col in debt_components if col in df.columns]
        if available_debt_components:
            df["총차입금"] = df[available_debt_components].apply(pd.to_numeric, errors="coerce").sum(axis=1)

        capex_candidates = ["유형자산의취득", "유형자산취득", "설비투자", "자본적지출"]
        for candidate in capex_candidates:
            if candidate in df.columns:
                df["CAPEX"] = pd.to_numeric(df[candidate], errors="coerce")
                break

        self.full_financial_df = df

    def _combine_data(self) -> pd.DataFrame:
        """분기별 재무 데이터와 주식 데이터 결합 수행"""
        analysis_financial = self.full_financial_df.loc[self.full_financial_df.index.astype(str).isin(self.analysis_quarters)]

        if self.stock_df is None or self.stock_df.empty:
            return analysis_financial

        stock_quarter_data = []
        for quarter in self.analysis_quarters:
            year, quarter_num = quarter.split("_")
            quarter_dates = {"Q1": f"{year}0331", "Q2": f"{year}0630", "Q3": f"{year}0930", "Q4": f"{year}1231"}
            target_date = quarter_dates[quarter_num]
            if target_date in self.stock_df.index:
                stock_data = self.stock_df.loc[target_date].copy()
                stock_data.name = quarter
                stock_quarter_data.append(stock_data)

        if not stock_quarter_data:
            return analysis_financial

        stock_df_quarters = pd.DataFrame(stock_quarter_data)
        return analysis_financial.join(stock_df_quarters, how="left")

    def _get_col(self, col_name: str) -> pd.Series:
        """컬럼명 기반 수치 시리즈 조회 및 대안 매칭 수행"""
        target_df = self.combined_df

        if col_name in target_df.columns:
            return pd.to_numeric(target_df[col_name], errors="coerce")

        similar_cols = [col for col in target_df.columns if col_name in str(col) or str(col) in col_name]
        if similar_cols:
            return pd.to_numeric(target_df[similar_cols[0]], errors="coerce")

        return pd.Series([np.nan] * len(self.analysis_quarters), index=self.analysis_quarters)

    def _safe_calc(
        self,
        num1: pd.Series,
        num2: pd.Series,
        operation: str = "divide",
        multiply: float = 1,
    ) -> pd.Series:
        """
        결측 및 0 나눗셈 방지 기반 안전 계산 수행
        Args:
            num1: 피연산자 시리즈
            num2: 피연산자 시리즈
            operation: 연산 종류 문자열
            multiply: 계산 결과 곱셈 계수
        Returns:
            계산 결과 시리즈 반환
        """
        if operation == "divide":
            result = np.where((num2 != 0) & ~pd.isna(num2) & ~pd.isna(num1), (num1 / num2) * multiply, np.nan)
        elif operation == "subtract":
            result = np.where(~pd.isna(num1) & ~pd.isna(num2), num1 - num2, np.nan)
        elif operation == "add":
            result = np.where(~pd.isna(num1) & ~pd.isna(num2), num1 + num2, np.nan)
        else:
            result = np.full(len(self.analysis_quarters), np.nan)

        return pd.Series(result, index=self.analysis_quarters)

    def _growth_rate_with_prev_quarter(self, col_name: str) -> pd.Series:
        """
        전 분기 대비 증가율 계산 수행
        Args:
            col_name: 증가율 계산 대상 컬럼명 문자열
        Returns:
            분기별 증가율 시리즈 반환
        """
        growth_rates: dict[str, float] = {}

        for quarter in self.analysis_quarters:
            current_year, current_q = quarter.split("_")

            if current_q == "Q1":
                prev_quarter = f"{int(current_year) - 1}_Q4"
            else:
                q_num = int(current_q[1]) - 1
                prev_quarter = f"{current_year}_Q{q_num}"

            if (
                prev_quarter in self.full_financial_df.index
                and col_name in self.full_financial_df.columns
                and quarter in self.combined_df.index
                and col_name in self.combined_df.columns
            ):
                prev_val = pd.to_numeric(self.full_financial_df.loc[prev_quarter, col_name], errors="coerce")
                curr_val = pd.to_numeric(self.combined_df.loc[quarter, col_name], errors="coerce")

                if not pd.isna(prev_val) and not pd.isna(curr_val) and float(prev_val) != 0:
                    growth_rates[quarter] = float(((curr_val - prev_val) / prev_val) * 100)
                else:
                    growth_rates[quarter] = np.nan
            else:
                growth_rates[quarter] = np.nan

        return pd.Series(growth_rates)

    def calculate_metrics(self) -> dict[str, pd.Series]:
        """
        분기별 재무 지표 계산 수행
        Returns:
            지표명 기준 시리즈 딕셔너리 반환
        """
        metrics: dict[str, pd.Series] = {}

        growth_items = {
            "매출액증가율": "매출액",
            "총자산증가율": "자산총계",
            "유형자산증가율": "유형자산",
            "영업이익증가율": "영업이익",
            "당기순이익증가율": "당기순이익",
            "자기자본증가율": "자본총계",
            "부채총계증가율": "부채총계",
        }

        for metric_name, col_name in growth_items.items():
            metrics[metric_name] = self._growth_rate_with_prev_quarter(col_name)

        metrics.update(
            {
                "CAPEX": self._get_col("CAPEX"),
                "FCF": self._safe_calc(self._get_col("영업활동현금흐름"), self._get_col("CAPEX"), "subtract"),
            }
        )

        metrics.update(
            {
                "매출액영업이익률": self._safe_calc(self._get_col("영업이익"), self._get_col("매출액"), "divide", 100),
                "매출액세전순이익률": self._safe_calc(self._get_col("법인세비용차감전순이익"), self._get_col("매출액"), "divide", 100),
                "당기순이익률": self._safe_calc(self._get_col("당기순이익"), self._get_col("매출액"), "divide", 100),
                "순이익률": self._safe_calc(self._get_col("당기순이익"), self._get_col("매출액"), "divide", 100),
                "ROE": self._safe_calc(self._get_col("당기순이익"), self._get_col("자본총계"), "divide", 100),
                "ROA": self._safe_calc(self._get_col("당기순이익"), self._get_col("자산총계"), "divide", 100),
                "OP마진": self._safe_calc(self._get_col("영업이익"), self._get_col("매출액"), "divide", 100),
                "매출총이익률": self._safe_calc(self._get_col("매출총이익"), self._get_col("매출액"), "divide", 100),
                "세전계속사업이익": self._get_col("법인세비용차감전순이익"),
                "당기순이익": self._get_col("당기순이익"),
                "지배주주순이익": self._get_col("지배기업소유주지분"),
            }
        )

        metrics.update(
            {
                "부채비율": self._safe_calc(self._get_col("부채총계"), self._get_col("자본총계"), "divide", 100),
                "차입금의존도": self._safe_calc(self._get_col("총차입금"), self._get_col("자산총계"), "divide", 100),
                "유동비율": self._safe_calc(self._get_col("유동자산"), self._get_col("유동부채"), "divide", 100),
                "자기자본비율": self._safe_calc(self._get_col("자본총계"), self._get_col("자산총계"), "divide", 100),
                "금융비용대부채비율": self._safe_calc(self._get_col("금융비용"), self._get_col("부채총계"), "divide", 100),
                "현금및유가증권비율": self._safe_calc(self._get_col("현금및현금성자산"), self._get_col("자산총계"), "divide", 100),
                "자산총계": self._get_col("자산총계"),
                "부채총계": self._get_col("부채총계"),
                "자본총계": self._get_col("자본총계"),
                "총차입금": self._get_col("총차입금"),
            }
        )

        metrics.update(
            {
                "총자본회전율": self._safe_calc(self._get_col("매출액"), self._get_col("자본총계"), "divide"),
                "총자산회전율": self._safe_calc(self._get_col("매출액"), self._get_col("자산총계"), "divide"),
                "재고자산회전율": self._safe_calc(self._get_col("매출원가"), self._get_col("재고자산"), "divide"),
                "유형자산회전율": self._safe_calc(self._get_col("매출액"), self._get_col("유형자산"), "divide"),
                "자기자본회전율": self._safe_calc(self._get_col("매출액"), self._get_col("자본총계"), "divide"),
            }
        )

        metrics.update(
            {
                "매출액": self._get_col("매출액"),
                "영업이익": self._get_col("영업이익"),
                "자본금": self._get_col("자본금"),
                "투자활동현금흐름": self._get_col("투자활동현금흐름"),
                "재무활동현금흐름": self._get_col("재무활동현금흐름"),
                "영업활동현금흐름": self._get_col("영업활동현금흐름"),
            }
        )

        market_cap_series = self._get_col("market_cap")
        if not market_cap_series.isna().all():
            metrics.update(
                {
                    "PER": self._get_col("PER"),
                    "PBR": self._get_col("PBR"),
                    "EPS": self._get_col("EPS"),
                    "BPS": self._get_col("BPS"),
                    "발행주식수": self._get_col("listed_shares"),
                    "현금DPS": self._get_col("DPS"),
                    "배당수익률": self._get_col("DIV"),
                    "시가총액": self._get_col("market_cap"),
                    "주가": self._get_col("close_price"),
                    "현금배당수익률": self._safe_calc(self._get_col("DPS"), self._get_col("close_price"), "divide", 100),
                }
            )

        return metrics

    def get_results_json(self) -> str:
        """계산 지표를 분기별 JSON 리스트로 변환 수행"""
        metrics = self.calculate_metrics()
        json_data: list[dict[str, float | str | None]] = []

        for quarter in self.analysis_quarters:
            row: dict[str, float | str | None] = {"분기": quarter}
            for name, series in metrics.items():
                val = series.get(quarter, None) if hasattr(series, "get") else None
                row[name] = float(val) if pd.notna(val) and not np.isinf(val) else None
            json_data.append(row)

        return json.dumps(json_data, ensure_ascii=False, indent=2)

    @property
    def analysis_period(self) -> list[str]:
        """실제 분석 대상 분기 목록 반환 수행"""
        return self.analysis_quarters


def main() -> None:
    """DART 재무제표와 주식 데이터 결합 기반 지표 계산 실행 예시"""
    corp = CorpInfo("삼성전자")
    corp_code = corp.corp_code

    financial_df = fetch_latest_n_years_reports(DART_API_KEY, corp_code, n_years=3)
    financial_df = prepare_finreports(financial_df)
    financial_df = clean_columns(financial_df)

    quarterly_dates = generate_dates_from_financial_df(financial_df)
    stock_info = StockInfo("005930", quarterly_dates)
    stock_df = stock_info._data

    calculator = MetricsCalculator(financial_df, stock_df)
    _ = calculator.get_results_json()


if __name__ == "__main__":
    main()