# /financial_analysis/__init__.py
"""
재무지표 이상치 탐지 모듈

이 모듈은 기업의 재무제표와 주식 데이터를 분석하여 이상치를 탐지
- 개별 기업 재무지표 수집
- 동종업계 평균과의 비교
- 시계열 분석을 통한 이상치 탐지
- 정량적 평가 및 분류별 분석
"""

from .main import analyze_corporation
from .calc_metrics import FinancialAnalyzer
from .finance_metric import get_company_financial_indicators, get_industry_average_indicators

__all__ = [
    'analyze_corporation',
    'FinancialAnalyzer', 
    'get_company_financial_indicators',
    'get_industry_average_indicators'
]