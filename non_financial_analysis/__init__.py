# /KB-CRACK/non_financial_analysis/__init__.py
# 비재무지표 이상치 탐지 패키지 공개 인터페이스 정의

"""
비재무지표 이상치 탐지 모듈

이 모듈은 DART 정기보고서를 분석하여 비재무 위험 요소를 탐지함
- 정기보고서 수집 및 처리 수행
- 텍스트 임베딩 및 벡터 검색 수행
- LLM 기반 위험 지표 평가 수행
- 5개 주요 비재무지표 분석 수행
"""

from .main import run_for_corp
from .evaluator import evaluate_quarter
from .indicators import load_indicators

__all__ = [
    "run_for_corp",
    "evaluate_quarter",
    "load_indicators",
]
