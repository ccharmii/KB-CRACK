# -*- coding: utf-8 -*-
"""
비재무지표 이상치 탐지 모듈

이 모듈은 DART 정기보고서를 분석하여 비재무 위험 요소를 탐지합니다.
- 정기보고서 수집 및 처리
- 텍스트 임베딩 및 벡터 검색
- LLM 기반 위험 지표 평가
- 5개 주요 비재무지표 분석 (산업위험, 경영위험, 영업위험, 재무위험(질적), 신뢰도)
"""

from .main import run_for_corp
from .evaluator import evaluate_quarter
from .indicators import load_indicators

__all__ = [
    'run_for_corp',
    'evaluate_quarter', 
    'load_indicators'
]