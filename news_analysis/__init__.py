# -*- coding: utf-8 -*-
"""
뉴스 이상징후 탐지 모듈

이 모듈은 기업 관련 뉴스를 분석하여 신용위험 징후를 탐지합니다.
- 일일 뉴스 검색 및 수집
- 기업 재무상황 고려한 위험도 평가
- LLM 기반 뉴스 분석 및 위험 분류
- 신용위험 징후 리포트 생성
"""

from .news_search import CreditRiskNewsAnalyzer

__all__ = [
    'CreditRiskNewsAnalyzer'
]