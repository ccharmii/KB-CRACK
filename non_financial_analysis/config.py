# /KB-CRACK/non_financial_analysis/config.py
# 비재무지표 분석 환경설정 및 지표 정의

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
DART_API_KEY = os.getenv("DART_API_KEY", "")

BASE_LIST_URL = "https://opendart.fss.or.kr/api/list.json"
BASE_DOC_URL = "https://opendart.fss.or.kr/api/document.xml"
LIST_LOOKBACK_DAYS = 450

DATA_DIR = "./data"
REPORTS_DIR_NAME = "reports"
INDEX_DIR = "index"
DB_FILE = "nfr.db"

CHUNK_SIZE = 1100
CHUNK_OVERLAP = 200
TOP_K = 12

CHAT_MODEL = "gpt-4o-mini"
EMBEDDING_MODEL = "text-embedding-3-small"

SCORE_GUIDE = """
                0: 중대한 부정 사실, 거절, 부적정 등, 1: 부정 징후, 미흡,
                2: 중립, 정보부족, 3: 관리 양호, 4: 우수
              """

REGULAR_TYPE = "A"
REGULAR_DETAIL_ALLOW = ("A001", "A002", "A003")

TITLE_INCLUDE_KEYS = ["사업보고서", "반기보고서", "분기보고서", "제1분기보고서", "제3분기보고서"]
TITLE_EXCLUDE_KEYS = ["현황공시", "대규모기업집단", "분기별공시", "대표회사용", "연1회공시", "연 1회 공시"]

INDICATORS = [
    {
        "pillar": "산업위험",
        "id": "IND_OUTLOOK",
        "name": "산업 전망",
        "desc": "해당 분기의 산업 수요/공급, 규제/정책 변화, 경쟁 강도, 원자재/환율 민감도 등을 근거로 0~4점",
        "cues": ["수요", "공급", "규제", "정책", "경쟁", "점유율", "원자재", "환율", "사이클", "전망", "리스크"],
    },
    {
        "pillar": "경영위험",
        "id": "MGT_STABILITY",
        "name": "경영관리·안정성",
        "desc": "지배구조 안정성, 이사회 독립성/전문성, 내부통제/준법, 경영진 변동/승계, 공시 성실성 등을 0~4점",
        "cues": ["지배구조", "이사회", "독립성", "내부통제", "준법", "감사위원회", "경영진", "승계", "공시 성실성", "내부회계"],
    },
    {
        "pillar": "영업위험",
        "id": "OPS_SALES",
        "name": "영업·구매·생산·판매 리스크",
        "desc": "고객 집중도/다변화, 장기계약/가격전가, 공급망 안정성, 생산/운영 중단, 품질/리콜 등 운영 리스크를 0~4점",
        "cues": ["고객 집중도", "다변화", "장기계약", "가격 전가", "공급망", "수급", "가동률", "생산중단", "운영중단", "품질", "리콜"],
    },
    {
        "pillar": "재무위험(질적)",
        "id": "FIN_QUAL",
        "name": "질적 재무위험",
        "desc": "감사의견/KAM, 내부회계관리제도 의견, 회계정책/추정, 우발/소송, 계속기업 관련 서술 등을 근거로 0~4점",
        "cues": ["감사의견", "한정", "부적정", "KAM", "내부회계관리제도", "회계정책", "회계추정", "우발채무", "소송", "계속기업"],
    },
    {
        "pillar": "신뢰도",
        "id": "TRUST_BEHAV",
        "name": "거래 신뢰도",
        "desc": "연체/체납/부도, 감독당국 제재, 협력사 분쟁, 개인정보/보안사고 등 대외 신뢰 사건을 근거로 0~4점",
        "cues": ["연체", "체납", "부도", "제재", "과징금", "경고", "분쟁", "민원", "개인정보 유출", "보안사고"],
    },
]
