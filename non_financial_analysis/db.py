# /KB-CRACK/non_financial_analysis/db.py
# 비재무 분석 결과 저장용 SQLite 유틸

import os
import sqlite3
from typing import Any, Dict, Iterable

SCHEMA = """
CREATE TABLE IF NOT EXISTS filings (
  rcept_no TEXT PRIMARY KEY,
  corp_code TEXT, corp_name TEXT,
  rcept_dt TEXT, report_nm TEXT,
  detail_ty TEXT,
  quarter TEXT, url TEXT, path TEXT
);

CREATE TABLE IF NOT EXISTS chunks (
  id TEXT PRIMARY KEY,
  rcept_no TEXT, quarter TEXT, corp_code TEXT,
  start INT, end INT, content TEXT
);

CREATE TABLE IF NOT EXISTS scores_json (
  id TEXT PRIMARY KEY,
  corp_code TEXT, quarter TEXT,
  indicator_id TEXT, indicator_name TEXT, pillar TEXT,
  score REAL,
  confidence REAL,
  rationale TEXT,
  evidence_json TEXT,
  created_at TEXT
);
"""


def open_db(db_path: str) -> sqlite3.Connection:
    """
    DB 파일을 열고 필요한 테이블 스키마를 보장
    입력: db_path
    출력: sqlite3.Connection
    """
    parent_dir = os.path.dirname(db_path)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)

    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn


def upsert_filings(conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> None:
    """filings 테이블에 메타데이터를 upsert 수행"""
    query = """
    INSERT OR REPLACE INTO filings
    (rcept_no, corp_code, corp_name, rcept_dt, report_nm, detail_ty, quarter, url, path)
    VALUES (:rcept_no, :corp_code, :corp_name, :rcept_dt, :report_nm, :detail_ty, :quarter, :url, :path)
    """
    conn.executemany(query, rows)
    conn.commit()


def insert_chunks(conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> None:
    """chunks 테이블에 청크 데이터를 upsert 수행"""
    query = """
    INSERT OR REPLACE INTO chunks
    (id, rcept_no, quarter, corp_code, start, end, content)
    VALUES (:id, :rcept_no, :quarter, :corp_code, :start, :end, :content)
    """
    conn.executemany(query, rows)
    conn.commit()


def insert_scores_json(conn: sqlite3.Connection, rows: Iterable[Dict[str, Any]]) -> None:
    """scores_json 테이블에 지표 점수 결과를 upsert 수행"""
    query = """
    INSERT OR REPLACE INTO scores_json
    (id, corp_code, quarter, indicator_id, indicator_name, pillar,
     score, confidence, rationale, evidence_json, created_at)
    VALUES (:id, :corp_code, :quarter, :indicator_id, :indicator_name, :pillar,
            :score, :confidence, :rationale, :evidence_json, :created_at)
    """
    conn.executemany(query, rows)
    conn.commit()
