import os, sqlite3
from typing import Iterable, Dict, Any

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

def open_db(db_path: str):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.executescript(SCHEMA)
    return conn

def upsert_filings(conn, rows: Iterable[Dict[str, Any]]):
    q = """INSERT OR REPLACE INTO filings
    (rcept_no, corp_code, corp_name, rcept_dt, report_nm, detail_ty, quarter, url, path)
    VALUES (:rcept_no, :corp_code, :corp_name, :rcept_dt, :report_nm, :detail_ty, :quarter, :url, :path)"""
    conn.executemany(q, rows); conn.commit()

def insert_chunks(conn, rows):
    q = """INSERT OR REPLACE INTO chunks
    (id, rcept_no, quarter, corp_code, start, end, content)
    VALUES (:id, :rcept_no, :quarter, :corp_code, :start, :end, :content)"""
    conn.executemany(q, rows); conn.commit()

def insert_scores_json(conn, rows):
    q = """INSERT OR REPLACE INTO scores_json
    (id, corp_code, quarter, indicator_id, indicator_name, pillar,
     score, confidence, rationale, evidence_json, created_at)
    VALUES (:id, :corp_code, :quarter, :indicator_id, :indicator_name, :pillar,
            :score, :confidence, :rationale, :evidence_json, :created_at)"""
    conn.executemany(q, rows); conn.commit()
