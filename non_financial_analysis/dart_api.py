# /KB-CRACK/non_financial_analysis/dart_api.py
# A유형 정기공시 수집 및 본문 텍스트 저장

import io
import os
import re
import time
import zipfile
from datetime import datetime, timedelta
from typing import Dict, List
from zipfile import BadZipFile

import requests
from lxml import etree, html
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from .config import (
    BASE_DOC_URL,
    BASE_LIST_URL,
    DART_API_KEY,
    LIST_LOOKBACK_DAYS,
    REGULAR_DETAIL_ALLOW,
    REGULAR_TYPE,
    REPORTS_DIR_NAME,
    TITLE_EXCLUDE_KEYS,
    TITLE_INCLUDE_KEYS,
)
from .quarter_utils import infer_quarter, last_n_complete_quarters

_sess = None


def _session() -> requests.Session:
    """재시도 설정이 적용된 requests 세션 반환"""
    global _sess
    if _sess:
        return _sess

    session = requests.Session()
    session.mount(
        "https://",
        HTTPAdapter(
            max_retries=Retry(
                total=5,
                backoff_factor=0.5,
                status_forcelist=(429, 500, 502, 503, 504),
                allowed_methods=("GET", "POST"),
            )
        ),
    )
    _sess = session
    return _sess


def _safe(text: str) -> str:
    """파일명에 사용할 수 있도록 문자열 정규화"""
    text = (text or "").strip()
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", " ", text)
    return text[:160]


def viewer_url(rcept_no: str) -> str:
    """DART 뷰어 URL 생성"""
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"


def _is_attachment_like(title: str) -> bool:
    """첨부성 문서 여부 판정"""
    title = title or ""
    return any(keyword in title for keyword in ["첨부정정", "추가서류", "추가 제출"])


def _title_says_regular(title: str) -> bool:
    """제목 기반 정기보고서 여부 판정"""
    normalized = (title or "").replace(" ", "")
    if any(keyword in normalized for keyword in TITLE_EXCLUDE_KEYS):
        return False

    base_keywords = set(
        (TITLE_INCLUDE_KEYS or [])
        + ["사업보고서", "반기보고서", "분기보고서", "제1분기", "제2분기", "제3분기", "제4분기"]
    )
    return any(keyword in normalized for keyword in base_keywords)


def list_regular_reports(corp_code: str, asof: datetime | None = None) -> List[Dict]:
    """
    최근 완료 분기 범위 내 정기보고서 메타데이터 목록 반환
    입력: corp_code, asof(기준일)
    출력: 정기보고서 메타데이터 리스트
    """
    end_dt = datetime(asof.year, asof.month, asof.day) if asof else datetime.today()
    start_dt = end_dt - timedelta(days=LIST_LOOKBACK_DAYS)

    session = _session()
    page = 1
    raw_rows: List[Dict] = []

    while True:
        params = {
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bgn_de": start_dt.strftime("%Y%m%d"),
            "end_de": end_dt.strftime("%Y%m%d"),
            "pblntf_ty": REGULAR_TYPE,
            "last_reprt_at": "Y",
            "page_no": page,
            "page_count": 100,
            "sort": "date",
            "sort_mth": "desc",
        }
        resp = session.get(BASE_LIST_URL, params=params, timeout=30)
        data = resp.json()
        items = data.get("list", []) or []

        if not items and data.get("status") not in ("000",):
            break

        for item in items:
            title = item.get("report_nm") or ""
            if _is_attachment_like(title):
                continue

            detail_type = (item.get("pblntf_detail_ty") or "").upper()
            if not ((detail_type in REGULAR_DETAIL_ALLOW) or _title_says_regular(title)):
                continue

            quarter_label = infer_quarter(title, detail_type, item.get("rcept_dt"))
            raw_rows.append(
                {
                    "corp_code": corp_code,
                    "corp_name": item.get("corp_name"),
                    "rcept_no": item.get("rcept_no"),
                    "rcept_dt": item.get("rcept_dt"),
                    "report_nm": title,
                    "detail_ty": detail_type,
                    "url": viewer_url(item.get("rcept_no")),
                    "quarter": quarter_label,
                }
            )

        if len(items) < 100:
            break
        page += 1

    unique_by_rcept_no: Dict[str, Dict] = {}
    for row in raw_rows:
        unique_by_rcept_no.setdefault(row["rcept_no"], row)

    rows = list(unique_by_rcept_no.values())
    targets = set(last_n_complete_quarters(4, asof=(asof or end_dt.date())))

    rows.sort(key=lambda x: (x["quarter"], x["rcept_dt"], x["rcept_no"]), reverse=True)
    return [row for row in rows if row["quarter"] in targets]


def _guess_decode(blob: bytes) -> str:
    """바이트 문자열을 인코딩 후보로 디코딩"""
    for enc in ("utf-8", "cp949", "euc-kr", "utf-16", "latin1"):
        try:
            return blob.decode(enc)
        except Exception:
            continue
    return blob.decode("utf-8", "ignore")


def _xml_to_text(xml_bytes: bytes) -> str:
    """XML 바이트를 텍스트로 변환"""
    try:
        root = etree.fromstring(xml_bytes)
        texts = root.xpath("//text()")
        text = "\n".join(t.strip() for t in texts if t and t.strip())
        text = html.fromstring(f"<div>{text}</div>").text_content()
        return re.sub(r"\n{2,}", "\n\n", text)
    except Exception:
        stripped = re.sub(br"<[^>]+>", b" ", xml_bytes)
        return re.sub(r"\s+", " ", stripped.decode("utf-8", "ignore"))


def _html_to_text(content: str | bytes) -> str:
    """HTML 문자열 또는 바이트를 텍스트로 변환"""
    if isinstance(content, (bytes, bytearray)):
        content = _guess_decode(content)

    try:
        doc = html.fromstring(content)
        text = doc.text_content()
    except Exception:
        text = re.sub(r"<[^>]+>", " ", content)

    text = re.sub(r"\n{2,}", "\n\n", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()


def download_text(rcept_no: str) -> str:
    """
    rcept_no에 해당하는 본문을 텍스트로 다운로드
    입력: rcept_no
    출력: 본문 텍스트
    """
    session = _session()
    resp = session.get(
        BASE_DOC_URL,
        params={"crtfc_key": DART_API_KEY, "rcept_no": rcept_no},
        timeout=60,
    )
    content = resp.content
    ctype = (resp.headers.get("Content-Type") or "").lower()
    cdisp = (resp.headers.get("Content-Disposition") or "").lower()

    is_zip = (len(content) >= 2 and content[:2] == b"PK") or ("zip" in ctype) or (".zip" in cdisp)
    if is_zip:
        try:
            zf = zipfile.ZipFile(io.BytesIO(content))
        except BadZipFile:
            is_zip = False
        else:
            names = zf.namelist()

            xml_candidates = [n for n in names if n.lower().endswith("document.xml")] or [
                n for n in names if n.lower().endswith(".xml")
            ]
            if xml_candidates:
                return _xml_to_text(zf.read(xml_candidates[0]))

            html_candidates = [n for n in names if n.lower().endswith((".html", ".htm"))]
            if html_candidates:
                largest = max(html_candidates, key=lambda n: zf.getinfo(n).file_size)
                return _html_to_text(zf.read(largest))

            text_candidates = [n for n in names if n.lower().endswith((".txt", ".csv", ".md", ".json"))]
            if text_candidates:
                return "\n\n".join(_guess_decode(zf.read(n)) for n in text_candidates).strip()

            parts: List[str] = []
            for name in names:
                try:
                    blob = zf.read(name)
                except Exception:
                    continue

                if any(name.lower().endswith(ext) for ext in (".png", ".jpg", ".jpeg", ".gif", ".pdf")):
                    continue

                if blob.strip().startswith(b"<"):
                    parts.append(_xml_to_text(blob))
                else:
                    parts.append(_guess_decode(blob))

            merged = "\n\n".join(p for p in parts if p.strip())
            return re.sub(r"\n{2,}", "\n\n", merged).strip()

    sniff = content[:4096].lower()
    if b"<html" in sniff or "text/html" in ctype:
        return _html_to_text(content)
    if b"<?xml" in content[:256] or "xml" in ctype:
        return _xml_to_text(content)

    return re.sub(r"\s{2,}", " ", _guess_decode(content)).strip()


def ensure_report_files(corp_root: str, filings: List[Dict]) -> List[Dict]:
    """
    보고서 목록을 분기 폴더에 TXT로 저장하고 저장 결과 반환
    입력: corp_root, filings
    출력: 저장 결과 리스트(path, meta, is_new)
    """
    saved: List[Dict] = []

    for filing in filings:
        quarter_dir = os.path.join(corp_root, REPORTS_DIR_NAME, filing["quarter"])
        os.makedirs(quarter_dir, exist_ok=True)

        filename = f'{filing["rcept_no"]}_{_safe(filing["report_nm"])}.txt'
        filepath = os.path.join(quarter_dir, filename)

        is_new = False
        if (not os.path.exists(filepath)) or (os.path.getsize(filepath) < 2000):
            try:
                text = download_text(filing["rcept_no"])
                with open(filepath, "w", encoding="utf-8") as w:
                    w.write(text)
                is_new = True
                time.sleep(0.4)
            except Exception as exc:
                print("다운로드 실패", filing["rcept_no"], exc)
                continue

        meta = dict(filing)
        meta["path"] = filepath
        saved.append({"path": filepath, "meta": meta, "is_new": is_new})

    return saved
