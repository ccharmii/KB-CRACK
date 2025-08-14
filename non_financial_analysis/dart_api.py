# A유형(사업/반기/분기) 공시 수집 -> 제목 우선 분기 재분류 -> TXT 저장
import io, os, re, time
from datetime import datetime, timedelta, date
from typing import List, Dict
import zipfile
from zipfile import BadZipFile
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from lxml import etree, html
from .config import (
    DART_API_KEY, BASE_LIST_URL, BASE_DOC_URL, LIST_LOOKBACK_DAYS,
    REGULAR_TYPE, REGULAR_DETAIL_ALLOW, TITLE_INCLUDE_KEYS, TITLE_EXCLUDE_KEYS,
    REPORTS_DIR_NAME
)
from .quarter_utils import infer_quarter, last_n_complete_quarters

# 세션 재시도
_sess = None
def _session():
    global _sess
    if _sess: 
        return _sess
    s = requests.Session()
    s.mount("https://", HTTPAdapter(max_retries=Retry(
        total=5, backoff_factor=0.5,
        status_forcelist=(429,500,502,503,504),
        allowed_methods=("GET","POST"),
    )))
    _sess = s
    return _sess

# 파일명 전처리
def _safe(s):
    s = (s or "").strip()
    s = re.sub(r"[\\/:*?\"<>|]", "_", s)
    s = re.sub(r"\s+", " ", s)
    return s[:160]

# DART 뷰어 링크 생성
def viewer_url(rcept_no):
    return f"https://dart.fss.or.kr/dsaf001/main.do?rcpNo={rcept_no}"

# “첨부정정/추가서류/추가 제출” 키워드로 첨부성 문서 판별 -> 제외
def _is_attachment_like(title):
    t = (title or "")
    return any(k in t for k in ["첨부정정", "추가서류", "추가 제출"])

# 제목에 정기보고서 키워드가 있는지(+exclude 키워드 제외) 간접 판정
def _title_says_regular(title):
    t = (title or "").replace(" ", "")
    if any(b in t for b in TITLE_EXCLUDE_KEYS): 
        return False
    base = set((TITLE_INCLUDE_KEYS or []) + ["사업보고서","반기보고서","분기보고서","제1분기","제2분기","제3분기","제4분기"])
    return any(k in t for k in base)

# 최근 4개 완료 분기 안에 속하는 보고서 정확히 반환
def list_regular_reports(corp_code, asof=None):
    end = datetime(asof.year, asof.month, asof.day) if asof else datetime.today()
    start = end - timedelta(days=LIST_LOOKBACK_DAYS)
    sess = _session()
    page = 1
    raw: List[Dict] = []

    while True:
        params = {
            "crtfc_key": DART_API_KEY,
            "corp_code": corp_code,
            "bgn_de": start.strftime("%Y%m%d"),
            "end_de": end.strftime("%Y%m%d"),
            "pblntf_ty": REGULAR_TYPE,     # 'A'
            "last_reprt_at": "Y",
            "page_no": page, "page_count": 100,
            "sort": "date", "sort_mth": "desc",
        }
        r = sess.get(BASE_LIST_URL, params=params, timeout=30)
        data = r.json()
        items = data.get("list", []) or []
        if not items and data.get("status") not in ("000",): 
            break

        for it in items:
            title = it.get("report_nm") or ""
            if _is_attachment_like(title): 
                continue
            det = (it.get("pblntf_detail_ty") or "").upper()
            if not ((det in REGULAR_DETAIL_ALLOW) or _title_says_regular(title)):
                continue

            qlabel = infer_quarter(title, det, it.get("rcept_dt"))
            raw.append({
                "corp_code": corp_code,
                "corp_name": it.get("corp_name"),
                "rcept_no": it.get("rcept_no"),
                "rcept_dt": it.get("rcept_dt"),
                "report_nm": title,
                "detail_ty": det,
                "url": viewer_url(it.get("rcept_no")),
                "quarter": qlabel,
            })

        if len(items) < 100: 
            break
        page += 1

    # rcept_no 중복 제외
    seen = {}
    for r in raw: seen.setdefault(r["rcept_no"], r)
    rows = list(seen.values())

    # 완료된 최근 4개 분기 범위만 남김
    targets = set(last_n_complete_quarters(4, asof=(asof or end.date())))
    print("hi!")
    rows.sort(key=lambda x: (x["quarter"], x["rcept_dt"], x["rcept_no"]), reverse=True)
    rows = [r for r in rows if r["quarter"] in targets]
    return rows

# 디코딩
def _guess_decode(b):
    for enc in ("utf-8","cp949","euc-kr","utf-16","latin1"):
        try: return b.decode(enc)
        except: pass
    return b.decode("utf-8","ignore")

# xml -> text
def _xml_to_text(xbytes):
    try:
        root = etree.fromstring(xbytes)
        texts = root.xpath("//text()")
        txt = "\n".join(t.strip() for t in texts if t and t.strip())
        txt = html.fromstring(f"<div>{txt}</div>").text_content()
        return re.sub(r"\n{2,}", "\n\n", txt)
    except Exception:
        import re
        t = re.sub(br"<[^>]+>", b" ", xbytes)
        return re.sub(r"\s+", " ", t.decode("utf-8", "ignore"))
    
# html -> text
def _html_to_text(h):
    if isinstance(h,(bytes,bytearray)): 
        h=_guess_decode(h)
    try:
        doc = html.fromstring(h)
        txt = doc.text_content()
    except Exception:
        txt = re.sub(r"<[^>]+>", " ", h)
    return re.sub(r"\s{2,}", " ", re.sub(r"\n{2,}", "\n\n", txt)).strip()

# ZIP 여부 스니핑 -> 안정적으로 텍스트 추출
def download_text(rcept_no):
    sess = _session()
    r = sess.get(BASE_DOC_URL, params={"crtfc_key": DART_API_KEY, "rcept_no": rcept_no}, timeout=60)
    content, ctype = r.content, (r.headers.get("Content-Type") or "").lower()
    cdisp = (r.headers.get("Content-Disposition") or "").lower()

    is_zip = (len(content)>=2 and content[:2]==b"PK") or ("zip" in ctype) or (".zip" in cdisp)
    if is_zip:
        try:
            z = zipfile.ZipFile(io.BytesIO(content))
        except BadZipFile:
            is_zip = False
        else:
            names = z.namelist()
            cand = [n for n in names if n.lower().endswith("document.xml")] \
                 or [n for n in names if n.lower().endswith(".xml")]
            if cand:
                return _xml_to_text(z.read(cand[0]))
            htmls = [n for n in names if n.lower().endswith((".html",".htm"))]
            if htmls:
                return _html_to_text(z.read(max(htmls, key=lambda n: z.getinfo(n).file_size)))
            textlikes = [n for n in names if n.lower().endswith((".txt",".csv",".md",".json"))]
            if textlikes:
                return "\n\n".join(_guess_decode(z.read(n)) for n in textlikes).strip()
            
            # fallback: 합쳐서 텍스트화
            parts=[]
            for n in names:
                try: b=z.read(n)
                except: continue
                if any(n.lower().endswith(ext) for ext in (".png",".jpg",".jpeg",".gif",".pdf")): 
                    continue
                if b.strip().startswith(b"<"):
                    try: 
                        parts.append(_xml_to_text(b))
                        continue
                    except: 
                        pass
                parts.append(_guess_decode(b))
            return re.sub(r"\n{2,}","\n\n","\n\n".join(p for p in parts if p.strip())).strip()

    # non-zip
    sniff = content[:4096].lower()
    if b"<html" in sniff or "text/html" in ctype:
        return _html_to_text(content)
    if b"<?xml" in content[:256] or "xml" in ctype:
        return _xml_to_text(content)

    return re.sub(r"\s{2,}"," ", _guess_decode(content)).strip()

# 분기 폴더 하위에 rcept_no_title.txt 저장
def ensure_report_files(corp_root, filings):
    saved = []
    for f in filings:
        qdir = os.path.join(corp_root, REPORTS_DIR_NAME, f["quarter"])
        os.makedirs(qdir, exist_ok=True)
        fname = f'{f["rcept_no"]}_{_safe(f["report_nm"])}.txt'
        fpath = os.path.join(qdir, fname)

        is_new = False
        if not os.path.exists(fpath) or os.path.getsize(fpath) < 2000:
            try:
                txt = download_text(f["rcept_no"])
                with open(fpath, "w", encoding="utf-8") as w:
                    w.write(txt)
                is_new = True
                time.sleep(0.4)
            except Exception as e:
                print("❌ 다운로드 실패:", f["rcept_no"], e)
                continue

        meta = dict(f); meta["path"] = fpath
        saved.append({"path": fpath, "meta": meta, "is_new": is_new})
    return saved
