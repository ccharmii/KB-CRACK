# /KB-CRACK/non_financial_analysis/explainer.py
# 비재무 보고서 기반 이상치 배경 설명 생성 로직

import json
import os
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from .config import CHAT_MODEL
from .indexer import build_or_load_faiss, get_retriever

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SNIPPET_MAX = 420
TOP_K_DEFAULT = 8

_BUCKET_RULES = [
    (
        "profit",
        r"(순이익률|총포괄이익률|매출총이익률|매출원가율|판관비율|영업수익경비율|세전계속사업이익률|ROE|자본금영업이익률|자기자본영업이익률|총자산영업이익률)",
    ),
    ("growth", r"(증가율|YoY|yoy|전년동기|전년 대비|성장률)"),
    ("leverage", r"(부채비율|자기자본비율|재무레버리지|금융비용부담률|유동비율|유동부채비율|비유동부채비율|비유동비율|비유동적합률|유동성)"),
    ("mix", r"(자산구성비율|비유동자산구성비율|유형자산구성비율|유동자산구성비율|재고자산구성비율|유동자산/비유동자산비율|재고자산/유동자산비율)"),
    ("assetchg", r"(총자산증가율|비유동자산증가율|유형자산증가율|부채총계증가율|자기자본증가율|유동자산증가율|재고자산증가율|유동부채증가율|비유동부채증가율)"),
    ("eff", r"(회전율|재고자산회전율|총자산회전율|비유동자산회전율|유형자산회전율|타인자본회전율|자기자본회전율|자본금회전율|총자본회전율|매출원가/재고자산)"),
    ("valuation", r"(PER|PBR|EPS|배당성향|market_cap|시가총액|close_price|주가|시총)"),
]

_EXTRAS_BY_BUCKET = {
    "profit": ["마진", "단가", "판가", "원가", "비용", "감가상각", "충당금", "손상차손", "판매믹스", "가동률", "수율", "환율", "일회성", "리콜", "소송"],
    "growth": ["수요", "출하", "수주", "가동률", "증설", "생산능력", "신규고객", "신제품", "가격전가", "환율", "채널재고", "시장점유율"],
    "leverage": ["차입", "이자비용", "회사채", "만기", "현금흐름", "운전자본", "차환", "담보", "유동성", "금리", "등급", "코버넌트"],
    "mix": ["자산 구성", "CAPEX", "설비투자", "취득", "매각", "재고", "평가손익", "감가상각", "유형자산", "무형자산"],
    "assetchg": ["증가 사유", "감소 사유", "취득", "매각", "유상증자", "배당", "환율 영향", "평가손익", "손상", "충당금"],
    "eff": ["재고회전", "매출채권", "매입채무", "운전자본", "재고일수", "현금회전", "리드타임", "수율", "가동률"],
    "valuation": ["주가", "시가총액", "투자심리", "멀티플", "가이던스", "실적전망", "규제", "소송", "리콜", "뉴스", "수급"],
}

_SYNONYM_BY_METRIC = {
    "ROE": ["Return on Equity", "자기자본이익률", "ROE"],
    "PER": ["PER", "Price Earnings Ratio", "멀티플"],
    "PBR": ["PBR", "Price to Book"],
    "EPS": ["EPS", "Earnings per Share"],
    "YoY": ["YoY", "Year over Year", "전년동기 대비", "전년 대비"],
}

_GENERIC_EXTRAS = ["원인", "요인", "사유", "배경", "변동", "증가", "감소", "가격", "재고", "환율", "수요", "공급", "경쟁", "정책", "규제"]

_SYSTEM_PROMPT = """
                당신은 한국 상장사의 비재무 보고서 텍스트로 '재무지표 이상치의 배경'을 설명하는 분석가입니다.
                규칙:
                - 아래 [CONTEXT]의 문장만 근거로 사용하세요(추측 금지).
                - 근거는 서로 다른 source_idx를 우선하며, 가능한 한 4~8개 제시하세요(품질 낮으면 4개 이상).
                - evidence 항목에는 반드시 source_idx, rcept_no, chunk_id, snippet을 넣으세요.
                - confidence는 0.00~1.00 범위로, 근거의 직접성/일관성/양을 반영하세요.
                - 출력은 JSON 하나만 반환(설명/코드블록 금지).
                """


def _log(msg: str, verbose: bool = True) -> None:
    """표준 로그 메시지 출력 수행"""
    if not verbose:
        return
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def _ensure_dir(p: str | Path) -> None:
    """디렉토리 미존재 시 생성 수행"""
    Path(p).mkdir(parents=True, exist_ok=True)


def _save_json(path: str | Path, data: Any, verbose: bool = True) -> str:
    """JSON 파일 저장 수행"""
    _ensure_dir(Path(path).parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return str(path)


def _corp_root(script_dir: str, corp_code: str) -> str:
    """기업별 데이터 루트 경로 생성 수행"""
    return os.path.join(".\\data", corp_code)


def _db_path(corp_root: str) -> str:
    """기업별 DB 파일 경로 생성 수행"""
    return os.path.join(corp_root, "nfr.db")


def _connect(db_path: str, verbose: bool = True):
    """SQLite DB 연결 객체 생성 수행"""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"DB가 없습니다: {db_path}")

    import sqlite3

    _log(f"DB 연결: {db_path}", verbose)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _latest_quarter(conn, corp_code: str, verbose: bool = True) -> str | None:
    """DB에서 최신 분기 조회 수행"""
    row = (
        conn.execute(
            "SELECT quarter FROM chunks WHERE corp_code=? AND quarter IS NOT NULL AND quarter<>'' "
            "ORDER BY quarter DESC LIMIT 1",
            (corp_code,),
        ).fetchone()
    )
    q = row["quarter"] if row else None
    _log(f"최신 분기 감지: {q or '없음'}", verbose)
    return q


def _bucketize_metric(metric_name: str) -> str:
    """지표명을 규칙 기반 버킷으로 분류 수행"""
    m = metric_name or ""
    for bucket, pat in _BUCKET_RULES:
        if re.search(pat, m, flags=re.I):
            return bucket

    if re.search(r"(PER|PBR|EPS|시가총액|market_cap|주가|close_price)", m, flags=re.I):
        return "valuation"
    if "회전" in m:
        return "eff"
    if "증가율" in m or "YoY" in m:
        return "growth"
    return "profit"


def _direction_terms(anomaly_text: str) -> List[str]:
    """이상치 문구에서 방향성 키워드 추출 수행"""
    t = anomaly_text or ""
    terms: List[str] = []
    if any(k in t for k in ["상회", "증가", "확대", "개선", "호전"]):
        terms += ["증가", "확대", "개선"]
    if any(k in t for k in ["하회", "감소", "축소", "악화", "둔화"]):
        terms += ["감소", "축소", "악화", "둔화"]
    return list(dict.fromkeys(terms))


def _metric_synonyms(metric_name: str) -> List[str]:
    """지표명에서 영문 및 동의어 키워드 보강 수행"""
    n = metric_name or ""
    out: List[str] = []
    if re.search(r"ROE", n, re.I):
        out += _SYNONYM_BY_METRIC["ROE"]
    if re.search(r"PER", n, re.I):
        out += _SYNONYM_BY_METRIC["PER"]
    if re.search(r"PBR", n, re.I):
        out += _SYNONYM_BY_METRIC["PBR"]
    if re.search(r"EPS", n, re.I):
        out += _SYNONYM_BY_METRIC["EPS"]
    if re.search(r"(YoY|전년|증가율)", n, re.I):
        out += _SYNONYM_BY_METRIC["YoY"]
    return list(dict.fromkeys(out))


def _build_query(metric: str, anomaly_text: str) -> str:
    """지표 및 이상치 기반 검색 쿼리 생성 수행"""
    bucket = _bucketize_metric(metric)
    extras = _EXTRAS_BY_BUCKET.get(bucket, []) + _GENERIC_EXTRAS
    syns = _metric_synonyms(metric)
    dirs = _direction_terms(anomaly_text)
    query = " ".join([metric, anomaly_text] + extras + syns + dirs)
    return " ".join(query.split()[:120])


def _user_prompt(metric: str, anomaly_text: str, corp_code: str, quarter: str, query_text: str) -> str:
    """LLM 입력 프롬프트 문자열 생성 수행"""
    return f"""
            [Task]
            다음 재무지표 이상치의 비재무적 배경을 [CONTEXT]에서 찾아 JSON으로 요약하세요.

            [입력]
            - corp_code: {corp_code}
            - quarter: {quarter}
            - metric: {metric}
            - anomaly: {anomaly_text}
            - query_text: {query_text}

            [출력 JSON 스키마]
            {{
            "metric": "{metric}",
            "anomaly_text": "{anomaly_text}",
            "quarter": "{quarter}",
            "explanation_ko": "핵심 원인 요약(최대 5문장, 500자 이내)",
            "drivers": ["핵심 요인 1", "핵심 요인 2", "핵심 요인 3"],
            "confidence": 0.00,
            "evidence": [
                {{"source_idx": 1, "rcept_no":"...", "chunk_id":"...", "snippet":"..."}}
            ]
            }}

            [CONTEXT]
            (각 블록의 [n] 번호는 evidence.source_idx에 대응)
            """


def _format_context_from_docs(docs: List[Any]) -> str:
    """리트리브 문서 목록을 LLM 컨텍스트 문자열로 변환 수행"""
    parts: List[str] = []
    for i, d in enumerate(docs, start=1):
        meta = d.metadata or {}
        head = f"[{i}] quarter={meta.get('quarter','')} rcept_no={meta.get('rcept_no','')} chunk_id={meta.get('chunk_id','')}"
        if meta.get("report_nm") or meta.get("url"):
            head += f" title={meta.get('report_nm','')} url={meta.get('url','')}"
        body = (d.page_content or "").strip()
        parts.append(head + "\n" + body)
    return "\n\n---\n\n".join(parts)


def _extract_json(raw: str, fallback: Dict[str, Any], verbose: bool = True) -> Dict[str, Any]:
    """LLM 출력에서 JSON 파싱 수행"""
    try:
        return json.loads(raw)
    except Exception:
        s, e = raw.find("{"), raw.rfind("}")
        if s != -1 and e != -1 and e > s:
            try:
                return json.loads(raw[s : e + 1])
            except Exception:
                pass

    _log("JSON 파싱 실패로 기본값 대체", verbose)
    return fallback


def _clamp_conf(x: Any, default: float = 0.5) -> float:
    """신뢰도 값을 0과 1 사이로 제한 수행"""
    try:
        v = float(x)
    except Exception:
        v = default
    return max(0.0, min(1.0, v))


def _pack_evidence(ev_items: List[Dict[str, Any]], docs: List[Any], limit: int = 8) -> List[Dict[str, Any]]:
    """evidence 배열을 메타데이터 기반으로 표준화 수행"""
    out: List[Dict[str, Any]] = []
    for ev in (ev_items or [])[:limit]:
        idx = ev.get("source_idx")
        didx = idx - 1 if isinstance(idx, int) else 0

        if not docs:
            meta = {}
        else:
            if 0 <= didx < len(docs):
                meta = docs[didx].metadata or {}
            else:
                meta = docs[0].metadata or {}

        out.append(
            {
                "source_idx": idx if isinstance(idx, int) else 1,
                "rcept_no": meta.get("rcept_no", ""),
                "chunk_id": meta.get("chunk_id", ""),
                "snippet": (ev.get("snippet") or "")[:SNIPPET_MAX],
            }
        )
    return out


def explain_one_metric_with_llm(
    llm: Any,
    retriever_fn: Any,
    corp_code: str,
    quarter: str,
    metric: str,
    anomaly_text: str,
    top_k: int = TOP_K_DEFAULT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    단일 지표 이상치에 대해 근거 문맥 검색과 LLM 요약 수행
    Args:
        llm: LangChain LLM 인스턴스
        retriever_fn: 쿼리 입력 시 문서 리스트를 반환하는 함수
        corp_code: 기업 코드
        quarter: 분기 문자열
        metric: 지표명
        anomaly_text: 이상치 설명 문구
        top_k: 리트리브 문맥 상한
        verbose: 로그 출력 여부
    Returns:
        요약 결과 딕셔너리 반환
    """
    _log(f"지표 처리 시작: {metric}", verbose)
    t0 = time.perf_counter()

    query_text = _build_query(metric, anomaly_text)
    _log(f"검색 쿼리 생성: {query_text[:160]}", verbose)

    docs = retriever_fn(query_text)
    if not docs:
        _log("후보 문맥 부재로 정보 부족", verbose)
        return {
            "metric": metric,
            "anomaly_text": anomaly_text,
            "quarter": quarter,
            "explanation_ko": "해당 분기 텍스트에서 근거를 찾지 못했습니다",
            "drivers": ["정보 부족"],
            "confidence": 0.30,
            "evidence": [],
        }

    context = _format_context_from_docs(docs)
    messages = [
        {"role": "system", "content": _SYSTEM_PROMPT},
        {"role": "user", "content": _user_prompt(metric, anomaly_text, corp_code, quarter, query_text) + "\n" + context},
    ]

    _log("LLM 호출 수행", verbose)
    t1 = time.perf_counter()
    try:
        raw = llm.invoke(messages).content
    except Exception as e:
        _log(f"LLM 호출 오류: {e}", verbose)
        raw = ""
    _log(f"LLM 응답 수신 완료 ({(time.perf_counter() - t1):.2f}s)", verbose)

    fallback = {
        "metric": metric,
        "anomaly_text": anomaly_text,
        "quarter": quarter,
        "explanation_ko": "모델 응답 오류로 요약 불가. 증거 블록을 직접 확인하세요.",
        "drivers": ["정보 부족"],
        "confidence": 0.30,
        "evidence": [],
    }
    data = _extract_json(raw, fallback, verbose=verbose)

    explanation = (data.get("explanation_ko") or "").strip()[:500]
    drivers = [d.strip() for d in (data.get("drivers") or []) if isinstance(d, str) and d.strip()][:6]
    conf = _clamp_conf(data.get("confidence", 0.5))
    evidence = _pack_evidence(data.get("evidence") or [], docs, limit=8)

    _log(f"지표 처리 완료: {metric} conf={conf:.2f} evidence={len(evidence)}개 elapsed={(time.perf_counter() - t0):.2f}s", verbose)

    return {
        "metric": metric,
        "anomaly_text": anomaly_text,
        "quarter": quarter,
        "explanation_ko": explanation or "근거는 있으나 요약이 충분치 않습니다",
        "drivers": drivers or ["증거 기반 요약 부족"],
        "confidence": conf,
        "evidence": evidence,
    }


def run_anomaly_explainer_min(
    anomalies_json_or_dict: Any,
    corp_code: str,
    script_dir: str,
    quarter: str | None = None,
    model: str = CHAT_MODEL,
    temperature: float = 0.2,
    top_k: int = TOP_K_DEFAULT,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    이상치 목록에 대해 근거 생성 파이프라인 수행
    Args:
        anomalies_json_or_dict: 이상치 입력 데이터 또는 파일 경로 또는 JSON 문자열
        corp_code: 기업 코드
        script_dir: 실행 스크립트 기준 경로
        quarter: 분석 분기 지정 값
        model: 사용 LLM 모델명
        temperature: LLM 온도 값
        top_k: 리트리브 문맥 상한
        verbose: 로그 출력 여부
    Returns:
        파이프라인 결과 딕셔너리 반환
    """
    _log("이상치 근거 생성 파이프라인 시작", verbose)

    if isinstance(anomalies_json_or_dict, dict):
        anomalies = anomalies_json_or_dict
        _log(f"입력 타입: dict({len(anomalies)})", verbose)
    else:
        if os.path.exists(str(anomalies_json_or_dict)):
            _log(f"입력 파일 로드: {anomalies_json_or_dict}", verbose)
            with open(anomalies_json_or_dict, "r", encoding="utf-8") as f:
                anomalies = json.load(f)
        else:
            _log("입력 JSON 문자열 파싱", verbose)
            anomalies = json.loads(anomalies_json_or_dict)

    corp_root = _corp_root(script_dir, corp_code)
    db_path = _db_path(corp_root)
    conn = _connect(db_path, verbose=verbose)
    try:
        q = quarter or _latest_quarter(conn, corp_code, verbose=verbose)
        if not q:
            raise RuntimeError("최신 분기 탐색 실패로 인덱싱 선행 필요")
    finally:
        conn.close()

    vs = build_or_load_faiss(corp_root)
    if vs is None:
        raise RuntimeError(f"FAISS 인덱스 미존재로 인덱싱 선행 필요: {os.path.join(corp_root, 'index', 'faiss_index')}")

    retriever = get_retriever(vs, quarter=q, top_k=top_k)

    if not OPENAI_API_KEY:
        _log("OPENAI_API_KEY 미설정 상태", verbose)

    _log(f"LLM 초기화: model={model} temp={temperature}", verbose)
    llm = ChatOpenAI(model=model, temperature=temperature, timeout=120, max_retries=3, api_key=OPENAI_API_KEY)

    results: List[Dict[str, Any]] = []
    _log(f"지표 처리 개수: {len(anomalies)}", verbose)

    for i, (metric, text) in enumerate(anomalies.items(), start=1):
        _log(f"[{i}/{len(anomalies)}] 지표 처리: {metric}", verbose)
        item = explain_one_metric_with_llm(
            llm=llm,
            retriever_fn=retriever,
            corp_code=corp_code,
            quarter=q,
            metric=metric,
            anomaly_text=text,
            top_k=top_k,
            verbose=verbose,
        )
        results.append(item)

    out = {"corp_code": corp_code, "quarter": q, "results": results}
    out_dir = os.path.join(corp_root, "anomaly_explanations")
    out_path = os.path.join(out_dir, f"{q}.json")
    _save_json(out_path, out, verbose=verbose)

    _log("이상치 근거 생성 파이프라인 완료", verbose)
    return out
