# /KB-CRACK/non_financial_analysis/evaluator.py
# 비재무지표별 점수와 근거를 LLM으로 평가

from __future__ import annotations

import json
import time
from typing import Dict, List

from langchain_openai import ChatOpenAI

from .config import CHAT_MODEL
from .prompt import SYSTEM_PROMPT, user_prompt

LABELS = {0: "위험", 1: "미흡", 2: "중립", 3: "양호", 4: "우수"}


def _clamp_score(x) -> int:
    """점수를 0~4 범위의 정수로 보정"""
    try:
        value = int(round(float(x)))
    except Exception:
        value = 2
    return 0 if value < 0 else 4 if value > 4 else value


def evaluate_quarter(retriever_fn, quarter, indicators, meta_search_text: str = "") -> List[Dict]:
    """
    분기별 비재무지표를 검색 컨텍스트 기반으로 평가하고 근거를 구조화하여 반환
    입력: retriever_fn, quarter, indicators, meta_search_text
    출력: List[Dict]
    """
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    results: List[Dict] = []

    for i, indicator in enumerate(indicators, start=1):
        query = f"{indicator['name']} {indicator.get('desc', '')} " + " ".join(indicator.get("cues", []))
        docs = retriever_fn(query)

        context_parts = []
        for j, doc in enumerate(docs, start=1):
            head = f"[{j}] {doc.metadata.get('report_nm', '')} ({doc.metadata.get('rcept_no', '')})"
            context_parts.append(head + "\n" + doc.page_content)
        context = "\n\n---\n\n".join(context_parts)

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(indicator, quarter) + "\n\n[컨텍스트]\n" + context},
        ]
        raw = llm.invoke(messages).content

        try:
            data = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start != -1 and end != -1:
                data = json.loads(raw[start : end + 1])
            else:
                data = {
                    "indicator_id": indicator["id"],
                    "score": 2,
                    "confidence": 0.3,
                    "rationale": "JSON 파싱 실패로 중립 처리",
                    "evidence": [],
                }

        score = _clamp_score(data.get("score", 2))
        confidence = float(data.get("confidence", 0.5))
        rationale = (data.get("rationale") or "")[:2000]

        evidence_out = []
        for ev in (data.get("evidence") or [])[:20]:
            source_idx = ev.get("source_idx")
            doc_idx = source_idx - 1 if isinstance(source_idx, int) else 0

            if docs and 0 <= doc_idx < len(docs):
                meta = docs[doc_idx].metadata
            else:
                meta = docs[0].metadata if docs else {}

            evidence_out.append(
                {
                    "report_title": meta.get("report_nm", ""),
                    "rcept_no": meta.get("rcept_no", ""),
                    "url": meta.get("url", ""),
                    "snippet": (ev.get("snippet") or "")[:300],
                }
            )

        results.append(
            {
                "indicator_id": indicator["id"],
                "indicator_name": indicator["name"],
                "pillar": indicator["pillar"],
                "score": score,
                "grade_label": LABELS.get(score, "중립"),
                "confidence": confidence,
                "rationale": rationale,
                "evidence": evidence_out,
            }
        )

        time.sleep(0.15)

    return results
