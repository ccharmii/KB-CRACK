# 각 비재무지표마다의 신용 평가 + 근거
from __future__ import annotations
import json, time
from typing import Dict, List
from langchain_openai import ChatOpenAI
from .prompt import SYSTEM_PROMPT, user_prompt
from .config import CHAT_MODEL

LABELS = {0: "위험", 1: "미흡", 2: "중립", 3: "양호", 4: "우수"}

# 0~4 보정 (점수 후처리)
def _clamp_score(x) -> int:
    try: 
        v = int(round(float(x)))
    except Exception: 
        v = 2
    return 0 if v < 0 else 4 if v > 4 else v


def evaluate_quarter(retriever_fn, quarter, indicators, meta_search_text=""):
    llm = ChatOpenAI(model=CHAT_MODEL, temperature=0.2)
    results: List[Dict] = []

    # 비재무지표마다 평가 시작
    for i, ind in enumerate(indicators, start=1):
        print(f"   🔸 [{i:02d}/{len(indicators)}] {ind['id']} - {ind['name']} … 검색/평가")
        query = f"{ind['name']} {ind.get('desc','')} " + " ".join(ind.get("cues", []))
        docs = retriever_fn(query) # 검색 및 상위 청크들 저장

        # 컨텍스트 구성 
        context_parts = []
        for j, d in enumerate(docs, start=1):
            head = f"[{j}] {d.metadata.get('report_nm', '')} ({d.metadata.get('rcept_no', '')})"
            context_parts.append(head + "\n" + d.page_content)
        context = "\n\n---\n\n".join(context_parts)

        # LLM 호출
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(ind, quarter) + "\n\n[컨텍스트]\n" + context},
        ]
        raw = llm.invoke(messages).content

        try:
            data = json.loads(raw)
        except Exception:
            s = raw.find("{"); e = raw.rfind("}")
            data = json.loads(raw[s:e+1]) if s!=-1 and e!=-1 else {
                "indicator_id": ind["id"], "score": 2, "confidence": 0.3,
                "rationale": "JSON 파싱 실패로 중립 처리", "evidence": []
            }

        # 후처리
        score = _clamp_score(data.get("score", 2))
        conf = float(data.get("confidence", 0.5))
        rationale = (data.get("rationale") or "")[:2000]

        # 근거 매핑
        ev_out = []
        for ev in (data.get("evidence") or [])[:20]:
            idx = ev.get("source_idx")
            didx = idx-1 if isinstance(idx, int) else 0
            meta = docs[didx].metadata if docs and 0 <= didx < len(docs) else (docs[0].metadata if docs else {})
            ev_out.append({
                "report_title": meta.get("report_nm",""),
                "rcept_no": meta.get("rcept_no",""),
                "url": meta.get("url",""),
                "snippet": (ev.get("snippet") or "")[:300]
            })

        # 최종 결과
        results.append({
            "indicator_id": ind["id"],
            "indicator_name": ind["name"],
            "pillar": ind["pillar"],
            "score": score,
            "grade_label": LABELS.get(score, "중립"),
            "confidence": conf,
            "rationale": rationale,
            "evidence": ev_out
        })

        print(f"      ↳ 점수={score}({LABELS.get(score)}) / conf={conf:.2f} / evid={len(ev_out)}")
        time.sleep(0.15)

    return results
