# /financial_analysis/fin_news_reason.py
# 재무지표 이상치에 대한 뉴스 근거를 하이브리드 관련성 평가로 분석하는 LangGraph 워크플로우


import json
import operator
import os
from datetime import datetime
from typing import Any, Dict, List

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_teddynote.tools.tavily import TavilySearch
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from typing_extensions import Annotated, TypedDict

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

embedding_model: SentenceTransformer | None = None


class NewsAnalysisState(TypedDict):
    """뉴스 분석 워크플로우 상태 정의"""

    result_dir: Annotated[str, "분석 결과 디렉토리 경로"]
    company_info: Annotated[Dict[str, Any], "회사 기본 정보"]
    anomalies: Annotated[List[Dict[str, Any]], "분석할 이상치 목록"]
    current_anomaly: Annotated[Dict[str, Any], "현재 분석 중인 이상치"]
    current_anomaly_index: Annotated[int, "현재 분석 중인 이상치 인덱스"]
    search_query: Annotated[str, "현재 검색 쿼리"]
    news_results: Annotated[List[Dict[str, Any]], "뉴스 검색 결과"]
    relevance_scores: Annotated[List[float], "관련성 점수"]
    hybrid_relevance_data: Annotated[Dict[str, Any], "하이브리드 관련성 분석 결과"]
    retry_count: Annotated[int, "재검색 횟수"]
    analysis_results: Annotated[List[Dict[str, Any]], "분석 결과 목록"]
    messages: Annotated[List[BaseMessage], operator.add]
    next_node: str


def get_embedding_model() -> SentenceTransformer | None:
    """임베딩 모델 초기화 및 반환 수행"""
    global embedding_model
    if embedding_model is not None:
        return embedding_model

    try:
        embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return embedding_model
    except Exception:
        return None


def create_reference_texts(company_name: str, metric_name: str, quarter: str, description: str) -> List[str]:
    """이상치 정보 기반 뉴스 검색용 참조 텍스트 생성 수행"""
    reference_texts = [
        f"{company_name} {metric_name} {quarter}",
        f"{company_name} {description}",
        f"{metric_name} 이상치 {quarter}",
        description,
        f"{company_name} 재무성과 {quarter}",
        f"{company_name} {metric_name} 변화",
    ]

    metric_keywords: dict[str, list[str]] = {
        "ROE": ["자기자본이익률", "수익성", "경영효율성", "투자수익"],
        "Sales Growth Rate": ["매출", "수익", "성장", "판매", "영업실적"],
        "PER": ["주가수익비율", "기업가치", "주식", "투자", "시장평가"],
        "Debt Ratio": ["부채", "재무안정성", "레버리지", "자본구조"],
        "Current Ratio": ["유동성", "단기지급능력", "현금흐름"],
        "Operating Profit Margin": ["영업이익", "수익성", "비용관리", "사업성과"],
    }

    if metric_name in metric_keywords:
        for keyword in metric_keywords[metric_name]:
            reference_texts.append(f"{company_name} {keyword}")
            reference_texts.append(f"{keyword} {quarter}")

    return reference_texts


def calculate_cosine_similarity_score(news_text: str, reference_texts: List[str]) -> float:
    """코사인 유사도 기반 관련성 점수 산출 수행"""
    model = get_embedding_model()
    if model is None:
        return 5.0

    try:
        news_embedding = model.encode([news_text])
        reference_embeddings = model.encode(reference_texts)
        similarities = cosine_similarity(news_embedding, reference_embeddings)[0]
        cosine_score = float(np.max(similarities) * 10)
        return cosine_score
    except Exception:
        return 5.0


def load_financial_data(state: NewsAnalysisState) -> NewsAnalysisState:
    """결과 디렉토리에서 재무 이상치 및 기업 정보 로드 수행"""
    result_dir = state["result_dir"]
    anomaly_file = os.path.join(result_dir, "financial_anomalies.json")
    company_file = os.path.join(result_dir, "financial_analysis.json")

    anomalies: List[Dict[str, Any]] = []
    company_info: Dict[str, Any] = {}

    try:
        if os.path.exists(anomaly_file):
            with open(anomaly_file, "r", encoding="utf-8") as f:
                raw_anomaly_data = json.load(f)

            if isinstance(raw_anomaly_data, dict):
                for metric_name, details in raw_anomaly_data.items():
                    if isinstance(details, dict):
                        anomalies.append(
                            {
                                "metric_name": details.get("metric_name", metric_name),
                                "description": details.get("description", ""),
                                "severity": details.get("severity", "Medium"),
                                "quarter": details.get("quarter", "Latest"),
                                "type": details.get("type", ""),
                                "source": details.get("source", ""),
                            }
                        )
                    else:
                        anomalies.append(
                            {
                                "metric_name": metric_name,
                                "description": str(details),
                                "severity": "Medium",
                                "quarter": "Latest",
                                "type": "unknown",
                                "source": "unknown",
                            }
                        )
            elif isinstance(raw_anomaly_data, list):
                anomalies = raw_anomaly_data

        if os.path.exists(company_file):
            with open(company_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            company_info = data.get("기업_정보", {})

    except Exception:
        anomalies = []
        company_info = {}

    new_state = dict(state)
    if anomalies:
        new_state.update(
            {
                "anomalies": anomalies,
                "company_info": company_info,
                "current_anomaly": anomalies[0],
                "current_anomaly_index": 0,
                "retry_count": 0,
                "analysis_results": [],
                "next_node": "generate_search_query",
            }
        )
    else:
        new_state.update(
            {
                "anomalies": [],
                "company_info": company_info,
                "analysis_results": [],
                "next_node": "save_results",
            }
        )

    return new_state


def generate_search_query(state: NewsAnalysisState) -> NewsAnalysisState:
    """이상치 정보 기반 뉴스 검색 쿼리 생성 수행"""
    query_prompt = ChatPromptTemplate.from_template(
        """
        다음 재무지표 이상치에 대한 뉴스 검색 쿼리를 생성해주세요

        회사명: {company_name}
        지표명: {metric_name}
        분기: {quarter}
        이상치 설명: {description}
        심각도: {severity}

        현재 시도: {retry_count}회
        이전 쿼리: {previous_query}

        요청사항
        - 한국어 뉴스 검색에 적합한 키워드로 구성
        - 회사명과 관련 재무지표를 포함
        - 재시도인 경우 다른 각도의 키워드 사용
        - 15단어 이내로 간결하게

        검색 쿼리만 출력해주세요
        """
            )

    current_anomaly = state["current_anomaly"]
    company_info = state["company_info"]
    retry_count = state.get("retry_count", 0)

    company_name = company_info.get("company_name", company_info.get("기업명", ""))
    previous_query = state.get("search_query", "")

    try:
        chain = query_prompt | llm
        search_query = chain.invoke(
            {
                "company_name": company_name,
                "metric_name": current_anomaly.get("metric_name", ""),
                "quarter": current_anomaly.get("quarter", ""),
                "description": current_anomaly.get("description", ""),
                "severity": current_anomaly.get("severity", ""),
                "retry_count": retry_count,
                "previous_query": previous_query,
            }
        ).content.strip()
    except Exception:
        search_query = f"{company_name} 재무 변화"

    new_state = dict(state)
    new_state.update({"search_query": search_query, "next_node": "search_news"})
    return new_state


def search_news(state: NewsAnalysisState) -> NewsAnalysisState:
    """Tavily 기반 뉴스 검색 수행"""
    search_query = state["search_query"]

    try:
        search_tool = TavilySearch(max_results=3, days=90)
        raw_results = search_tool.search(query=search_query)
    except Exception:
        raw_results = []

    news_results: List[Dict[str, Any]] = []
    if isinstance(raw_results, list):
        for result in raw_results:
            if not isinstance(result, dict):
                continue
            news_results.append(
                {
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", "")[:1000],
                    "published_date": result.get("published_date", ""),
                    "source": result.get("source", ""),
                }
            )

    new_state = dict(state)
    new_state.update({"news_results": news_results, "next_node": "check_relevance_hybrid"})
    return new_state


def check_relevance_hybrid(state: NewsAnalysisState) -> NewsAnalysisState:
    """코사인 유사도와 LLM 의미 분석 결합 기반 관련성 평가 수행"""
    current_anomaly = state["current_anomaly"]
    company_info = state["company_info"]
    news_results = state.get("news_results", [])

    if not news_results:
        new_state = dict(state)
        new_state.update(
            {
                "relevance_scores": [],
                "hybrid_relevance_data": {},
                "next_node": "rewrite_query_or_analyze",
            }
        )
        return new_state

    semantic_analysis_prompt = ChatPromptTemplate.from_template(
        """
        다음 뉴스와 재무지표 이상치 간의 의미적 연관성을 분석해주세요

        회사명: {company_name}
        지표명: {metric_name}
        분기: {quarter}
        이상치 설명: {description}

        제목: {news_title}
        내용: {news_content}

        다음 JSON 형식으로 답변해주세요
        {
        "causal_relationship": {"score": 8, "explanation": "설명"},
        "temporal_relevance": {"score": 7, "explanation": "설명"},
        "business_alignment": {"score": 9, "explanation": "설명"},
        "financial_impact": {"score": 6, "explanation": "설명"},
        "overall_semantic_score": 7.5,
        "reasoning": "종합 판단 근거"
        }
        """
            )

    company_name = company_info.get("company_name", company_info.get("기업명", ""))
    metric_name = current_anomaly.get("metric_name", "")
    quarter = current_anomaly.get("quarter", "")
    description = current_anomaly.get("description", "")

    reference_texts = create_reference_texts(company_name, metric_name, quarter, description)

    hybrid_results: List[Dict[str, Any]] = []
    final_scores: List[float] = []

    for i, news in enumerate(news_results):
        news_title = news.get("title", "")
        news_content = news.get("content", "")
        news_text = f"{news_title} {news_content}"

        cosine_score = calculate_cosine_similarity_score(news_text, reference_texts)

        try:
            chain = semantic_analysis_prompt | llm | JsonOutputParser()
            semantic_analysis = chain.invoke(
                {
                    "company_name": company_name,
                    "metric_name": metric_name,
                    "quarter": quarter,
                    "description": description,
                    "news_title": news_title,
                    "news_content": news_content,
                }
            )
            semantic_score = float(semantic_analysis.get("overall_semantic_score", 5.0))
        except Exception:
            semantic_analysis = {"overall_semantic_score": 5.0, "reasoning": "LLM 분석 실패"}
            semantic_score = 5.0

        hybrid_score = float(min(10.0, max(0.0, cosine_score * 0.3 + semantic_score * 0.7)))

        hybrid_results.append(
            {
                "news_index": i,
                "cosine_similarity_score": float(round(cosine_score, 2)),
                "semantic_analysis_score": float(round(semantic_score, 2)),
                "hybrid_relevance_score": float(round(hybrid_score, 2)),
                "semantic_analysis": semantic_analysis,
                "news_info": {
                    "title": news_title,
                    "url": news.get("url", ""),
                    "published_date": news.get("published_date", ""),
                },
            }
        )
        final_scores.append(hybrid_score)

    hybrid_results.sort(key=lambda x: x["hybrid_relevance_score"], reverse=True)

    avg_hybrid_score = float(np.mean(final_scores)) if final_scores else 0.0
    max_hybrid_score = float(np.max(final_scores)) if final_scores else 0.0

    relevant_threshold = 6.0
    has_relevant_news = any(score >= relevant_threshold for score in final_scores)

    hybrid_relevance_data = {
        "analysis_method": "hybrid_cosine_llm",
        "weights": {"cosine": 0.3, "semantic": 0.7},
        "results": hybrid_results,
        "statistics": {
            "total_news_count": len(news_results),
            "average_hybrid_score": round(avg_hybrid_score, 2),
            "max_hybrid_score": round(max_hybrid_score, 2),
            "relevant_news_count": len([s for s in final_scores if s >= relevant_threshold]),
        },
    }

    new_state = dict(state)
    new_state.update(
        {
            "relevance_scores": final_scores,
            "hybrid_relevance_data": hybrid_relevance_data,
            "next_node": "analyze_news" if has_relevant_news else "rewrite_query_or_analyze",
        }
    )
    return new_state


def rewrite_query_or_analyze(state: NewsAnalysisState) -> NewsAnalysisState:
    """관련성 점수 기반 재검색 또는 분석 진행 분기 수행"""
    retry_count = state.get("retry_count", 0)
    relevance_scores = state.get("relevance_scores", [])

    has_relevant = any(score >= 6 for score in relevance_scores) if relevance_scores else False

    new_state = dict(state)

    max_retry = 2
    if not has_relevant and retry_count < max_retry:
        new_state.update({"retry_count": retry_count + 1, "next_node": "generate_search_query"})
        return new_state

    new_state.update({"next_node": "analyze_news"})
    return new_state


def analyze_news(state: NewsAnalysisState) -> NewsAnalysisState:
    """관련 뉴스 기반 이상치 원인 분석 수행"""
    current_anomaly = state["current_anomaly"]
    company_info = state["company_info"]
    news_results = state.get("news_results", [])
    hybrid_relevance_data = state.get("hybrid_relevance_data", {})

    analysis_prompt = ChatPromptTemplate.from_template(
        """
        다음 뉴스들을 바탕으로 재무지표 이상치의 원인을 분석해주세요

        지표명: {metric_name}
        분기: {quarter}
        이상치 설명: {description}
        심각도: {severity}
        회사명: {company_name}

        관련 뉴스
        {relevant_news}

        하이브리드 분석 정보
        - 평균 관련성 점수: {avg_relevance_score}/10
        - 관련성 높은 뉴스: {relevant_count}개

        다음 JSON 형식으로 답변해주세요
        {
        "primary_cause": "주요 원인",
        "confidence_level": 8,
        "supporting_evidence": ["근거1", "근거2"],
        "detailed_explanation": "상세 설명",
        "news_sources": ["URL1", "URL2"],
        "impact_assessment": "영향 평가",
        "risk_level": "낮음/보통/높음",
        "relevance_quality": "품질 평가"
        }
        """
            )

    relevant_news: List[Dict[str, Any]] = []
    if hybrid_relevance_data.get("results"):
        for result in hybrid_relevance_data["results"]:
            if result.get("hybrid_relevance_score", 0) < 5.0:
                continue
            news_idx = result.get("news_index", -1)
            if 0 <= news_idx < len(news_results):
                news = news_results[news_idx]
                relevant_news.append(
                    {
                        **news,
                        "hybrid_score": result.get("hybrid_relevance_score", 0),
                        "cosine_score": result.get("cosine_similarity_score", 0),
                        "semantic_score": result.get("semantic_analysis_score", 0),
                    }
                )

    if not relevant_news:
        relevant_news_text = "관련 뉴스 부재"
        stats = {"average_hybrid_score": 0, "relevant_news_count": 0}
    else:
        news_text_blocks: List[str] = []
        for i, news in enumerate(relevant_news):
            news_text_blocks.append(
                "\n".join(
                    [
                        f"[뉴스 {i + 1}] (하이브리드 점수: {news.get('hybrid_score', 'N/A')})",
                        f"제목: {news.get('title', '')}",
                        f"내용: {news.get('content', '')}",
                        f"출처: {news.get('source', '')}",
                        f"URL: {news.get('url', '')}",
                    ]
                )
            )
        relevant_news_text = "\n\n".join(news_text_blocks)
        stats = hybrid_relevance_data.get("statistics", {"average_hybrid_score": 0, "relevant_news_count": 0})

    avg_relevance = stats.get("average_hybrid_score", 0)
    relevant_count = stats.get("relevant_news_count", 0)

    try:
        chain = analysis_prompt | llm | JsonOutputParser()
        analysis_result = chain.invoke(
            {
                "metric_name": current_anomaly.get("metric_name", ""),
                "quarter": current_anomaly.get("quarter", ""),
                "description": current_anomaly.get("description", ""),
                "severity": current_anomaly.get("severity", ""),
                "company_name": company_info.get("company_name", company_info.get("기업명", "")),
                "relevant_news": relevant_news_text,
                "avg_relevance_score": avg_relevance,
                "relevant_count": relevant_count,
            }
        )
        analysis_payload: Dict[str, Any] = analysis_result
    except Exception as e:
        analysis_payload = {"error": f"분석 오류 {str(e)}"}

    result_data = {
        "anomaly_info": current_anomaly,
        "analysis": analysis_payload,
        "news_evidence": relevant_news,
        "hybrid_relevance_analysis": hybrid_relevance_data,
        "search_info": {
            "final_query": state.get("search_query", ""),
            "retry_count": state.get("retry_count", 0),
            "total_news_found": len(news_results),
            "relevant_news_count": len(relevant_news),
        },
        "timestamp": datetime.now().isoformat(),
    }

    current_results = state.get("analysis_results", [])
    current_results.append(result_data)

    new_state = dict(state)
    new_state.update({"analysis_results": current_results, "next_node": "check_remaining_anomalies"})
    return new_state


def check_remaining_anomalies(state: NewsAnalysisState) -> NewsAnalysisState:
    """잔여 이상치 존재 여부 확인 및 다음 이상치 설정 수행"""
    anomalies = state.get("anomalies", [])
    current_index = state.get("current_anomaly_index", 0)
    next_index = current_index + 1

    new_state = dict(state)

    if next_index < len(anomalies):
        new_state.update(
            {
                "current_anomaly": anomalies[next_index],
                "current_anomaly_index": next_index,
                "retry_count": 0,
                "news_results": [],
                "relevance_scores": [],
                "hybrid_relevance_data": {},
                "next_node": "generate_search_query",
            }
        )
    else:
        new_state.update({"next_node": "save_results"})

    return new_state


def save_results(state: NewsAnalysisState) -> NewsAnalysisState:
    """뉴스 분석 결과 JSON 파일 저장 수행"""
    result_dir = state["result_dir"]
    analysis_results = state.get("analysis_results", [])
    company_info = state.get("company_info", {})

    final_results = {
        "success": True,
        "analysis_method": "hybrid_relevance_check",
        "methodology": {
            "cosine_similarity_weight": 0.3,
            "llm_semantic_weight": 0.7,
            "relevance_threshold": 6.0,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2",
        },
        "company_info": company_info,
        "analysis_summary": {
            "total_anomalies": len(analysis_results),
            "successful_analyses": len([r for r in analysis_results if "error" not in r.get("analysis", {})]),
            "analysis_timestamp": datetime.now().isoformat(),
        },
        "anomaly_news_analyses": analysis_results,
    }

    output_file = os.path.join(result_dir, "anomaly_news_analysis.json")
    try:
        os.makedirs(result_dir, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

    new_state = dict(state)
    new_state.update({"next_node": "END"})
    return new_state


def route_next_node(state: NewsAnalysisState) -> str:
    """상태 값 기반 다음 노드명 반환 수행"""
    return state.get("next_node", "END")


def create_news_analysis_graph():
    """뉴스 분석 그래프 구성 및 컴파일 수행"""
    workflow = StateGraph(NewsAnalysisState)

    workflow.add_node("load_financial_data", load_financial_data)
    workflow.add_node("generate_search_query", generate_search_query)
    workflow.add_node("search_news", search_news)
    workflow.add_node("check_relevance_hybrid", check_relevance_hybrid)
    workflow.add_node("rewrite_query_or_analyze", rewrite_query_or_analyze)
    workflow.add_node("analyze_news", analyze_news)
    workflow.add_node("check_remaining_anomalies", check_remaining_anomalies)
    workflow.add_node("save_results", save_results)

    workflow.set_entry_point("load_financial_data")

    workflow.add_conditional_edges(
        "load_financial_data",
        route_next_node,
        {"generate_search_query": "generate_search_query", "save_results": "save_results"},
    )
    workflow.add_edge("generate_search_query", "search_news")
    workflow.add_edge("search_news", "check_relevance_hybrid")

    workflow.add_conditional_edges(
        "check_relevance_hybrid",
        route_next_node,
        {"rewrite_query_or_analyze": "rewrite_query_or_analyze", "analyze_news": "analyze_news"},
    )
    workflow.add_conditional_edges(
        "rewrite_query_or_analyze",
        route_next_node,
        {"generate_search_query": "generate_search_query", "analyze_news": "analyze_news"},
    )

    workflow.add_edge("analyze_news", "check_remaining_anomalies")

    workflow.add_conditional_edges(
        "check_remaining_anomalies",
        route_next_node,
        {"generate_search_query": "generate_search_query", "save_results": "save_results"},
    )

    workflow.add_edge("save_results", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    return app


def run_anomaly_news_analysis(result_dir: str) -> Dict[str, Any]:
    """
    재무지표 이상치 뉴스 근거 분석 워크플로우 실행 수행

    Args:
        result_dir: 분석 결과 디렉토리 경로 문자열

    Returns:
        실행 결과 딕셔너리 반환
    """
    get_embedding_model()

    app = create_news_analysis_graph()

    initial_state: NewsAnalysisState = {
        "result_dir": result_dir,
        "messages": [],
        "analysis_results": [],
        "current_anomaly_index": 0,
        "retry_count": 0,
        "next_node": "load_financial_data",
        "company_info": {},
        "anomalies": [],
        "current_anomaly": {},
        "search_query": "",
        "news_results": [],
        "relevance_scores": [],
        "hybrid_relevance_data": {},
    }

    config = {
        "configurable": {"thread_id": f"news_analysis_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"},
        "recursion_limit": 100,
    }

    final_state: Dict[str, Any] | None = None
    max_steps = 50
    step_count = 0

    for event in app.stream(initial_state, config):
        step_count += 1
        node_name = list(event.keys())[0]
        node_data = event[node_name]
        final_state = node_data

        if node_data.get("next_node", "END") == "END":
            break
        if step_count >= max_steps:
            break

    if final_state is None:
        final_state = app.get_state(config).values

    analysis_results = final_state.get("analysis_results", [])

    total_news_analyzed = 0
    avg_hybrid_scores: List[float] = []

    for result in analysis_results:
        hybrid_data = result.get("hybrid_relevance_analysis", {})
        stats = hybrid_data.get("statistics", {})
        if not stats:
            continue

        total_news_analyzed += int(stats.get("total_news_count", 0))
        avg_score = float(stats.get("average_hybrid_score", 0))
        if avg_score > 0:
            avg_hybrid_scores.append(avg_score)

    overall_avg_score = float(np.mean(avg_hybrid_scores)) if avg_hybrid_scores else 0.0

    return {
        "success": True,
        "analysis_method": "hybrid_relevance_check",
        "total_anomalies_analyzed": len(analysis_results),
        "total_news_analyzed": total_news_analyzed,
        "average_relevance_score": round(overall_avg_score, 2),
        "analysis_results": analysis_results,
        "output_file": os.path.join(result_dir, "anomaly_news_analysis.json"),
    }


def main() -> None:
    test_result_dir = "analysis_results/20250812_123456"
    run_anomaly_news_analysis(test_result_dir)


if __name__ == "__main__":
    main()
