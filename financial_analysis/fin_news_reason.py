# -*- coding: utf-8 -*-
"""
재무지표 이상치 뉴스 근거 분석 모듈 (LangGraph 기반)
하이브리드 관련성 체크 (코사인 유사도 + LLM) 적용
"""

import json
import operator
import os
import numpy as np
from typing_extensions import TypedDict, Annotated, List, Dict, Any
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import JsonOutputParser
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_teddynote.tools.tavily import TavilySearch
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# 상태 정의
class NewsAnalysisState(TypedDict):
    """뉴스 분석 시스템의 상태"""
    # 입력 데이터
    result_dir: Annotated[str, "분석 결과 디렉토리 경로"]
    company_info: Annotated[Dict[str, Any], "회사 기본 정보"]
    anomalies: Annotated[List[Dict], "분석할 이상치 목록"]
    
    # 분석 과정 데이터
    current_anomaly: Annotated[Dict[str, Any], "현재 분석 중인 이상치"]
    current_anomaly_index: Annotated[int, "현재 분석 중인 이상치 인덱스"]
    search_query: Annotated[str, "현재 검색 쿼리"]
    news_results: Annotated[List[Dict], "뉴스 검색 결과"]
    relevance_scores: Annotated[List[float], "관련성 점수"]
    hybrid_relevance_data: Annotated[Dict, "하이브리드 관련성 분석 결과"]
    retry_count: Annotated[int, "재검색 횟수"]
    
    # 결과 데이터
    analysis_results: Annotated[List[Dict], "분석 결과 목록"]
    
    # 메시지와 다음 노드
    messages: Annotated[List[BaseMessage], operator.add]
    next_node: str

# LLM 설정
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# 임베딩 모델 초기화 (전역 변수로 한 번만 로드)
embedding_model = None

def get_embedding_model():
    """임베딩 모델을 가져오거나 초기화"""
    global embedding_model
    if embedding_model is None:
        try:
            embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        except Exception as e:
            print(f"    ⚠️ 임베딩 모델 로드 실패: {e}")
            return None
    return embedding_model

def create_reference_texts(company_name: str, metric_name: str, 
                         quarter: str, description: str) -> List[str]:
    """
    재무지표 이상치 정보를 기반으로 참조 텍스트(뉴스 검색 쿼리용, 한국어 문자열)를 생성합니다.
    """
    reference_texts = [
        f"{company_name} {metric_name} {quarter}",
        f"{company_name} {description}",
        f"{metric_name} 이상치 {quarter}",
        description,
        f"{company_name} 재무성과 {quarter}",
        f"{company_name} {metric_name} 변화"
    ]
    
    # 지표별 특화 키워드 추가
    metric_keywords = {
        "ROE": ["자기자본이익률", "수익성", "경영효율성", "투자수익"],
        "Sales Growth Rate": ["매출", "수익", "성장", "판매", "영업실적"],
        "PER": ["주가수익비율", "기업가치", "주식", "투자", "시장평가"],
        "Debt Ratio": ["부채", "재무안정성", "레버리지", "자본구조"],
        "Current Ratio": ["유동성", "단기지급능력", "현금흐름"],
        "Operating Profit Margin": ["영업이익", "수익성", "비용관리", "사업성과"]
    }
    
    if metric_name in metric_keywords:
        for keyword in metric_keywords[metric_name]:
            reference_texts.append(f"{company_name} {keyword}")
            reference_texts.append(f"{keyword} {quarter}")
    
    return reference_texts

def calculate_cosine_similarity_score(news_text: str, reference_texts: List[str]) -> float:
    """
    코사인 유사도 계산
    """
    try:
        model = get_embedding_model()
        if model is None:
            return 5.0  # 기본값
        
        # 텍스트 임베딩 생성
        news_embedding = model.encode([news_text])
        reference_embeddings = model.encode(reference_texts)
        
        # 코사인 유사도 계산
        similarities = cosine_similarity(news_embedding, reference_embeddings)[0]
        
        # 최대 유사도를 0-10 스케일로 변환
        max_similarity = np.max(similarities)
        cosine_score = max_similarity * 10
        
        # numpy.float64를 Python float로 변환
        return float(cosine_score)
    except Exception as e:
        print(f"    ⚠️ 코사인 유사도 계산 오류: {e}")
        return 5.0

def load_financial_data(state: NewsAnalysisState) -> NewsAnalysisState:
    """결과 디렉토리에서 재무분석 결과와 회사 정보를 로드"""
    
    result_dir = state["result_dir"]
    
    # 이상치 데이터 로드
    anomaly_file = os.path.join(result_dir, "financial_anomalies.json")
    company_file = os.path.join(result_dir, "financial_analysis.json")

    try:
        # 이상치 데이터 로드 및 변환
        if os.path.exists(anomaly_file):
            with open(anomaly_file, 'r', encoding='utf-8') as f:
                raw_anomaly_data = json.load(f)
            
            # 딕셔너리 형태를 리스트로 변환
            anomalies = []
            if isinstance(raw_anomaly_data, dict):
                for metric_name, details in raw_anomaly_data.items():
                    if isinstance(details, dict):
                        anomaly_item = {
                            "metric_name": details.get("metric_name", metric_name),
                            "description": details.get("description", ""),
                            "severity": details.get("severity", "Medium"),
                            "quarter": details.get("quarter", "Latest"),
                            "type": details.get("type", ""),
                            "source": details.get("source", "")
                        }
                        anomalies.append(anomaly_item)
                    else:
                        anomaly_item = {
                            "metric_name": metric_name,
                            "description": str(details),
                            "severity": "Medium",
                            "quarter": "Latest",
                            "type": "unknown",
                            "source": "unknown"
                        }
                        anomalies.append(anomaly_item)
            elif isinstance(raw_anomaly_data, list):
                anomalies = raw_anomaly_data
            else:
                print(f"    ⚠️ 예상치 못한 이상치 데이터 형태: {type(raw_anomaly_data)}")
                anomalies = []
                
            print(f"    📊 이상치 데이터 변환 완료: {len(anomalies)}개")
            for i, anomaly in enumerate(anomalies):
                print(f"        {i+1}. {anomaly['metric_name']}: {anomaly['description'][:50]}...")
        else:
            print(f"    ⚠️ 이상치 파일을 찾을 수 없습니다: {anomaly_file}")
            anomalies = []
        
        # 회사 정보 로드
        company_info = {}
        if os.path.exists(company_file):
            with open(company_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
                company_info = data.get("기업_정보", {})
        
        # 상태를 완전히 새로 생성하여 반환
        new_state = dict(state)  # 기존 상태 복사
        if anomalies:
            new_state.update({
                "anomalies": anomalies,
                "company_info": company_info,
                "current_anomaly": anomalies[0],
                "current_anomaly_index": 0,
                "retry_count": 0,
                "analysis_results": [],
                "next_node": "generate_search_query"
            })
        else:
            new_state.update({
                "anomalies": [],
                "company_info": company_info,
                "analysis_results": [],
                "next_node": "save_results"
            })
        return new_state
            
    except Exception as e:
        print(f"    ❌ 데이터 로딩 오류: {str(e)}")
        new_state = dict(state)
        new_state.update({
            "anomalies": [],
            "company_info": {},
            "analysis_results": [],
            "next_node": "save_results"
        })
        return new_state

def generate_search_query(state: NewsAnalysisState) -> NewsAnalysisState:
    """이상치 정보를 바탕으로 뉴스 검색 쿼리 생성"""
    
    query_prompt = ChatPromptTemplate.from_template("""
    다음 재무지표 이상치에 대한 뉴스 검색 쿼리를 생성해주세요.

    ## 회사 정보
    회사명: {company_name}
    
    ## 이상치 정보
    지표명: {metric_name}
    분기: {quarter}
    이상치 설명: {description}
    심각도: {severity}
    
    ## 재시도 정보
    현재 시도: {retry_count}회
    이전 쿼리: {previous_query}
    
    ## 요청사항
    - 한국어 뉴스 검색에 적합한 키워드로 구성
    - 회사명과 관련 재무지표를 포함
    - 재시도인 경우 다른 각도의 키워드 사용
    - 15단어 이내로 간결하게
    
    검색 쿼리만 출력해주세요.
    """)
    
    current_anomaly = state["current_anomaly"]
    company_info = state["company_info"]
    retry_count = state.get("retry_count", 0)
    
    company_name = company_info.get("company_name", company_info.get("기업명", ""))
    previous_query = state.get("search_query", "")
    
    try:
        chain = query_prompt | llm
        
        search_query = chain.invoke({
            "company_name": company_name,
            "metric_name": current_anomaly.get("metric_name", ""),
            "quarter": current_anomaly.get("quarter", ""),
            "description": current_anomaly.get("description", ""),
            "severity": current_anomaly.get("severity", ""),
            "retry_count": retry_count,
            "previous_query": previous_query
        }).content.strip()
        
        print(f"    🔍 검색 쿼리 생성: {search_query}")
        
        new_state = dict(state)
        new_state.update({
            "search_query": search_query,
            "next_node": "search_news"
        })
        return new_state
        
    except Exception as e:
        print(f"    ❌ 쿼리 생성 오류: {str(e)}")
        new_state = dict(state)
        new_state.update({
            "search_query": f"{company_name} 재무 변화",
            "next_node": "search_news"
        })
        return new_state

def search_news(state: NewsAnalysisState) -> NewsAnalysisState:
    """Tavily를 사용하여 뉴스 검색"""
    
    search_query = state["search_query"]
    
    try:
        search_tool = TavilySearch(max_results=3, days=90)
        
        print(f"    📰 뉴스 검색 중: {search_query}")
        
        # 뉴스 검색 실행
        raw_results = search_tool.search(query=search_query)
        
        # 결과 정리
        news_results = []
        if isinstance(raw_results, list):
            for result in raw_results:
                if isinstance(result, dict):
                    news_results.append({
                        "title": result.get("title", ""),
                        "url": result.get("url", ""),
                        "content": result.get("content", "")[:1000],
                        "published_date": result.get("published_date", ""),
                        "source": result.get("source", "")
                    })
        
        new_state = dict(state)
        new_state.update({
            "news_results": news_results,
            "next_node": "check_relevance_hybrid"
        })
        return new_state
        
    except Exception as e:
        print(f"    ❌ 뉴스 검색 오류: {str(e)}")
        new_state = dict(state)
        new_state.update({
            "news_results": [],
            "next_node": "check_relevance_hybrid"
        })
        return new_state

def check_relevance_hybrid(state: NewsAnalysisState) -> NewsAnalysisState:
    """하이브리드 방식으로 뉴스 관련성 체크 (코사인 유사도 + LLM)"""
    
    semantic_analysis_prompt = ChatPromptTemplate.from_template("""
    다음 뉴스와 재무지표 이상치 간의 의미적 연관성을 분석해주세요.

    ## 재무지표 이상치 정보
    회사명: {company_name}
    지표명: {metric_name}
    분기: {quarter}
    이상치 설명: {description}
    
    ## 뉴스 내용
    제목: {news_title}
    내용: {news_content}
    
    ## 분석 요청사항
    1. 인과관계 분석: 이 뉴스가 해당 재무지표에 직접적/간접적 영향을 미칠 수 있는가?
    2. 시간적 적합성: 뉴스 시점과 재무지표 측정 시점이 논리적으로 연결되는가?
    3. 사업영역 일치성: 뉴스 내용이 해당 회사의 주요 사업영역과 관련있는가?
    4. 재무영향도: 이 뉴스가 실제로 재무성과에 측정 가능한 영향을 미칠 수 있는가?
    
    다음 예시처럼 JSON 형식으로 답변해주세요.
                                                                
    {{
        "causal_relationship": {{
            "score": 8,
            "explanation": "뉴스에서 언급된 신제품 출시가 매출 증가에 직접적 영향"
        }},
        "temporal_relevance": {{
            "score": 7,
            "explanation": "뉴스 시점이 해당 분기와 일치함"
        }},
        "business_alignment": {{
            "score": 9,
            "explanation": "회사의 핵심 사업영역과 완전히 일치"
        }},
        "financial_impact": {{
            "score": 6,
            "explanation": "단기적으로는 영향 제한적이나 중장기적 영향 예상"
        }},
        "overall_semantic_score": 7.5,
        "reasoning": "종합적인 판단 근거"
    }}
    """)
    
    current_anomaly = state["current_anomaly"]
    company_info = state["company_info"]
    news_results = state.get("news_results", [])
    
    if not news_results:
        new_state = dict(state)
        new_state.update({
            "relevance_scores": [],
            "hybrid_relevance_data": {},
            "next_node": "rewrite_query_or_analyze"
        })
        return new_state
    
    company_name = company_info.get("company_name", company_info.get("기업명", ""))
    metric_name = current_anomaly.get("metric_name", "")
    quarter = current_anomaly.get("quarter", "")
    description = current_anomaly.get("description", "")
    
    # 참조 텍스트 생성 (코사인 유사도용)
    reference_texts = create_reference_texts(company_name, metric_name, quarter, description)
    
    # 하이브리드 분석 결과 저장
    hybrid_results = []
    final_scores = []
    
    try:
        for i, news in enumerate(news_results):
            news_title = news.get("title", "")
            news_content = news.get("content", "")
            news_text = f"{news_title} {news_content}"
            
            # 1. 코사인 유사도 계산
            cosine_score = calculate_cosine_similarity_score(news_text, reference_texts)
            
            # 2. LLM 의미 분석
            try:
                chain = semantic_analysis_prompt | llm | JsonOutputParser()
                
                semantic_analysis = chain.invoke({
                    "company_name": company_name,
                    "metric_name": metric_name,
                    "quarter": quarter,
                    "description": description,
                    "news_title": news_title,
                    "news_content": news_content
                })
                
                semantic_score = semantic_analysis.get("overall_semantic_score", 5.0)
                
            except Exception as e:
                print(f"    ⚠️ LLM 의미 분석 오류 (뉴스 {i+1}): {e}")
                semantic_analysis = {
                    "overall_semantic_score": 5.0,
                    "reasoning": f"분석 오류: {str(e)}"
                }
                semantic_score = 5.0
            
            # 3. 하이브리드 점수 계산 (코사인 30%, 의미 분석 70%)
            hybrid_score = (cosine_score * 0.3 + semantic_score * 0.7)
            hybrid_score = min(10.0, max(0.0, hybrid_score))
            
            # 결과 저장 - 모든 숫자를 Python 기본 타입으로 변환
            result = {
                "news_index": i,
                "cosine_similarity_score": float(round(cosine_score, 2)),
                "semantic_analysis_score": float(round(semantic_score, 2)),
                "hybrid_relevance_score": float(round(hybrid_score, 2)),
                "semantic_analysis": semantic_analysis,
                "news_info": {
                    "title": news_title,
                    "url": news.get("url", ""),
                    "published_date": news.get("published_date", "")
                }
            }
            
            hybrid_results.append(result)
            final_scores.append(float(hybrid_score))
        
        # 결과 정렬 (하이브리드 점수 기준)
        hybrid_results.sort(key=lambda x: x["hybrid_relevance_score"], reverse=True)
        
        # 통계 계산
        avg_hybrid_score = float(np.mean(final_scores)) if final_scores else 0.0
        max_hybrid_score = float(np.max(final_scores)) if final_scores else 0.0
        
        # 관련성 기준 (점수 6.0 이상을 관련있음으로 판단)
        relevant_threshold = 6.0
        has_relevant_news = any(score >= relevant_threshold for score in final_scores)
        
        # 하이브리드 관련성 데이터 구성
        hybrid_relevance_data = {
            "analysis_method": "hybrid_cosine_llm",
            "weights": {"cosine": 0.3, "semantic": 0.7},
            "results": hybrid_results,
            "statistics": {
                "total_news_count": len(news_results),
                "average_hybrid_score": round(avg_hybrid_score, 2),
                "max_hybrid_score": round(max_hybrid_score, 2),
                "relevant_news_count": len([s for s in final_scores if s >= relevant_threshold])
            }
        }
        
        print(f"    📊 하이브리드 관련성 점수: {avg_hybrid_score:.1f}/10 (최고: {max_hybrid_score:.1f})")
        print(f"    - 관련성 높은 뉴스: {len([s for s in final_scores if s >= relevant_threshold])}개")
        
        new_state = dict(state)
        new_state.update({
            "relevance_scores": final_scores,
            "hybrid_relevance_data": hybrid_relevance_data,
            "next_node": "rewrite_query_or_analyze" if not has_relevant_news else "analyze_news"
        })
        return new_state
        
    except Exception as e:
        print(f"    ❌ 하이브리드 관련성 체크 오류: {str(e)}")
        new_state = dict(state)
        new_state.update({
            "relevance_scores": [],
            "hybrid_relevance_data": {},
            "next_node": "rewrite_query_or_analyze"
        })
        return new_state

def rewrite_query_or_analyze(state: NewsAnalysisState) -> NewsAnalysisState:
    """관련성이 낮으면 쿼리를 다시 작성하거나 분석 진행"""
    
    retry_count = state.get("retry_count", 0)
    relevance_scores = state.get("relevance_scores", [])
    
    # 관련성이 높은 뉴스가 있는지 확인 (점수 6 이상)
    has_relevant = any(score >= 6 for score in relevance_scores) if relevance_scores else False
    
    new_state = dict(state)
    
    if not has_relevant and retry_count < 0: # 2로 수정
        # 재검색
        print(f"    🔄 관련성이 낮아 재검색 ({retry_count + 1}회차)")
        new_state.update({
            "retry_count": retry_count + 1,
            "next_node": "generate_search_query"
        })
    else:
        # 분석 진행 (재시도 한계 도달 또는 관련 뉴스 발견)
        new_state.update({
            "next_node": "analyze_news"
        })
    
    return new_state

def analyze_news(state: NewsAnalysisState) -> NewsAnalysisState:
    """뉴스를 분석하여 이상치 원인을 파악"""
    
    analysis_prompt = ChatPromptTemplate.from_template("""
    다음 뉴스들을 바탕으로 재무지표 이상치의 원인을 분석해주세요.

    ## 이상치 정보
    지표명: {metric_name}
    분기: {quarter}
    이상치 설명: {description}
    심각도: {severity}
    회사명: {company_name}

    ## 관련 뉴스 (하이브리드 관련성 점수순)
    {relevant_news}

    ## 하이브리드 분석 정보
    - 분석 방법: 코사인 유사도 (30%) + LLM 의미 분석 (70%)
    - 평균 관련성 점수: {avg_relevance_score}/10
    - 관련성 높은 뉴스: {relevant_count}개

    ## 분석 요청
    위 뉴스들을 바탕으로 이상치의 원인을 분석하고 다음 JSON 형식으로 답변해주세요:
    {{
        "primary_cause": "주요 원인 (한줄 요약)",
        "confidence_level": 8,
        "supporting_evidence": ["뉴스 근거1", "뉴스 근거2", "뉴스 근거3"],
        "detailed_explanation": "상세 분석 (3-4문장)",
        "news_sources": ["뉴스 URL1", "뉴스 URL2"],
        "impact_assessment": "영향 평가",
        "risk_level": "낮음/보통/높음",
        "relevance_quality": "하이브리드 분석을 통한 뉴스 관련성 품질 평가"
    }}
    """)
    
    current_anomaly = state["current_anomaly"]
    company_info = state["company_info"]
    news_results = state.get("news_results", [])
    relevance_scores = state.get("relevance_scores", [])
    hybrid_relevance_data = state.get("hybrid_relevance_data", {})
    
    # 하이브리드 관련성이 높은 뉴스만 선별 (점수 5 이상)
    relevant_news = []
    if hybrid_relevance_data.get("results"):
        for result in hybrid_relevance_data["results"]:
            if result["hybrid_relevance_score"] >= 5.0:
                news_idx = result["news_index"]
                if news_idx < len(news_results):
                    news = news_results[news_idx]
                    relevant_news.append({
                        **news,
                        "hybrid_score": result["hybrid_relevance_score"],
                        "cosine_score": result["cosine_similarity_score"],
                        "semantic_score": result["semantic_analysis_score"]
                    })
    
    # 관련 뉴스가 없으면 빈 결과 처리
    if not relevant_news:
        # 빈 뉴스에 대한 기본 처리
        relevant_news_text = "관련 뉴스를 찾을 수 없음"
        stats = {"average_hybrid_score": 0, "relevant_news_count": 0}
    else:
        # 뉴스를 텍스트로 변환
        news_text = []
        for i, news in enumerate(relevant_news):
            # 안전한 방식으로 hybrid_score 가져오기
            if isinstance(news, dict):
                hybrid_score = news.get("hybrid_score", "N/A")
                title = news.get("title", "")
                content = news.get("content", "")
                source = news.get("source", "")
                url = news.get("url", "")
            else:
                # news가 dict가 아닌 경우 (예상치 못한 상황)
                hybrid_score = "N/A"
                title = str(news)
                content = ""
                source = ""
                url = ""
            
            text = f"[뉴스 {i+1}] (하이브리드 점수: {hybrid_score})\n"
            text += f"제목: {title}\n"
            text += f"내용: {content}\n"
            text += f"출처: {source}\n"
            text += f"URL: {url}\n"
            news_text.append(text)
        
        relevant_news_text = "\n---\n".join(news_text)
        stats = hybrid_relevance_data.get("statistics", {"average_hybrid_score": 0, "relevant_news_count": 0})
    
    # 통계 정보
    avg_relevance = stats.get("average_hybrid_score", 0)
    relevant_count = stats.get("relevant_news_count", 0)
    
    try:
        chain = analysis_prompt | llm | JsonOutputParser()
        
        analysis_result = chain.invoke({
            "metric_name": current_anomaly.get("metric_name", ""),
            "quarter": current_anomaly.get("quarter", ""),
            "description": current_anomaly.get("description", ""),
            "severity": current_anomaly.get("severity", ""),
            "company_name": company_info.get("company_name", company_info.get("기업명", "")),
            "relevant_news": relevant_news_text,
            "avg_relevance_score": avg_relevance,
            "relevant_count": relevant_count
        })
        
        # 분석 결과 저장
        result_data = {
            "anomaly_info": current_anomaly,
            "analysis": analysis_result,
            "news_evidence": relevant_news,
            "hybrid_relevance_analysis": hybrid_relevance_data,
            "search_info": {
                "final_query": state.get("search_query", ""),
                "retry_count": state.get("retry_count", 0),
                "total_news_found": len(news_results),
                "relevant_news_count": len(relevant_news)
            },
            "timestamp": datetime.now().isoformat()
        }
        
        current_results = state.get("analysis_results", [])
        current_results.append(result_data)
        
        print(f"    ✅ 하이브리드 분석 완료: {current_anomaly.get('metric_name', 'Unknown')}")
        
        new_state = dict(state)
        new_state.update({
            "analysis_results": current_results,
            "next_node": "check_remaining_anomalies"
        })
        return new_state
        
    except Exception as e:
        print(f"    ❌ 뉴스 분석 오류: {str(e)}")
        
        # 오류 결과 저장
        error_result = {
            "anomaly_info": current_anomaly,
            "analysis": {"error": f"분석 오류: {str(e)}"},
            "news_evidence": relevant_news,
            "hybrid_relevance_analysis": hybrid_relevance_data,
            "timestamp": datetime.now().isoformat()
        }
        
        current_results = state.get("analysis_results", [])
        current_results.append(error_result)
        
        new_state = dict(state)
        new_state.update({
            "analysis_results": current_results,
            "next_node": "check_remaining_anomalies"
        })
        return new_state

def check_remaining_anomalies(state: NewsAnalysisState) -> NewsAnalysisState:
    """분석할 이상치가 더 있는지 확인"""
    
    anomalies = state.get("anomalies", [])
    current_index = state.get("current_anomaly_index", 0)
    next_index = current_index + 1
    
    new_state = dict(state)
    
    if next_index < len(anomalies):
        # 다음 이상치로 이동
        print(f"    ➡️ 다음 이상치로 이동 ({next_index + 1}/{len(anomalies)})")
        new_state.update({
            "current_anomaly": anomalies[next_index],
            "current_anomaly_index": next_index,
            "retry_count": 0,  # 초기화
            "news_results": [],  # 초기화
            "relevance_scores": [],  # 초기화
            "hybrid_relevance_data": {},  # 초기화
            "next_node": "generate_search_query"
        })
    else:
        # 모든 이상치 분석 완료
        print(f"    ✅ 모든 이상치 분석 완료 ({len(anomalies)}개)")
        new_state.update({
            "next_node": "save_results"
        })
    
    return new_state

def save_results(state: NewsAnalysisState) -> NewsAnalysisState:
    """분석 결과를 JSON 파일로 저장"""
    
    result_dir = state["result_dir"]
    analysis_results = state.get("analysis_results", [])
    company_info = state.get("company_info", {})
    
    # 결과 데이터 구성
    final_results = {
        "success": True,
        "analysis_method": "hybrid_relevance_check",
        "methodology": {
            "cosine_similarity_weight": 0.3,
            "llm_semantic_weight": 0.7,
            "relevance_threshold": 6.0,
            "embedding_model": "paraphrase-multilingual-MiniLM-L12-v2"
        },
        "company_info": company_info,
        "analysis_summary": {
            "total_anomalies": len(analysis_results),
            "successful_analyses": len([r for r in analysis_results if "error" not in r.get("analysis", {})]),
            "analysis_timestamp": datetime.now().isoformat()
        },
        "anomaly_news_analyses": analysis_results
    }
    
    # 파일 저장
    output_file = os.path.join(result_dir, "anomaly_news_analysis.json")
    
    try:
        # 디렉토리가 없으면 생성
        os.makedirs(result_dir, exist_ok=True)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(final_results, f, ensure_ascii=False, indent=2)
        
        print(f"    💾 하이브리드 분석 결과 저장 완료: {output_file}")
        
        new_state = dict(state)
        new_state.update({
            "next_node": "END"
        })
        return new_state
        
    except Exception as e:
        print(f"    ❌ 파일 저장 오류: {str(e)}")
        new_state = dict(state)
        new_state.update({
            "next_node": "END"
        })
        return new_state

def route_next_node(state: NewsAnalysisState) -> str:
    """다음 노드를 결정하는 라우팅 함수"""
    return state.get("next_node", "END")

def create_news_analysis_graph():
    """뉴스 분석 그래프를 생성"""
    
    workflow = StateGraph(NewsAnalysisState)
    
    # 노드 추가
    workflow.add_node("load_financial_data", load_financial_data)
    workflow.add_node("generate_search_query", generate_search_query)
    workflow.add_node("search_news", search_news)
    workflow.add_node("check_relevance_hybrid", check_relevance_hybrid)
    workflow.add_node("rewrite_query_or_analyze", rewrite_query_or_analyze)
    workflow.add_node("analyze_news", analyze_news)
    workflow.add_node("check_remaining_anomalies", check_remaining_anomalies)
    workflow.add_node("save_results", save_results)
    
    # 시작점 설정
    workflow.set_entry_point("load_financial_data")
    
    # 엣지 추가
    workflow.add_conditional_edges(
        "load_financial_data",
        route_next_node,
        {
            "generate_search_query": "generate_search_query",
            "save_results": "save_results"
        }
    )
    
    workflow.add_edge("generate_search_query", "search_news")
    workflow.add_edge("search_news", "check_relevance_hybrid")
    
    workflow.add_conditional_edges(
        "check_relevance_hybrid",
        route_next_node,
        {
            "rewrite_query_or_analyze": "rewrite_query_or_analyze",
            "analyze_news": "analyze_news"
        }
    )
    
    workflow.add_conditional_edges(
        "rewrite_query_or_analyze",
        route_next_node,
        {
            "generate_search_query": "generate_search_query",
            "analyze_news": "analyze_news"
        }
    )
    
    workflow.add_edge("analyze_news", "check_remaining_anomalies")
    
    workflow.add_conditional_edges(
        "check_remaining_anomalies",
        route_next_node,
        {
            "generate_search_query": "generate_search_query",
            "save_results": "save_results"
        }
    )
    
    workflow.add_edge("save_results", END)
    
    # 메모리 설정
    memory = MemorySaver()
    
    # 그래프 컴파일
    app = workflow.compile(checkpointer=memory)
    
    return app

def run_anomaly_news_analysis(result_dir: str) -> Dict[str, Any]:
    """
    재무지표 이상치에 대한 뉴스 근거 분석 실행 (하이브리드 관련성 체크)
    
    Args:
        result_dir: 분석 결과가 저장된 디렉토리 경로
        
    Returns:
        Dict: 분석 결과
    """
    
    print("📰 재무지표 이상치 뉴스 근거 분석 시작... (하이브리드 관련성 체크)")
    print("    - 분석 방법: 코사인 유사도 (30%) + LLM 의미 분석 (70%)")
    
    # 임베딩 모델 미리 로드
    try:
        print("    🔧 임베딩 모델 로딩 중...")
        get_embedding_model()
        print("    ✅ 임베딩 모델 로딩 완료")
    except Exception as e:
        print(f"    ⚠️ 임베딩 모델 로딩 실패: {e}")
        print("    🔍 기본 LLM 분석으로 진행...")
    
    # 그래프 생성
    app = create_news_analysis_graph()
    
    # 초기 상태 설정
    initial_state = {
        "result_dir": result_dir,
        "messages": [],
        "analysis_results": [],
        "current_anomaly_index": 0,
        "retry_count": 0,
        "next_node": "load_financial_data"
    }
    
    # 그래프 실행
    config = {
        "configurable": {
            "thread_id": f"news_analysis_hybrid_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        },
        "recursion_limit": 100
    }
    
    try:
        print("    🚀 LangGraph 실행 시작...")
        
        # 그래프 실행 with 단계별 디버깅
        final_state = None
        step_count = 0
        max_steps = 50
        
        for event in app.stream(initial_state, config):
            step_count += 1
            node_name = list(event.keys())[0]
            node_data = event[node_name]
            
            print(f"    🔍 단계 {step_count}: {node_name}")
            
            # 다음 노드 정보 출력 (디버깅용)
            next_node = node_data.get("next_node", "END")
            if next_node != "END":
                print(f"        → 다음: {next_node}")
            
            # 무한 루프 방지
            if step_count >= max_steps:
                print(f"    ⚠️ 최대 단계 수({max_steps}) 도달, 강제 종료")
                break
                
            final_state = node_data
            
            # END 노드 도달 시 종료
            if next_node == "END":
                print("    ✅ 분석 완료 (END 노드 도달)")
                break
        
        if final_state is None:
            print("    ⚠️ final_state가 None, app.get_state()로 상태 조회")
            final_state = app.get_state(config).values
        
        analysis_results = final_state.get("analysis_results", [])
        
        # 하이브리드 분석 통계 계산
        total_news_analyzed = 0
        avg_hybrid_scores = []
        
        for result in analysis_results:
            hybrid_data = result.get("hybrid_relevance_analysis", {})
            stats = hybrid_data.get("statistics", {})
            
            if stats:
                total_news_analyzed += stats.get("total_news_count", 0)
                avg_score = stats.get("average_hybrid_score", 0)
                if avg_score > 0:
                    avg_hybrid_scores.append(float(avg_score))
        
        overall_avg_score = float(np.mean(avg_hybrid_scores)) if avg_hybrid_scores else 0.0
        
        return {
            "success": True,
            "analysis_method": "hybrid_relevance_check",
            "total_anomalies_analyzed": len(analysis_results),
            "total_news_analyzed": total_news_analyzed,
            "average_relevance_score": round(overall_avg_score, 2),
            "analysis_results": analysis_results,
            "output_file": os.path.join(result_dir, "anomaly_news_analysis.json")
        }
        
    except Exception as e:
        print(f"❌ 하이브리드 뉴스 분석 실행 오류: {str(e)}")
        return {
            "success": False,
            "error": str(e),
            "total_anomalies_analyzed": 0,
            "analysis_method": "hybrid_relevance_check"
        }

if __name__ == "__main__":
    # 테스트 실행
    test_result_dir = "analysis_results/20250812_123456"  # 예시 경로
    
    result = run_anomaly_news_analysis(test_result_dir)

    print("\n=== 하이브리드 뉴스 분석 결과 ===")
    print(f"성공 여부: {result['success']}")
    print(f"분석 방법: {result.get('analysis_method', 'N/A')}")
    print(f"분석된 이상치 수: {result['total_anomalies_analyzed']}")
    
    if result['success']:
        print(f"총 분석된 뉴스 수: {result.get('total_news_analyzed', 0)}")
        print(f"평균 관련성 점수: {result.get('average_relevance_score', 0)}/10")
        print(f"결과 파일: {result['output_file']}")
    
    if result.get('error'):
        print(f"오류: {result['error']}")