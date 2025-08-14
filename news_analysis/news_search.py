import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Tuple
from langchain_teddynote.tools.tavily import TavilySearch
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from pydantic import BaseModel, Field


class NewsRiskItem(BaseModel):
    """뉴스 위험 항목 모델"""
    title: str = Field(description="뉴스 제목")
    content_summary: str = Field(description="뉴스 내용 요약 (1-2문장)")
    url: str = Field(description="뉴스 URL")
    risk_category: str = Field(description="위험 카테고리 (재무위험/비재무위험/복합위험)")
    risk_level: int = Field(description="위험도 (1-10, 10이 가장 위험)")
    evidence_basis: str = Field(description="위험 판단 근거")
    anomaly_connection: str = Field(description="탐지된 이상치와의 연관성")
    published_date: str = Field(description="발행일", default="")


class CreditAssessment(BaseModel):
    """신용등급 평가 모델"""
    current_grade: str = Field(description="현재 추정 신용등급")
    predicted_grade: str = Field(description="예상 신용등급")
    change_probability: float = Field(description="등급 변경 확률 (0-1)")
    change_direction: str = Field(description="변경 방향 (상승/하락/유지)")
    reasoning: str = Field(description="등급 평가 근거")


class DailyRiskSummary(BaseModel):
    """일일 위험 요약 모델"""
    analysis_date: str = Field(description="분석 날짜")
    company_name: str = Field(description="기업명")
    total_risk_news: int = Field(description="위험 뉴스 총 개수")
    high_risk_count: int = Field(description="고위험 뉴스 개수 (위험도 7-10)")
    medium_risk_count: int = Field(description="중위험 뉴스 개수 (위험도 4-6)")
    key_risk_summary: str = Field(description="주요 위험 요약 (2-3문장)")
    risk_news_items: List[NewsRiskItem] = Field(description="위험 뉴스 목록")
    credit_assessment: CreditAssessment = Field(description="신용등급 평가")


class NewsRelevance(BaseModel):
    """뉴스 관련성 판단 모델"""
    is_relevant: bool = Field(description="뉴스가 신용위험 분석과 관련이 있는지 여부")
    reason: str = Field(description="관련성 판단에 대한 간략한 근거")


class CreditRiskNewsAnalyzer:
    def __init__(self, max_search_results=10, openai_api_key=None, tavily_api_key=None):
        """
        신용위험 징후 뉴스 분석 시스템 초기화
        
        Args:
            max_search_results (int): 뉴스 검색 최대 결과 수
            openai_api_key (str, optional): OpenAI API 키
            tavily_api_key (str, optional): Tavily API 키
        """
        # API 키 설정
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        if tavily_api_key:
            os.environ["TAVILY_API_KEY"] = tavily_api_key
        
        self.tavily_search = TavilySearch(max_results=max_search_results)
        self.llm = ChatOpenAI(model="gpt-4o", temperature=0)
        
        # 신용위험 징후 기준 정의
        self.credit_risk_criteria = {
            "재무 위험 징후": {
                "수익성 악화": ["매출 감소", "이익률 하락", "손실 확대", "영업적자"],
                "재무구조 악화": ["부채 증가", "차입금 증가", "신용등급 하락", "금융비용 증가"],
                "유동성 위기": ["현금 부족", "유동성 경색", "자금조달 어려움", "운전자금 부족"],
                "투자 능력 저하": ["투자 축소", "설비투자 감소", "연구개발비 삭감", "신규투자 중단"]
            },
            "비재무 위험 징후": {
                "거버넌스 이슈": ["지배구조 문제", "경영진 교체", "내부 갈등", "주주 분쟁"],
                "규제/법적 리스크": ["규제 위반", "과징금", "소송", "법적 분쟁", "제재 조치"],
                "운영 리스크": ["생산 중단", "공급망 차질", "품질 문제", "사고 발생"],
                "ESG 리스크": ["환경 오염", "사회적 물의", "노사 갈등", "평판 악화"],
                "시장 리스크": ["시장점유율 하락", "경쟁 심화", "고객 이탈", "브랜드 가치 하락"]
            }
        }

    def load_analysis_results(self, result_dir: str) -> Dict[str, Any]:
        """
        분석 결과 디렉토리에서 재무분석과 비재무분석 결과를 로드
        
        Args:
            result_dir (str): 분석 결과 디렉토리 경로
            
        Returns:
            Dict: 통합된 분석 결과
        """
        analysis_data = {
            "company_name": "",
            "current_credit_grade": "A",  # 기본값
            "financial_anomalies": [],
            "non_financial_anomalies": [],
            "financial_characteristics": {},
            "business_context": {}
        }
        
        # 1. 재무 이상치 결과 로드 (financial_anomalies.json)
        financial_anomalies_file = os.path.join(result_dir, "financial_anomalies.json")
        if os.path.exists(financial_anomalies_file):
            try:
                with open(financial_anomalies_file, 'r', encoding='utf-8') as f:
                    raw_financial_anomalies = json.load(f)
                
                # 딕셔너리 형태를 리스트로 변환
                financial_anomalies = []
                if isinstance(raw_financial_anomalies, dict):
                    for metric_name, details in raw_financial_anomalies.items():
                        if isinstance(details, dict):
                            anomaly_item = {
                                "indicator": details.get("metric_name", metric_name),
                                "severity": details.get("severity", "medium").lower(),
                                "description": details.get("description", ""),
                                "type": details.get("type", ""),
                                "quarter": details.get("quarter", ""),
                                "source": details.get("source", ""),
                                "metric_name": details.get("metric_name", metric_name)
                            }
                            financial_anomalies.append(anomaly_item)
                        else:
                            # fallback: 간단한 형태로 변환
                            anomaly_item = {
                                "indicator": metric_name,
                                "severity": "medium",
                                "description": str(details),
                                "type": "unknown",
                                "quarter": "Latest",
                                "source": "unknown",
                                "metric_name": metric_name
                            }
                            financial_anomalies.append(anomaly_item)
                elif isinstance(raw_financial_anomalies, list):
                    # 이미 리스트 형태인 경우
                    for anomaly in raw_financial_anomalies:
                        financial_anomalies.append({
                            "indicator": anomaly.get("metric_name", ""),
                            "severity": anomaly.get("severity", "medium").lower(),
                            "description": anomaly.get("description", ""),
                            "type": anomaly.get("type", ""),
                            "quarter": anomaly.get("quarter", ""),
                            "source": anomaly.get("source", "")
                        })
                
                analysis_data["financial_anomalies"] = financial_anomalies
                print(f"✅ 재무 이상치 로드 완료: {len(financial_anomalies)}개")
                
            except Exception as e:
                print(f"❌ 재무 이상치 로드 오류: {str(e)}")
        
        # 2. 기업 정보 추출
        financial_analysis_file = os.path.join(result_dir, "financial_analysis.json")
        if os.path.exists(financial_analysis_file):
            try:
                with open(financial_analysis_file, 'r', encoding='utf-8') as f:
                    financial_data = json.load(f)
                
                # 기업 정보 추출
                company_info = financial_data.get("기업_정보", {})
                analysis_data["company_name"] = company_info.get("기업명", "Unknown")
                analysis_data["current_credit_grade"] = company_info.get("Current_credit_grade", "A")
                
                # 비즈니스 컨텍스트 설정
                analysis_data["business_context"] = {
                    "company_info": company_info,
                    "industry": company_info.get("업종", ""),
                    "business_area": company_info.get("제품군", "")
                }
                
                print(f"✅ 기업 정보 로드 완료: {analysis_data['company_name']}")
                
            except Exception as e:
                print(f"❌ 기업 정보 로드 오류: {str(e)}")
                # 기본값 설정
                analysis_data["company_name"] = "삼성전자"
        
        # 3. 비재무분석 결과 로드 (non_financial_reasoning.json)
        nfr_file = os.path.join(result_dir, "non_financial_reasoning.json")
        if os.path.exists(nfr_file):
            try:
                with open(nfr_file, 'r', encoding='utf-8') as f:
                    nfr_data = json.load(f)
                
                # JSON 구조 확인: results 배열이 있는지 체크
                if nfr_data.get("success") and "results" in nfr_data:
                    corp_code = nfr_data.get("corp_code", "")
                    quarter = nfr_data.get("quarter", "")
                    
                    # results 배열의 각 항목을 처리
                    for result in nfr_data["results"]:
                        metric = result.get("metric", "")
                        anomaly_text = result.get("anomaly_text", "")
                        explanation = result.get("explanation_ko", "")
                        confidence = result.get("confidence", 0)
                        
                        # 심각도 계산
                        severity_text = result.get("severity", "Medium")
                        severity = severity_text.lower() if severity_text in ["High", "Medium", "Low"] else "medium"
                        
                        # 분석 데이터에 추가
                        analysis_data["non_financial_anomalies"].append({
                            "indicator": metric,
                            "pillar": "Unknown",
                            "score": 0,
                            "grade_label": anomaly_text,
                            "severity": severity,
                            "description": explanation,
                            "confidence": confidence,
                            "quarter": quarter,
                            "indicator_id": metric
                        })
                    
                    print(f"✅ 비재무 분석 결과 로드 완료: {len(analysis_data['non_financial_anomalies'])}개 항목")
                    
                else:
                    print(f"❌ 비재무분석 JSON 구조가 예상과 다름: success={nfr_data.get('success')}")
                    
            except Exception as e:
                print(f"❌ 비재무분석 로드 오류: {str(e)}")
        
        # 4. 재무 특성 분석
        analysis_data["financial_characteristics"] = self._analyze_financial_characteristics(analysis_data["financial_anomalies"])
        
        return analysis_data

    def _analyze_financial_characteristics(self, anomalies: List[Dict]) -> Dict[str, Any]:
        """재무 이상치에서 특성을 분석하는 헬퍼 메서드"""
        characteristics = {
            "severity_distribution": {"high": 0, "medium": 0, "low": 0},
            "type_distribution": {"peer_comparison": 0, "time_series": 0, "other": 0},
            "indicator_types": {},
            "quarters_affected": set(),
            "risk_indicators": [],
            "total_anomalies": len(anomalies)
        }
        
        for anomaly in anomalies:
            # 심각도 분포
            severity = anomaly.get("severity", "medium").lower()
            if severity in characteristics["severity_distribution"]:
                characteristics["severity_distribution"][severity] += 1
            
            # 유형 분포
            anomaly_type = anomaly.get("type", "other")
            if anomaly_type in characteristics["type_distribution"]:
                characteristics["type_distribution"][anomaly_type] += 1
            else:
                characteristics["type_distribution"]["other"] += 1
            
            # 지표 유형 분포
            indicator = anomaly.get("indicator", "")
            if indicator:
                characteristics["indicator_types"][indicator] = characteristics["indicator_types"].get(indicator, 0) + 1
            
            # 영향받은 분기
            quarter = anomaly.get("quarter", "")
            if quarter:
                characteristics["quarters_affected"].add(quarter)
            
            # 고위험 지표
            if severity == "high":
                characteristics["risk_indicators"].append(indicator)
        
        # set을 list로 변환 (JSON 직렬화를 위해)
        characteristics["quarters_affected"] = list(characteristics["quarters_affected"])
        
        return characteristics
    
    def _generate_augmented_queries(self, company_name) -> List[str]:
        """
        회사 이름과 다양한 위험 키워드를 조합하여 검색 쿼리 목록을 생성합니다.
        """
        base_query = f'"{company_name}"' # 정확한 회사명 검색을 위해 큰따옴표 사용

        # 재무적 위험 키워드
        financial_risk_keywords = [
            "유동성 위기", "자금난", "부채 과다", "영업손실", "적자 지속",
            "신용등급 하향", "채무 불이행", "워크아웃", "법정관리",
            "감사의견 거절", "자본잠식", "어닝쇼크"
        ]

        # 비재무적 위험 키워드
        non_financial_risk_keywords = [
            "횡령", "배임", "분식회계", "소송", "공정위 조사", "압수수색",
            "영업정지", "사업 중단", "경영진 사퇴", "구조조정"
        ]

        # 기본 쿼리 추가
        queries = [f'{base_query} 신용위험', f'{base_query} 부도 가능성']

        # 키워드 조합 쿼리 추가
        for keyword in financial_risk_keywords:
            queries.append(f'{base_query} {keyword}')

        for keyword in non_financial_risk_keywords:
            queries.append(f'{base_query} {keyword}')

        return list(set(queries)) # 중복 쿼리 제거

    def search_targeted_news(self, analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        분석된 이상치를 바탕으로 어제와 오늘의 뉴스를 타겟팅하여 검색
        
        Args:
            analysis_data: 재무/비재무 분석 결과
            
        Returns:
            List[Dict]: 검색된 뉴스 리스트
        """
        company_name = analysis_data.get("company_name", "삼성전자")
        financial_anomalies = analysis_data.get("financial_anomalies", [])
        nfr_anomalies = analysis_data.get("non_financial_anomalies", [])
        
        # 검색 키워드 생성


        print("🔍 증강된 쿼리를 사용하여 뉴스 검색을 시작합니다...")
        search_keywords = self._generate_augmented_queries(company_name)
        print(f"  - 생성된 쿼리 수: {len(search_keywords)}개")
        
        search_keywords.append( f"{company_name} 뉴스")

        seen_urls = set()
        all_news = []

        for keyword in search_keywords:
            try:

                results = self.tavily_search.search(
                    query=keyword,
                    search_depth="advanced",
                    # topic="news",  # <-- 이 부분을 추가하여 뉴스로 검색 범위를 한정합니다.
                    include_answer=False,
                    include_raw_content=False,
                    max_results=5,
                    # days=2  # 날짜 범위 필터: 어제부터 오늘까지
                )
                
                if results:
                    if isinstance(results, list):
                        for result in results:
                            news_item = self._process_search_result_clean(result, keyword)
                            if news_item:
                                all_news.append(news_item)
                    else:
                        news_item = self._process_search_result_clean(results, keyword)
                        if news_item:
                            all_news.append(news_item)
                            
            except Exception as e:
                print(f"   - 뉴스 검색 오류 ({keyword}): {str(e)}")
                continue
        
        # 중복 제거 (URL 기준)
        seen_urls = set()
        unique_news = []
        for news in all_news:
            url = news.get("url", "")
            if url and url not in seen_urls:
                seen_urls.add(url)
                unique_news.append(news)
        
        return unique_news
    
    def _pre_filter_relevant_news(self, news_list: List[Dict[str, Any]], analysis_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        LLM을 사용하여 검색된 뉴스 목록을 사전 필터링하여 신용위험과 관련 있는 뉴스만 선별합니다.

        Args:
            news_list (List[Dict[str, Any]]): 검색된 뉴스 목록
            analysis_data (Dict[str, Any]): 분석 데이터

        Returns:
            List[Dict[str, Any]]: 필터링된 관련 뉴스 목록
        """
        if not news_list:
            return []

        print("   - 뉴스 사전 필터링 시작...")
        
        prompt_template = ChatPromptTemplate.from_messages([
            ("system", """
You are a financial analyst's assistant. Your task is to quickly determine if a news article is potentially relevant for a company's credit risk assessment.
You will be given the company's name, a summary of detected financial and non-financial anomalies, and general credit risk definitions.
Based on this context, evaluate the provided news article. If the news article does not contain anomalies, but you determine it is related to the detected financial or non-financial anomalies, evaluate it as an anomaly.
Respond ONLY with a JSON object with two keys: "is_relevant" (boolean) and "reason" (a brief one-sentence explanation).
"""),
            ("human", """
**Context for Assessment**
- Company Name: {company_name}
- Detected Financial Anomalies Summary: {financial_anomalies_summary}
- Detected Non-Financial Anomalies Summary: {non_financial_anomalies_summary}
- General Credit Risk Criteria: {credit_risk_criteria}D

**News Article to Evaluate**
- Title: {news_title}
- Content: {news_content}

Is this news article potentially relevant for a credit risk assessment?
Provide your answer in JSON format.
""")
        ])

        relevance_chain = prompt_template | self.llm | JsonOutputParser(pydantic_object=NewsRelevance)

        relevant_news = []
        # 프롬프트에 사용할 컨텍스트 요약 정보 생성
        financial_anomalies = analysis_data.get("financial_anomalies", [])
        nfr_anomalies = analysis_data.get("non_financial_anomalies", [])

        financial_summary = f"{len(financial_anomalies)} financial anomalies detected. High severity indicators include: {', '.join([a['indicator'] for a in financial_anomalies if a.get('severity') == 'high'][:3])}"
        nfr_summary = f"{len(nfr_anomalies)} non-financial anomalies detected."
        risk_criteria_summary = "Keywords include: revenue decline, debt increase, legal disputes, governance issues, market share loss."

        for news in news_list:
            try:
                result = relevance_chain.invoke({
                    "company_name": analysis_data.get("company_name"),
                    "financial_anomalies_summary": financial_summary,
                    "non_financial_anomalies_summary": nfr_summary,
                    "credit_risk_criteria": risk_criteria_summary,
                    "news_title": news.get("title"),
                    "news_content": news.get("content", "")[:800]  # 빠른 판단을 위해 내용 길이 제한
                })

                if isinstance(result, dict) and result.get("is_relevant"):
                    print(f"      - ✅ RELEVANT: {news.get('title')}")
                    relevant_news.append(news)
                else:
                    print(f"      - ❌ IRRELEVANT: {news.get('title')}")

            except Exception as e:
                print(f"      - ⚠️ Pre-filtering error for news '{news.get('title')}': {e}")
                continue
                
        print(f"   - 사전 필터링 완료: {len(relevant_news)} / {len(news_list)} 뉴스 선별")
        return relevant_news

    def _process_search_result_clean(self, result: Any, keyword: str) -> Dict[str, Any]:
        """
        검색 결과를 깨끗하게 정리하여 표준 형식으로 변환
        
        Args:
            result: 검색 결과 (dict 또는 str)
            keyword: 검색 키워드
            
        Returns:
            Dict: 정리된 뉴스 아이템
        """
        try:
            if isinstance(result, dict):
                # 제목 정리
                title = result.get('title', '제목 없음')
                title = self._clean_title(title)
                
                # 내용 정리
                content = result.get('content', result.get('snippet', ''))
                content = self._clean_content(content)
                
                # URL 검증
                url = result.get('url', '')
                
                # 내용이 너무 짧으면 스킵
                if len(content) < 30:
                    return None
                
                return {
                    'title': title,
                    'url': url,
                    'content': content,
                    'published_date': result.get('published_date', result.get('date', '')),
                    'search_keyword': keyword,
                    'search_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
            else:
                return None
                
        except Exception as e:
            print(f"결과 처리 오류: {str(e)}")
            return None
    
    def _clean_title(self, title: str) -> str:
        """뉴스 제목 정리"""
        if not title:
            return "제목 없음"
        
        import re
        title = re.sub(r'<[^>]+>', '', title)
        title = title.replace('\n', ' ').replace('\t', ' ')
        title = re.sub(r'\s+', ' ', title).strip()
        
        if len(title) > 100:
            title = title[:100] + "..."
        
        return title
    
    def _clean_content(self, content: str) -> str:
        """뉴스 내용 정리"""
        if not content:
            return "내용 없음"
        
        import re
        
        content = re.sub(r'<[^>]+>', '', content)
        content = re.sub(r'', '', content, flags=re.DOTALL)
        content = re.sub(r'javascript:[^"\']*', '', content)
        content = re.sub(r'https?://[^\s]+', '', content)
        
        unwanted_patterns = [
            r'이미지.*?jpg|png|gif',
            r'검색.*?하기',
            r'페이지.*?링크',
            r'Copyright.*?\d{4}',
            r'사업자등록번호.*?\d',
        ]
        
        for pattern in unwanted_patterns:
            content = re.sub(pattern, '', content, flags=re.IGNORECASE)
        
        content = content.replace('\n', ' ').replace('\t', ' ')
        content = re.sub(r'\s+', ' ', content).strip()
        
        # 너무 긴 내용은 자르기
        if len(content) > 1000:
            content = content[:1000] + "..."
        
        return content

    def create_enhanced_risk_assessment_prompt(self) -> ChatPromptTemplate:
        """
        이상치 정보를 반영한 향상된 신용위험 판단 프롬프트 생성
        
        Returns:
            ChatPromptTemplate: 향상된 위험 평가 프롬프트
        """
        system_message = """
당신은 신용위험 분석 전문가입니다. 기업의 재무/비재무 이상치 분석 결과를 고려하여 뉴스가 신용위험 징후인지 판단하고, 구조화된 결과를 제공해주세요.

**중요한 판단 원칙**:
1. **선별된 뉴스 분석**: 입력된 뉴스는 이미 이상 징후 및 신용위험 기준에 따라 1차 선별된 것입니다. 이 뉴스들이 왜, 그리고 얼마나 위험한지 심층적으로 분석하세요.
2. **명확한 근거 기반 판단**: 각 뉴스에 대해 구체적이고 설명 가능한 근거를 제시해야 합니다.
3. **이상치와의 연관성 중시**: 탐지된 재무/비재무 이상치와 뉴스의 연관성을 중점적으로 분석하세요.
4. **복합적 위험 평가**: 뉴스 단독으로는 정상이지만 이상치와 함께 보면 위험한 경우를 식별하세요.
5. **정량적 위험도 평가**: 위험도를 1-10점으로 수치화하여 평가하세요.

**복합 위험 판단 예시**:
- 뉴스: "기업이 새로운 사업을 시작"
- 재무 이상치: "부채비율이 높음"
- 판단: 부채비율이 높은 상황에서 신규 사업 투자는 재무 부담 가중으로 위험 (위험도 7점)

**탐지된 이상치**:
재무 이상치: {financial_anomalies_summary}
비재무 이상치: {non_financial_anomalies_summary}
현재 신용등급: {current_credit_grade}

**신용위험 징후 기준**:
{credit_risk_criteria}

다음 JSON 형식으로 응답해주세요:
{{
    "analysis_date": {analysis_date},
    "company_name": "기업명",
    "total_risk_news": 위험뉴스수,
    "high_risk_count": 고위험뉴스수,
    "medium_risk_count": 중위험뉴스수,
    "key_risk_summary": "주요 위험 요약 (2-3문장)",
    "risk_news_items": [
        {{
            "title": "뉴스 제목",
            "content_summary": "뉴스 내용 요약 (1-2문장)",
            "url": "뉴스 URL",
            "risk_category": "재무위험/비재무위험/복합위험",
            "risk_level": 위험도(1-10),
            "evidence_basis": "위험 판단 근거 (구체적이고 명확하게)",
            "anomaly_connection": "탐지된 이상치와의 연관성",
            "published_date": "발행일"
        }}
    ],
    "credit_assessment": {{
        "current_grade": "현재 추정 신용등급",
        "predicted_grade": "예상 신용등급",
        "change_probability": 변경확률(0-1),
        "change_direction": "상승/하락/유지",
        "reasoning": "등급 평가 근거"
    }}
}}

**위험도 평가 기준**:
- 9-10점: 심각한 위험 (신용등급 즉시 하락 요인)
- 7-8점: 높은 위험 (단기내 신용등급에 부정적 영향)
- 5-6점: 중간 위험 (지속 모니터링 필요)
- 3-4점: 낮은 위험 (잠재적 우려사항)
- 1-2점: 미미한 위험 (일반적 비즈니스 활동)

**중요**: 위험도 5점 이상인 뉴스만 risk_news_items에 포함하세요. 위험 뉴스가 하나도 없는 경우 risk_news_items를 빈 리스트로 반환하세요.
"""
        
        human_message = """
## 탐지된 이상치 정보

### 재무 이상치
{financial_anomalies_detail}

### 비재무 이상치  
{non_financial_anomalies_detail}

### 기업 재무 특성
{financial_characteristics}

## 분석 대상 뉴스 (1차 선별됨)
{news_data}

위 이상치 분석 결과를 바탕으로 신용위험 징후 분석을 수행하고 구조화된 JSON 결과를 제공해주세요.
"""
        
        return ChatPromptTemplate.from_messages([
            ("system", system_message),
            ("human", human_message)
        ])
    
    def assess_credit_risk_with_anomalies(self, analysis_data: Dict[str, Any], news_list: List[Dict[str, Any]]) -> DailyRiskSummary:
        """
        이상치 정보를 반영하여 뉴스의 신용위험 징후 여부 판단
        
        Args:
            analysis_data (Dict): 재무/비재무 분석 결과
            news_list (List[Dict]): **사전 필터링된** 뉴스 리스트
            
        Returns:
            DailyRiskSummary: 구조화된 일일 위험 분석 결과
        """
        if not news_list:
            # 뉴스가 없는 경우 기본 응답
            return DailyRiskSummary(
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                company_name=analysis_data.get("company_name", "Unknown"),
                total_risk_news=0,
                high_risk_count=0,
                medium_risk_count=0,
                key_risk_summary="오늘 관련된 위험 뉴스가 발견되지 않았습니다. (사전 필터링 결과 관련 뉴스 없음)",
                risk_news_items=[],
                credit_assessment=CreditAssessment(
                    current_grade=analysis_data.get("current_credit_grade", "A"),
                    predicted_grade=analysis_data.get("current_credit_grade", "A"),
                    change_probability=0.0,
                    change_direction="유지",
                    reasoning="위험 관련 뉴스가 없어 현재 등급 유지 예상"
                )
            )
        
        try:
            prompt = self.create_enhanced_risk_assessment_prompt()
            
            # JSON 출력 파서 설정
            json_parser = JsonOutputParser(pydantic_object=DailyRiskSummary)
            chain = prompt | self.llm | json_parser
            
            # 뉴스 데이터를 텍스트로 변환 (길이 제한)
            news_text = "\n\n".join([
                f"[뉴스 {i+1}]\n제목: {news.get('title', '')}\n내용: {news.get('content', '')[:500]}...\nURL: {news.get('url', 'URL 없음')}\n발행일: {news.get('published_date', '')}"
                for i, news in enumerate(news_list[:8])  # 최대 8개로 제한
            ])
            
            # 이상치 상세 정보 생성
            financial_detail = "\n".join([
                f"- {anomaly.get('indicator', '')}: {anomaly.get('description', '')} (심각도: {anomaly.get('severity', '')})"
                for anomaly in analysis_data.get("financial_anomalies", [])
            ]) or "탐지된 재무 이상치 없음"
            
            nfr_detail = "\n".join([
                f"- {anomaly.get('indicator', '')} ({anomaly.get('pillar', '')}): {anomaly.get('description', '')} (심각도: {anomaly.get('severity', '')})"
                for anomaly in analysis_data.get("non_financial_anomalies", [])[:10]  # 최대 10개로 제한
            ]) or "탐지된 비재무 이상치 없음"
            
            # 이상치 요약 생성
            financial_summary = f"{len(analysis_data.get('financial_anomalies', []))}개 재무 이상치"
            nfr_summary = f"{len(analysis_data.get('non_financial_anomalies', []))}개 비재무 이상치"
            
            result = chain.invoke({
                "analysis_date" : datetime.now().strftime("%Y-%m-%d"),
                "financial_anomalies_summary": financial_summary,
                "non_financial_anomalies_summary": nfr_summary,
                "current_credit_grade": analysis_data.get("current_credit_grade", "A"),
                "credit_risk_criteria": json.dumps(self.credit_risk_criteria, ensure_ascii=False, indent=1),
                "financial_anomalies_detail": financial_detail,
                "non_financial_anomalies_detail": nfr_detail,
                "financial_characteristics": json.dumps(analysis_data.get("financial_characteristics", {}), ensure_ascii=False),
                "news_data": news_text
            })
            
            # Pydantic 모델로 결과 검증 및 변환
            if isinstance(result, dict):
                return DailyRiskSummary(**result)
            else:
                return result
            
        except Exception as e:
            print(f"⚠️ LLM API 오류 발생: {str(e)}")
            # 오류 시 기본 응답
            return DailyRiskSummary(
                analysis_date=datetime.now().strftime("%Y-%m-%d"),
                company_name=analysis_data.get("company_name", "Unknown"),
                total_risk_news=0,
                high_risk_count=0,
                medium_risk_count=0,
                key_risk_summary=f"뉴스 분석 중 오류가 발생했습니다: {str(e)}",
                risk_news_items=[],
                credit_assessment=CreditAssessment(
                    current_grade=analysis_data.get("current_credit_grade", "A"),
                    predicted_grade=analysis_data.get("current_credit_grade", "A"),
                    change_probability=0.0,
                    change_direction="유지",
                    reasoning="분석 오류로 인해 등급 변경 예측 불가"
                )
            )
    
    def generate_daily_summary_report(self, risk_summary: DailyRiskSummary) -> str:
        """
        일일 신용위험 알림 요약 리포트 생성
        
        Args:
            risk_summary: 일일 위험 분석 결과
            
        Returns:
            str: 알림 형식의 요약 리포트
        """
        
        # 알림 헤더
        report = f"""🚨 {risk_summary.company_name} 신용위험 알림 요약

    📅 분석일자: {risk_summary.analysis_date}
    📊 위험 뉴스: {risk_summary.total_risk_news}개 (고위험 {risk_summary.high_risk_count}개, 중위험 {risk_summary.medium_risk_count}개)
    ⚖️ 신용등급: {risk_summary.credit_assessment.current_grade} → {risk_summary.credit_assessment.predicted_grade} ({risk_summary.credit_assessment.change_probability:.1%})

    """
    
        if risk_summary.risk_news_items:
            # 위험도 순으로 정렬
            sorted_news = sorted(risk_summary.risk_news_items, key=lambda x: x.risk_level, reverse=True)
            
            for news in sorted_news:
                # 위험도에 따른 이모지 설정
                if news.risk_level >= 8:
                    risk_emoji = "🔥"
                    alert_level = "긴급"
                elif news.risk_level >= 6:
                    risk_emoji = "⚠️"
                    alert_level = "주의"
                else:
                    risk_emoji = "ℹ️"
                    alert_level = "모니터링"
                    
                report += f"""
    ═══════════════════════════════════════════════════
    {risk_emoji} {news.risk_category} 리스크 {alert_level} 경고
    [{news.published_date}] "{news.title}"

    """
                
                # 이상치 연관성이 있는 경우 특별 강조
                if news.anomaly_connection and news.anomaly_connection.strip() != "해당없음":
                    report += f"""🔍 이상징후 탐지 근거:
    {news.anomaly_connection}

    📈 뉴스 분석:
    {news.content_summary}

    🤖 AI 위험도 평가:
    {news.evidence_basis}
    (위험도: {news.risk_level}/10)

    """
                else:
                    report += f"""📈 상황 분석:
    {news.content_summary}

    🤖 AI 위험도 평가:
    {news.evidence_basis}
    (위험도: {news.risk_level}/10)

    """
        else:
            report += """
    ═══════════════════════════════════════════════════
    ✅ 오늘 탐지된 신용위험 관련 뉴스가 없습니다.

    """
        
        # 종합 평가
        report += f"""
    ═══════════════════════════════════════════════════
    📋 종합 위험도 평가

    {risk_summary.key_risk_summary}

    💡 신용등급 변동 전망:
    현재 {risk_summary.credit_assessment.current_grade}등급에서 {risk_summary.credit_assessment.predicted_grade}등급으로 {risk_summary.credit_assessment.change_direction}할 확률이 {risk_summary.credit_assessment.change_probability:.1%}입니다.

    근거: {risk_summary.credit_assessment.reasoning}

    ───────────────────────────────────────────────────
    ⏰ 알림 생성: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
    🔄 다음 모니터링: {(datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")}
    """
        
        return report
    
    def analyze_credit_risk_with_results(self, result_dir: str) -> Dict[str, Any]:
        """
        분석 결과를 바탕으로 신용위험 징후 종합 분석
        
        Args:
            result_dir (str): 분석 결과 디렉토리 경로
            
        Returns:
            Dict: 종합 분석 결과
        """
        print("=== 신용위험 징후 뉴스 분석 시작 ===")
        
        # 1. 분석 결과 로드
        print("1. 분석 결과 로드 중...")
        analysis_data = self.load_analysis_results(result_dir)
        
        company_name = analysis_data.get("company_name", "Unknown")
        financial_anomalies_count = len(analysis_data.get("financial_anomalies", []))
        nfr_anomalies_count = len(analysis_data.get("non_financial_anomalies", []))
        
        print(f"   - 기업명: {company_name}")
        print(f"   - 재무 이상치: {financial_anomalies_count}개")
        print(f"   - 비재무 분석 항목: {nfr_anomalies_count}개")
        
        # 2. 타겟팅된 뉴스 검색
        print("\n2. 이상치 기반 뉴스 검색 중...")
        targeted_news = self.search_targeted_news(analysis_data)
        print(f"   - 검색된 총 뉴스 수: {len(targeted_news)}개")
        
        # 2.5. 신용위험 관련 뉴스 사전 필터링 (핵심 변경사항)
        print("\n2.5. 신용위험 관련 뉴스 사전 필터링 중...")
        relevant_news = self._pre_filter_relevant_news(targeted_news, analysis_data)

        # 3. 신용위험 징후 심층 분석
        print("\n3. 신용위험 징후 심층 분석 중...")
        risk_summary = self.assess_credit_risk_with_anomalies(analysis_data, relevant_news)
        
        # 4. 마크다운 리포트 생성
        print("\n4. 일일 요약 리포트 생성 중...")
        daily_report = self.generate_daily_summary_report(risk_summary)
        
        # 5. 결과 정리
        analysis_result = {
            "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "company": company_name,
            "total_news_count": len(targeted_news),
            "relevant_news_count": len(relevant_news),
            "financial_anomalies_count": financial_anomalies_count,
            "non_financial_anomalies_count": nfr_anomalies_count,
            "company_analysis_summary": self._create_analysis_summary(analysis_data),
            "relevant_news": relevant_news, # 필터링된 뉴스
            "daily_risk_summary": risk_summary.dict(),
            "daily_summary_report": daily_report,
            "analysis_basis": "detected_anomalies_and_pre_filtered_news"
        }
        
        # 6. 결과 저장
        try:
            # JSON 결과 저장
            json_output_file = os.path.join(result_dir, "daily_news_risk_analysis.json")
            with open(json_output_file, 'w', encoding='utf-8') as f:
                json.dump(analysis_result, f, ensure_ascii=False, indent=2)
            print(f"\n✅ JSON 분석 결과 저장: {json_output_file}")
            
            # 마크다운 리포트 저장
            report_output_file = os.path.join(result_dir, "daily_risk_summary.md")
            with open(report_output_file, 'w', encoding='utf-8') as f:
                f.write(daily_report)
            print(f"✅ 일일 요약 리포트 저장: {report_output_file}")
            
        except Exception as e:
            print(f"❌ 결과 저장 오류: {str(e)}")
        
        print("\n=== 분석 완료 ===")
        return analysis_result
    
    def _create_analysis_summary(self, analysis_data: Dict[str, Any]) -> Dict[str, Any]:
        """분석 데이터를 바탕으로 요약 정보 생성"""
        
        financial_anomalies = analysis_data.get("financial_anomalies", [])
        nfr_anomalies = analysis_data.get("non_financial_anomalies", [])
        
        # 위험 수준 계산 (재무)
        financial_high_risk = sum(1 for a in financial_anomalies if a.get("severity") in ["high", "critical"])
        financial_medium_risk = sum(1 for a in financial_anomalies if a.get("severity") == "medium")
        
        # 위험 수준 계산 (비재무)
        nfr_high_risk = sum(1 for a in nfr_anomalies if a.get("severity") in ["high"])
        nfr_medium_risk = sum(1 for a in nfr_anomalies if a.get("severity") == "medium")
        
        total_high_risk = financial_high_risk + nfr_high_risk
        total_anomalies = len(financial_anomalies) + len(nfr_anomalies)
        
        # 전체 위험 수준 결정
        if total_high_risk >= 3 or total_anomalies >= 10:
            overall_risk = "높음"
        elif total_high_risk >= 1 or total_anomalies >= 6:
            overall_risk = "주의"
        elif total_anomalies >= 3:
            overall_risk = "보통"
        else:
            overall_risk = "낮음"
        
        # 주요 우려사항 추출
        concerns = []
        
        # 재무 우려사항
        for anomaly in financial_anomalies:
            if anomaly.get("severity") in ["high", "medium", "critical"]:
                concerns.append(f"재무: {anomaly.get('description', '')}")
        
        # 비재무 우려사항
        for anomaly in nfr_anomalies:
            if anomaly.get("severity") in ["high", "medium"]:
                concerns.append(f"비재무({anomaly.get('pillar', '')}): {anomaly.get('description', '')}")
        
        return {
            "overall_risk_level": overall_risk,
            "total_anomalies": total_anomalies,
            "high_risk_anomalies": total_high_risk,
            "financial_risk_count": len(financial_anomalies),
            "non_financial_risk_count": len(nfr_anomalies),
            "financial_characteristics": analysis_data.get("financial_characteristics", {}),
            "key_concerns": concerns[:5]  # 상위 5개
        }


# 사용 예제
def main():
    """
    메인 실행 함수 (이상치 기반 분석)
    """
    try:
        # API 키 설정
        from dotenv import load_dotenv
        load_dotenv()
        
        import os
        if not os.getenv("OPENAI_API_KEY"):
            print("⚠️ OPENAI_API_KEY가 설정되지 않았습니다.")
            return
            
        if not os.getenv("TAVILY_API_KEY"):
            print("⚠️ TAVILY_API_KEY가 설정되지 않았습니다.")
            return
        
        # 신용위험 뉴스 분석기 초기화
        print("=== 신용위험 뉴스 분석기 초기화 ===")
        analyzer = CreditRiskNewsAnalyzer(max_search_results=5)
        
        # 예시 분석 결과 디렉토리 (실제 환경에서는 run.py에서 전달)
        test_result_dir = "analysis_results/삼성전자"
        if not os.path.exists(test_result_dir):
            os.makedirs(test_result_dir)
            print(f"테스트 디렉토리 생성: {test_result_dir}")
        
        # 이상치 기반 신용위험 징후 분석 실행
        print("\n=== 이상치 기반 신용위험 징후 분석 실행 ===")
        analysis_result = analyzer.analyze_credit_risk_with_results(test_result_dir)
        
        # 결과 출력
        print("\n" + "="*60)
        print("📰 신용위험 징후 뉴스 분석 완료")
        print("="*60)
        
        risk_summary = analysis_result.get("daily_risk_summary", {})
        
        print(f"📊 분석 요약:")
        print(f"- 분석 대상: {analysis_result['company']}")
        print(f"- 총 검색된 뉴스: {analysis_result['total_news_count']}개")
        print(f"- 신용위험 관련 뉴스: {analysis_result['relevant_news_count']}개 (선별됨)")
        print(f"- 최종 분석된 위험 뉴스: {risk_summary.get('total_risk_news', 0)}개")
        print(f"  - 고위험: {risk_summary.get('high_risk_count', 0)}개, 중위험: {risk_summary.get('medium_risk_count', 0)}개")
        
        credit_assessment = risk_summary.get('credit_assessment', {})
        print(f"\n📈 신용등급 평가:")
        print(f"- 현재 신용등급: {credit_assessment.get('current_grade', 'Unknown')}")
        print(f"- 예상 신용등급: {credit_assessment.get('predicted_grade', 'Unknown')}")
        print(f"- 변경 확률: {credit_assessment.get('change_probability', 0):.1%}")
        print(f"- 변경 방향: {credit_assessment.get('change_direction', 'Unknown')}")
        
        print(f"\n📄 생성된 파일:")
        print(f"- {os.path.join(test_result_dir, 'daily_news_risk_analysis.json')}")
        print(f"- {os.path.join(test_result_dir, 'daily_risk_summary.md')}")
                
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()