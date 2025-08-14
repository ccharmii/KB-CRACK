# -*- coding: utf-8 -*-
"""
통합 신용위험 분석 최종 리포트 생성기 (수정된 버전)
뉴스 분석을 제외한 재무지표, 비재무지표, 근거 분석을 통합하여 최종 리포트 생성
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv

load_dotenv()

class CreditRiskReportGenerator:
    """신용위험 분석 리포트 생성기"""
    
    def __init__(self, model="gpt-4o-mini"):
        """
        리포트 생성기 초기화
        
        Args:
            model: 사용할 LLM 모델
        """
        self.llm = ChatOpenAI(model=model, temperature=0.3)
        
    def load_analysis_results(self, result_dir: str) -> Dict[str, Any]:
        """
        분석 결과 디렉토리에서 모든 분석 결과를 로드
        
        Args:
            result_dir: 분석 결과 디렉토리 경로
            
        Returns:
            Dict: 통합된 분석 결과
        """
        results = {
            "financial_analysis": None,
            "financial_anomalies": None,
            "non_financial_analysis": None,
            "integrated_anomaly_analysis": None
        }
        
        # 재무분석 원본 데이터 로드
        financial_analysis_file = os.path.join(result_dir, "financial_analysis.json")
        if os.path.exists(financial_analysis_file):
            try:
                with open(financial_analysis_file, 'r', encoding='utf-8') as f:
                    results["financial_analysis"] = json.load(f)
                print("✅ 재무분석 원본 데이터 로드 완료")
            except Exception as e:
                print(f"❌ 재무분석 원본 데이터 로드 오류: {str(e)}")
        
        # 재무지표 이상치 로드
        financial_anomalies_file = os.path.join(result_dir, "financial_anomalies.json")
        if os.path.exists(financial_anomalies_file):
            try:
                with open(financial_anomalies_file, 'r', encoding='utf-8') as f:
                    results["financial_anomalies"] = json.load(f)
                print(f"✅ 재무지표 이상치 로드 완료: {len(results['financial_anomalies'])}개")
            except Exception as e:
                print(f"❌ 재무지표 이상치 로드 오류: {str(e)}")
        
        # 비재무분석 결과 로드
        nfr_file = os.path.join(result_dir, "non_financial_analysis.json")
        if os.path.exists(nfr_file):
            try:
                with open(nfr_file, 'r', encoding='utf-8') as f:
                    results["non_financial_analysis"] = json.load(f)
                print("✅ 비재무분석 결과 로드 완료")
            except Exception as e:
                print(f"❌ 비재무분석 결과 로드 오류: {str(e)}")
        
        # 통합 이상치 분석 결과 로드
        integrated_file = os.path.join(result_dir, "integrated_anomaly_report.json")
        if os.path.exists(integrated_file):
            try:
                with open(integrated_file, 'r', encoding='utf-8') as f:
                    integrated_data = json.load(f)
                    results["integrated_anomaly_analysis"] = integrated_data.get("통합_이상치_분석", [])
                print("✅ 통합 이상치 분석 결과 로드 완료")
            except Exception as e:
                print(f"❌ 통합 이상치 분석 결과 로드 오류: {str(e)}")
        
        return results
        
    def _calculate_comprehensive_risk_score(self, all_results: Dict[str, Any]) -> Dict[str, Any]:
        """종합 신용위험 점수 계산"""
        
        base_score = 100
        
        # 재무지표 이상치로 인한 감점
        financial_penalty = 0
        financial_anomalies = all_results.get("financial_anomalies")
        if financial_anomalies:
            for metric_name, details in financial_anomalies.items():
                severity = details.get("severity", "Medium") if isinstance(details, dict) else "Medium"
                if severity == "High":
                    financial_penalty += 15
                elif severity == "Medium":
                    financial_penalty += 8
                else:
                    financial_penalty += 3
        
        # 비재무지표로 인한 감점
        nfr_penalty = 0
        nfr_data = all_results.get("non_financial_analysis")
        if nfr_data:
            risk_summary = nfr_data.get("risk_summary", {})
            risk_level = risk_summary.get("overall_risk_level", "보통")
            
            if risk_level == "높음":
                nfr_penalty += 25
            elif risk_level == "주의":
                nfr_penalty += 15
            elif risk_level == "보통":
                nfr_penalty += 8
            
            # 개별 지표 점수도 고려
            latest_results = nfr_data.get("latest_quarter_results", [])
            for result in latest_results:
                score = result.get("score", 3)
                if score <= 1:
                    nfr_penalty += 5
                elif score <= 2:
                    nfr_penalty += 3
        
        # 최종 점수 계산
        final_score = max(0, base_score - financial_penalty - nfr_penalty)
        
        # 등급 결정
        if final_score >= 90:
            grade = "AAA"
            risk_level = "매우 낮음"
        elif final_score >= 80:
            grade = "AA"
            risk_level = "낮음"
        elif final_score >= 70:
            grade = "A"
            risk_level = "보통"
        elif final_score >= 60:
            grade = "BBB"
            risk_level = "주의"
        elif final_score >= 50:
            grade = "BB"
            risk_level = "위험"
        else:
            grade = "B"
            risk_level = "고위험"
        
        return {
            "score": final_score,
            "grade": grade,
            "risk_level": risk_level,
            "breakdown": {
                "financial_penalty": financial_penalty,
                "non_financial_penalty": nfr_penalty
            },
            "component_scores": {
                "financial": max(0, 100 - financial_penalty),
                "non_financial": max(0, 100 - nfr_penalty)
            }
        }
    
    def _extract_quarterly_trend_data(self, financial_data: Dict[str, Any]) -> Dict[str, Any]:
        """분기별 재무지표 변화 추세 데이터 추출"""
        
        if not financial_data:
            return {}
            
        # 주요지표 시계열 분석 데이터 추출
        quarterly_data = financial_data.get("주요지표_시계열_분석", {})
        
        # 분기별 데이터를 시간순으로 정렬
        sorted_quarters = sorted(quarterly_data.keys())
        
        trend_analysis = {}
        
        for i, quarter in enumerate(sorted_quarters):
            if i > 0:  # 이전 분기와 비교
                prev_quarter = sorted_quarters[i-1]
                current_data = quarterly_data[quarter]
                prev_data = quarterly_data[prev_quarter]
                
                quarter_trend = {}
                for metric, current_value in current_data.items():
                    if metric in prev_data:
                        prev_value = prev_data[metric]
                        if prev_value != 0:
                            change_rate = ((current_value - prev_value) / prev_value) * 100
                            quarter_trend[metric] = {
                                "current": current_value,
                                "previous": prev_value,
                                "change_rate": round(change_rate, 2),
                                "direction": "증가" if change_rate > 0 else "감소" if change_rate < 0 else "변화없음"
                            }
                
                trend_analysis[quarter] = quarter_trend
        
        return trend_analysis
    
    def _find_similar_cases(self, anomalies: Dict[str, Any], company_info: Dict[str, Any]) -> List[Dict]:
        """유사한 이상치 사례 검색 (예시 데이터)"""
        
        # 실제 구현에서는 데이터베이스에서 유사 사례를 검색
        similar_cases = [
            {
                "company": "LG전자",
                "year": "2023",
                "anomaly_type": "ROE 하락",
                "initial_grade": "A+",
                "final_grade": "A",
                "recovery_period": "6개월",
                "actions_taken": ["비용절감", "사업구조 개편", "R&D 투자 증대"]
            },
            {
                "company": "SK하이닉스",
                "year": "2022",
                "anomaly_type": "매출액증가율 둔화",
                "initial_grade": "AA-",
                "final_grade": "A+",
                "recovery_period": "9개월",
                "actions_taken": ["신제품 출시", "해외시장 확장", "원가절감"]
            }
        ]
        
        return similar_cases
    
    def generate_ai_analysis_summary(self, all_results: Dict[str, Any], risk_assessment: Dict[str, Any]) -> str:
        """AI 종합 분석 요약 생성"""
        
        summary_prompt = ChatPromptTemplate.from_template("""
당신은 신용위험 분석 전문가입니다. 다음 분석 결과를 바탕으로 1-2줄의 간결한 AI 종합 분석 요약을 작성해주세요.

재무지표 이상치: {financial_anomalies_count}개
비재무지표 위험수준: {nfr_risk_level}
종합 위험등급: {risk_grade}
종합 점수: {risk_score}/100

핵심 이상치:
{key_anomalies}

1-2줄로 핵심 리스크와 전반적인 신용상태를 요약해주세요.
        """)
        
        chain = summary_prompt | self.llm | StrOutputParser()
        
        try:
            financial_anomalies = all_results.get("financial_anomalies", {})
            nfr_data = all_results.get("non_financial_analysis", {})
            
            key_anomalies = []
            if financial_anomalies:
                for metric, details in list(financial_anomalies.items())[:3]:  # 상위 3개만
                    if isinstance(details, dict):
                        key_anomalies.append(f"- {metric}: {details.get('description', '')}")
            
            summary = chain.invoke({
                "financial_anomalies_count": len(financial_anomalies) if financial_anomalies else 0,
                "nfr_risk_level": nfr_data.get("risk_summary", {}).get("overall_risk_level", "알 수 없음"),
                "risk_grade": risk_assessment["grade"],
                "risk_score": risk_assessment["score"],
                "key_anomalies": "\n".join(key_anomalies) or "주요 이상치 없음"
            })
            
            return summary.strip()
            
        except Exception as e:
            return f"AI 분석 요약 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_executive_summary(self, all_results: Dict[str, Any], risk_assessment: Dict[str, Any], 
                                 company_info: Dict[str, Any]) -> str:
        """경영진용 요약 리포트 생성"""
        
        summary_prompt = ChatPromptTemplate.from_template("""
당신은 신용위험 분석 전문가입니다. 다음 정보를 바탕으로 경영진용 종합 요약 리포트를 작성해주세요.

## 기업 정보
기업명: {company_name}
현재 신용등급: {current_grade}
분석 대상 등급: {new_grade}

## 종합 위험 평가
- 점수: {score}점/100점
- 등급: {grade}
- 위험수준: {risk_level}

## 재무지표 이상치 분석
{financial_anomalies_summary}

## 비재무지표 분석
{non_financial_summary}

## 근거 기반 인사이트
{evidence_insights}

다음 구조로 작성해주세요:

# {company_name} 신용위험 분석 요약 리포트

## 🔍 AI 종합 분석 요약
{ai_summary}

## 📊 예상 신용등급 변화
현재: {current_grade} → 예상: {new_grade}
변화 사유: [주요 위험요인 요약]

## ⚠️ 주요 위험 요소
### 재무지표 위험
[재무지표 이상치별 위험도 및 영향]

### 비재무지표 위험  
[비재무지표 위험수준 및 세부사항]

## 💡 필요 조치사항
### 즉시 대응 필요
[긴급 조치사항]

### 중장기 개선방안
[전략적 개선방안]

## 📈 인사이트 및 권고사항
[데이터 기반 핵심 인사이트와 구체적 권고사항]
        """)
        
        chain = summary_prompt | self.llm | StrOutputParser()
        
        try:
            # 재무지표 이상치 요약
            financial_anomalies = all_results.get("financial_anomalies", {})
            financial_summary = []
            if financial_anomalies:
                for metric, details in financial_anomalies.items():
                    if isinstance(details, dict):
                        severity = details.get("severity", "Medium")
                        desc = details.get("description", "")
                        financial_summary.append(f"- {metric} ({severity}): {desc}")
            
            # 비재무지표 요약
            nfr_data = all_results.get("non_financial_analysis", {})
            nfr_summary = f"전체 위험수준: {nfr_data.get('risk_summary', {}).get('overall_risk_level', '알 수 없음')}"
            
            # 근거 기반 인사이트
            integrated_data = all_results.get("integrated_anomaly_analysis", [])
            evidence_insights = []
            for anomaly in integrated_data[:3]:  # 상위 3개
                if isinstance(anomaly, dict):
                    metric = anomaly.get("메트릭명", "")
                    news_analysis = anomaly.get("뉴스_분석", {})
                    primary_cause = news_analysis.get("주요_원인", "")
                    if primary_cause:
                        evidence_insights.append(f"- {metric}: {primary_cause}")
            
            # AI 요약 생성
            ai_summary = self.generate_ai_analysis_summary(all_results, risk_assessment)
            
            summary = chain.invoke({
                "company_name": company_info.get("기업명", "분석 대상 기업"),
                "current_grade": company_info.get("Current_credit_grade", "알 수 없음"),
                "new_grade": risk_assessment["grade"],
                "score": risk_assessment["score"],
                "grade": risk_assessment["grade"],
                "risk_level": risk_assessment["risk_level"],
                "financial_anomalies_summary": "\n".join(financial_summary) or "이상치 없음",
                "non_financial_summary": nfr_summary,
                "evidence_insights": "\n".join(evidence_insights) or "근거 기반 인사이트 없음",
                "ai_summary": ai_summary
            })
            
            return summary
            
        except Exception as e:
            return f"요약 리포트 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_detailed_analysis(self, all_results: Dict[str, Any]) -> str:
        """상세 분석 리포트 생성"""
        
        detailed_prompt = ChatPromptTemplate.from_template(
"""
다음 종합 신용위험 분석 결과를 바탕으로 상세한 분석 리포트를 작성해주세요.
근거는 반드시 {integrated_analysis}에서 직접 발췌한 문장으로 표기하세요.

## 재무지표 분석 결과
{financial_data}

## 비재무지표 분석 결과  
{non_financial_data}

## 통합 이상치 분석 (근거 데이터)
해당 근거 데이터로 리포트를 작성해주세요.
{integrated_analysis}

다음 구조로 상세 분석을 작성해주세요:

## 1. 재무 위험 분석
- 탐지된 이상치와 그 의미
- 이상치에 대한 뉴스 근거 : 뉴스 제목, url, 내용 일부 
- 이상치에 대한 비재무 근거 : 보고서 종류, 관련 문장 
- 분기별 재무지표 변화 트렌드
- 동종업계 대비 상대적 위치

## 2. 비재무 위험 분석
- 5개 핵심 영역별 위험도 평가 (산업위험, 경영위험, 영업위험, 재무위험(질적), 신뢰도)
- 정기보고서 기반 질적 위험 요소
- 거버넌스 및 운영 위험 

## 3. 근거 기반 원인 분석
- 각 이상치의 근본 원인과 증거 
- 뉴스 및 시장 정보 기반 외부 요인 분석(근거문장, url 도출)
- 내부 경영진의 의사결정 영향

## 4. 통합 분석 및 종합 의견
- 재무·비재무 분석의 일관성
- 종합적 위험 평가
- 향후 전망 및 시나리오
""")
        
        chain = detailed_prompt | self.llm | StrOutputParser()
        
        try:
            detailed_analysis = chain.invoke({
                "financial_data": json.dumps(all_results.get("financial_anomalies"), ensure_ascii=False, indent=2) if all_results.get("financial_anomalies") else "재무 이상치 없음",
                "non_financial_data": json.dumps(all_results.get("non_financial_analysis"), ensure_ascii=False, indent=2) if all_results.get("non_financial_analysis") else "비재무 분석 결과 없음",
                "integrated_analysis": json.dumps(all_results.get("integrated_anomaly_analysis"), ensure_ascii=False, indent=2) if all_results.get("integrated_anomaly_analysis") else "통합 분석 결과 없음"
            })
            
            return detailed_analysis
            
        except Exception as e:
            return f"상세 분석 리포트 생성 중 오류가 발생했습니다: {str(e)}"
    
    def generate_integrated_report_from_dir(self, result_dir: str) -> Dict[str, Any]:
        """
        분석 결과 디렉토리로부터 통합 리포트 생성
        
        Args:
            result_dir: 분석 결과 디렉토리 경로
            
        Returns:
            Dict: 통합 리포트
        """
        
        print(f"📊 통합 리포트 생성 중... (결과 디렉토리: {result_dir})")
        
        # 1. 모든 분석 결과 로드
        all_results = self.load_analysis_results(result_dir)
        
        # 기업 정보 추출
        financial_data = all_results.get("financial_analysis", {})
        company_info = financial_data.get("기업_정보", {})
        company_name = company_info.get("기업명", "분석 대상 기업")
        
        # 2. 종합 위험 평가
        risk_assessment = self._calculate_comprehensive_risk_score(all_results)
        
        # 3. AI 분석 요약 생성
        ai_summary = self.generate_ai_analysis_summary(all_results, risk_assessment)
        
        # 4. 분기별 트렌드 분석
        quarterly_trends = self._extract_quarterly_trend_data(financial_data)
        
        # 5. 유사 사례 검색
        similar_cases = self._find_similar_cases(all_results.get("financial_anomalies", {}), company_info)
        
        # 6. 경영진용 요약 리포트 생성
        executive_summary = self.generate_executive_summary(all_results, risk_assessment, company_info)
        
        # 7. 상세 분석 리포트 생성
        detailed_analysis = self.generate_detailed_analysis(all_results)
        
        # 8. 최종 JSON 리포트 구성
        final_report_json = {
            "분석_메타데이터": {
                "생성_시간": datetime.now().isoformat(),
                "분석_대상": company_name,
                "분석_버전": "3.0",
                "결과_디렉토리": result_dir
            },
            "기업_정보": company_info,
            "AI_종합_분석_요약": ai_summary,
            "예상_신용등급_변화": {
                "현재_등급": company_info.get("Current_credit_grade", "알 수 없음"),
                "예상_등급": risk_assessment["grade"],
                "등급_변화_사유": "재무지표 이상치 및 비재무 위험요소 종합 평가 결과"
            },
            "재무지표_분석": {
                "이상치_목록": all_results.get("financial_anomalies", {}),
                "통합_근거_분석": all_results.get("integrated_anomaly_analysis", []),
                "분기별_변화_추이": quarterly_trends,
                "위험도_평가": risk_assessment["breakdown"]
            },
            "비재무지표_분석": {
                "이상치_탐지_기준": "5개 핵심 영역 평가 (산업위험, 경영위험, 영업위험, 재무위험(질적), 신뢰도)",
                "탐지된_이상치": all_results.get("non_financial_analysis", {}).get("latest_quarter_results", []),
                "위험도": all_results.get("non_financial_analysis", {}).get("risk_summary", {}),
            },
            "유사_사례_분석": similar_cases,
            "종합_위험평가": risk_assessment,
        }
        
        # 9. 통합 리포트 구성
        integrated_report = {
            "metadata": {
                "company_name": company_name,
                "analysis_date": datetime.now().isoformat(),
                "report_version": "3.0",
                "analyst": "AI Credit Risk Analyzer",
                "result_directory": result_dir
            },
            "risk_assessment": risk_assessment,
            "executive_summary": executive_summary,
            "detailed_analysis": detailed_analysis,
            "final_report_json": final_report_json,
            "source_data": {
                "financial_analysis_available": all_results.get("financial_analysis") is not None,
                "financial_anomalies_available": all_results.get("financial_anomalies") is not None,
                "non_financial_analysis_available": all_results.get("non_financial_analysis") is not None,
                "integrated_analysis_available": all_results.get("integrated_anomaly_analysis") is not None
            }
        }
        
        return integrated_report


def generate_final_report(result_dir: str) -> Dict[str, Any]:
    """
    결과 디렉토리로부터 최종 통합 리포트 생성 (메인 함수)
    
    Args:
        result_dir: 분석 결과가 저장된 디렉토리 경로
        
    Returns:
        Dict: 통합 리포트
    """
    
    generator = CreditRiskReportGenerator()
    integrated_report = generator.generate_integrated_report_from_dir(result_dir)
    
    # JSON 리포트 저장
    json_report_path = os.path.join(result_dir, "final_comprehensive_report.json")
    with open(json_report_path, 'w', encoding='utf-8') as f:
        json.dump(integrated_report["final_report_json"], f, ensure_ascii=False, indent=2)
    
    print(f"📄 최종 JSON 리포트 저장: {json_report_path}")
    
    return integrated_report


if __name__ == "__main__":
    # 테스트 실행
    test_result_dir = "analysis_results/삼성전자"
    
    try:
        report = generate_final_report(test_result_dir)
        
        print("=== 통합 리포트 생성 완료 ===")
        print(f"기업명: {report['metadata']['company_name']}")
        print(f"종합 등급: {report['risk_assessment']['grade']}")
        print(f"위험 수준: {report['risk_assessment']['risk_level']}")
        print(f"종합 점수: {report['risk_assessment']['score']}/100")
        
        # 요약 리포트 미리보기
        print("\n=== 경영진용 요약 (미리보기) ===")
        summary = report['executive_summary']
        print(summary[:500] + "..." if len(summary) > 500 else summary)
        
    except Exception as e:
        print(f"테스트 실행 중 오류: {str(e)}")