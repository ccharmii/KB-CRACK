#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
이상치 분석 결과 통합 정리 모듈
재무 이상치에 대한 비재무 근거, 뉴스 근거, 관련 보고서 등을 통합하여 JSON으로 출력
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
from pathlib import Path


class AnomalyIntegrator:
    """이상치 분석 결과 통합 클래스"""
    
    def __init__(self, output_dir: str = None):
        """
        초기화
        
        Args:
            output_dir (str): 결과 저장 디렉토리
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./integrated_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_analysis_files(self, news_analysis_path: str, non_financial_path: str) -> tuple:
        """
        분석 결과 파일들을 로드
        
        Args:
            news_analysis_path (str): 뉴스 분석 결과 파일 경로
            non_financial_path (str): 비재무 분석 결과 파일 경로
            
        Returns:
            tuple: (뉴스 분석 결과, 비재무 분석 결과)
        """
        try:
            # 뉴스 분석 결과 로드
            with open(news_analysis_path, 'r', encoding='utf-8') as f:
                news_data = json.load(f)
                
            # 비재무 분석 결과 로드
            with open(non_financial_path, 'r', encoding='utf-8') as f:
                non_financial_data = json.load(f)
                
            return news_data, non_financial_data
            
        except Exception as e:
            print(f"❌ 파일 로드 중 오류 발생: {str(e)}")
            return None, None
    
    def extract_company_info(self, news_data: Dict) -> Dict:
        """
        기업 정보 추출
        
        Args:
            news_data (Dict): 뉴스 분석 데이터
            
        Returns:
            Dict: 기업 정보
        """
        company_info = news_data.get("company_info", {})
        
        return {
            "기업명_한글": company_info.get("기업명", ""),
            "기업명_영문": company_info.get("영문기업명", ""),
            "종목코드": company_info.get("종목코드", ""),
            "대표자명": company_info.get("대표자명", ""),
            "업종": company_info.get("업종", ""),
            "신용등급": company_info.get("Current_credit_grade", ""),
            "주소": company_info.get("주소", "")
        }
    
    def match_anomalies_by_metric(self, news_anomalies: List[Dict], 
                                non_financial_results: List[Dict]) -> List[Dict]:
        """
        메트릭명을 기준으로 뉴스 이상치와 비재무 결과를 매칭
        
        Args:
            news_anomalies (List[Dict]): 뉴스 분석에서 탐지된 이상치 목록
            non_financial_results (List[Dict]): 비재무 분석 결과 목록
            
        Returns:
            List[Dict]: 통합된 이상치 분석 결과
        """
        integrated_anomalies = []
        
        # 비재무 결과를 메트릭별로 인덱싱
        nf_by_metric = {}
        for nf_result in non_financial_results:
            metric = nf_result.get("metric", "")
            nf_by_metric[metric] = nf_result
        
        # 뉴스 이상치를 기준으로 통합
        for news_anomaly in news_anomalies:
            anomaly_info = news_anomaly.get("anomaly_info", {})
            metric_name = anomaly_info.get("metric_name", "")
            
            # 매칭되는 비재무 결과 찾기
            matching_nf = nf_by_metric.get(metric_name)
            
            # 통합 결과 구성
            integrated_anomaly = self._create_integrated_anomaly(
                news_anomaly, matching_nf, metric_name
            )
            
            integrated_anomalies.append(integrated_anomaly)
        
        return integrated_anomalies
    
    def _create_integrated_anomaly(self, news_anomaly: Dict, 
                                 non_financial_result: Optional[Dict], 
                                 metric_name: str) -> Dict:
        """
        개별 이상치에 대한 통합 결과 생성
        
        Args:
            news_anomaly (Dict): 뉴스 분석 이상치
            non_financial_result (Optional[Dict]): 매칭되는 비재무 결과
            metric_name (str): 메트릭명
            
        Returns:
            Dict: 통합된 이상치 결과
        """
        anomaly_info = news_anomaly.get("anomaly_info", {})
        analysis = news_anomaly.get("analysis", {})
        news_evidence = news_anomaly.get("news_evidence", [])
        
        # 기본 이상치 정보
        result = {
            "메트릭명": metric_name,
            "이상치_설명": anomaly_info.get("description", ""),
            "심각도": anomaly_info.get("severity", ""),
            "분기": anomaly_info.get("quarter", ""),
            "탐지_유형": anomaly_info.get("type", ""),
            "분석_시점": datetime.now().isoformat()
        }
        
        # 뉴스 기반 분석 결과
        result["뉴스_분석"] = {
            "주요_원인": analysis.get("primary_cause", ""),
            "신뢰도": analysis.get("confidence_level", 0),
            "상세_설명": analysis.get("detailed_explanation", ""),
            "영향_평가": analysis.get("impact_assessment", ""),
            "위험_수준": analysis.get("risk_level", ""),
            "지원_증거": analysis.get("supporting_evidence", []),
            "뉴스_소스": analysis.get("news_sources", []),
            "관련성_품질점수": analysis.get("relevance_quality", "")
        }
        
        # 뉴스 증거 정보
        result["뉴스_증거"] = []
        for news in news_evidence:
            news_info = {
                "제목": news.get("title", ""),
                "URL": news.get("url", ""),
                "발행일": news.get("published_date", ""),
                "출처": news.get("source", ""),
                "관련성_점수": news.get("hybrid_score", 0),
                "내용_요약": news.get("content", "")[:200] + "..." if news.get("content") else ""
            }
            result["뉴스_증거"].append(news_info)
        
        # 비재무 분석 결과 (있는 경우)
        if non_financial_result:
            result["비재무_분석"] = {
                "설명": non_financial_result.get("explanation_ko", ""),
                "주요_원인": non_financial_result.get("drivers", []),
                "신뢰도": non_financial_result.get("confidence", 0),
                "관련_보고서": [],
                "관련_문장": []
            }
            
            # 증거 문서 정보 추가
            evidence = non_financial_result.get("evidence", [])
            for ev in evidence:
                report_info = {
                    "보고서_번호": ev.get("rcept_no", ""),
                    "문서_ID": ev.get("chunk_id", ""),
                    "관련_문장": ev.get("snippet", ""),
                    "소스_인덱스": ev.get("source_idx", "")
                }
                result["비재무_분석"]["관련_보고서"].append(report_info)
                result["비재무_분석"]["관련_문장"].append(ev.get("snippet", ""))
        else:
            result["비재무_분석"] = {
                "설명": "매칭되는 비재무 분석 결과를 찾을 수 없습니다.",
                "주요_원인": [],
                "신뢰도": 0,
                "관련_보고서": [],
                "관련_문장": []
            }
        
        return result
    
    def create_summary_statistics(self, integrated_anomalies: List[Dict]) -> Dict:
        """
        통합 분석 결과 요약 통계 생성
        
        Args:
            integrated_anomalies (List[Dict]): 통합된 이상치 목록
            
        Returns:
            Dict: 요약 통계
        """
        total_anomalies = len(integrated_anomalies)
        
        # 심각도별 분포
        severity_count = {}
        # 메트릭별 분포
        metric_count = {}
        # 뉴스 신뢰도 평균
        news_confidence_scores = []
        # 비재무 신뢰도 평균
        nf_confidence_scores = []
        
        for anomaly in integrated_anomalies:
            # 심각도 집계
            severity = anomaly.get("심각도", "Unknown")
            severity_count[severity] = severity_count.get(severity, 0) + 1
            
            # 메트릭 집계
            metric = anomaly.get("메트릭명", "Unknown")
            metric_count[metric] = metric_count.get(metric, 0) + 1
            
            # 신뢰도 점수 수집
            news_conf = anomaly.get("뉴스_분석", {}).get("신뢰도", 0)
            if news_conf > 0:
                news_confidence_scores.append(news_conf)
                
            nf_conf = anomaly.get("비재무_분석", {}).get("신뢰도", 0)
            if nf_conf > 0:
                nf_confidence_scores.append(nf_conf)
        
        return {
            "총_이상치_수": total_anomalies,
            "심각도별_분포": severity_count,
            "메트릭별_분포": metric_count,
            "평균_뉴스_신뢰도": sum(news_confidence_scores) / len(news_confidence_scores) if news_confidence_scores else 0,
            "평균_비재무_신뢰도": sum(nf_confidence_scores) / len(nf_confidence_scores) if nf_confidence_scores else 0,
            "뉴스_증거_보유_이상치": sum(1 for a in integrated_anomalies if a.get("뉴스_증거")),
            "비재무_증거_보유_이상치": sum(1 for a in integrated_anomalies if a.get("비재무_분석", {}).get("관련_보고서"))
        }
    
    def generate_integrated_report(self, news_analysis_path: str, 
                                 non_financial_path: str) -> Dict:
        """
        통합 이상치 분석 리포트 생성
        
        Args:
            news_analysis_path (str): 뉴스 분석 결과 파일 경로
            non_financial_path (str): 비재무 분석 결과 파일 경로
            
        Returns:
            Dict: 통합된 분석 리포트
        """
        print("🔄 이상치 분석 결과 통합 시작...")
        
        # 분석 파일 로드
        news_data, non_financial_data = self.load_analysis_files(
            news_analysis_path, non_financial_path
        )
        
        if not news_data or not non_financial_data:
            return {"error": "분석 파일 로드 실패"}
        
        # 기업 정보 추출
        company_info = self.extract_company_info(news_data)
        
        # 이상치 매칭 및 통합
        news_anomalies = news_data.get("anomaly_news_analyses", [])
        non_financial_results = non_financial_data.get("results", [])
        
        integrated_anomalies = self.match_anomalies_by_metric(
            news_anomalies, non_financial_results
        )
        
        # 요약 통계 생성
        summary_stats = self.create_summary_statistics(integrated_anomalies)
        
        # 최종 리포트 구성
        integrated_report = {
            "분석_메타데이터": {
                "생성_시간": datetime.now().isoformat(),
                "뉴스_분석_파일": str(news_analysis_path),
                "비재무_분석_파일": str(non_financial_path),
                "분석_방법론": {
                    "뉴스_분석": news_data.get("methodology", {}),
                    "통합_방식": "메트릭명 기준 매칭"
                }
            },
            "기업_정보": company_info,
            "요약_통계": summary_stats,
            "통합_이상치_분석": integrated_anomalies
        }
        
        print(f"✅ 통합 분석 완료: {len(integrated_anomalies)}개 이상치 처리")
        return integrated_report
    
    def save_integrated_report(self, report: Dict, filename: str = None) -> str:
        """
        통합 리포트를 JSON 파일로 저장
        
        Args:
            report (Dict): 통합 리포트
            filename (str): 저장할 파일명 (기본값: 자동 생성)
            
        Returns:
            str: 저장된 파일 경로
        """
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            corp_name = report.get("기업_정보", {}).get("기업명_한글", "Unknown")
            filename = f"integrated_anomaly_report.json"
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        
        print(f"💾 통합 리포트 저장: {filepath}")
        return str(filepath)
    

def main():
    """메인 실행 함수 - 예시"""
    
    # 예시 파일 경로 (실제 경로로 수정 필요)
    news_analysis_file = "analysis_results/삼성전자/anomaly_news_analysis.json"
    non_financial_file = "analysis_results/삼성전자/non_financial_reasoning.json"
    
    # 통합기 초기화
    integrator = AnomalyIntegrator(output_dir="./삼성전자")
    
    try:
        # 통합 리포트 생성
        report = integrator.generate_integrated_report(
            news_analysis_file, 
            non_financial_file
        )
        
        if "error" not in report:
            # JSON 리포트 저장
            json_path = integrator.save_integrated_report(report)
            
            # Excel 요약 리포트 생성 (옵션)
            
            print("\n" + "="*60)
            print("🎉 이상치 분석 결과 통합 완료!")
            print(f"📋 JSON 리포트: {json_path}")

            print("="*60)
            
            # 간단한 요약 출력
            summary = report.get("요약_통계", {})
            print(f"\n📊 요약:")
            print(f"   - 총 이상치: {summary.get('총_이상치_수', 0)}개")
            print(f"   - 평균 뉴스 신뢰도: {summary.get('평균_뉴스_신뢰도', 0):.2f}")
            print(f"   - 평균 비재무 신뢰도: {summary.get('평균_비재무_신뢰도', 0):.2f}")
            
        else:
            print(f"❌ 통합 분석 실패: {report.get('error')}")
            
    except Exception as e:
        print(f"❌ 실행 중 오류 발생: {str(e)}")


if __name__ == "__main__":
    main()
