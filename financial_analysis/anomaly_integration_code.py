# /financial_analysis/anomaly_integration_code.py
# 이상치 분석 결과를 뉴스 및 비재무 근거와 통합하여 JSON 리포트로 생성


import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


class AnomalyIntegrator:
    """이상치 분석 결과 통합 처리 클래스"""

    def __init__(self, output_dir: Optional[str] = None):
        """
        통합 결과 저장 경로 초기화 수행
        Args:
            output_dir: 통합 결과 저장 디렉토리 경로 문자열
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./integrated_results")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def load_analysis_files(self, news_analysis_path: str, non_financial_path: str) -> tuple[Optional[Dict], Optional[Dict]]:
        """
        뉴스 분석 결과와 비재무 분석 결과 JSON 로드 수행
        Args:
            news_analysis_path: 뉴스 분석 결과 JSON 파일 경로 문자열
            non_financial_path: 비재무 분석 결과 JSON 파일 경로 문자열
        Returns:
            (news_data, non_financial_data) 튜플 반환
        """
        try:
            with open(news_analysis_path, "r", encoding="utf-8") as f:
                news_data = json.load(f)

            with open(non_financial_path, "r", encoding="utf-8") as f:
                non_financial_data = json.load(f)

            return news_data, non_financial_data
        except Exception:
            return None, None

    def extract_company_info(self, news_data: Dict[str, Any]) -> Dict[str, str]:
        """
        뉴스 분석 데이터에서 기업 기본 정보 추출 수행
        Args:
            news_data: 뉴스 분석 결과 딕셔너리
        Returns:
            기업 정보 딕셔너리 반환
        """
        company_info = news_data.get("company_info", {})

        return {
            "기업명_한글": company_info.get("기업명", ""),
            "기업명_영문": company_info.get("영문기업명", ""),
            "종목코드": company_info.get("종목코드", ""),
            "대표자명": company_info.get("대표자명", ""),
            "업종": company_info.get("업종", ""),
            "신용등급": company_info.get("Current_credit_grade", ""),
            "주소": company_info.get("주소", ""),
        }

    def match_anomalies_by_metric(
        self,
        news_anomalies: List[Dict[str, Any]],
        non_financial_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        메트릭명을 기준으로 뉴스 이상치와 비재무 결과 매칭 및 통합 수행
        Args:
            news_anomalies: 뉴스 기반 이상치 목록
            non_financial_results: 비재무 분석 결과 목록
        Returns:
            통합 이상치 목록 반환
        """
        nf_by_metric: Dict[str, Dict[str, Any]] = {}
        for nf_result in non_financial_results:
            metric = nf_result.get("metric", "")
            if metric:
                nf_by_metric[metric] = nf_result

        integrated_anomalies: List[Dict[str, Any]] = []
        for news_anomaly in news_anomalies:
            anomaly_info = news_anomaly.get("anomaly_info", {})
            metric_name = anomaly_info.get("metric_name", "")
            matching_nf = nf_by_metric.get(metric_name)

            integrated_anomalies.append(
                self._create_integrated_anomaly(news_anomaly, matching_nf, metric_name)
            )

        return integrated_anomalies

    def _create_integrated_anomaly(
        self,
        news_anomaly: Dict[str, Any],
        non_financial_result: Optional[Dict[str, Any]],
        metric_name: str,
    ) -> Dict[str, Any]:
        """
        개별 이상치 단위의 통합 결과 생성 수행
        Args:
            news_anomaly: 뉴스 분석 이상치 단위 결과
            non_financial_result: 매칭된 비재무 분석 단위 결과
            metric_name: 메트릭명 문자열
        Returns:
            개별 이상치 통합 결과 딕셔너리 반환
        """
        anomaly_info = news_anomaly.get("anomaly_info", {})
        analysis = news_anomaly.get("analysis", {})
        news_evidence = news_anomaly.get("news_evidence", [])

        result: Dict[str, Any] = {
            "메트릭명": metric_name,
            "이상치_설명": anomaly_info.get("description", ""),
            "심각도": anomaly_info.get("severity", ""),
            "분기": anomaly_info.get("quarter", ""),
            "탐지_유형": anomaly_info.get("type", ""),
            "분석_시점": datetime.now().isoformat(),
        }

        result["뉴스_분석"] = {
            "주요_원인": analysis.get("primary_cause", ""),
            "신뢰도": analysis.get("confidence_level", 0),
            "상세_설명": analysis.get("detailed_explanation", ""),
            "영향_평가": analysis.get("impact_assessment", ""),
            "위험_수준": analysis.get("risk_level", ""),
            "지원_증거": analysis.get("supporting_evidence", []),
            "뉴스_소스": analysis.get("news_sources", []),
            "관련성_품질점수": analysis.get("relevance_quality", ""),
        }

        result["뉴스_증거"] = [
            {
                "제목": news.get("title", ""),
                "URL": news.get("url", ""),
                "발행일": news.get("published_date", ""),
                "출처": news.get("source", ""),
                "관련성_점수": news.get("hybrid_score", 0),
                "내용_요약": (news.get("content", "")[:200] + "...") if news.get("content") else "",
            }
            for news in news_evidence
        ]

        if non_financial_result:
            nf_analysis: Dict[str, Any] = {
                "설명": non_financial_result.get("explanation_ko", ""),
                "주요_원인": non_financial_result.get("drivers", []),
                "신뢰도": non_financial_result.get("confidence", 0),
                "관련_보고서": [],
                "관련_문장": [],
            }

            for ev in non_financial_result.get("evidence", []):
                nf_analysis["관련_보고서"].append(
                    {
                        "보고서_번호": ev.get("rcept_no", ""),
                        "문서_ID": ev.get("chunk_id", ""),
                        "관련_문장": ev.get("snippet", ""),
                        "소스_인덱스": ev.get("source_idx", ""),
                    }
                )
                nf_analysis["관련_문장"].append(ev.get("snippet", ""))

            result["비재무_분석"] = nf_analysis
        else:
            result["비재무_분석"] = {
                "설명": "매칭되는 비재무 분석 결과 부재",
                "주요_원인": [],
                "신뢰도": 0,
                "관련_보고서": [],
                "관련_문장": [],
            }

        return result

    def create_summary_statistics(self, integrated_anomalies: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        통합 이상치 목록 기반 요약 통계 산출 수행
        Args:
            integrated_anomalies: 통합 이상치 목록
        Returns:
            요약 통계 딕셔너리 반환
        """
        total_anomalies = len(integrated_anomalies)

        severity_count: Dict[str, int] = {}
        metric_count: Dict[str, int] = {}
        news_confidence_scores: List[float] = []
        nf_confidence_scores: List[float] = []

        for anomaly in integrated_anomalies:
            severity = anomaly.get("심각도", "Unknown")
            severity_count[severity] = severity_count.get(severity, 0) + 1

            metric = anomaly.get("메트릭명", "Unknown")
            metric_count[metric] = metric_count.get(metric, 0) + 1

            news_conf = anomaly.get("뉴스_분석", {}).get("신뢰도", 0)
            if isinstance(news_conf, (int, float)) and news_conf > 0:
                news_confidence_scores.append(float(news_conf))

            nf_conf = anomaly.get("비재무_분석", {}).get("신뢰도", 0)
            if isinstance(nf_conf, (int, float)) and nf_conf > 0:
                nf_confidence_scores.append(float(nf_conf))

        avg_news_conf = sum(news_confidence_scores) / len(news_confidence_scores) if news_confidence_scores else 0
        avg_nf_conf = sum(nf_confidence_scores) / len(nf_confidence_scores) if nf_confidence_scores else 0

        return {
            "총_이상치_수": total_anomalies,
            "심각도별_분포": severity_count,
            "메트릭별_분포": metric_count,
            "평균_뉴스_신뢰도": avg_news_conf,
            "평균_비재무_신뢰도": avg_nf_conf,
            "뉴스_증거_보유_이상치": sum(1 for a in integrated_anomalies if a.get("뉴스_증거")),
            "비재무_증거_보유_이상치": sum(
                1 for a in integrated_anomalies if a.get("비재무_분석", {}).get("관련_보고서")
            ),
        }

    def generate_integrated_report(self, news_analysis_path: str, non_financial_path: str) -> Dict[str, Any]:
        """
        뉴스 및 비재무 분석 결과를 통합 리포트로 생성 수행
        Args:
            news_analysis_path: 뉴스 분석 결과 JSON 파일 경로 문자열
            non_financial_path: 비재무 분석 결과 JSON 파일 경로 문자열
        Returns:
            통합 리포트 딕셔너리 반환
        """
        news_data, non_financial_data = self.load_analysis_files(news_analysis_path, non_financial_path)
        if not news_data or not non_financial_data:
            return {"error": "분석 파일 로드 실패"}

        company_info = self.extract_company_info(news_data)

        news_anomalies = news_data.get("anomaly_news_analyses", [])
        non_financial_results = non_financial_data.get("results", [])

        integrated_anomalies = self.match_anomalies_by_metric(news_anomalies, non_financial_results)
        summary_stats = self.create_summary_statistics(integrated_anomalies)

        return {
            "분석_메타데이터": {
                "생성_시간": datetime.now().isoformat(),
                "뉴스_분석_파일": str(news_analysis_path),
                "비재무_분석_파일": str(non_financial_path),
                "분석_방법론": {
                    "뉴스_분석": news_data.get("methodology", {}),
                    "통합_방식": "메트릭명 기준 매칭",
                },
            },
            "기업_정보": company_info,
            "요약_통계": summary_stats,
            "통합_이상치_분석": integrated_anomalies,
        }

    def save_integrated_report(self, report: Dict[str, Any], filename: Optional[str] = None) -> str:
        """
        통합 리포트 JSON 저장 수행
        Args:
            report: 통합 리포트 딕셔너리
            filename: 저장 파일명 문자열
        Returns:
            저장된 파일 경로 문자열 반환
        """
        if not filename:
            filename = "integrated_anomaly_report.json"

        filepath = self.output_dir / filename

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

        return str(filepath)


def main() -> None:
    """뉴스 및 비재무 결과 통합 리포트 생성 실행 예시"""
    news_analysis_file = "analysis_results/삼성전자/anomaly_news_analysis.json"
    non_financial_file = "analysis_results/삼성전자/non_financial_reasoning.json"

    integrator = AnomalyIntegrator(output_dir="./삼성전자")

    report = integrator.generate_integrated_report(news_analysis_file, non_financial_file)
    if "error" in report:
        return

    json_path = integrator.save_integrated_report(report)
    summary = report.get("요약_통계", {})

    print(f"JSON 리포트 저장 경로 {json_path}")
    print(f"총 이상치 수 {summary.get('총_이상치_수', 0)}개")
    print(f"평균 뉴스 신뢰도 {summary.get('평균_뉴스_신뢰도', 0):.2f}")
    print(f"평균 비재무 신뢰도 {summary.get('평균_비재무_신뢰도', 0):.2f}")


if __name__ == "__main__":
    main()
