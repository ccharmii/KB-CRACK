# /KB-CRACK/dashboard_server.py
# 정적 대시보드와 분석 결과 API를 제공하는 HTTP 서버 파일

import json
import os
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from urllib.parse import unquote, urlparse


class DashboardHandler(SimpleHTTPRequestHandler):
    """대시보드 정적 파일 서빙과 기업 분석 결과 API 라우팅 처리"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)

    def do_GET(self):
        """요청 경로에 따라 API 응답 또는 정적 파일 서빙 수행"""
        parsed_path = urlparse(self.path)

        if parsed_path.path == "/api/companies":
            self.send_companies_data()
            return

        if parsed_path.path.startswith("/api/company/"):
            company_name = parsed_path.path.split("/")[-1]
            self.send_company_detail(company_name)
            return

        super().do_GET()

    def send_companies_data(self) -> None:
        """기업 목록 요약 데이터 JSON 응답 수행"""
        try:
            companies = []
            analysis_dir = Path("analysis_results")

            if analysis_dir.exists():
                for company_dir in analysis_dir.iterdir():
                    if not company_dir.is_dir():
                        continue

                    company_data = self.load_company_summary(company_dir)
                    if company_data:
                        companies.append(company_data)

            self.send_json_response(companies)
        except Exception as e:
            self.send_error_response(str(e))

    def send_company_detail(self, company_name: str) -> None:
        """기업 상세 데이터 JSON 응답 수행"""
        try:
            company_name_decoded = unquote(company_name)
            company_dir = Path("analysis_results") / company_name_decoded
            detail_data = self.load_company_detail(company_dir)
            self.send_json_response(detail_data)
        except Exception as e:
            self.send_error_response(str(e))

    def load_company_summary(self, company_dir: Path) -> Dict[str, object] | None:
        """기업 요약 데이터를 파일에서 로드 및 변환 수행"""
        try:
            comprehensive_file = company_dir / "final_comprehensive_report.json"
            daily_file = company_dir / "daily_news_risk_analysis.json"

            if not comprehensive_file.exists():
                return None

            with open(comprehensive_file, "r", encoding="utf-8") as f:
                comprehensive_data = json.load(f)

            daily_data = {}
            if daily_file.exists():
                with open(daily_file, "r", encoding="utf-8") as f:
                    daily_data = json.load(f)

            return {
                "name": comprehensive_data.get("기업_정보", {}).get("기업명", company_dir.name),
                "currentGrade": comprehensive_data.get("기업_정보", {}).get("Current_credit_grade", "N/A"),
                "predictedGrade": comprehensive_data.get("예상_신용등급_변화", {}).get("예상_등급", "N/A"),
                "riskLevel": daily_data.get("company_analysis_summary", {}).get("overall_risk_level", "보통"),
                "anomalies": len(comprehensive_data.get("재무지표_분석", {}).get("이상치_목록", {})),
                "lastAnalysis": (comprehensive_data.get("분석_메타데이터", {}).get("생성_시간", "") or "")[:10],
            }
        except Exception as e:
            print(f"Error loading company summary for {company_dir.name}: {e}")
            return None

    def load_company_detail(self, company_dir: Path) -> Dict[str, object]:
        """기업 상세 데이터를 파일에서 로드 및 변환 수행"""
        try:
            comprehensive_file = company_dir / "final_comprehensive_report.json"
            daily_file = company_dir / "daily_news_risk_analysis.json"

            with open(comprehensive_file, "r", encoding="utf-8") as f:
                comprehensive_data = json.load(f)

            daily_data = {}
            if daily_file.exists():
                with open(daily_file, "r", encoding="utf-8") as f:
                    daily_data = json.load(f)

            financial_anomalies = []
            for metric, data in comprehensive_data.get("재무지표_분석", {}).get("이상치_목록", {}).items():
                financial_anomalies.append(
                    {
                        "metric": metric,
                        "description": data.get("description", ""),
                        "severity": data.get("severity", "Medium"),
                    }
                )

            non_financial_risks = []
            for risk in comprehensive_data.get("비재무지표_분석", {}).get("탐지된_이상치", []):
                score = risk.get("score", 3)
                if score <= 2:
                    level = "high"
                elif score <= 3:
                    level = "medium"
                else:
                    level = "low"

                non_financial_risks.append(
                    {
                        "indicator": risk.get("indicator_name", ""),
                        "rationale": risk.get("rationale", ""),
                        "grade": risk.get("grade_label", ""),
                        "level": level,
                    }
                )

            news_analysis = []
            for news in daily_data.get("relevant_news", [])[:5]:
                content = news.get("content", "") or ""
                news_analysis.append(
                    {
                        "title": news.get("title", ""),
                        "summary": (content[:200] + "...") if content else "",
                        "url": news.get("url", "#"),
                    }
                )

            historical_cases = []
            for case in comprehensive_data.get("유사_사례_분석", []):
                historical_cases.append(
                    {
                        "company": case.get("company", ""),
                        "year": case.get("year", ""),
                        "anomalyType": case.get("anomaly_type", ""),
                        "initialGrade": case.get("initial_grade", ""),
                        "finalGrade": case.get("final_grade", ""),
                        "recoveryPeriod": case.get("recovery_period", ""),
                        "actions": case.get("actions_taken", []),
                    }
                )

            return {
                "aiSummary": comprehensive_data.get("AI_종합_분석_요약", ""),
                "currentGrade": comprehensive_data.get("기업_정보", {}).get("Current_credit_grade", "N/A"),
                "predictedGrade": comprehensive_data.get("예상_신용등급_변화", {}).get("예상_등급", "N/A"),
                "gradeReason": comprehensive_data.get("예상_신용등급_변화", {}).get("등급_변화_사유", ""),
                "financialAnomalies": financial_anomalies,
                "nonFinancialRisks": non_financial_risks,
                "newsAnalysis": news_analysis,
                "historicalCases": historical_cases,
            }
        except Exception as e:
            print(f"Error loading company detail: {e}")
            return {"error": str(e)}

    def send_json_response(self, data: object) -> None:
        """JSON 응답 헤더 설정 및 바디 전송 수행"""
        self.send_response(200)
        self.send_header("Content-type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode("utf-8"))

    def send_error_response(self, error_msg: str) -> None:
        """에러 JSON 응답 전송 수행"""
        self.send_response(500)
        self.send_header("Content-type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"error": error_msg}, ensure_ascii=False).encode("utf-8"))


def start_dashboard_server() -> None:
    """HTTP 대시보드 서버 시작 및 브라우저 자동 오픈"""
    port = 8000
    server = HTTPServer(("localhost", port), DashboardHandler)
    print(f"대시보드 서버가 http://localhost:{port} 에서 실행 중입니다...")

    def open_browser() -> None:
        """대시보드 주소를 기본 브라우저로 오픈"""
        time.sleep(1)
        webbrowser.open(f"http://localhost:{port}")

    threading.Thread(target=open_browser, daemon=True).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n대시보드 서버를 종료합니다.")
        server.shutdown()
