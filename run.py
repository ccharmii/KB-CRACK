# /KB-CRACK/run.py
# 재무·비재무·뉴스 분석을 통합 실행하고 최종 리포트를 생성

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 신용위험 분석 시스템
재무지표 및 비재무지표 이상치 탐지와 근거 생성, 뉴스 이상징후 탐지, 최종 리포트 생성을 수행
"""

import argparse
import json
import os
import sys
from datetime import datetime, date
from pathlib import Path

from financial_analysis.anomaly_integration_code import AnomalyIntegrator
from financial_analysis.fin_news_reason import run_anomaly_news_analysis
from financial_analysis.main import analyze_corporation, extract_financial_anomalies
from news_analysis.news_search import CreditRiskNewsAnalyzer
from non_financial_analysis.explainer import run_anomaly_explainer_min
from non_financial_analysis.main import run_for_corp
from report_generator import generate_final_report


project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))


class IntegratedCreditRiskAnalyzer:
    """통합 신용위험 분석을 단계별로 수행하는 클래스"""

    def __init__(self, corp_name: str, config_path: str | None = None):
        """
        분석기 초기화

        입력: corp_name(기업명), config_path(설정 파일 경로)
        """
        self.corp_name = corp_name
        self.config = self._load_config(config_path)
        self.results = {}
        self.output_dir = Path(f"analysis_results/{self.corp_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self, config_path: str | None):
        """설정 파일을 로드하고 기본값과 병합"""
        default_config = {
            "financial_analysis": {
                "krx_file_path": "financial_analysis/업종분류현황_250809.csv",
                "n_years": 2,
                "max_peers": 5,
            },
            "non_financial_analysis": {
                "data_dir": "./data",
                "force_rerun": False,
            },
            "news_analysis": {
                "max_search_results": 10,
            },
            "output": {
                "save_intermediate": True,
                "report_format": "json",
            },
        }

        if config_path and os.path.exists(config_path):
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)

            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                    continue

                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue

            return config

        return default_config

    def run_financial_analysis(self, corp_name: str, current_grade: str):
        """
        재무지표 이상치 탐지 실행
        입력: corp_name(기업명), current_grade(현재 신용등급)
        출력: 성공 여부와 이상치 목록을 포함한 딕셔너리 또는 None
        """
        print("=" * 60)
        print("1. 재무지표 이상치 탐지 시작")
        print("=" * 60)

        try:
            config = self.config["financial_analysis"]

            financial_results = analyze_corporation(
                output_dir=self.output_dir,
                corp_name=corp_name,
                file_path=config["krx_file_path"],
                n_years=config["n_years"],
                max_peers=config["max_peers"],
            )

            if "error" in financial_results:
                print(f"재무지표 분석 실패: {financial_results['error']}")
                return None

            financial_results["기업_정보"]["Current_credit_grade"] = current_grade
            self.results["financial_analysis"] = financial_results

            with open(self.output_dir / "financial_analysis.json", "w", encoding="utf-8") as f:
                json.dump(financial_results, f, ensure_ascii=False, indent=4)

            anomalies = extract_financial_anomalies(financial_results, self.output_dir)
            if anomalies:
                print(f"{corp_name} 재무지표 분석 완료")
                print(f"이상치 탐지: {len(anomalies)}개")
            else:
                print(f"{corp_name} 재무지표 분석 완료")
                print("이상치 탐지: 없음")

            return {
                "success": True,
                "anomalies": anomalies,
                "total_anomalies": len(anomalies) if anomalies else 0,
            }

        except Exception as e:
            print(f"재무지표 분석 중 오류: {str(e)}")
            return None

    def run_non_financial_analysis(self, corp_code: str):
        """
        비재무지표 이상치 탐지 실행
        입력: corp_code(DART 기업코드)
        출력: 성공 여부와 최신 분기 및 위험요약을 포함한 딕셔너리 또는 None
        """
        print("\n" + "=" * 60)
        print("2. 비재무지표 이상치 탐지 시작")
        print("=" * 60)

        try:
            config = self.config["non_financial_analysis"]

            print(f"{corp_code} 비재무지표 분석 시작")
            nfr_results = run_for_corp(
                corp_code=corp_code,
                asof=date.today(),
                force=config["force_rerun"],
            )

            evaluation_results = nfr_results.get("evaluation_results_by_quarter", [])
            latest_quarter = nfr_results.get("latest_quarter")

            self.results["non_financial_analysis"] = evaluation_results

            if self.config["output"]["save_intermediate"]:
                with open(self.output_dir / "non_financial_analysis.json", "w", encoding="utf-8") as f:
                    json.dump(nfr_results, f, ensure_ascii=False, indent=2)

            with open(self.output_dir / "non_financial_analysis_last_quater.json", "w", encoding="utf-8") as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

            print("비재무지표 분석 완료")
            print(f"분석된 분기: {len(nfr_results.get('analyzed_quarters', []))}개")
            print(f"최신 분기: {latest_quarter}")
            print(f"위험수준 요약: {nfr_results.get('risk_summary', {}).get('overall_risk_level', 'Unknown')}")

            return {
                "success": True,
                "latest_quarter": latest_quarter,
                "anomalies": evaluation_results,
                "analyzed_quarters": nfr_results.get("analyzed_quarters", []),
                "risk_summary": nfr_results.get("risk_summary", {}),
            }

        except Exception as e:
            print(f"비재무지표 분석 중 오류: {str(e)}")
            return None

    def run_financial_reasoning(self, corp_code: str):
        """
        재무지표 이상치 근거 생성 수행
        입력: corp_code(DART 기업코드)
        출력: 통합 이상치 분석 목록 또는 None
        """
        print("\n" + "=" * 60)
        print("3. 재무지표 이상치 근거 생성 시작")
        print("=" * 60)

        anomalies_path = self.output_dir / "financial_anomalies.json"
        anomalies = {}

        try:
            if anomalies_path.exists():
                with open(anomalies_path, "r", encoding="utf-8") as f:
                    raw_anomalies = json.load(f)

                for metric, details in raw_anomalies.items():
                    if isinstance(details, dict) and "description" in details:
                        anomalies[metric] = details["description"]
                    else:
                        anomalies[metric] = str(details)
            else:
                print("재무지표 이상치 데이터를 찾을 수 없음")

        except Exception as e:
            print(f"재무지표 이상치 데이터 로드 중 오류: {str(e)}")

        try:
            if anomalies:
                script_dir = os.path.dirname(os.path.abspath(__file__))
                nonf_reason = run_anomaly_explainer_min(
                    anomalies_json_or_dict=anomalies,
                    corp_code=corp_code,
                    verbose=True,
                    script_dir=script_dir,
                )
                nonf_reason["success"] = True

                if nonf_reason.get("success"):
                    with open(self.output_dir / "non_financial_reasoning.json", "w", encoding="utf-8") as f:
                        json.dump(nonf_reason, f, ensure_ascii=False, indent=2)
                    print("비재무 정보 기반 근거 생성 완료")
                else:
                    print("비재무 정보 기반 근거 생성 실패")
            else:
                print("분석할 재무지표 이상치가 없어 비재무 근거 분석 생략")

        except Exception as e:
            print(f"비재무 정보 기반 근거 생성 중 오류: {str(e)}")

        try:
            run_anomaly_news_analysis(str(self.output_dir))

            anomaly_news_path = self.output_dir / "anomaly_news_analysis.json"
            if anomaly_news_path.exists():
                with open(anomaly_news_path, "r", encoding="utf-8") as f:
                    fin_news_reason = json.load(f)
            else:
                fin_news_reason = None
                print("뉴스 기반 근거 파일을 찾을 수 없음")

            if fin_news_reason and fin_news_reason.get("success"):
                print("뉴스 기반 근거 생성 완료")
                print(f"{len(anomalies)}개 이상치에 대한 뉴스 근거 분석 완료")
            else:
                print("뉴스 기반 근거 생성 실패")

        except Exception as e:
            print(f"뉴스 기반 근거 생성 중 오류: {str(e)}")

        integrator = AnomalyIntegrator(output_dir=self.output_dir)
        report = integrator.generate_integrated_report(
            news_analysis_path=self.output_dir / "anomaly_news_analysis.json",
            non_financial_path=self.output_dir / "non_financial_reasoning.json",
        )
        integrator.save_integrated_report(report)

        integrated_report_path = self.output_dir / "integrated_anomaly_report.json"
        if integrated_report_path.exists():
            with open(integrated_report_path, "r", encoding="utf-8") as f:
                integrated_report = json.load(f)
            self.results["financial_reasoning"] = integrated_report.get("통합_이상치_분석")
        else:
            self.results["financial_reasoning"] = []

        print("재무지표 이상치 근거 생성 완료")
        return (integrated_report.get("통합_이상치_분석") if "integrated_report" in locals() else None)

    def run_news_analysis(self):
        """
        뉴스 이상징후 탐지 실행
        출력: 뉴스 분석 결과 딕셔너리 또는 None
        """
        print("\n" + "=" * 60)
        print("4. 뉴스 이상징후 탐지 시작")
        print("=" * 60)

        try:
            config = self.config["news_analysis"]

            analyzer = CreditRiskNewsAnalyzer(max_search_results=config["max_search_results"])
            news_results = analyzer.analyze_credit_risk_with_results(str(self.output_dir))

            self.results["news_analysis"] = news_results

            total_news = news_results.get("total_news_count", 0)
            financial_anomalies = news_results.get("financial_anomalies_count", 0)
            nfr_anomalies = news_results.get("non_financial_anomalies_count", 0)

            print(f"뉴스 분석 완료: {total_news}개 뉴스 분석")
            print(f"기반 이상치: 재무 {financial_anomalies}개, 비재무 {nfr_anomalies}개")

            return news_results

        except Exception as e:
            print(f"뉴스 분석 중 오류: {str(e)}")
            import traceback

            print(f"상세 오류: {traceback.format_exc()}")
            return None

    def generate_final_report(self):
        """
        결과 디렉터리 기반 최종 종합 리포트 생성 수행
        출력: 최종 리포트 딕셔너리 또는 None
        """
        print("\n" + "=" * 60)
        print("5. 최종 종합 리포트 생성 시작")
        print("=" * 60)

        try:
            final_report = generate_final_report(str(self.output_dir))
            self.results["final_report"] = final_report

            report_path = self.output_dir / "final_integrated_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)

            summary_path = self.output_dir / "executive_summary.md"
            with open(summary_path, "w", encoding="utf-8") as f:
                f.write(final_report.get("executive_summary", ""))

            detailed_path = self.output_dir / "detailed_analysis.md"
            with open(detailed_path, "w", encoding="utf-8") as f:
                f.write(final_report.get("detailed_analysis", ""))

            company_name = final_report.get("metadata", {}).get("company_name", "Unknown")
            risk_assessment = final_report.get("risk_assessment", {})
            risk_grade = risk_assessment.get("grade", "Unknown")
            risk_score = risk_assessment.get("score", 0)

            print("최종 리포트 생성 완료")
            print(f"기업: {company_name}")
            print(f"신용등급: {risk_grade}")
            print(f"위험점수: {risk_score}/100")
            print(f"최종 리포트: {report_path}")
            print(f"요약 리포트: {summary_path}")
            print(f"상세 리포트: {detailed_path}")

            return final_report

        except Exception as e:
            print(f"최종 리포트 생성 중 오류: {str(e)}")
            return None

    def run_full_analysis(self, corp_name: str, current_grade: str, corp_code: str | None = None):
        """
        전체 분석 파이프라인 실행
        입력: corp_name(기업명), current_grade(현재 신용등급), corp_code(DART 기업코드)
        출력: results(단계별 결과), analysis_summary(실행 요약)
        """
        print(f"{corp_name} 통합 신용위험 분석 시작")
        print(f"결과 저장 경로: {self.output_dir}")
        print("=" * 80)

        start_time = datetime.now()
        analysis_summary = {
            "company_name": corp_name,
            "current_credit_grade": current_grade,
            "corp_code": corp_code,
            "start_time": start_time.isoformat(),
            "output_directory": str(self.output_dir),
            "steps_completed": [],
            "steps_failed": [],
        }

        financial_result = self.run_financial_analysis(corp_name, current_grade)
        if financial_result and financial_result.get("success"):
            analysis_summary["steps_completed"].append("financial_analysis")
        else:
            analysis_summary["steps_failed"].append("financial_analysis")

        if corp_code:
            nfr_result = self.run_non_financial_analysis(corp_code)
            if nfr_result and nfr_result.get("success"):
                analysis_summary["steps_completed"].append("non_financial_analysis")
                analysis_summary["latest_quarter"] = nfr_result.get("latest_quarter")
            else:
                analysis_summary["steps_failed"].append("non_financial_analysis")
        else:
            print("corp_code가 제공되지 않아 비재무지표 분석 생략")
            nfr_result = None

        if corp_code:
            reasoning_result = self.run_financial_reasoning(corp_code)
            if reasoning_result:
                analysis_summary["steps_completed"].append("financial_reasoning")
            else:
                analysis_summary["steps_failed"].append("financial_reasoning")
        else:
            print("corp_code가 제공되지 않아 근거 분석 생략")
            reasoning_result = None

        news_result = self.run_news_analysis()
        if news_result:
            analysis_summary["steps_completed"].append("news_analysis")
        else:
            analysis_summary["steps_failed"].append("news_analysis")

        final_report = self.generate_final_report()
        if final_report:
            analysis_summary["steps_completed"].append("final_report")
        else:
            analysis_summary["steps_failed"].append("final_report")

        end_time = datetime.now()
        duration = end_time - start_time

        analysis_summary["end_time"] = end_time.isoformat()
        analysis_summary["total_duration"] = str(duration)

        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(analysis_summary, f, ensure_ascii=False, indent=2)

        print("\n" + "=" * 80)
        print("통합 분석 완료")
        print(f"총 소요시간: {duration}")
        print(f"분석 요약: {summary_path}")
        print(f"결과 저장 경로: {self.output_dir}")

        return self.results, analysis_summary


def launch_dashboard():
    """Flask 기반 대시보드 실행을 보조하는 함수"""
    import subprocess

    try:
        import flask  # noqa: F401
    except ImportError:
        print("Flask 설치 필요")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])

    if not os.path.exists("templates"):
        os.makedirs("templates", exist_ok=True)

    print("\n" + "=" * 60)
    print("KB 국민은행 AI 신용위험 분석 대시보드를 시작합니다")
    print("웹 브라우저 접속 주소: http://127.0.0.1:5000/")
    print("종료는 Ctrl+C")
    print("=" * 60 + "\n")

    try:
        subprocess.run([sys.executable, "dashboard_app.py"])
    except KeyboardInterrupt:
        print("대시보드 종료")
    except Exception as e:
        print(f"대시보드 실행 중 오류: {e}")


def main():
    """커맨드라인 인자 기반 통합 분석 실행"""
    parser = argparse.ArgumentParser(description="통합 신용위험 분석 시스템")
    parser.add_argument("--company", "-c", required=True, help="분석할 회사명")
    parser.add_argument("--config", help="설정 파일 경로")
    parser.add_argument("--output_dir", "-o", help="결과 저장 디렉터리")

    args = parser.parse_args()

    try:
        analyzer = IntegratedCreditRiskAnalyzer(corp_name=args.company, config_path=args.config)

        if args.output_dir:
            analyzer.output_dir = Path(args.output_dir)
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)

        from financial_analysis.load_corpinfo import CorpInfo

        try:
            with open("./dataset/credit_grade_fake.json", "r", encoding="utf-8") as f:
                grade_data = json.load(f)

            if args.company not in grade_data:
                print(f"신용등급 데이터에 '{args.company}' 정보가 없어 기본값 사용")
                current_grade = "B-"
            else:
                current_grade = grade_data[args.company].get("current_grade", "B-")

        except FileNotFoundError:
            print("신용등급 데이터 파일을 찾을 수 없어 기본값 사용")
            current_grade = "B-"

        results, summary = analyzer.run_full_analysis(
            corp_name=args.company,
            current_grade=current_grade,
            corp_code=CorpInfo(args.company).corp_code,
        )

        failed_count = len(summary.get("steps_failed", []))
        if failed_count == 0:
            launch_dashboard()
            return 0

        if failed_count < 3:
            return 0

        return 1

    except KeyboardInterrupt:
        print("사용자에 의해 분석 중단")
        return 1

    except Exception as e:
        print(f"예상치 못한 오류 발생: {str(e)}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    main()
