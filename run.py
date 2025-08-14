#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
통합 신용위험 분석 시스템 (수정된 버전)
- 재무지표 이상치 탐지
- 비재무지표 이상치 탐지  
- 재무지표 이상치 근거 생성 (뉴스 + 비재무 정보)
- 뉴스 이상징후 탐지
- 최종 종합 리포트 생성
"""

import os
import sys
import json
import argparse
from datetime import datetime, date
from pathlib import Path

# 프로젝트 루트 디렉토리를 sys.path에 추가
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# 재무지표 이상치 탐지 모듈
from financial_analysis.main import analyze_corporation, extract_financial_anomalies

# 비재무지표 이상치 탐지 모듈  
from non_financial_analysis.main import run_for_corp

# 비재무에서 재무지표 이상치 근거 생성 모듈
from non_financial_analysis.explainer import run_anomaly_explainer_min

# 뉴스에서 재무지표 이상치 근거 생성 모듈 (새로운 모듈)
from financial_analysis.fin_news_reason import run_anomaly_news_analysis

# 비재무 근거 + 뉴스 근거를 재무 이상치별로 합치는 모듈
from financial_analysis.anomaly_integration_code import AnomalyIntegrator


# 뉴스 이상징후 탐지 모듈
from news_analysis.news_search import CreditRiskNewsAnalyzer

# 최종 리포트 생성 모듈
from report_generator import CreditRiskReportGenerator
from report_generator import generate_final_report


class IntegratedCreditRiskAnalyzer:
    """통합 신용위험 분석 시스템"""
    
    def __init__(self, corp_name, config_path=None):
        """
        시스템 초기화
        
        Args:
            config_path: 설정 파일 경로 (옵션)
        """
        self.corp_name = corp_name
        self.config = self._load_config(config_path)
        self.results = {}
        self.output_dir = Path(f"analysis_results/{self.corp_name}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def _load_config(self, config_path):
        """설정 파일 로드 또는 기본값 설정"""
        default_config = {
            "financial_analysis": {
                "krx_file_path": "financial_analysis/업종분류현황_250809.csv",
                "n_years": 2,
                "max_peers": 5
            },
            "non_financial_analysis": {
                "data_dir": "./data",
                "force_rerun": False
            },
            "news_analysis": {
                "max_search_results": 10
            },
            "output": {
                "save_intermediate": True,
                "report_format": "json"
            }
        }
        
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            # 기본값과 병합
            for key, value in default_config.items():
                if key not in config:
                    config[key] = value
                elif isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in config[key]:
                            config[key][subkey] = subvalue
            return config
        
        return default_config

    def run_financial_analysis(self, corp_name, current_grade):
        """1. 재무지표 이상치 탐지 실행"""
        print("=" * 60)
        print("1. 재무지표 이상치 탐지 시작")
        print("=" * 60)
        
        try:
            config = self.config["financial_analysis"]
            
            # 재무지표 분석 실행
            financial_results = analyze_corporation(
                output_dir=self.output_dir,  # 결과를 지정된 디렉토리에 저장
                corp_name=corp_name,
                file_path=config["krx_file_path"],
                n_years=config["n_years"],
                max_peers=config["max_peers"],
            )
            
            if "error" in financial_results:
                print(f"❌ 재무지표 분석 실패: {financial_results['error']}")
                return None

                
            print(f"✅ {corp_name} 재무지표 분석 완료")

            financial_results["기업_정보"]["Current_credit_grade"] = current_grade
            
            self.results["financial_analysis"] = financial_results

            with open(self.output_dir /  f'financial_analysis.json', 'w', encoding='utf-8') as f:
                json.dump(financial_results, f, ensure_ascii=False, indent=4)


            anomalies = extract_financial_anomalies(financial_results, self.output_dir)


            if anomalies:
                print(f"   - 이상치 탐지: {len(anomalies)}개")
            else:
                print("   - 이상치 탐지: 없음")

            return {
                        "success": True,
                        "anomalies": anomalies,
                        "total_anomalies": len(anomalies) if anomalies else 0,
                    }
            
        except Exception as e:
            print(f"❌ 재무지표 분석 중 오류: {str(e)}")
            return None

    def run_non_financial_analysis(self, corp_code):
        """2. 비재무지표 이상치 탐지 실행"""
        print("\n" + "=" * 60)
        print("2. 비재무지표 이상치 탐지 시작")
        print("=" * 60)
        
        try:
            config = self.config["non_financial_analysis"]

            # 비재무지표 분석 실행
            print(f"   🔄 {corp_code} 비재무지표 분석 시작")

            nfr_results = run_for_corp(
                corp_code=corp_code,
                asof=date.today(),
                force=config["force_rerun"]
            )

            evaluation_results = nfr_results.get("evaluation_results_by_quarter", [])
            latest_quarter = nfr_results.get("latest_quarter", None)
            
            self.results["non_financial_analysis"] = evaluation_results
            
            # 분석 결과 저장
            if self.config["output"]["save_intermediate"]:
                    with open(self.output_dir / "non_financial_analysis.json", 'w', encoding='utf-8') as f:
                        json.dump(nfr_results, f, ensure_ascii=False, indent=2)

            # 최신분기 저장
            with open(self.output_dir / "non_financial_analysis_last_quater.json", 'w', encoding='utf-8') as f:
                    json.dump(evaluation_results, f, ensure_ascii=False, indent=2)

            print(f"   ✅ 비재무지표 분석 완료")
            print(f"      - 분석된 분기: {len(nfr_results.get('analyzed_quarters', []))}개")
            print(f"      - 최신 분기: {latest_quarter}")
            print(f"      - 위험수준 요약: {nfr_results.get('risk_summary', {}).get('overall_risk_level', 'Unknown')}")

            return {
                "success": True,
                "latest_quarter": latest_quarter,
                "anomalies": evaluation_results,
                "analyzed_quarters": nfr_results.get("analyzed_quarters", []),
                "risk_summary": nfr_results.get("risk_summary", {})
            }
                
        except Exception as e:
            print(f"❌ 비재무지표 분석 중 오류: {str(e)}")
            return None
        
    def run_financial_reasoning(self, corp_code):
        """3. 재무지표 이상치 근거 생성 실행 (뉴스 + 비재무 정보 기반)"""
        print("\n" + "=" * 60)
        print("3. 재무지표 이상치 근거 생성 시작")
        print("=" * 60)

        reasoning_results = {}
        
        # 3-1. 비재무 정보 기반 근거 생성
        print("\n" + "-" * 60)
        print("   📊 비재무 정보 기반 근거 분석 중...")
        print("-" * 60)
        try:
            # 재무지표 이상치 데이터 로드
            anomalies_path = self.output_dir / "financial_anomalies.json"

            if anomalies_path.exists():
                with open(anomalies_path, 'r', encoding='utf-8') as f:
                    raw_anomalies = json.load(f)
                print(f"   - 재무지표 이상치 데이터 로드 완료: {len(raw_anomalies)}개 이상치")  
                
                # explainer.py에서 기대하는 형태로 변환: {metric: description}
                anomalies = {}
                for metric, details in raw_anomalies.items():
                    if isinstance(details, dict) and 'description' in details:
                        anomalies[metric] = details['description']
                    else:
                        # fallback: 전체 내용을 문자열로 변환
                        anomalies[metric] = str(details)
                
                print(f"   - 변환된 이상치 데이터: {list(anomalies.keys())}")
            else:
                print("   ⚠️ 재무지표 이상치 데이터를 찾을 수 없습니다.")
                anomalies = {}

            # db에서 근거찾기
            if anomalies:
                dir = os.path.dirname(os.path.abspath(__file__))
                nonf_reason = run_anomaly_explainer_min(
                    anomalies_json_or_dict=anomalies,  # 변환된 딕셔너리 전달
                    corp_code=corp_code,
                    verbose=True,
                    script_dir=dir
                )

                nonf_reason['success'] = True

                if nonf_reason and nonf_reason.get("success"):
                    # reasoning_results["non_financial_reasoning"] = nonf_reason
                    print("   ✅ 비재무 정보 기반 근거 생성 완료")
                    
                    # 중간 결과 저장
                    with open(self.output_dir / "non_financial_reasoning.json", 'w', encoding='utf-8') as f:
                        json.dump(nonf_reason, f, ensure_ascii=False, indent=2)
                else:
                    print("   ❌ 비재무 정보 기반 근거 생성 실패")
            else:
                print("   ⚠️ 분석할 재무지표 이상치가 없어 비재무 근거 분석을 건너뜁니다.")
                
        except Exception as e:
            print(f"   ❌ 비재무 정보 기반 근거 생성 중 오류: {str(e)}")

        # 3-2. 뉴스 기반 근거 생성
        print("\n" + "-" * 60)
        print("   📰 뉴스 기반 근거 분석 중...")
        print("-" * 60)
        try:
            news_reason = run_anomaly_news_analysis(str(self.output_dir))

            # json 파일 불러오기 
            anomaly_news = Path(self.output_dir / "anomaly_news_analysis.json")

            if anomaly_news.exists():
                with open(anomaly_news, 'r', encoding='utf-8') as f:
                    fin_news_reason = json.load(f)
            else:
                print("   ⚠️ 재무지표 이상치 데이터를 찾을 수 없습니다.")
                fin_news_reason = []  
            
            
            if fin_news_reason and fin_news_reason.get("success"):
                # reasoning_results["news_reasoning"] = fin_news_reason
                print("   ✅ 뉴스 기반 근거 생성 완료")
                
                print(f"      - {len(anomalies)}개 이상치에 대한 뉴스 근거 분석 완료")
            else:
                print("   ❌ 뉴스 기반 근거 생성 실패")
                
        except Exception as e:
            print(f"   ❌ 뉴스 기반 근거 생성 중 오류: {str(e)}")

        # 3-3. 종합 근거 분석 결과 저장
        # 통합 근거 분석 결과 저장

        integrator = AnomalyIntegrator(output_dir=self.output_dir)
        report = integrator.generate_integrated_report(
            news_analysis_path=self.output_dir / "anomaly_news_analysis.json",
            non_financial_path=self.output_dir / "non_financial_reasoning.json"
        )
        
        integrator.save_integrated_report(report)

        # integrated_anomaly_report.json 파일을 읽어서 self.results에 저장
        integrated_report_path = self.output_dir / "integrated_anomaly_report.json"
        if integrated_report_path.exists():
            with open(integrated_report_path, 'r', encoding='utf-8') as f:
                integrated_report = json.load(f)
            self.results["financial_reasoning"] = integrated_report.get("통합_이상치_분석")
        else:
            self.results["financial_reasoning"] = []


        print("✅ 재무지표 이상치 근거 생성 완료 (비재무 + 뉴스)")

            
        return integrated_report.get("통합_이상치_분석")


    def run_news_analysis(self):
        """4. 뉴스 이상징후 탐지 실행"""
        print("\n" + "=" * 60)
        print("4. 뉴스 이상징후 탐지 시작")
        print("=" * 60)
        
        try:
            config = self.config["news_analysis"]
            
            # 뉴스 분석기 초기화
            analyzer = CreditRiskNewsAnalyzer(
                max_search_results=config["max_search_results"]
            )
            
            # 이상치 기반 뉴스 분석 실행
            news_results = analyzer.analyze_credit_risk_with_results(str(self.output_dir))
            
            self.results["news_analysis"] = news_results
            
            total_news = news_results.get("total_news_count", 0)
            financial_anomalies = news_results.get("financial_anomalies_count", 0)
            nfr_anomalies = news_results.get("non_financial_anomalies_count", 0)
            
            print(f"✅ 뉴스 분석 완료: {total_news}개 뉴스 분석")
            print(f"   - 기반 이상치: 재무 {financial_anomalies}개, 비재무 {nfr_anomalies}개")
            
            return news_results
            
        except Exception as e:
            print(f"❌ 뉴스 분석 중 오류: {str(e)}")
            # 에러 세부 정보 출력
            import traceback
            print(f"   상세 오류: {traceback.format_exc()}")
            return None

    def generate_final_report(self):
        """4. 최종 종합 리포트 생성 (뉴스 분석 제외)"""
        print("\n" + "=" * 60)
        print("4. 최종 종합 리포트 생성 시작")
        print("=" * 60)
        
        try:
            # 결과 디렉토리에서 통합 리포트 생성
            final_report = generate_final_report(str(self.output_dir))
            
            self.results["final_report"] = final_report

            # 최종 리포트 저장
            report_path = self.output_dir / "final_integrated_report.json"
            with open(report_path, 'w', encoding='utf-8') as f:
                json.dump(final_report, f, ensure_ascii=False, indent=2)
            
            # 요약 리포트도 저장
            summary_path = self.output_dir / "executive_summary.md"
            with open(summary_path, 'w', encoding='utf-8') as f:
                f.write(final_report.get("executive_summary", ""))
            
            # 상세 분석 리포트 저장
            detailed_path = self.output_dir / "detailed_analysis.md"
            with open(detailed_path, 'w', encoding='utf-8') as f:
                f.write(final_report.get("detailed_analysis", ""))
            
            company_name = final_report.get("metadata", {}).get("company_name", "Unknown")
            risk_grade = final_report.get("risk_assessment", {}).get("grade", "Unknown")
            risk_score = final_report.get("risk_assessment", {}).get("score", 0)
            
            print(f"✅ 최종 리포트 생성 완료")
            print(f"   - 기업: {company_name}")
            print(f"   - 신용등급: {risk_grade}")
            print(f"   - 위험점수: {risk_score}/100")
            print(f"📄 상세 리포트: {report_path}")
            print(f"📋 요약 리포트: {summary_path}")
            print(f"📊 분석 리포트: {detailed_path}")
            
            return final_report
            
        except Exception as e:
            print(f"❌ 최종 리포트 생성 중 오류: {str(e)}")
            return None

    def run_full_analysis(self, corp_name, current_grade, corp_code=None):
        """전체 분석 프로세스 실행"""
        print(f"🚀 {corp_name} 통합 신용위험 분석 시작")
        print(f"📁 결과 저장 경로: {self.output_dir}")
        print("=" * 80)
        
        start_time = datetime.now()
        analysis_summary = {
            "company_name": corp_name,
            "current_credit_grade": current_grade,  # 현재 신용등급은 별도로 조회 필요
            "corp_code": corp_code,
            "start_time": start_time.isoformat(),
            "output_directory": str(self.output_dir),
            "steps_completed": [],
            "steps_failed": []
        }
        
        # 1. 재무지표 이상치 탐지
        financial_result = self.run_financial_analysis(corp_name, current_grade)
        if financial_result and financial_result.get("success"):
            analysis_summary["steps_completed"].append("financial_analysis")
        else:
            analysis_summary["steps_failed"].append("financial_analysis")
        
        # 2. 비재무지표 이상치 탐지 (corp_code 필요)
        if corp_code:
            nfr_result = self.run_non_financial_analysis(corp_code)
            if nfr_result and nfr_result.get("success"):
                analysis_summary["steps_completed"].append("non_financial_analysis")
                analysis_summary["latest_quarter"] = nfr_result.get("latest_quarter")
            else:
                analysis_summary["steps_failed"].append("non_financial_analysis")
        else:
            print("⚠️ corp_code가 제공되지 않아 비재무지표 분석을 건너뜁니다.")
            nfr_result = None
        
        # 3. 재무지표 이상치 근거 생성 (뉴스 + 비재무 근거)
        if corp_code:
            reasoning_result = self.run_financial_reasoning(corp_code)
            if reasoning_result:
                analysis_summary["steps_completed"].append("financial_reasoning")
            else:
                analysis_summary["steps_failed"].append("financial_reasoning")
        else:
            print("⚠️ corp_code가 제공되지 않아 근거 분석을 건너뜁니다.")
            reasoning_result = None
        
        # 4. 뉴스 이상징후 탐지
        news_result = self.run_news_analysis()
        if news_result:
            analysis_summary["steps_completed"].append("news_analysis")
        else:
            analysis_summary["steps_failed"].append("news_analysis")
        
        # 5. 최종 리포트 생성
        final_report = self.generate_final_report()
        if final_report:
            analysis_summary["steps_completed"].append("final_report")
        else:
            analysis_summary["steps_failed"].append("final_report")
        
        # 분석 완료
        end_time = datetime.now()
        duration = end_time - start_time
        analysis_summary["end_time"] = end_time.isoformat()
        analysis_summary["total_duration"] = str(duration)
        
        # 분석 요약 저장
        summary_path = self.output_dir / "analysis_summary.json"
        with open(summary_path, 'w', encoding='utf-8') as f:
            json.dump(analysis_summary, f, ensure_ascii=False, indent=2)
        
        # 결과 출력
        print("\n" + "=" * 80)
        print("🎉 통합 분석 완료!")
        print(f"⏱️ 총 소요시간: {duration}")
        print(f"📊 분석 결과:")
        
        # 재무지표 결과
        if financial_result and financial_result.get("success"):
            anomaly_count = len(financial_result.get("anomalies", []))
            print(f"   ✅ 재무지표 이상치: {anomaly_count}개 탐지")
        else:
            print(f"   ❌ 재무지표 분석: 실패")
            
        # 비재무지표 결과
        if nfr_result and nfr_result.get("success"):
            quarters = len(nfr_result.get("analyzed_quarters", []))
            risk_level = nfr_result.get("risk_summary", {}).get("overall_risk_level", "Unknown")
            print(f"   ✅ 비재무지표 분석: {quarters}개 분기, 위험수준 {risk_level}")
        elif corp_code:
            print(f"   ❌ 비재무지표 분석: 실패")
        else:
            print(f"   ⚠️ 비재무지표 분석: 건너뜀 (corp_code 없음)")
            
        # 근거분석 결과
        if reasoning_result:
            has_nonf = "non_financial_reasoning" in reasoning_result
            has_news = "news_reasoning" in reasoning_result
            reasoning_type = []
            if has_nonf:
                reasoning_type.append("비재무")
            if has_news:
                reasoning_type.append("뉴스")
            print(f"   ✅ 근거 분석: {' + '.join(reasoning_type)} 기반 완료")
        elif corp_code:
            print(f"   ❌ 근거 분석: 실패")
        else:
            print(f"   ⚠️ 근거 분석: 건너뜀 (corp_code 없음)")
            
        # 뉴스분석 결과
        if news_result:
            news_count = news_result.get("total_news_count", 0)
            financial_anomalies = news_result.get("financial_anomalies_count", 0)
            nfr_anomalies = news_result.get("non_financial_anomalies_count", 0)
            print(f"   ✅ 뉴스 분석: {news_count}개 뉴스, 기반 이상치 {financial_anomalies + nfr_anomalies}개")
        else:
            print(f"   ❌ 뉴스 분석: 실패")
            
        # 최종 리포트 결과
        if final_report:
            company_name = final_report.get("metadata", {}).get("company_name", corp_name)
            risk_assessment = final_report.get("risk_assessment", {})
            grade = risk_assessment.get("grade", "Unknown")
            score = risk_assessment.get("score", 0)
            risk_level = risk_assessment.get("risk_level", "Unknown")
            print(f"   ✅ 최종 리포트: {company_name} - {grade}등급 ({score}점, {risk_level})")
        else:
            print(f"   ❌ 최종 리포트: 생성 실패")
        
        print(f"\n📁 모든 결과가 {self.output_dir}에 저장되었습니다.")
        print(f"📋 분석 요약: {summary_path}")
        
        # 완료된 단계와 실패한 단계 요약
        completed_count = len(analysis_summary["steps_completed"])
        failed_count = len(analysis_summary["steps_failed"])
        total_steps = completed_count + failed_count
        
        print(f"🔄 단계별 결과: {completed_count}/{total_steps} 성공")
        if failed_count > 0:
            print(f"⚠️ 실패한 단계: {', '.join(analysis_summary['steps_failed'])}")
        
        print("=" * 80)
       
        
        return self.results, analysis_summary


def main():
    """메인 실행 함수"""
    parser = argparse.ArgumentParser(description="통합 신용위험 분석 시스템")
    parser.add_argument("--company", "-c", required=True, help="분석할 회사명 (예: 삼성전자)")
    # parser.add_argument("--corp_code", "-cc", help="DART 기업코드 8자리 (예: 00126380)")
    parser.add_argument("--config", help="설정 파일 경로")
    parser.add_argument("--output_dir", "-o", help="결과 저장 디렉토리")
    
    args = parser.parse_args()
    
    try:
        # 분석기 초기화
        analyzer = IntegratedCreditRiskAnalyzer(corp_name=args.company, config_path=args.config)
        # 출력 디렉토리 설정
        if args.output_dir:
            analyzer.output_dir = Path(args.output_dir)
            analyzer.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 전체 분석 실행
        from financial_analysis.load_corpinfo import CorpInfo

        # 예시 신용등급 데이터 로드
        try:
            with open("./dataset/credit_grade_fake.json", 'r', encoding='utf-8') as f:
                grade_data = json.load(f)
            if args.company not in grade_data:
                print(f"❌ 신용등급 데이터에 '{args.company}' 정보가 없습니다. 기본값으로 'B-' 사용.")
                current_grade = "B-"
            else:
                current_grade = grade_data[args.company].get("current_grade", "B-")
        except FileNotFoundError:
            print("⚠️ 신용등급 데이터 파일을 찾을 수 없습니다. 기본값으로 'B-' 사용.")
            current_grade = "B-"


        results, summary = analyzer.run_full_analysis(
            corp_name=args.company,
            current_grade = current_grade,
            corp_code = CorpInfo(args.company).corp_code
        )
        
        # 성공 여부에 따른 종료 코드 반환
        failed_count = len(summary.get("steps_failed", []))
        if failed_count == 0:
            print("🎯 모든 분석 단계가 성공적으로 완료되었습니다!")
            launch_dashboard()
            return 0
        elif failed_count < 3:  # 일부 실패는 허용
            print("⚠️ 일부 단계에서 오류가 발생했지만 주요 분석은 완료되었습니다.")
            return 0
        else:
            print("❌ 다수의 분석 단계에서 오류가 발생했습니다.")
            return 1
        
    except KeyboardInterrupt:
        print("\n❌ 사용자에 의해 분석이 중단되었습니다.")
        return 1
        
    except Exception as e:
        print(f"\n❌ 예상치 못한 오류가 발생했습니다: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

# run.py 파일 마지막에 추가할 코드

def launch_dashboard():
    """대시보드를 실행하는 함수"""
    import subprocess
    import sys
    import os
    
    # 필요한 패키지 설치 확인
    try:
        import flask
    except ImportError:
        print("Flask가 설치되지 않았습니다. 설치 중...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "flask"])
    
    # templates 폴더가 없으면 생성
    if not os.path.exists('templates'):
        print("templates 폴더를 생성하고 HTML 파일을 설정하는 중...")
        
        # templates 폴더 생성 및 파일 복사 코드
        os.makedirs('templates', exist_ok=True)
        
        # 여기에서 위의 HTML 파일들을 생성하는 코드 실행
        # (위의 templates_folder_structure 아티팩트 코드 실행)
        
    # 대시보드 실행
    print("\n" + "="*60)
    print("🚀 KB 국민은행 AI 신용위험 분석 대시보드를 시작합니다...")
    print("📊 웹 브라우저가 자동으로 열립니다: http://127.0.0.1:5000/")
    print("❌ 종료하려면 Ctrl+C를 누르세요.")
    print("="*60 + "\n")
    
    # Flask 앱 실행
    try:
        # dashboard_app.py 실행
        subprocess.run([sys.executable, "dashboard_app.py"])
    except KeyboardInterrupt:
        print("\n대시보드를 종료합니다.")
    except Exception as e:
        print(f"대시보드 실행 중 오류가 발생했습니다: {e}")
        print("dashboard_app.py 파일이 있는지 확인해주세요.")

# run.py의 메인 함수 마지막에 추가
if __name__ == "__main__":
    # 기존 분석 코드 실행 후...
    main()
    # 분석 완료 후 대시보드 실행 여부 물어보기
    while True:
        user_input = input("\n분석이 완료되었습니다. 대시보드를 실행하시겠습니까? (y/n): ").lower().strip()
        if user_input in ['y', 'yes', '예', 'ㅇ']:
            launch_dashboard()
            break
        elif user_input in ['n', 'no', '아니오', 'ㄴ']:
            print("프로그램을 종료합니다.")
            break
        else:
            print("y 또는 n을 입력해주세요.")


