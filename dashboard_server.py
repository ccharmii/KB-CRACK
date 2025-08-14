
import os
import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from urllib.parse import urlparse, parse_qs
import time

class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=os.getcwd(), **kwargs)
    
    def do_GET(self):
        parsed_path = urlparse(self.path)
        
        if parsed_path.path == '/api/companies':
            self.send_companies_data()
        elif parsed_path.path.startswith('/api/company/'):
            company_name = parsed_path.path.split('/')[-1]
            self.send_company_detail(company_name)
        else:
            super().do_GET()
    
    def send_companies_data(self):
        try:
            companies = []
            analysis_dir = Path('analysis_results')
            
            if analysis_dir.exists():
                for company_dir in analysis_dir.iterdir():
                    if company_dir.is_dir():
                        company_data = self.load_company_summary(company_dir)
                        if company_data:
                            companies.append(company_data)
            
            self.send_json_response(companies)
        except Exception as e:
            self.send_error_response(str(e))
    
    def send_company_detail(self, company_name):
        try:
            from urllib.parse import unquote
            company_name = unquote(company_name)
            
            analysis_dir = Path('analysis_results') / company_name
            detail_data = self.load_company_detail(analysis_dir)
            
            self.send_json_response(detail_data)
        except Exception as e:
            self.send_error_response(str(e))
    
    def load_company_summary(self, company_dir):
        try:
            comprehensive_file = company_dir / 'final_comprehensive_report.json'
            daily_file = company_dir / 'daily_news_risk_analysis.json'
            
            if not comprehensive_file.exists():
                return None
            
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                comprehensive_data = json.load(f)
            
            daily_data = {}
            if daily_file.exists():
                with open(daily_file, 'r', encoding='utf-8') as f:
                    daily_data = json.load(f)
            
            return {
                'name': comprehensive_data.get('기업_정보', {}).get('기업명', company_dir.name),
                'currentGrade': comprehensive_data.get('기업_정보', {}).get('Current_credit_grade', 'N/A'),
                'predictedGrade': comprehensive_data.get('예상_신용등급_변화', {}).get('예상_등급', 'N/A'),
                'riskLevel': daily_data.get('company_analysis_summary', {}).get('overall_risk_level', '보통'),
                'anomalies': len(comprehensive_data.get('재무지표_분석', {}).get('이상치_목록', {})),
                'lastAnalysis': comprehensive_data.get('분석_메타데이터', {}).get('생성_시간', '')[:10]
            }
        except Exception as e:
            print(f"Error loading company summary for {company_dir.name}: {e}")
            return None
    
    def load_company_detail(self, company_dir):
        try:
            comprehensive_file = company_dir / 'final_comprehensive_report.json'
            daily_file = company_dir / 'daily_news_risk_analysis.json'
            
            with open(comprehensive_file, 'r', encoding='utf-8') as f:
                comprehensive_data = json.load(f)
            
            daily_data = {}
            if daily_file.exists():
                with open(daily_file, 'r', encoding='utf-8') as f:
                    daily_data = json.load(f)
            
            # 재무지표 이상치 변환
            financial_anomalies = []
            for metric, data in comprehensive_data.get('재무지표_분석', {}).get('이상치_목록', {}).items():
                financial_anomalies.append({
                    'metric': metric,
                    'description': data.get('description', ''),
                    'severity': data.get('severity', 'Medium')
                })
            
            # 비재무지표 위험 변환
            non_financial_risks = []
            for risk in comprehensive_data.get('비재무지표_분석', {}).get('탐지된_이상치', []):
                non_financial_risks.append({
                    'indicator': risk.get('indicator_name', ''),
                    'rationale': risk.get('rationale', ''),
                    'grade': risk.get('grade_label', ''),
                    'level': 'high' if risk.get('score', 3) <= 2 else 'medium' if risk.get('score', 3) <= 3 else 'low'
                })
            
            # 뉴스 분석 변환
            news_analysis = []
            for news in daily_data.get('relevant_news', [])[:5]:  # 최대 5개
                news_analysis.append({
                    'title': news.get('title', ''),
                    'summary': news.get('content', '')[:200] + '...',
                    'url': news.get('url', '#')
                })
            
            # 유사 사례 변환
            historical_cases = []
            for case in comprehensive_data.get('유사_사례_분석', []):
                historical_cases.append({
                    'company': case.get('company', ''),
                    'year': case.get('year', ''),
                    'anomalyType': case.get('anomaly_type', ''),
                    'initialGrade': case.get('initial_grade', ''),
                    'finalGrade': case.get('final_grade', ''),
                    'recoveryPeriod': case.get('recovery_period', ''),
                    'actions': case.get('actions_taken', [])
                })
            
            return {
                'aiSummary': comprehensive_data.get('AI_종합_분석_요약', ''),
                'currentGrade': comprehensive_data.get('기업_정보', {}).get('Current_credit_grade', 'N/A'),
                'predictedGrade': comprehensive_data.get('예상_신용등급_변화', {}).get('예상_등급', 'N/A'),
                'gradeReason': comprehensive_data.get('예상_신용등급_변화', {}).get('등급_변화_사유', ''),
                'financialAnomalies': financial_anomalies,
                'nonFinancialRisks': non_financial_risks,
                'newsAnalysis': news_analysis,
                'historicalCases': historical_cases
            }
        except Exception as e:
            print(f"Error loading company detail: {e}")
            return {'error': str(e)}
    
    def send_json_response(self, data):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps(data, ensure_ascii=False).encode('utf-8'))
    
    def send_error_response(self, error_msg):
        self.send_response(500)
        self.send_header('Content-type', 'application/json')
        self.end_headers()
        self.wfile.write(json.dumps({'error': error_msg}, ensure_ascii=False).encode('utf-8'))

def start_dashboard_server():
    port = 8000
    server = HTTPServer(('localhost', port), DashboardHandler)
    print(f"대시보드 서버가 http://localhost:{port} 에서 실행 중입니다...")
    
    # 브라우저 열기
    def open_browser():
        time.sleep(1)  # 서버가 시작될 때까지 잠시 대기
        webbrowser.open(f'http://localhost:{port}')
    
    threading.Thread(target=open_browser, daemon=True).start()
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n대시보드 서버를 종료합니다.")
        server.shutdown()
