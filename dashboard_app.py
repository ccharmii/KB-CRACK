import os
import json
from flask import Flask, render_template, request, jsonify
import webbrowser
import threading
import time
from datetime import datetime

app = Flask(__name__)

def get_companies_data():
    """analysis_results 폴더에서 기업별 데이터를 읽어오는 함수"""
    companies = {}
    analysis_results_path = "analysis_results"
    
    if not os.path.exists(analysis_results_path):
        return companies
    
    for company_folder in os.listdir(analysis_results_path):
        company_path = os.path.join(analysis_results_path, company_folder)
        if os.path.isdir(company_path):
            company_data = {}
            
            # final_comprehensive_report.json 읽기
            comprehensive_report_path = os.path.join(company_path, "final_comprehensive_report.json")
            if os.path.exists(comprehensive_report_path):
                with open(comprehensive_report_path, 'r', encoding='utf-8') as f:
                    company_data['comprehensive'] = json.load(f)
            
            # daily_news_risk_analysis.json 읽기
            daily_news_path = os.path.join(company_path, "daily_news_risk_analysis.json")
            if os.path.exists(daily_news_path):
                with open(daily_news_path, 'r', encoding='utf-8') as f:
                    company_data['daily_news'] = json.load(f)
            
            if company_data:
                companies[company_folder] = company_data
    
    return companies

@app.route('/')
def index():
    """기업 목록 페이지"""
    companies = get_companies_data()
    return render_template('index.html', companies=companies)

@app.route('/company/<company_name>')
def company_dashboard(company_name):
    """특정 기업의 대시보드 페이지"""
    companies = get_companies_data()
    if company_name not in companies:
        return "기업을 찾을 수 없습니다.", 404
    
    company_data = companies[company_name]
    return render_template('dashboard.html', 
                         company_name=company_name, 
                         company_data=company_data)

@app.route('/api/company/<company_name>')
def api_company_data(company_name):
    """기업 데이터 API"""
    companies = get_companies_data()
    if company_name not in companies:
        return jsonify({"error": "기업을 찾을 수 없습니다."}), 404
    
    return jsonify(companies[company_name])

def open_browser():
    """브라우저를 자동으로 여는 함수"""
    time.sleep(1.5)  # 서버가 시작될 때까지 잠시 대기
    webbrowser.open('http://127.0.0.1:5000/')

if __name__ == '__main__':
    # 브라우저 자동 열기
    threading.Thread(target=open_browser).start()
    
    # Flask 앱 실행
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
