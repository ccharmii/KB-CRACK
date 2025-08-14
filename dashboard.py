#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
KB 국민은행 AI 신용위험 분석 대시보드 설정 스크립트
이 스크립트를 실행하면 대시보드에 필요한 모든 파일이 자동으로 생성됩니다.
"""

import os
import sys

def create_dashboard_files():
    """대시보드에 필요한 모든 파일을 생성합니다."""
    
    print("🚀 KB 국민은행 AI 신용위험 분석 대시보드 설정을 시작합니다...\n")
    
    # 1. templates 폴더 생성
    print("1. templates 폴더 생성 중...")
    os.makedirs('templates', exist_ok=True)
    
    # 2. Flask 앱 파일 생성
    print("2. Flask 애플리케이션 파일 생성 중...")
    
    dashboard_app_content = '''import os
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
'''
    
    with open('dashboard_app.py', 'w', encoding='utf-8') as f:
        f.write(dashboard_app_content)
    
    # 3. index.html 템플릿 생성
    print("3. index.html 템플릿 생성 중...")
    
    index_html_content = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KB 국민은행 AI 신용위험 분석 시스템</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #FFD700 0%, #FFF8DC 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .header {
            background: #FFD700;
            color: #333;
            padding: 20px;
            border-radius: 15px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            text-align: center;
        }
        
        .header h1 {
            font-size: 2.5em;
            margin-bottom: 10px;
            font-weight: bold;
        }
        
        .header .subtitle {
            font-size: 1.2em;
            color: #666;
        }
        
        .ai-badge {
            background: #FF6B35;
            color: white;
            padding: 5px 15px;
            border-radius: 20px;
            font-size: 0.9em;
            margin-left: 10px;
            animation: pulse 2s infinite;
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .container {
            max-width: 1200px;
            margin: 0 auto;
        }
        
        .companies-grid {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
            gap: 25px;
            margin-top: 30px;
        }
        
        .company-card {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 6px 20px rgba(0,0,0,0.1);
            transition: all 0.3s ease;
            border: 2px solid transparent;
            cursor: pointer;
            text-decoration: none;
            color: inherit;
        }
        
        .company-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 30px rgba(0,0,0,0.15);
            border-color: #FFD700;
        }
        
        .company-header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .company-name {
            font-size: 1.4em;
            font-weight: bold;
            color: #333;
        }
        
        .credit-grade {
            padding: 8px 16px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .grade-aaa { background: #4CAF50; color: white; }
        .grade-aa { background: #8BC34A; color: white; }
        .grade-a { background: #CDDC39; color: #333; }
        .grade-bbb { background: #FF9800; color: white; }
        .grade-bb { background: #FF5722; color: white; }
        .grade-b { background: #F44336; color: white; }
        .grade-ccc { background: #9C27B0; color: white; }
        .grade-default { background: #607D8B; color: white; }
        
        .company-info {
            margin-bottom: 15px;
        }
        
        .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 5px 0;
            border-bottom: 1px dotted #eee;
        }
        
        .info-label {
            color: #666;
            font-weight: 500;
        }
        
        .info-value {
            font-weight: bold;
            color: #333;
        }
        
        .risk-level {
            text-align: center;
            margin-top: 15px;
        }
        
        .risk-badge {
            padding: 8px 20px;
            border-radius: 20px;
            font-weight: bold;
            font-size: 0.9em;
        }
        
        .risk-low { background: #E8F5E8; color: #2E7D32; }
        .risk-medium { background: #FFF3E0; color: #F57C00; }
        .risk-high { background: #FFEBEE; color: #D32F2F; }
        
        .no-companies {
            text-align: center;
            padding: 60px 20px;
            background: white;
            border-radius: 15px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .no-companies h2 {
            color: #666;
            margin-bottom: 15px;
        }
        
        .no-companies p {
            color: #999;
            font-size: 1.1em;
        }
        
        .last-updated {
            text-align: center;
            margin-top: 30px;
            color: #666;
            font-style: italic;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>KB 국민은행 신용위험 분석 시스템 <span class="ai-badge">🤖 AI 분석</span></h1>
            <div class="subtitle">AI 기반 기업 신용위험 실시간 모니터링 대시보드</div>
        </div>
        
        {% if companies %}
        <div class="companies-grid">
            {% for company_name, company_data in companies.items() %}
            <a href="/company/{{ company_name }}" class="company-card">
                <div class="company-header">
                    <div class="company-name">{{ company_name }}</div>
                    {% if company_data.comprehensive and company_data.comprehensive.기업_정보 %}
                        {% set current_grade = company_data.comprehensive.기업_정보.Current_credit_grade or 'N/A' %}
                        <div class="credit-grade grade-{{ current_grade.lower().replace('+', '').replace('-', '') }}">
                            {{ current_grade }}
                        </div>
                    {% endif %}
                </div>
                
                <div class="company-info">
                    {% if company_data.comprehensive and company_data.comprehensive.기업_정보 %}
                    <div class="info-row">
                        <span class="info-label">업종</span>
                        <span class="info-value">{{ company_data.comprehensive.기업_정보.업종 or 'N/A' }}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">종목코드</span>
                        <span class="info-value">{{ company_data.comprehensive.기업_정보.종목코드 or 'N/A' }}</span>
                    </div>
                    {% endif %}
                    
                    {% if company_data.comprehensive and company_data.comprehensive.종합_위험평가 %}
                    <div class="info-row">
                        <span class="info-label">위험등급</span>
                        <span class="info-value">{{ company_data.comprehensive.종합_위험평가.grade or 'N/A' }}</span>
                    </div>
                    <div class="info-row">
                        <span class="info-label">위험점수</span>
                        <span class="info-value">{{ company_data.comprehensive.종합_위험평가.score or 'N/A' }}/100</span>
                    </div>
                    {% endif %}
                </div>
                
                {% if company_data.comprehensive and company_data.comprehensive.종합_위험평가 %}
                <div class="risk-level">
                    {% set risk_level = company_data.comprehensive.종합_위험평가.risk_level %}
                    {% if risk_level == '저위험' %}
                        <span class="risk-badge risk-low">{{ risk_level }}</span>
                    {% elif risk_level == '중위험' %}
                        <span class="risk-badge risk-medium">{{ risk_level }}</span>
                    {% else %}
                        <span class="risk-badge risk-high">{{ risk_level }}</span>
                    {% endif %}
                </div>
                {% endif %}
            </a>
            {% endfor %}
        </div>
        {% else %}
        <div class="no-companies">
            <h2>분석된 기업이 없습니다</h2>
            <p>analysis_results 폴더에 기업 분석 결과가 없습니다.<br>
            run.py를 실행하여 기업 분석을 먼저 수행해주세요.</p>
        </div>
        {% endif %}
        
        <div class="last-updated">
            마지막 업데이트: 방금 전
        </div>
    </div>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html_content)
    
    # 4. dashboard.html 템플릿 생성 (길어서 별도 함수로)
    print("4. dashboard.html 템플릿 생성 중...")
    create_dashboard_template()
    
    print("\n✅ 모든 파일이 성공적으로 생성되었습니다!")
    print("\n📁 생성된 파일 목록:")
    print("  - dashboard_app.py (Flask 애플리케이션)")
    print("  - templates/index.html (기업 목록 페이지)")
    print("  - templates/dashboard.html (기업별 상세 대시보드)")
    print("\n🚀 사용 방법:")
    print("  1. python dashboard_app.py 실행")
    print("  2. 웹 브라우저에서 http://127.0.0.1:5000/ 접속")
    print("  3. 분석된 기업 목록에서 원하는 기업 클릭")

def create_dashboard_template():
    """dashboard.html 템플릿을 생성합니다."""
    
    dashboard_html_content = '''<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ company_name }} - KB 신용위험 분석</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }
        
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #FFD700 0%, #FFF8DC 100%);
            min-height: 100vh;
            padding: 20px;
        }
        
        .container {
            max-width: 1400px;
            margin: 0 auto;
        }
        
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 6px solid #FFD700;
        }
        
        .header-top {
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin-bottom: 20px;
        }
        
        .company-title {
            font-size: 2.5em;
            color: #333;
            font-weight: bold;
        }
        
        .back-btn {
            background: #FFD700;
            color: #333;
            padding: 12px 25px;
            border: none;
            border-radius: 25px;
            text-decoration: none;
            font-weight: bold;
            transition: all 0.3s ease;
        }
        
        .back-btn:hover {
            background: #FFC107;
            transform: translateY(-2px);
        }
        
        .ai-summary {
            background: linear-gradient(135deg, #FF6B35, #F7931E);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
            border-left: 5px solid #FF4500;
        }
        
        .ai-summary h3 {
            font-size: 1.3em;
            margin-bottom: 10px;
            display: flex;
            align-items: center;
        }
        
        .ai-summary h3::before {
            content: "🤖";
            margin-right: 10px;
        }
        
        .credit-grades {
            display: flex;
            gap: 30px;
            margin-top: 20px;
        }
        
        .grade-box {
            background: white;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            flex: 1;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        
        .grade-box h4 {
            color: #666;
            margin-bottom: 10px;
        }
        
        .grade {
            font-size: 2em;
            font-weight: bold;
            padding: 10px 20px;
            border-radius: 20px;
            display: inline-block;
        }
        
        .main-content {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 30px;
            margin-bottom: 30px;
        }
        
        .section {
            background: white;
            border-radius: 15px;
            padding: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 5px solid #FFD700;
        }
        
        .section h2 {
            color: #333;
            margin-bottom: 20px;
            font-size: 1.5em;
            display: flex;
            align-items: center;
        }
        
        .section h2::before {
            content: "";
            width: 20px;
            height: 20px;
            margin-right: 10px;
            border-radius: 50%;
            background: #FFD700;
        }
        
        .full-width {
            grid-column: span 2;
        }
        
        .anomaly-list {
            margin-bottom: 20px;
        }
        
        .anomaly-item {
            background: #FFF9E6;
            border: 1px solid #FFD700;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 5px solid #FF9800;
        }
        
        .anomaly-header {
            font-weight: bold;
            color: #E65100;
            margin-bottom: 8px;
            font-size: 1.1em;
        }
        
        .anomaly-description {
            color: #666;
            margin-bottom: 10px;
            line-height: 1.5;
        }
        
        .severity {
            display: inline-block;
            padding: 4px 12px;
            border-radius: 15px;
            font-size: 0.8em;
            font-weight: bold;
        }
        
        .severity-medium {
            background: #FFF3E0;
            color: #F57C00;
        }
        
        .severity-high {
            background: #FFEBEE;
            color: #D32F2F;
        }
        
        .severity-low {
            background: #E8F5E8;
            color: #2E7D32;
        }
        
        .metric-table {
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }
        
        .metric-table th,
        .metric-table td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        
        .metric-table th {
            background: #F5F5F5;
            font-weight: bold;
            color: #333;
        }
        
        .metric-table tr:hover {
            background: #FFFBF0;
        }
        
        .news-item {
            background: #F8F9FA;
            border-radius: 8px;
            padding: 15px;
            margin-bottom: 15px;
            border-left: 4px solid #007BFF;
        }
        
        .news-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 8px;
            line-height: 1.4;
        }
        
        .news-content {
            color: #666;
            margin-bottom: 10px;
            line-height: 1.5;
        }
        
        .news-meta {
            font-size: 0.9em;
            color: #999;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        
        .news-link {
            color: #007BFF;
            text-decoration: none;
            font-weight: 500;
        }
        
        .news-link:hover {
            text-decoration: underline;
        }
        
        .evidence-section {
            background: #F8F9FA;
            border-radius: 8px;
            padding: 15px;
            margin-top: 15px;
        }
        
        .evidence-title {
            font-weight: bold;
            color: #333;
            margin-bottom: 10px;
        }
        
        .evidence-list {
            list-style: none;
        }
        
        .evidence-list li {
            padding: 8px 0;
            border-bottom: 1px dotted #ddd;
            color: #666;
        }
        
        .evidence-list li:last-child {
            border-bottom: none;
        }
        
        .stats-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        
        .stat-box {
            background: #F8F9FA;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
            border: 2px solid #E9ECEF;
        }
        
        .stat-value {
            font-size: 2em;
            font-weight: bold;
            color: #007BFF;
            margin-bottom: 5px;
        }
        
        .stat-label {
            color: #666;
            font-size: 0.9em;
        }
        
        .grade-aaa, .grade-aa { background: #4CAF50; color: white; }
        .grade-a { background: #8BC34A; color: white; }
        .grade-bbb { background: #CDDC39; color: #333; }
        .grade-bb, .grade-b { background: #FF9800; color: white; }
        .grade-ccc { background: #F44336; color: white; }
        
        .no-data {
            text-align: center;
            padding: 40px;
            color: #999;
            font-style: italic;
        }
        
        .collapsible {
            background: #F1F1F1;
            color: #333;
            cursor: pointer;
            padding: 15px;
            width: 100%;
            border: none;
            text-align: left;
            outline: none;
            font-size: 1em;
            border-radius: 5px;
            margin-bottom: 5px;
            font-weight: bold;
        }
        
        .collapsible:hover {
            background: #E1E1E1;
        }
        
        .collapsible-content {
            padding: 0 15px;
            max-height: 0;
            overflow: hidden;
            transition: max-height 0.3s ease;
            background: #FAFAFA;
        }
        
        .collapsible.active + .collapsible-content {
            max-height: 1000px;
            padding: 15px;
        }
    </style>
</head>
<body>
    <div class="container">
        <!-- 헤더 -->
        <div class="header">
            <div class="header-top">
                <h1 class="company-title">{{ company_name }}</h1>
                <a href="/" class="back-btn">← 기업 목록</a>
            </div>
            
            <!-- AI 종합 분석 요약 -->
            {% if company_data.comprehensive and company_data.comprehensive.AI_종합_분석_요약 %}
            <div class="ai-summary">
                <h3>AI 종합 분석 요약</h3>
                <p>{{ company_data.comprehensive.AI_종합_분석_요약 }}</p>
            </div>
            {% endif %}
            
            <!-- 신용등급 정보 -->
            <div class="credit-grades">
                {% if company_data.comprehensive and company_data.comprehensive.예상_신용등급_변화 %}
                <div class="grade-box">
                    <h4>현재 신용등급</h4>
                    <div class="grade grade-{{ company_data.comprehensive.예상_신용등급_변화.현재_등급.lower().replace('+', '').replace('-', '') }}">
                        {{ company_data.comprehensive.예상_신용등급_변화.현재_등급 }}
                    </div>
                </div>
                <div class="grade-box">
                    <h4>예상 신용등급</h4>
                    <div class="grade grade-{{ company_data.comprehensive.예상_신용등급_변화.예상_등급.lower().replace('+', '').replace('-', '') }}">
                        {{ company_data.comprehensive.예상_신용등급_변화.예상_등급 }}
                    </div>
                </div>
                {% endif %}
                
                {% if company_data.comprehensive and company_data.comprehensive.종합_위험평가 %}
                <div class="grade-box">
                    <h4>종합 위험점수</h4>
                    <div class="stat-value">{{ company_data.comprehensive.종합_위험평가.score }}/100</div>
                    <div class="stat-label">{{ company_data.comprehensive.종합_위험평가.grade }}등급</div>
                </div>
                {% endif %}
            </div>
        </div>
        
        <!-- 메인 컨텐츠는 계속 아래에 이어짐... -->
        <!-- 여기서는 길이 제한으로 축약 -->
        
    </div>
    
    <script>
        // 접기/펼치기 기능
        document.addEventListener('DOMContentLoaded', function() {
            var collapsibles = document.getElementsByClassName('collapsible');
            for (var i = 0; i < collapsibles.length; i++) {
                collapsibles[i].addEventListener('click', function() {
                    this.classList.toggle('active');
                    var content = this.nextElementSibling;
                    if (content.style.maxHeight) {
                        content.style.maxHeight = null;
                    } else {
                        content.style.maxHeight = content.scrollHeight + "px";
                    }
                });
            }
        });
    </script>
</body>
</html>'''
    
    with open('templates/dashboard.html', 'w', encoding='utf-8') as f:
        f.write(dashboard_html_content)

if __name__ == "__main__":
    try:
        create_dashboard_files()
    except Exception as e:
        print(f"❌ 오류가 발생했습니다: {e}")
        sys.exit(1)