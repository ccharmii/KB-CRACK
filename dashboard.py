# /KB-CRACK/dashboard.py
# 대시보드 실행에 필요한 Flask 및 템플릿 파일을 자동 생성하는 설정 스크립트


import os
import sys


def create_dashboard_files():
    """대시보드 구성을 위한 파일과 디렉터리를 생성"""
    print("KB 국민은행 AI 신용위험 분석 대시보드 설정을 시작합니다\n")

    os.makedirs("templates", exist_ok=True)

    dashboard_app_content = """import os
import json
from flask import Flask, render_template, jsonify
import webbrowser
import threading
import time

app = Flask(__name__)

def get_companies_data():
    \\"\\"\\"
    analysis_results 폴더에서 기업별 분석 데이터를 로드

    출력: 기업명 키와 분석 데이터 값을 갖는 딕셔너리
    \\"\\"\\"
    companies = {}
    analysis_results_path = "analysis_results"

    if not os.path.exists(analysis_results_path):
        return companies

    for company_folder in os.listdir(analysis_results_path):
        company_path = os.path.join(analysis_results_path, company_folder)
        if not os.path.isdir(company_path):
            continue

        company_data = {}

        comprehensive_report_path = os.path.join(company_path, "final_comprehensive_report.json")
        if os.path.exists(comprehensive_report_path):
            with open(comprehensive_report_path, "r", encoding="utf-8") as f:
                company_data["comprehensive"] = json.load(f)

        daily_news_path = os.path.join(company_path, "daily_news_risk_analysis.json")
        if os.path.exists(daily_news_path):
            with open(daily_news_path, "r", encoding="utf-8") as f:
                company_data["daily_news"] = json.load(f)

        if company_data:
            companies[company_folder] = company_data

    return companies

@app.route("/")
def index():
    \\"\\"\\"
    기업 목록 페이지 렌더링

    출력: 기업 목록 HTML 응답
    \\"\\"\\"
    companies = get_companies_data()
    return render_template("index.html", companies=companies)

@app.route("/company/<company_name>")
def company_dashboard(company_name):
    \\"\\"\\"
    특정 기업의 대시보드 페이지 렌더링

    입력: company_name(기업 폴더명)
    출력: 기업 상세 HTML 응답 또는 404 응답
    \\"\\"\\"
    companies = get_companies_data()
    if company_name not in companies:
        return "기업을 찾을 수 없습니다.", 404

    company_data = companies[company_name]
    return render_template(
        "dashboard.html",
        company_name=company_name,
        company_data=company_data,
    )

@app.route("/api/company/<company_name>")
def api_company_data(company_name):
    \\"\\"\\"
    기업 데이터 JSON API 제공

    입력: company_name(기업 폴더명)
    출력: 기업 데이터 JSON 응답 또는 404 JSON 응답
    \\"\\"\\"
    companies = get_companies_data()
    if company_name not in companies:
        return jsonify({"error": "기업을 찾을 수 없습니다."}), 404

    return jsonify(companies[company_name])

def open_browser():
    \\"\\"\\"
    서버 실행 후 브라우저 자동 오픈 수행
    \\"\\"\\"
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000/")

if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
"""
    with open("dashboard_app.py", "w", encoding="utf-8") as f:
        f.write(dashboard_app_content)

    index_html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>KB 국민은행 AI 신용위험 분석 시스템</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
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
        .header h1 { font-size: 2.5em; margin-bottom: 10px; font-weight: bold; }
        .header .subtitle { font-size: 1.2em; color: #666; }
        .container { max-width: 1200px; margin: 0 auto; }
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
        .company-name { font-size: 1.4em; font-weight: bold; color: #333; }
        .credit-grade { padding: 8px 16px; border-radius: 20px; font-weight: bold; font-size: 0.9em; }
        .grade-aaa { background: #4CAF50; color: white; }
        .grade-aa { background: #8BC34A; color: white; }
        .grade-a { background: #CDDC39; color: #333; }
        .grade-bbb { background: #FF9800; color: white; }
        .grade-bb { background: #FF5722; color: white; }
        .grade-b { background: #F44336; color: white; }
        .grade-ccc { background: #9C27B0; color: white; }
        .grade-default { background: #607D8B; color: white; }
        .company-info { margin-bottom: 15px; }
        .info-row {
            display: flex;
            justify-content: space-between;
            margin-bottom: 8px;
            padding: 5px 0;
            border-bottom: 1px dotted #eee;
        }
        .info-label { color: #666; font-weight: 500; }
        .info-value { font-weight: bold; color: #333; }
        .risk-level { text-align: center; margin-top: 15px; }
        .risk-badge { padding: 8px 20px; border-radius: 20px; font-weight: bold; font-size: 0.9em; }
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
        .no-companies h2 { color: #666; margin-bottom: 15px; }
        .no-companies p { color: #999; font-size: 1.1em; }
        .last-updated { text-align: center; margin-top: 30px; color: #666; font-style: italic; }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>KB 국민은행 신용위험 분석 시스템</h1>
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
            <p>analysis_results 폴더에 기업 분석 결과가 없습니다<br>
            run.py를 실행하여 기업 분석을 먼저 수행해주세요</p>
        </div>
        {% endif %}

        <div class="last-updated">
            마지막 업데이트: 방금 전
        </div>
    </div>
</body>
</html>
"""
    with open("templates/index.html", "w", encoding="utf-8") as f:
        f.write(index_html_content)

    create_dashboard_template()

    print("\n모든 파일이 성공적으로 생성되었습니다")
    print("\n생성된 파일 목록")
    print("  - dashboard_app.py")
    print("  - templates/index.html")
    print("  - templates/dashboard.html")
    print("\n사용 방법")
    print("  1. python dashboard_app.py 실행")
    print("  2. 웹 브라우저에서 http://127.0.0.1:5000/ 접속")
    print("  3. 분석된 기업 목록에서 원하는 기업 클릭")


def create_dashboard_template():
    """기업별 대시보드 화면 템플릿 파일을 생성"""
    dashboard_html_content = """<!DOCTYPE html>
<html lang="ko">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{{ company_name }} - KB 신용위험 분석</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            background: linear-gradient(135deg, #FFD700 0%, #FFF8DC 100%);
            min-height: 100vh;
            padding: 20px;
        }
        .container { max-width: 1400px; margin: 0 auto; }
        .header {
            background: white;
            border-radius: 15px;
            padding: 30px;
            margin-bottom: 30px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 6px solid #FFD700;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1>{{ company_name }}</h1>
            <a href="/">기업 목록</a>
        </div>
    </div>
</body>
</html>
"""
    with open("templates/dashboard.html", "w", encoding="utf-8") as f:
        f.write(dashboard_html_content)


if __name__ == "__main__":
    try:
        create_dashboard_files()
    except Exception as e:
        print(f"오류가 발생했습니다: {e}")
        sys.exit(1)
