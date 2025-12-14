# /KB-CRACK/dashboard_app.py
# 분석 결과를 웹 대시보드로 제공하는 Flask 애플리케이션 파일

import json
import os
import threading
import time
import webbrowser
from datetime import datetime

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)


def get_companies_data():
    """
    analysis_results 디렉토리에서 기업별 분석 결과 로드 수행
    """
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
    """기업 목록 페이지 렌더링 수행"""
    companies = get_companies_data()
    return render_template("index.html", companies=companies)


@app.route("/company/<company_name>")
def company_dashboard(company_name):
    """
    특정 기업 대시보드 페이지 렌더링 수행
    """
    companies = get_companies_data()
    if company_name not in companies:
        return "기업을 찾을 수 없습니다", 404

    company_data = companies[company_name]
    return render_template(
        "dashboard.html",
        company_name=company_name,
        company_data=company_data,
    )


@app.route("/api/company/<company_name>")
def api_company_data(company_name):
    """
    기업별 분석 데이터 JSON API 제공
    """
    companies = get_companies_data()
    if company_name not in companies:
        return jsonify({"error": "기업을 찾을 수 없습니다"}), 404

    return jsonify(companies[company_name])


def open_browser():
    """로컬 서버 주소를 기본 브라우저로 자동 오픈"""
    time.sleep(1.5)
    webbrowser.open("http://127.0.0.1:5000/")


if __name__ == "__main__":
    threading.Thread(target=open_browser).start()
    app.run(debug=True, use_reloader=False, host="0.0.0.0", port=5000)
