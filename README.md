# KB-CRACK: KB 스타일 AI 신용위험 분석 & 대시보드

## 개요
KB-CRACK은 기업별 **재무 이상치 탐지 + 비재무 리스크 평가 + 뉴스 기반 신용위험 징후 분석 + 최종 리포트 생성**을 통합 실행하고, 결과를 **웹 대시보드(Flask)** 로 시각화합니다.

- 통합 실행 진입점: `run.py`
- 대시보드 실행: `dashboard_app.py` (Flask), `dashboard_server.py` (정적 HTTP 서버)
- 결과 저장 경로: `analysis_results/<기업명>/`

---

## 프로젝트 구조 (요약)
```
KB-CRACK/
run.py
dashboard_app.py
dashboard_server.py
templates/
index.html
dashboard.html
financial_analysis/
non_financial_analysis/
news_analysis/
report_generator.py
requirements.txt
analysis_results/
```

---

## 실행 전 준비

### 1) 패키지 설치
```bash
pip install -r requirements.txt
```
### 2) 환경변수 설정
뉴스 분석 및 리포트 생성 과정에서 LLM(OpenAI)과 검색(Tavily)을 사용합니다.
프로젝트 루트 경로에다가 `.env`를 생성해서 `OPENAI_API_KEY`, `TAVILY_API_KEY`를 넣어주세요. 

## 빠른 실행
### 1) 통합 분석 실행 (+ 성공 시 대시보드 자동 실행)
```
python run.py -c 삼성전자
# 성공 시:
# - analysis_results/삼성전자/ 에 결과 생성
# - http://127.0.0.1:5000/ 대시보드 자동 실행
```
- 옵션:
  - ```-c, --company (필수)```: 분석할 회사명
  - ```--config``` : 설정 파일 경로(JSON)
  - ```-o, --output_dir``` : 결과 저장 디렉토리 변경
- 예시:
```python run.py -c 삼성생명 -o analysis_results/삼성생명```

### 2) 대시보드만 실행 (이미 결과가 있는 경우)
```python dashboard_app.py
# http://127.0.0.1:5000/
```

## 분석 파이프라인 설명 (run.py 기준)
1. 재무 분석 (financial_analysis)
  - `financial_analysis/main.py`: `analyze_corporation()`
  - 결과: `financial_analysis.json`, `financial_anomalies.json` (이상치 추출)
2. 비재무 분석 (non_financial_analysis)
  - `non_financial_analysis/main.py`: `run_for_corp()`
  - 최근 4개 분기 정기보고서를 수집/인덱싱 후 지표 평가
  - 결과: `non_financial_analysis.json`, `non_financial_analysis_last_quater.json
3. 근거 생성/통합 (financial_reasoning + news evidence)
  - `non_financial_analysis/explainer.py` 기반 비재무 문장 근거 생성
  - `financial_analysis/fin_news_reason.py` 기반 뉴스 근거 생성
  - `financial_analysis/anomaly_integration_code.py`로 통합 리포트 생성
  - 결과: `non_financial_reasoning.json`, `anomaly_news_analysis.json`, `integrated_anomaly_report.json`
4. 뉴스 위험 분석 (news_analysis)
  - `news_analysis/news_search.py`: `CreditRiskNewsAnalyzer`
  - 결과: `daily_news_risk_analysis.json`, `daily_risk_summary.md`
5. 최종 종합 리포트 생성 (`report_generator.py`)
  - `report_generator.py`: `generate_final_report()`
  - 결과: `final_integrated_report.json`, `executive_summary.md`, `detailed_analysis.md`, (추가 생성되는 경우) `final_comprehensive_report.json`, `structured_final_report.json` 등

## 예시 (이미 생성된 결과)
현재 레포에는 다음 기업 결과가 존재합니다.
- 삼성전자, 삼성생명, 한화시스템, 바이오솔루션, LG전자, sk하이닉스 등
(각 기업별 결과는 `analysis_results/<기업명>/`에서 확인)
