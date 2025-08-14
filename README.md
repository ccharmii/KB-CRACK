
# KB 스타일 신용위험 대시보드

## 개요
- `run.py` 실행 후 생성되는 `analysis_results/<기업명>/` 내 JSON들을 읽어 자동으로 웹 대시보드를 띄웁니다.
- 색상은 KB 국민은행 느낌의 **노란색 + 흰색** 톤이며, **AI 자동분석** 배지를 명시했습니다.

## 파일 구성
```
kb_credit_dashboard/
  app.py
  templates/
    base.html
    multi_company_dashboard.html
    company_detail_dashboard.html
    company_detail_legacy.html
    no_data.html
  static/
    css/kb.css
    js/main.js
```

## 실행
분석이 완료된 상태라면:
```bash
python app.py
# → http://localhost:5000
```

또는 `run.py`에서 자동 실행(아래 패치 적용 참고).
