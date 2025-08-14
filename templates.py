import os

def create_templates_folder():
    """templates 폴더와 필요한 HTML 파일들을 생성하는 함수"""
    
    # templates 폴더 생성
    os.makedirs('templates', exist_ok=True)
    
    # index.html 파일 생성
    index_html = '''<!DOCTYPE html>
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
            마지막 업데이트: {{ moment().format('YYYY-MM-DD HH:mm:ss') if moment else '' }}
        </div>
    </div>
</body>
</html>'''
    
    with open('templates/index.html', 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    print("templates/index.html 파일이 생성되었습니다.")
    print("templates/dashboard.html 파일은 위의 dashboard 아티팩트를 복사하여 생성해주세요.")

if __name__ == "__main__":
    create_templates_folder()