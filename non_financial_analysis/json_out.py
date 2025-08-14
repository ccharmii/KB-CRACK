import os, json
from datetime import datetime

def _ensure_dir(p): 
    os.makedirs(p, exist_ok=True)

# 분기별 결과를 JSON 파일로 저장
def write_quarter_json(corp_root, corp_code, corp_name, quarter, indicators):
    _ensure_dir(os.path.join(corp_root, "results"))
    obj = {
        "corp_code": corp_code,
        "corp_name": corp_name,
        "quarter": quarter,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "indicators": indicators
    }
    path = os.path.join(corp_root, "results", f"{quarter}_nfr_scores.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    return path, obj

# 분기 결과를 JSONL에 한 줄로 넣기
def append_global_jsonl(result_dir, quarter_obj):
    _ensure_dir(result_dir)
    path = os.path.join(result_dir, "nfr_scores_all.jsonl")
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(quarter_obj, ensure_ascii=False) + "\n")
    return path



