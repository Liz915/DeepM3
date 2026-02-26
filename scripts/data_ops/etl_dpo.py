import json
import os

# 
REQ_FILE = "logs/online_requests.jsonl"
FB_FILE = "logs/online_feedback.jsonl"
OUT_FILE = "assets/dpo_dataset_final.jsonl"

def etl_process():
    if not os.path.exists(REQ_FILE) or not os.path.exists(FB_FILE):
        print(f" No logs found at {REQ_FILE}. Please run docker cp first.")
        return

    print(" Processing logs...")
    
    # 1. 
    req_map = {}
    with open(REQ_FILE, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                req_map[d['impression_id']] = d
            except: pass # 
            
    # 2. 
    fb_map = {}
    with open(FB_FILE, 'r') as f:
        for line in f:
            try:
                d = json.loads(line)
                fb_map[d['impression_id']] = d['click']
            except: pass
            
    # 3. Join ()  DPO 
    dpo_pairs = []
    for imp_id, clicked in fb_map.items():
        if imp_id not in req_map: continue
        
        req = req_map[imp_id]
        
        # [Fix]  Key  'input'
        user_history = req.get('input', req.get('input_items', []))
        reasoning = req.get('reasoning', '')
        
        # 
        prompt = f"Recommend for User {req['user_id']} based on History {user_history}"
        
        if clicked:
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": reasoning,
                "rejected": "Generic recommendation." # 
            })
        else:
            dpo_pairs.append({
                "prompt": prompt,
                "chosen": "Perfect deep reasoning analysis.", # 
                "rejected": reasoning
            })
            
    # 4. 
    os.makedirs("assets", exist_ok=True)
    with open(OUT_FILE, 'w') as f:
        for p in dpo_pairs:
            f.write(json.dumps(p, ensure_ascii=False) + "\n")
            
    print(f" Generated {len(dpo_pairs)} training samples in {OUT_FILE}")

if __name__ == "__main__":
    etl_process()