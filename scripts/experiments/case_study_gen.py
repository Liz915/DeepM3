import torch
import yaml
import numpy as np
import pickle
import sys
import os

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model

def generate_case_study():
    device = torch.device("mps")
    with open('configs/config.yaml', 'r') as f: config = yaml.safe_load(f)
    
    # 1.  Metadata (ID -> Title)
    print(" Loading Metadata...")
    with open('data/ml1m_meta.pkl', 'rb') as f:
        meta = pickle.load(f) # {id: "Title | Genre"}
    
    # 2. 
    dataset = MovieLensDataset(mode='test')

    # 3.  (RK4)
    model = DeepM3Model(config, n_items=dataset.n_items, solver='rk4').to(device)
    model.load_state_dict(torch.load("checkpoints/model_ode_rk4.pth", map_location=device))
    model.eval()
    
    print(" Searching for High-Entropy Case (Ambiguous User)...")
    
    best_case = None
    max_entropy = -1.0
    
    #  500 
    indices = np.random.choice(len(dataset), 500, replace=False)
    
    with torch.no_grad():
        for idx in indices:
            sample = dataset[idx]
            x = sample['x'].unsqueeze(0).to(device)
            t = sample['t'].unsqueeze(0).to(device)
            dt = sample['dt'].unsqueeze(0).to(device) if 'dt' in sample else None
            
            #  User Embedding
            user_emb = model(x, t, dt=dt) # [1, D]
            
            # 
            all_items = torch.arange(1, dataset.n_items).to(device)
            item_embs = model.get_item_embedding(all_items)
            logits = torch.matmul(user_emb, item_embs.t()) # [1, 3999]
            
            #  (Entropy)
            probs = torch.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            
            if entropy > max_entropy:
                max_entropy = entropy
                #  Top 3 
                _, topk = torch.topk(logits, 3)
                
                best_case = {
                    'user_id': sample['x'].tolist(), # 
                    'target': sample['pos'].item(),  # 
                    'preds': topk[0].tolist(),       # 
                    'entropy': entropy
                }

    # 4.  Case Report ()
    print("\n" + "="*50)
    print(f" CASE STUDY REPORT (Entropy: {best_case['entropy']:.4f})")
    print("="*50)
    
    print("\n [User History] (Last 5 items):")
    history_ids = best_case['user_id'][-5:]
    genres_count = {}
    for mid in history_ids:
        info = meta.get(mid, "Unknown")
        print(f"  - {info}")
        #  Genre
        g = info.split('|')[-1].strip()
        genres_count[g] = genres_count.get(g, 0) + 1
        
    print(f"\n [Ground Truth Next Item]:")
    print(f"  -> {meta.get(best_case['target'], 'Unknown')}")
    
    print(f"\n [System 1 Prediction] (Top 3):")
    for pid in best_case['preds']:
        print(f"  ? {meta.get(pid, 'Unknown')}")
        
    print("\n [System 2 Reasoning Trace] (Simulated):")
    dominant_genre = max(genres_count, key=genres_count.get)
    print(f"\"User history shows strong interest in {dominant_genre}. ")
    print(f"System 1 is uncertain (High Entropy). ")
    print(f"Reasoning: The user recently watched a sequence of {dominant_genre} movies with specific actors.")
    print(f"Therefore, reranking candidates to prioritize {dominant_genre} over generic popular items.\"")
    print("="*50)

if __name__ == "__main__":
    generate_case_study()
