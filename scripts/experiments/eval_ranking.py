import torch
import yaml
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import argparse

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model
from src.dynamics.gru_baseline import GRUBaseline

def compute_metrics(scores, k=10):
    _, indices = torch.topk(scores, k, dim=-1)
    hits = 0
    ndcg = 0
    for row in indices:
        if 0 in row:
            hits += 1
            rank = (row == 0).nonzero(as_tuple=True)[0].item()
            ndcg += 1 / np.log2(rank + 2)
    return hits, ndcg

def evaluate(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    with open('configs/config.yaml', 'r') as f: config = yaml.safe_load(f)
    # 
    dataset = MovieLensDataset(mode='test')
    
    # 
    if args.solver == 'baseline':
        print(" Loading GRU Baseline...")
        model = GRUBaseline(n_items=dataset.n_items, hidden_dim=config['model']['hidden_dim']).to(device)
    else:
        print(f" Loading DeepM3 ODE ({args.solver})...")
        model = DeepM3Model(config, n_items=dataset.n_items, solver=args.solver).to(device)
        
    # 
    print(f" Loading weights from {args.model_path}")
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()
    
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    total_hits = 0
    total_ndcg = 0
    N = len(dataset)
    
    with torch.no_grad():
        for batch in tqdm(test_loader):
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            dt = batch['dt'].to(device) if 'dt' in batch else None
            pos = batch['pos'].to(device)
            
            # Forward
            if args.solver == 'baseline':
                user_emb = model(x)
            else:
                user_emb = model(x, t, dt=dt)
            
            # Ranking
            neg_candidates = torch.randint(1, dataset.n_items, (100,)).to(device)
            candidates = torch.cat([pos, neg_candidates])
            
            cand_embs = model.get_item_embedding(candidates)
            scores = torch.matmul(user_emb, cand_embs.t())
            
            h, n = compute_metrics(scores, k=10)
            total_hits += h
            total_ndcg += n
            
    print(f"\n Results for {args.solver}:")
    print(f"   HR@10   : {total_hits/N:.4f}")
    print(f"   NDCG@10 : {total_ndcg/N:.4f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--solver', type=str, required=True)
    args = parser.parse_args()
    evaluate(args)
