import torch
import yaml
import numpy as np
import pandas as pd
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model
from src.dynamics.gru_baseline import GRUBaseline

def compute_ndcg_per_user(scores, k=10):
    _, indices = torch.topk(scores, k, dim=-1)
    if 0 in indices[0]:
        rank = (indices[0] == 0).nonzero(as_tuple=True)[0].item()
        return 1 / np.log2(rank + 2)
    return 0.0

def run_irregular_sig():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open('configs/config.yaml', 'r') as f: config = yaml.safe_load(f)
    
    print(" Loading Models (Baseline vs Euler)...")
    
    dataset = MovieLensDataset(mode='test')

    # Baseline
    base_model = GRUBaseline(n_items=dataset.n_items, hidden_dim=config['model']['hidden_dim']).to(device)
    base_model.load_state_dict(torch.load("checkpoints/model_gru_baseline.pth", map_location=device))
    base_model.eval()
    
    # DeepM3 (Euler)
    ode_model = DeepM3Model(config, n_items=dataset.n_items, solver='euler').to(device)
    ode_model.load_state_dict(torch.load("checkpoints/model_ode_euler.pth", map_location=device))
    ode_model.eval()
    
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    
    results = []
    
    print(" Collecting metrics & calculating CV...")
    with torch.no_grad():
        for batch in tqdm(loader):
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            dt = batch['dt'].to(device) if 'dt' in batch else None
            pos = batch['pos'].to(device)
            
            # Irregularity should be measured on inter-event gaps, not cumulative time.
            if dt is None:
                dt_local = torch.zeros_like(t)
                dt_local[:, 1:] = t[:, 1:] - t[:, :-1]
            else:
                dt_local = dt
            dt_valid = dt_local[:, 1:] if dt_local.size(1) > 1 else dt_local
            t_std = torch.std(dt_valid).item()
            t_mean = torch.mean(dt_valid).item()
            cv = t_std / (t_mean + 1e-6)
            
            #  NDCG
            neg = torch.randint(1, dataset.n_items, (100,)).to(device)
            cand = torch.cat([pos, neg])
            
            u_base = base_model(x)
            s_base = torch.matmul(u_base, base_model.get_item_embedding(cand).t())
            n_base = compute_ndcg_per_user(s_base)
            
            u_ode = ode_model(x, t, dt=dt)
            s_ode = torch.matmul(u_ode, ode_model.get_item_embedding(cand).t())
            n_ode = compute_ndcg_per_user(s_ode)
            
            results.append({
                'cv': cv,
                'ndcg_base': n_base,
                'ndcg_ode': n_ode
            })
            
    df = pd.DataFrame(results)
    
    #  CV  33% (Irregular Users)
    threshold = df['cv'].quantile(0.67)
    df_irr = df[df['cv'] > threshold]
    
    print("\n" + "="*50)
    print(f" Significance Test on Irregular Users (Top 33% CV > {threshold:.2f})")
    print(f"Sample Size: {len(df_irr)}")
    print("="*50)
    
    t_stat, p_val = stats.ttest_rel(df_irr['ndcg_base'], df_irr['ndcg_ode'])
    
    print(f"Mean NDCG (Base) : {df_irr['ndcg_base'].mean():.4f}")
    print(f"Mean NDCG (Euler): {df_irr['ndcg_ode'].mean():.4f}")
    print(f"Improvement      : +{((df_irr['ndcg_ode'].mean() - df_irr['ndcg_base'].mean()) / df_irr['ndcg_base'].mean()) * 100:.2f}%")
    print(f"P-value          : {p_val:.4e}")
    
    if p_val < 0.05:
        print(" RESULT: Significant on Irregular Users!")
    else:
        print(" RESULT: Not significant.")

if __name__ == "__main__":
    run_irregular_sig()
