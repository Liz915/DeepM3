import sys
import os
import torch
import numpy as np
import yaml
import argparse
import time
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from src.data.dataset import RealDataDataset
from src.dynamics.modeling import DeepM3Model

def compute_metrics(logits, targets, k=10):
    _, indices = torch.topk(logits, k, dim=-1)
    targets = targets.view(-1, 1)
    hits = (indices == targets).float()
    hr = hits.sum(dim=1).mean().item()
    
    positions = torch.arange(k).to(logits.device).expand_as(hits)
    discounts = 1.0 / torch.log2(positions + 2.0)
    ndcg = (hits * discounts).sum(dim=1).mean().item()
    return hr, ndcg

def evaluate(args):
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
        
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    
    # Data
    val_ds = RealDataDataset("data/processed.pt", is_train=False)
    val_loader = DataLoader(val_ds, batch_size=64, shuffle=False, num_workers=0)
    
    # Model
    model = DeepM3Model(cfg, n_items=val_ds.n_items).to(device)
    
    # Load specific checkpoint
    if not os.path.exists(args.model_path):
        print(f"❌ Error: Model not found at {args.model_path}")
        return

    # weights_only=False 解决 numpy 加载问题
    state_dict = torch.load(args.model_path, map_location=device, weights_only=False)
    model.load_state_dict(state_dict)
    model.eval()
    
    total_hr, total_ndcg = 0.0, 0.0
    latencies = []
    
    # Warmup
    if args.verbose:
        print(f"📊 Evaluating {args.model_path} on {device}...")
        iterator = tqdm(val_loader)
    else:
        iterator = val_loader

    with torch.no_grad():
        for batch in iterator:
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            start = time.time()
            logits = model(x, t)
            # Sync for accurate timing on GPU
            if torch.cuda.is_available(): torch.cuda.synchronize()
            if torch.backends.mps.is_available(): torch.mps.synchronize()
            end = time.time()
            
            latencies.append((end - start) * 1000)
            
            hr, ndcg = compute_metrics(logits, y, k=10)
            total_hr += hr
            total_ndcg += ndcg
            
    avg_hr = total_hr / len(val_loader)
    avg_ndcg = total_ndcg / len(val_loader)
    avg_lat = np.mean(latencies)
    
    # 输出 CSV 格式结果 (方便 Shell 脚本抓取)
    # Format: ModelPath, HR@10, NDCG@10, Latency
    print(f"{args.model_path},{avg_hr:.4f},{avg_ndcg:.4f},{avg_lat:.2f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    parser.add_argument("--model_path", required=True, help="Path to .pth checkpoint")
    parser.add_argument("--verbose", action="store_true", help="Show progress bar")
    
    args = parser.parse_args()
    evaluate(args)