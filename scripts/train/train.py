import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model
# Baseline model
from src.dynamics.gru_baseline import GRUBaseline
from src.utils.seeder import set_seed

# Pairwise ranking loss
def bpr_loss(user_emb, pos_item_emb, neg_item_emb):
    pos_score = torch.sum(user_emb * pos_item_emb, dim=-1)
    neg_score = torch.sum(user_emb * neg_item_emb, dim=-1)
    loss = -torch.mean(torch.nn.functional.logsigmoid(pos_score - neg_score))
    return loss

def train(args):
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f" Training on {device} | Solver: {args.solver}")

    with open(args.config, 'r') as f: config = yaml.safe_load(f)
    if args.hidden_dim > 0:
        config['model']['hidden_dim'] = args.hidden_dim

    # 1) Build training dataloader (num_workers=0 for stability)
    train_dataset = MovieLensDataset(mode='train')
    train_loader = DataLoader(
        train_dataset, 
        batch_size=config['train']['batch_size'], 
        shuffle=True,
        num_workers=0 
    )
    print(f" Train Samples: {len(train_dataset)}")

    # 2) Build model
    if args.solver == 'baseline':
        print(" Using Standard GRU Baseline (High Speed)")
        model = GRUBaseline(
            n_items=train_dataset.n_items,
            hidden_dim=config['model']['hidden_dim']
        ).to(device)
    else:
        print(f" Using DeepM3 ODE Model (Solver: {args.solver})")
        model = DeepM3Model(
            config, 
            n_items=train_dataset.n_items, 
            solver=args.solver
        ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # 3) Train loop
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start = time.time()
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            dt = batch['dt'].to(device) if 'dt' in batch else None
            pos = batch['pos'].to(device)
            neg = batch['neg'].to(device)
            
            # Forward pass
            if args.solver == 'baseline':
                # Baseline ignores time features
                user_emb = model(x)
            else:
                # DeepM3 consumes both absolute time and deltas
                user_emb = model(x, t, dt=dt)
            
            pos_emb = model.get_item_embedding(pos) 
            neg_emb = model.get_item_embedding(neg)
            
            loss = bpr_loss(user_emb, pos_emb, neg_emb)
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            total_loss += loss.item()
            
            # Logging
            if i % 100 == 0:
                print(f"\rStep {i}/{len(train_loader)} Loss: {loss.item():.4f}", end="")
            
        print(f"\nEpoch {epoch+1} | BPR Loss: {total_loss/len(train_loader):.4f} | Time: {time.time()-start:.1f}s")
        
        # Save latest checkpoint each epoch
        os.makedirs("checkpoints", exist_ok=True)
        torch.save(model.state_dict(), f"checkpoints/{args.save_name}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--solver', type=str, default='rk4', choices=['baseline', 'rk4', 'euler', 'none'])
    parser.add_argument('--save_name', type=str, default='model.pth')
    parser.add_argument('--hidden_dim', type=int, default=0, help='Override hidden_dim in config (0 = use config default)')
    args = parser.parse_args()
    train(args)
