import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time

# Path Hack
sys.path.append(os.getcwd())

from src.data.dataset import UserBehaviorDataset 
from src.dynamics.modeling import DeepM3Model
from src.utils.seeder import set_seed
from src.utils.env_check import print_env_fingerprint

def train(args):
    # 1. 环境准备
    print_env_fingerprint()
    set_seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on Device: {device}")
    print(f"⚙️  Solver Strategy: {args.solver.upper()}") # 打印当前策略

    # 2. 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 数据准备
    try:
        train_dataset = UserBehaviorDataset(mode='train', config=config)
        # 为了 Ablation 跑得快一点，如果是演示，可以减少 epoch 或数据量
        n_items = train_dataset.n_items
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        return

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)

    # 4. 模型初始化
    # [Mod] 这里的关键：把 args.solver 传给模型
    model = DeepM3Model(config, n_items=n_items, solver=args.solver).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=float(config['train']['learning_rate']))
    criterion = nn.CrossEntropyLoss()

    # 5. 训练循环
    print(f"🔥 Start Training for {args.epochs} Epochs...")
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        start_time = time.time()
        
        for batch in train_loader:
            optimizer.zero_grad()
            x = batch['x'].to(device)
            t = batch['t'].to(device)
            y = batch['y'].to(device)
            
            # Forward (内部会根据 solver='none'/'rk4' 走不同路径)
            logits = model(x, t)
            
            loss = criterion(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # 6. 保存逻辑
        # [Mod] 使用 args.save_name 动态决定保存文件名
        if avg_loss < best_loss:
            best_loss = avg_loss
            os.makedirs("checkpoints", exist_ok=True)
            save_path = f"checkpoints/{args.save_name}"
            torch.save(model.state_dict(), save_path)
            
    print(f"✅ Training Complete. Best Loss: {best_loss:.4f}")
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--seed', type=int, default=42)
    
    # [Mod] 新增消融实验参数
    parser.add_argument('--solver', type=str, default='rk4', choices=['none', 'euler', 'rk4'], help="ODE solver method")
    parser.add_argument('--save_name', type=str, default='model_ode_rk4.pth', help="Checkpoint filename")
    
    args = parser.parse_args()
    train(args)