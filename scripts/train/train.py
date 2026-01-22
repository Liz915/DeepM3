import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import sys
import os
import time

# Path Hack: 确保能找到 src
sys.path.append(os.getcwd())

# 引入我们整理好的模块
# [Fix] 这里的类名要和 src/data/dataset.py 里的一致
from src.data.dataset import UserBehaviorDataset 
from src.dynamics.modeling import DeepM3Model
from src.utils.seeder import set_seed
from src.utils.env_check import print_env_fingerprint

def train(args):
    # 1. 环境准备
    print_env_fingerprint() # [Task 8] 打印环境指纹
    set_seed(args.seed)     # [Task 8] 锁定种子
    
    device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
    print(f"🚀 Training on Device: {device}")

    # 2. 加载配置
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # 3. 数据准备
    print("🔄 Loading Data...")
    try:
        train_dataset = UserBehaviorDataset(mode='train', config=config)
        test_dataset = UserBehaviorDataset(mode='test', config=config)
    except Exception as e:
        print(f"❌ Data Load Error: {e}")
        print("💡 Tip: Did you run 'python src/data/preprocessor.py' first?")
        return

    # [Fix] 获取 n_items 用于模型初始化
    n_items = train_dataset.n_items
    print(f"📊 Items: {n_items} | Train Size: {len(train_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=config['train']['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['train']['batch_size'], shuffle=False)

    # 4. 模型初始化
    model = DeepM3Model(config, n_items=n_items).to(device)
    lr = float(config['train'].get('learning_rate', 1e-3))
    weight_decay = float(config['train'].get('weight_decay', 1e-5))
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
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
            
            x = batch['x'].to(device) # [B, Seq]
            t = batch['t'].to(device) # [B, Seq]
            y = batch['y'].to(device) # [B] (Next Item Label)
            
            # Forward
            logits = model(x, t) # [B, n_items]
            
            # 这种简单的 Auto-regressive 任务通常取最后一个时间步预测下一个
            # 这里的 logits 已经是 head 输出的 [B, n_items]
            
            loss = criterion(logits, y)
            loss.backward()
            
            # 梯度裁剪 (ODE 训练稳定性关键)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        avg_loss = total_loss / len(train_loader)
        epoch_time = time.time() - start_time
        
        print(f"   Epoch {epoch+1}/{args.epochs} | Loss: {avg_loss:.4f} | Time: {epoch_time:.1f}s")
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            # [Fix] 保存为我们在 README 里承诺的文件名
            save_path = "checkpoints/model_ode_rk4.pth"
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(model.state_dict(), save_path)
            
    print(f"✅ Training Complete. Best Loss: {best_loss:.4f}")
    print(f"💾 Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='configs/config.yaml')
    parser.add_argument('--epochs', type=int, default=5) # 默认改为5
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    
    train(args)