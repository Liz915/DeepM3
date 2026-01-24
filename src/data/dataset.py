import torch
from torch.utils.data import Dataset
import numpy as np
import os

class UserBehaviorDataset(Dataset):
    def __init__(self, mode='train', config=None):
        self.mode = mode
        
        # 1. 路径处理
        # 假设 processed.pt 在 data/ 目录下
        data_path = "data/processed.pt"
        if config and 'data' in config and 'data_path' in config['data']:
            data_path = config['data']['data_path']
            
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}. Please run training first or check path.")

        # 2. 加载数据
        # weights_only=False 是为了兼容旧版 PyTorch 保存的字典格式
        print(f"📦 Loading dataset from {data_path}...")
        payload = torch.load(data_path, map_location='cpu', weights_only=False)
        
        # 兼容两种保存格式：可能是 dict，也可能直接是 list
        if isinstance(payload, dict):
            self.sequences = payload.get('sequences', [])
            self.n_items = payload.get('n_items', 3707)
        else:
            self.sequences = payload
            self.n_items = 3707 # Fallback
            
        # 3. 划分 Train/Test
        total_len = len(self.sequences)
        train_size = int(0.8 * total_len)
        
        if mode == 'train':
            self.data = self.sequences[:train_size]
        else:
            self.data = self.sequences[train_size:]
            
        print(f"✅ Loaded {len(self.data)} sequences for {mode}.")

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # data[idx] 结构: (items_list, times_list, label_item)
        # 或者 (items, times) 取决于预处理逻辑。
        # 这里假设是标准的 seq 格式
        
        seq_data = self.data[idx]
        items = seq_data[0]
        times = seq_data[1]
        
        # 截断与转换
        # Input: 0 ~ T-1
        # Target: 1 ~ T (Next Item Prediction) -> 这里为了简化 ODE 训练，通常用 Auto-regressive 方式
        
        # 转换为 Tensor
        x = torch.tensor(items[:-1], dtype=torch.long)
        y = torch.tensor(items[1:], dtype=torch.long) # 简单的 Next Item 监督
        
        # 时间戳处理
        t_raw = np.array(times[:-1], dtype=np.float32)
        
        # 强制单调递增 (Monotonicity Check)
        # 防止数据噪音导致 dt < 0，这会让 Neural ODE 求解器崩溃
        t_safe = np.maximum.accumulate(t_raw)
        
        # 防止完全相同的时间戳 (dt=0)，加上极小扰动
        # 比如: [0.1, 0.1] -> [0.1, 0.10001]
        epsilon = 1e-5
        t_safe = t_safe + np.arange(len(t_safe)) * epsilon
        
        t = torch.tensor(t_safe, dtype=torch.float32)
        
        # 如果需要，这里可以只返回最后一个 target 用于评估
        target_item = torch.tensor(items[-1], dtype=torch.long)

        return {
            "x": x,           # [Seq_len]
            "t": t,           # [Seq_len] (Strictly Increasing)
            "y": target_item  # Scalar
        }