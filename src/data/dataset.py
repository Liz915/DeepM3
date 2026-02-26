import torch
from torch.utils.data import Dataset
import pickle
import random
import os


def _infer_n_items(samples, data_dir):
    """
    Infer vocabulary size from metadata first, then fallback to sample scan.
    Returns n_items including padding index 0.
    """
    meta_path = os.path.join(data_dir, "ml1m_meta.pkl")
    if os.path.exists(meta_path):
        try:
            with open(meta_path, "rb") as f:
                meta = pickle.load(f)
            if isinstance(meta, dict):
                if "n_items" in meta and isinstance(meta["n_items"], int):
                    return int(meta["n_items"])
                # ML-1M metadata is usually {item_id: "title | genre"}.
                int_keys = [int(k) for k in meta.keys() if isinstance(k, int) or str(k).isdigit()]
                if int_keys:
                    return max(int_keys) + 1
        except Exception:
            pass

    max_id = 0
    for sample in samples:
        y = int(sample.get("y", 0))
        max_id = max(max_id, y)
        for v in sample.get("x", []):
            max_id = max(max_id, int(v))
    # +1 for zero-padding index
    return max(max_id + 1, 2)

class MovieLensDataset(Dataset):
    def __init__(self, mode='train', data_dir='data'):
        file_path = f"{data_dir}/ml1m_{mode}.pkl"
        print(f" Loading {mode} data from {file_path}...")
        
        with open(file_path, 'rb') as f:
            self.samples = pickle.load(f)
            
        self.mode = mode
        self.n_items = _infer_n_items(self.samples, data_dir)
        
        # Check if new format (has 'dt' field)
        self.has_dt = len(self.samples) > 0 and 'dt' in self.samples[0]
        
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        x = torch.tensor(sample['x'], dtype=torch.long)
        t = torch.tensor(sample['t'], dtype=torch.float32)
        pos = torch.tensor(sample['y'], dtype=torch.long)
        
        # dt: inter-event deltas (new format) or derive from t
        if self.has_dt:
            dt = torch.tensor(sample['dt'], dtype=torch.float32)
        else:
            # Fallback: derive dt from cumulative t
            dt = torch.zeros_like(t)
            dt[1:] = t[1:] - t[:-1]
        
        result = {'x': x, 't': t, 'dt': dt, 'pos': pos}
        
        if self.mode == 'train':
            neg = random.randint(1, self.n_items - 1)
            while neg == sample['y'] or neg in sample['x']:
                neg = random.randint(1, self.n_items - 1)
            result['neg'] = torch.tensor(neg, dtype=torch.long)
            
        return result
