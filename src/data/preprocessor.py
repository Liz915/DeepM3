import pandas as pd
import numpy as np
import torch
import os
import sys

class MovieLensProcessor:
    def __init__(self, data_path="data/raw/ml-1m/ratings.dat", seq_len=20, min_len=5):
        # 获取当前脚本运行的根目录，确保能找到 data 文件夹
        current_dir = os.getcwd()
        # 如果 data_path 是相对路径，拼接上当前目录
        if not os.path.isabs(data_path):
            self.data_path = os.path.join(current_dir, data_path)
        else:
            self.data_path = data_path
            
        self.seq_len = seq_len
        self.min_len = min_len
        self.user2id = {}
        self.item2id = {}
        self.n_items = 0
        
    def load_data(self):
        print(f"🔄 Loading MovieLens data from: {self.data_path}")
        
        if not os.path.exists(self.data_path):
            raise FileNotFoundError(f"Raw data not found at {self.data_path}")

        # MovieLens 1M format: UserID::MovieID::Rating::Timestamp
        df = pd.read_csv(self.data_path, sep='::', header=None, engine='python',
                         names=['uid', 'mid', 'rating', 'timestamp'])
        
        # 1. 重新编码 ID (0 号留给 padding)
        self.item2id = {mid: i+1 for i, mid in enumerate(df['mid'].unique())}
        self.n_items = len(self.item2id) + 1
        df['item_idx'] = df['mid'].map(self.item2id)
        
        # 2. 按用户分组并排序
        print("🔄 Grouping by user and sorting time...")
        data_group = df.sort_values(['uid', 'timestamp']).groupby('uid')
        
        sequences = []
        for uid, group in data_group:
            items = group['item_idx'].values.tolist()
            times = group['timestamp'].values.tolist()
            
            if len(items) < self.min_len:
                continue
                
            # 切分序列 (只取最近的一段 seq_len)
            seq_items = items[-self.seq_len:]
            seq_times = times[-self.seq_len:]
            
            # Padding (不够长的补 0)
            pad_len = self.seq_len - len(seq_items)
            if pad_len > 0:
                seq_items = [0] * pad_len + seq_items
                seq_times = [seq_times[0]] * pad_len + seq_times
            
            # Time Normalization
            # 归一化时间戳，防止数值过大导致 ODE 梯度爆炸
            seq_times = np.array(seq_times)
            t0 = seq_times[0]
            # 缩放到 [0, 10] 左右的区间
            norm_times = (seq_times - t0) / 3600.0 / 24.0 / 10.0 
            norm_times = np.maximum.accumulate(norm_times)
            
            sequences.append((seq_items, norm_times.tolist()))
            
        return sequences

    def save_processed(self, save_path="data/processed.pt"):
        try:
            seqs = self.load_data()
            print(f"✅ Processed {len(seqs)} sequences. Saving to {save_path}...")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save({
                "sequences": seqs,
                "n_items": self.n_items,
                "item2id": self.item2id
            }, save_path)
            print("🎉 Data processing complete!")
        except Exception as e:
            print(f"❌ Error during processing: {e}")


if __name__ == "__main__":
    # 自动检查并尝试下载
    raw_path = "data/raw/ml-1m/ratings.dat"
    zip_path = "data/raw/ml-1m.zip"
    
    if not os.path.exists(raw_path):
        print(f"⚠️ Data not found at {raw_path}")
        print("🔄 Attempting automatic download...")
        try:
            os.makedirs("data/raw", exist_ok=True)
            # 使用 curl 下载
            if os.system(f"curl -o {zip_path} https://files.grouplens.org/datasets/movielens/ml-1m.zip") == 0:
                print("✅ Download success. Unzipping...")
                os.system(f"unzip -o {zip_path} -d data/raw")
            else:
                raise Exception("Curl command failed.")
        except Exception as e:
            print(f"❌ Automatic download failed: {e}")
            print("💡 Please manually download ml-1m.zip from grouplens.org and place it in 'data/raw/'.")
            sys.exit(1)

    print("🚀 Starting Preprocessor...")
    processor = MovieLensProcessor(data_path=raw_path)
    processor.save_processed()