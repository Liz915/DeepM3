import torch
import yaml
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os

sys.path.append(os.getcwd())
from src.data.dataset import MovieLensDataset
from src.dynamics.modeling import DeepM3Model

def plot_entropy_dist():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    with open('configs/config.yaml', 'r') as f: config = yaml.safe_load(f)
    dataset = MovieLensDataset(mode='test')
    
    #  RK4 RK4 
    #  Euler  appendix  RK4
    model = DeepM3Model(config, n_items=dataset.n_items, solver='rk4').to(device)
    model.load_state_dict(torch.load("checkpoints/model_ode_rk4.pth", map_location=device))
    model.eval()

    #  1000 
    indices = np.random.choice(len(dataset), 1000, replace=False)
    
    entropy_list = []
    
    print(" Calculating Entropy for distribution plot...")
    with torch.no_grad():
        for idx in tqdm(indices):
            sample = dataset[idx]
            x = torch.tensor(sample['x']).unsqueeze(0).to(device)
            t = torch.tensor(sample['t']).unsqueeze(0).to(device)
            dt = torch.tensor(sample['dt']).unsqueeze(0).to(device) if 'dt' in sample else None
            
            user_emb = model(x, t, dt=dt)
            all_items = model.get_item_embedding(torch.arange(1, dataset.n_items).to(device))
            logits = torch.matmul(user_emb, all_items.t())
            
            # 
            probs = torch.softmax(logits, dim=-1)
            # Shannon Entropy
            entropy = -torch.sum(probs * torch.log(probs + 1e-9)).item()
            entropy_list.append(entropy)
            
    # 
    plt.figure(figsize=(8, 6))
    sns.set_style("whitegrid")
    
    #  + 
    sns.histplot(entropy_list, kde=True, color="purple", bins=30)
    
    #  Router  ( Top 20% )
    threshold = np.percentile(entropy_list, 80)
    plt.axvline(threshold, color='red', linestyle='--', label=f'Router Threshold (Top 20%)')
    
    plt.title("Distribution of Prediction Entropy (System 1)", fontsize=14)
    plt.xlabel("Entropy (Uncertainty)", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.legend()
    
    save_path = "assets/entropy_distribution_appendix.png"
    os.makedirs("assets", exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f" Plot saved to {save_path}")

if __name__ == "__main__":
    plot_entropy_dist()
