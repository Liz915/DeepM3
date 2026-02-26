import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np
import os

def plot_phase_portrait(hidden_states, times, save_path="assets/attractor.png"):
    """
     2D 
    
    hidden_states: [seq_len, hidden_dim] ()
    times: [seq_len] 
    """
    # 1.  PCA  (Principal Component Analysis)
    # "" (Major Dynamics)
    pca = PCA(n_components=2)
    trajectory_2d = pca.fit_transform(hidden_states.detach().cpu().numpy())
    
    # 2. 
    plt.figure(figsize=(10, 8))
    ax = plt.gca()
    
    #  (Trajectory)
    #  (Time Evolution)
    #  -> ""
    scatter = ax.scatter(trajectory_2d[:, 0], trajectory_2d[:, 1], 
                         c=times.detach().cpu().numpy(), 
                         cmap='viridis', s=100, alpha=0.8, edgecolors='w')
    
    ax.plot(trajectory_2d[:, 0], trajectory_2d[:, 1], 
            c='gray', alpha=0.3, linestyle='--', linewidth=1)
    
    # 3.  (Math Style)
    plt.title("Latent Dynamics Phase Portrait (Neural ODE)", fontsize=16)
    plt.xlabel("Principal Component 1 (Trend)", fontsize=12)
    plt.ylabel("Principal Component 2 (Cycle)", fontsize=12)
    plt.colorbar(scatter, label="Time Evolution")
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # 4. 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    print(f" Visualization saved to: {save_path}")
    plt.close()