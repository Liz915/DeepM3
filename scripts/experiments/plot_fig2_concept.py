import matplotlib.pyplot as plt
import numpy as np
import os

def plot_concept_comparison():
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    #  ()
    t_obs = [1, 3, 6, 8]
    h_obs = [2, 4, 3, 5]
    
    # --- Left: RNN (Discrete) ---
    # Step function behavior
    t_rnn = [0, 1, 1, 3, 3, 6, 6, 8, 8, 10]
    h_rnn = [2, 2, 4, 4, 3, 3, 5, 5, 5, 5] # Last hold
    
    ax1.plot(t_rnn, h_rnn, color='gray', linestyle='--', linewidth=2, label='State Holding')
    ax1.scatter(t_obs, h_obs, color='black', s=50, zorder=5, label='Observation')
    
    # Draw vertical updates (Jumps)
    for i in range(len(t_obs)):
        if i > 0:
            ax1.arrow(t_obs[i], h_obs[i-1], 0, h_obs[i]-h_obs[i-1]-0.2, 
                      head_width=0.2, head_length=0.2, fc='black', ec='black', alpha=0.5)

    ax1.set_title("(a) RNN / Discrete Models", fontsize=12)
    ax1.set_xlabel("Time", fontsize=11)
    ax1.set_ylabel("User Latent State $h(t)$", fontsize=11)
    ax1.set_ylim(0, 6)
    ax1.grid(True, linestyle=':', alpha=0.3)
    ax1.legend(loc='lower right')
    
    # Annotation
    ax1.text(4.5, 4.5, "State constant\nbetween steps", ha='center', fontsize=9, color='gray')

    # --- Right: Neural ODE (Continuous) ---
    # Interpolation curve (Spline or smooth function)
    t_ode = np.linspace(0, 10, 200)
    #  ()
    p = np.polyfit(t_obs, h_obs, 3)
    h_ode = np.polyval(p, t_ode)
    
    ax2.plot(t_ode, h_ode, color='#D62728', linewidth=2.5, label='ODE Trajectory')
    ax2.scatter(t_obs, h_obs, color='black', s=50, zorder=5, label='Observation')
    
    # Draw arrows indicating flow
    ax2.arrow(2, np.polyval(p, 2), 0.1, (np.polyval(p, 2.1)-np.polyval(p, 2)), 
              head_width=0, head_length=0, fc='#D62728', ec='#D62728')

    ax2.set_title("(b) DeepM3 (Neural ODE)", fontsize=12)
    ax2.set_xlabel("Time", fontsize=11)
    ax2.set_yticks([]) # Y
    ax2.set_ylim(0, 6)
    ax2.grid(True, linestyle=':', alpha=0.3)
    ax2.legend(loc='lower right')

    # Annotation
    ax2.text(4.5, 1.0, "Continuous Evolution\n(Drift)", ha='center', fontsize=9, color='#D62728')

    # Save
    plt.tight_layout()
    plt.savefig("assets/Fig2_Concept.pdf", format='pdf')
    print(" Figure 2 saved to assets/Fig2_Concept.pdf")

if __name__ == "__main__":
    plot_concept_comparison()