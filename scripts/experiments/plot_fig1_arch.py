import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os

def draw_neural_ode_arch():
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis('off')

    # --- 1. Input Sequence () ---
    for i in range(4):
        # Items (Squares)
        rect = patches.Rectangle((0.5 + i*1.2, 2.0), 0.6, 0.6, linewidth=1.5, edgecolor='#333333', facecolor='#E0E0E0')
        ax.add_patch(rect)
        ax.text(0.8 + i*1.2, 2.3, f"$x_{i+1}$", ha='center', va='center', fontsize=10)
        
        # Time Intervals (Clocks)
        ax.text(0.8 + i*1.2, 1.5, r"$\Delta t_" + str(i+1) + "$", ha='center', va='center', fontsize=9, color='blue')

    ax.text(2.5, 0.8, "Input Sequence", ha='center', fontsize=11, fontweight='bold')

    # --- 2. Neural ODE () ---
    #  hidden state 
    circles_x = [5.5, 6.5, 7.5]
    for x in circles_x:
        circle = patches.Circle((x, 2.3), 0.25, linewidth=1.5, edgecolor='#1f77b4', facecolor='white')
        ax.add_patch(circle)
        
    #  ODE  (Spiral Trajectory)
    # 
    t = np.linspace(0, 10, 100)
    x_spiral = 5.5 + t * 0.1
    y_spiral = 2.3 + 0.5 * np.sin(t * 3) * np.exp(-0.1 * t)
    # 
    t_seg = np.linspace(5.75, 6.25, 50)
    y_seg = 2.3 + 0.3 * np.sin((t_seg-5.5)*20) 
    ax.plot(t_seg, y_seg, color='#1f77b4', linestyle='-', linewidth=1.5)
    ax.text(6.0, 2.8, "ODE Solve", ha='center', fontsize=8, color='#1f77b4')

    t_seg2 = np.linspace(6.75, 7.25, 50)
    y_seg2 = 2.3 + 0.3 * np.sin((t_seg2-6.5)*20)
    ax.plot(t_seg2, y_seg2, color='#1f77b4', linestyle='-', linewidth=1.5)

    #  ODE 
    ax.text(6.5, 3.5, r"$\frac{dh(t)}{dt} = f(h(t), t)$", ha='center', fontsize=10, color='#1f77b4', 
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.8))

    # --- 3. Router () ---
    router = patches.RegularPolygon((8.5, 2.3), numVertices=4, radius=0.4, orientation=0, 
                                    facecolor='#fff9c4', edgecolor='#fbc02d', linewidth=1.5)
    ax.add_patch(router)
    ax.text(8.5, 2.3, "Entropy\nCheck", ha='center', va='center', fontsize=8)

    # --- 4. Branches () ---
    # Path A (Fast)
    ax.arrow(8.9, 2.3, 0.8, 0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(9.5, 2.5, "Low Entropy\n(Fast Path)", ha='center', fontsize=8, color='green')
    
    # Path B (Slow / System 2)
    ax.arrow(8.5, 2.7, 0, 1.0, head_width=0.1, head_length=0.1, fc='black', ec='black')
    ax.text(8.8, 3.2, "High\nEntropy", ha='left', fontsize=8, color='red')
    
    # System 2 Box
    rect_sys2 = patches.Rectangle((8.0, 3.8), 1.0, 0.8, linewidth=1.5, edgecolor='#d62728', facecolor='#ffcdd2')
    ax.add_patch(rect_sys2)
    ax.text(8.5, 4.2, "System 2\n(Reasoning)", ha='center', va='center', fontsize=9, color='#d62728')

    # Output
    ax.text(9.8, 2.3, "Top-K", ha='left', va='center', fontsize=10, fontweight='bold')

    # Save
    os.makedirs("assets", exist_ok=True)
    plt.savefig("assets/Fig1_Architecture.pdf", format='pdf', bbox_inches='tight')
    print(" Figure 1 saved to assets/Fig1_Architecture.pdf")

if __name__ == "__main__":
    draw_neural_ode_arch()