# DeepM3: Continuous-Time Dynamics for Session-based Recommendation via Neural ODEs

DeepM3 resolves the intrinsic limitation of discrete-time sequential models (e.g., RNNs, Transformers) when processing real-world interactions subject to irregular time intervals. By framing user intention evolution as a **Continuous-Time Dynamical System**, DeepM3 enables intentions to evolve smoothly in the latent space during periods of inactivity.

Furthermore, we position the recommendation environment — characterized by shifting user interests and item availability — as a **Random Dynamical System (RDS)**. Under high environmental noise (measurable via *Random Topological Entropy*), discrete models shatter due to over-sensitivity or lack of time-awareness. Our ODE-based architecture, backed by Lipschitz continuity, provides extreme robustness, mathematically mitigating the destructive impact of stochastic sequence corruption.

---

## 🚀 Key Results (KBS Submission)

Our experiments comprehensively contrast highly structured data (**MovieLens-1M**) with extremely sparse, irregular data (**Amazon Books**).

### 1. Robustness Against High Irregularity
DeepM3 excels precisely when time intervals become chaotic (High Coefficient of Variation - CV). 
* **ML-1M (Dense):** Continuous integration yields **+2.27%** significant improvement specifically in the most irregular sequences.
* **Amazon Books (Sparse & High Noise):** The model showcases immense robustness when time noise is synthetically injected, suffering only a **0.36% NDCG drop**, whereas traditional methods collapse or ignore structural time mechanics.

### 2. Efficiency Trade-offs
We offer mathematically rigorous solvers for different industrial constraints:
* **DeepM3 (RK4)**: Highest precision (NDCG@10: 0.4078 on ML-1M).
* **DeepM3 (Euler)**: Low-latency industrial adaptation (Latency halved compared to RK4, while maintaining strong statistical baseline improvements).
* **DeepM3 (800K Params)** rivals or surpasses **SASRec (2.6M Params)** acting as a lightweight framework.

---

## ⚙️ Reproducibility Pipeline

To ensure absolute reproducibility for KBS peer review, we provide 1-click execution scripts.

### 1. Requirements
```bash
conda create -n DeepM3 python=3.10
conda activate DeepM3
pip install torch pandas numpy scipy pyyaml tqdm
```

### 2. Run the Full ML-1M Pipeline (Dense Data)
Downloads the dataset, tunes baselines, trains DEEPM3 variants, and runs the entire Phase-2 experiment suite (Sensitivity, Robustness, Efficiency, Significance, Irregularity).
```bash
DEVICE=auto bash scripts/experiments/run_phase2.sh
# Results will be generated in `results/ml1m/`
```

### 3. Run the Full Amazon Pipeline (Sparse/Noisy Data)
Automatically subsamples Amazon 5-core books, processes sequences, and outputs comparative validations.
```bash
DEVICE=auto bash scripts/experiments/run_amazon.sh
# Results will be generated in `results/amazon/`
```

---

## 🧠 Architectural Highlights
1. **Time-Aware ODE Integration:** Unobserved spans dictate the ODE integration step $(dt)$, adjusting hidden states without artificial sequence padding.
2. **RDS Resilience:** Handles random time perturbation and sequential dropout natively without re-training.
3. **Pluggable Solvers:** Supports `Euler` (O(1) step) and `RK4` (O(4) step) tailored for edge-computing or datacenter capacities.

## Citation
*(Anonymous for KBS Double-Blind Review)*
