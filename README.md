# DeepM3: Continuous-Time Dynamics for Irregular Session-based Recommendation via Neural ODEs

[**English**](./README.md) | [**中文版**](./README.zh-CN.md)

<div align="center">
  <img src="assets/cover.webp" alt="DeepM3 Architecture Concept" width="800">
</div>

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Anonymous](https://img.shields.io/badge/Status-Double--Blind%20Review-red)]()

> **Note**: This repository contains the official implementation of **DeepM3**, designed for anonymous peer review. Institutional affiliations and author information have been removed.

## 📖 Abstract

Traditional sequential recommendation models (e.g., RNNs, Transformers) implicitly assume uniform time intervals between user interactions. By discretizing sequences, they fail to capture the true continuous nature of user intention evolution, leading to brittle performance when faced with high temporal irregularity.

This paper proposes **DeepM3**, a novel recommendation framework built upon **Continuous-Time Dynamical Systems**. By modeling user intention as a continuous latent trajectory integrating Ordinary Differential Equations (ODEs), DeepM3 smoothly evolves hidden states across unobserved time spans.

Furthermore, we conceptualize real-world industrial recommendations (e.g., highly volatile user interests, shifting item availability) as **Random Dynamical Systems (RDS)**. Under environments with high *Random Topological Entropy* (indicative of extreme temporal noise and structural sparsity), discrete sequence alignment shatters. Our empirically validated ODE architecture—anchored by Lipschitz continuity—provides extreme mathematical robustness against stochastic temporal corruption.

---

## 🚀 Key Results & Contributions

Our extensive evaluation spans heavily structured, dense data (MovieLens-1M) and extremely sparse, highly variable long-tail data (Amazon Books). DeepM3 consistently isolates the core mathematical principles governing structural time versus chaotic time.

### 1. Robustness Against High Irregularity
DeepM3 excels structurally when time intervals become chaotic (measured via Coefficient of Variation, *CV*). 
* **Structuring Density:** Continuous integration yields **+2.27%** significant improvement over baselines specifically within the highest irregularity grouping of ML-1M.
* **Resilience Under RDS Noise (Amazon):** When time noise and packet-dropout are synthetically injected into extreme long-tail sequences, traditional methods collapse. DeepM3 showcases immense robustness, suffering only a **0.36% NDCG drop**, proving its resistance to high-entropy temporal structures.

### 2. Efficiency & Industrial Adaptability
We provide interchangeable ODE solvers tuned for hardware constraints:
* **DeepM3 (RK4)**: 4th-order Runge-Kutta for theoretical maximal precision.
* **DeepM3 (Euler)**: 1st-order solver tailored for low-latency production environments (Halves inference latency with negligible metric degradation).
* **DeepM3 (Lightweight)**: At ~800K parameters, DeepM3 outperforms massive Transformer baselines (e.g., SASRec/TiSASRec at 2.6M params), proving that continuous mechanics trump raw parameter scaling.

---

## ⚙️ Reproducibility Pipeline

To ensure absolute reproducibility for peer review, we provide end-to-end sandbox scripts executing tuning, training, and 8-stage comprehensive analysis.

### 1. Requirements & Setup
```bash
conda create -n DeepM3 python=3.10
conda activate DeepM3
pip install torch pandas numpy scipy pyyaml tqdm
```

### 2. Full Suite (Structured Regular Data)
Downloads `ml-1m`, executes baseline grid tuning, trains four DeepM3 variants, and generates Phase-2 artifacts (Sensitivity, Robustness, Efficiency, Paired Significance, Irregularity Analysis).
```bash
DEVICE=auto bash scripts/experiments/run_phase2.sh
# All artifacts saved to `results/ml1m/`
```

### 3. Full Suite (Sparse / High Topological Entropy Data)
Processes the massive `Amazon 5-core Books` dataset. Computes the same 8-stage rigorous corruption/evaluation suite specifically to measure RDS variance.
```bash
DEVICE=auto bash scripts/experiments/run_amazon.sh
# All artifacts saved to `results/amazon/`
```

---

## 🧠 Architectural Insights

1. **Continuous Time Integration (Non-Uniform):** The span between observed events explicitly dictates the ODE integration bounds $([t_{i-1}, t_i] \to dt)$. The hidden intention vector gracefully drifts without zero-padding alignments.
2. **Lipschitz-Bounded Evolution:** Unlike recurrent blocks prone to gradient explosion under volatile steps, our neural derivatives tightly bound extreme trajectory leaps, making DeepM3 naturally immune to timeline scrambling.
