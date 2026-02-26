# DeepM3: 基于神经 ODE 的连续时间动态会话推荐模型

![DeepM3 核心架构](assets/Fig2_Concept.pdf) <!-- 请替换为您稍后生成的封面图 -->

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![匿名审核中](https://img.shields.io/badge/Status-Double--Blind%20Review-red)]()

> **注意**: 本开源仓库为 **DeepM3** 的官方匿名实现，专供双盲同行评审（Double-Blind Peer Review）使用。相关的机构与作者署名已被永久性隐藏，直至随论文正式发表同步公开。

## 📖 研究摘要

传统的序列推荐模型（如深度循环神经网络 RNNs 或自注意力模型 Transformers）在构建底层特征演化时，隐式地假设了用户交互时间是等距切分的。这种激进的离散化（Discretization）破坏了用户意图演化的真实连续性，导致模型在面对高度不规则（High Irregularity）的工业长尾数据时极其脆弱。

本研究提出了 **DeepM3**，一个构建于**连续时间动力系统 (Continuous-Time Dynamical Systems)** 基础之上的全新范式。我们将用户意图在潜在空间中的演变视作微积分连续轨迹，并嵌入**常微分方程 (Ordinary Differential Equations, ODEs)**。此举彻底消除了对虚拟事件的零点填充 (Zero-padding) 要求。

我们更进一步论证指出，现实世界中充斥着变数的用户偏好实际上构成了一个充满环境扰动的**随机动力系统 (Random Dynamical Systems, RDS)**。在此类环境中，用*随机拓扑熵 (Random Topological Entropy)* 衡量可知，序列结构具有极高考验。本基于数值解法并拥有李普希茨连续 (Lipschitz continuity) 约束兜底的架构，从纯数学维度提供了难以置信的系统鲁棒性，免疫序列时间摧毁造成的性能崩塌。

---

## 🚀 核心贡献与性能闭环

为了支撑严谨的动力系统框架，本次大范围实验严格对比了极度高质量的稠密数据 (MovieLens-1M) 与极度稀疏、跨度长尾的混沌数据 (Amazon Books)。

### 1. 免疫长尾不确定性 (High Topological Entropy)
当交互序列的变异系数（CV）陡升，甚至完全偏离统计平稳分布时，常态模型结构迎来坍塌。
* **高质序列修复与增幅:** 在面对 ML-1M 验证集中最不规律、最混乱的交互群组时，连续时间积分引擎成功实现了“以平滑逆转杂乱”，带来高达 **+2.27%** 的显著指标回涨。
* **极限 RDS 噪音抗性测试 (Amazon Books):** 我们对序列施以最严苛的打乱 (Shuffle)、高斯抖动与掉线丢失。在基线被彻底撕裂的同时，DeepM3 仅极其微小地掉了不到 **0.36% (NDCG)**，这种军工级的“抗震防噪机制”验证了理论模型。

### 2. 精确度与落地工程效率的双赢
鉴于边端部署限制，DeepM3 放出了自适配的 ODE 算术求解降级策略：
* **理论上限 (RK4 龙格库塔)**: 以 4 阶积分步骤追求绝对最高精度。
* **工业提速 (Euler 欧拉)**: 定制化的低延迟 1 阶版本，砍半甚至缩减三分之二计算延迟的同时，换取了优于当前 SOTA 的统计增幅。
* **强力减重**: DeepM3 的基础物理参数量约 **80 万**左右，轻松打败庞大且具有 **260 万**参数的 SASRec / TiSASRec 巨型编码器。事实证明，贴近本源客观规律的设计，比暴力的参数膨胀更具科研吸引力。

---

## ⚙️ 一键复现套件 (Reproducibility Sandbox)

作为可复现科学体系的遵循者，我们公开底层沙盒脚本供一键触发多阶段训练、探测和可视化流程。

### 1. 宿主环境依赖搭建
```bash
conda create -n DeepM3 python=3.10
conda activate DeepM3
pip install torch pandas numpy scipy pyyaml tqdm
```

### 2. 执行常规稠密场景分析 (ML-1M)
系统将全自动拉取原始结构，并在 MPS / Cuda 完成包括最佳泛化搜索集、核心训练以及 8 项完整深度透析实验：
```bash
DEVICE=auto bash scripts/experiments/run_phase2.sh
# 分析结果和统计图表将在 `results/ml1m/` 内落盘。
```

### 3. 执行稀疏高噪场景分析 (Amazon 5-core)
拉取巨大 Amazon 语料，执行自动流转化与序列筛选压缩。计算模型在随机扰动的 RDS 环境中的极限能力。
```bash
DEVICE=auto bash scripts/experiments/run_amazon.sh
# 分析结果和统计图表将在 `results/amazon/` 内落盘。
```

---

## 📜 论文引用 (Citation)

*(当前正处于国际双盲同行评审阶段，相关引用标记、开放 DOI 与预印本链接将于近期随录用结果对外开放获取)*
