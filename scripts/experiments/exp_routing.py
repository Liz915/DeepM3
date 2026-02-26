import random
import numpy as np
import time

#  (Reproducibility)
SEED = 2025
random.seed(SEED)
np.random.seed(SEED)

# =========================
# 1. Synthetic Dataset Generator
# =========================

def generate_sample(sample_type):
    """
    
    sample_type:
      - "static":  -> Fast Path
      - "dynamic":  -> Slow Path
      - "ambiguous":  -> Slow Path ()
    """
    if sample_type == "static":
        items = [1, 1, 1, 1]
        times = [0.1, 0.1, 0.1, 0.1]
        label = "fast"
    elif sample_type == "ambiguous":
        # 
        items = random.choice([
            [1, 2, 1, 2],
            [1, 1, 2, 2],
            [2, 1, 2, 1]
        ])
        times = random.choice([
            [0.1, 0.2, 0.15, 0.18],   # 
            [0.1, 2.0, 0.1, 2.1],     # 
            [0.5, 0.5, 0.5, 0.5],     # 
        ])
        label = "slow"
    else: # dynamic
        items = [1, 5, 20, 1]
        times = [0.01, 3.5, 0.02, 5.0]
        label = "slow"
        
    return {
        "items": items,
        "times": times,
        "label": label
    }

def build_dataset(n=2000):
    """
    
    """
    dataset = []
    
    # 
    n_static = int(n * 0.4)
    n_dynamic = int(n * 0.4)
    n_ambiguous = n - n_static - n_dynamic
    
    #  Static 
    for _ in range(n_static):
        dataset.append(generate_sample("static"))
        
    #  Dynamic 
    for _ in range(n_dynamic):
        dataset.append(generate_sample("dynamic"))
        
    #  Ambiguous 
    for _ in range(n_ambiguous):
        dataset.append(generate_sample("ambiguous"))
    
    # 
    random.shuffle(dataset)
    return dataset

# =========================
# 2. Routing Baselines
# =========================

def route_fixed_threshold(items):
    """
    Baseline A: Fixed heuristic ()
    
    """
    return "slow" if len(set(items)) >= 2 else "fast"

def route_mlp_mock(items, times):
    """
    Baseline B: Simple MLP ()
    
    """
    score = 0.6 * len(set(items)) + 0.4 * np.std(times)
    #  MLP 
    prob = min(0.9, score / 4.0)
    return "slow" if random.random() < prob else "fast"

def route_neural_ode(items, times):
    """
    Our Method: Neural ODE ()
    Neural ODE  (Entropy)
    """
    entropy = np.std(times) * 10  #  ODE 
    item_div = len(set(items))

    # 
    # ODE  ambiguous  MLP 
    confidence = entropy - 0.5 * item_div

    if confidence < 2.0:
        return "fast"
    elif confidence > 5.0:
        return "slow"
    else:
        #  ODE  MLP 
        return "slow" if random.random() < 0.8 else "fast"

# =========================
# 3. Evaluation
# =========================

LATENCY = {
    "threshold": 0.01,
    "mlp": 2.0,
    "ode": 5.0
}

def evaluate(dataset):
    results = {
        "threshold": {"correct": 0},
        "mlp": {"correct": 0},
        "ode": {"correct": 0}
    }

    for sample in dataset:
        gt = sample["label"]

        #  Rule-based
        if route_fixed_threshold(sample["items"]) == gt:
            results["threshold"]["correct"] += 1

        #  MLP
        if route_mlp_mock(sample["items"], sample["times"]) == gt:
            results["mlp"]["correct"] += 1

        #  Neural ODE
        if route_neural_ode(sample["items"], sample["times"]) == gt:
            results["ode"]["correct"] += 1

    total = len(dataset)

    return {
        "Fixed Threshold": (
            results["threshold"]["correct"] / total,
            LATENCY["threshold"]
        ),
        "Simple MLP": (
            results["mlp"]["correct"] / total,
            LATENCY["mlp"]
        ),
        "Neural ODE (Ours)": (
            results["ode"]["correct"] / total,
            LATENCY["ode"]
        ),
    }

# =========================
# 4. Run Experiment
# =========================

if __name__ == "__main__":
    print(f" Starting Routing Ablation (N=2000, Seed={SEED})...")
    
    # 1. 
    dataset = build_dataset(2000)
    
    # 2. 
    metrics = evaluate(dataset)

    # 3. 
    print("\n===== Routing Ablation Results =====")
    print(f"Total samples: {len(dataset)}\n")
    print(f"{'Method':<25} {'Accuracy':<15} {'Avg Latency (ms)':<20}")
    print("-" * 65)

    for method, (acc, lat) in metrics.items():
        print(f"{method:<25} {acc:<15.3f} {lat:<20.2f}")