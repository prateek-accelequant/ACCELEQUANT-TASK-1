"""
Module: utils.py
Description: Centralized mathematical utilities, persistent compute state tracking, 
             and finely-grained hierarchical quantum kernel matrix evaluators.
"""
import os
import json
import gc
import numpy as np
from scipy import stats
import config

def get_cache_dir(dataset_name, suite_name, ablation_params=None):
    """
    Creates a granular, highly organized cache hierarchy:
    QKE_Cache/[Dataset]/main/
    QKE_Cache/[Dataset]/ablation/[map]_[ent]_bw[bw]_noise[noise]/
    """
    clean_ds = "".join([c if c.isalnum() else "_" for c in dataset_name])
    
    if suite_name == "kernels_main":
        path = os.path.join(config.CACHE_DIR_BASE, clean_ds, "main")
    elif suite_name == "kernels_ablation" and ablation_params:
        fmap, ent, bw, noise = ablation_params
        sub_folder = f"{fmap}_ent{ent}_bw{bw}_noise{noise}"
        path = os.path.join(config.CACHE_DIR_BASE, clean_ds, "ablation", sub_folder)
    else:
        path = os.path.join(config.CACHE_DIR_BASE, clean_ds, suite_name)
        
    os.makedirs(path, exist_ok=True)
    return path

def get_compute_state_file(dataset_name):
    clean_ds = "".join([c if c.isalnum() else "_" for c in dataset_name])
    return os.path.join("results", f"compute_footprint_{clean_ds}.json")

def load_persistent_compute_state(dataset_name):
    filepath = get_compute_state_file(dataset_name)
    if os.path.exists(filepath):
        try:
            with open(filepath, "r") as f:
                return json.load(f).get("total_elapsed_seconds", 0.0)
        except Exception:
            return 0.0
    return 0.0

def save_persistent_compute_state(dataset_name, additional_seconds):
    os.makedirs("results", exist_ok=True)
    current_total = load_persistent_compute_state(dataset_name) + additional_seconds
    with open(get_compute_state_file(dataset_name), "w") as f:
        json.dump({"total_elapsed_seconds": current_total}, f, indent=4)
    return current_total

def calculate_95_ci(data):
    mean = np.mean(data)
    n = len(data)
    if n <= 1: return mean, 0.0
    se = stats.sem(data)
    return mean, se * stats.t.ppf((1 + 0.95) / 2., n - 1)

def calculate_kernel_target_alignment(K_train, y_train):
    y_mapped = np.where(y_train == 0, -1, 1)
    y_vec = np.reshape(y_mapped, (-1, 1))
    K_target = y_vec @ y_vec.T
    inner_product = np.sum(K_train * K_target)
    norm_K = np.linalg.norm(K_train, ord='fro')
    norm_target = np.linalg.norm(K_target, ord='fro')
    if norm_K == 0 or norm_target == 0:
        return 0.0
    return inner_product / (norm_K * norm_target)

def evaluate_kernel_in_chunks(qkernel, X1, X2=None, chunk_size=100, verbose=False):
    symmetric = False
    if X2 is None:
        X2 = X1
        symmetric = True

    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))
    current_chunk = 0

    if verbose:
        print(f"   [Quantum Engine] Starting vectorized chunked evaluation ({n1}x{n2}, chunk size {chunk_size})...")

    for i in range(0, n1, chunk_size):
        end_i = min(i + chunk_size, n1)
        X1_chunk = X1[i:end_i]
        
        for j in range(0, n2, chunk_size):
            end_j = min(j + chunk_size, n2)
            if symmetric and j < i:
                continue
                
            X2_chunk = X2[j:end_j]
            current_chunk += 1
            
            if verbose:
                print(f"   [Quantum Engine] Processing chunk block [{i}:{end_i}, {j}:{end_j}]...")
                
            K_block = qkernel.evaluate(x_vec=X1_chunk, y_vec=X2_chunk)
            K[i:end_i, j:end_j] = K_block
            
            if symmetric and i != j:
                K[j:end_j, i:end_i] = K_block.T
        gc.collect()

    return K