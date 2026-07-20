"""
Module: ablation_studies.py
Description: Parallelized, GPU-accelerated multi-dimensional ablation suite built with Joblib, 
             Joblib progress tracking, and Qiskit Aer statevector multithreading. Automatically 
             serializes all summary text logs, quantitative performance ledgers, and visual alignment 
             plots directly into the results archive repository.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from sklearn.svm import SVC
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

import config
from data_pipeline import QKEDataPipeline
from models_quantum import ProductionQuantumKernelManager

def calculate_kernel_target_alignment(K_train, y_train):
    """Computes the Quantum Kernel Target Alignment metric."""
    y_mapped = np.where(y_train == 0, -1, 1)
    y_vec = np.reshape(y_mapped, (-1, 1))
    
    K_target = y_vec @ y_vec.T
    inner_product = np.sum(K_train * K_target)
    
    norm_K = np.linalg.norm(K_train, ord='fro')
    norm_target = np.linalg.norm(K_target, ord='fro')
    
    if norm_K == 0 or norm_target == 0:
        return 0.0
    return inner_product / (norm_K * norm_target)

def run_single_ablation(fmap_type, ent, bw, use_noise, X_train_raw, X_test_raw, y_train, y_test, clean_ds):
    """Isolated worker process for parallel CPU/GPU execution."""
    config.ENTANGLEMENT = ent
    config.USE_NISQ_NOISE = use_noise
    
    cache_suffix = f"map_{fmap_type}_ent_{ent}_bw_{bw}_noise_{use_noise}_seed{config.SEED}"
    train_cache_file = f"cache_ablation_{clean_ds}_{cache_suffix}_train.npy"
    test_cache_file = f"cache_ablation_{clean_ds}_{cache_suffix}_test.npy"
    
    pipeline = QKEDataPipeline(seed=config.SEED)
    pca_train = pipeline.preprocess(X_train_raw)
    pca_test = pipeline.preprocess(X_test_raw)
    
    X_train_scaled = pca_train * bw
    X_test_scaled = pca_test * bw
    
    mgr = ProductionQuantumKernelManager(map_type=fmap_type, use_noise=use_noise)
    res_specs = mgr.get_resource_counts()
    
    if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        K_train = np.load(train_cache_file)
        K_test = np.load(test_cache_file)
    else:
        K_train = mgr.kernel.evaluate(x_vec=X_train_scaled)
        K_test = mgr.kernel.evaluate(x_vec=X_test_scaled, y_vec=X_train_scaled)
        np.save(train_cache_file, K_train)
        np.save(test_cache_file, K_test)
    
    alignment = calculate_kernel_target_alignment(K_train, y_train)
    spectral = mgr.calculate_spectral_diagnostics(K_train)
    
    svc = SVC(kernel='precomputed', C=1.0, probability=True, random_state=config.SEED)
    svc.fit(K_train, y_train)
    
    preds = svc.predict(K_test)
    probs = svc.predict_proba(K_test)[:, 1]
    
    return {
        'Feature Map': fmap_type,
        'Entanglement': ent,
        'Bandwidth': bw,
        'Noisy Backend': use_noise,
        'Qubits': res_specs['qubits'],
        'CNOT Count': res_specs['cnot_gates'],
        'Target Alignment': alignment,
        'Spectral Variance': spectral['variance'],
        'Condition Number': spectral['condition_number'],
        'Test F1': f1_score(y_test, preds),
        'Test AUC': roc_auc_score(y_test, probs)
    }

def run_ablation_matrix_suite(dataset_name, X_raw_all, y_all):
    clean_ds = "".join([c if c.isalnum() else "_" for c in dataset_name])
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_raw_all, y_all))
    
    # Bounded truncation to completely prevent exponential O(N^2) stalls on large real data pools
    max_train_pool = min(1000, len(train_idx))
    max_test_pool = min(500, len(test_idx))
    train_idx = train_idx[:max_train_pool]
    test_idx = test_idx[:max_test_pool]
    
    X_train_raw, X_test_raw = X_raw_all[train_idx], X_raw_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    
    feature_maps = ['ZZ', 'CPMap']
    entanglements = ['linear', 'full']
    bandwidths = [0.5, 1.0, 2.0]
    noise_options = [False, True]
    
    tasks = []
    for fmap_type in feature_maps:
        for ent in entanglements:
            for bw in bandwidths:
                for use_noise in noise_options:
                    tasks.append((fmap_type, ent, bw, use_noise))
                    
    print(f"Launching parallel GPU/CPU ablation suite for '{dataset_name}' ({len(tasks)} parallel configurations)...")
    
    # Parallel execution with verbose joblib logging status updates
    ablation_records = Parallel(n_jobs=-1, verbose=10)(
        delayed(run_single_ablation)(
            *task, X_train_raw, X_test_raw, y_train, y_test, clean_ds
        ) for task in tasks
    )
    
    df_results = pd.DataFrame(ablation_records)
    os.makedirs("results", exist_ok=True)
    
    # 1. RECORD/SAVE QUANTITATIVE CSV LEDGER
    csv_out = os.path.join("results", f"ablation_report_{clean_ds}.csv")
    df_results.to_csv(csv_out, index=False)
    
    # 2. RECORD/SAVE TEXT REPORT SUMMARY
    txt_out = os.path.join("results", f"ablation_summary_{clean_ds}.txt")
    with open(txt_out, "w") as f:
        f.write(f"=== ABLATION STUDY SUMMARY: {dataset_name} ===\n\n")
        f.write(df_results.to_string(index=False))
        
    # 3. RECORD/SAVE VISUAL PLOTS (Target Alignment & F1 vs Bandwidth)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for fmap in feature_maps:
        sub_df = df_results[df_results['Feature Map'] == fmap]
        grouped = sub_df.groupby('Bandwidth')['Target Alignment'].mean()
        axes[0].plot(grouped.index, grouped.values, marker='o', label=f"{fmap} Map", linewidth=2)
        
        grouped_f1 = sub_df.groupby('Bandwidth')['Test F1'].mean()
        axes[1].plot(grouped_f1.index, grouped_f1.values, marker='s', label=f"{fmap} Map", linewidth=2)
        
    axes[0].set_title(f"{dataset_name}: Kernel Target Alignment vs Bandwidth", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Bandwidth Multiplier ($\sigma$)", fontsize=10)
    axes[0].set_ylabel("Kernel Target Alignment", fontsize=10)
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    axes[1].set_title(f"{dataset_name}: Test F1-Score vs Bandwidth", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Bandwidth Multiplier ($\sigma$)", fontsize=10)
    axes[1].set_ylabel("Test F1 Score", fontsize=10)
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.suptitle("Automated Parallel Quantum Ablation Diagnostics", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_out = os.path.join("results", f"ablation_plots_{clean_ds}.png")
    fig.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    
    print(f"\n[Artifact Hub] Ablation results safely recorded:")
    print(f" -> CSV Ledger: {csv_out}")
    print(f" -> Text Summary: {txt_out}")
    print(f" -> Visual Plots: {plot_out}\n")
    
    return df_results

if __name__ == "__main__":
    pipeline = QKEDataPipeline(seed=config.SEED)
    X_synth_raw, y_synth = pipeline.generate_synthetic_shells()
    run_ablation_matrix_suite("Primary Synthetic Shells", X_synth_raw, y_synth)