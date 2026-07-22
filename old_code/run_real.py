"""
Module: real_engine.py
Description: Dedicated orchestration engine for the Rebalanced Real-World Fraud Dataset.
             Features memory-safe truncation pools and custom chunking tailored for large tabular inputs.
"""
import os
import sys

# =====================================================================
# 1. RUNTIME RESOURCE ALLOCATION
# =====================================================================
print("\n" + "="*60)
print("   Quantum Kernel Evaluation - REAL DATA ENGINE (Fraud)   ")
print("="*60)
try:
    _user_input = input("Enter the number of parallel workers to use [Default: 4]: ").strip()
    N_CORES = int(_user_input) if _user_input else 4
except ValueError:
    print("[!] Invalid input. Defaulting to 4 workers for stability.")
    N_CORES = 4

print(f"\n[*] Initializing environment with {N_CORES} protected workers...")
safe_threads = str(N_CORES)
os.environ['OMP_NUM_THREADS'] = safe_threads
os.environ['MKL_NUM_THREADS'] = safe_threads
os.environ['OPENBLAS_NUM_THREADS'] = safe_threads
os.environ['RAY_NUM_THREADS'] = safe_threads

# =====================================================================
# 2. HEAVY LIBRARY IMPORTS & HEADLESS GUI FIX
# =====================================================================
import time
import json
import gc
import multiprocessing
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC

from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.num_workers = 1

import config
from data_pipeline import QKEDataPipeline
from models_classical import ClassicalBaselineManager
from models_quantum import ProductionQuantumKernelManager

# =====================================================================
# 3. CACHE & COMPUTE MANAGERS
# =====================================================================
COMPUTE_STATE_FILE = os.path.join("results", "compute_footprint_real.json")

def load_persistent_compute_state():
    if os.path.exists(COMPUTE_STATE_FILE):
        try:
            with open(COMPUTE_STATE_FILE, "r") as f:
                data = json.load(f)
                return data.get("total_elapsed_seconds", 0.0)
        except Exception:
            return 0.0
    return 0.0

def save_persistent_compute_state(additional_seconds):
    os.makedirs("results", exist_ok=True)
    current_total = load_persistent_compute_state() + additional_seconds
    with open(COMPUTE_STATE_FILE, "w") as f:
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

def evaluate_kernel_in_chunks(qkernel, X1, X2=None, chunk_size=50):
    """
    Memory-safe and deadlock-free chunked kernel evaluator optimized for real data.
    Uses a smaller chunk size (50) and includes active progress tracking.
    """
    symmetric = False
    if X2 is None:
        X2 = X1
        symmetric = True

    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))

    total_chunks = ((n1 + chunk_size - 1) // chunk_size) * ((n2 + chunk_size - 1) // chunk_size)
    current_chunk = 0

    print(f"   [Quantum Engine] Starting vectorized chunked evaluation ({n1}x{n2} matrix, chunk size {chunk_size})...")

    for i in range(0, n1, chunk_size):
        end_i = min(i + chunk_size, n1)
        X1_chunk = X1[i:end_i]
        
        for j in range(0, n2, chunk_size):
            end_j = min(j + chunk_size, n2)
            if symmetric and j < i:
                continue
                
            X2_chunk = X2[j:end_j]
            current_chunk += 1
            
            # Print explicit feedback so you know it's moving past K(98,99)
            print(f"   [Quantum Engine] Processing chunk block [{i}:{end_i}, {j}:{end_j}] ({current_chunk} total blocks)...")
            
            K_block = qkernel.evaluate(x_vec=X1_chunk, y_vec=X2_chunk)
            K[i:end_i, j:end_j] = K_block
            
            if symmetric and i != j:
                K[j:end_j, i:end_i] = K_block.T
                
        gc.collect()

    print("   [Quantum Engine] Kernel evaluation matrix successfully completed.")
    return K

# =====================================================================
# 4. EXECUTION ENGINES
# =====================================================================

def run_sample_efficiency_suite(X_all, y_all, dataset_name):
    print(f"\n==================================================================")
    print(f"STARTING SAMPLE-EFFICIENCY SUITE FOR: {dataset_name} (Total Loaded: {len(X_all)})")
    print(f"==================================================================")
    
    start_time = time.time()
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    # --- MEMORY PROTECTION: Cap Real Data Pool to Avoid 30M Circuit Explosion ---
    max_train_pool = min(1000, len(train_idx))
    max_test_pool = min(500, len(test_idx))
    train_idx = train_idx[:max_train_pool]
    test_idx = test_idx[:max_test_pool]
    print(f"   [Real Data Limiter] Bounded pool to {len(train_idx)} train / {len(test_idx)} test points.")
    
    X_train_full, X_test = X_all[train_idx], X_all[test_idx]
    y_train_full, y_test = y_all[train_idx], y_all[test_idx]
    
    active_models = [m for m, active in config.RUN_MODELS.items() if active]
    performance_log = {m: {N: {'f1': [], 'auc': []} for N in config.N_LIST} for m in active_models}
    cost_log = {m: {N: {'circuits': 0, 'single_qubit': 0, 'cnot': 0} for N in config.N_LIST} for m in active_models}
    
    rng = np.random.default_rng(config.SEED)
    classical_mgr = ClassicalBaselineManager(seed=config.SEED)
    
    master_kernels = {}
    quantum_managers = {}
    clean_ds_name = "".join([c if c.isalnum() else "_" for c in dataset_name])
    
    for q_name, map_type in [('Quantum-ZZ', 'ZZ'), ('Quantum-CPMap', 'CPMap')]:
        if config.RUN_MODELS[q_name]:
            mgr = ProductionQuantumKernelManager(map_type=map_type)
            quantum_managers[q_name] = mgr
            
            raw_train_name = f"cache_{clean_ds_name}_{q_name}_seed{config.SEED}_train.npy"
            raw_test_name = f"cache_{clean_ds_name}_{q_name}_seed{config.SEED}_test.npy"
            
            train_cache_file = os.path.join(config.CACHE_DIR_KERNELS_MAIN, raw_train_name)
            test_cache_file = os.path.join(config.CACHE_DIR_KERNELS_MAIN, raw_test_name)
            
            if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
                print(f"   [Cache Hit] Loading precomputed {q_name} kernel matrices instantly...")
                K_train_master = np.load(train_cache_file)
                K_test_master = np.load(test_cache_file)
            else:
                print(f"   [Cache Miss] Computing {q_name} kernel matrices for {dataset_name} in chunks...")
                K_train_master = evaluate_kernel_in_chunks(mgr.kernel, X1=X_train_full, chunk_size=100)
                K_test_master = evaluate_kernel_in_chunks(mgr.kernel, X1=X_test, X2=X_train_full, chunk_size=100)
                np.save(train_cache_file, K_train_master)
                np.save(test_cache_file, K_test_master)
                
            master_kernels[q_name] = {'train': K_train_master, 'test': K_test_master}

    spectral_log = {}
    for q_name in master_kernels:
        spectral_log[q_name] = quantum_managers[q_name].calculate_spectral_diagnostics(master_kernels[q_name]['train'])

    summary_data = {m: {'N': [], 'F1': [], 'AUC': []} for m in active_models}

    for N in config.N_LIST:
        print(f"\n--- Running Sweep Size: N = {N} ---")
        split_iterator = tqdm(range(config.N_SPLITS), desc=f"Evaluating N={N}", leave=True)
        
        for split in split_iterator:
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            
            replace_pos = (N // 2 > len(pos_idx))
            replace_neg = (N // 2 > len(neg_idx))
            
            sampled_pos = rng.choice(pos_idx, size=N // 2, replace=replace_pos)
            sampled_neg = rng.choice(neg_idx, size=N // 2, replace=replace_neg)
            sub_idx = rng.permutation(np.concatenate([sampled_pos, sampled_neg]))
            
            X_train_sub = X_train_full[sub_idx]
            y_train_sub = y_train_full[sub_idx]
            
            if 'RBF-SVC' in active_models:
                svc_clf, best_params = classical_mgr.fit_rbf_svc(X_train_sub, y_train_sub)
                performance_log['RBF-SVC'][N]['f1'].append(f1_score(y_test, svc_clf.predict(X_test)))
                performance_log['RBF-SVC'][N]['auc'].append(roc_auc_score(y_test, svc_clf.predict_proba(X_test)[:, 1]))
                
                if 'From-Scratch SVM' in active_models:
                    scratch_clf = classical_mgr.fit_scratch_svm(X_train_sub, y_train_sub, best_params)
                    performance_log['From-Scratch SVM'][N]['f1'].append(f1_score(y_test, scratch_clf.predict(X_test)))
                    performance_log['From-Scratch SVM'][N]['auc'].append(roc_auc_score(y_test, scratch_clf.decision_function(X_test)))

            if 'XGBoost' in active_models:
                xgb_clf = classical_mgr.fit_xgboost(X_train_sub, y_train_sub)
                performance_log['XGBoost'][N]['f1'].append(f1_score(y_test, xgb_clf.predict(X_test)))
                performance_log['XGBoost'][N]['auc'].append(roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]))
                
            for q_name in ['Quantum-ZZ', 'Quantum-CPMap']:
                if q_name in active_models:
                    K_train_sub = master_kernels[q_name]['train'][np.ix_(sub_idx, sub_idx)]
                    K_test_sub = master_kernels[q_name]['test'][:, sub_idx]
                    
                    n_train, n_test = len(X_train_sub), len(X_test)
                    total_circs = (n_train * (n_train - 1)) // 2 + n_test * n_train
                    res_counts = quantum_managers[q_name].get_resource_counts()
                    
                    cost_log[q_name][N]['circuits'] += total_circs
                    cost_log[q_name][N]['single_qubit'] += total_circs * res_counts['single_qubit_gates']
                    cost_log[q_name][N]['cnot'] += total_circs * res_counts['cnot_gates']
                    
                    q_clf = quantum_managers[q_name].fit_quantum_svc(K_train_sub, y_train_sub)
                    performance_log[q_name][N]['f1'].append(f1_score(y_test, q_clf.predict(K_test_sub)))
                    performance_log[q_name][N]['auc'].append(roc_auc_score(y_test, q_clf.predict_proba(K_test_sub)[:, 1]))

        for model in active_models:
            f1_m, _ = calculate_95_ci(performance_log[model][N]['f1'])
            auc_m, _ = calculate_95_ci(performance_log[model][N]['auc'])
            if N not in summary_data[model]['N']:
                summary_data[model]['N'].append(N)
                summary_data[model]['F1'].append(f1_m)
                summary_data[model]['AUC'].append(auc_m)
            else:
                idx = summary_data[model]['N'].index(N)
                summary_data[model]['F1'][idx] = f1_m
                summary_data[model]['AUC'][idx] = auc_m
        gc.collect()

    session_elapsed = time.time() - start_time
    total_cumulative_seconds = save_persistent_compute_state(session_elapsed)
    cpu_cores = multiprocessing.cpu_count()
    cumulative_cpu_years = (total_cumulative_seconds * cpu_cores) / (365.25 * 24 * 3600)

    report_str = f"\n{'='*66}\nSAMPLE-EFFICIENCY METRICS REPORT FOR: {dataset_name}\n{'='*66}\n"
    report_str += f"[Compute Footprint Tracker]\n"
    report_str += f" -> Cumulative CPU Computing Years: {cumulative_cpu_years:.2e} years\n\n"
    
    for model in active_models:
        report_str += f">> Model Family: {model}\n"
        for N in config.N_LIST:
            f1_m, f1_ci = calculate_95_ci(performance_log[model][N]['f1'])
            auc_m, auc_ci = calculate_95_ci(performance_log[model][N]['auc'])
            report_str += f"   N={N:3d} | F1: {f1_m:.4f} ± {f1_ci:.4f} | AUC: {auc_m:.4f} ± {auc_ci:.4f}\n"

    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", f"summary_sample_eff_{clean_ds_name}.txt")
    with open(log_path, "w") as f:
        f.write(report_str)

    # --- Headless Plot Generation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    colors = {'RBF-SVC': '#1f77b4', 'From-Scratch SVM': '#aec7e8', 'XGBoost': '#d62728', 'Quantum-ZZ': '#9467bd', 'Quantum-CPMap': '#ff7f0e'}
    markers = {'RBF-SVC': 'o', 'From-Scratch SVM': 'x', 'XGBoost': 's', 'Quantum-ZZ': '^', 'Quantum-CPMap': 'D'}
    
    for model in summary_data.keys():
        if len(summary_data[model]['N']) == 0: continue
        axes[0].plot(summary_data[model]['N'], summary_data[model]['F1'], marker=markers.get(model, 'o'), color=colors.get(model, '#333'), label=model, linewidth=1.8)
        axes[1].plot(summary_data[model]['N'], summary_data[model]['AUC'], marker=markers.get(model, 'o'), color=colors.get(model, '#333'), label=model, linewidth=1.8)
        
    axes[0].set_title(f"{dataset_name}: F1-Score vs N", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Sample Budget Size (N)")
    axes[0].set_ylabel("F1 Score")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    axes[1].set_title(f"{dataset_name}: ROC-AUC vs N", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Sample Budget Size (N)")
    axes[1].set_ylabel("ROC-AUC")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    for ax in axes.flat:
        ax.set_xticks(config.N_LIST)
        ax.set_ylim([0.35, 1.05])
        
    plt.suptitle(f"Sample Efficiency: {dataset_name}", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    plot_path = os.path.join("results", f"sample_eff_plot_{clean_ds_name}.png")
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"[Incremental Plot Hub] Finalized plot saved to '{plot_path}'")

    return summary_data


def run_single_ablation(fmap_type, ent, bw, use_noise, X_train_preprocessed, X_test_preprocessed, y_train, y_test, clean_ds):
    """Isolated worker process for parallel structural ablations with memory cleanup."""
    from qiskit_algorithms.utils import algorithm_globals
    algorithm_globals.num_workers = 1

    config.ENTANGLEMENT = ent
    config.USE_NISQ_NOISE = use_noise
    
    raw_train_name = f"cache_ablation_{clean_ds}_map_{fmap_type}_ent_{ent}_bw_{bw}_noise_{use_noise}_seed{config.SEED}_train.npy"
    raw_test_name = f"cache_ablation_{clean_ds}_map_{fmap_type}_ent_{ent}_bw_{bw}_noise_{use_noise}_seed{config.SEED}_test.npy"
    
    train_cache_file = os.path.join(config.CACHE_DIR_KERNELS_ABLATION, raw_train_name)
    test_cache_file = os.path.join(config.CACHE_DIR_KERNELS_ABLATION, raw_test_name)
    
    X_train_scaled = X_train_preprocessed * bw
    X_test_scaled = X_test_preprocessed * bw
    
    mgr = ProductionQuantumKernelManager(map_type=fmap_type, use_noise=use_noise)
    res_specs = mgr.get_resource_counts()
    
    if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
        K_train = np.load(train_cache_file)
        K_test = np.load(test_cache_file)
    else:
        K_train = evaluate_kernel_in_chunks(mgr.kernel, X1=X_train_scaled, chunk_size=100)
        K_test = evaluate_kernel_in_chunks(mgr.kernel, X1=X_test_scaled, X2=X_train_scaled, chunk_size=100)
        np.save(train_cache_file, K_train)
        np.save(test_cache_file, K_test)
    
    alignment = calculate_kernel_target_alignment(K_train, y_train)
    spectral = mgr.calculate_spectral_diagnostics(K_train)
    
    svc = SVC(kernel='precomputed', C=1.0, probability=True, random_state=config.SEED)
    svc.fit(K_train, y_train)
    
    preds = svc.predict(K_test)
    probs = svc.predict_proba(K_test)[:, 1]
    
    result = {
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
    gc.collect()
    return result


def run_ablation_matrix_suite(dataset_name, X_all, y_all):
    clean_ds = "".join([c if c.isalnum() else "_" for c in dataset_name])
    start_time = time.time()
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    # --- MEMORY PROTECTION: Cap Real Data Pool for Ablation Suite ---
    max_train_pool = min(1000, len(train_idx))
    max_test_pool = min(500, len(test_idx))
    train_idx = train_idx[:max_train_pool]
    test_idx = test_idx[:max_test_pool]
    
    X_train_preprocessed, X_test_preprocessed = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    
    feature_maps = ['ZZ', 'CPMap']
    entanglements = ['linear', 'full']
    bandwidths = [0.5, 1.0, 2.0]
    noise_options = [False, True]
    
    # Filter out the invalid CPMap + full entanglement combination
    tasks = [
        (fmap, ent, bw, noise) 
        for fmap in feature_maps 
        for ent in entanglements 
        for bw in bandwidths 
        for noise in noise_options
        if not (fmap == 'CPMap' and ent == 'full')
    ]
    
    print(f"\n[Ablation Suite] Launching parallel architectural matrix for '{dataset_name}' ({len(tasks)} configs)...")
    
    ablation_records = Parallel(n_jobs=N_CORES, verbose=5, batch_size=2)(
        delayed(run_single_ablation)(*task, X_train_preprocessed, X_test_preprocessed, y_train, y_test, clean_ds) for task in tasks
    )
    
    session_elapsed = time.time() - start_time
    total_cumulative_seconds = save_persistent_compute_state(session_elapsed)
    cpu_cores = multiprocessing.cpu_count()
    cumulative_cpu_years = (total_cumulative_seconds * cpu_cores) / (365.25 * 24 * 3600)

    df_results = pd.DataFrame(ablation_records)
    os.makedirs("results", exist_ok=True)
    
    csv_out = os.path.join("results", f"ablation_report_{clean_ds}.csv")
    df_results.to_csv(csv_out, index=False)
    
    txt_out = os.path.join("results", f"ablation_summary_{clean_ds}.txt")
    with open(txt_out, "w") as f:
        f.write(f"=== ABLATION STUDY SUMMARY: {dataset_name} ===\n")
        f.write(f" -> Cumulative Compute Footprint: CPU Years = {cumulative_cpu_years:.2e}\n\n")
        f.write(df_results.to_string(index=False))

    # --- Headless Plot Generation ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for fmap in feature_maps:
        sub_df = df_results[df_results['Feature Map'] == fmap]
        g_align = sub_df.groupby('Bandwidth')['Target Alignment'].mean()
        g_f1 = sub_df.groupby('Bandwidth')['Test F1'].mean()
        axes[0].plot(g_align.index, g_align.values, marker='o', label=f"{fmap} Map", linewidth=2)
        axes[1].plot(g_f1.index, g_f1.values, marker='s', label=f"{fmap} Map", linewidth=2)
        
    axes[0].set_title(f"{dataset_name}: Alignment vs Bandwidth", fontsize=11, fontweight='bold')
    axes[0].set_xlabel("Bandwidth Multiplier ($\sigma$)")
    axes[0].set_ylabel("Kernel Target Alignment")
    axes[0].grid(True, linestyle='--', alpha=0.5)
    axes[0].legend()
    
    axes[1].set_title(f"{dataset_name}: Test F1 vs Bandwidth", fontsize=11, fontweight='bold')
    axes[1].set_xlabel("Bandwidth Multiplier ($\sigma$)")
    axes[1].set_ylabel("Test F1 Score")
    axes[1].grid(True, linestyle='--', alpha=0.5)
    axes[1].legend()
    
    plt.suptitle(f"Ablation Analytics: {dataset_name}", fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    plot_out = os.path.join("results", f"ablation_incremental_plot_{clean_ds}.png")
    fig.savefig(plot_out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"\n[Ablation Artifact Hub] Results recorded. Plot saved to: {plot_out}\n")
    return df_results


if __name__ == "__main__":
    print("[Real Data Engine] Preloading secondary dataset...")
    
    pipeline = QKEDataPipeline(seed=config.SEED)
    
    try:
        X_real_raw, y_real = pipeline.load_and_rebalance_real_data()
        X_real = pipeline.preprocess(X_real_raw)
        print(f"   [Data Loaded] Rebalanced real dataset successfully loaded. Shape: {X_real.shape}")
    except FileNotFoundError:
        print(f"\n[!] Error: '{config.CSV_PATH}' not found in runtime space. Please place the CSV file in the workspace.\n")
        sys.exit(1)

    name = "Rebalanced Real Data"
    clean_ds = "".join([c if c.isalnum() else "_" for c in name])
    
    sample_eff_txt = os.path.join("results", f"summary_sample_eff_{clean_ds}.txt")
    sample_eff_plot = os.path.join("results", f"sample_eff_plot_{clean_ds}.png")
    ablation_csv = os.path.join("results", f"ablation_report_{clean_ds}.csv")
    ablation_txt = os.path.join("results", f"ablation_summary_{clean_ds}.txt")
    ablation_plot = os.path.join("results", f"ablation_incremental_plot_{clean_ds}.png")
    
    if os.path.exists(sample_eff_txt) and os.path.exists(sample_eff_plot):
        print(f"\n[Skip] Sample-Efficiency suite for '{name}' already completed. Artifacts found.")
    else:
        algorithm_globals.num_workers = N_CORES
        run_sample_efficiency_suite(X_real, y_real, name)
    
    if os.path.exists(ablation_csv) and os.path.exists(ablation_txt) and os.path.exists(ablation_plot):
        print(f"\n[Skip] Ablation suite for '{name}' already completed. Artifacts found.")
    else:
        run_ablation_matrix_suite(name, X_real, y_real)

    print("\n[Execution Hub Complete] Real Data Pipeline successfully finalized.")