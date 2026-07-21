"""
Module: main.py
Optimized Edition: Employs persistent disk caching (.npy) in a centralized hub to bypass Aer calculations.
Hardware Optimized: Features dynamic worker allocation, memory-safe matrix chunking, and compute-year tracking.
"""
import os
import sys

# =====================================================================
# 1. RUNTIME RESOURCE ALLOCATION (Must execute before C++ library imports)
# =====================================================================
print("\n" + "="*60)
print("   Quantum Kernel Evaluation - Main Sample-Efficiency Suite   ")
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
os.environ['RAY_NUM_THREADS'] = safe_threads
os.environ['MKL_NUM_THREADS'] = safe_threads
os.environ['OPENBLAS_NUM_THREADS'] = safe_threads

import time
import gc
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
from tqdm import tqdm
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

import config
from data_pipeline import QKEDataPipeline
from models_classical import ClassicalBaselineManager
from models_quantum import ProductionQuantumKernelManager

warnings.filterwarnings('ignore')

# Unlock Qiskit's internal classical preprocessing
from qiskit_algorithms.utils import algorithm_globals
algorithm_globals.num_workers = N_CORES


def calculate_95_ci(data):
    mean = np.mean(data)
    n = len(data)
    if n <= 1: return mean, 0.0
    se = stats.sem(data)
    return mean, se * stats.t.ppf((1 + 0.95) / 2., n - 1)


def evaluate_kernel_in_chunks(qkernel, X1, X2=None, chunk_size=100):
    """
    Evaluates the quantum kernel matrix in memory-safe chunks to prevent 
    Qiskit circuit bloat and RAM exhaustion (SIGKILL).
    """
    symmetric = False
    if X2 is None:
        X2 = X1
        symmetric = True

    n1, n2 = len(X1), len(X2)
    K = np.zeros((n1, n2))

    for i in range(0, n1, chunk_size):
        end_i = min(i + chunk_size, n1)
        X1_chunk = X1[i:end_i]
        
        for j in range(0, n2, chunk_size):
            end_j = min(j + chunk_size, n2)
            
            # If symmetric, we only need to compute the upper triangle and mirror it
            if symmetric and j < i:
                continue
                
            X2_chunk = X2[j:end_j]
            K_block = qkernel.evaluate(x_vec=X1_chunk, y_vec=X2_chunk)
            K[i:end_i, j:end_j] = K_block
            
            if symmetric and i != j:
                K[j:end_j, i:end_i] = K_block.T
                
        gc.collect()

    return K


def run_central_experiment(X_all, y_all, dataset_name):
    print(f"\n==================================================================")
    print(f"STARTING OPTIMIZED RUN FOR: {dataset_name} (Total Samples: {len(X_all)})")
    print(f"==================================================================")
    
    start_time = time.time()
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
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
            
            res = mgr.get_resource_counts()
            print(f"   [{q_name} Layout] Physical Qubits: {res['qubits']} mapped to {config.QUBIT_BUDGET} Data Features.")
            
            raw_train_name = f"cache_{clean_ds_name}_{q_name}_seed{config.SEED}_train.npy"
            raw_test_name = f"cache_{clean_ds_name}_{q_name}_seed{config.SEED}_test.npy"
            
            train_cache_file = os.path.join(config.CACHE_DIR_KERNELS_MAIN, raw_train_name)
            test_cache_file = os.path.join(config.CACHE_DIR_KERNELS_MAIN, raw_test_name)
            
            if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
                print(f"   [Disk Cache Hub] Found matching arrays! Loading precomputed matrices instantly...")
                K_train_master = np.load(train_cache_file)
                K_test_master = np.load(test_cache_file)
            else:
                print(f"   [Simulation Engine] Executing chunked memory-safe sweep for {q_name}...")
                K_train_master = evaluate_kernel_in_chunks(mgr.kernel, X1=X_train_full, chunk_size=100)
                K_test_master = evaluate_kernel_in_chunks(mgr.kernel, X1=X_test, X2=X_train_full, chunk_size=100)
                
                np.save(train_cache_file, K_train_master)
                np.save(test_cache_file, K_test_master)
                print(f"   [Disk Cache Hub] Quantum matrices cached to disk successfully.")
            
            master_kernels[q_name] = {'train': K_train_master, 'test': K_test_master}

    spectral_log = {}
    for q_name in master_kernels:
        spectral_log[q_name] = quantum_managers[q_name].calculate_spectral_diagnostics(master_kernels[q_name]['train'])

    # --- SWEEP AND SAMPLE LOOPS ---
    for N in config.N_LIST:
        print(f"\n--- Running Sweep Size: N = {N} (Slicing Cached Matrices) ---")
        
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
                split_iterator.set_postfix_str("Tuning & Training: RBF-SVC")
                svc_clf, best_params = classical_mgr.fit_rbf_svc(X_train_sub, y_train_sub)
                performance_log['RBF-SVC'][N]['f1'].append(f1_score(y_test, svc_clf.predict(X_test)))
                performance_log['RBF-SVC'][N]['auc'].append(roc_auc_score(y_test, svc_clf.predict_proba(X_test)[:, 1]))
                
                if 'From-Scratch SVM' in active_models:
                    split_iterator.set_postfix_str("Training: From-Scratch SVM")
                    scratch_clf = classical_mgr.fit_scratch_svm(X_train_sub, y_train_sub, best_params)
                    performance_log['From-Scratch SVM'][N]['f1'].append(f1_score(y_test, scratch_clf.predict(X_test)))
                    performance_log['From-Scratch SVM'][N]['auc'].append(roc_auc_score(y_test, scratch_clf.decision_function(X_test)))

            if 'XGBoost' in active_models:
                split_iterator.set_postfix_str("Tuning & Training: XGBoost")
                xgb_clf = classical_mgr.fit_xgboost(X_train_sub, y_train_sub)
                performance_log['XGBoost'][N]['f1'].append(f1_score(y_test, xgb_clf.predict(X_test)))
                performance_log['XGBoost'][N]['auc'].append(roc_auc_score(y_test, xgb_clf.predict_proba(X_test)[:, 1]))
                
            for q_name in ['Quantum-ZZ', 'Quantum-CPMap']:
                if q_name in active_models:
                    split_iterator.set_postfix_str(f"Fitting: {q_name}")
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

    # --- COMPUTE USAGE TRACKING ---
    elapsed_seconds = time.time() - start_time
    cpu_cores = multiprocessing.cpu_count()
    seconds_per_year = 365.25 * 24 * 3600
    cpu_compute_years = (elapsed_seconds * cpu_cores) / seconds_per_year
    gpu_compute_years = (elapsed_seconds * 1) / seconds_per_year if any('Quantum' in m for m in active_models) else 0.0

    report_str = f"\n{'='*66}\nMETRICS REPORT FOR: {dataset_name}\n{'='*66}\n"
    report_str += f"[Compute Footprint Tracker]\n"
    report_str += f" -> Wall-Clock Time: {elapsed_seconds:.2f} seconds\n"
    report_str += f" -> CPU Compute Years Used: {cpu_compute_years:.2e} years ({cpu_cores} cores active)\n"
    report_str += f" -> GPU Compute Years Used: {gpu_compute_years:.2e} years\n\n"
    
    summary_data = {m: {'N': [], 'F1': [], 'AUC': []} for m in active_models}
    
    for model in active_models:
        report_str += f">> Model Family: {model}\n"
        for N in config.N_LIST:
            f1_m, f1_ci = calculate_95_ci(performance_log[model][N]['f1'])
            auc_m, auc_ci = calculate_95_ci(performance_log[model][N]['auc'])
            circs = cost_log[model][N]['circuits'] if model in cost_log else 0
            single_g = cost_log[model][N]['single_qubit'] if model in cost_log else 0
            cnot_g = cost_log[model][N]['cnot'] if model in cost_log else 0
            
            summary_data[model]['N'].append(N)
            summary_data[model]['F1'].append(f1_m)
            summary_data[model]['AUC'].append(auc_m)
            
            report_str += f"   N={N:3d} | F1: {f1_m:.4f} ± {f1_ci:.4f} | AUC: {auc_m:.4f} ± {auc_ci:.4f}\n"
            if model in ['Quantum-ZZ', 'Quantum-CPMap']:
                report_str += f"         [Accumulated Resource Cost] Circuits: {circs} | Single Gates: {single_g} | CNOTs: {cnot_g}\n"
            
        if model in ['Quantum-ZZ', 'Quantum-CPMap']:
            report_str += f"      [Global Spectral Diagnostics] Concentration Variance: {spectral_log[model]['variance']:.2e} | Condition No: {spectral_log[model]['condition_number']:.2e}\n"
            
    print(report_str)
    
    os.makedirs("results", exist_ok=True)
    log_path = os.path.join("results", f"summary_{clean_ds_name}.txt")
    with open(log_path, "w") as f:
        f.write(report_str)
        
    print(f"\n[Log Hub] Metrics report safely saved to '{log_path}'")
    return summary_data


if __name__ == "__main__":
    print("[Pipeline Setup] Preloading data distributions...")
    
    zz_mgr = ProductionQuantumKernelManager(map_type='ZZ')
    cpmap_mgr = ProductionQuantumKernelManager(map_type='CPMap')
    
    pipeline = QKEDataPipeline(seed=config.SEED)
    X_synth_raw, y_synth = pipeline.generate_synthetic_shells()
    X_synth = pipeline.preprocess(X_synth_raw)
    
    try:
        X_real_raw, y_real = pipeline.load_and_rebalance_real_data()
        X_real = pipeline.preprocess(X_real_raw)
    except FileNotFoundError:
        X_real, y_real = None, None

    X_qnative_ZZ, y_qnative_ZZ = pipeline.generate_quantum_native(X_synth, zz_mgr.kernel, map_name="ZZ")
    X_qnative_CPMap, y_qnative_CPMap = pipeline.generate_quantum_native(X_synth, cpmap_mgr.kernel, map_name="CPMap")
    
    datasets_to_run = [
        ("Primary Synthetic (Shells)", X_synth, y_synth),
        ("Positive Control (ZZ-Generated)", X_qnative_ZZ, y_qnative_ZZ),
        ("Positive Control (CPMap-Generated)", X_qnative_CPMap, y_qnative_CPMap),
    ]
    if X_real is not None:
        datasets_to_run.append(("Rebalanced Real Data", X_real, y_real))
    
    all_dataset_plots = {}
    for name, X_d, y_d in datasets_to_run:
        if X_d is not None:
            clean_ds = "".join([c if c.isalnum() else "_" for c in name])
            sample_eff_txt = os.path.join("results", f"summary_{clean_ds}.txt")
            
            if os.path.exists(sample_eff_txt):
                print(f"\n[Skip] Main Sample-Efficiency suite for '{name}' already completed. Artifacts found.")
                # We still need to load data into the plotting dictionary if we skip execution
                # For simplicity, bypassing plot aggregation if skipped, as plots are saved locally.
            else:
                all_dataset_plots[name] = run_central_experiment(X_d, y_d, name)
            
    # --- VISUALIZATION PLOT GENERATION (Only runs if new data was processed) ---
    if all_dataset_plots:
        fig, axes = plt.subplots(len(all_dataset_plots), 2, figsize=(14, 5 * len(all_dataset_plots)))
        if len(all_dataset_plots) == 1:
            axes = np.expand_dims(axes, axis=0)
            
        colors = {'RBF-SVC': '#1f77b4', 'From-Scratch SVM': '#aec7e8', 'XGBoost': '#d62728', 'Quantum-ZZ': '#9467bd', 'Quantum-CPMap': '#ff7f0e'}
        markers = {'RBF-SVC': 'o', 'From-Scratch SVM': 'x', 'XGBoost': 's', 'Quantum-ZZ': '^', 'Quantum-CPMap': 'D'}
        
        for row_idx, (d_name, plot_data) in enumerate(all_dataset_plots.items()):
            for model in plot_data.keys():
                if len(plot_data[model]['N']) == 0: continue
                axes[row_idx, 0].plot(plot_data[model]['N'], plot_data[model]['F1'], marker=markers[model], color=colors[model], label=model, linewidth=1.8)
                axes[row_idx, 1].plot(plot_data[model]['N'], plot_data[model]['AUC'], marker=markers[model], color=colors[model], label=model, linewidth=1.8)
                
            axes[row_idx, 0].set_title(f"{d_name}: F1-Score vs N", fontsize=11, fontweight='bold')
            axes[row_idx, 0].set_ylabel("F1 Score")
            axes[row_idx, 0].grid(True, linestyle='--', alpha=0.5)
            axes[row_idx, 0].legend()
            
            axes[row_idx, 1].set_title(f"{d_name}: ROC-AUC vs N", fontsize=11, fontweight='bold')
            axes[row_idx, 1].set_ylabel("ROC-AUC")
            axes[row_idx, 1].grid(True, linestyle='--', alpha=0.5)
            axes[row_idx, 1].legend()

        for ax in axes.flat:
            ax.set_xlabel("Sample Budget Size (N)")
            ax.set_xticks(config.N_LIST)
            ax.set_ylim([0.35, 1.05])
            
        plt.suptitle("Unified Model Sample-Efficiency Curves Comparison", fontsize=14, fontweight='bold', y=0.98)
        plt.tight_layout()
        
        os.makedirs("results", exist_ok=True)
        save_path = os.path.join("results", "8_qubit_Task1_run.png")
        fig.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"\n[Plot Hub] Execution complete. Figure safely saved to '{save_path}'")