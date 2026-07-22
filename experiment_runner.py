"""
Module: experiment_runner.py
Description: The singular, unified execution engine for both Sample Efficiency and 
             Ablation matrix suites. Supports dynamic memory pooling and targeted execution.
"""
import os
import time
import gc
import multiprocessing
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.svm import SVC
from qiskit_algorithms.utils import algorithm_globals

import config
import utils
from models_classical import ClassicalBaselineManager
from models_quantum import ProductionQuantumKernelManager

def run_sample_efficiency_suite(X_all, y_all, dataset_name, is_real_data=False, n_cores=4):
    algorithm_globals.num_workers = n_cores
    clean_ds_name = "".join([c if c.isalnum() else "_" for c in dataset_name])
    
    # Artifact Bypass Check
    sample_eff_txt = os.path.join("results", f"summary_sample_eff_{clean_ds_name}.txt")
    if os.path.exists(sample_eff_txt):
        print(f"\n[Skip] Sample-Efficiency suite for '{dataset_name}' already completed. Artifacts found.")
        return {}
        
    print(f"\n==================================================================")
    print(f"STARTING SAMPLE-EFFICIENCY SUITE FOR: {dataset_name}")
    print(f"==================================================================")
    
    start_time = time.time()
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    # Memory Protection Layer
    chunk_sz = 200
    if is_real_data:
        train_idx = train_idx[:min(1000, len(train_idx))]
        test_idx = test_idx[:min(500, len(test_idx))]
        chunk_sz = 50
        print(f"   [Real Data Limiter] Bounded pool to {len(train_idx)} train / {len(test_idx)} test points.")
        
    X_train_full, X_test = X_all[train_idx], X_all[test_idx]
    y_train_full, y_test = y_all[train_idx], y_all[test_idx]
    
    active_models = [m for m, active in config.RUN_MODELS.items() if active]
    performance_log = {m: {N: {'f1': [], 'auc': []} for N in config.N_LIST} for m in active_models}
    cost_log = {m: {N: {'circuits': 0, 'single_qubit': 0, 'cnot': 0} for N in config.N_LIST} for m in active_models}
    
    rng = np.random.default_rng(config.SEED)
    classical_mgr = ClassicalBaselineManager(seed=config.SEED)
    master_kernels, quantum_managers = {}, {}
    
    # Isolate Matrix Pre-computation entirely to Active Models
    for q_name, map_type in [('Quantum-ZZ', 'ZZ'), ('Quantum-CPMap', 'CPMap')]:
        if q_name in active_models:
            mgr = ProductionQuantumKernelManager(map_type=map_type)
            quantum_managers[q_name] = mgr
            
            raw_train_name = f"cache_{q_name}_seed{config.SEED}_train.npy"
            raw_test_name = f"cache_{q_name}_seed{config.SEED}_test.npy"
            
            cache_dir = utils.get_cache_dir(dataset_name, "kernels_main")
            train_cache_file = os.path.join(cache_dir, raw_train_name)
            test_cache_file = os.path.join(cache_dir, raw_test_name)
            
            if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
                print(f"   [Cache Hit] Loading {q_name} matrices from disk...")
                K_train_master = np.load(train_cache_file)
                K_test_master = np.load(test_cache_file)
            else:
                print(f"   [Simulation] Computing {q_name} kernel matrices...")
                K_train_master = utils.evaluate_kernel_in_chunks(mgr.kernel, X1=X_train_full, chunk_size=chunk_sz, verbose=is_real_data)
                K_test_master = utils.evaluate_kernel_in_chunks(mgr.kernel, X1=X_test, X2=X_train_full, chunk_size=chunk_sz, verbose=is_real_data)
                np.save(train_cache_file, K_train_master)
                np.save(test_cache_file, K_test_master)
                
            master_kernels[q_name] = {'train': K_train_master, 'test': K_test_master}

    spectral_log = {q_name: quantum_managers[q_name].calculate_spectral_diagnostics(master_kernels[q_name]['train']) for q_name in master_kernels}
    summary_data = {m: {'N': [], 'F1': [], 'AUC': []} for m in active_models}

    # Main Sweep Execution
    for N in config.N_LIST:
        print(f"\n--- Running Sweep Size: N = {N} ---")
        split_iterator = tqdm(range(config.N_SPLITS), desc=f"Evaluating N={N}", leave=True)
        
        for split in split_iterator:
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            
            sampled_pos = rng.choice(pos_idx, size=N // 2, replace=(N // 2 > len(pos_idx)))
            sampled_neg = rng.choice(neg_idx, size=N // 2, replace=(N // 2 > len(neg_idx)))
            sub_idx = rng.permutation(np.concatenate([sampled_pos, sampled_neg]))
            
            X_train_sub, y_train_sub = X_train_full[sub_idx], y_train_full[sub_idx]
            
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
            f1_m, _ = utils.calculate_95_ci(performance_log[model][N]['f1'])
            auc_m, _ = utils.calculate_95_ci(performance_log[model][N]['auc'])
            if N not in summary_data[model]['N']:
                summary_data[model]['N'].append(N)
                summary_data[model]['F1'].append(f1_m)
                summary_data[model]['AUC'].append(auc_m)
            else:
                idx = summary_data[model]['N'].index(N)
                summary_data[model]['F1'][idx] = f1_m
                summary_data[model]['AUC'][idx] = auc_m
        gc.collect()

    # Hardware compute logging and artifact saving
    total_cumulative_seconds = utils.save_persistent_compute_state(dataset_name, time.time() - start_time)
    cumulative_cpu_years = (total_cumulative_seconds * multiprocessing.cpu_count()) / (365.25 * 24 * 3600)

    report_str = f"\n{'='*66}\nSAMPLE-EFFICIENCY METRICS REPORT FOR: {dataset_name}\n{'='*66}\n"
    report_str += f"[Compute Footprint Tracker]\n -> Cumulative CPU Computing Years: {cumulative_cpu_years:.2e} years\n\n"
    
    for model in active_models:
        report_str += f">> Model Family: {model}\n"
        for N in config.N_LIST:
            f1_m, f1_ci = utils.calculate_95_ci(performance_log[model][N]['f1'])
            auc_m, auc_ci = utils.calculate_95_ci(performance_log[model][N]['auc'])
            report_str += f"   N={N:3d} | F1: {f1_m:.4f} ± {f1_ci:.4f} | AUC: {auc_m:.4f} ± {auc_ci:.4f}\n"

    with open(sample_eff_txt, "w") as f:
        f.write(report_str)
        
    print(f"\n[Artifact Hub] Core suite finalized. Report saved to: {sample_eff_txt}\n")
    return summary_data


def run_single_ablation(fmap_type, ent, bw, use_noise, X_train_p, X_test_p, y_train, y_test, ds_name, chunk_sz):
    """Isolated memory-protected worker process for structural ablations with dimension-validated caching."""
    # Force underlying linear algebra libraries in subprocesses to single-thread 
    # so Joblib processes can cleanly distribute across multiple CPU cores without locking.
    os.environ['OMP_NUM_THREADS'] = '1'
    os.environ['MKL_NUM_THREADS'] = '1'
    os.environ['OPENBLAS_NUM_THREADS'] = '1'
    
    algorithm_globals.num_workers = 1
    config.ENTANGLEMENT = ent
    config.USE_NISQ_NOISE = use_noise
    
    cache_dir = utils.get_cache_dir(ds_name, "kernels_ablation", ablation_params=(fmap_type, ent, bw, use_noise))
    train_cache = os.path.join(cache_dir, "train_kernel.npy")
    test_cache = os.path.join(cache_dir, "test_kernel.npy")
    
    X_train_scaled, X_test_scaled = X_train_p * bw, X_test_p * bw
    mgr = ProductionQuantumKernelManager(map_type=fmap_type, use_noise=use_noise)
    
    # Check cache existence and validate matrix dimensions against current y_train / y_test
    cache_valid = False
    if os.path.exists(train_cache) and os.path.exists(test_cache):
        K_train = np.load(train_cache)
        K_test = np.load(test_cache)
        if K_train.shape[0] == len(y_train) and K_test.shape[0] == len(y_test) and K_test.shape[1] == len(y_train):
            cache_valid = True

    if not cache_valid:
        K_train = utils.evaluate_kernel_in_chunks(mgr.kernel, X1=X_train_scaled, chunk_size=chunk_sz)
        K_test = utils.evaluate_kernel_in_chunks(mgr.kernel, X1=X_test_scaled, X2=X_train_scaled, chunk_size=chunk_sz)
        np.save(train_cache, K_train)
        np.save(test_cache, K_test)
    
    spectral = mgr.calculate_spectral_diagnostics(K_train)
    svc = SVC(kernel='precomputed', C=1.0, probability=True, random_state=config.SEED).fit(K_train, y_train)
    
    result = {
        'Feature Map': fmap_type, 'Entanglement': ent, 'Bandwidth': bw, 'Noisy Backend': use_noise,
        'Qubits': mgr.get_resource_counts()['qubits'], 'CNOT Count': mgr.get_resource_counts()['cnot_gates'],
        'Target Alignment': utils.calculate_kernel_target_alignment(K_train, y_train),
        'Spectral Variance': spectral['variance'], 'Condition Number': spectral['condition_number'],
        'Test F1': f1_score(y_test, svc.predict(K_test)), 'Test AUC': roc_auc_score(y_test, svc.predict_proba(K_test)[:, 1])
    }
    gc.collect()
    return result


def run_ablation_matrix_suite(X_all, y_all, dataset_name, is_real_data=False, n_cores=4):
    clean_ds = "".join([c if c.isalnum() else "_" for c in dataset_name])
    ablation_csv = os.path.join("results", f"ablation_report_{clean_ds}.csv")
    
    if os.path.exists(ablation_csv):
        print(f"\n[Skip] Ablation suite for '{dataset_name}' already completed.")
        return pd.DataFrame()
        
    start_time = time.time()
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    # --- STRICT TRAINING SAMPLE LIMITER (Max 500 samples) ---
    max_train_pool = min(500, len(train_idx))
    max_test_pool = min(250, len(test_idx))
    train_idx = train_idx[:max_train_pool]
    test_idx = test_idx[:max_test_pool]
    print(f"   [Ablation Pool Limiter] Bounded training pool to exactly {len(train_idx)} samples.")

    chunk_sz = 200
    if is_real_data:
        chunk_sz = 50

    X_train_p, X_test_p = X_all[train_idx], X_all[test_idx]
    y_train, y_test = y_all[train_idx], y_all[test_idx]
    
    # Generate structural combinations while filtering invalid architectural states
    tasks = [
        (fmap, ent, bw, noise) 
        for fmap in ['ZZ', 'CPMap'] 
        for ent in ['linear', 'full'] 
        for bw in [0.5, 1.0, 2.0] 
        for noise in [False, True]
        if not (fmap == 'CPMap' and ent == 'full')
    ]
    
    # Enforce maximum 12 core global limit safely
    effective_cores = min(n_cores, 12)
    
    print(f"\n[Ablation Suite] Launching parallel architectural matrix for '{dataset_name}' using {effective_cores} cores...")
    
    # Restored explicit batch_size=2 alongside loky backend for balanced parallel scheduling
    ablation_records = Parallel(n_jobs=effective_cores, backend='loky', batch_size=2, verbose=5)(
        delayed(run_single_ablation)(*task, X_train_p, X_test_p, y_train, y_test, dataset_name, chunk_sz) for task in tasks
    )
    
    total_cumulative_seconds = utils.save_persistent_compute_state(dataset_name, time.time() - start_time)
    df_results = pd.DataFrame(ablation_records)
    
    df_results.to_csv(ablation_csv, index=False)
    with open(os.path.join("results", f"ablation_summary_{clean_ds}.txt"), "w") as f:
        f.write(f"=== ABLATION STUDY SUMMARY: {dataset_name} ===\n")
        f.write(f" -> Cumulative Compute Footprint: CPU Years = {(total_cumulative_seconds * multiprocessing.cpu_count()) / (365.25 * 24 * 3600):.2e}\n\n")
        f.write(df_results.to_string(index=False))

    print(f"\n[Ablation Artifact Hub] Results recorded. CSV saved to: {ablation_csv}\n")
    return df_results