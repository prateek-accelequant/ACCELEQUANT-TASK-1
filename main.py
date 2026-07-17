"""
Module: main.py
Optimized Edition: Employs persistent disk caching (.npy) to bypass Aer calculations on duplicate runs.
"""
import numpy as np
import matplotlib.pyplot as plt
import os
import warnings
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

import config
from data_pipeline import QKEDataPipeline, get_all_data
from models_classical import ClassicalBaselineManager
from models_quantum import ProductionQuantumKernelManager
from tqdm import tqdm

warnings.filterwarnings('ignore')

def calculate_95_ci(data):
    mean = np.mean(data)
    n = len(data)
    if n <= 1: return mean, 0.0
    se = stats.sem(data)
    return mean, se * stats.t.ppf((1 + 0.95) / 2., n - 1)

def run_central_experiment(X_all, y_all, dataset_name):
    print(f"\n==================================================================")
    print(f"STARTING OPTIMIZED RUN FOR: {dataset_name} (Total Samples: {len(X_all)})")
    print(f"==================================================================")
    
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    X_train_full, X_test = X_all[train_idx], X_all[test_idx]
    y_train_full, y_test = y_all[train_idx], y_all[test_idx]
    
    active_models = [m for m, active in config.RUN_MODELS.items() if active]
    performance_log = {m: {N: {'f1': [], 'auc': []} for N in config.N_LIST} for m in active_models}
    cost_log = {m: {N: {'circuits': 0, 'single_qubit': 0, 'cnot': 0} for N in config.N_LIST} for m in active_models}
    
    rng = np.random.default_rng(config.SEED)
    classical_mgr = ClassicalBaselineManager(seed=config.SEED)
    
    # --- PERSISTENT DISK CACHING OPTIMIZATION ENGINE ---
    master_kernels = {}
    quantum_managers = {}
    
    clean_ds_name = "".join([c if c.isalnum() else "_" for c in dataset_name])
    
    for q_name, map_type in [('Quantum-ZZ', 'ZZ'), ('Quantum-CPMap', 'CPMap')]:
        if config.RUN_MODELS[q_name]:
            mgr = ProductionQuantumKernelManager(map_type=map_type)
            quantum_managers[q_name] = mgr
            
            res = mgr.get_resource_counts()
            print(f"   [{q_name} Layout] Physical Qubits: {res['qubits']} mapped to {config.QUBIT_BUDGET} Data Features.")
            
            train_cache_file = f"cache_{clean_ds_name}_{q_name}_seed{config.SEED}_train.npy"
            test_cache_file = f"cache_{clean_ds_name}_{q_name}_seed{config.SEED}_test.npy"
            
            if os.path.exists(train_cache_file) and os.path.exists(test_cache_file):
                print(f"   [Disk Cache Hub] Found matching arrays on disk! Loading precomputed matrices instantly...")
                K_train_master = np.load(train_cache_file)
                K_test_master = np.load(test_cache_file)
            else:
                print(f"   [Simulation Engine] Cache not found. Executing quantum hardware simulator sweeps...")
                K_train_master = mgr.kernel.evaluate(x_vec=X_train_full)
                K_test_master = mgr.kernel.evaluate(x_vec=X_test, y_vec=X_train_full)
                
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
        
        # Wrap the split loop in a tqdm progress bar
        split_iterator = tqdm(range(config.N_SPLITS), desc=f"Evaluating N={N}", leave=True)
        
        for split in split_iterator:
            pos_idx = np.where(y_train_full == 1)[0]
            neg_idx = np.where(y_train_full == 0)[0]
            
            sampled_pos = rng.choice(pos_idx, size=N // 2, replace=False)
            sampled_neg = rng.choice(neg_idx, size=N // 2, replace=False)
            sub_idx = rng.permutation(np.concatenate([sampled_pos, sampled_neg]))
            
            X_train_sub = X_train_full[sub_idx]
            y_train_sub = y_train_full[sub_idx]
            
            # --- Classical Baselines ---
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
                
            # --- Quantum Baselines ---
            for q_name in ['Quantum-ZZ', 'Quantum-CPMap']:
                if q_name in active_models:
                    split_iterator.set_postfix_str(f"Fitting: {q_name}")
                    K_train_sub = master_kernels[q_name]['train'][np.ix_(sub_idx, sub_idx)]
                    K_test_sub = master_kernels[q_name]['test'][:, sub_idx]
                    
                    # Exact unique circuit evaluations
                    n_train = len(X_train_sub)
                    n_test = len(X_test)
                    unique_train_circs = (n_train * (n_train - 1)) // 2
                    unique_test_circs = n_test * n_train
                    total_circs = unique_train_circs + unique_test_circs
                    
                    # Extract exact native circuit costs
                    res_counts = quantum_managers[q_name].get_resource_counts()
                    cost_log[q_name][N]['circuits'] += total_circs
                    cost_log[q_name][N]['single_qubit'] += total_circs * res_counts['single_qubit_gates']
                    cost_log[q_name][N]['cnot'] += total_circs * res_counts['cnot_gates']
                    
                    q_clf = quantum_managers[q_name].fit_quantum_svc(K_train_sub, y_train_sub)
                    performance_log[q_name][N]['f1'].append(f1_score(y_test, q_clf.predict(K_test_sub)))
                    performance_log[q_name][N]['auc'].append(roc_auc_score(y_test, q_clf.predict_proba(K_test_sub)[:, 1]))

    # --- TEXT FILE REPORT GENERATOR ---
    report_str = f"\n{'='*66}\nMETRICS REPORT FOR: {dataset_name}\n{'='*66}\n"
    summary_data = {m: {'N': [], 'F1': [], 'AUC': []} for m in active_models}
    
    for model in active_models:
        report_str += f"\n>> Model Family: {model}\n"
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

    X_qnative_ZZ, y_qnative_ZZ = pipeline.generate_quantum_native(X_synth, zz_mgr.kernel)
    X_qnative_CPMap, y_qnative_CPMap = pipeline.generate_quantum_native(X_synth, cpmap_mgr.kernel)
    
    datasets_to_run = [
        ("Primary Synthetic (Shells)", X_synth, y_synth),
        ("Positive Control (ZZ-Generated)", X_qnative_ZZ, y_qnative_ZZ),
        ("Positive Control (CPMap-Generated)", X_qnative_CPMap, y_qnative_CPMap),
        ("Rebalanced Real Data", X_real, y_real)
    ]
    
    all_dataset_plots = {}
    for name, X_d, y_d in datasets_to_run:
        if X_d is not None:
            all_dataset_plots[name] = run_central_experiment(X_d, y_d, name)
            
    # --- VISUALIZATION PLOT GENERATION ---
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
    
    plt.show()