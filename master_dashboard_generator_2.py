"""
Script: master_dashboard_generator.py
Description: Master reporting and evaluation orchestrator. 
             Generates publication-ready plots for:
               - F1 Score vs N (with 95% CI bands)
               - Accuracy vs N (with 95% CI bands)
               - ROC-AUC Score vs N (with 95% CI bands)
               - Gram Matrix Condition Number (kappa) vs N
               - Kernel Target Alignment vs N
               - Samples to Reach Target F1 Score (Optimal N)
               - Hardware Resource Scaling & Circuit Evaluations
               - Spectral Decomposition & Off-Diagonal Variance
               - Ablation Dashboards
             Includes dynamic shape-binding and NaN-safe exception handling 
             for robust dataset and Gram matrix cache evaluation.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import f1_score, accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from qiskit import transpile
from qiskit.circuit.library import ZZFeatureMap
from tqdm import tqdm

# Enable Matplotlib mathtext for LaTeX formatting
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

try:
    import config
    import utils
    from models_quantum import ProductionQuantumKernelManager
except ImportError:
    class MockConfig:
        CACHE_DIR_BASE = "QKE_Cache"
        CACHE_DIR_DATASETS = os.path.join("QKE_Cache", "_datasets")
        QUBIT_BUDGET = 8
        N_LIST = [50, 100, 200, 500]
        SEED = 42
        OUTER_SPLITS = 5
    config = MockConfig()
    class MockUtils:
        @staticmethod
        def calculate_95_ci(data):
            valid_data = [x for x in data if not np.isnan(x)]
            n = len(valid_data)
            if n == 0: return np.nan, np.nan
            mean = np.mean(valid_data)
            if n <= 1: return mean, 0.0
            se = stats.sem(valid_data)
            return mean, se * stats.t.ppf((1 + 0.95) / 2., n - 1)
        @staticmethod
        def calculate_kernel_target_alignment(K_train, y_train):
            y_mapped = np.where(y_train == 0, -1, 1).reshape(-1, 1)
            K_target = y_mapped @ y_mapped.T
            norm_K = np.linalg.norm(K_train, 'fro')
            norm_T = np.linalg.norm(K_target, 'fro')
            return np.sum(K_train * K_target) / (norm_K * norm_T) if norm_K * norm_T > 0 else 0.0
    utils = MockUtils()

warnings.filterwarnings('ignore')
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

MODEL_COLORS = {
    'Classical (RBF)': '#1f77b4',
    'Quantum-ZZ (Linear)': '#9467bd',
    'Quantum-ZZ (Full)': '#d62728',
    'Quantum-CPMap': '#ff7f0e'
}


# =====================================================================
# 1. HARDWARE RESOURCE SCALING & DYNAMIC BASIS GATE INSPECTION
# =====================================================================
def inspect_live_circuit_gates(map_type='CPMap', entanglement='linear', reps=1):
    """
    Instantiates live Qiskit feature map objects and unrolls them down 
    to fundamental basis gates ['cx', 'rz', 'ry', 'h'] via Qiskit transpile.
    """
    if map_type == 'ZZ':
        fmap = ZZFeatureMap(
            feature_dimension=config.QUBIT_BUDGET, 
            reps=reps, 
            entanglement=entanglement
        )
        qubits = config.QUBIT_BUDGET
    else: 
        mgr = ProductionQuantumKernelManager(map_type='CPMap')
        fmap = mgr.feature_map
        qubits = mgr.num_qubits

    overlap_circuit = fmap.compose(fmap.inverse())
    unrolled_circ = transpile(overlap_circuit, basis_gates=['cx', 'rz', 'ry', 'h'], optimization_level=0)
    ops = unrolled_circ.count_ops()

    cnot_count = ops.get('cx', 0)
    sq_gates = ops.get('h', 0) + ops.get('rz', 0) + ops.get('ry', 0) + ops.get('rx', 0)

    return {
        'Qubits': qubits,
        'CNOT_per_circuit': cnot_count,
        'SQ_gates_per_circuit': sq_gates,
        'Total_gates_per_circuit': cnot_count + sq_gates
    }


def generate_resource_table():
    """Outputs hardware resource CSVs and a 4-panel visual dashboard."""
    print("\n[*] Dynamically inspecting live Qiskit circuit objects...")
    
    configs_to_inspect = [
        ('Quantum-ZZ (Linear)', 'ZZ', 'linear'),
        ('Quantum-ZZ (Full)', 'ZZ', 'full'),
        ('Quantum-CPMap', 'CPMap', 'linear')
    ]
    
    records = []
    for label, m_type, ent in configs_to_inspect:
        info = inspect_live_circuit_gates(map_type=m_type, entanglement=ent)
        records.append({
            'Configuration': label,
            'Qubits': info['Qubits'],
            'CNOTs / Overlap Circuit (U^dagger U)': info['CNOT_per_circuit'],
            '1-Qubit Gates / Overlap Circuit': info['SQ_gates_per_circuit'],
            'Total Operations / Overlap Circuit': info['Total_gates_per_circuit']
        })
        
    df_hw = pd.DataFrame(records)
    csv_path = os.path.join(RESULTS_DIR, "dynamic_circuit_gate_counts.csv")
    df_hw.to_csv(csv_path, index=False)
    
    print(r"\n--- DYNAMICALLY EXTRACTED GATE COUNTS (PER OVERLAP CIRCUIT U^\dagger U) ---")
    print(df_hw.to_string(index=False))
    
    max_n = max(config.N_LIST)
    n_sweep = np.arange(10, max_n + 1, 10)
    n_test = 200
    
    scaling_rows = []
    for n_train in n_sweep:
        pair_circuits = int((n_train * (n_train - 1)) // 2 + (n_test * n_train))
        for label, m_type, ent in configs_to_inspect:
            info = inspect_live_circuit_gates(map_type=m_type, entanglement=ent)
            scaling_rows.append({
                'N_Train_Samples': n_train,
                'Configuration': label,
                'Pair_Circuits_Evaluated': pair_circuits,
                'CNOTs_Per_Circuit': info['CNOT_per_circuit'],
                'Cumulative_CNOT_Operations': pair_circuits * info['CNOT_per_circuit'],
                'Cumulative_Total_Operations': pair_circuits * info['Total_gates_per_circuit']
            })
            
    df_scaling = pd.DataFrame(scaling_rows)
    df_scaling.to_csv(os.path.join(RESULTS_DIR, "resource_scaling_by_N.csv"), index=False)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    
    ax1 = axes[0, 0]
    sub_c = df_scaling[df_scaling['Configuration'] == 'Quantum-CPMap']
    ax1.plot(sub_c['N_Train_Samples'], sub_c['Pair_Circuits_Evaluated'], color='#333333', linewidth=2.5)
    ax1.set_title(r"Gram Matrix Circuit Scaling ($\mathcal{O}(N^2)$ Pair Evaluations)", fontweight='bold', fontsize=12)
    ax1.set_xlabel(r"Training Sample Size ($N$)")
    ax1.set_ylabel(r"Total Overlap Circuits ($N_{\mathrm{circ}}$)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    ax2 = axes[0, 1]
    for label in MODEL_COLORS:
        if label == 'Classical (RBF)': continue
        sub = df_scaling[df_scaling['Configuration'] == label]
        if not sub.empty:
            ax2.plot(sub['N_Train_Samples'], sub['Cumulative_CNOT_Operations'], label=label, color=MODEL_COLORS[label], linewidth=2.2)
    ax2.set_title(r"Cumulative CNOT Operations to Convergence", fontweight='bold', fontsize=12)
    ax2.set_xlabel(r"Training Sample Size ($N$)")
    ax2.set_ylabel(r"Cumulative CNOT Count")
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    ax3 = axes[1, 0]
    bars = ax3.bar(df_hw['Configuration'], df_hw['CNOTs / Overlap Circuit (U^dagger U)'], 
                   color=[MODEL_COLORS.get(c, '#333') for c in df_hw['Configuration']], alpha=0.85, width=0.4, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{int(yval)} CNOTs", ha='center', va='bottom', fontweight='bold')
    ax3.set_title(r"Per-Circuit Hardware Overhead ($U^\dagger U$)", fontweight='bold', fontsize=12)
    ax3.set_ylabel(r"CNOT Count per Overlap Circuit")
    plt.setp(ax3.get_xticklabels(), rotation=15, ha='right', fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    ax4 = axes[1, 1]
    for label in MODEL_COLORS:
        if label == 'Classical (RBF)': continue
        sub = df_scaling[df_scaling['Configuration'] == label]
        if not sub.empty:
            ax4.plot(sub['N_Train_Samples'], sub['Cumulative_Total_Operations'], label=label, color=MODEL_COLORS[label], linewidth=2.2)
    ax4.set_title(r"Total Quantum Operations (1-Qubit + CNOTs)", fontweight='bold', fontsize=12)
    ax4.set_xlabel(r"Training Sample Size ($N$)")
    ax4.set_ylabel(r"Total Operations Count")
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plot_resource_scaling_dashboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [Saved] Hardware Resource Dashboard -> '{plot_path}'")


def plot_circuit_evaluations_vs_N(max_n=500, n_test=200, step=10):
    """Generates dedicated O(N^2) circuit evaluations plot."""
    print("\n[*] Generating Circuit Evaluations vs. Training Samples (N) Plot...")

    n_train_vec = np.arange(10, max_n + 1, step)
    train_gram_circuits = (n_train_vec * (n_train_vec - 1)) // 2
    test_gram_circuits = n_test * n_train_vec
    total_circuits = train_gram_circuits + test_gram_circuits

    df_circuits = pd.DataFrame({
        'N_Train_Samples': n_train_vec,
        'N_Test_Samples': n_test,
        'Train_Gram_Matrix_Circuits': train_gram_circuits,
        'Test_Gram_Matrix_Circuits': test_gram_circuits,
        'Total_Circuit_Evaluations': total_circuits
    })
    
    csv_path = os.path.join(RESULTS_DIR, "circuit_evaluations_vs_N.csv")
    df_circuits.to_csv(csv_path, index=False)

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(n_train_vec, total_circuits, label=r"Total Circuit Evaluations ($N_{\mathrm{circ}}$)", 
            color='#1f77b4', linewidth=2.8, zorder=3)
    ax.plot(n_train_vec, train_gram_circuits, label=r"Training Gram Matrix ($\frac{N(N-1)}{2}$)", 
            color='#9467bd', linestyle='--', linewidth=1.8, alpha=0.85)
    ax.plot(n_train_vec, test_gram_circuits, label=r"Test Gram Matrix ($N_{\mathrm{test}} \times N$)", 
            color='#2ca02c', linestyle=':', linewidth=1.8, alpha=0.85)

    benchmark_n = [50, 100, 200, 500]
    for n_val in benchmark_n:
        idx = np.where(n_train_vec == n_val)[0][0]
        tot_val = total_circuits[idx]
        
        ax.scatter([n_val], [tot_val], color='#d62728', s=60, zorder=5, edgecolor='black')
        ax.annotate(
            f"$N={n_val}$\n{tot_val:,} circ", 
            xy=(n_val, tot_val), 
            xytext=(-35, 18 if n_val < 500 else -35),
            textcoords="offset points", 
            fontsize=9,
            fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#d62728", alpha=0.9),
            arrowprops=dict(arrowstyle="->", color="#d62728", lw=1.2)
        )

    ax.set_title(
        r"Quantum Kernel Circuit Evaluation Scaling $\mathcal{O}(N^2)$" + "\n" +
        rf"(Gram Matrix Pair Evaluations for $N_{{\mathrm{{test}}}} = {n_test}$)", 
        fontweight='bold', fontsize=13, pad=12
    )
    ax.set_xlabel(r"Number of Training Samples ($N$)", fontsize=11)
    ax.set_ylabel(r"Total Circuit Evaluations ($N_{\mathrm{circ}}$)", fontsize=11)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.legend(loc='upper left', fontsize=10, frameon=True)
    ax.get_yaxis().set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, p: f"{int(x):,}"))

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plot_circuit_evaluations_vs_N.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [Saved] Circuit Evaluations Plot -> '{plot_path}'")


# =====================================================================
# 2. DATASET LOADER & FLEXIBLE FILE MAPPING (Fixed Index Capping)
# =====================================================================
def load_dataset_and_split(ds_name):
    """
    Robust loader supporting both old and new dataset file naming formats.
    Caps training indices to max 500 samples to match cached Gram matrix sizes.
    """
    if not os.path.exists(config.CACHE_DIR_DATASETS):
        print(f"   [Notice] Datasets directory '{config.CACHE_DIR_DATASETS}' does not exist.")
        return None, None, None, None

    all_files = os.listdir(config.CACHE_DIR_DATASETS)
    target_file = None

    keywords = []
    if "Primary" in ds_name or "Shells" in ds_name or "Synthetic" in ds_name:
        keywords = ["synthetic", "shells", "Primary"]
    elif "ZZ" in ds_name:
        keywords = ["qnative_ZZ", "ZZ_Generated", "ZZ"]
    elif "CPMap" in ds_name:
        keywords = ["qnative_CPMap", "CPMap_Generated", "CPMap"]
    elif "Real" in ds_name or "Fraud" in ds_name or "balanced" in ds_name:
        keywords = ["balanced", "Real", "fraud"]

    for f in all_files:
        if any(kw in f for kw in keywords):
            target_file = os.path.join(config.CACHE_DIR_DATASETS, f)
            break

    if not target_file:
        print(f"   [Notice] No raw dataset file found matching '{ds_name}' in '{config.CACHE_DIR_DATASETS}'.")
        return None, None, None, None

    if target_file.endswith(".npz"):
        data = np.load(target_file)
        X_all, y_all = data['X'], data['y']
    elif target_file.endswith(".csv"):
        df = pd.read_csv(target_file)
        y_all = df.iloc[:, 0].to_numpy()
        X_all = df.iloc[:, 1:].to_numpy()
    else:
        print(f"   [Notice] File '{target_file}' is neither .npz nor .csv.")
        return None, None, None, None

    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    max_train_samples = 1000 if ("Real" in ds_name or "Fraud" in ds_name or "balanced" in target_file) else 500
    max_test_samples = 500 if ("Real" in ds_name or "Fraud" in ds_name or "balanced" in target_file) else 250

    train_idx = train_idx[:min(max_train_samples, len(train_idx))]
    test_idx = test_idx[:min(max_test_samples, len(test_idx))]
        
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]


# =====================================================================
# 3. METRIC SWEEP (F1, ACCURACY, ROC-AUC, ALIGNMENT, CONDITION NUMBER)
# =====================================================================
def execute_continuous_sweep_and_all_plots():
    print("\n[*] Executing Continuous Sample Efficiency Sweep (F1, Accuracy, ROC-AUC, Alignment, Condition Number)...")
    
    if not os.path.exists(config.CACHE_DIR_BASE):
        print(f"   [Notice] Base cache dir '{config.CACHE_DIR_BASE}' does not exist.")
        return

    dataset_dirs = glob.glob(os.path.join(config.CACHE_DIR_BASE, "*"))
    step = 10
    max_n = max(config.N_LIST)
    fine_n_list = np.arange(10, max_n + 1, step) 
    rng = np.random.default_rng(config.SEED)
    
    quantum_model_keys = ['Quantum-ZZ (Linear)', 'Quantum-ZZ (Full)', 'Quantum-CPMap']

    for ds_dir in dataset_dirs:
        if not os.path.isdir(ds_dir) or os.path.basename(ds_dir) == "_datasets":
            continue
            
        ds_name = os.path.basename(ds_dir)
        print(f"\n   -> Processing Dataset Cache Folder: '{ds_name}'")
        
        X_train_full, X_test, y_train_full, y_test = load_dataset_and_split(ds_name)
        if y_train_full is None:
            continue
            
        f1_data, f1_ci_data = {}, {}
        auc_data, auc_ci_data = {}, {}
        acc_data, acc_ci_data = {}, {}
        cond_num_data = {}
        align_data = {}
        quantum_intersections = {}
        
        df_sweep = pd.DataFrame({'N_Samples': fine_n_list})

        # -------------------------------------------------------------
        # A. Classical Baseline Evaluation
        # -------------------------------------------------------------
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]

        clf_f1s, clf_f1_cis = [], []
        clf_aucs, clf_auc_cis = [], []
        clf_accs, clf_acc_cis = [], []
        
        for n in tqdm(fine_n_list, desc=f"   Sweeping Classical RBF", leave=False):
            n_pos = min(n // 2, len(pos_idx))
            n_neg = min(n - n_pos, len(neg_idx))
            
            # Safe np.nan insertion to prevent dropping valid curve coordinates
            if n_pos == 0 or n_neg == 0:
                clf_f1s.append(np.nan); clf_f1_cis.append(np.nan)
                clf_aucs.append(np.nan); clf_auc_cis.append(np.nan)
                clf_accs.append(np.nan); clf_acc_cis.append(np.nan)
                continue
                
            splits_f1, splits_auc, splits_acc = [], [], []
            for _ in range(3):
                sub_idx = rng.permutation(np.concatenate([
                    rng.choice(pos_idx, size=n_pos, replace=False),
                    rng.choice(neg_idx, size=n_neg, replace=False)
                ]))
                
                try:
                    clf = SVC(kernel='rbf', C=1.0, probability=True).fit(X_train_full[sub_idx], y_train_full[sub_idx])
                    y_pred = clf.predict(X_test)
                    y_prob = clf.predict_proba(X_test)[:, 1]
                    
                    splits_f1.append(f1_score(y_test, y_pred))
                    splits_auc.append(roc_auc_score(y_test, y_prob))
                    splits_acc.append(accuracy_score(y_test, y_pred))
                except ValueError:
                    splits_f1.append(np.nan)
                    splits_auc.append(np.nan)
                    splits_acc.append(np.nan)
                
            m_f1, ci_f1 = utils.calculate_95_ci(splits_f1)
            m_auc, ci_auc = utils.calculate_95_ci(splits_auc)
            m_acc, ci_acc = utils.calculate_95_ci(splits_acc)
            
            clf_f1s.append(m_f1); clf_f1_cis.append(ci_f1)
            clf_aucs.append(m_auc); clf_auc_cis.append(ci_auc)
            clf_accs.append(m_acc); clf_acc_cis.append(ci_acc)
            
        f1_data['Classical (RBF)'] = clf_f1s; f1_ci_data['Classical (RBF)'] = clf_f1_cis
        auc_data['Classical (RBF)'] = clf_aucs; auc_ci_data['Classical (RBF)'] = clf_auc_cis
        acc_data['Classical (RBF)'] = clf_accs; acc_ci_data['Classical (RBF)'] = clf_acc_cis
        
        df_sweep['Classical_RBF_F1'] = clf_f1s
        df_sweep['Classical_RBF_ROC_AUC'] = clf_aucs
        df_sweep['Classical_RBF_Accuracy'] = clf_accs
        
        # Guard retrieval to avoid referencing completely NaN lists
        valid_clf_f1s = [v for v in clf_f1s if not np.isnan(v)]
        classical_target_f1 = valid_clf_f1s[-1] if valid_clf_f1s else np.nan
        df_sweep['Classical_Target_F1_N500'] = classical_target_f1

        # -------------------------------------------------------------
        # B. Quantum Models Evaluation
        # -------------------------------------------------------------
        for q_key in quantum_model_keys:
            K_train_master, K_test_master = locate_kernel_matrices(ds_dir, q_key)
            if K_train_master is None:
                continue

            # Ensure data matches exact matrix dimensions of K_train_master and K_test_master
            n_train_matrix = K_train_master.shape[0]
            y_tr_bounded = y_train_full[:n_train_matrix]
            
            n_test_matrix = K_test_master.shape[0]
            y_te_bounded = y_test[:n_test_matrix]
            
            pos_idx_q = np.where(y_tr_bounded == 1)[0]
            neg_idx_q = np.where(y_tr_bounded == 0)[0]
                
            q_f1s, q_f1_cis = [], []
            q_aucs, q_auc_cis = [], []
            q_accs, q_acc_cis = [], []
            q_conds, q_aligns = [], []
            
            exact_n_reached = None
            target_reached_list = []

            for n in tqdm(fine_n_list, desc=f"   Sweeping {q_key[:20]}", leave=False):
                n_pos = min(n // 2, len(pos_idx_q))
                n_neg = min(n - n_pos, len(neg_idx_q))
                
                # Handling empty data frames gracefully
                if n_pos == 0 or n_neg == 0:
                    q_f1s.append(np.nan); q_f1_cis.append(np.nan)
                    q_aucs.append(np.nan); q_auc_cis.append(np.nan)
                    q_accs.append(np.nan); q_acc_cis.append(np.nan)
                    q_conds.append(np.nan)
                    q_aligns.append(np.nan)
                    target_reached_list.append(False)
                    continue

                splits_f1, splits_auc, splits_acc, splits_align = [], [], [], []
                for _ in range(3):
                    sub_idx = rng.permutation(np.concatenate([
                        rng.choice(pos_idx_q, size=n_pos, replace=False),
                        rng.choice(neg_idx_q, size=n_neg, replace=False)
                    ]))
                    
                    K_tr_sub = K_train_master[np.ix_(sub_idx, sub_idx)]
                    K_te_sub = K_test_master[:, sub_idx]
                    
                    try:
                        clf = SVC(kernel='precomputed', C=1.0, probability=True).fit(K_tr_sub, y_tr_bounded[sub_idx])
                        y_pred = clf.predict(K_te_sub)
                        y_prob = clf.predict_proba(K_te_sub)[:, 1]
                        
                        splits_f1.append(f1_score(y_te_bounded, y_pred))
                        splits_auc.append(roc_auc_score(y_te_bounded, y_prob))
                        splits_acc.append(accuracy_score(y_te_bounded, y_pred))
                        splits_align.append(utils.calculate_kernel_target_alignment(K_tr_sub, y_tr_bounded[sub_idx]))
                        
                    except ValueError:
                        splits_f1.append(np.nan)
                        splits_auc.append(np.nan)
                        splits_acc.append(np.nan)
                        splits_align.append(np.nan)

                m_f1, ci_f1 = utils.calculate_95_ci(splits_f1)
                m_auc, ci_auc = utils.calculate_95_ci(splits_auc)
                m_acc, ci_acc = utils.calculate_95_ci(splits_acc)
                m_align, _ = utils.calculate_95_ci(splits_align)
                
                q_f1s.append(m_f1); q_f1_cis.append(ci_f1)
                q_aucs.append(m_auc); q_auc_cis.append(ci_auc)
                q_accs.append(m_acc); q_acc_cis.append(ci_acc)
                q_aligns.append(m_align)

                eigs = np.sort(np.linalg.eigvalsh(K_tr_sub))[::-1]
                cond = eigs[0] / max(eigs[-1], 1e-12)
                q_conds.append(cond)

                is_met = m_f1 >= classical_target_f1 if not np.isnan(m_f1) and not np.isnan(classical_target_f1) else False
                target_reached_list.append(is_met)
                if exact_n_reached is None and is_met:
                    exact_n_reached = n

            f1_data[q_key] = q_f1s; f1_ci_data[q_key] = q_f1_cis
            auc_data[q_key] = q_aucs; auc_ci_data[q_key] = q_auc_cis
            acc_data[q_key] = q_accs; acc_ci_data[q_key] = q_acc_cis
            cond_num_data[q_key] = q_conds
            align_data[q_key] = q_aligns

            df_sweep[f'{q_key}_F1'] = q_f1s
            df_sweep[f'{q_key}_Accuracy'] = q_accs
            df_sweep[f'{q_key}_ROC_AUC'] = q_aucs
            df_sweep[f'{q_key}_Alignment'] = q_aligns
            df_sweep[f'{q_key}_Condition_Number'] = q_conds
            df_sweep[f'{q_key}_Reached_Target'] = target_reached_list

            if exact_n_reached is not None:
                idx_f = fine_n_list.tolist().index(exact_n_reached)
                quantum_intersections[q_key] = (exact_n_reached, q_f1s[idx_f])
                print(f"      [Target Reached] {q_key} hit classical target at N = {exact_n_reached}")

        csv_save_path = os.path.join(RESULTS_DIR, f"optimal_N_search_{ds_name}.csv")
        df_sweep.to_csv(csv_save_path, index=False)
        print(f"   [Saved] Sweep evolution ledger -> '{csv_save_path}'")

        clean_ds_title = ds_name.replace('_', ' ')

        # -------------------------------------------------------------
        # PLOT 1: F1 Score vs N
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6))
        for m_name, f1s in f1_data.items():
            f1_arr, cis = np.array(f1s, dtype=float), np.array(f1_ci_data[m_name], dtype=float)
            c = MODEL_COLORS.get(m_name, '#333')
            ax.plot(fine_n_list, f1_arr, label=m_name, color=c, linewidth=2.2)
            ax.fill_between(fine_n_list, f1_arr - cis, f1_arr + cis, color=c, alpha=0.12)

        ax.set_title(rf"Sample Efficiency Curve: {clean_ds_title}" + "\n" + r"($F_1$ Score vs. Training Sample Size $N$ with 95% CI)", fontweight='bold', fontsize=13, pad=12)
        ax.set_xlabel(r"Training Sample Size ($N$)", fontsize=11)
        ax.set_ylabel(r"Test $F_1$ Score (Mean $\pm$ 95% CI)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_sample_efficiency_F1_{ds_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        # -------------------------------------------------------------
        # PLOT 2: Accuracy vs N
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6))
        for m_name, accs in acc_data.items():
            acc_arr, cis = np.array(accs, dtype=float), np.array(acc_ci_data[m_name], dtype=float)
            c = MODEL_COLORS.get(m_name, '#333')
            ax.plot(fine_n_list, acc_arr, label=m_name, color=c, linewidth=2.2)
            ax.fill_between(fine_n_list, acc_arr - cis, acc_arr + cis, color=c, alpha=0.12)

        ax.set_title(rf"Model Accuracy: {clean_ds_title}" + "\n" + r"(Accuracy vs. Training Sample Size $N$ with 95% CI)", fontweight='bold', fontsize=13, pad=12)
        ax.set_xlabel(r"Training Sample Size ($N$)", fontsize=11)
        ax.set_ylabel(r"Test Accuracy (Mean $\pm$ 95% CI)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_sample_efficiency_Accuracy_{ds_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # -------------------------------------------------------------
        # PLOT 3: ROC-AUC Score vs N
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6))
        for m_name, aucs in auc_data.items():
            auc_arr, cis = np.array(aucs, dtype=float), np.array(auc_ci_data[m_name], dtype=float)
            c = MODEL_COLORS.get(m_name, '#333')
            ax.plot(fine_n_list, auc_arr, label=m_name, color=c, linewidth=2.2)
            ax.fill_between(fine_n_list, auc_arr - cis, auc_arr + cis, color=c, alpha=0.12)

        ax.set_title(rf"Classification Performance: {clean_ds_title}" + "\n" + r"(ROC-AUC Score vs. Training Sample Size $N$ with 95% CI)", fontweight='bold', fontsize=13, pad=12)
        ax.set_xlabel(r"Training Sample Size ($N$)", fontsize=11)
        ax.set_ylabel(r"Test ROC-AUC Score (Mean $\pm$ 95% CI)", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_sample_efficiency_ROC_AUC_{ds_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)

        # -------------------------------------------------------------
        # PLOT 4: Condition Number (kappa) vs N
        # -------------------------------------------------------------
        if cond_num_data:
            fig, ax = plt.subplots(figsize=(11, 6))
            for q_key, conds in cond_num_data.items():
                c = MODEL_COLORS.get(q_key, '#333')
                ax.plot(fine_n_list, np.array(conds, dtype=float), label=q_key, color=c, linewidth=2.2, marker='.')

            ax.set_title(rf"Gram Matrix Condition Number Decay: {clean_ds_title}" + "\n" + r"($\kappa = \lambda_{\max} / \lambda_{\min}$ vs. $N$ - Exponential Concentration Diagnostic)", fontweight='bold', fontsize=13, pad=12)
            ax.set_xlabel(r"Training Sample Size ($N$)", fontsize=11)
            ax.set_ylabel(r"Condition Number ($\kappa$, log scale)", fontsize=11)
            ax.set_yscale('log')
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper left', fontsize=10, frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"plot_condition_number_vs_N_{ds_name}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)
            
        # -------------------------------------------------------------
        # PLOT 5: Kernel Target Alignment vs N
        # -------------------------------------------------------------
        if align_data:
            fig, ax = plt.subplots(figsize=(11, 6))
            for q_key, aligns in align_data.items():
                c = MODEL_COLORS.get(q_key, '#333')
                ax.plot(fine_n_list, np.array(aligns, dtype=float), label=q_key, color=c, linewidth=2.2, marker='^')

            ax.set_title(rf"Kernel Target Alignment vs N: {clean_ds_title}", fontweight='bold', fontsize=13, pad=12)
            ax.set_xlabel(r"Training Sample Size ($N$)", fontsize=11)
            ax.set_ylabel(r"Target Alignment", fontsize=11)
            ax.grid(True, linestyle='--', alpha=0.4)
            ax.legend(loc='upper left', fontsize=10, frameon=True)
            plt.tight_layout()
            plt.savefig(os.path.join(RESULTS_DIR, f"plot_target_alignment_vs_N_{ds_name}.png"), dpi=300, bbox_inches='tight')
            plt.close(fig)

        # -------------------------------------------------------------
        # PLOT 6: Samples to Reach Target F1 Score (Optimal N Annotations)
        # -------------------------------------------------------------
        fig, ax = plt.subplots(figsize=(11, 6.5))
        for m_name, f1s in f1_data.items():
            ax.plot(fine_n_list, np.array(f1s, dtype=float), label=m_name, color=MODEL_COLORS.get(m_name, '#333'), linewidth=2.2, marker='.', markersize=4)

        if not np.isnan(classical_target_f1):
            ax.axhline(classical_target_f1, color='red', linestyle='--', alpha=0.75, 
                       label=rf"Classical Target $F_1$ ({classical_target_f1:.3f} at $N={max_n}$)")

        offset_mult = 1
        for q_key, (n_val, f1_val) in quantum_intersections.items():
            c = MODEL_COLORS.get(q_key)
            ax.axvline(n_val, color=c, linestyle=':', alpha=0.8, linewidth=1.5)
            ax.scatter([n_val], [f1_val], color=c, zorder=6, s=110, edgecolor='black', marker='*')
            
            y_off = 25 if (offset_mult % 2 == 1) else -35
            ax.annotate(
                f"{q_key}\nOptimal Target N = {n_val}", 
                xy=(n_val, f1_val), 
                xytext=(20, y_off),
                textcoords="offset points", 
                color=c, 
                fontweight='bold',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c, alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2)
            )
            offset_mult += 1

        ax.set_title(
            f"Samples to Reach Target F1 Score: {clean_ds_title}\n"
            f"(Target Baseline = Classical F1 at N={max_n})", 
            fontweight='bold', fontsize=13, pad=12
        )
        ax.set_xlabel(r"Number of Training Samples ($N$)", fontsize=11)
        ax.set_ylabel(r"Test $F_1$ Score", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10, frameon=True)

        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"plot_samples_to_target_{ds_name}.png"), dpi=300, bbox_inches='tight')
        plt.close(fig)


def locate_kernel_matrices(ds_dir, q_map_name):
    """
    Locates cached kernel matrices in main/ and ablation/ folders.
    Prints informative notice if the matrix file is missing.
    """
    if q_map_name == 'Quantum-ZZ (Linear)':
        train_p = os.path.join(ds_dir, "main", f"cache_Quantum-ZZ_seed{config.SEED}_train.npy")
        test_p = os.path.join(ds_dir, "main", f"cache_Quantum-ZZ_seed{config.SEED}_test.npy")
    elif q_map_name == 'Quantum-CPMap':
        train_p = os.path.join(ds_dir, "main", f"cache_Quantum-CPMap_seed{config.SEED}_train.npy")
        test_p = os.path.join(ds_dir, "main", f"cache_Quantum-CPMap_seed{config.SEED}_test.npy")
    elif q_map_name == 'Quantum-ZZ (Full)':
        ablation_folder = os.path.join(ds_dir, "ablation", "ZZ_entfull_bw1.0_noiseFalse")
        train_p = os.path.join(ablation_folder, "train_kernel.npy")
        test_p = os.path.join(ablation_folder, "test_kernel.npy")
    else:
        return None, None

    if os.path.exists(train_p) and os.path.exists(test_p):
        return np.load(train_p), np.load(test_p)
    else:
        print(f"   [Notice] Kernel matrix for '{q_map_name}' in '{os.path.basename(ds_dir)}' not ready yet.")
        return None, None



# =====================================================================
# 4. ABLATION INSIGHT DASHBOARDS
# =====================================================================
def render_ablation_dashboards():
    print("\n[*] Processing Ablation Dashboards...")
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("ablation_report_") and f.endswith(".csv")]
    
    if not csv_files:
        print("   [Notice] No ablation CSV reports found in 'results/'. Skipping dashboards.")
        return

    for csv_file in csv_files:
        try:
            ds_slug = csv_file.replace("ablation_report_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
            
            fig = plt.figure(figsize=(24, 14))
            fig.suptitle(rf"Ablation Insights & Diagnostics: {ds_slug.replace('_', ' ')}", fontsize=18, fontweight='bold', y=0.98)
            
            ax1, ax2, ax3, ax4 = plt.subplot(2,3,1), plt.subplot(2,3,2), plt.subplot(2,3,3), plt.subplot(2,3,4)
            ax5 = plt.subplot(2,3,(5,6))
            
            fmaps = df['Feature Map'].unique()
            abl_colors = {'ZZ': '#9467bd', 'CPMap': '#ff7f0e'}
            
            for fmap in fmaps:
                sub = df[df['Feature Map'] == fmap]
                g1 = sub.groupby('Bandwidth')['Target Alignment'].mean()
                ax1.plot(g1.index, g1.values, marker='o', label=fmap, color=abl_colors.get(fmap, '#333'))
                g2 = sub.groupby('Bandwidth')['Test F1'].mean()
                ax2.plot(g2.index, g2.values, marker='s', label=fmap, color=abl_colors.get(fmap, '#333'))
                g3 = sub.groupby('Bandwidth')['Condition Number'].mean()
                ax3.plot(g3.index, g3.values, marker='^', label=fmap, color=abl_colors.get(fmap, '#333'))

            ax1.set_title(r"Kernel Target Alignment vs Bandwidth ($\sigma$)"); ax1.set_xlabel(r"Bandwidth ($\sigma$)"); ax1.legend(); ax1.grid(True, alpha=0.5)
            ax2.set_title(r"Test $F_1$ Score vs Bandwidth ($\sigma$)"); ax2.set_xlabel(r"Bandwidth ($\sigma$)"); ax2.legend(); ax2.grid(True, alpha=0.5)
            ax3.set_title(r"Gram Matrix Condition No. vs Bandwidth ($\sigma$)"); ax3.set_xlabel(r"Bandwidth ($\sigma$)"); ax3.set_yscale('log'); ax3.legend(); ax3.grid(True, alpha=0.5)

            res_df = df[['Feature Map', 'CNOT Count']].drop_duplicates().groupby('Feature Map').mean()
            ax4.bar(res_df.index, res_df['CNOT Count'], color=[abl_colors.get(x, '#333') for x in res_df.index], edgecolor='black', alpha=0.85)
            ax4.set_title(r"Entangling Gate Overhead ($N_{\mathrm{CNOT}}$)")
            ax4.set_ylabel("Count")
            
            ax5.axis('off')
            ax5.text(0.05, 0.85, "Ablation matrix successfully processed from CSV ledger.", fontsize=14)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join(RESULTS_DIR, f"dashboard_ablation_{ds_slug}.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"   [Warning] Failed to generate ablation dashboard for {csv_file}: {e}")


# =====================================================================
# 5. SPECTRAL DIAGNOSTICS & VARIANCE
# =====================================================================
def plot_all_spectral_diagnostics():
    print("\n[*] Extracting Spectral Diagnostics and Off-Diagonal Variance...")
    kernel_files = glob.glob(os.path.join(config.CACHE_DIR_BASE, "**", "*_train.npy"), recursive=True)
    
    if not kernel_files:
        print("   [Notice] No cached .npy kernel files found for spectral diagnostics.")
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    labels, variances = [], []

    for idx, k_file in enumerate(kernel_files):
        try:
            K = np.load(k_file)
            if K.ndim != 2 or K.shape[0] != K.shape[1]: 
                continue
                
            raw_label = os.path.basename(k_file).replace(".npy", "").replace("cache_", "")
            c = cmap[idx % len(cmap)]
            
            eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
            eigs_norm = eigs / np.max(eigs)
            
            ax1.plot(eigs_norm, label=raw_label[:30], color=c, linewidth=2)
            
            off_diag = K[~np.eye(len(K), dtype=bool)]
            var = np.var(off_diag)
            
            labels.append(raw_label[:20])
            variances.append(var)
            
            ax2.bar(idx, var, color=c, alpha=0.85, edgecolor='black')
        except Exception:
            continue

    ax1.set_title(r"Eigenvalue Spectrum Decay ($\lambda_i / \lambda_{\max}$)", fontweight='bold', fontsize=13)
    ax1.set_xlabel(r"Eigenvalue Index ($i$)", fontsize=11)
    ax1.set_ylabel(r"Normalized Eigenvalue ($\lambda_i / \lambda_{\max}$)", fontsize=11)
    ax1.set_yscale('log')
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.legend(fontsize=8, loc='upper right')

    ax2.set_title(r"Off-Diagonal Kernel Variance ($\sigma^2$)", fontweight='bold', fontsize=13)
    ax2.set_ylabel(r"Variance ($\sigma^2$)", fontsize=11)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=35, ha='right', fontsize=8)
    ax2.grid(axis='y', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plot_kernel_spectral_diagnostics.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [Saved] Kernel Spectral Diagnostics -> '{plot_path}'")


# =====================================================================
# MAIN EXECUTION ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    generate_resource_table()
    plot_circuit_evaluations_vs_N()
    execute_continuous_sweep_and_all_plots()
    render_ablation_dashboards()
    plot_all_spectral_diagnostics()