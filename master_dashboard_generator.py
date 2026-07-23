"""
Script: master_dashboard_generator.py
Description: Master reporting and evaluation orchestrator. Inspects live Qiskit circuit objects
             dynamically to measure exact gate counts per overlap circuit (U^\dagger U),
             tracks sample efficiency continuous sweeps, generates spectral diagnostic plots,
             renders ablation dashboards, and exports comprehensive CSV ledgers.
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
from sklearn.decomposition import PCA
import xgboost as xgb
from qiskit.circuit.library import ZZFeatureMap
from tqdm import tqdm

# Enable Matplotlib mathtext for publication-grade LaTeX formatting
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

# Import local project modules safely
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
            mean = np.mean(data)
            n = len(data)
            if n <= 1: return mean, 0.0
            se = stats.sem(data)
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


# =====================================================================
# 1. DYNAMIC CIRCUIT INSPECTION & HARDWARE RESOURCE ENGINE
# =====================================================================
def inspect_live_circuit_gates(map_type='CPMap', entanglement='linear', reps=1):
    """
    Instantiates live Qiskit feature map objects, constructs the exact compute-uncompute 
    fidelity circuit (U^\dagger U), unrolls custom blocks down to native basis gates, 
    and reads gate counts directly from the Qiskit circuit DAG/ops dictionary.
    """
    if map_type == 'ZZ':
        fmap = ZZFeatureMap(
            feature_dimension=config.QUBIT_BUDGET, 
            reps=reps, 
            entanglement=entanglement
        )
        qubits = config.QUBIT_BUDGET
    else: # CPMap
        mgr = ProductionQuantumKernelManager(map_type='CPMap')
        fmap = mgr.feature_map
        qubits = mgr.num_qubits

    # Fully decompose custom gates down to fundamental basis gates (cx, rz, ry, h)
    decomposed_fmap = fmap.decompose()
    if map_type == 'CPMap':
        decomposed_fmap = decomposed_fmap.decompose()

    # Build exact compute-uncompute overlap circuit: U^\dagger(x') U(x)
    overlap_circuit = decomposed_fmap.compose(decomposed_fmap.inverse())
    ops = overlap_circuit.count_ops()

    cnot_count = ops.get('cx', 0)
    sq_gates = ops.get('h', 0) + ops.get('rz', 0) + ops.get('ry', 0) + ops.get('rx', 0)

    return {
        'Qubits': qubits,
        'CNOT_per_circuit': cnot_count,
        'SQ_gates_per_circuit': sq_gates,
        'Total_gates_per_circuit': cnot_count + sq_gates
    }


def generate_resource_table():
    """Generates dynamic hardware cost ledgers and a 4-panel visual dashboard."""
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
    print("\n--- DYNAMICALLY EXTRACTED GATE COUNTS (PER OVERLAP CIRCUIT U^\dagger U) ---")
    print(df_hw.to_string(index=False))
    
    # Calculate resource scaling across N
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
    
    # Render 4-Panel Resource Visual Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {'Quantum-ZZ (Linear)': '#9467bd', 'Quantum-ZZ (Full)': '#d62728', 'Quantum-CPMap': '#ff7f0e'}
    
    # Panel 1: Circuit Pair Evaluations
    ax1 = axes[0, 0]
    sub_c = df_scaling[df_scaling['Configuration'] == 'Quantum-CPMap']
    ax1.plot(sub_c['N_Train_Samples'], sub_c['Pair_Circuits_Evaluated'], color='#333333', linewidth=2.5)
    ax1.set_title(r"Gram Matrix Circuit Scaling ($O(N^2)$ Pair Evaluations)", fontweight='bold', fontsize=12)
    ax1.set_xlabel(r"Training Sample Size ($N$)")
    ax1.set_ylabel(r"Total Overlap Circuits ($N_{\mathrm{circ}}$)")
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Panel 2: Cumulative CNOT Scaling
    ax2 = axes[0, 1]
    for label in colors:
        sub = df_scaling[df_scaling['Configuration'] == label]
        ax2.plot(sub['N_Train_Samples'], sub['Cumulative_CNOT_Operations'], label=label, color=colors[label], linewidth=2.2)
    ax2.set_title(r"Cumulative CNOT Operations to Convergence", fontweight='bold', fontsize=12)
    ax2.set_xlabel(r"Training Sample Size ($N$)")
    ax2.set_ylabel(r"Cumulative CNOT Count ($N_{\mathrm{CNOT}}$)")
    ax2.legend(loc='upper left', fontsize=9)
    ax2.grid(True, linestyle='--', alpha=0.5)

    # Panel 3: Per-Circuit CNOT Overhead
    ax3 = axes[1, 0]
    bars = ax3.bar(df_hw['Configuration'], df_hw['CNOTs / Overlap Circuit (U^dagger U)'], 
                   color=[colors.get(c, '#333') for c in df_hw['Configuration']], alpha=0.85, width=0.4, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{int(yval)} CNOTs", ha='center', va='bottom', fontweight='bold')
    ax3.set_title(r"Per-Circuit Hardware Overhead ($U^\dagger U$)", fontweight='bold', fontsize=12)
    ax3.set_ylabel(r"CNOT Count per Overlap Circuit")
    plt.setp(ax3.get_xticklabels(), rotation=15, ha='right', fontsize=9)
    ax3.grid(axis='y', linestyle='--', alpha=0.5)

    # Panel 4: Cumulative Total Operations
    ax4 = axes[1, 1]
    for label in colors:
        sub = df_scaling[df_scaling['Configuration'] == label]
        ax4.plot(sub['N_Train_Samples'], sub['Cumulative_Total_Operations'], label=label, color=colors[label], linewidth=2.2)
    ax4.set_title(r"Total Quantum Operations (1-Qubit + CNOTs)", fontweight='bold', fontsize=12)
    ax4.set_xlabel(r"Training Sample Size ($N$)")
    ax4.set_ylabel(r"Total Operations Count")
    ax4.legend(loc='upper left', fontsize=9)
    ax4.grid(True, linestyle='--', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plot_resource_scaling_dashboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [Saved] Verified Hardware Resource Dashboard -> '{plot_path}'")


# =====================================================================
# 2. DATASET LOADER WITH ROBUST FILENAME MAPPING
# =====================================================================
def load_dataset_and_split(ds_name):
    file_map = {
        "Primary_Synthetic": "Primary_Synthetic_Shells",
        "Positive_Control__ZZ": "Positive_Control_ZZ_Generated",
        "Positive_Control__CPMap": "Positive_Control_CPMap_Generated",
        "Rebalanced_Real": "Rebalanced_Real_Data"
    }
    
    target_file = None
    for key, val in file_map.items():
        if key in ds_name or val in ds_name:
            files = os.listdir(config.CACHE_DIR_DATASETS)
            for f in files:
                if val in f or key in f:
                    target_file = os.path.join(config.CACHE_DIR_DATASETS, f)
                    break
            break
            
    if not target_file:
        return None, None, None, None
        
    if target_file.endswith(".npz"):
        data = np.load(target_file)
        X_all, y_all = data['X'], data['y']
    elif target_file.endswith(".csv"):
        df = pd.read_csv(target_file)
        y_all = df.iloc[:, 0].to_numpy()
        X_all = df.iloc[:, 1:].to_numpy()
    else:
        return None, None, None, None

    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    if "Real" in ds_name or "Fraud" in ds_name or "Rebalanced" in target_file:
        train_idx = train_idx[:min(1000, len(train_idx))]
        test_idx = test_idx[:min(500, len(test_idx))]
        
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]


# =====================================================================
# 3. CONTINUOUS SAMPLE EFFICIENCY SWEEP & TARGET F1 SEARCH (FROM N=10)
# =====================================================================
def execute_continuous_sweep_and_plot():
    print("\n[*] Executing Fine-Grained Sample Efficiency Sweep (Target F1 Search)...")
    
    dataset_dirs = glob.glob(os.path.join(config.CACHE_DIR_BASE, "*"))
    step = 10
    max_n = max(config.N_LIST)
    fine_n_list = np.arange(10, max_n + 1, step) 
    rng = np.random.default_rng(config.SEED)
    
    for ds_dir in dataset_dirs:
        if not os.path.isdir(ds_dir) or os.path.basename(ds_dir) == "_datasets":
            continue
            
        ds_name = os.path.basename(ds_dir)
        X_train_full, X_test, y_train_full, y_test = load_dataset_and_split(ds_name)
        if y_train_full is None:
            continue
            
        pos_idx = np.where(y_train_full == 1)[0]
        neg_idx = np.where(y_train_full == 0)[0]
        
        models_to_plot = {}
        quantum_intersections = {}
        df_sweep = pd.DataFrame({'N_Samples': fine_n_list})
        
        # 1. Classical Baseline Continuous Sweep
        clf_f1s = []
        for n in tqdm(fine_n_list, desc=f"RBF-SVC ({ds_name[:15]})", leave=False):
            n_pos = min(n // 2, len(pos_idx))
            n_neg = min(n - n_pos, len(neg_idx))
            if n_pos == 0 or n_neg == 0:
                clf_f1s.append(0.0)
                continue
                
            splits = []
            for _ in range(3):
                sub_idx = rng.permutation(np.concatenate([
                    rng.choice(pos_idx, size=n_pos, replace=False),
                    rng.choice(neg_idx, size=n_neg, replace=False)
                ]))
                clf = SVC(kernel='rbf', C=1.0).fit(X_train_full[sub_idx], y_train_full[sub_idx])
                splits.append(f1_score(y_test, clf.predict(X_test)))
            clf_f1s.append(np.mean(splits))
            
        models_to_plot['Classical (RBF)'] = clf_f1s
        df_sweep['Classical_RBF_F1'] = clf_f1s
        
        # METHODOLOGY TARGET: Classical performance at full convergence (N = 500)
        classical_converged_f1 = clf_f1s[-1]
        target_threshold = classical_converged_f1
        df_sweep['Classical_Target_F1_At_N500'] = target_threshold
        
        # 2. Quantum Baselines Continuous Sweep via Precomputed Matrices
        for q_map in ['Quantum-ZZ', 'Quantum-CPMap']:
            train_k_path = os.path.join(ds_dir, "main", f"cache_{q_map}_seed{config.SEED}_train.npy")
            test_k_path = os.path.join(ds_dir, "main", f"cache_{q_map}_seed{config.SEED}_test.npy")
            
            if not (os.path.exists(train_k_path) and os.path.exists(test_k_path)):
                continue
                
            K_train_master = np.load(train_k_path)
            K_test_master = np.load(test_k_path)
            
            q_f1s = []
            exact_n_reached = None
            target_reached_list = []
            
            for n in tqdm(fine_n_list, desc=f"{q_map} ({ds_name[:15]})", leave=False):
                n_pos = min(n // 2, len(pos_idx))
                n_neg = min(n - n_pos, len(neg_idx))
                if n_pos == 0 or n_neg == 0:
                    q_f1s.append(0.0)
                    target_reached_list.append(False)
                    continue
                    
                splits = []
                for _ in range(3):
                    sub_idx = rng.permutation(np.concatenate([
                        rng.choice(pos_idx, size=n_pos, replace=False),
                        rng.choice(neg_idx, size=n_neg, replace=False)
                    ]))
                    clf = SVC(kernel='precomputed', C=1.0).fit(K_train_master[np.ix_(sub_idx, sub_idx)], y_train_full[sub_idx])
                    splits.append(f1_score(y_test, clf.predict(K_test_master[:, sub_idx])))
                    
                m = np.mean(splits)
                q_f1s.append(m)
                
                is_met = m >= target_threshold
                target_reached_list.append(is_met)
                
                if exact_n_reached is None and is_met:
                    exact_n_reached = n
            
            models_to_plot[q_map] = q_f1s
            df_sweep[f'{q_map}_F1'] = q_f1s
            df_sweep[f'{q_map}_Reached_Target'] = target_reached_list
            
            if exact_n_reached is not None:
                quantum_intersections[q_map] = (exact_n_reached, q_f1s[fine_n_list.tolist().index(exact_n_reached)])
                print(f"      [Target Reached] {q_map} hit classical target at N = {exact_n_reached}")
        
        # Save exact sample evolution CSV
        csv_save_path = os.path.join(RESULTS_DIR, f"optimal_N_search_{ds_name}.csv")
        df_sweep.to_csv(csv_save_path, index=False)

        # Plot Generation with Clean Formatting
        fig, ax = plt.subplots(figsize=(11, 6.5))
        colors = {'Classical (RBF)': '#1f77b4', 'Quantum-ZZ': '#9467bd', 'Quantum-CPMap': '#ff7f0e'}
        
        for name, f1s in models_to_plot.items():
            ax.plot(fine_n_list, f1s, label=name, color=colors.get(name, '#333'), linewidth=2.2, marker='.', markersize=4)
            
        ax.axhline(target_threshold, color='red', linestyle='--', alpha=0.75, 
                   label=rf"Classical Target $F_1$ ({target_threshold:.3f} at $N={max_n}$)")
        
        # Dynamic offsets to eliminate label collision
        offset_multiplier = 1
        for q_map, (n_val, f1_val) in quantum_intersections.items():
            c = colors.get(q_map)
            ax.axvline(n_val, color=c, linestyle=':', alpha=0.8, linewidth=1.5)
            ax.scatter([n_val], [f1_val], color=c, zorder=6, s=110, edgecolor='black', marker='*')
            
            text_y_offset = 25 if (offset_multiplier % 2 == 1) else -35
            ax.annotate(
                f"{q_map}\nOptimal Target N = {n_val}", 
                xy=(n_val, f1_val), 
                xytext=(20, text_y_offset),
                textcoords="offset points", 
                color=c, 
                fontweight='bold',
                fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor=c, alpha=0.85),
                arrowprops=dict(arrowstyle="->", color=c, lw=1.2)
            )
            offset_multiplier += 1

        clean_title_ds = ds_name.replace('_', ' ')
        ax.set_title(
            f"Samples to Reach Target F1 Score: {clean_title_ds}\n"
            f"(Target Baseline = Classical F1 at N={max_n})", 
            fontweight='bold', fontsize=13, pad=12
        )
        
        ax.set_xlabel(r"Number of Training Samples ($N$)", fontsize=11)
        ax.set_ylabel(r"Test $F_1$ Score", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"plot_samples_to_target_{ds_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)


# =====================================================================
# 4. ABLATION INSIGHT DASHBOARDS
# =====================================================================
def render_ablation_dashboards():
    print("\n[*] Processing Ablation Dashboards...")
    csv_files = [f for f in os.listdir(RESULTS_DIR) if f.startswith("ablation_report_") and f.endswith(".csv")]
    
    for csv_file in csv_files:
        try:
            ds_slug = csv_file.replace("ablation_report_", "").replace(".csv", "")
            df = pd.read_csv(os.path.join(RESULTS_DIR, csv_file))
            
            fig = plt.figure(figsize=(24, 14))
            fig.suptitle(rf"Ablation Insights & Diagnostics: {ds_slug.replace('_', ' ')}", fontsize=18, fontweight='bold', y=0.98)
            
            ax1, ax2, ax3, ax4 = plt.subplot(2,3,1), plt.subplot(2,3,2), plt.subplot(2,3,3), plt.subplot(2,3,4)
            ax5 = plt.subplot(2,3,(5,6))
            
            fmaps, colors = df['Feature Map'].unique(), {'ZZ': '#9467bd', 'CPMap': '#ff7f0e'}
            
            for fmap in fmaps:
                sub = df[df['Feature Map'] == fmap]
                g1 = sub.groupby('Bandwidth')['Target Alignment'].mean()
                ax1.plot(g1.index, g1.values, marker='o', label=fmap, color=colors.get(fmap, '#333'))
                g2 = sub.groupby('Bandwidth')['Test F1'].mean()
                ax2.plot(g2.index, g2.values, marker='s', label=fmap, color=colors.get(fmap, '#333'))
                g3 = sub.groupby('Bandwidth')['Condition Number'].mean()
                ax3.plot(g3.index, g3.values, marker='^', label=fmap, color=colors.get(fmap, '#333'))

            ax1.set_title(r"Kernel Target Alignment vs Bandwidth ($\sigma$)"); ax1.set_xlabel(r"Bandwidth ($\sigma$)"); ax1.legend(); ax1.grid(True, alpha=0.5)
            ax2.set_title(r"Test $F_1$ Score vs Bandwidth ($\sigma$)"); ax2.set_xlabel(r"Bandwidth ($\sigma$)"); ax2.legend(); ax2.grid(True, alpha=0.5)
            ax3.set_title(r"Gram Matrix Condition No. vs Bandwidth ($\sigma$)"); ax3.set_xlabel(r"Bandwidth ($\sigma$)"); ax3.set_yscale('log'); ax3.legend(); ax3.grid(True, alpha=0.5)

            res_df = df[['Feature Map', 'CNOT Count']].drop_duplicates().groupby('Feature Map').mean()
            ax4.bar(res_df.index, res_df['CNOT Count'], color=[colors.get(x, '#333') for x in res_df.index], edgecolor='black', alpha=0.85)
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


# =====================================================================
# MAIN EXECUTION ORCHESTRATOR
# =====================================================================
if __name__ == "__main__":
    generate_resource_table()
    execute_continuous_sweep_and_plot()
    render_ablation_dashboards()
    plot_all_spectral_diagnostics()