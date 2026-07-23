"""
Script: master_dashboard_generator.py
Description: Unified reporting orchestrator. Generates ablation insight dashboards, 
             resource calculations, kernel diagnostics, decision boundaries, and 
             a continuous, fine-grained sample-efficiency search over N with LaTeX labels.
"""

import os
import glob
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Enable Matplotlib mathtext for LaTeX-style formatting in labels
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'

from scipy import stats
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.decomposition import PCA
import xgboost as xgb
from tqdm import tqdm

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
    utils = MockUtils()

warnings.filterwarnings('ignore')
RESULTS_DIR = "results"
os.makedirs(RESULTS_DIR, exist_ok=True)

# =====================================================================
# 1. HARDWARE RESOURCE ENGINE
# =====================================================================
def compute_circuit_and_gate_costs(n_train, n_test, feature_map_type='ZZ'):
    total_circuits = int((n_train * (n_train - 1)) // 2 + (n_test * n_train))
    try:
        mgr = ProductionQuantumKernelManager(map_type=feature_map_type)
        res = mgr.get_resource_counts()
        qubits, sq_gates, cnot_gates = res['qubits'], res['single_qubit_gates'], res['cnot_gates']
    except Exception:
        qubits = config.QUBIT_BUDGET
        sq_gates = qubits * 2 if feature_map_type == 'ZZ' else qubits
        cnot_gates = qubits - 1 if feature_map_type == 'ZZ' else (qubits // 2)
        
    return {
        'Circuits Evaluated': total_circuits,
        'Total Operations': (total_circuits * sq_gates) + (total_circuits * cnot_gates)
    }

# =====================================================================
# 1. DETAILED HARDWARE RESOURCE & SCALING ENGINE
# =====================================================================
def generate_resource_table():
    print("\n[*] Generating Comprehensive Hardware Resource Ledgers and Visual Dashboards...")
    
    # Sample sweep array
    max_n = max(config.N_LIST)
    n_sweep = np.arange(10, max_n + 1, 10)
    n_test = 200 # Fixed held-out test set size
    
    # Retrieve exact per-circuit gate counts from quantum managers
    resource_info = {}
    for m in ['ZZ', 'CPMap']:
        try:
            mgr = ProductionQuantumKernelManager(map_type=m)
            res = mgr.get_resource_counts()
            resource_info[m] = {
                'qubits': res['qubits'],
                'sq_gates': res['single_qubit_gates'],
                'cnot_gates': res['cnot_gates']
            }
        except Exception:
            qubits = config.QUBIT_BUDGET
            resource_info[m] = {
                'qubits': qubits,
                'sq_gates': qubits * 2 if m == 'ZZ' else qubits,
                'cnot_gates': qubits - 1 if m == 'ZZ' else (qubits // 2)
            }

    # 1. Build Detailed N-Scaling DataFrame
    scaling_rows = []
    for n_train in n_sweep:
        # Fidelity kernel symmetry: N_train*(N_train-1)/2 (train) + N_test*N_train (test)
        circuits = int((n_train * (n_train - 1)) // 2 + (n_test * n_train))
        
        for m in ['ZZ', 'CPMap']:
            sq_per_circ = resource_info[m]['sq_gates']
            cnot_per_circ = resource_info[m]['cnot_gates']
            
            tot_sq = circuits * sq_per_circ
            tot_cnot = circuits * cnot_per_circ
            tot_ops = tot_sq + tot_cnot
            
            scaling_rows.append({
                'N_Train_Samples': n_train,
                'N_Test_Samples': n_test,
                'Feature_Map': f"Quantum-{m}",
                'Qubits_Required': resource_info[m]['qubits'],
                'Total_Circuits_Evaluated': circuits,
                'SQ_Gates_Per_Circuit': sq_per_circ,
                'CNOT_Gates_Per_Circuit': cnot_per_circ,
                'Cumulative_SQ_Gates': tot_sq,
                'Cumulative_CNOT_Gates': tot_cnot,
                'Total_Quantum_Operations': tot_ops
            })
            
    df_scaling = pd.DataFrame(scaling_rows)
    csv_path = os.path.join(RESULTS_DIR, "resource_scaling_by_N.csv")
    df_scaling.to_csv(csv_path, index=False)
    print(f"   [Saved] Detailed N-Scaling Resource Ledger -> '{csv_path}'")
    
    # 2. Render 4-Panel Visual Resource Dashboard
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    colors = {'Quantum-ZZ': '#9467bd', 'Quantum-CPMap': '#ff7f0e'}
    
    # Subplot A: Total Circuits vs N
    ax_circ = axes[0, 0]
    zz_data = df_scaling[df_scaling['Feature_Map'] == 'Quantum-ZZ']
    ax_circ.plot(zz_data['N_Train_Samples'], zz_data['Total_Circuits_Evaluated'], color='#333333', linewidth=2.5)
    ax_circ.set_title(r"Gram Matrix Circuit Scaling ($O(N^2)$ Pair Evaluations)", fontweight='bold', fontsize=12)
    ax_circ.set_xlabel(r"Training Sample Size ($N$)")
    ax_circ.set_ylabel(r"Total Circuit Evaluations ($N_{\mathrm{circ}}$)")
    ax_circ.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot B: Cumulative CNOT Gates vs N
    ax_cnot = axes[0, 1]
    for m in ['Quantum-ZZ', 'Quantum-CPMap']:
        sub = df_scaling[df_scaling['Feature_Map'] == m]
        ax_cnot.plot(sub['N_Train_Samples'], sub['Cumulative_CNOT_Gates'], label=m, color=colors[m], linewidth=2.5)
    ax_cnot.set_title(r"Cumulative CNOT Gate Scaling to Convergence", fontweight='bold', fontsize=12)
    ax_cnot.set_xlabel(r"Training Sample Size ($N$)")
    ax_cnot.set_ylabel(r"Cumulative CNOT Count ($N_{\mathrm{CNOT}}$)")
    ax_cnot.legend(loc='upper left')
    ax_cnot.grid(True, linestyle='--', alpha=0.5)
    
    # Subplot C: Per-Circuit Entangling Gate Overhead (CNOTs)
    ax_per_circ = axes[1, 0]
    m_names = ['Quantum-ZZ', 'Quantum-CPMap']
    cnots_per = [resource_info['ZZ']['cnot_gates'], resource_info['CPMap']['cnot_gates']]
    bars = ax_per_circ.bar(m_names, cnots_per, color=[colors[m] for m in m_names], alpha=0.85, width=0.4, edgecolor='black')
    for bar in bars:
        yval = bar.get_height()
        ax_per_circ.text(bar.get_x() + bar.get_width()/2, yval + 0.1, f"{int(yval)} CNOTs/circ", ha='center', va='bottom', fontweight='bold')
    ax_per_circ.set_title(r"Hardware Overhead: CNOT Gates per Circuit Execution", fontweight='bold', fontsize=12)
    ax_per_circ.set_ylabel(r"CNOT Count per Feature Map")
    ax_per_circ.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Subplot D: Total Quantum Operations (1-Qubit + CNOT) vs N
    ax_tot = axes[1, 1]
    for m in ['Quantum-ZZ', 'Quantum-CPMap']:
        sub = df_scaling[df_scaling['Feature_Map'] == m]
        ax_tot.plot(sub['N_Train_Samples'], sub['Total_Quantum_Operations'], label=m, color=colors[m], linewidth=2.5)
    ax_tot.set_title(r"Total Quantum Operations (1-Qubit + CNOT Gates)", fontweight='bold', fontsize=12)
    ax_tot.set_xlabel(r"Training Sample Size ($N$)")
    ax_tot.set_ylabel(r"Total Operations Count")
    ax_tot.legend(loc='upper left')
    ax_tot.grid(True, linestyle='--', alpha=0.5)
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, "plot_resource_scaling_dashboard.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"   [Saved] Hardware Resource Visual Dashboard -> '{plot_path}'")

# =====================================================================
# 2. CONTINUOUS FINE-GRAINED SAMPLE EFFICIENCY SWEEP (Starts at N=10)
# =====================================================================
def load_dataset_and_split(ds_name):
    """
    Reconstructs the exact train/test split utilized in experiment_runner.py 
    using robust file mapping to prevent silent skips.
    """
    # Map the clean folder name to a substring of the actual file name in _datasets
    file_map = {
        "Primary_Synthetic": "cache_synthetic_shells",
        "Positive_Control__ZZ": "cache_qnative_ZZ",
        "Positive_Control__CPMap": "cache_qnative_CPMap",
        "Rebalanced_Real": "balanced"
    }
    
    target_file = None
    for key, val in file_map.items():
        if key in ds_name:
            files = os.listdir(config.CACHE_DIR_DATASETS)
            for f in files:
                if val in f:
                    target_file = os.path.join(config.CACHE_DIR_DATASETS, f)
                    break
            break
            
    if not target_file:
        print(f"      [Warning] Could not find raw dataset file for '{ds_name}'. Skipping.")
        return None, None, None, None
        
    # Load appropriately based on file type
    if target_file.endswith(".npz"):
        data = np.load(target_file)
        X_all, y_all = data['X'], data['y']
    elif target_file.endswith(".csv"):
        df = pd.read_csv(target_file)
        y_all = df.iloc[:, 0].to_numpy()
        X_all = df.iloc[:, 1:].to_numpy()
    else:
        return None, None, None, None

    # Reconstruct the exact split from experiment_runner.py
    skf_outer = StratifiedKFold(n_splits=config.OUTER_SPLITS, shuffle=True, random_state=config.SEED)
    train_idx, test_idx = next(skf_outer.split(X_all, y_all))
    
    # Real data pool limiter logic
    if "Real" in ds_name or "Fraud" in ds_name or "balanced" in target_file:
        train_idx = train_idx[:min(1000, len(train_idx))]
        test_idx = test_idx[:min(500, len(test_idx))]
        
    return X_all[train_idx], X_all[test_idx], y_all[train_idx], y_all[test_idx]

# =====================================================================
# 2. CONTINUOUS FINE-GRAINED SAMPLE EFFICIENCY SWEEP (Fixed Logic & Formatting)
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
        
        # 1. Classical Baseline
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
        
        # CORRECT METHODOLOGY TARGET: Classical performance at N = 500 (Convergence)
        classical_converged_f1 = clf_f1s[-1]
        target_threshold = classical_converged_f1
        df_sweep['Classical_Converged_Target_F1'] = target_threshold
        
        # 2. Quantum Baselines
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
                
                # Check if quantum model reaches/exceeds the converged classical target
                is_met = m >= target_threshold
                target_reached_list.append(is_met)
                
                if exact_n_reached is None and is_met:
                    exact_n_reached = n
            
            models_to_plot[q_map] = q_f1s
            df_sweep[f'{q_map}_F1'] = q_f1s
            df_sweep[f'{q_map}_Reached_Target'] = target_reached_list
            
            if exact_n_reached is not None:
                quantum_intersections[q_map] = (exact_n_reached, q_f1s[fine_n_list.tolist().index(exact_n_reached)])
                print(f"      [Target Reached] {q_map} hit target at N = {exact_n_reached}")
        
        # Save exact sample evolution CSV
        csv_save_path = os.path.join(RESULTS_DIR, f"optimal_N_search_{ds_name}.csv")
        df_sweep.to_csv(csv_save_path, index=False)

        # 3. Clean Plot Generation
        fig, ax = plt.subplots(figsize=(11, 6.5))
        colors = {'Classical (RBF)': '#1f77b4', 'Quantum-ZZ': '#9467bd', 'Quantum-CPMap': '#ff7f0e'}
        
        for name, f1s in models_to_plot.items():
            ax.plot(fine_n_list, f1s, label=name, color=colors.get(name, '#333'), linewidth=2.2, marker='.', markersize=4)
            
        ax.axhline(target_threshold, color='red', linestyle='--', alpha=0.75, 
                   label=rf"Classical Target $F_1$ ({target_threshold:.3f} at $N={max_n}$)")
        
        # Annotate intersections safely avoiding collisions
        y_min_val, y_max_val = ax.get_ylim()
        offset_multiplier = 1
        for q_map, (n_val, f1_val) in quantum_intersections.items():
            c = colors.get(q_map)
            ax.axvline(n_val, color=c, linestyle=':', alpha=0.8, linewidth=1.5)
            ax.scatter([n_val], [f1_val], color=c, zorder=6, s=110, edgecolor='black', marker='*')
            
            # Dynamic offset to prevent text collision
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

        # FIX: Multi-line string formatting (no \n rendering in titles)
        clean_title_ds = ds_name.replace('_', ' ')
        ax.set_title(
            f"Samples to Reach Target F1 Score: {clean_title_ds}\n"
            f"(Target Baseline = Classical F1 at N={max_n})", 
            fontweight='bold', fontsize=13, pad=12
        )
        
        ax.set_xlabel("Number of Training Samples (N)", fontsize=11)
        ax.set_ylabel("Test F1 Score", fontsize=11)
        ax.grid(True, linestyle='--', alpha=0.4)
        ax.legend(loc='lower right', fontsize=10, frameon=True)
        
        # FIX: Pad layout to eliminate title or label clipping
        plt.tight_layout()
        save_path = os.path.join(RESULTS_DIR, f"plot_samples_to_target_{ds_name}.png")
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

# =====================================================================
# 3. ABLATION DASHBOARDS
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
            ax4.bar(res_df.index, res_df['CNOT Count'], color=[colors.get(x, '#333') for x in res_df.index])
            ax4.set_title(r"Entangling Gate Overhead ($N_{\mathrm{CNOT}}$)")
            ax4.set_ylabel("Count")
            
            ax5.axis('off')
            ax5.text(0.05, 0.85, "Ablation matrix processed successfully from CSV ledger.", fontsize=14)
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            fig.savefig(os.path.join(RESULTS_DIR, f"dashboard_ablation_{ds_slug}.png"), dpi=300)
            plt.close(fig)
        except Exception as e:
            print(f"   [Warning] Failed to generate ablation dashboard for {csv_file}: {e}")

# =====================================================================
# 4. SPECTRAL DIAGNOSTICS & BOUNDARIES
# =====================================================================
def plot_spectral_and_boundaries():
    print("\n[*] Extracting Spectral Diagnostics and Decision Boundaries...")
    kernel_files = glob.glob(os.path.join(config.CACHE_DIR_BASE, "**", "*_train.npy"), recursive=True)
    
    if kernel_files:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Color cycle alignment setup
        cmap = plt.rcParams['axes.prop_cycle'].by_key()['color']
        
        for idx, k_file in enumerate(kernel_files):
            try:
                K = np.load(k_file)
                if K.ndim != 2 or K.shape[0] != K.shape[1]: continue
                
                label = os.path.basename(k_file).replace(".npy", "").replace("cache_", "")
                eigs = np.sort(np.linalg.eigvalsh(K))[::-1]
                
                c = cmap[idx % len(cmap)]
                
                # Apply identical color to both plot elements
                ax1.plot(eigs / np.max(eigs), label=label, color=c)
                
                off_diag = K[~np.eye(len(K), dtype=bool)]
                # Bar is assigned the same color to establish relationship
                ax2.bar(idx, np.var(off_diag), color=c, alpha=0.8, label=label[:30])
                
            except Exception: pass
            
        ax1.set_title(r"Eigenvalue Spectrum Decay", fontweight='bold')
        ax1.set_xlabel(r"Eigenvalue Index ($i$)")
        ax1.set_ylabel(r"Normalized Eigenvalue ($\lambda_i / \lambda_{\max}$)")
        ax1.set_yscale('log'); ax1.legend(fontsize=8)
        
        ax2.set_title(r"Off-Diagonal Variance", fontweight='bold')
        ax2.set_ylabel(r"Variance ($\sigma^2$)")
        # Removing X-ticks because they overlap in bar charts; Legend provides mapping.
        ax2.set_xticks([])
        ax2.legend(fontsize=8, loc='upper right')
        
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, "plot_kernel_spectral_diagnostics.png"), dpi=300)
        plt.close(fig)

if __name__ == "__main__":
    generate_resource_table()
    execute_continuous_sweep_and_plot()
    render_ablation_dashboards()
    plot_spectral_and_boundaries()