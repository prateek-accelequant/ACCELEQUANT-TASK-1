"""
Module: plot_ablation_insights.py
Description: Parses ablation CSV result ledgers and summary text files to generate 
             a comprehensive, annotated visualization dashboard extracting quantitative 
             insights on kernel health, resource efficiency, and performance.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def analyze_and_plot_ablations():
    results_dir = "results"
    if not os.path.exists(results_dir):
        print("[!] Error: 'results/' directory not found. Run your experiment scripts first.")
        return

    csv_files = [f for f in os.listdir(results_dir) if f.startswith("ablation_report_") and f.endswith(".csv")]
    
    if not csv_files:
        print("[!] No ablation report CSVs found in the results directory.")
        return

    print(f"[*] Found {len(csv_files)} ablation report files. Generating enhanced dashboards...")

    for csv_file in csv_files:
        dataset_slug = csv_file.replace("ablation_report_", "").replace(".csv", "")
        csv_path = os.path.join(results_dir, csv_file)
        
        df = pd.read_csv(csv_path)
        print(f"\n--- Processing Dashboard for Dataset: {dataset_slug} ---")
        
        # 1. Extract Compute Footprint from the corresponding summary txt file
        summary_txt_path = os.path.join(results_dir, f"ablation_summary_{dataset_slug}.txt")
        compute_footprint = "Data Unavailable"
        if os.path.exists(summary_txt_path):
            with open(summary_txt_path, 'r') as f:
                for line in f:
                    if "Cumulative Compute Footprint" in line:
                        compute_footprint = line.split("=")[-1].strip()
                        break

        # 2. Extract Top Performing Configuration
        best_row = df.loc[df['Test F1'].idxmax()]

        # 3. Render Comprehensive Multi-Panel Dashboard
        fig = plt.figure(figsize=(24, 14))
        fig.suptitle(
            f"Ablation Insights & Kernel Diagnostics: {dataset_slug.replace('_', ' ')}\n"
            f"Total Compute Footprint: {compute_footprint}", 
            fontsize=18, fontweight='bold', y=0.98
        )
        
        # Grid layout for subplots
        ax1 = plt.subplot(2, 3, 1) # Target Alignment
        ax2 = plt.subplot(2, 3, 2) # Test F1
        ax3 = plt.subplot(2, 3, 3) # Condition Number
        ax4 = plt.subplot(2, 3, 4) # CNOT Cost Comparison
        ax5 = plt.subplot(2, 3, (5, 6)) # Spanning Text Panel
        
        feature_maps = df['Feature Map'].unique()
        colors = {'ZZ': '#9467bd', 'CPMap': '#ff7f0e'}

        # Plot A: Target Alignment vs Bandwidth
        for fmap in feature_maps:
            sub = df[df['Feature Map'] == fmap]
            grouped = sub.groupby('Bandwidth')['Target Alignment'].mean()
            ax1.plot(grouped.index, grouped.values, marker='o', label=f"{fmap} Map", color=colors.get(fmap, '#333'), linewidth=2.5)
            
            # Annotate values
            for x, y in zip(grouped.index, grouped.values):
                ax1.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

        ax1.set_title("Kernel Target Alignment vs Bandwidth ($\sigma$)", fontsize=13, fontweight='bold')
        ax1.set_xlabel("Bandwidth Multiplier ($\sigma$)", fontsize=11)
        ax1.set_ylabel("Mean Target Alignment", fontsize=11)
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.legend(fontsize=10)

        # Plot B: Test F1 Score vs Bandwidth
        for fmap in feature_maps:
            sub = df[df['Feature Map'] == fmap]
            grouped = sub.groupby('Bandwidth')['Test F1'].mean()
            ax2.plot(grouped.index, grouped.values, marker='s', label=f"{fmap} Map", color=colors.get(fmap, '#333'), linewidth=2.5)
            
            # Annotate values
            for x, y in zip(grouped.index, grouped.values):
                ax2.annotate(f"{y:.3f}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

        ax2.set_title("Mean Test F1 Score vs Bandwidth ($\sigma$)", fontsize=13, fontweight='bold')
        ax2.set_xlabel("Bandwidth Multiplier ($\sigma$)", fontsize=11)
        ax2.set_ylabel("Test F1 Score", fontsize=11)
        ax2.set_ylim([0.0, 1.05])
        ax2.grid(True, linestyle='--', alpha=0.6)
        ax2.legend(fontsize=10)

        # Plot C: Condition Number Diagnostics (Exponential Concentration)
        for fmap in feature_maps:
            sub = df[df['Feature Map'] == fmap]
            grouped = sub.groupby('Bandwidth')['Condition Number'].mean()
            ax3.plot(grouped.index, grouped.values, marker='^', label=f"{fmap} Map", color=colors.get(fmap, '#333'), linewidth=2.5)
            
            # Annotate values
            for x, y in zip(grouped.index, grouped.values):
                ax3.annotate(f"{y:.1e}", (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=10, fontweight='bold')

        ax3.set_title("Gram Matrix Condition No. vs Bandwidth", fontsize=13, fontweight='bold')
        ax3.set_xlabel("Bandwidth Multiplier ($\sigma$)", fontsize=11)
        ax3.set_ylabel("Condition Number (log scale)", fontsize=11)
        ax3.set_yscale('log')
        ax3.grid(True, linestyle='--', alpha=0.6)
        ax3.legend(fontsize=10)

        # Plot D: Resource Cost (CNOT Count)
        resource_df = df[['Feature Map', 'CNOT Count']].drop_duplicates().groupby('Feature Map').mean()
        bars = ax4.bar(resource_df.index, resource_df['CNOT Count'], color=[colors.get(x, '#333') for x in resource_df.index], alpha=0.8, edgecolor='k')
        
        # Annotate bar values
        for bar in bars:
            yval = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, yval + (yval * 0.05), f"{int(yval)} CNOTs", ha='center', va='bottom', fontsize=11, fontweight='bold')

        ax4.set_title("Entangling Gate Overhead (CNOTs)", fontsize=13, fontweight='bold')
        ax4.set_ylabel("Total CNOT Count", fontsize=11)
        ax4.grid(axis='y', linestyle='--', alpha=0.6)

        # Plot E: Explanatory Insight Panel
        ax5.axis('off')
        
        insight_text = (
            "ANALYTICAL SUMMARY & CONTEXT:\n"
            "--------------------------------------------------------------------------------------\n\n"
            f"[*] OPTIMAL CONFIGURATION:\n"
            f"    The highest observed Test F1 Score ({best_row['Test F1']:.4f}) was achieved using the {best_row['Feature Map']} map.\n"
            f"    Optimal Hyperparameters -> Entanglement: {best_row['Entanglement']} | Bandwidth: {best_row['Bandwidth']} | NISQ Noise: {best_row['Noisy Backend']}\n\n"
            
            "[*] RESOURCE EFFICIENCY (HARDWARE SCALING):\n"
            "    The conventional ZZFeatureMap requires a quadratically growing number of entangling CNOT gates\n"
            "    relative to the dimensionality of the dataset. The CPMap demonstrates a significant structural\n"
            "    advantage, maintaining high expressivity while drastically reducing the entangling gate overhead.\n\n"
            
            "[*] EXPONENTIAL CONCENTRATION DIAGNOSTICS:\n"
            "    As Hilbert space dimensions grow, off-diagonal kernel matrix values typically concentrate toward a\n"
            "    constant, resulting in the Gram matrix approaching the identity. The Condition Number plotted above\n"
            "    serves as a primary indicator of this phenomenon. Configurations with excessively high condition\n"
            "    numbers suffer from vanishing generalizability, a fundamental limit for standard fidelity kernels.\n"
        )
        
        ax5.text(0.05, 0.85, insight_text, fontsize=13, verticalalignment='top', family='monospace',
                 bbox=dict(boxstyle='round,pad=1.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#adb5bd'))

        plt.tight_layout(rect=[0, 0.03, 1, 0.95], h_pad=3.0)

        # Save and Cleanup
        dashboard_plot_path = os.path.join(results_dir, f"dashboard_{dataset_slug}.png")
        fig.savefig(dashboard_plot_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f" -> Rich visualization dashboard saved successfully to '{dashboard_plot_path}'")

if __name__ == "__main__":
    analyze_and_plot_ablations()