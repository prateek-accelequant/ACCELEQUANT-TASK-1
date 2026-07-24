"""
Module: manager.py
Description: The single entry point for the QKE ecosystem. 
             Provides a unified interface to trigger specific data pipelines and 
             orchestrate experiments centrally.
"""
import os
import sys
import config
from data_pipeline import QKEDataPipeline
from models_quantum import ProductionQuantumKernelManager
from experiment_runner import run_sample_efficiency_suite, run_ablation_matrix_suite

def main():
    print("\n" + "="*60)
    print("   Quantum Kernel Evaluation - Master Orchestrator   ")
    print("="*60)
    
    try:
        _user_input = input("Enter the number of parallel workers to use [Default: 4]: ").strip()
        N_CORES = int(_user_input) if _user_input else 4
    except ValueError:
        print("[!] Invalid input. Defaulting to 4 workers for stability.")
        N_CORES = 4

    os.environ['OMP_NUM_THREADS'] = str(N_CORES)
    os.environ['RAY_NUM_THREADS'] = str(N_CORES)
    
    print("\n[1] Run Synthetic Data Suite (Primary & Controls)")
    print("[2] Run Real Fraud Data Suite (Secondary)")
    print("[3] Run Both")
    choice = input("Select execution path (1/2/3): ").strip()
    
    if choice not in ['1', '2', '3']:
        print("[!] Invalid selection. Exiting.")
        sys.exit(1)
        
    pipeline = QKEDataPipeline(seed=config.SEED)
    
    if choice in ['1', '3']:
        print("\n" + "="*45)
        print("   SYNTHETIC DATA PIPELINE INITIATED   ")
        print("="*45)
        X_synth_raw, y_synth = pipeline.generate_synthetic_shells()
        X_synth = pipeline.preprocess(X_synth_raw)
        
        zz_mgr = ProductionQuantumKernelManager(map_type='ZZ')
        cpmap_mgr = ProductionQuantumKernelManager(map_type='CPMap')
        
        print("   [Master Orchestrator] Generating quantum-native baselines...")
        X_qnative_ZZ, y_qnative_ZZ = pipeline.generate_quantum_native(X_synth, zz_mgr.kernel, map_name="ZZ")
        X_qnative_CPMap, y_qnative_CPMap = pipeline.generate_quantum_native(X_synth, cpmap_mgr.kernel, map_name="CPMap")
        
        datasets = [
            ("Primary Synthetic (Shells)", X_synth, y_synth),
            ("Positive Control (ZZ-Generated)", X_qnative_ZZ, y_qnative_ZZ),
            ("Positive Control (CPMap-Generated)", X_qnative_CPMap, y_qnative_CPMap)
        ]
        
        for name, X_d, y_d in datasets:
            run_sample_efficiency_suite(X_d, y_d, name, is_real_data=False, n_cores=N_CORES)
            run_ablation_matrix_suite(X_d, y_d, name, is_real_data=False, n_cores=N_CORES)
            
    if choice in ['2', '3']:
        print("\n" + "="*45)
        print("   REAL DATA PIPELINE INITIATED        ")
        print("="*45)
        try:
            X_real_raw, y_real = pipeline.load_and_rebalance_real_data()
            X_real = pipeline.preprocess(X_real_raw)
            run_sample_efficiency_suite(X_real, y_real, "Rebalanced Real Data", is_real_data=True, n_cores=N_CORES)
            run_ablation_matrix_suite(X_real, y_real, "Rebalanced Real Data", is_real_data=True, n_cores=N_CORES)
        except FileNotFoundError:
            print(f"\n[!] Error: '{config.CSV_PATH}' not found in runtime space. Skipping real dataset benchmarks.")
            
    print("\n[Master Orchestrator] All requested pipelines successfully finalized.")

if __name__ == "__main__":
    main()