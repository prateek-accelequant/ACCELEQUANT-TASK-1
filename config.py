"""
Module: config.py
Description: Central parameters defining execution flags, shared experimental constraints,
             and global cache directory routing.
"""
import os

# --- CENTRAL CACHE ROUTING ---
CACHE_DIR_BASE = "QKE_Cache"
CACHE_DIR_DATASETS = os.path.join(CACHE_DIR_BASE, "datasets")
CACHE_DIR_KERNELS_MAIN = os.path.join(CACHE_DIR_BASE, "kernels_main")
CACHE_DIR_KERNELS_ABLATION = os.path.join(CACHE_DIR_BASE, "kernels_ablation")

for d in [CACHE_DIR_DATASETS, CACHE_DIR_KERNELS_MAIN, CACHE_DIR_KERNELS_ABLATION]:
    os.makedirs(d, exist_ok=True)

# Global Replication Control
SEED = 42

# Model Execution Selector Switches (True = Evaluate, False = Skip)
RUN_MODELS = {
    'RBF-SVC': True,
    'From-Scratch SVM': True,
    'XGBoost': True,
    'Quantum-ZZ': True,
    'Quantum-CPMap': True
}

# Data Pipeline Parameters
QUBIT_BUDGET = 8           
N_PER_CLASS_SYNTH = 400    
SYNTH_RADII = (1.0, 2.0)   
SYNTH_NOISE = 0.08         
REAL_DATA_RATIO = 1.0      
CSV_PATH = "processed_fraud_data.csv"

# Positive Control Setup
N_ANCHORS = 30
MARGIN_PERCENTAGE = 0.1

# Quantum Execution Parameters
REPS = 1                   
ENTANGLEMENT = 'linear'    
SHOTS = 1024               
USE_NISQ_NOISE = False     

# Experimental Sweep Strategy
N_LIST = [50, 100, 200, 500]  
N_SPLITS = 5               
OUTER_SPLITS = 5           

# Hyperparameter Search Grids
SVC_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

QSVC_PARAM_GRID = {
    'C': [0.1, 1, 10, 100]
}

XGB_PARAM_GRID = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100],
    'n_jobs': [-1],
    'device': ['cuda'] # Update to 'cpu' if running on a non-CUDA instance
}