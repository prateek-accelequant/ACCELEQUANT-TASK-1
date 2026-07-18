"""
Module: config.py
Description: Central parameters defining execution flags and shared experimental constraints.
"""
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
QUBIT_BUDGET = 8           # Maximum qubits upper bound. Data PCA targets exactly this many features.
N_PER_CLASS_SYNTH = 400    # Synthetic dataset boundaries
SYNTH_RADII = (1.0, 2.0)   # Concentric ring boundaries
SYNTH_NOISE = 0.08         # Random perturbation width
REAL_DATA_RATIO = 1.0      # Balance mapping ratio for fraud inputs
CSV_PATH = "processed_fraud_data.csv"

# Positive Control Setup
N_ANCHORS = 30
MARGIN_PERCENTAGE = 0.1

# Quantum Execution Parameters
REPS = 1                   # Number of feature-map repetitions down the timeline
ENTANGLEMENT = 'linear'    # ZZ Map connection topology ('linear' or 'full')
SHOTS = 1024               # Sample iterations per circuit evaluation
USE_NISQ_NOISE = False     # Toggle between clean Aer vs Noisy Aer models

# Experimental Sweep Strategy
N_LIST = [50, 100, 200, 500]  # Sample budget sweep sizes
N_SPLITS = 5               # Resampling iterations per sweep budget
OUTER_SPLITS = 5

# Hyperparameter Search Grids
SVC_PARAM_GRID = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 'auto', 0.01, 0.1]
}

XGB_PARAM_GRID = {
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'n_estimators': [50, 100],
    'n_jobs': [-1],
    'device': ['cuda']
}