"""
Module: models_quantum.py
Description: Pure Quantum Execution Engine running exclusively on AerSimulator primitives.
             Hardware Optimized: Hands full control to C++ OpenMP for 100% CPU core saturation.
"""
import numpy as np
import warnings
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

warnings.filterwarnings('ignore')

# --- VERSION-SAFE PRIMITIVE IMPORTS ---
try:
    from qiskit_aer.primitives import Sampler as AerSampler
    from qiskit_algorithms.statevector_fidelity import ComputeUncompute
    USE_AER_PRIMITIVE = True
except ImportError:
    try:
        from qiskit.primitives import Sampler as AerSampler
        from qiskit_algorithms.statevector_fidelity import ComputeUncompute
        USE_AER_PRIMITIVE = False
    except ImportError:
        from qiskit.primitives import StatevectorSampler as AerSampler
        from qiskit.quantum_info import Statevector
        USE_AER_PRIMITIVE = "V2"

class ProductionQuantumKernelManager:
    def __init__(self, map_type='CPMap', use_noise=False, shots=1024):
        self.map_type = map_type
        self.use_noise = use_noise
        
        import config
        self.shots = config.SHOTS 
        
        self.num_features = config.QUBIT_BUDGET
        self.num_qubits = self._calculate_optimal_qubits()
        self.feature_map = self._generate_feature_map()
        
        if USE_AER_PRIMITIVE == "V2":
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        else:
            self.sampler = self._initialize_sampler_primitive()
            self.fidelity = ComputeUncompute(sampler=self.sampler)
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=self.fidelity)


    def _calculate_optimal_qubits(self):
        import config
        if self.map_type == 'ZZ':
            return self.num_features
        elif self.map_type == 'CPMap':
            for n in range(1, config.QUBIT_BUDGET + 1):
                capacity = 0
                temp_active = n
                while temp_active >= 1:
                    capacity += temp_active
                    if temp_active == 1:
                        break
                    temp_active = (temp_active + 1) // 2
                if capacity >= self.num_features:
                    return n
            return config.QUBIT_BUDGET

    def _initialize_sampler_primitive(self):
            import multiprocessing
            cores = multiprocessing.cpu_count()

            # --- HARDWARE SATURATION ENGINE ---
            # By targeting the CPU for 8-qubit circuits, we bypass PCI-E GPU latency. 
            # We force max_parallel_experiments to match total system cores, enabling 
            # C++ OpenMP to crunch the statevectors perfectly in parallel.
            cpu_options = {
                "max_parallel_threads": 0,
                "max_parallel_experiments": cores,
                "max_parallel_shots": 1,
                "batched_shots_optimization": True
            }

            # Add GPU targeting flag
            gpu_options = cpu_options.copy()
            gpu_options["device"] = "GPU"

            if not self.use_noise:
                return AerSampler(
                    backend_options={"method": "statevector", **gpu_options}, 
                    options={"shots": None}
                )

            if USE_AER_PRIMITIVE == "V2":
                from qiskit_aer import AerSimulator
                backend = AerSimulator(method="statevector", device="CPU", **cpu_options)
                return None
                
            if not self.use_noise:
                return AerSampler(
                    backend_options={"method": "statevector", "device": "CPU", **cpu_options}, 
                    options={"shots": None}
                )
                
            from qiskit_aer.noise import NoiseModel, depolarizing_error, thermal_relaxation_error
            noise_model = NoiseModel()
            p_depol = 0.02
            error_depol = depolarizing_error(p_depol, 2)
            noise_model.add_all_qubit_quantum_error(error_depol, ['cx'])
            
            t1, t2, gate_time = 50e-6, 70e-6, 35e-9
            error_thermal = thermal_relaxation_error(t1, t2, gate_time)
            noise_model.add_all_qubit_quantum_error(error_thermal, ['rz', 'x', 'h'])
            
            if USE_AER_PRIMITIVE:
                return AerSampler(
                    backend_options={"noise_model": noise_model, "method": "density_matrix", "device": "CPU", **cpu_options}, 
                    options={"shots": self.shots}
                )
            else:
                return AerSampler(
                    options={
                        "backend_options": {"method": "density_matrix", "noise_model": noise_model, "device": "CPU", **cpu_options}, 
                        "shots": self.shots
                    }
                )

    @staticmethod
    def get_custom_c_gate():
        qc = QuantumCircuit(2, name='C')
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 3, 0)
        qc.ry(np.pi / 6, 1)
        qc.cx(0, 1)
        qc.ry(-np.pi / 9, 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 2, 0)
        return qc.to_gate()

    @staticmethod
    def get_custom_p_gate():
        qc = QuantumCircuit(2, name='P')
        qc.rz(-np.pi / 2, 1)
        qc.cx(1, 0)
        qc.rz(np.pi / 7, 0)
        qc.ry(np.pi / 9, 1)
        qc.cx(0, 1)
        qc.ry(-np.pi / 7, 1)
        return qc.to_gate()

    def _generate_feature_map(self):
        import config
        if self.map_type == 'ZZ':
            return ZZFeatureMap(
                feature_dimension=self.num_features, 
                reps=config.REPS, 
                entanglement=config.ENTANGLEMENT
            )
        elif self.map_type == 'CPMap':
            x = ParameterVector('x', length=self.num_features)
            circuit = QuantumCircuit(self.num_qubits)
            
            c_gate = self.get_custom_c_gate()
            p_gate = self.get_custom_p_gate()
            
            for r in range(config.REPS):
                active_qubits = list(range(self.num_qubits))
                feature_idx = 0
                
                while len(active_qubits) >= 2 and feature_idx < self.num_features:
                    n_active = len(active_qubits)
                    
                    # 1. Feature Encoding
                    for q in active_qubits:
                        if feature_idx < self.num_features:
                            circuit.h(q)
                            circuit.rz(x[feature_idx], q)
                            feature_idx += 1
                        
                    # 2. C-Gates on Even Pairs
                    for i in range(0, n_active - 1, 2):
                        circuit.append(c_gate, [active_qubits[i], active_qubits[i+1]])
                    
                    # 3. C-Gates on Odd Pairs (Staggered Entanglement across ALL layers)
                    for i in range(1, n_active - 1, 2):
                        circuit.append(c_gate, [active_qubits[i], active_qubits[i+1]])
                        
                    # 4. P-Gates (Pooling) on Even Pairs
                    next_layer_qubits = []
                    for i in range(0, n_active - 1, 2):
                        circuit.append(p_gate, [active_qubits[i], active_qubits[i+1]])
                        next_layer_qubits.append(active_qubits[i])
                        
                    # Carry forward the last unpaired qubit if n_active is odd
                    if n_active % 2 != 0:
                        next_layer_qubits.append(active_qubits[-1])
                        
                    active_qubits = next_layer_qubits
                
                # Encode remaining feature if exactly 1 qubit is left
                if len(active_qubits) == 1 and feature_idx < self.num_features:
                    circuit.h(active_qubits[0])
                    circuit.rz(x[feature_idx], active_qubits[0])
                    feature_idx += 1
                    
                if r < config.REPS - 1:
                    circuit.barrier()
            return circuit

    def get_resource_counts(self):
        """
        Calculates exact hardware cost by decomposing custom feature map blocks 
        down to native 1-qubit and 2-qubit (CX) operations for the compute-uncompute circuit.
        """
        import config
        # Fully unroll custom gate blocks (C, P, ZZ gates) down to basis gates
        decomposed_map = self.feature_map.decompose()
        if self.map_type == 'CPMap':
            decomposed_map = decomposed_map.decompose()

        # Construct exact compute-uncompute overlap circuit: U^\dagger(x') U(x)
        overlap_circuit = decomposed_map.compose(decomposed_map.inverse())
        ops = overlap_circuit.count_ops()

        cnot_count = ops.get('cx', 0)
        sq_gates = ops.get('h', 0) + ops.get('rz', 0) + ops.get('ry', 0) + ops.get('rx', 0)

        return {
            'qubits': self.num_qubits,
            'single_qubit_gates': sq_gates,
            'cnot_gates': cnot_count,
            'total_gates_per_circuit': cnot_count + sq_gates
        }

    def calculate_spectral_diagnostics(self, K_train):
        """Analyzes indicators of exponential kernel matrix concentration."""
        off_diag = K_train[~np.eye(len(K_train), dtype=bool)]
        conc_var = off_diag.var() if len(off_diag) > 0 else 0.0
        eigs = np.linalg.eigvalsh(K_train)
        condition_num = eigs[-1] / max(eigs[0], 1e-12)
        return {'variance': conc_var, 'spectrum': eigs, 'condition_number': condition_num}

    def fit_quantum_svc(self, K_train, y_train):
        """Trains a version-safe probability enabled precomputed SVC model with uniform hyperparameter tuning."""
        import config
        grid = GridSearchCV(
            SVC(kernel='precomputed', probability=True, random_state=config.SEED), 
            config.QSVC_PARAM_GRID, 
            cv=3, 
            scoring='f1', 
            n_jobs=-1
        )
        grid.fit(K_train, y_train)
        return grid.best_estimator_