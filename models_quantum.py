"""
Module: models_quantum.py
Description: Pure Quantum Execution Engine running exclusively on AerSimulator primitives.
             Dynamically solves for optimal qubit spaces for CPMap based on incoming feature dimensions.
             Features version-safe fallback handling for modern Qiskit primitives.
"""
import numpy as np
import warnings
from qiskit import QuantumCircuit, transpile
from qiskit.circuit import ParameterVector
from qiskit.circuit.library import ZZFeatureMap
from qiskit_machine_learning.kernels import FidelityQuantumKernel
from sklearn.svm import SVC

# Suppress warnings to maintain a clean command-line interface
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
        self.shots = shots
        
        import config
        self.num_features = config.QUBIT_BUDGET
        self.num_qubits = self._calculate_optimal_qubits()
        self.feature_map = self._generate_feature_map()
        
        if USE_AER_PRIMITIVE == "V2":
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map)
        else:
            self.sampler = self._initialize_sampler_primitive()
            self.fidelity = ComputeUncompute(sampler=self.sampler)
            self.kernel = FidelityQuantumKernel(feature_map=self.feature_map, fidelity=self.fidelity)

        self._inject_progress_bar()

    def _inject_progress_bar(self, batch_size=1000):
            import types
            import numpy as np
            from tqdm import tqdm
            from concurrent.futures import ThreadPoolExecutor, as_completed
            import multiprocessing
            
            original_evaluate = self.kernel.evaluate
            
            def batched_evaluate(kernel_self, x_vec, y_vec=None):
                target_y = x_vec if y_vec is None else y_vec
                N_x = len(x_vec)
                
                K = np.zeros((N_x, len(target_y)))
                total_overlaps = N_x * len(target_y)
                
                print(f"\n   [Quantum Hardware] Routing {total_overlaps} circuit overlaps via ThreadPool (GPU-Safe)...")
                
                # Utilize all CPU cores via Threads to construct circuits simultaneously, 
                # bypassing the GIL without corrupting the shared CUDA memory context.
                max_threads = multiprocessing.cpu_count()
                
                with ThreadPoolExecutor(max_workers=max_threads) as executor:
                    futures = {}
                    for i in range(0, N_x, batch_size):
                        x_batch = x_vec[i : i + batch_size]
                        
                        # Offload the parameter binding and primitive execution to background threads
                        future = executor.submit(original_evaluate, x_vec=x_batch, y_vec=target_y)
                        futures[future] = i
                        
                    for future in tqdm(as_completed(futures), total=len(futures), desc=f"[{self.map_type} Kernel Evaluation]", leave=True):
                        i = futures[future]
                        K[i : i + batch_size, :] = future.result()
                        
                return K
                
            self.kernel.evaluate = types.MethodType(batched_evaluate, self.kernel)

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
            # Define universal multi-core CPU options to accelerate preprocessing and fallbacks
            cpu_options = {
                "max_parallel_threads": 0,       # 0 = Use all available CPU cores automatically
                "max_parallel_experiments": 0,   # 0 = Run maximum simultaneous circuits
                "max_parallel_shots": 1          # Dedicate threads to circuits, not individual shots
            }

            if USE_AER_PRIMITIVE == "V2":
                from qiskit_aer import AerSimulator
                # Merge GPU flag with CPU threading limits
                backend = AerSimulator(method="statevector", device="GPU", **cpu_options)
                return None
                
            if not self.use_noise:
                return AerSampler(
                    backend_options={"method": "statevector", "device": "GPU", **cpu_options}, 
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
                    backend_options={"noise_model": noise_model, "method": "density_matrix", "device": "GPU", **cpu_options}, 
                    options={"shots": self.shots}
                )
            else:
                return AerSampler(
                    options={
                        "backend_options": {"method": "density_matrix", "noise_model": noise_model, "device": "GPU", **cpu_options}, 
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
                is_layer_zero = True
                
                while len(active_qubits) >= 2 and feature_idx < self.num_features:
                    n_active = len(active_qubits)
                    for q in active_qubits:
                        if feature_idx < self.num_features:
                            circuit.h(q)
                            circuit.rz(x[feature_idx], q)
                            feature_idx += 1
                        
                    for i in range(0, n_active - 1, 2):
                        circuit.append(c_gate, [active_qubits[i], active_qubits[i+1]])
                    
                    if is_layer_zero:
                        for i in range(1, n_active - 1, 2):
                            circuit.append(c_gate, [active_qubits[i], active_qubits[i+1]])
                        is_layer_zero = False
                        
                    next_layer_qubits = []
                    for i in range(0, n_active - 1, 2):
                        circuit.append(p_gate, [active_qubits[i], active_qubits[i+1]])
                        next_layer_qubits.append(active_qubits[i])
                        
                    if n_active % 2 != 0:
                        next_layer_qubits.append(active_qubits[-1])
                    active_qubits = next_layer_qubits
                
                if len(active_qubits) == 1 and feature_idx < self.num_features:
                    circuit.h(active_qubits[0])
                    circuit.rz(x[feature_idx], active_qubits[0])
                    feature_idx += 1
                    
                if r < config.REPS - 1:
                    circuit.barrier()
            return circuit

    def get_resource_counts(self):
        """
        Calculates exact hardware cost by natively constructing the full 
        compute-uncompute circuit without transpiler optimization.
        """
        exact_map = self.feature_map.decompose()
        compute_uncompute_circ = exact_map.compose(exact_map.inverse())
        gate_counts = compute_uncompute_circ.count_ops()
        
        single_gates = (gate_counts.get('h', 0) + 
                        gate_counts.get('rz', 0) + 
                        gate_counts.get('ry', 0) + 
                        gate_counts.get('rx', 0))
        
        return {
            'qubits': self.num_qubits,
            'single_qubit_gates': single_gates,
            'cnot_gates': gate_counts.get('cx', 0)
        }

    def calculate_spectral_diagnostics(self, K_train):
        """Analyzes indicators of exponential kernel matrix concentration."""
        off_diag = K_train[~np.eye(len(K_train), dtype=bool)]
        conc_var = off_diag.var() if len(off_diag) > 0 else 0.0
        eigs = np.linalg.eigvalsh(K_train)
        condition_num = eigs[-1] / max(eigs[0], 1e-12)
        return {'variance': conc_var, 'spectrum': eigs, 'condition_number': condition_num}

    def fit_quantum_svc(self, K_train, y_train):
        """Trains a version-safe probability enabled precomputed SVC model."""
        clf = SVC(kernel='precomputed', C=1.0, probability=True)
        clf.fit(K_train, y_train)
        return clf