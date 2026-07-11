import cupy as cp    
import numpy as np
from itertools import combinations
import qrc_circuit
import torch.nn as nn
from readout import RidgeReadout
import pennylane as qml
from copy import deepcopy
        
class QRC_Model(nn.Module):
    def __init__(self, num_qubits=0, crc_size=0, use_gpu=False, backends=None, f_bs=[0.1], b=-0.31, ridge_param=1.e-6, seed =0):
        """The Onion Classical Quantum Reservoir Computing Model
        
        HAR + QRC residual architecture (Layer 5):
            - HAR du doan phan chinh:      y_HAR (precomputed, truyen vao train/forward)
            - QRC (Head A) hoc residual:   e_{t+1} = y_{t+1} - y_HAR_{t+1}
            - Forecast cuoi (skip conn.):  y_final = y_HAR + e_hat
            - Ve volatility scale:         RV_hat = exp(y_final)
        
        Args:
            num_qubits (int): The number of qubits for the quantum reservoir.
            crc_size(int): The size of the classical reservoir.
            use_classical(bool): Whether to include a classical reservoir
            use_quantum(bool): Whether to include the quantum reservoir(s)
            backends(List(Backend)): The number of backends determines the number of quantum reservoirs
            b(float): The b parameter for the QRC circuit
            f_bs(List[Float]): The different feedback parameters for the quantum reservoir circuit
            ridge_param(float): The regularization parameter for the ridge regression
            use_gpu(bool): If true, use qiskit_aer gpu and cupy
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.crc_size = crc_size
        self.observables = self.observables_qrc_circuit()
        self.circuit = self.full_circuit()

        self.J = self.generate_J(sigma=1.0, seed=seed)
        self.use_gpu = use_gpu
        
        self.train_results = []
        self.train_outputs = []


        self.backends = backends
        self.pmc = [qml.QNode(self.circuit, backend) for backend in self.backends]

        self.last_output = [[0] for _ in range(len(self.backends))]
        
        self.ridge = RidgeReadout(n_qubits=self.num_qubits, alpha=ridge_param,
                                  pair=False, n_reservoirs=len(self.backends))

        # Generate initial states for QRC evolution
        self.init_qrc()
        
    def generate_J(self, sigma=1.0, seed=None):
        rng = np.random.default_rng(seed)

        J = rng.normal(0, sigma, size=(self.num_qubits, self.num_qubits))
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)

        return J

    def observables_qrc_circuit(self):
        """Generate the observables for a num_qubits quantum reservoir."""
        
        # First, all the single qubit Z_q terms
        observables = []

        # Single-qubit Z_i terms
        for i in range(self.num_qubits):
            observables.append(qml.PauliZ(i))

        # Pairwise Z_i Z_j terms
        for i, j in combinations(range(self.num_qubits), 2):
            observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

        # Generate integer represenation of observables for quick calculations
        self._int_observables = np.zeros(len(observables), dtype=int)

        for i in range(self.num_qubits):
            mask = 1 << i
            self._int_observables[i] = mask

        for i, j in combinations(range(self.num_qubits), 2):
            mask = (1 << i) | (1 << j)
            self._int_observables[len(self._int_observables) - 1] = mask

        return observables
    
    def full_circuit(self):
        """Return the full qrc circuit """
        return qrc_circuit(self.num_qubits)

    def calc_observables(self, samples):
        """Given the samples from the quantum reservoir, calculate all observables"""
        exps = np.zeros(len(self.observables), dtype=float)
        for i, r in enumerate(samples): #.items():
            for o, obs in enumerate(self._int_observables):
                exps[o] += (-1)**bin(obs & i).count("1")*r  # /self.n_shots
        return exps

    def init_qrc(self):
        """Initialize the quantum reservoir with Hadamard-state expectation values."""

        # For Hadamard initialization, <Z_i> = 0 for every qubit.
        self.last_output = [list(np.zeros(self.num_qubits)) for _ in range(len(self.backends))]

        self.initial_exp_values = deepcopy(self.last_output)
        
    def reset_reservoir(self):
        """Dua reservoir ve trang thai ban dau truoc moi chuoi thoi gian."""
        self.last_output = deepcopy(self.initial_exp_values)

    def evolve_qrc(self, t0):
        """Evolve the quantum reservoir given an input signal t0"""
        obsvs = []
        for c, qnode in enumerate(self.qnodes):
            previous_z = self.last_output[c]

            new_z = np.asarray(qnode(t0, previous_z), dtype=float,)

            # These n values become memory for the next time step.
            self.last_output[c] = new_z.copy()

            # These values become the QRC feature vector at this step.
            obsvs.extend(new_z.tolist())

        return obsvs
    
    def _warmup_and_evolution_series(self, ts_ith, n_washout=2):
        """Warm-up roi teacher-force reservoir qua chuoi da biet.
 
        Returns:
            features: list cua f_t, voi f_t la feature SAU khi da thay ts_ith[t]
                      (f_t dung de du doan buoc t+1).
        """
        self.reset_reservoir()
 
        # warm-up / washout: move the reservoir away from its artificial initial state
        for _ in range(n_washout):
            _ = self.evolve_qrc(ts_ith[0])
 
        features = []
        for value in ts_ith:
            features.append(self.evolve_qrc(value))
        return features
    
    
    def train(self, x, y_HAR):
        """Evolve each set of time evolutions and train on results"""
        ts_indices = x.shape[0]
        y_HAR = np.asarray(y_HAR, dtype=float)
        assert y_HAR.shape == (x.shape[0], x.shape[1] - 1), (
            f"y_HAR phải có shape (N, T-1) = {(x.shape[0], x.shape[1] - 1)}, "
            f"nhận được {y_HAR.shape}"
        )

        for idx in range(ts_indices):
            residuals = []
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            ts_ith = x[idx]
            har_ith = y_HAR[idx]
            
            # warm-up or washout period: move the reservoir away from its artificial initial state before training
            self.reset_reservoir()
            features = self._warmup_and_evolution_series(ts_ith, n_washout=2)
            # residual targets: e_{t+1} = y_{t+1} - y_HAR_{t+1}
            residuals = ts_ith[1:] - har_ith
            self.train_results += features
            self.train_outputs += residuals.tolist()
            
        self.fit()

    def fit(self):
        """Perform ridge regression on all the evolved training data"""
        obs = np.asarray(self.train_results, dtype=float)
        y = np.asarray(self.train_outputs, dtype=float)
        self.W_out = self.ridge.weight_output(obs, y_train=y)
        return self.W_out

    def forward(self, x, num_predict, y_HAR):
        """Given a set of initial evolutions, predict num_predict time steps into the future
        
        Moi buoc:
            e_hat_{t+1}  = ridge.predict(f_t)                # Head A: residual
            y_final_{t+1} = y_HAR_{t+1} + e_hat_{t+1}        # skip connection
            RV_hat_{t+1}  = exp(y_final_{t+1})               # ve volatility scale
        
        """
        ts_indices = x.shape[0]
        y_HAR = np.asarray(y_HAR, dtype=float)
        assert y_HAR.shape == (x.shape[0], num_predict)

        all_outputs = np.zeros((ts_indices, num_predict))

        for idx in range(ts_indices):
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            ts_ith = x[idx]
            har_ith = y_HAR[idx]
            
            # warm-up or washout period: move the reservoir away from its artificial initial state before training
            self.reset_reservoir()
            features = self._warmup_and_evolution_series(ts_ith, n_washout=2)
            feedbacks = features[-1]  # last feature vector after warm-up
        
            # Run prediction
            y_final_vec = []
            for step in range(num_predict):
                # Get residual 
                e_hat = self.ridge.predict(feedbacks)
                # Get final output by adding HAR prediction and residual
                y_final = e_hat + har_ith[step]
                y_final_vec.append(y_final)

                if step < num_predict - 1:
                    # feed forecast (log scale) back into the reservoir
                    feedbacks = self.evolve_qrc(y_final)  # update feedbacks for next step

            all_outputs[idx] = np.exp(np.asarray(y_final_vec, dtype=float))
            
        return all_outputs