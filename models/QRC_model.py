import numpy as np
from itertools import combinations
import models.qrc_circuit as qrc_circuit
import torch.nn as nn
from models.QRC_readout import RidgeReadout
import pennylane as qml
from copy import deepcopy
from tqdm import tqdm
        
class QRC_Model(nn.Module):
    def __init__(self, num_qubits=0, backends=None, ridge_param=1.e-6, f_bs=(0.1,), dt=0.1,  seed=0):
        """The Quantum Reservoir Computing Model
        
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
            dt(float): The time step for the QRC circuit
            f_bs(List[Float]): The different feedback parameters for the quantum reservoir circuit
            ridge_param(float): The regularization parameter for the ridge regression
            use_gpu(bool): If true, use qiskit_aer gpu and cupy
        """
        super().__init__()
        self.num_qubits = num_qubits
        self.observables = self.observables_qrc_circuit()
        self.J = self.generate_J(sigma=1.0, seed=seed)
        
        self.train_features = []
        self.train_outputs = []


        self.backends = backends
        self.qnodes = [
            qml.QNode(self.full_circuit(num_qubits, self.J, dt=dt, f_b=fb), dev)
            for dev, fb in zip(backends, f_bs)
        ]
        
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

        for k, (i, j) in enumerate(combinations(range(self.num_qubits), 2)):
            self._int_observables[self.num_qubits + k] = (1 << i) | (1 << j)

        return observables
    
    def full_circuit(self, num_qubits, J, dt, f_b):
        """Return the full qrc circuit """
        return qrc_circuit.qrc_circuit(num_qubits, J=J, dt=dt, f_b=f_b)

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
    
    def _warmup_and_evolution_series(self, residuals_ticker, n_washout=2):
        """Warm-up roi teacher-force reservoir qua residual.
 
        Returns:
            features: list cua f_t, voi f_t la feature SAU khi da thay ts_ticker[t]
                      (f_t dung de du doan buoc t+1).
        """
        self.reset_reservoir()
 
        # warm-up or washout: move the reservoir away from its artificial initial state
        for _ in range(n_washout):
            _ = self.evolve_qrc(residuals_ticker[0])
 
        features = []
        targets = []
        for t in tqdm(range(len(residuals_ticker) - 1)):
            features.append(self.evolve_qrc(residuals_ticker[t]))
            targets.append(residuals_ticker[t+1])
        return features, targets
    
    
    def train(self, x, y_HAR):
        """Evolve each set of time evolutions and train on results"""
        ts_tickers = x.shape[0]
        y_HAR = np.asarray(y_HAR, dtype=float)
        assert y_HAR.shape == x.shape, (
            f"y_HAR phải có shape (N, T) = {(x.shape[0], x.shape[1])}, "
            f"nhận được {y_HAR.shape}"
        )

        for idx in tqdm(range(ts_tickers)):
            ts_ticker = x[idx]
            har_ticker = y_HAR[idx]
            # features from time 1 to T
            residuals_ticker = ts_ticker - har_ticker
            residual_features, residuals_targets = self._warmup_and_evolution_series(residuals_ticker, n_washout=2)
            # residual targets: e_{t+1} = y_{t+1} - y_HAR_{t+1}
            self.train_features += residual_features
            self.train_outputs += residuals_targets
            print("Session complete for ticker", idx)

    def fit(self):
        """Perform ridge regression on all the evolved training data"""
        obs = np.asarray(self.train_features, dtype=float)
        y = np.asarray(self.train_outputs, dtype=float)
        self.W_out = self.ridge.weight_output(obs, y_train=y)
        return self.W_out
    
    
    def forward_one_shot(self, x, y_HAR):
        """
        One-step-ahead over the whole series. x, y_HAR: (N, T), same-index.
        Returns preds_log: (N, T-1) where preds_log[n, t] is the forecast of x[n, t+1],
        aligned with x[:, 1:] and y_HAR[:, 1:].  LOG scale — caller exps.
        """
        x = np.asarray(x, float); y_HAR = np.asarray(y_HAR, float)
        assert y_HAR.shape == x.shape
        N, T = x.shape
        preds_log = np.zeros((N, T - 1))

        for idx in range(N):
            residuals = x[idx] - y_HAR[idx]
            self.reset_reservoir()
            # washout
            for _ in range(2):                                   
                self.evolve_qrc(residuals[0])
            for t in range(T - 1):
                # state after e_t
                f_t = np.asarray(self.evolve_qrc(residuals[t]))  
                e_hat = float(self.ridge.ridge_model.predict(f_t[None, :])[0])
                # skip conn: HAR forecast OF t+1
                preds_log[idx, t] = y_HAR[idx, t + 1] + e_hat    
        return preds_log

    def forward_multi_shot(self, x, num_predict, y_HAR):
        """Given a set of initial evolutions, predict num_predict time steps into the future
        
        Moi buoc:
            e_hat_{t+1}  = ridge.predict(f_t)                # Head A: residual
            y_final_{t+1} = y_HAR_{t+1} + e_hat_{t+1}        # skip connection
            RV_hat_{t+1}  = exp(y_final_{t+1})               # ve volatility scale
        
        """
        ts_tickers = x.shape[0]
        y_HAR = np.asarray(y_HAR, dtype=float)
        assert y_HAR.shape == x.shape, (
            f"y_HAR phải có shape (N, T) = {(x.shape[0], x.shape[1])}, "
            f"nhận được {y_HAR.shape}"
        )

        RV_outputs = np.zeros((ts_tickers, num_predict))

        for idx in range(ts_tickers):
            ts_ticker = x[idx]
            har_ticker = y_HAR[idx]
            residuals_ticker = ts_ticker - har_ticker
            
            # warm-up or washout period: move the reservoir away from its artificial initial state before training
            residual_features, _ = self._warmup_and_evolution_series(residuals_ticker, n_washout=2)
            feedbacks = np.asarray(self.evolve_qrc(residual_features[-1]))  # last feature vector after warm-up
        
            # Run prediction
            y_final_vec = []
            for step in range(num_predict):
                # Get residual 
                e_hat = float(self.ridge.predict(feedbacks[None, :])[0])
                # Get final output by adding HAR prediction and residual
                y_final = e_hat + har_ticker[step]
                y_final_vec.append(y_final)

                if step < num_predict - 1:
                    # feed residues back into the reservoir
                    feedbacks = np.asarray(self.evolve_qrc(e_hat))  # update feedbacks for next step

            RV_outputs[idx] = np.asarray(y_final_vec, dtype=float)
            
        return RV_outputs