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
        self.pmc = []
        for b in self.backends:
            self.pmc.append(qml.QNode(self.circuit, b))

        self.last_output = [[0] for _ in range(len(self.backends))]
        
        self.ridge = RidgeReadout(n_qubits=self.num_qubits, alpha=ridge_param, pair=True)

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

    def evolve_qrc(self, t0):
        """Evolve the quantum reservoir given an input signal t0"""
        obsvs = []
        for c, qnode in enumerate(self.qnodes):
            previous_z = self.last_output[c]

            new_z = np.asarray(
                qnode(t0, previous_z),
                dtype=float,
            )

            # These n values become memory for the next time step.
            self.last_output[c] = new_z.copy()

            # These values become the QRC feature vector at this step.
            obsvs.extend(new_z.tolist())

        return obsvs
    
    
    def train(self, x):
        """Evolve each set of time evolutions and train on results"""
        ts_indices = x.shape[0]

        for idx in range(ts_indices):
            exp_values = []
            output_values = []
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            ts_ith = x[idx]
            
            # warm-up or washout period: move the reservoir away from its artificial initial state before training
            for _ in range(2):
                _ = self.evolve_qrc(ts_ith[0])

            # obtain initial reservoir state after warm-up period and the corresponding output value
            exp_values += [self.evolve_qrc(ts_ith[0])]
            output_values += [ts_ith[1]]

            # training evolution
            for value_next in ts_ith[2:]:
                exp_values += [self.evolve_qrc(output_values[-1])]
                output_values += [value_next]
            self.train_results += exp_values
            self.train_outputs += output_values
        self.fit()

    def fit(self):
        """Perform ridge regression on all the evolved training data"""
        self.ridge.weight_output(self.train_results, y_train=self.train_outputs)

    def forward(self, x, num_predict):
        """Given a set of initial evolutions, predict num_predict time steps into the future"""
        ts_indices = x.shape[0]

        all_outputs = np.zeros((ts_indices, num_predict))
        output_values = [0.]
        exp_values = []
        for idx in range(ts_indices):
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            ts_ith = x[idx]
            
            # warm-up or washout period: move the reservoir away from its artificial initial state before training
            for _ in range(2):
                _ = (self.evolve_qrc(ts_ith[0]))
            exp_values += [self.evolve_qrc(ts_ith[0])]
            output_values += [ts_ith[1]]
            
            # training evolution
            for value_next in ts_ith[2:]:
                exp_values += [self.evolve_qrc(output_values[-1])]
                output_values += [value_next]
        
            # Fix here
            qrc_outputs = [self.evolve_qrc(output_values[-1])]
            test_values = []
            
            for _ in range(num_predict-1):
                qrc_outputs += [self.evolve_qrc(test_values[-1])]
                test_values += []
            all_outputs[idx, :] = test_values
        
        return all_outputs