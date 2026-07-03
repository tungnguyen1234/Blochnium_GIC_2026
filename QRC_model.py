import cupy as cp    
import numpy as np
from itertools import combinations
import qrc_circuit
import torch.nn as nn
from sklearn.linear_model import Ridge

class QRC_Model(nn.Module):
    def __init__(self, num_qubits=0, crc_size=0, use_gpu=False, backends=None, f_bs=[0.1], b=-0.31, ridge_param=1.e-6):
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


        self.b = b
        self.use_gpu = use_gpu
        self.f_bs = f_bs
        
        self.train_results = []
        self.train_outputs = []


        self.backends = backends
        self.pms = []
        self.pmc = []
        for b in self.backends:
            pass
        self.last_output = [[0] for i in range(len(self.backends))]
        
        self.ridge = Ridge(ridge_param)

        # Generate initial states for QRC evolution
        self.init_qrc()

    def observables_qrc_circuit(self):
        """Generate the observables for a num_qubits quantum reservoir."""
        pass
    
    def full_circuit(self):
        """Return the full qrc circuit as described in arXiv:2505.22837 """
        return qrc_circuit(self.num_qubits)

    def calc_observables(self, samples):
        """Given the samples from the quantum reservoir, calculate all observables"""
        exps = np.zeros(len(self.observables), dtype=float)
        for i, r in enumerate(samples): #.items():
            for o, obs in enumerate(self._int_observables):
                pass
        return exps

    def init_qrc(self):
        """Initial state for the quantum reservoir"""
        self.last_output = [list(np.ones(self.num_qubits)*0.5) for _ in range(len(self.backends))]
        from copy import deepcopy
        self.initial_exp_values = deepcopy(self.last_output)

    def evolve_qrc(self, t0):
        """Evolve the quantum reservoir given an input signal t0"""
        obsvs = []
        #### Fill in the blanks here

        return list(np.array(obsvs))

    def combined(self, t0):
        """Evolve both quantum and classical and return combined data"""
        return self.evolve_qrc(t0)
    
    
    def train(self, x):
        """Evolve each set of time evolutions and train on results"""
        batch_size = x.shape[0]

        for b in range(batch_size):
            exp_values = []
            output_values = []
            from copy import deepcopy
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)

            for _ in range(2):
                _ = self.combined(x[b, 0])

            exp_values += [self.combined(x[b, 0])]

            output_values += [x[b, 1]]
            p = 1
            for value in x[b, 2:]:
                exp_values += [self.combined(output_values[-1])]
                output_values += [value]
                p+=1
            self.train_results += exp_values
            self.train_outputs += output_values
        self.fit()

    def fit(self):
        """Perform ridge regression on all the evolved training data"""
        cw = np.array(self.train_results)
        self.ridge.fit(cw, y=self.train_outputs)

    def forward(self, x, num_predict):
        """Given a set of initial evolutions, predict num_predict time steps into the future"""
        batch_size = x.shape[0]

        all_outputs = np.zeros((batch_size, num_predict))
        output_values = [0.]
        exp_values = []
        for b in range(batch_size):
            from copy import deepcopy
            self.last_output = deepcopy(self.initial_exp_values)
            self.x = deepcopy(self.init_xinit)
            
            for _ in range(2):
                _ = (self.combined(x[b, 0]))
            
            exp_values += [self.combined(x[b, 0])]

            output_values += [x[b, 1]]
            p = 1
            for value in x[b, 2:]:

                exp_values += [self.combined(output_values[-1])]
                p += 1
                output_values += [value]
        
            qrc_outputs = [self.combined(output_values[-1])]

            test_values = [self.ridge.predict(np.array(qrc_outputs[-1]).reshape(1, len(qrc_outputs[-1])))[0]]

            p += 1
            
            for _ in range(num_predict-1):

                qrc_outputs += [self.combined(test_values[-1])]
                test_values += [self.ridge.predict(np.array(qrc_outputs[-1]).reshape(1, len(qrc_outputs[-1])))[0]]
                
                p += 1 
            all_outputs[b, :] = test_values
        
        return all_outputs