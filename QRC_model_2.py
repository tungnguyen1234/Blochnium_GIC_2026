"""
QRC_model_2.py
--------------
Direct-prediction Quantum Reservoir Computing model.

    QRC(O_t, H_t, C_t, V_t, logRV_t)  ->  logRV_{t+1}

No HAR anywhere: the ridge readout regresses reservoir features straight
onto y_{t+1}.

Changes vs the previous upload:
    - removed `import cupy` (crashes on non-GPU machines; cp was never used)
    - removed torch nn.Module base (no torch functionality used)
    - n_trotter is now forwarded to the circuit (was silently dropped,
      always falling back to the circuit default)
    - removed the leftover residual-era `_warmup_and_evolution_series`
      (referenced residuals that no longer exist in this architecture)
"""

from copy import deepcopy
from itertools import combinations

import numpy as np
import pennylane as qml
from tqdm import tqdm

import qrc_circuit_2
from QRC_readout import RidgeReadout


class QRC_Model_2:
    def __init__(self, num_qubits=5, backends=None, ridge_param=1.0e-6,
                 f_bs=(1.0,), dt=0.1, n_trotter=2, feedback_scale=1.0,
                 seed=0):
        """
        Args:
            num_qubits: reservoir qubits == number of input features per step.
            backends: list of PennyLane devices; one reservoir per device.
            ridge_param: ridge regression regularization alpha.
            f_bs: input-scaling parameter per reservoir (len == len(backends)).
            dt: Trotter step, scales the IsingZZ couplings.
            n_trotter: Trotter repetitions inside the circuit.
            feedback_scale: multiplier on arccos(prev_z) for the RX feedback
                angles. Keep at 1.0; dt-scaling collapses the reservoir.
            seed: rng seed for the coupling matrix J.
        """
        if backends is None or len(backends) == 0:
            raise ValueError("provide at least one PennyLane device in `backends`")
        if len(f_bs) != len(backends):
            raise ValueError(
                f"len(f_bs)={len(f_bs)} must equal len(backends)={len(backends)}"
            )

        self.num_qubits = num_qubits
        self.observables = self.observables_qrc_circuit()
        self.J = self.generate_J(sigma=1.0, seed=seed)

        self.train_features = []
        self.train_outputs = []

        self.backends = backends
        self.qnodes = [
            qml.QNode(
                self.full_circuit(num_qubits, self.J, dt=dt, f_b=fb,
                                  n_trotter=n_trotter,
                                  feedback_scale=feedback_scale),
                dev,
            )
            for dev, fb in zip(backends, f_bs)
        ]

        self.ridge = RidgeReadout(n_qubits=self.num_qubits, alpha=ridge_param,
                                  pair=False, n_reservoirs=len(self.backends))

        self.init_qrc()

    # -------------------------------------------------- construction
    def generate_J(self, sigma=1.0, seed=None):
        rng = np.random.default_rng(seed)
        J = rng.normal(0, sigma, size=(self.num_qubits, self.num_qubits))
        J = (J + J.T) / 2
        np.fill_diagonal(J, 0)
        return J

    def observables_qrc_circuit(self):
        """Z_i and Z_i Z_j observables + integer masks (for counts readout)."""
        observables = []
        for i in range(self.num_qubits):
            observables.append(qml.PauliZ(i))
        for i, j in combinations(range(self.num_qubits), 2):
            observables.append(qml.PauliZ(i) @ qml.PauliZ(j))

        self._int_observables = np.zeros(len(observables), dtype=int)
        for i in range(self.num_qubits):
            self._int_observables[i] = 1 << i
        for k, (i, j) in enumerate(combinations(range(self.num_qubits), 2)):
            self._int_observables[self.num_qubits + k] = (1 << i) | (1 << j)

        return observables

    def full_circuit(self, num_qubits, J, dt, f_b, n_trotter, feedback_scale):
        """Build the reservoir circuit, forwarding ALL knobs."""
        return qrc_circuit_2.qrc_circuit(
            num_qubits, J=J, dt=dt, f_b=f_b,
            n_trotter=n_trotter, feedback_scale=feedback_scale,
        )

    def calc_observables(self, counts):
        """Expectation values from an {int basis state: count} dict.

        Only needed for counts-based (hardware) readout.
        """
        exps = np.zeros(len(self.observables), dtype=float)
        total = sum(counts.values())
        for state, n in counts.items():
            for o, mask in enumerate(self._int_observables):
                exps[o] += (-1) ** bin(mask & int(state)).count("1") * n
        return exps / total

    # -------------------------------------------------- reservoir state
    def init_qrc(self):
        """Hadamard-state initialization: <Z_i> = 0 on every qubit."""
        self.last_output = [
            list(np.zeros(self.num_qubits)) for _ in range(len(self.backends))
        ]
        self.initial_exp_values = deepcopy(self.last_output)

    def reset_reservoir(self):
        """Return the reservoir to its initial state before each new series."""
        self.last_output = deepcopy(self.initial_exp_values)

    def evolve_qrc(self, v_t):
        """One reservoir step driven by the feature VECTOR v_t.

        Args:
            v_t: array of shape (num_qubits,), entries in [-1, 1].
        Returns:
            list of num_qubits * n_reservoirs concatenated <Z_i>.
        """
        v_t = np.asarray(v_t, dtype=float)
        if v_t.shape != (self.num_qubits,):
            raise ValueError(
                f"v_t must have shape ({self.num_qubits},), got {v_t.shape}"
            )

        obsvs = []
        for c, qnode in enumerate(self.qnodes):
            previous_z = self.last_output[c]
            new_z = np.asarray(qnode(v_t, previous_z), dtype=float)
            self.last_output[c] = new_z.copy()
            obsvs.extend(new_z.tolist())
        return obsvs

    # -------------------------------------------------- training
    def train(self, V, y_next):
        """Accumulate (feature, y_{t+1}) pairs across tickers.

        Args:
            V:      (N, T, num_qubits) scaled feature vectors.
            y_next: (N, T) target series; y_next[n, t] is logRV at row t+1
                    of the original dataframe.
        """
        V = np.asarray(V, float)
        y_next = np.asarray(y_next, float)
        assert V.shape[:2] == y_next.shape, (
            f"V {V.shape[:2]} and y_next {y_next.shape} must align"
        )
        assert V.shape[2] == self.num_qubits

        for idx in tqdm(range(y_next.shape[0]), desc="tickers"):
            self.reset_reservoir()
            for _ in range(2):                       # washout
                self.evolve_qrc(V[idx, 0])
            # ALIGNMENT: f_t (state after row t) predicts the value at
            # row t+1; evaluation compares against y_next[:, 1:].
            for t in range(y_next.shape[1] - 1):
                self.train_features.append(self.evolve_qrc(V[idx, t]))
                self.train_outputs.append(y_next[idx, t + 1])

    def fit(self):
        """Ridge regression on all accumulated training pairs."""
        obs = np.asarray(self.train_features, dtype=float)
        y = np.asarray(self.train_outputs, dtype=float)
        self.W_out = self.ridge.weight_output(obs, y_train=y)
        return self.W_out

    # -------------------------------------------------- prediction
    def forward_one_shot(self, V, y_next):
        """One-step-ahead direct forecasts.

        Returns preds_log of shape (N, T-1); preds_log[n, t] forecasts
        y_next[n, t+1], aligned with y_next[:, 1:].
        """
        V = np.asarray(V, float)
        y_next = np.asarray(y_next, float)
        N, T = y_next.shape
        preds_log = np.zeros((N, T - 1))

        for idx in range(N):
            self.reset_reservoir()
            for _ in range(2):                       # washout
                self.evolve_qrc(V[idx, 0])
            for t in range(T - 1):
                f_t = np.asarray(self.evolve_qrc(V[idx, t]))
                preds_log[idx, t] = float(self.ridge.predict(f_t[None, :])[0])
        return preds_log
