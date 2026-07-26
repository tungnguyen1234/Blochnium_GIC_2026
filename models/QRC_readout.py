from itertools import combinations

import numpy as np
from sklearn.linear_model import Ridge

__all__ = ["RidgeReadout"]


class RidgeReadout:

    def __init__(self, n_qubits: int, alpha: float = 1e-6, pair: bool = False,
                 n_reservoirs: int = 1, **ridge_kwargs):

        if n_qubits < 1:
            raise ValueError("n_qubits must be >= 1")

        self.n_qubits = n_qubits
        self.n_single = n_qubits
        self.n_pair = n_qubits * (n_qubits - 1) // 2
        self.n_reservoirs = n_reservoirs
 
        per_reservoir = self.n_single + (self.n_pair if pair else 0)
        self.in_dim = per_reservoir * n_reservoirs
        self.ridge_model = Ridge(alpha=alpha, **ridge_kwargs)

    def _check_dim(self, obs: np.ndarray) -> None:
        if obs.shape[-1] != self.in_dim:
            raise ValueError(
                f"Expected {self.in_dim} features for {self.n_qubits} qubits "
                f"(including {self.n_single} <Zi> + {self.n_pair} <ZiZj>), "
                f"but obs has {obs.shape[-1]} features."
            )

    def weight_output(self, obs, y_train):
        """
        Args:
            obs_train: array-like [T, in_dim]
            y_train: array-like [T] (or [T, out_dim], currently unused)
        Returns:
            Ridge regression weights, shape = (in_dim,)
        """
        y_train = np.asarray(y_train, dtype=float).ravel()
        obs = np.asarray(obs)
        assert obs.shape[0] == y_train.shape[0]
        self._check_dim(obs)
        self.ridge_model.fit(obs, y_train)
        self._fitted = True
        return self.ridge_model.coef_ # W_out has dim 1 x d
    
    def predict(self, obs):
        """
        Args:
            obs: [in_dim] for one step, or [N, in_dim] for a batch.
        Returns:
            A scalar predicted residual for one input step, or an np.ndarray [N] for a batch.
        """
        if not self._fitted:
            raise RuntimeError("Ridge readout has not been fitted. Call weight_output() first.")
        obs = np.asarray(obs)
        single_step = obs.ndim == 1 # Whether it is only 1 vector
        if single_step:
            obs = obs.reshape(1, -1)
        else:
            assert obs.ndim == 2
        self._check_dim(obs)
        residual = self.ridge_model.predict(obs)
        if single_step:
            return residual[0]
        return residual
        
        

    def split_features(self, obs):
        """Split obs into separate <Zi> and <ZiZj> components for analysis/debugging."""
        obs = np.asarray(obs)
        self._check_dim(obs)
        z = obs[..., : self.n_single]
        zz = obs[..., self.n_single :]
        return z, zz

    def feature_names(self):
        names = [f"Z{i}" for i in range(self.n_qubits)]
        names += [f"Z{i}Z{j}" for i, j in combinations(range(self.n_qubits), 2)]
        return names

    def __repr__(self):
        return (
            f"RidgeReadout(n_qubits={self.n_qubits}, "
            f"in_dim={self.in_dim}, alpha={self.ridge_model.alpha})"
        )


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n_qubits = 6
    T_horizon = 20
    
    readout = RidgeReadout(n_qubits=n_qubits, alpha=1e-6, pair = False)
    print(readout)
    print("feature_names:", readout.feature_names())

    # Feature matrix produced by reservoir computing, with shape T x d
    obs = rng.uniform(-1, 1, size=(T_horizon, readout.in_dim)) 
    # Target values across the time horizon
    y_train = rng.uniform(-1, 1, size=T_horizon) 
    # HAR forecast (precomputed)
    y_HAR = y_train + rng.normal(0, 0.1, T_horizon)   
    
    # Head A learns the residual rather than predicting y directly 
    residuals = y_train - y_HAR
    W_out = readout.weight_output(obs, residuals)
    print("W_out:", W_out)
    print("Residual at time t is:", readout.predict(obs))
