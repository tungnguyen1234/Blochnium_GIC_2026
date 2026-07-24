"""
QRC model executed through the qBraid SDK.

Each call to evolve_qrc() submits one qBraid task per reservoir. The returned
measurement counts are converted into single-qubit Pauli-Z expectation values,
which become the recurrent reservoir state and the classical ridge features.

Required project files:
    QRC_readout.py

Required packages:
    qbraid
    qiskit
    numpy
    tqdm
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit
from tqdm import tqdm

from QRC_readout import RidgeReadout


class QRC_Model_QBraid:
    """HAR + QRC residual forecasting model using qBraid remote execution.

    Architecture:
        HAR forecast:
            y_hat_HAR[t+1]

        QRC target:
            residual[t+1] = y[t+1] - y_hat_HAR[t+1]

        Final forecast:
            y_hat_final[t+1] = y_hat_HAR[t+1] + residual_hat[t+1]

    Important:
        Every reservoir evolution is a separate qBraid task. Because the next
        circuit uses expectation values from the previous circuit, the time
        steps are sequential and cannot all be submitted in advance.
    """

    def __init__(
        self,
        num_qubits: int,
        qbraid_device: Any,
        ridge_param: float = 1.0e-6,
        f_bs: Sequence[float] = (0.1,),
        dt: float = 0.1,
        n_trotter: int = 2,
        shots: int = 100,
        seed: int = 0,
        n_washout: int = 2,
        max_tasks: int | None = None,
        print_job_ids: bool = True,
    ):
        if num_qubits < 2:
            raise ValueError("num_qubits must be at least 2.")
        if qbraid_device is None:
            raise ValueError("qbraid_device cannot be None.")
        if shots <= 0:
            raise ValueError("shots must be positive.")
        if n_trotter <= 0:
            raise ValueError("n_trotter must be positive.")
        if n_washout < 0:
            raise ValueError("n_washout cannot be negative.")
        if not f_bs:
            raise ValueError("f_bs must contain at least one feedback value.")
        if max_tasks is not None and max_tasks <= 0:
            raise ValueError("max_tasks must be positive when provided.")

        self.num_qubits = int(num_qubits)
        self.qbraid_device = qbraid_device
        self.ridge_param = float(ridge_param)
        self.f_bs = tuple(float(value) for value in f_bs)
        self.dt = float(dt)
        self.n_trotter = int(n_trotter)
        self.shots = int(shots)
        self.n_washout = int(n_washout)
        self.max_tasks = max_tasks
        self.print_job_ids = bool(print_job_ids)
        self.n_reservoirs = len(self.f_bs)

        self.J = self.generate_J(sigma=1.0, seed=seed)

        self.train_features: list[list[float]] = []
        self.train_outputs: list[float] = []
        self.W_out = None

        self.submitted_tasks = 0
        self.job_ids: list[str] = []

        self.ridge = RidgeReadout(
            n_qubits=self.num_qubits,
            alpha=self.ridge_param,
            pair=False,
            n_reservoirs=self.n_reservoirs,
        )

        self.init_qrc()

    def generate_J(self, sigma: float = 1.0, seed: int | None = None):
        """Create a fixed symmetric random Ising-coupling matrix."""
        rng = np.random.default_rng(seed)
        J = rng.normal(0.0, sigma, size=(self.num_qubits, self.num_qubits))
        J = (J + J.T) / 2.0
        np.fill_diagonal(J, 0.0)
        return J

    def init_qrc(self):
        """Initialize every reservoir with <Z_i> = 0."""
        self.last_output = [
            np.zeros(self.num_qubits, dtype=float)
            for _ in range(self.n_reservoirs)
        ]
        self.initial_exp_values = deepcopy(self.last_output)

    def reset_reservoir(self):
        """Return all reservoirs to their initial recurrent state."""
        self.last_output = deepcopy(self.initial_exp_values)

    def reset_training_data(self):
        """Clear accumulated ridge-training samples."""
        self.train_features.clear()
        self.train_outputs.clear()
        self.W_out = None

    @staticmethod
    def estimate_required_tasks(
        train_steps: int,
        test_steps: int,
        n_reservoirs: int = 1,
        n_tickers: int = 1,
        n_washout: int = 2,
    ) -> int:
        """Estimate tasks used by train() followed by forward_one_shot().

        Per ticker and reservoir:
            training = n_washout + (train_steps - 1)
            testing  = n_washout + (test_steps - 1)
        """
        if train_steps < 2 or test_steps < 2:
            raise ValueError("train_steps and test_steps must each be at least 2.")

        per_ticker_reservoir = (
            n_washout
            + train_steps
            - 1
            + n_washout
            + test_steps
            - 1
        )
        return int(n_tickers * n_reservoirs * per_ticker_reservoir)

    @staticmethod
    def estimate_credits(
        tasks: int,
        shots: int,
        per_task_credit: float,
        per_shot_credit: float,
    ) -> float:
        """Estimate total credits under a per-task plus per-shot price model."""
        return float(tasks) * (
            float(per_task_credit) + float(shots) * float(per_shot_credit)
        )

    def build_qbraid_circuit(
        self,
        t0: float,
        previous_z: Sequence[float],
        f_b: float,
    ) -> QuantumCircuit:
        """Build one measured Qiskit circuit for a QRC evolution step."""
        t0 = float(t0)
        if not np.isfinite(t0):
            raise ValueError(f"Input t0 must be finite; received {t0!r}.")

        previous_z_array = np.asarray(previous_z, dtype=float)
        if previous_z_array.shape != (self.num_qubits,):
            raise ValueError(
                "previous_z must have shape "
                f"({self.num_qubits},), received {previous_z_array.shape}."
            )

        previous_z_array = np.clip(previous_z_array, -1.0, 1.0)
        feedback_angles = np.arccos(previous_z_array) * self.dt

        circuit = QuantumCircuit(self.num_qubits, self.num_qubits)

        # Current residual input on qubit 0.
        input_angle = float(f_b) * self.dt * np.arctan(t0)
        circuit.ry(float(input_angle), 0)

        # Recurrent memory from the previous measured Z expectations.
        for qubit in range(1, self.num_qubits):
            circuit.ry(float(feedback_angles[qubit - 1]), qubit)

        # Fixed transverse-field Ising reservoir evolution.
        for _ in range(self.n_trotter):
            for i in range(self.num_qubits):
                for j in range(i + 1, self.num_qubits):
                    coupling = float(self.J[i, j])
                    if coupling != 0.0:
                        circuit.rzz(2.0 * coupling * self.dt, i, j)

            for qubit in range(self.num_qubits):
                circuit.rx(2.0 * float(feedback_angles[qubit]), qubit)

        # Explicit q[i] -> c[i] mapping. In Qiskit count strings, c[0] is
        # the rightmost bit, which is handled by counts_to_z_expectations().
        circuit.measure(
            list(range(self.num_qubits)),
            list(range(self.num_qubits)),
        )
        return circuit

    # @staticmethod
    # def _normalize_counts(raw_counts: Any) -> Mapping[str, int]:
    #     """Validate and normalize a single-circuit counts payload."""
    #     counts = raw_counts

    #     # Some result containers can wrap a single dictionary in a list.
    #     if isinstance(counts, list):
    #         if len(counts) != 1:
    #             raise ValueError(
    #                 "Expected counts for one circuit, but received "
    #                 f"{len(counts)} count dictionaries."
    #             )
    #         counts = counts[0]

    #     if not isinstance(counts, Mapping):
    #         raise TypeError(
    #             "Expected qBraid result.data.get_counts() to return a mapping; "
    #             f"received {type(counts).__name__}."
    #         )

    #     normalized: dict[str, int] = {}
    #     for key, value in counts.items():
    #         normalized[str(key).replace(" ", "")] = int(value)

    #     if not normalized:
    #         raise ValueError("qBraid returned an empty counts dictionary.")

    #     return normalized
    
    @staticmethod
    def _extract_counts(result):
        """Extract counts from qBraid results across provider formats."""

        data = result.data

        # Standard qBraid interface
        try:
            counts = data.get_counts()
            if counts:
                return QRC_Model_QBraid._normalize_counts(counts)
        except ValueError:
            pass

        # Direct measurement_counts attribute
        measurement_counts = getattr(data, "measurement_counts", None)

        if measurement_counts:
            return QRC_Model_QBraid._normalize_counts(
                measurement_counts
            )

        # Construct counts from shot-by-shot measurements
        measurements = getattr(data, "measurements", None)

        if measurements is not None:
            measurements = np.asarray(measurements)

            if measurements.size > 0:
                counts = {}

                for shot in measurements:
                    # Reverse because qBraid/Qiskit count keys are usually
                    # displayed with the highest classical bit first.
                    bitstring = "".join(
                        str(int(bit)) for bit in shot[::-1]
                    )
                    counts[bitstring] = counts.get(bitstring, 0) + 1

                return counts

        # Useful diagnostic information if the provider returned no samples
        available = {
            name: getattr(data, name, None)
            for name in (
                "measurement_counts",
                "measurements",
                "measurement_probabilities",
                "probabilities",
            )
        }

        raise RuntimeError(
            "The quantum job completed, but no measurement samples "
            "were returned. Ensure the circuit contains final measurements. "
            f"Result data type: {type(data).__name__}; "
            f"available fields: {available}"
        )

    @staticmethod
    def counts_to_z_expectations(
        counts: Mapping[str, int],
        num_qubits: int,
    ):
        """Convert bitstring counts into <Z_i> for every qubit.

        A measured 0 contributes +1 and a measured 1 contributes -1.
        Qiskit places qubit/classical bit 0 at the right side of a count key.
        """
        total_shots = int(sum(int(value) for value in counts.values()))
        if total_shots <= 0:
            raise ValueError("The total number of returned shots must be positive.")

        z_values = np.zeros(num_qubits, dtype=float)

        for raw_bitstring, count in counts.items():
            bitstring = str(raw_bitstring).replace(" ", "")

            if bitstring.startswith("0x"):
                decimal_value = int(bitstring, 16)
                bitstring = format(decimal_value, f"0{num_qubits}b")
            elif bitstring.startswith("0b"):
                decimal_value = int(bitstring, 2)
                bitstring = format(decimal_value, f"0{num_qubits}b")
            else:
                bitstring = bitstring.zfill(num_qubits)

            if len(bitstring) != num_qubits or any(bit not in "01" for bit in bitstring):
                raise ValueError(
                    f"Unexpected measurement key {raw_bitstring!r} for "
                    f"{num_qubits} qubits."
                )

            for qubit in range(num_qubits):
                bit = bitstring[-1 - qubit]
                z_values[qubit] += int(count) if bit == "0" else -int(count)

        return z_values / float(total_shots)

    def _reserve_task(self):
        """Prevent submission after the configured task cap is reached."""
        if self.max_tasks is not None and self.submitted_tasks >= self.max_tasks:
            raise RuntimeError(
                "Task limit reached before submission: "
                f"{self.submitted_tasks}/{self.max_tasks}. "
                "Increase max_tasks only after checking the credit estimate."
            )

    def _run_qbraid_circuit(self, circuit: QuantumCircuit):
        """Submit one qBraid task and return its Z expectation vector."""
        self._reserve_task()

        job = self.qbraid_device.run(circuit, shots=self.shots)

        # Count the task immediately after successful submission. A failed
        # execution may still be billable depending on the selected provider.
        self.submitted_tasks += 1

        job_id = str(getattr(job, "id", "unknown"))
        self.job_ids.append(job_id)

        if self.print_job_ids:
            print(
                f"Submitted qBraid task {self.submitted_tasks}"
                + (
                    f"/{self.max_tasks}" if self.max_tasks is not None else ""
                )
                + f": {job_id}"
            )

        result = job.result()
        counts = self._extract_counts(result)
        return self.counts_to_z_expectations(counts, self.num_qubits)

    def evolve_qrc(self, t0: float) -> list[float]:
        """Advance all reservoirs by one recurrent time step."""
        features: list[float] = []

        for reservoir_index, f_b in enumerate(self.f_bs):
            previous_z = self.last_output[reservoir_index]
            circuit = self.build_qbraid_circuit(t0, previous_z, f_b)
            new_z = self._run_qbraid_circuit(circuit)

            self.last_output[reservoir_index] = new_z.copy()
            features.extend(new_z.tolist())

        return features

    def _warmup_and_evolution_series(
        self,
        residuals_ticker: Sequence[float],
    ) -> tuple[list[list[float]], list[float]]:
        """Teacher-force a continuous residual sequence through the reservoir."""
        residuals = np.asarray(residuals_ticker, dtype=float)
        if residuals.ndim != 1 or len(residuals) < 2:
            raise ValueError("Each residual series must be one-dimensional with length >= 2.")

        self.reset_reservoir()

        for _ in range(self.n_washout):
            self.evolve_qrc(float(residuals[0]))

        features: list[list[float]] = []
        targets: list[float] = []

        for t in tqdm(
            range(len(residuals) - 1),
            desc="QRC train evolution",
            leave=False,
        ):
            features.append(self.evolve_qrc(float(residuals[t])))
            targets.append(float(residuals[t + 1]))

        return features, targets

    def train(self, x, y_HAR):
        """Create QRC features and residual targets for every ticker."""
        x_array = np.asarray(x, dtype=float)
        y_har_array = np.asarray(y_HAR, dtype=float)

        if x_array.ndim != 2:
            raise ValueError(f"x must have shape (N, T), received {x_array.shape}.")
        if y_har_array.shape != x_array.shape:
            raise ValueError(
                f"y_HAR must match x shape {x_array.shape}; "
                f"received {y_har_array.shape}."
            )

        for ticker_index in tqdm(
            range(x_array.shape[0]),
            desc="QRC training tickers",
        ):
            residuals = x_array[ticker_index] - y_har_array[ticker_index]
            features, targets = self._warmup_and_evolution_series(residuals)
            self.train_features.extend(features)
            self.train_outputs.extend(targets)
            print(f"Remote QRC sequence complete for ticker index {ticker_index}.")

    def fit(self):
        """Fit the classical ridge readout to all accumulated QRC features."""
        if not self.train_features:
            raise RuntimeError("No training features exist. Call train() before fit().")

        observations = np.asarray(self.train_features, dtype=float)
        targets = np.asarray(self.train_outputs, dtype=float)

        self.W_out = self.ridge.weight_output(observations, y_train=targets)
        return self.W_out

    def _predict_residual(self, feature_vector: Sequence[float]) -> float:
        if not hasattr(self.ridge, "ridge_model"):
            raise RuntimeError("Ridge model is not fitted. Call fit() first.")

        feature = np.asarray(feature_vector, dtype=float).reshape(1, -1)
        return float(self.ridge.ridge_model.predict(feature)[0])

    def forward_one_shot(self, x, y_HAR):
        """Produce teacher-forced one-step-ahead HAR + QRC forecasts."""
        x_array = np.asarray(x, dtype=float)
        y_har_array = np.asarray(y_HAR, dtype=float)

        if x_array.ndim != 2:
            raise ValueError(f"x must have shape (N, T), received {x_array.shape}.")
        if y_har_array.shape != x_array.shape:
            raise ValueError(
                f"y_HAR must match x shape {x_array.shape}; "
                f"received {y_har_array.shape}."
            )

        n_tickers, n_steps = x_array.shape
        if n_steps < 2:
            raise ValueError("At least two test steps are required.")

        predictions = np.zeros((n_tickers, n_steps - 1), dtype=float)

        for ticker_index in tqdm(
            range(n_tickers),
            desc="QRC testing tickers",
        ):
            residuals = x_array[ticker_index] - y_har_array[ticker_index]
            self.reset_reservoir()

            for _ in range(self.n_washout):
                self.evolve_qrc(float(residuals[0]))

            for t in tqdm(
                range(n_steps - 1),
                desc=f"QRC test evolution {ticker_index}",
                leave=False,
            ):
                feature_t = self.evolve_qrc(float(residuals[t]))
                residual_hat = self._predict_residual(feature_t)

                predictions[ticker_index, t] = (
                    y_har_array[ticker_index, t + 1] + residual_hat
                )

        return predictions

    def forward_multi_shot(
        self,
        x,
        num_predict: int,
        y_HAR,
    ):
        """Autoregressively feed predicted residuals back into the reservoir.

        This method requires externally supplied HAR forecasts for each requested
        future step. The returned values remain on the log-volatility scale.
        """
        x_array = np.asarray(x, dtype=float)
        y_har_array = np.asarray(y_HAR, dtype=float)

        if x_array.ndim != 2:
            raise ValueError(f"x must have shape (N, T), received {x_array.shape}.")
        if y_har_array.shape != x_array.shape:
            raise ValueError(
                f"y_HAR must match x shape {x_array.shape}; "
                f"received {y_har_array.shape}."
            )
        if not 1 <= num_predict <= x_array.shape[1]:
            raise ValueError(
                f"num_predict must be in [1, {x_array.shape[1]}]."
            )

        outputs = np.zeros((x_array.shape[0], num_predict), dtype=float)

        for ticker_index in range(x_array.shape[0]):
            residuals = x_array[ticker_index] - y_har_array[ticker_index]
            self._warmup_and_evolution_series(residuals)

            # _warmup_and_evolution_series ends after residuals[-2]. Evolve
            # the final observed scalar residual once to obtain the feature
            # used for the first genuinely future prediction.
            feature = np.asarray(
                self.evolve_qrc(float(residuals[-1])),
                dtype=float,
            )

            for step in range(num_predict):
                residual_hat = self._predict_residual(feature)
                outputs[ticker_index, step] = (
                    y_har_array[ticker_index, step] + residual_hat
                )

                if step < num_predict - 1:
                    feature = np.asarray(
                        self.evolve_qrc(residual_hat),
                        dtype=float,
                    )

        return outputs
