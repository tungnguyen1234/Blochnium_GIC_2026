"""QRC circuit and measurement helpers used by the transition workflow."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from qiskit import QuantumCircuit


def generate_j(num_qubits: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    j_mat = rng.normal(0.0, 1.0, size=(num_qubits, num_qubits))
    j_mat = (j_mat + j_mat.T) / 2.0
    np.fill_diagonal(j_mat, 0.0)
    return j_mat


def build_qrc_circuit(
    num_qubits: int,
    j_mat: np.ndarray,
    t0: float,
    previous_z: Sequence[float],
    dt: float,
    f_b: float,
    n_trotter: int,
    input_scale: float = 1.0,
    feedback_scale: float | None = None,
) -> QuantumCircuit:
    previous_z_array = np.clip(np.asarray(previous_z, dtype=float), -1.0, 1.0)
    if feedback_scale is None:
        feedback_scale = dt
    feedback_angles = np.arccos(previous_z_array) * feedback_scale

    circuit = QuantumCircuit(num_qubits, num_qubits)
    circuit.ry(float(input_scale * f_b * np.arctan(float(t0))), 0)

    for qubit in range(1, num_qubits):
        circuit.ry(float(feedback_angles[qubit - 1]), qubit)

    for _ in range(n_trotter):
        for i in range(num_qubits):
            for j in range(i + 1, num_qubits):
                coupling = float(j_mat[i, j])
                if coupling != 0.0:
                    circuit.rzz(2.0 * coupling * dt, i, j)
        for qubit in range(num_qubits):
            circuit.rx(2.0 * float(feedback_angles[qubit]), qubit)

    circuit.measure(list(range(num_qubits)), list(range(num_qubits)))
    return circuit


def normalize_counts(raw_counts: Any) -> Mapping[str, int]:
    counts = raw_counts
    if isinstance(counts, list):
        if len(counts) != 1:
            raise ValueError(f"Expected one counts dict, got {len(counts)}.")
        counts = counts[0]
    if not isinstance(counts, Mapping):
        raise TypeError(f"Expected counts mapping, got {type(counts).__name__}.")

    normalized = {}
    for key, value in counts.items():
        normalized[str(key).replace(" ", "")] = int(value)
    if not normalized:
        raise ValueError("qBraid returned empty counts.")
    return normalized


def extract_counts(result):
    data = result.data
    try:
        counts = data.get_counts()
        if counts:
            return normalize_counts(counts)
    except Exception:
        pass

    measurement_counts = getattr(data, "measurement_counts", None)
    if measurement_counts:
        return normalize_counts(measurement_counts)

    measurements = getattr(data, "measurements", None)
    if measurements is not None:
        measurements = np.asarray(measurements)
        if measurements.size > 0:
            counts = {}
            for shot in measurements:
                bitstring = "".join(str(int(bit)) for bit in shot[::-1])
                counts[bitstring] = counts.get(bitstring, 0) + 1
            return counts

    probabilities = getattr(data, "measurement_probabilities", None)
    if probabilities:
        return normalize_counts(
            {key: round(float(value) * 1000000) for key, value in probabilities.items()}
        )

    raise RuntimeError(f"No counts-like measurement data returned: {type(data).__name__}")


def counts_to_z(counts: Mapping[str, int], num_qubits: int) -> np.ndarray:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        raise ValueError("Total counts must be positive.")

    z_values = np.zeros(num_qubits, dtype=float)
    for raw_bitstring, count in counts.items():
        bitstring = str(raw_bitstring).replace(" ", "")
        if bitstring.startswith("0x"):
            bitstring = format(int(bitstring, 16), f"0{num_qubits}b")
        elif bitstring.startswith("0b"):
            bitstring = format(int(bitstring, 2), f"0{num_qubits}b")
        else:
            bitstring = bitstring.zfill(num_qubits)

        for qubit in range(num_qubits):
            bit = bitstring[-1 - qubit]
            z_values[qubit] += int(count) if bit == "0" else -int(count)

    return z_values / float(total)


def local_z_expectations(
    num_qubits: int,
    j_mat: np.ndarray,
    t0: float,
    previous_z: Sequence[float],
    dt: float,
    f_b: float,
    n_trotter: int,
    input_scale: float = 1.0,
    feedback_scale: float | None = None,
) -> np.ndarray:
    import pennylane as qml

    dev = qml.device("default.qubit", wires=num_qubits)

    @qml.qnode(dev)
    def circuit():
        previous_z_array = np.clip(np.asarray(previous_z, dtype=float), -1.0, 1.0)
        scale = dt if feedback_scale is None else feedback_scale
        feedback_angles = np.arccos(previous_z_array) * scale
        qml.RY(float(input_scale * f_b * np.arctan(float(t0))), wires=0)

        for qubit in range(1, num_qubits):
            qml.RY(float(feedback_angles[qubit - 1]), wires=qubit)

        for _ in range(n_trotter):
            for i in range(num_qubits):
                for j in range(i + 1, num_qubits):
                    coupling = float(j_mat[i, j])
                    if coupling != 0.0:
                        qml.IsingZZ(2.0 * coupling * dt, wires=[i, j])
            for qubit in range(num_qubits):
                qml.RX(2.0 * float(feedback_angles[qubit]), wires=qubit)

        return [qml.expval(qml.PauliZ(qubit)) for qubit in range(num_qubits)]

    return np.asarray(circuit(), dtype=float)
