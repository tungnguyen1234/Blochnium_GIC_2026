import warnings

import numpy as np
import pennylane as qml

warnings.filterwarnings("ignore", category=DeprecationWarning)


def qrc_circuit(num_qubits, J, dt=0.1, f_b=0.1, n_trotter=2, feedback_scale=None):
    """Build the QRC reservoir circuit function.

    Args:
        num_qubits (int): number of reservoir qubits.
        J (np.ndarray): (num_qubits, num_qubits) symmetric coupling matrix
            with zero diagonal.
        dt (float): Trotter time step, applied to the IsingZZ coupling.
        f_b (float): input scaling for the encoding layer.
        n_trotter (int): number of Trotter steps.
        feedback_scale (float, optional): multiplier on arccos(previous_z)
            when forming the transverse-rotation angles. Defaults to `dt`,
            which reproduces the original behaviour -- but see the note below.

    Note on feedback_scale:
        arccos(z) naturally spans [0, pi]. Multiplying it by dt=0.1 compresses
        the RX angles into [0, 0.31], so the reservoir map contracts onto a
        fixed point near <Z_i> = 1 and stops responding to input. Measured
        across 200 steps with 4 qubits:

            feedback_scale=dt,  f_b=0.1  ->  feature std 0.007   (dead)
            feedback_scale=1.0, f_b=0.1  ->  feature std 0.673   (alive)

        A reservoir needs to sit at the edge of chaos: contracting enough to
        forget slowly, expansive enough to encode. Pass feedback_scale=1.0
        (and consider raising f_b) if you want usable features.

    Returns:
        callable: circuit(v_0, previous_z) -> list of <Z_i>, to be wrapped
        in a qml.QNode.
    """
    J = np.asarray(J, dtype=float)
    if J.shape != (num_qubits, num_qubits):
        raise ValueError(f"J must be ({num_qubits}, {num_qubits}), got {J.shape}")

    if feedback_scale is None:
        feedback_scale = dt

    edges = [
        (i, j)
        for i in range(num_qubits)
        for j in range(i + 1, num_qubits)
        if J[i, j] != 0
    ]

    def circuit(v_0, previous_z):
        # Numerical protection for arccos.
        previous_z = qml.math.clip(previous_z, -1.0, 1.0)

        # Previous reservoir output converted to angles.
        feedback_angles = qml.math.arccos(previous_z) * feedback_scale

        # ---------------------------------------
        # Layer 1: encode the current input vector.
        # One feature per qubit: O_t, H_t, V_t, y_t, C_t, ...
        # ---------------------------------------
        for i in range(num_qubits):
            qml.RY(f_b * v_0[i], wires=i)

        # Layers 2..k: alternate coupling + transverse rotation.
        for _ in range(n_trotter):
            for i, j in edges:
                qml.IsingZZ(2.0 * J[i, j] * dt, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(2.0 * feedback_angles[i], wires=i)

        # One Z observable per qubit.
        return [qml.expval(qml.PauliZ(i)) for i in range(num_qubits)]

    return circuit


def generate_J(num_qubits, sigma=1.0, seed=None):
    """Symmetric coupling matrix with zero diagonal (matches QRC_Model)."""
    rng = np.random.default_rng(seed)
    J = rng.normal(0.0, sigma, size=(num_qubits, num_qubits))
    J = (J + J.T) / 2.0
    np.fill_diagonal(J, 0.0)
    return J


def _run_series(qnode, num_qubits, num_steps, seed=1):
    """Drive the reservoir with random input and collect the feature series."""
    rng = np.random.default_rng(seed)
    z = np.zeros(num_qubits)          # Hadamard init: <Z_i> = 0
    feats = []
    for _ in range(num_steps):
        v_0 = rng.normal(0.0, 1.0, size=num_qubits)
        z = np.asarray(qnode(v_0, z), dtype=float)
        feats.append(z)
    return np.asarray(feats)


if __name__ == "__main__":
    num_qubits = 4
    dt = 0.1
    n_trotter = 2
    J = generate_J(num_qubits, sigma=1.0, seed=0)
    dev = qml.device("default.qubit", wires=num_qubits)

    print("Reservoir richness diagnostic (200 steps, discarding 50 as washout)")
    print(f"{'feedback_scale':>15}  {'f_b':>5}  {'mean|Z|':>8}  {'feature std':>12}")
    for fs, f_b in [(dt, 0.1), (dt, 1.0), (1.0, 0.1), (1.0, 1.0)]:
        qn = qml.QNode(
            qrc_circuit(num_qubits, J=J, dt=dt, f_b=f_b,
                        n_trotter=n_trotter, feedback_scale=fs),
            dev,
        )
        F = _run_series(qn, num_qubits, 200)[50:]
        print(f"{fs:>15.2f}  {f_b:>5.2f}  {np.abs(F).mean():>8.4f}  "
              f"{F.std(axis=0).mean():>12.5f}")

    print("\nSample trajectory (feedback_scale=1.0, f_b=1.0):")
    qn = qml.QNode(
        qrc_circuit(num_qubits, J=J, dt=dt, f_b=1.0,
                    n_trotter=n_trotter, feedback_scale=1.0),
        dev,
    )
    F = _run_series(qn, num_qubits, 6)
    for step, z in enumerate(F):
        print(f"  step {step}: {np.array2string(z, precision=4, suppress_small=True)}")

    print("\nCircuit structure:")
    print(qml.draw(qn)(np.zeros(num_qubits), np.zeros(num_qubits)))