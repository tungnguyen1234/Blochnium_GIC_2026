import warnings
import pennylane as qml
import numpy as np

# Ignore the DeprecationWarnings coming from qiskit EfficientSU2 that can not be avoided.
warnings.filterwarnings("ignore", category=DeprecationWarning)

def qrc_circuit(num_qubits, t, num_steps, f_b, params=[], J=[]):
    edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits) if J[i][j] != 0]
    dt = t / num_steps
    # Numerical protection for arccos.
    previous_z = qml.math.clip(previous_z, -1.0, 1.0)

    # Previous reservoir output converted to angles.
    feedback_angles = qml.math.arccos(previous_z) * dt

    # ---------------------------------------
    # Layer 1: current input + previous memory
    # ---------------------------------------
    
    # Current residual x_t goes to qubit 0.
    qml.RY(f_b, wires=0)

    # Remaining qubits receive y_0 ... y_(n-2).
    for i in range(1, num_qubits):
        qml.RY(feedback_angles[i - 1], wires=i,)

    # ---------------------------------------
    # Layer 2: all previous feedback values
    # ---------------------------------------
    for i in range(num_qubits):
        qml.RX(params[i], wires=i,)

    # ---------------------------------------
    # Layer 3: fixed Ising ZZ evolution
    # ---------------------------------------
    for i , j in edges:
        if J[i, j] != 0.0:
            qml.IsingZZ(2.0 * J[i, j] * dt, wires=[i, j],)

    drawer = qml.draw(circuit)
    print(drawer())
    # Only one Z observable per qubit.
    return [
        qml.expval(qml.PauliZ(i))
        for i in range(num_qubits)
    ]

    

    return circuit


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    num_qubits = 4
    t = 1.0
    num_steps = 2
    params = rng.uniform(0, 1, num_qubits)
    J = rng.uniform(-1, 1, size=(num_qubits, num_qubits))
    

    circuit = qrc_circuit(num_qubits, t, num_steps, params=params, J=J)
    