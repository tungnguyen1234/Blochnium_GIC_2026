import warnings
import pennylane as qml
import numpy as np

# Ignore the DeprecationWarnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def qrc_circuit(num_qubits, J, dt=0.1, f_b=0.1, n_trotter=2):
    edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits) if J[i][j] != 0]
    
    def circuit(t0, previous_z):
        # Numerical protection for arccos.
        previous_z = qml.math.clip(previous_z, -1.0, 1.0)

        # Previous reservoir output converted to angles.
        feedback_angles = qml.math.arccos(previous_z) * dt

        # ---------------------------------------
        # Layer 1: current input + previous memory
        # ---------------------------------------
        
        # Current residual x_t goes to qubit 0.
        qml.RY(f_b * dt * np.arctan(t0), wires=0)

        # Remaining qubits receive y_0 ... y_(n-2).
        for i in range(1, num_qubits):
            qml.RY(feedback_angles[i - 1], wires=i,)

        # Layers 2 to k: alternate coupling and transverse rotation
        for _ in range(n_trotter):        
            for i, j in edges:
                qml.IsingZZ(2.0 * J[i, j] * dt, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(2.0 * feedback_angles[i], wires=i)
                
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
    