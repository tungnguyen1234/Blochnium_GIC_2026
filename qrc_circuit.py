import warnings
import pennylane as qml
import numpy as np

# Ignore the DeprecationWarnings coming from qiskit EfficientSU2 that can not be avoided.
warnings.filterwarnings("ignore", category=DeprecationWarning)

def qrc_circuit(num_qubits, t, num_steps, h, J, edges):
    dt = t / num_steps # Get number of steps from time t gate
    dev = qml.device("default.qubit", wires=num_qubits)
    
    @qml.qnode(dev)
    def circuit():
        for step in range(num_steps):
            for i in range(num_qubits):
                qml.RX(2 * h[i] * dt, wires=i)
            for i, j in edges:
                qml.IsingZZ(2 * J[i, j] * dt, wires=[i, j])
    
    drawer = qml.draw(circuit)
    print(drawer())
    return circuit


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    num_qubits = 4
    t = 1.0
    num_steps = 10
    h = rng.uniform(0, 1, num_qubits)
    J = rng.uniform(-1, 1, size=(num_qubits, num_qubits))
    edges = [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits) if J[i][j] != 0]

    circuit = qrc_circuit(num_qubits, t, num_steps, h, J, edges)
    