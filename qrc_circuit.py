import warnings
import pennylane as qml

# Ignore the DeprecationWarnings coming from qiskit EfficientSU2 that can not be avoided.
warnings.filterwarnings("ignore", category=DeprecationWarning)

def qrc_circuit(num_qubits, t, num_steps, h, J, edges):
    dt = t / num_steps # Get number of steps from time t gate
    qml.device("default.qubit", wires=num_qubits)
    for step in range(num_steps):
        for i in range(num_qubits):
            qml.RX(2 * h[i] * dt, wires=i)
        for i, j in edges:
            qml.IsingZZ(2 * J[i, j] * dt, wires=[i, j])
