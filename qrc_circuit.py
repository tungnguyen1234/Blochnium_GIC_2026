import warnings
import pennylane as qml

# Ignore the DeprecationWarnings coming from qiskit EfficientSU2 that can not be avoided.
warnings.filterwarnings("ignore", category=DeprecationWarning)

def qrc_circuit(num_qubits):
    qml.device("default.qubit", wires=num_qubits)
    pass
