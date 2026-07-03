from sklearn.metrics import r2_score
from QRC_model import QRC_Model
import numpy as np
import pennylane as qml

npts = 26  # Total evolution time
npred = 20  # Number of time-steps to predict
n_train = 10  # Number of training samples
n_test = 4  # The number of test samples.
# Generate simple model which just averages over the train data
train_data = np.zeros((n_train, npts))
test_data = np.zeros((n_test,npts))
simple_model = np.mean(train_data.reshape(n_train,npts)[:,npts-npred:].T, axis=1)
use_gpu = True

# Reshape test data
all_tests = test_data.reshape(n_test, npts)[:,npts-npred:]

simple_r2 = r2_score(np.array(all_tests).flatten(), np.array(list(simple_model)*n_test).flatten())
print(f"The simple model has an R^2 of {simple_r2}")

# The hyper parameters for the OCQRC model
b = -0.33
f_b = 0.11
num_qubits = 6
n_qrc_layers = 3
f_bs = [f_b, f_b*1.25, f_b*1.125]
crc_size = 850
ridge_regularization = 3.e-4

r2scores = []
all_predictions = []

if use_gpu:
    dev = qml.device("lightning.gpu", wires=num_qubits)
else:
    dev = qml.device("default.qubit", wires=num_qubits)

for use_classical, use_quantum in [(True, False), (False, True), (True, True)]:

    model = QRC_Model(num_qubits=num_qubits, crc_size=crc_size,
                        use_classical=use_classical, use_quantum=use_quantum,
                        backends=[dev]*n_qrc_layers,
                        b=b, f_bs=f_bs,
                        ridge_param=ridge_regularization)

    # Run through the 10 instances of training series of length 26
    model.train(train_data.reshape(n_train, npts)[:, :])

    # Run through the 2 instances of test series to predict 20 points of data given the first 6 points
    all_predictions += [model.forward(test_data.reshape(n_test, npts)[:,:-npred], npred)]
    
    r2scores += [r2_score(np.array(all_tests).flatten(), np.array(all_predictions[-1]).flatten())]
    label = "classical" if use_classical else ""
    label += "+" if use_classical and use_quantum else ""
    label += "quantum" if use_quantum else ""
    print(f'The {label} model has an R^2 of {r2scores[-1]}')