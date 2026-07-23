"""
qrc_qbraid_backend.py
---------------------
Run the QRC reservoir evolution step on a real QPU via the qBraid runtime,
instead of PennyLane's local `default.qubit` simulator.

Design notes
------------
The reservoir is sequentially feedback-driven: the rotation angles at step t
are derived from the MEASURED expectation values at step t-1.  This means
circuits cannot be batched -- each job must complete before the next circuit
can even be constructed.  Budget accordingly.

Usage
-----
    from qrc_qbraid_backend import QPUReservoir, QPUBackedQRCModel

    res = QPUReservoir(num_qubits=6, J=J, dt=0.1, f_b=0.1,
                       device_id="<id from provider.get_devices()>",
                       shots=1000)

    model = QPUBackedQRCModel(num_qubits=6, backends=[None], f_bs=[0.1],
                              dt=0.1, seed=0)
    model.attach_qpu(res)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pennylane as qml
from qbraid import QbraidProvider, transpile


# --------------------------------------------------------------------------
# Tape construction
# --------------------------------------------------------------------------
def build_reservoir_tape(num_qubits, J, t0, previous_z, dt=0.1, f_b=0.1,
                         n_trotter=2):
    """Mirror of qrc_circuit.qrc_circuit, but returns a measurable QuantumTape.

    The local version ends in `qml.expval(PauliZ(i))` for each wire.  Hardware
    cannot do that directly -- you get bitstring counts and reconstruct the
    expectation values yourself (see z_expvals_from_counts below).
    """
    edges = [(i, j)
             for i in range(num_qubits)
             for j in range(i + 1, num_qubits)
             if J[i][j] != 0]

    previous_z = np.clip(np.asarray(previous_z, dtype=float), -1.0, 1.0)
    feedback_angles = np.arccos(previous_z) * dt

    with qml.tape.QuantumTape() as tape:
        # Layer 1: current input on wire 0, previous memory on the rest
        qml.RY(f_b * dt * np.arctan(t0), wires=0)
        for i in range(1, num_qubits):
            qml.RY(feedback_angles[i - 1], wires=i)

        # Trotterized coupling + transverse field
        for _ in range(n_trotter):
            for i, j in edges:
                qml.IsingZZ(2.0 * J[i, j] * dt, wires=[i, j])
            for i in range(num_qubits):
                qml.RX(2.0 * feedback_angles[i], wires=i)

        qml.counts(wires=range(num_qubits))

    return tape


# --------------------------------------------------------------------------
# Counts -> expectation values
# --------------------------------------------------------------------------
def z_expvals_from_counts(counts, num_qubits, little_endian=True):
    """Reconstruct <Z_i> for each wire from a bitstring counts dict.

    <Z_i> = sum_b p(b) * (-1)^{b_i}

    IMPORTANT: bit ordering differs between providers.  Validate this against
    a simulator run of the SAME circuit before trusting hardware numbers --
    a silent endianness flip will look like plausible-but-wrong physics.
    """
    total = sum(counts.values())
    if total == 0:
        raise ValueError("Empty counts dictionary")

    z = np.zeros(num_qubits, dtype=float)
    for bitstr, n in counts.items():
        b = str(bitstr).replace(" ", "")
        if len(b) != num_qubits:
            b = b.zfill(num_qubits)
        for i in range(num_qubits):
            idx = (num_qubits - 1 - i) if little_endian else i
            z[i] += n * (1.0 - 2.0 * int(b[idx]))

    return z / total


def zz_expvals_from_counts(counts, pairs, num_qubits, little_endian=True):
    """Reconstruct <Z_i Z_j> for the given (i, j) pairs from the same counts.

    Note: you get every two-body correlator for free from one measurement --
    no extra shots needed, since they all commute in the computational basis.
    """
    total = sum(counts.values())
    zz = np.zeros(len(pairs), dtype=float)

    for bitstr, n in counts.items():
        b = str(bitstr).replace(" ", "")
        if len(b) != num_qubits:
            b = b.zfill(num_qubits)
        for k, (i, j) in enumerate(pairs):
            ii = (num_qubits - 1 - i) if little_endian else i
            jj = (num_qubits - 1 - j) if little_endian else j
            parity = int(b[ii]) ^ int(b[jj])
            zz[k] += n * (1.0 - 2.0 * parity)

    return zz / total


# --------------------------------------------------------------------------
# QPU-backed reservoir
# --------------------------------------------------------------------------
class QPUReservoir:
    """One reservoir whose evolution step executes on a qBraid device."""

    def __init__(self, num_qubits, J, device_id, dt=0.1, f_b=0.1,
                 n_trotter=2, shots=1000, api_key=None,
                 cache_path="qrc_qpu_cache.jsonl", poll_seconds=10,
                 little_endian=True):
        self.num_qubits = num_qubits
        self.J = np.asarray(J, dtype=float)
        self.dt = dt
        self.f_b = f_b
        self.n_trotter = n_trotter
        self.shots = shots
        self.poll_seconds = poll_seconds
        self.little_endian = little_endian

        self.provider = QbraidProvider(api_key=api_key or os.getenv("QBRAID_API_KEY"))
        self.device = self.provider.get_device(device_id)

        # Every completed step is appended to disk.  Queue waits are long and
        # scripts die; without this you lose the whole run to one timeout.
        self.cache_path = cache_path
        self._cache = self._load_cache()
        self.step_index = 0

    # -- persistence -------------------------------------------------------
    def _load_cache(self):
        cache = {}
        if self.cache_path and os.path.exists(self.cache_path):
            with open(self.cache_path) as fh:
                for line in fh:
                    rec = json.loads(line)
                    cache[rec["key"]] = rec
        return cache

    def _append_cache(self, rec):
        self._cache[rec["key"]] = rec
        if self.cache_path:
            with open(self.cache_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")

    # -- dry run -----------------------------------------------------------
    def preview_qasm(self, t0=0.0, previous_z=None):
        """Print the transpiled circuit. ALWAYS run this before spending credits."""
        if previous_z is None:
            previous_z = np.zeros(self.num_qubits)
        tape = build_reservoir_tape(self.num_qubits, self.J, t0, previous_z,
                                    dt=self.dt, f_b=self.f_b,
                                    n_trotter=self.n_trotter)
        return transpile(tape, "qasm2")

    # -- the actual step ---------------------------------------------------
    def evolve(self, t0, previous_z):
        """Submit one reservoir step, block for the result, return <Z_i> vector."""
        key = f"{self.step_index}"
        if key in self._cache:
            return np.asarray(self._cache[key]["z"], dtype=float)

        tape = build_reservoir_tape(self.num_qubits, self.J, t0, previous_z,
                                    dt=self.dt, f_b=self.f_b,
                                    n_trotter=self.n_trotter)

        job = self.device.run(tape, shots=self.shots)

        while not job.is_terminal_state():
            time.sleep(self.poll_seconds)

        counts = job.result().data.get_counts()
        z = z_expvals_from_counts(counts, self.num_qubits,
                                  little_endian=self.little_endian)

        self._append_cache({
            "key": key,
            "job_id": str(getattr(job, "id", "")),
            "t0": float(t0),
            "counts": {str(k): int(v) for k, v in counts.items()},
            "z": z.tolist(),
        })
        self.step_index += 1
        return z


# --------------------------------------------------------------------------
# Drop-in model override
# --------------------------------------------------------------------------
def make_qpu_backed_model(QRC_Model, reservoirs):
    """Return a subclass of your QRC_Model whose evolve_qrc() hits real hardware.

    `reservoirs` is a list of QPUReservoir, one per entry in f_bs -- matching
    the number of qnodes the original model builds.
    """

    class QPUBackedQRCModel(QRC_Model):
        def evolve_qrc(self, t0):
            obsvs = []
            for c, res in enumerate(reservoirs):
                previous_z = self.last_output[c]
                new_z = res.evolve(t0, previous_z)
                self.last_output[c] = new_z.copy()
                obsvs.extend(new_z.tolist())
            return obsvs

    return QPUBackedQRCModel


# --------------------------------------------------------------------------
if __name__ == "__main__":
    # Discovery: which devices are actually online right now?
    provider = QbraidProvider()
    for d in provider.get_devices():
        meta = d.metadata()
        if meta.get("device_type") == "QPU":
            print(d.id, meta.get("status"), meta.get("num_qubits"),
                  "queue:", meta.get("queue_depth"))
