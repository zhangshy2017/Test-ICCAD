from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitConverter
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE, AdaptVQE
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, PUCCD, SUCCD
from qiskit.algorithms.optimizers import SLSQP, SPSA, QNSPSA
from qiskit_aer.primitives import Estimator
from qiskit.primitives import Sampler

from qiskit import QuantumCircuit, transpile, QuantumRegister
from qiskit.circuit import Parameter, ParameterVector
from qiskit.circuit.random import random_circuit
from qiskit.quantum_info import Kraus, SuperOp, SparsePauliOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer

from qiskit.providers.fake_provider import FakeMontreal, FakeKolkata, FakeCairo
from qiskit import Aer, pulse, execute
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error)
import qiskit_aer.noise as noise
from qiskit_nature import settings

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab
import math
import time
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')


# Hamiltonian
def generate_Hamiltonian():
    ultra_simplified_ala_string = """
    O 0.0 0.0 0.0
    H 0.45 -0.1525 -0.8454
    """

    driver = PySCFDriver(
        atom=ultra_simplified_ala_string.strip(),
        basis='sto3g',
        charge=1,
        spin=0,
        unit=DistanceUnit.ANGSTROM
    )
    qmolecule = driver.run()

    hamiltonian = qmolecule.hamiltonian
    coefficients = hamiltonian.electronic_integrals
    second_q_op = hamiltonian.second_q_op()

    mapper = JordanWignerMapper()
    converter = QubitConverter(mapper=mapper, two_qubit_reduction=False)
    qubit_op = converter.convert(second_q_op)

    return qmolecule, mapper, qubit_op

# classical solution
def getReference(qmolecule):
    solver = GroundStateEigensolver(
        JordanWignerMapper(),
        NumPyMinimumEigensolver(),
    )

    result = solver.solve(qmolecule)
    ref_value = result.computed_energies + result.nuclear_repulsion_energy
    # print(f" the computed energies: {result.computed_energies} \n the nuclear replusion energy: {result.nuclear_repulsion_energy} \n the reference energy: {ref_value}")
    return result, ref_value

def readAnsatz(qc_path, params_path):
    with open(qc_path, 'rb') as file:
        loaded_qc = pickle.load(file)
    optimal_params = list(np.load(params_path, allow_pickle=True).item().values())
    return loaded_qc, optimal_params

# construct noise model
def constructNoise(path='./NoiseModel/fakecairo.pkl'):
    with open(path, 'rb') as file:
        noise_model = pickle.load(file)
    noise_model_ = noise.NoiseModel()
    noise_modelreal = noise_model_.from_dict(noise_model)
    return noise_modelreal

# Circuit Transpile
def transpileCircuit(ansatz, opt_level=None):
    system_model = FakeCairo()   #######
    transpiled_circuit = transpile(ansatz, backend=system_model, optimization_level=opt_level)
    return system_model, transpiled_circuit

# remove the idle qubits in quantum circuit
def remove_idle_qubits(circ, ancilla):
    dag = circuit_to_dag(circ)
    for qubit in dag.qubits[:]:
        if qubit.index in ancilla:
            dag.remove_qubits(qubit)
    circ_ = dag_to_circuit(dag)
    return circ_

def reorder(transpiled_circ):
    curr_orders = []
    original = []
    ancilla = []
    for i, q in transpiled_circ.layout.initial_layout._p2v.items():
        if q.register.name == 'ancilla':
            ancilla.append(i)
            continue
        # print(i, q.register.name, q.index)
        curr_orders.append(i)
        original.append(q.index)
    sorted_curr_orders = np.sort(curr_orders).tolist()
    transpiled_curr_orders = [sorted_curr_orders.index(i) for i in curr_orders]
    final_order = [0]*12
    for o in transpiled_curr_orders:
        final_order[o] = original[transpiled_curr_orders.index(o)]
    
    circ = remove_idle_qubits(transpiled_circ, ancilla)
    res_circ = QuantumCircuit(12)
    res_circ = res_circ.compose(circ, final_order)
    return res_circ

# Calculate the Accuracy
def getAccuracy(res, ref_value):
    result = res.computed_energies + res.nuclear_repulsion_energy
    error_rate = abs(abs(ref_value - result) / ref_value)
    acc = 1 - error_rate
    return error_rate, acc

# Obtain the Duration of Quantum Circuit
def getDuration(backend, ansatz):
    with pulse.build(backend) as program:
        with pulse.transpiler_settings(optimization_level=3):
            pulse.call(ansatz)
    return program.duration





