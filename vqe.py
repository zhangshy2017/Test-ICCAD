from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitConverter
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE, AdaptVQE
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, PUCCD, SUCCD
from qiskit.algorithms.optimizers import SLSQP, SPSA
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

import numpy as np
import matplotlib.pyplot as plt
import pickle
import pylab
import time, math
from itertools import permutations
import warnings
warnings.filterwarnings('ignore')

from .utils import generate_Hamiltonian, getReference, readAnsatz, constructNoise, transpileCircuit, reorder, getAccuracy, getDuration

# hyper-parameters
seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
iterations = 125
shot = 6000

# VQE
def runVQE(ansatz, init_params, qmolecule, mapper, noise_model):
    estimator = Estimator(
        backend_options = {
            'method': 'statevector',
            'device': 'CPU',
            'noise_model': noise_model
            },
        run_options = {
            'shots': shot,
            'seed': seeds,
        },
        skip_transpilation=True
        # transpile_options = {
        #     'seed_transpiler': seed_transpiler
        # }
    )

    vqe_solver = VQE(estimator, ansatz, SLSQP())
    vqe_solver.initial_point = init_params

    start_time = time.time()
    calc = GroundStateEigensolver(mapper, vqe_solver)
    res = calc.solve(qmolecule)
    end_time = time.time()
    timeCost = end_time - start_time
    return res, ansatz, timeCost




if __name__ == "__main__":
    qmolecule, mapper, qubit_op = generate_Hamiltonian()

    result, ref_value = getReference(qmolecule)
    print(f" the computed energies: {result.computed_energies} \n the nuclear replusion energy: {result.nuclear_repulsion_energy} \n the reference energy: {ref_value}")

    ansatz, params = readAnsatz(f'./optimal.qc', f'./optimal_params.npy')
    # params = params[:ansatz.num_parameters]

    noise_modelreal = constructNoise()
    system_model, transpiled_circuit = transpileCircuit(ansatz, opt_level=3)
    transpiled_ansatz = reorder(transpiled_circuit)

    res, transpiled_ansatz, timeCost = runVQE(transpiled_ansatz, params, qmolecule, mapper, noise_modelreal)
    print(f'vqe time cost: {timeCost}')
    print(res)
    with open(f'./vqe.res', 'wb') as f:
        pickle.dump(res, f)

    error_rate, acc = getAccuracy(res, ref_value)
    print("Error rate: %f%%, Accuracy: %f%%" % (error_rate*100, acc*100))

    # duration = getDuration(system_model, transpiled_ansatz)
    # print(f"the duration of quantum circuit: {duration}")

