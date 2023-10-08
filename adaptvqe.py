from qiskit_nature.units import DistanceUnit
from qiskit_nature.second_q.drivers import PySCFDriver
from qiskit_nature.second_q.mappers import JordanWignerMapper, ParityMapper, QubitConverter
from qiskit_nature.second_q.algorithms import GroundStateEigensolver
from qiskit.algorithms.minimum_eigensolvers import NumPyMinimumEigensolver, VQE, AdaptVQE
from qiskit.circuit.library import EvolvedOperatorAnsatz
from qiskit_nature.second_q.circuit.library import HartreeFock, UCCSD, PUCCD, SUCCD
from qiskit.algorithms.optimizers import SLSQP, SPSA, QNSPSA
from qiskit_aer.primitives import Estimator

from qiskit import QuantumCircuit, transpile
from qiskit.quantum_info import Kraus, SuperOp, SparsePauliOp
from qiskit.utils import QuantumInstance, algorithm_globals
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.tools.visualization import dag_drawer

from qiskit.providers.fake_provider import FakeMontreal, FakeKolkata
from qiskit import Aer, pulse
from qiskit_aer.noise import (NoiseModel, QuantumError, ReadoutError, pauli_error, depolarizing_error, thermal_relaxation_error)
import qiskit_aer.noise as noise
from qiskit_nature import settings

import numpy as np
import pickle
import pylab
import time
import warnings
warnings.filterwarnings('ignore')

from .utils import generate_Hamiltonian, getReference, getAccuracy

# hyper-parameters
seeds = 170
algorithm_globals.random_seed = seeds
seed_transpiler = seeds
iterations = 25
shot = 4096
settings.use_pauli_sum_op = True


# construct Ansatz
def constructAnsatz(qmolecule, mapper):
    uccsd = UCCSD(
        qmolecule.num_spatial_orbitals,
        qmolecule.num_particles,
        mapper,
    )

    # ops_pool = uccsd.operators[:16]+ [uccsd.operators[39]]
    # ops_pool = []
    # for i, ops in enumerate(uccsd.operators):
    #     pauli_string = str(ops._primitive._pauli_list[0])
    #     if pauli_string[:6] == 'IIIIII' or pauli_string[-6:] == 'IIIIII':
    #         ops_pool.append(ops)

    ansatz = EvolvedOperatorAnsatz(
        operators=uccsd.operators,
        initial_state=HartreeFock(
            qmolecule.num_spatial_orbitals,
            qmolecule.num_particles,
            mapper,
        ),
    )
    return ansatz


# AdaptVQE
def runAdaptVQE(ansatz, qubit_op, iterations):
    estimator = Estimator(
        backend_options = {
            'method': 'statevector',
            'device': 'CPU',
            # 'noise_model': noise_model
            },
        run_options = {
            'shots': shot,
            'seed': seeds,
        },
        # skip_transpilation=True
        transpile_options = {
            'seed_transpiler': seed_transpiler
        },
        approximation = True
    )
    
    vqe = VQE(estimator, ansatz, SLSQP())
    vqe.initial_point = np.zeros(ansatz.num_parameters)
    adapt_vqe = AdaptVQE(vqe, max_iterations=iterations, gradient_threshold=1e-12, eigenvalue_threshold=1e-8)
    # adapt_vqe.supports_aux_operators = lambda: True


    start_time = time.time()
    result = adapt_vqe.compute_minimum_eigenvalue(qubit_op)
    end_time = time.time()
    timeCost = end_time - start_time
    return result, ansatz, timeCost


if __name__ == "__main__":
    qmolecule, mapper, qubit_op = generate_Hamiltonian()

    ref_res, ref_value = getReference(qmolecule)
    print(f" the computed energies: {ref_res.computed_energies} \n the nuclear replusion energy: {ref_res.nuclear_repulsion_energy} \n the reference energy: {ref_value} \n")

    ansatz = constructAnsatz(qmolecule, mapper)
    
    res, ansatz, timeCost = runAdaptVQE(ansatz, qubit_op, iterations)  # iterations=1
    print(f'vqe time cost: {timeCost}\n')
    print(res, end='\n')
    with open(f'optimal.qc', 'wb') as f:
        pickle.dump(res.optimal_circuit.decompose(), f)
    np.save(f'./optimal_params.npy', res.optimal_parameters)

    error_rate, acc_rate = getAccuracy(res, ref_value)
    print("Error rate: %f%% \n" % (error_rate))
    print("Accuracy: %f%%\n" % (acc_rate))

    print('successfuly done!')