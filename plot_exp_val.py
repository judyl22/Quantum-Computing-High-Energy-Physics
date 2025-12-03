import qiskit
import qiskit_ibm_runtime
import pylatexenc
from qiskit import QuantumCircuit
from qiskit.quantum_info import SparsePauliOp
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import EstimatorV2 as Estimator
from qiskit.circuit import Parameter
from qiskit.circuit.library import UGate
from qiskit.circuit.library import RXGate
from qiskit.quantum_info import SparsePauliOp
from qiskit_ibm_runtime.fake_provider import FakeAlmadenV2
from qiskit.primitives import StatevectorEstimator
import numpy as np
from numpy import pi
from itertools import product
from matplotlib import pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
from pytplot import get_data
from matplotlib import pyplot as plt
# %matplotlib tk
from matplotlib.colors import LogNorm
import logging

t = Parameter("t")
g = Parameter("g")

theta_E = (8 / 3) * g**2 * t
theta_B = 1 / (g**2 * np.sqrt(2)) * t

def one_step_new(n, N):
    qc = QuantumCircuit(n)
    
    for q in range(n):
        qc.rz(theta_E/N, q)

    for i in range(1, n - 1):       
        qc.x(i - 1)
        qc.x(i + 1)
        
        controlled_rx = RXGate(theta_B/N).control(2)  # 2 control qubits
        qc.append(controlled_rx, [i - 1, i + 1, i])

        qc.x(i - 1)
        qc.x(i + 1)
        
    return qc

def N_steps_compose_new(n, N):
    one_step_circuit = one_step_new(n, N)
    full_circuit = QuantumCircuit(n)
    for _ in range(N):
        full_circuit.compose(one_step_circuit, inplace=True)
    return full_circuit

# this is the helper function that can plot expectation values for different N
# T = total time evolution, used in later functions
# N = number of steps, t = T/N = step size
def plot_measurement_loop_one_observable(n, N, o_index, t):
    circuit = N_steps_compose_new(n, N)
    pm = generate_preset_pass_manager(optimization_level=1)
    isa_circuit = pm.run(circuit)
    mapped_observables = [
        observables[o_index].apply_layout(isa_circuit.layout)
    ]

    estimator = StatevectorEstimator()
    job = estimator.run([(isa_circuit, mapped_observables, [1.0, t])])
    
    result = job.result()
 
    pub_result = job.result()[0]

    values = pub_result.data.evs
    labels = N
    return labels, values

# Obervables
observables_labels = ["IZI", "ZZI", "IZZ", "ZIZ", "IXI", "XXI", "IXX", "XIX", "III"]
observables = [SparsePauliOp(label) for label in observables_labels]

# this plotting fixes N * t = 1.0, takes N as an argument
# default o_index[1] = ZZI
def exp_val_one_observable(N, n=3, o_index=1):
    exp_vals = []
    N_s = np.arange(1, N+1, 1)
    t_s = 1.0 / N_s
    for t, N in zip(t_s, N_s):
        exp_val = plot_measurement_loop_one_observable(n, N, o_index, t)[1][0]
        exp_vals.append(exp_val)

    plt.plot(N_s, exp_vals, "-o")
    plt.xlabel("N (where t * N = 1)")
    plt.ylabel(f"{observables_labels[o_index]} Expectation Value")
    plt.title(f"Expectation vs N for t*N=1, observable {observables_labels[o_index]}")
    plt.xticks(np.arange(1, N+1, 1))
    plt.grid(True)
    plt.show()

# this function is similar to above, but can plot t as a seperate x-axis on the top of the graph
def exp_val_one_observable_dual_axis(N, o_index=1, n=3):
    exp_vals = []
    N_s = np.arange(1, N+1, 1)
    t_s = 1.0 / N_s
    for t, N in zip(t_s, N_s):
        exp_val = plot_measurement_loop_one_observable(n, N, o_index, t)[1][0]
        exp_vals.append(exp_val)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(N_s, exp_vals, "o-")
    ax1.set_xlabel("N")
    ax1.set_ylabel(f"{observables_labels[o_index]} Expectation Value")
    ax1.set_xticks(np.arange(1, N+1, 1))
    ax1.set_title(f"Expectation vs N (top axis: t = 1/N)")
    ax1.grid(True)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(N_s)
    ax2.set_xticklabels(f"{t:.3f}" for t in t_s)
    ax2.tick_params(axis="x", labelrotation=45, pad=10)
    ax2.set_xlabel("t = 1 / N")
    
    plt.tight_layout()
    plt.show()

# This function takes in a parameter T which is the total time T=N*t
# N the number of steps is set to default 40
def exp_val_given_T(T, N=40, o_index=1, n=3):
    exp_vals = []
    N_s = np.arange(1, N+1, 1)
    t_s = T / N_s
    for t, N in zip(t_s, N_s):
        exp_val = plot_measurement_loop_one_observable(n, N, o_index, t)[1][0]
        exp_vals.append(exp_val)

    fig = plt.figure(figsize=(12, 5))
    ax1 = fig.add_subplot(111)
    ax2 = ax1.twiny()

    ax1.plot(N_s, exp_vals, "o-")
    ax1.set_xlabel("N")
    ax1.set_ylabel(f"{observables_labels[o_index]} Expectation Value")
    ax1.set_xticks(np.arange(1, N+1, 1))
    ax1.set_title(f"Expectation vs N (top axis: t = 1/N)")
    ax1.grid(True)

    ax2.set_xlim(ax1.get_xlim())
    ax2.set_xticks(N_s)
    ax2.set_xticklabels(f"{t:.3f}" for t in t_s)
    ax2.tick_params(axis="x", labelrotation=45, pad=10)
    ax2.set_xlabel("t = 1 / N")

    plt.tight_layout()
    plt.show()

# Plots an overlapped plot of different Ts
# the number of steps N set to default 40, the observable is set to default ZZI
# Abandons the dual axis because different T and same N generate different t axis
# Caution to use, can be slow
def plot_multiple_T(T, o_index=1, N=40):
    fig = plt.figure(figsize=(12, 5))
    
    for T in np.arange(1.0, T+1.0, 1.0):
        exp_vals = []
        N_s = np.arange(1, N+1, 1)
        t_s = T / N_s
        for t, N in zip(t_s, N_s):
            exp_val = plot_measurement_loop_one_observable(3, N, o_index, t)[1][0]
            exp_vals.append(exp_val)
        plt.plot(N_s, exp_vals, "o-", label=f"T = {T}")

    plt.xlabel("N")
    plt.ylabel(f"{observables_labels[o_index]} Expectation Value")
    plt.xticks(np.arange(1, N+1, 1))
    plt.title(f"Expectation vs N (top axis: t = 1/N)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()