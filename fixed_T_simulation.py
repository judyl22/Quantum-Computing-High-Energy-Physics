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
from matplotlib.colors import LogNorm

# generate pop-up plot
# from pytplot import get_data
# %matplotlib tk

import logging
import itertools
from matplotlib.backends.backend_pdf import PdfPages

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

def plot_measurement_loop_one_observable(n, N, observable, t, g):
    circuit = N_steps_compose_new(n, N)
    pm = generate_preset_pass_manager(optimization_level=1)
    isa_circuit = pm.run(circuit)
    mapped_observables = [
        SparsePauliOp(observable).apply_layout(isa_circuit.layout)
    ]

    estimator = StatevectorEstimator()
    job = estimator.run([(isa_circuit, mapped_observables, [g, t])])
    
    result = job.result()
 
    pub_result = job.result()[0]

    values = pub_result.data.evs
    labels = N
    return labels, values

def exp_val_vs_T(n=3, observable="IXI", Ns=[2, 5, 10, 20], save_fig=False, g=1.0):
    fig, ax = plt.subplots(figsize=(12, 5))
    for N in Ns:
        T_vals = []
        for t in range(0, 15): 
            T_vals.extend(np.linspace(t, t+1, N+1))
        exp_vals = []
        for T in T_vals:
            val = plot_measurement_loop_one_observable(n, N, observable, T, g)[1][0]
            exp_vals.append(val)

        ax.plot(T_vals, exp_vals, "o-", label=f"N={N}, Δt={1.0/N:.3f}")

    ax.set_xlabel("Total time T")
    ax.set_ylabel(f"{observable} Expectation Value")
    ax.set_title(f"Expectation vs T, for g = {g}")
    ax.grid(True)
    ax.legend(title="Steps N")

    fig.tight_layout()
    if save_fig:
        fig.savefig(f"{observable}_vs_T_steps.png")
    plt.show()

def exp_val_vs_T_diff_g(n=3, N=10, observable="IXI", save_fig=False, gs=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5], num_delta_t=10, total_time=15):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Blues(np.linspace(0, 1, len(gs)+1))[1:]
    
    for g, color in zip(gs,colors):
        T_vals = []
        for t in range(0, total_time): 
            T_vals.extend(np.linspace(t, t+1, num_delta_t+1)[:-1])
        exp_vals = []
        for T in T_vals:
            val = plot_measurement_loop_one_observable(n, N, observable, T, g)[1][0]
            exp_vals.append(val)

        ax.plot(T_vals, exp_vals, "o-", markersize=4, label=f"g={g}, Δt={1.0/num_delta_t}", color=color)

    ax.set_xlabel("Total time T")
    ax.set_ylabel(f"{observable} Expectation Value")
    ax.set_title(f"Expectation vs T, for g = {gs}")
    ax.grid(True)
    ax.legend(title="Steps N")

    fig.tight_layout()
    if save_fig:
        fig.savefig(f"{observable}_vs_T_steps.png")
    plt.show()

def avg_T_diff_g(n=3, N=10, observable="IXI", save_fig=False, gs=[0.25, 0.5, 0.75, 1.0, 1.25, 1.5], num_delta_t=10, total_time=15):
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = plt.cm.Blues(np.linspace(0, 1, len(gs)+1))[1:]
    
    for g, color in zip(gs,colors):
        T_vals = []
        for t in range(0, total_time): 
            T_vals.extend(np.linspace(t, t+1, num_delta_t+1)[:-1])
        exp_vals = []
        sum_exp_vals = []
        for T in T_vals[1:]:
            val = (plot_measurement_loop_one_observable(n, N, observable, T, g)[1][0])
            exp_vals.append(val)
            # sum_exp_val = np.sum(exp_vals) / T
            # sum_exp_val = np.mean(exp_vals)
            sum_exp_val = np.trapz(exp_vals, T_vals[:len(exp_vals)]) / T
            sum_exp_vals.append(sum_exp_val)

        ax.plot(T_vals[1:], sum_exp_vals, "o-", markersize=4, label=f"g={g}, Δt={1.0/num_delta_t}", color=color)

    ax.set_xlabel("Total time T")
    ax.set_ylabel(f"{observable} Expectation Value Averaging")
    ax.set_title(f"Expectation vs T, for g = {gs}")
    ax.grid(True)
    ax.legend(title="Steps N")

    fig.tight_layout()
    if save_fig:
        fig.savefig(f"{observable}_vs_T_steps.png")
    plt.show()

# N = number of actual time steps
# num_delta_t = number of time steps to plot on the graph
def comp_obs(observables, n=4, N=20, g=1.5, num_delta_t=2, total_time=100):
    fig, ax = plt.subplots(figsize=(12, 5))

    for observable in observables:
        T_vals = []
        for t in range(0, total_time): 
            T_vals.extend(np.linspace(t, t+1, num_delta_t+1)[:-1])
        exp_vals = []
        sum_exp_vals = []
        for T in T_vals[1:]:
            val = (plot_measurement_loop_one_observable(n, N, observable, T, g)[1][0])
            exp_vals.append(val)
            sum_exp_val = np.trapz(exp_vals, T_vals[:len(exp_vals)]) / T
            sum_exp_vals.append(sum_exp_val)

        ax.plot(T_vals[1:], sum_exp_vals, "o-", markersize=4, label=f"observable={observable}")

    ax.set_xlabel("Total time T")
    ax.set_ylabel(f"{observable} Expectation Value Averaging")
    ax.set_title(f"Expectation vs T")
    ax.grid(True)
    ax.legend(title="Observable")
    fig.tight_layout()
    plt.show()