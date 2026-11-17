# Quantum Computing for High Energy Physics

Exploratory Qiskit notebooks and Python utilities for simulating quantum time evolution and studying Pauli-observable expectation values. The experiments compare evolution step counts, coupling strengths, system sizes, and observables, with later work focused on time averages and identifying flat or plateau regions.

This repository is a collection of research experiments. The numbered notebooks record different approaches and parameter choices; they are not a single sequential pipeline.

## Setup

From the repository root, create a Python environment and install the dependencies used by the simulation utilities and analysis notebooks:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install qiskit qiskit-ibm-runtime pylatexenc "numpy<2" matplotlib scipy scikit-learn jupyterlab
jupyter lab
```

Dependency versions are not locked, and this installation recipe has not been validated against a fresh environment. The NumPy constraint preserves `np.trapz`, which the current averaging utilities use. The code also imports `FakeAlmadenV2` from `qiskit_ibm_runtime.fake_provider`; availability depends on the installed Runtime version.

Some introductory or older examples need additional packages:

```bash
python -m pip install qiskit-aer pytplot
```

`hello_world.ipynb` imports Qiskit Aer. `plot_exp_val.py` and `ex8_t*N_x-axis.ipynb` import `pytplot`. Run notebooks from the repository root so that imports of local modules resolve, and select the Python environment where you installed the dependencies.

## Run a small simulation

In a notebook, evaluate one observable at a chosen evolution time:

```python
from fixed_T_simulation import plot_measurement_loop_one_observable

steps, values = plot_measurement_loop_one_observable(
    n=3,
    N=10,
    observable="IXI",
    t=1.0,
    g=1.5,
)
print(f"N={steps}, expectation={values[0]}")
```

The shared utility constructs a circuit from repeated steps containing Z rotations and doubly controlled X rotations, then evaluates the observable with Qiskit's local `StatevectorEstimator`. This example does not require an IBM Quantum account.

The main parameters are:

| Parameter | Meaning |
| --- | --- |
| `n` | Number of qubits. |
| `N` | Number of repeated evolution steps; rotation angles are divided by `N`. |
| `t` | Total evolution time in the shared circuit utilities. |
| `g` | Nonzero coupling parameter. |
| `observable` | Pauli string with exactly `n` characters, such as `IXI`. |
| `num_delta_t` | Number of sampling points per unit time for curve generation, independent of `N`. |
| `total_time` | Upper time bound; the curve helpers sample below this bound. |

To generate and plot a time-averaged observable curve:

```python
import matplotlib.pyplot as plt
from fixed_T_simulation import return_vals

times, averages = return_vals(
    "IXXI",
    n=4,
    N=10,
    g=1.5,
    num_delta_t=2,
    total_time=10,
)

plt.plot(times, averages, "o-")
plt.xlabel("Total time T")
plt.ylabel("Time-averaged IXXI expectation")
plt.grid(True)
plt.show()
```

`return_vals` returns sampled times excluding zero and the corresponding numerical running averages. It uses the averaging implementation currently in the research code; check its integration convention before using the results for quantitative comparisons.

## Repository guide

| Files | Focus |
| --- | --- |
| `hello_world.ipynb`, `install_qiskit.ipynb` | Initial Qiskit examples and setup experiments. |
| `Exact simulation with Qiskit SDK primitives.ipynb` | Estimator and Sampler examples. |
| `ex1.ipynb`–`ex10_4_qubits.ipynb` | Circuit construction, controlled-gate variants, time-axis experiments, and small-system simulations. |
| `ex11_fixed_T.ipynb`–`ex15_greater-T.ipynb` | Fixed-time studies, step counts, four-qubit observables, and longer evolution times. |
| `ex16_diff-g_draft.ipynb`–`ex22_fft_of_diff-g.ipynb` | Coupling sweeps, observable comparisons, and Fourier analysis. |
| `ex23_average_T.ipynb`–`ex26_compare_obs.ipynb` | Time averaging, larger systems, and comparisons across observables. |
| `ex27_flat_region_try.ipynb`, `ex28_flat_region.ipynb` | Flat-region exploration. |
| `ex29_central_difference_approximation.ipynb`, `ex29_central_difference_approximation-Copy1.ipynb` | Central-difference analysis. |
| `fixed_T_simulation.py` | Shared circuit, expectation-value, plotting, time-average, and stable-point helpers. |
| `plot_exp_val.py` | Earlier plotting utilities for expectation values and step-count comparisons. |
| `flat_region_ml_baseline_v1.py` | Random-forest baseline using 21 manually labeled six- and eight-qubit observables. |
| `flat_region_ml_baseline_v2.py` | Smaller baseline using 10 observables and a shorter simulation window. |

## Flat-region classification

Run the smaller baseline from the repository root:

```bash
python flat_region_ml_baseline_v2.py
```

It generates simulation curves, extracts eight features describing slopes, variation, and low-slope fractions, and trains a random forest. Output includes dataset counts, three-fold cross-validation accuracy, feature importances, and predictions for the training observables.

Labels are assigned by visual inspection. The dataset is small, and the final per-observable predictions are made on the training data, so treat this as an exploratory baseline. Version 1 uses more observables and longer curves and runs its simulation and training immediately when imported.

## Working with the experiments

- Start with small systems and short time windows. Statevector memory grows exponentially with qubit count, and the helpers rebuild and transpile circuits for each sampled time.
- Keep observable length equal to the qubit count, use a positive step count, and avoid `g=0` because the circuit angles include an inverse-square coupling term.
- Inspect notebook parameters before running all cells; some experiments contain large sweeps or long evolution windows.
- `calc_stable_pt` uses hard-coded index sections designed around a 199-point curve. Adjust those sections before applying it to other sampling configurations.
- Saved notebook outputs reflect earlier runs. Record package versions and rerun the relevant cells when reproducing an experiment.
