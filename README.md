# Dendritic Neuron from Scratch (Python)

This repository implements a **dendritic neuron** with multiple branches from scratch in Python, alongside a standard point-neuron baseline. The goal is to provide a clear, inspectable playground for dendritic computation and dendritic optimization concepts inspired by recent work and hackathon material.

---

## Overview

This project explores, at a small scale, how adding structure inside a single neuron changes what it can represent:

- **DendriticNeuron**:
  - Multiple branches, each receiving a subset of input features.
  - Per-branch weights and biases.
  - A soma that combines branch outputs via its own weights and bias.
- **PointNeuron**:
  - Standard neuron with a single weight vector and scalar bias.
- **Experiments**:
  - Synthetic 2D-Gaussian branch-structured dataset.
  - Side‑by‑side training and evaluation of dendritic vs point neurons.
- **Tests**:
  - Forward correctness, shape checks, and numerical gradient tests to validate the implementation.

There is also a **learning script** that documents the process of understanding dendritic optimization concepts while following the Perforated AI / Weights & Biases “Dendritic Optimization” hackathon talk. This script is intentionally more exploratory and is kept in the repo as a study artifact.

---

## Dendritic Optimization (Conceptual Background)

Modern deep learning typically uses point neurons: all inputs are combined into a single weighted sum and passed through one nonlinearity. Dendritic optimization takes inspiration from neuroscience, where biological neurons possess rich dendritic trees that perform additional nonlinear computation before signals reach the soma.

Key conceptual ideas:

- **Branches as local experts**  
  Each dendritic branch receives only a subset of the inputs and learns its own local transformation. This allows the neuron to respond differently in distinct regions of the input space.

- **Error-aware assistant signals**  
  In dendritic optimization approaches, auxiliary “dendritic” units can be trained to detect patterns where a neuron tends to make mistakes (false positives/negatives). Their outputs are then fed back to adjust the neuron’s effective decision boundary without redesigning the entire network.

- **Perforated backpropagation idea (high level)**  
  Backpropagation still flows through the main neuron graph. Additional dendritic components:
  - Observe the same inputs and neuron errors.
  - Use a separate learning rule to become specialized “outlier detectors.”
  - Connect back to neurons through additional weights, upgrading each neuron’s decision capability while preserving the original network structure.

This repository does **not** re‑implement a full PyTorch dendritic library, but instead recreates a minimal, from‑scratch version of the core structural idea at the single‑neuron level. The design and comments were informed by carefully working through an educational dendritic optimization talk and rephrasing the key concepts into plain Python code.

---

## Repository Structure
```
dendritic-neuron/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── src/
│ ├── init.py
│ ├── dendritic_neuron.py # Dendritic neuron class: branches + soma
│ ├── point_neuron.py # Baseline point neuron
│ ├── activations.py # Activation functions
│ ├── losses.py # Loss functions and derivatives
│ ├── training.py # Generic training / evaluation loops
│ └── utils.py # Initialization, splits, batching, small math helpers
│
├── experiments/
│ ├── init.py
│ ├── synthetic_dataset.py # Toy dataset + branch grouping
│ ├── train_dendritic.py # Train dendritic neuron
│ ├── train_point.py # Train point neuron
│ └── evaluate_neurons.py # Compare trained models and noise robustness
│
├── notebooks/
│ └── 01_dendritic_vs_point.ipynb # Optional analysis / visualization
│
├── results/
│ ├── logs/ # Training logs (JSON, metrics, etc.)
│ ├── models/ # Saved neuron parameters (.npz)
│ └── summaries/ # Comparison metrics (JSON/CSV)
│
├── tests/
│ ├── test_dendritic_forward.py # Forward-shape & math tests
│ └── test_gradient.py # Numerical gradient checks
│
└── script/
└── ... # Personal learning script(s) written while
# studying dendritic optimization from the talk
```


The `script/` directory (or `script.py` if you use a single file) is intentionally kept separate from the core library and experiments. It contains notes, trial code, and intermediate steps developed while learning dendritic optimization from the referenced video. It is not part of the main experiment pipeline, but documents the reasoning path and can be useful for readers who want to see the "scratch work" behind the final implementation.

---

## Installation

1. **Clone the repository**
```
git clone https://github.com/laila-kz/Dendritic-neurons-from-scratch.git
cd dendritic-neuron
```


3. **Create and activate a virtual environment** (recommended)
```
python -m venv .venv
```

Windows (PowerShell)
```
.venv\Scripts\Activate.ps1
```


macOS / Linux
```
source .venv/bin/activate
```


3. **Install dependencies**
```
pip install -r requirements.txt
```


Recommended contents of `requirements.txt`:

- `numpy`
- `matplotlib`
- `scikit-learn`
- `pytest`

---

## Usage

All commands assume you are in the project root (`dendritic-neuron/`) with the virtual environment activated.

### 1. Train the dendritic neuron
```
python -m experiments.train_dendritic
```


This script will:

- Generate a 6‑dimensional synthetic dataset (three branches × two features per branch) for binary classification.  
- Split the data into train and validation sets.  
- Construct a `DendriticNeuron` with a specified branch structure and activations.  
- Train for a fixed number of epochs, printing training and validation loss/accuracy.  
- Save trained parameters to:
results/models/dendritic_params.npz


### 2. Train the point neuron baseline
```
python -m experiments.train_point
```

This script will:

- Use the same dataset and training hyperparameters as the dendritic run.  
- Train a `PointNeuron` with a single weight vector and bias.  
- Save:
results/models/point_neuron_params.npz
results/logs/point_neuron_history.json


### 3. Evaluate and compare neurons

After training both models:
```
python -m experiments.evaluate_neurons
```

This will:

- Regenerate a fresh test set.  
- Load the saved dendritic and point neuron parameters.  
- Evaluate both models on clean test data and on a noisy variant with Gaussian perturbations.  
- Print loss and accuracy (including noisy accuracy) for both models.  
- Write a JSON summary to:

results/summaries/comparison.json


---

## Running Tests

The tests import the core code via `src`, so the project root must be visible to Python.

From the project root:
Windows (PowerShell)
```
$env:PYTHONPATH = (Get-Location).Path
pytest tests
```


This runs:

- `tests/test_dendritic_forward.py`  
  - Shape and type checks for single and batch forward passes.  
  - Known-value test for manual dendritic math.  
  - Branch isolation and batch consistency tests.
- `tests/test_gradient.py`  
  - Numerical gradient checks (finite differences) for:
    - `PointNeuron` weights.  
    - `DendriticNeuron` branch weights.

These tests provide a sanity check that the forward and backward implementations behave as expected.

---

## The Learning Script

The **`script`** directory (or `script.py`) is a record of the learning process:

- Rough calculations, experiments, and notes made while working through a dendritic optimization talk.  
- Early or alternative implementations that helped build intuition before settling on the final `DendriticNeuron` and `PointNeuron` classes.  
- Additional comments and diagrams that may not fit the production code style but are useful pedagogically.

This material is not required to run the main experiments, but it is left in the repository on purpose as a transparent record of how the final design was reached. Readers who want to see how the concepts evolved from the talk into code can start by browsing the script and then compare it to the final `src/` modules.

---

## Possible Extensions

- Introduce different activation functions per branch or per group of branches.  
- Replace the single-neuron experiments with shallow networks where some layers use dendritic units.  
- Add more complex synthetic datasets to test when dendritic structure gives the largest benefit.  
- Integrate experiment tracking and hyperparameter sweeps to systematically explore:
  - Number of branches.  
  - Branch-to-input mappings.  
  - Noise levels and regularization strategies.

---




