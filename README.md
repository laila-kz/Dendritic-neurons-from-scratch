# Dendritic Neuron from Scratch (Python)

A **dendritic neuron** with multiple branches, implemented from scratch in Python, alongside a standard point-neuron baseline. The point is to provide a clear, inspectable playground for dendritic computation and dendritic optimization concepts inspired by the Perforated AI / Weights & Biases "Dendritic Optimization" hackathon talk.

Everything is plain Python. There is no PyTorch, no autograd, and the core library has **zero runtime dependencies** — so the forward and backward passes can be read line by line.

---

## Overview

- **`DendriticNeuron`**
  - Multiple branches, each receiving a subset of input features.
  - Per-branch weights and biases.
  - A soma that combines branch outputs via its own weights and bias.
- **`PointNeuron`**
  - Standard neuron with a single weight vector and scalar bias.
- **Experiments**
  - Synthetic 6D branch-structured Gaussian dataset.
  - Side-by-side training and evaluation of dendritic vs point neurons on an identical split.
- **Tests**
  - Forward correctness, numerical gradient checks, edge cases and input-validation coverage.

There is also a **reference script** under [`script/`](script/) documenting how the dendritic-optimization API fits together. It is a study artifact, not part of the experiment pipeline.

---

## Quick start

```bash
git clone https://github.com/laila-kz/Dendritic-neurons-from-scratch.git
cd Dendritic-neurons-from-scratch

python -m venv .venv
source .venv/bin/activate          # macOS / Linux
# .venv\Scripts\Activate.ps1       # Windows PowerShell

pip install -r requirements.txt
pip install -e .                   # editable install

pytest                             # 140+ tests
python -m experiments.train_dendritic
python -m experiments.train_point
python -m experiments.evaluate_neurons
```

> **No `PYTHONPATH` needed.** `pip install -e .` puts `src` and `experiments` on the import path, and `pyproject.toml` also sets `pythonpath = ["."]` for pytest, so `pytest` works straight from a fresh clone even if you skip the editable install.

> **Dependencies.** The library in `src/` is pure Python with no runtime
> dependencies — only `numpy` and `matplotlib` (experiments and plots) plus
> `pytest` (tests) are needed. `scikit-learn` is *not* required: the stratified
> train/val split is implemented in `experiments/common.py`.

---

## Repository structure

```
Dendritic-neurons-from-scratch/
├── README.md
├── pyproject.toml                 # packaging + pytest config
├── requirements.txt               # pinned dependency ranges
├── .gitignore
│
├── src/                           # the library (pure Python, no deps)
│   ├── __init__.py                # re-exports the public API
│   ├── dendritic_neuron.py        # branches + soma
│   ├── point_neuron.py            # baseline point neuron
│   ├── activations.py             # activations and derivatives
│   ├── losses.py                  # loss functions and derivatives
│   ├── training.py                # generic training / evaluation loops
│   └── utils.py                   # init, splits, batching, math helpers
│
├── experiments/
│   ├── __init__.py
│   ├── common.py                  # shared config: one split, one source of truth
│   ├── synthetic_dataset.py       # toy dataset + branch grouping
│   ├── train_dendritic.py         # train the dendritic neuron
│   ├── train_point.py             # train the point neuron baseline
│   └── evaluate_neurons.py        # compare models + noise robustness
│
├── notebooks/
│   └── 01_dendritic_vs_point.ipynb   # visual analysis and comparison
│
├── results/                       # generated output (git-ignored contents)
│   ├── models/                    # .npz parameters
│   ├── logs/                      # per-epoch history JSON
│   └── summaries/                 # comparison JSON
│
├── tests/
│   ├── conftest.py
│   ├── test_dendritic_forward.py  # forward shapes, math, edge cases
│   ├── test_point_neuron.py       # baseline + dimension mismatches
│   ├── test_gradient.py           # finite-difference gradient checks
│   ├── test_activations_and_losses.py
│   ├── test_utils_and_training.py
│   └── test_experiments.py
│
└── script/
    └── perforated_ai_dendrite_sweep.py   # study notes, import-safe
```

---

## Dendritic Optimization (conceptual background)

Modern deep learning typically uses point neurons: all inputs are combined into a single weighted sum and passed through one nonlinearity. Dendritic optimization takes inspiration from neuroscience, where biological neurons possess rich dendritic trees that perform additional nonlinear computation before signals reach the soma.

Key conceptual ideas:

- **Branches as local experts**  
  Each dendritic branch receives only a subset of the inputs and learns its own local transformation. This allows the neuron to respond differently in distinct regions of the input space.

- **Error-aware assistant signals**  
  In dendritic optimization approaches, auxiliary "dendritic" units can be trained to detect patterns where a neuron tends to make mistakes (false positives/negatives). Their outputs are then fed back to adjust the neuron's effective decision boundary without redesigning the entire network.

- **Perforated backpropagation idea (high level)**  
  Backpropagation still flows through the main neuron graph. Additional dendritic components:
  - Observe the same inputs and neuron errors.
  - Use a separate learning rule to become specialized "outlier detectors."
  - Connect back to neurons through additional weights, upgrading each neuron's decision capability while preserving the original network structure.

This repository does **not** re-implement a full PyTorch dendritic library. It recreates a minimal, from-scratch version of the core structural idea at the single-neuron level. See [`script/perforated_ai_dendrite_sweep.py`](script/perforated_ai_dendrite_sweep.py) for the Perforated AI + W&B API shape.

---

## Usage

All commands run from the project root.

### 1. Train the dendritic neuron

```bash
python -m experiments.train_dendritic
```

Generates a 6-dimensional synthetic dataset (three branches × two features per branch), splits it, builds a `DendriticNeuron`, and trains it with mini-batch gradient descent. Writes:

- `results/models/dendritic_params.npz`
- `results/logs/dendritic_history.json`

### 2. Train the point neuron baseline

```bash
python -m experiments.train_point
```

Uses the **same dataset, split and hyperparameters** as the dendritic run. Writes:

- `results/models/point_neuron_params.npz`
- `results/logs/point_neuron_history.json`

### 3. Evaluate and compare

```bash
python -m experiments.evaluate_neurons
```

Rebuilds both neurons **with the exact activations they were trained with**, loads the saved parameters, and evaluates on a fresh test set plus a Gaussian-noise variant. Writes:

- `results/summaries/comparison.json`

Both seeds are fixed, so repeated runs give identical numbers.

### 4. Notebook

```bash
jupyter notebook notebooks/01_dendritic_vs_point.ipynb
```

Trains both models, plots learning curves, inspects what each branch learned, visualizes decision boundaries and sweeps input-noise levels.

---

## Running tests

```bash
pytest
```

Or explicitly:

```bash
pytest tests -v
```

Coverage includes:

- **Forward math** — known-value tests against hand-computed branch/soma results, batch vs single-sample consistency, branch isolation.
- **Gradients** — finite-difference checks for every parameter of both neuron types, including through nonlinear activations.
- **Batch scaling** — a regression test that gradients are *summed*, not divided by batch size a second time.
- **Edge cases** — single branch, single feature per branch, overlapping branch groups, unused input features.
- **Input mismatches** — wrong and ragged input dimensions raise a clear `ValueError` rather than failing silently.
- **Experiments** — the training and evaluation paths build identical architectures, and the split is shared and stratified.

---

## Notes on the implementation

A few details worth knowing if you are reading the code:

- **Losses are mean reductions.** `mse_derivative` and `binary_cross_entropy_derivative` return the gradient of the *mean* loss, so they already include a `1/batch_size` factor. `backward()` therefore **sums** the per-sample contributions rather than averaging again.
- **Seeding is isolated.** Each neuron uses a private `random.Random(seed)` instance, so constructing a neuron never perturbs dataset generation or batch shuffling elsewhere.
- **Activations live in one place.** `src/activations.py` is the single source of truth; both neurons resolve their nonlinearity through it, and `sigmoid` is evaluated in a numerically stable way.
- **Saved parameters need no pickle.** `branch_weights` is ragged, so each branch is stored under its own numeric key (`branch_weight_0`, ...) instead of an object array. `load_params()` therefore runs with `allow_pickle` off, and saving to a new path creates that path's parents rather than the default output directory.
- **One source of truth for the experiment.** `experiments/common.py` holds every hyperparameter, the dataset and the split, so the two models can never drift apart.

---

## Possible extensions

- Introduce different activation functions per branch.
- Replace the single-neuron experiments with shallow networks where some layers use dendritic units.
- Add more complex synthetic datasets to test when dendritic structure gives the largest benefit.
- Integrate experiment tracking and hyperparameter sweeps to systematically explore number of branches, branch-to-input mappings, noise levels and regularization strategies.

---

## License

MIT — see [`LICENSE`](LICENSE).
