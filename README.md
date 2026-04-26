# A Neurocomputational Theory of Adaptive Forgetting

This repository contains the official PyTorch implementation for the paper **"Moving On: Toward a Neurocomputational Theory of Adaptive Forgetting."**

The implementation focuses on continual learning under task sequences, with a specific emphasis on the BICL mechanism and its stability–plasticity tradeoff.

## Repository Layout

At the repository root:

- `bicl-framework/`: main Python project package and experiment code.
- `Final_BICL_Investigation.ipynb`: exploratory notebook.
- `Moving_On_V2.pdf`: manuscript/reference document.

Inside `bicl-framework/`:

- `configs/`: YAML experiment configurations.
- `scripts/`: runnable entrypoints (`run_experiment.py`, plotting scripts).
- `src/`: core implementation.
  - `frameworks.py`: BICL and EWC logic.
  - `experiment.py`: training/evaluation orchestration over task streams.
  - `data.py`: synthetic task generation and benchmark task wrappers.
  - `model.py`: model factory (ResNet18, MLP, TinyNet).
  - `utils.py`, `plotting.py`: helpers and visualization support.
- `tests/`: framework-level tests.
- `requirements.txt`: pinned Python dependencies.

## Architecture Overview

The codebase is organized around three layers:

1. **Task/data layer (`src/data.py`)**  
   Produces continual-learning task sequences from either benchmark datasets (e.g., Split CIFAR, Permuted MNIST) or synthetic generators.

2. **Learning framework layer (`src/frameworks.py`)**  
   Encapsulates method-specific regularization logic:
   - `EWC`: Fisher-based consolidation baseline.
   - `BICLFramework`: bio-inspired consolidation + homeostatic regularization with online importance updates.

3. **Experiment orchestration layer (`src/experiment.py`, `scripts/run_experiment.py`)**  
   Drives end-to-end runs: configuration loading, trial generation, per-task training, periodic evaluation, and metric/result export.

This separation keeps method logic independent from data generation and experiment orchestration, making framework comparisons and ablations easier to run.

## What Is Novel Here

Compared with standard continual-learning baselines, the BICL implementation introduces a specific training-time structure:

- **Loss–update decoupling for autograd compatibility**:  
  BICL computes regularized loss (`calculate_loss`) during forward/backward, then performs importance-weight updates in a dedicated post-backward hook (`after_backward_update`) before optimizer step.

- **Dual regularization perspective**:  
  It combines:
  - **Synaptic consolidation** (parameter anchoring to previous task state using learned importance weights), and
  - **Homeostatic regulation** (parameter-range regularization to preserve stable internal dynamics).

- **Sequential task-state memory**:  
  At task boundaries, reference parameters are explicitly snapshot (`on_task_finish`) to carry forward biologically motivated memory constraints.

Together, these pieces provide a practical and explicit implementation of adaptive forgetting dynamics rather than only a static penalty term.

## How to Run

Install dependencies:

```bash
cd bicl-framework
python -m pip install -r requirements.txt
```

Run the main experiment:

```bash
cd bicl-framework
python scripts/run_experiment.py
```

The default run reads `configs/experiment_config.yaml`, executes configured methods/trials, and writes timestamped outputs (metrics/config/logs) under the configured output directory.

## Customization

Edit `bicl-framework/configs/experiment_config.yaml` to change:

- benchmark/task setup (`data`)
- model family and capacity (`model`)
- optimization controls (`training`)
- framework-specific hyperparameters (`frameworks`)
- trial definitions and number of statistical runs (`experiment`)

## Notes

- The current test suite exists under `bicl-framework/tests/`.
- If you compare methods, ensure `experiment.methods_to_run` includes each target framework (`vanilla`, `ewc`, `bicl`) and matching framework parameters.
