# A Neurocomputational Theory of Adaptive Forgetting

This repository contains the official PyTorch implementation for the paper "Moving On: Toward a Neurocomputational Theory of Adaptive Forgetting".

## Project Structure

- `configs/`: YAML configuration files for experiments.
- `scripts/`: Executable scripts to run experiments and plot results.
- `src/`: Main source code.
  - `data.py`: Data loading and task generation (e.g., SplitMNIST).
  - `model.py`: Neural network architectures.
  - `frameworks.py`: Implementations of the Continual Learning methods (Vanilla, EWC, BICL).
  - `experiment.py`: The main experiment runner class.
  - `utils.py`: Helper functions.
  - `plotting.py`: (Optional) Code to generate plots.
- `README.md`: This file.
- `requirements.txt`: Python package dependencies.

## How to Run

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Run the Main Experiment:**
    ```bash
    python scripts/run_experiment.py
    ```

    The script will load the configuration from `configs/experiment_config.yaml`, run the specified number of experiments for each method, and print a formatted summary of the results, including statistical tests.

3.  **Customize Your Experiment:**
    Modify `configs/experiment_config.yaml` to change hyperparameters, the number of runs, the model architecture, or the methods being tested.