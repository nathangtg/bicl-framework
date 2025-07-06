# scripts/run_experiment.py
import argparse
import yaml
import numpy as np
import torch
import pandas as pd
import os
import sys
import logging
from datetime import datetime

# ==============================================================================
# Setup & Configuration
# ==============================================================================

# Add project root to Python path for robust module imports
# This allows you to run this script from anywhere in the project
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiment import ExperimentRunner # Assumes ExperimentRunner is in src/experiment.py

def setup_logging(log_dir: str):
    """
    Configures a logger to output to both a file and the console. This is
    essential for capturing detailed experiment traces for debugging and records.

    Args:
        log_dir (str): The directory where the log file will be saved.
    """
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    # Log to file for a persistent record
    file_handler = logging.FileHandler(os.path.join(log_dir, "experiment.log"))
    file_handler.setFormatter(log_formatter)
    root_logger.addHandler(file_handler)

    # Log to console for real-time feedback
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    root_logger.addHandler(console_handler)

# ==============================================================================
# Main Execution Logic
# ==============================================================================

def main(args):
    """
    Main function to orchestrate the continual learning experiment.
    
    This function handles:
    1. Loading configuration from the specified YAML file.
    2. Setting up timestamped output directories to prevent overwriting results.
    3. Configuring a robust logging system.
    4. Seeding all random number generators for full reproducibility.
    5. Running the experiment via the ExperimentRunner class.
    6. Processing and saving all results (accuracies, forgetting, complexity) to CSV files.
    """
    # --- 1. Load Configuration ---
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line args for flexibility
    config['experiment']['seed'] = args.seed
    config['experiment']['num_runs'] = args.num_runs

    # --- 2. Setup Output Directory & Logging ---
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(args.output_dir, f"{config['experiment']['name']}_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    setup_logging(output_dir)
    logging.info(f"Experiment results will be saved to: {output_dir}")

    # Save the exact config file used for this run. This is CRITICAL for reproducibility.
    with open(os.path.join(output_dir, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, default_flow_style=False)
    logging.info("Saved a copy of the configuration file for this run.")

    # --- 3. Set Seeds for Reproducibility ---
    seed = config['experiment']['seed']
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        logging.info(f"Running on CUDA with random seed {seed}.")
    else:
        logging.info(f"Running on CPU with random seed {seed}.")

    # --- 4. Run Experiment ---
    try:
        logging.info("Initializing ExperimentRunner...")
        runner = ExperimentRunner(config)
        
        logging.info("Starting comprehensive experiment runs...")
        # The runner now returns dictionaries where keys are trial IDs and values are lists of results
        accuracies, bwt_results, complexity = runner.run_comprehensive_experiment()
        logging.info("Comprehensive experiment finished successfully.")

        # --- 5. Process and Save Results ---
        logging.info("Processing and saving results to CSV files...")
        
        # FIX: The 'accuracies' and 'bwt_results' dicts are already in a format 
        # suitable for a DataFrame, where keys are trial IDs (columns) and 
        # each list item is a row (a statistical run).
        accuracies_df = pd.DataFrame(accuracies)
        accuracies_df.to_csv(os.path.join(output_dir, "accuracy_results.csv"))

        bwt_df = pd.DataFrame(bwt_results)
        bwt_df.to_csv(os.path.join(output_dir, "bwt_results.csv"))

        # The complexity processing logic was already correct
        complexity_df = pd.DataFrame.from_dict({
            (trial_id, metric): values 
            for trial_id, metrics_dict in complexity.items() 
            for metric, values in metrics_dict.items()
        }, orient='index')
        complexity_df.index = pd.MultiIndex.from_tuples(complexity_df.index, names=['trial_id', 'metric'])
        complexity_df.to_csv(os.path.join(output_dir, "complexity_results.csv"))
        
        logging.info(f"\n✅ All raw results have been successfully saved to '{output_dir}'")
        logging.info(f"To generate plots, run: python scripts/plot_results.py --results_dir {output_dir}")

    except Exception as e:
        logging.error("An error occurred during the experiment run.", exc_info=True)
        sys.exit(1) # Exit with an error code

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Bio-Inspired Continual Learning (BICL) experiment.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml', 
                        help='Path to the master YAML configuration file.')
    parser.add_argument('--output_dir', type=str, default='results', 
                        help='Parent directory to save experiment results.')
    parser.add_argument('--num_runs', type=int, default=5, 
                        help='Number of statistical runs (overrides config).')
    parser.add_argument('--seed', type=int, default=42, 
                        help='Random seed for reproducibility (overrides config).')
    
    args = parser.parse_args()
    main(args)
