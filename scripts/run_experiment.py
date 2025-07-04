import argparse
import yaml
import numpy as np
import torch
import pandas as pd
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.experiment import ExperimentRunner

def main(args):
    """Loads config, runs the comprehensive experiment, and saves results to CSV files."""
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override config with command-line args if provided
    config['experiment']['seed'] = args.seed
    config['experiment']['num_runs'] = args.num_runs

    # Set seeds for reproducibility
    np.random.seed(config['experiment']['seed'])
    torch.manual_seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config['experiment']['seed'])

    print("BIOLOGICAL-INSPIRED CONTINUAL LEARNING EXPERIMENT")
    print("=" * 60)
    
    runner = ExperimentRunner(config)
    results, forgetting, complexity = runner.run_comprehensive_experiment()

    # Create results directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Process and save results
    results_flat = {(m, t): res for m, tasks in results.items() for t, res in tasks.items()}
    results_df = pd.DataFrame.from_dict(results_flat, orient='index')
    results_df.index = pd.MultiIndex.from_tuples(results_df.index, names=['method', 'task'])
    results_df.to_csv(os.path.join(args.output_dir, "raw_accuracy_results.csv"))

    forgetting_df = pd.DataFrame(forgetting)
    forgetting_df.to_csv(os.path.join(args.output_dir, "forgetting_results.csv"))

    complexity_df = pd.DataFrame.from_dict({(m, k): v for m, d in complexity.items() for k, v in d.items()}, orient='index')
    complexity_df.index = pd.MultiIndex.from_tuples(complexity_df.index, names=['method', 'metric'])
    complexity_df.to_csv(os.path.join(args.output_dir, "complexity_results.csv"))
    
    print(f"\n✅ All raw results saved to '{args.output_dir}'")
    print("To generate plots, run: python scripts/plot_results.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the BICL experiment.")
    parser.add_argument('--config', type=str, default='configs/experiment_config.yaml', help='Path to config file.')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results.')
    parser.add_argument('--num_runs', type=int, default=5, help='Number of statistical runs (overrides config).')
    parser.add_argument('--seed', type=int, default=42, help='Random seed (overrides config).')
    args = parser.parse_args()
    main(args)