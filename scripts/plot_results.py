import pandas as pd
import argparse
import os
import sys

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.plotting import create_publication_quality_figures, compute_statistics_from_df

def main(args):
    """Loads saved results, computes stats, and generates publication-quality figures."""
    print("📊 Generating figures from saved results...")
    
    # Load data
    try:
        results_df = pd.read_csv(os.path.join(args.results_dir, "raw_accuracy_results.csv"), index_col=[0, 1])
        forgetting_df = pd.read_csv(os.path.join(args.results_dir, "forgetting_results.csv"), index_col=0)
        complexity_raw = pd.read_csv(os.path.join(args.results_dir, "complexity_results.csv"), index_col=[0, 1])
    except FileNotFoundError:
        print(f"Error: Results files not found in '{args.results_dir}'.")
        print("Please run 'python scripts/run_experiment.py' first.")
        return
        
    # Compute summary stats for complexity
    complexity_stats = complexity_raw.T.describe().T[['mean', 'std']]
    complexity_stats['sem'] = complexity_stats['std'] / (len(complexity_raw.columns)**0.5)

    # Compute stats for plotting
    stats_results, forgetting_stats, significance_tests = compute_statistics_from_df(results_df, forgetting_df)
    
    # Generate figure
    create_publication_quality_figures(stats_results, forgetting_stats, significance_tests, complexity_stats)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from experiment results.")
    parser.add_argument('--results_dir', type=str, default='results', help='Directory containing saved results.')
    args = parser.parse_args()
    main(args)