import pandas as pd
import matplotlib.pyplot as plt
import argparse
import os

def create_final_plot(args):
    """
    Loads experiment results and generates a plot.
    This version robustly handles data format, naming, and type errors.
    """
    print("📊 Generating final plot from your results...")

    try:
        # Load the data based on its true structure
        accuracy_df = pd.read_csv(
            os.path.join(args.results_dir, "accuracy_results.csv"),
            index_col=0
        )
        bwt_df = pd.read_csv(
            os.path.join(args.results_dir, "bwt_results.csv"),
            index_col=0
        )
        complexity_df = pd.read_csv(
            os.path.join(args.results_dir, "complexity_results.csv"),
            index_col=0
        ).T # Transpose to easily access values

    except FileNotFoundError as e:
        print(f"❌ Error: Could not find a results file. {e}")
        return
    except Exception as e:
        print(f"❌ An error occurred while loading data: {e}")
        return

    # --- Plotting ---
    fig, ax = plt.subplots(figsize=(10, 6))

    task_labels = [f"Task {i+1}" for i in accuracy_df.index]
    ax.bar(task_labels, accuracy_df['bicl'], label='Per-Task Accuracy', color='skyblue')

    ax.set_title('Experiment Results for BICL Method', fontsize=16, pad=20)
    ax.set_xlabel('Tasks', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    for i, acc in enumerate(accuracy_df['bicl']):
        ax.text(i, acc + 0.02, f'{acc:.2%}', ha='center', fontweight='bold')

    # --- Robust Annotations for other metrics ---
    # This 'errors='coerce'' argument is the key fix. It turns non-numeric
    # strings like "time" into a special 'Not a Number' (NaN) value.
    bwt_value = pd.to_numeric(bwt_df.iloc[0, 0], errors='coerce')
    
    complexity_metric_name = complexity_df.index[0]
    complexity_value = pd.to_numeric(complexity_df.iloc[0, 0], errors='coerce')

    # Format values for display, checking for NaN to show 'N/A' if needed.
    bwt_str = f"{bwt_value:.4f}" if pd.notna(bwt_value) else "N/A"
    complexity_str = f"{complexity_value:.2f}" if pd.notna(complexity_value) else "N/A"

    stats_text = (
        f"Key Metrics:\n"
        f"-----------------\n"
        f"Forgetting (BWT): {bwt_str}\n"
        f"{complexity_metric_name}: {complexity_str}"
    )
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=props)
    
    output_path = os.path.join(args.results_dir, 'final_results_plot.png')
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    print(f"\n🎉 Success! Plot saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a final plot from experiment results.")
    parser.add_argument('--results_dir', type=str, required=True, help='Directory containing saved results.')
    args = parser.parse_args()
    create_final_plot(args)