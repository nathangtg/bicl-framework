# Bio-Inspired Continual Learning Framework (BICL)

A comprehensive research framework for evaluating bio-inspired continual learning algorithms, focusing on catastrophic forgetting mitigation and adaptive learning strategies.

## 🧠 Overview

This project implements and compares multiple continual learning approaches inspired by biological neural mechanisms. The framework provides a systematic evaluation of how different algorithms handle sequential task learning while preserving previously acquired knowledge.

### Key Features

- **Enhanced Unified Framework**: Bio-inspired approach combining adaptive regularization with experience replay
- **Enhanced Elastic Weight Consolidation (EWC)**: Fisher information-based parameter protection
- **Comprehensive Evaluation**: Statistical analysis across multiple runs with significance testing
- **Progressive Task Difficulty**: Dynamically adjusting task complexity to simulate realistic learning scenarios
- **Publication-Quality Visualizations**: Automated generation of research-grade figures and plots

## 🏗️ Architecture

### Core Components

```
src/
├── frameworks.py      # Continual learning algorithms (Unified, EWC)
├── model.py          # Neural network architecture definitions
├── data.py           # Task generation and data management
├── experiment.py     # Experimental pipeline and evaluation
├── plotting.py       # Statistical analysis and visualization
└── utils.py          # Utility functions and helpers
```

### Algorithms Implemented

1. **Enhanced Unified Framework**
   - Adaptive β-parameter tuning based on performance trends
   - Importance-weighted parameter divergence
   - Experience replay with memory buffer management
   - Multiple divergence metrics (L2, adaptive KL-divergence)

2. **Enhanced Elastic Weight Consolidation (EWC)**
   - Fisher Information Matrix computation
   - Progressive parameter importance accumulation
   - Regularization-based catastrophic forgetting prevention

3. **Vanilla Baseline**
   - Standard sequential training without continual learning mechanisms

## 📊 Experimental Design

### Task Generation
- **Progressive Difficulty**: Tasks become increasingly challenging through reduced class separation and increased feature complexity
- **Rotation & Scaling**: Geometric transformations to simulate domain shift
- **Controlled Noise**: Systematic noise injection for robustness evaluation

### Evaluation Metrics
- **Task-wise Accuracy**: Performance on individual tasks after sequential training
- **Catastrophic Forgetting**: Quantitative measure of knowledge loss
- **Forward Transfer**: Knowledge transfer to new tasks
- **Computational Complexity**: Training time and memory usage analysis
- **Statistical Significance**: Rigorous statistical testing across multiple runs

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt
```

### Running Experiments

1. **Configure Experiment Parameters**
   ```bash
   # Edit configuration file
   vim configs/experiment_config.yaml
   ```

2. **Execute Full Experimental Pipeline**
   ```bash
   # Run with default configuration
   python scripts/run_experiment.py

   # Custom configuration
   python scripts/run_experiment.py --config custom_config.yaml --num_runs 10 --seed 123
   ```

3. **Generate Publication Figures**
   ```bash
   python scripts/plot_results.py --results_dir results/
   ```

### Configuration Options

Key parameters in `configs/experiment_config.yaml`:

```yaml
experiment:
  num_runs: 5              # Statistical runs for significance
  methods_to_run: ['vanilla', 'unified', 'ewc']

data:
  num_tasks: 5             # Sequential learning tasks
  n_samples_per_task: 2000 # Dataset size per task
  task_difficulty: 'progressive'  # Difficulty progression

frameworks:
  ewc:
    lambda: 1000.0         # Regularization strength
  unified:
    beta_values: [0.1, 0.5, 1.0, 2.0, 5.0]  # Stability-plasticity trade-off
    memory_buffer_size: 1000  # Experience replay capacity
```

## 📈 Results & Analysis

The framework generates comprehensive analysis including:

### Performance Metrics
- **Panel A**: Task-wise performance comparison across methods
- **Panel B**: Catastrophic forgetting quantification
- **Panel C**: Statistical significance heatmaps (p-values)
- **Panel D**: Computational complexity analysis

### Behavioral Analysis
- **Panel E**: Learning curve dynamics
- **Panel F**: Forward transfer capabilities
- **Panel G**: Memory efficiency trends
- **Panel H**: Hyperparameter sensitivity analysis

### Output Files
```
results/
├── raw_accuracy_results.csv    # Task-wise accuracy data
├── forgetting_results.csv      # Catastrophic forgetting metrics
├── complexity_results.csv      # Computational complexity data
└── comprehensive_results.png   # Publication-quality figure
```

## 🧪 Testing

Run the test suite to verify algorithm implementations:

```bash
# Run all tests
pytest tests/

# Specific test cases
pytest tests/test_frameworks.py::test_ewc_loss_increases_with_divergence
pytest tests/test_frameworks.py::test_unified_framework_memory_buffer_capacity
```

## 📚 Research Applications

This framework is designed for:

- **Continual Learning Research**: Algorithm development and comparison
- **Catastrophic Forgetting Analysis**: Systematic evaluation of forgetting mitigation
- **Bio-Inspired Learning**: Investigation of brain-inspired computational mechanisms
- **Educational Purposes**: Teaching continual learning concepts and implementations

## 🔬 Key Research Questions Addressed

1. **How do different regularization strategies affect knowledge retention?**
2. **What is the optimal balance between stability and plasticity in sequential learning?**
3. **How does task difficulty progression impact continual learning performance?**
4. **Which bio-inspired mechanisms are most effective for catastrophic forgetting prevention?**

## 📊 Statistical Rigor

- **Multiple Runs**: All experiments conducted across multiple random seeds
- **Significance Testing**: Statistical hypothesis testing for method comparison
- **Effect Size Analysis**: Cohen's d and confidence intervals
- **Reproducibility**: Comprehensive random seed management

## 🛠️ Customization & Extension

### Adding New Algorithms

1. Implement algorithm class in `src/frameworks.py`
2. Add method to `methods_to_run` in configuration
3. Update experimental pipeline in `src/experiment.py`

### Custom Task Generation

Modify `src/data.py` to implement domain-specific task sequences:

```python
def generate_custom_tasks(config):
    # Implement custom task generation logic
    pass
```

### Advanced Visualization

Extend `src/plotting.py` for custom analysis plots:

```python
def create_custom_analysis(data):
    # Implement domain-specific visualizations
    pass
```

## 📖 Citation

If you use this framework in your research, please cite:

```bibtex
@software{bicl_framework,
  title={Bio-Inspired Continual Learning Framework},
  author={[Your Name]},
  year={2025},
  url={https://github.com/your-repo/bicl-framework}
}
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch (`git checkout -b feature/new-algorithm`)
3. Commit changes (`git commit -am 'Add new continual learning algorithm'`)
4. Push to branch (`git push origin feature/new-algorithm`)
5. Create Pull Request

## 📧 Contact

For questions, issues, or collaboration opportunities:
- **Issues**: [GitHub Issues](https://github.com/your-repo/bicl-framework/issues)
- **Discussions**: [GitHub Discussions](https://github.com/your-repo/bicl-framework/discussions)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Note**: This framework is designed for research purposes. For production applications, additional optimization and validation may be required.
