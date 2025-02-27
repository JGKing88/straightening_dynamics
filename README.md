# Experiment Management System

This repository contains an experiment management system for training language models. The system provides a structured way to configure, run, track, and analyze model training experiments.

## Features

- **Configuration Management**: Define all hyperparameters in YAML files
- **Experiment Tracking**: Each run is assigned a unique ID and tracked in a central dataframe
- **Hyperparameter Sweeps**: Easily run experiments across multiple hyperparameter combinations
- **Model Checkpointing**: Automatically save model checkpoints with organized directory structure
- **Gradient Saving**: Save model gradients at each checkpoint for analysis
- **DeepSpeed Integration**: Leverage DeepSpeed for efficient training on GPUs
- **Flexible Hyperparameters**: Add new hyperparameters in config files without modifying code

## Getting Started

### Prerequisites

- Python 3.7+
- PyTorch
- Transformers
- Accelerate
- DeepSpeed
- Pandas
- CUDA-compatible GPU

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/straightening_dynamics.git
   cd straightening_dynamics
   ```

2. Install the package:
   ```bash
   pip install -e .
   ```

## Running Experiments

### Single Training Run

To run a single training job with the default configuration:

```bash
sbatch straightening_dynamics/train/training_scripts/train_with_config.sh
```

This will:
1. Create a unique run ID
2. Save the configuration to `saved_models/{run_id}/config.yaml`
3. Register the run in the runs dataframe
4. Train the model with the specified configuration
5. Save model checkpoints to `saved_models/{run_id}/checkpoint_{step}`
6. Save the final model to `saved_models/{run_id}/final_checkpoint`

To use a custom configuration file:

```bash
sbatch straightening_dynamics/train/training_scripts/train_with_config.sh /path/to/your/config.yaml
```

### Hyperparameter Sweeps

To run a hyperparameter sweep with the default configurations:

```bash
sbatch straightening_dynamics/train/training_scripts/run_sweep.sh
```

This will:
1. Generate multiple configurations based on the sweep configuration
2. Run each configuration sequentially
3. Track all runs in the runs dataframe with the same sweep name

To use custom configuration files:

```bash
sbatch straightening_dynamics/train/training_scripts/run_sweep.sh /path/to/base/config.yaml /path/to/sweep/config.yaml
```

## Configuration Files

### Training Configuration

The training configuration file (`straightening_dynamics/train/configs/train_config.yaml`) contains all hyperparameters for training:

```yaml
# Training configuration
model:
  name: "gpt2"  # Model name/type
  context_length: 1024  # Context length for the model

data:
  dataset_size: "10M"  # Dataset size (e.g., "10M")
  dataset_path: "/om2/user/jackking/straightening_dynamics/data"  # Path to dataset

training:
  batch_size: 32  # Batch size for training
  max_gpu_batch_size: 16  # Maximum batch size per GPU
  eval_batch_size: 16  # Batch size for evaluation
  gradient_accumulation_steps: 8  # Number of gradient accumulation steps
  learning_rate: 0.0006  # Learning rate
  num_epochs: 1  # Number of training epochs
  correct_bias: true  # Whether to correct bias in optimizer
  seed: 42  # Random seed
  scheduler: "cosine"  # Learning rate scheduler (linear, cosine)
  warmup_steps: 100  # Number of warmup steps
  save_steps: 500  # Save model every X steps
  eval_steps: 500  # Evaluate model every X steps

run:
  name: "test"  # Run name
  init_type: "normal"  # Initialization type (normal, gaussian)
  checkpoint: null  # Checkpoint to start from (null for none)
```

You can add any new hyperparameters to this file without modifying the code. The system will automatically track all hyperparameters in the runs dataframe.

### Sweep Configuration

The sweep configuration file (`straightening_dynamics/train/configs/sweep_config.yaml`) defines the hyperparameters to sweep over:

```yaml
# Hyperparameter sweep configuration
sweep_name: "learning_rate_sweep"  # Name of the sweep

parameters:
  # Example of sweeping over initialization parameters
  initialization.init_std:
    values: [0.01, 0.02]
  
  initialization.init_mean:
    values: [0.0]

# Strategy for selecting combinations (grid, random)
strategy: "grid"

# Number of runs to perform with different combinations if strategy is random.
num_runs: 3
```

You can specify either `grid` or `random` as the strategy:
- `grid`: Try all possible combinations of hyperparameters
- `random`: Randomly sample `num_runs` combinations of hyperparameters

## Viewing and Analyzing Results

### Viewing Runs

To view all runs:

```bash
python straightening_dynamics/train/view_runs.py
```

To list all available columns:

```bash
python straightening_dynamics/train/view_runs.py --list_columns
```

To filter by sweep name:

```bash
python straightening_dynamics/train/view_runs.py --filter_sweep learning_rate_sweep
```

To filter by specific column values:

```bash
python straightening_dynamics/train/view_runs.py --filter "model.name=gpt2" "training.learning_rate=0.0006"
```

To sort by a specific column (e.g., validation loss):

```bash
python straightening_dynamics/train/view_runs.py --sort_by final_valid_loss
```

To sort in descending order:

```bash
python straightening_dynamics/train/view_runs.py --sort_by final_valid_loss --descending
```

To show only specific columns:

```bash
python straightening_dynamics/train/view_runs.py --show run_id run_name final_valid_loss final_valid_perplexity
```

To hide specific columns:

```bash
python straightening_dynamics/train/view_runs.py --hide timestamp
```

To limit the number of rows displayed:

```bash
python straightening_dynamics/train/view_runs.py --limit 10
```

### Run Directory Structure

Each run is stored in a directory with the following structure:

```
saved_models/
├── runs.csv                      # Central dataframe of all runs
├── {run_id}/                     # Directory for a specific run
│   ├── config.yaml               # Configuration used for this run
│   ├── checkpoint_{step}/        # Model checkpoint at a specific step
│   │   ├── pytorch_model.bin     # Model weights
│   │   ├── config.json           # Model configuration
│   │   ├── gradients.pt          # Model gradients at this step
│   │   └── accelerator_states    # Optimizer and scheduler states
│   └── final_checkpoint/         # Final model checkpoint
│       ├── pytorch_model.bin     # Model weights
│       ├── config.json           # Model configuration
│       ├── gradients.pt          # Final model gradients
│       └── accelerator_states    # Optimizer and scheduler states
```

## Creating Custom Configurations

To create a custom configuration:

1. Copy the default configuration file:
   ```bash
   cp straightening_dynamics/train/configs/train_config.yaml my_config.yaml
   ```

2. Edit the configuration file with your desired hyperparameters. You can add any new hyperparameters you need.

3. Run the training script with your custom configuration:
   ```bash
   sbatch straightening_dynamics/train/training_scripts/train_with_config.sh my_config.yaml
   ```

## Creating Custom Sweeps

To create a custom sweep:

1. Copy the default sweep configuration file:
   ```bash
   cp straightening_dynamics/train/configs/sweep_config.yaml my_sweep.yaml
   ```

2. Edit the sweep configuration file with your desired hyperparameters to sweep over.

3. Run the sweep script with your custom configurations:
   ```bash
   sbatch straightening_dynamics/train/training_scripts/run_sweep.sh my_config.yaml my_sweep.yaml
   ```

## Tips for Efficient Experimentation

1. **Start Small**: Begin with small models and datasets to verify your setup.

2. **Monitor Resources**: Use `squeue` and `sinfo` to monitor your jobs on the cluster.

3. **Organize Sweeps**: Use descriptive sweep names to easily identify related experiments.

4. **Analyze Results**: Use the `view_runs.py` script to compare results across different runs.

5. **Checkpoint Management**: Older checkpoints can be safely deleted to save disk space, as the final model is always saved.

6. **Add Custom Metrics**: You can update the runs dataframe with custom metrics using the `update_metrics` method of the `RunManager` class.

## Troubleshooting

- **Out of Memory Errors**: Reduce batch size or increase gradient accumulation steps.
- **Job Failures**: Check the slurm output files for error messages.
- **Dataset Issues**: Verify the dataset path in your configuration file.

## Git Configuration

This repository is configured to exclude model checkpoints and gradients from version control while preserving the directory structure. This approach keeps the repository size manageable while ensuring collaborators have the same directory structure.

### What's excluded from Git:
- All checkpoint directories (`saved_models/**/checkpoint_*/`)
- Final checkpoint directories (`saved_models/**/final_checkpoint/`)
- Model files (`.pt`, `.bin`, `.json`, `.h5`)
- Gradient files (included in the `.pt` exclusion)

### How the directory structure is preserved:
- The `saved_models` directory itself is tracked
- Empty `.gitkeep` files maintain the directory structure

### For collaborators:
When cloning this repository, you'll need to:
1. Train your own models, or
2. Obtain model checkpoints through other means (shared drive, cloud storage)
3. Place them in the appropriate `saved_models/{run_id}/` directories

### For backing up important models:
Since model checkpoints aren't in Git, use alternative backup strategies:
- Copy important checkpoints to a separate storage location
- Use cloud storage for long-term archiving
- Document which runs produced the best results

## Analyzing Gradients

The system now automatically saves model gradients at each checkpoint. These gradients are stored as PyTorch tensors in a dictionary format, where the keys are parameter names and the values are the corresponding gradient tensors.

To load and analyze the gradients:

```python
import torch
import matplotlib.pyplot as plt
import numpy as np

# Load gradients from a checkpoint
gradients = torch.load('saved_models/{run_id}/checkpoint_{step}/gradients.pt')

# Example: Compute gradient norm for each layer
gradient_norms = {}
for name, grad in gradients.items():
    gradient_norms[name] = torch.norm(grad).item()

# Plot gradient norms
plt.figure(figsize=(12, 8))
plt.bar(range(len(gradient_norms)), list(gradient_norms.values()))
plt.xticks(range(len(gradient_norms)), list(gradient_norms.keys()), rotation=90)
plt.xlabel('Layer')
plt.ylabel('Gradient Norm')
plt.title('Gradient Norms Across Layers')
plt.tight_layout()
plt.savefig('gradient_norms.png')
```

You can also compare gradients across different checkpoints to analyze how they evolve during training:

```python
# Load gradients from multiple checkpoints
checkpoint_steps = [500, 1000, 1500, 2000]
all_gradients = {}

for step in checkpoint_steps:
    all_gradients[step] = torch.load(f'saved_models/{run_id}/checkpoint_{step}/gradients.pt')

# Example: Compare gradient norms for a specific layer across checkpoints
layer_name = 'transformer.h.0.mlp.c_fc.weight'
layer_gradient_norms = []

for step in checkpoint_steps:
    if layer_name in all_gradients[step]:
        layer_gradient_norms.append(torch.norm(all_gradients[step][layer_name]).item())
    else:
        layer_gradient_norms.append(0)

plt.figure(figsize=(10, 6))
plt.plot(checkpoint_steps, layer_gradient_norms, marker='o')
plt.xlabel('Training Step')
plt.ylabel('Gradient Norm')
plt.title(f'Gradient Norm Evolution for {layer_name}')
plt.grid(True)
plt.savefig('gradient_evolution.png')
```

These are just basic examples. You can perform more sophisticated analyses such as:

- Comparing gradient directions between layers
- Analyzing gradient variance across batches
- Identifying layers with vanishing or exploding gradients
- Correlating gradient behavior with model performance

## Advanced Usage

### Continuing from a Checkpoint

To continue training from a checkpoint, set the `checkpoint` parameter in your configuration:

```yaml
run:
  name: "continued_run"
  init_type: "normal"
  checkpoint: "saved_models/abcd1234/checkpoint_1000"  # Path to the checkpoint
```

### Custom Initialization

To use Gaussian initialization instead of the default initialization:

```yaml
run:
  name: "gaussian_init"
  init_type: "gaussian"
  checkpoint: null
```

### Adding Custom Metrics

You can add custom metrics to the runs dataframe by updating the train.py script:

```python
# Update metrics in the runs dataframe
run_manager.update_metrics(run_id, {
    "my_custom_metric": value,
    "another_metric": another_value
})
```

These metrics will be automatically added as columns to the runs dataframe and can be used for filtering and sorting. 