import argparse
import os
import yaml
from pathlib import Path
import subprocess
import sys
from straightening_dynamics.train.sweep_utils import generate_sweep_configs
from straightening_dynamics.train.run_manager import RunManager
from straightening_dynamics.train.train import train


def run_sweep(base_config_path, sweep_config_path):
    """
    Run a hyperparameter sweep.
    
    Args:
        base_config_path: Path to the base configuration file
        sweep_config_path: Path to the sweep configuration file
    """
    # Generate configurations for the sweep
    configs, sweep_name = generate_sweep_configs(base_config_path, sweep_config_path)
    
    # Initialize run manager
    run_manager = RunManager()
    
    # Run each configuration
    for i, config in enumerate(configs):
        print(f"Running configuration {i+1}/{len(configs)}")
        print(f"Configuration: {config}")
        
        # Register the run and get a run ID
        run_id = run_manager.setup_run(config, sweep_name)
        
        # Train the model with this configuration
        train(config, run_id)
        
        print(f"Completed run {i+1}/{len(configs)} with run ID: {run_id}")
    
    print(f"Sweep '{sweep_name}' completed with {len(configs)} runs.")


def main():
    parser = argparse.ArgumentParser(description="Run a hyperparameter sweep")
    parser.add_argument("--base_config", type=str, default="straightening_dynamics/train/configs/train_config.yaml",
                        help="Path to the base configuration file")
    parser.add_argument("--sweep_config", type=str, default="straightening_dynamics/train/configs/sweep_config.yaml",
                        help="Path to the sweep configuration file")
    args = parser.parse_args()
    
    run_sweep(args.base_config, args.sweep_config)


if __name__ == "__main__":
    main() 