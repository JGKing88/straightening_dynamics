import yaml
import random
import itertools
from typing import Dict, Any, List, Tuple, Optional
import copy
import os
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load a configuration file.
    
    Args:
        config_path: Path to the configuration file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def set_nested_value(config: Dict[str, Any], key_path: str, value: Any) -> Dict[str, Any]:
    """
    Set a value in a nested dictionary using a dot-separated key path.
    
    Args:
        config: Configuration dictionary
        key_path: Dot-separated key path (e.g., "training.learning_rate")
        value: Value to set
        
    Returns:
        Updated configuration dictionary
    """
    keys = key_path.split(".")
    current = config
    
    # Navigate to the nested dictionary
    for key in keys[:-1]:
        if key not in current:
            current[key] = {}
        current = current[key]
    
    # Set the value
    current[keys[-1]] = value
    
    return config


def generate_grid_configs(base_config: Dict[str, Any], 
                         sweep_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Generate configurations for a grid search.
    
    Args:
        base_config: Base configuration dictionary
        sweep_config: Sweep configuration dictionary
        
    Returns:
        List of configuration dictionaries
    """
    # Extract parameters to sweep over
    param_dict = {}
    for param_path, param_config in sweep_config["parameters"].items():
        param_dict[param_path] = param_config["values"]
    
    # Generate all combinations
    param_names = list(param_dict.keys())
    param_values = [param_dict[name] for name in param_names]
    combinations = list(itertools.product(*param_values))
    
    # Create configurations for each combination
    configs = []
    for combo in combinations:
        config = copy.deepcopy(base_config)
        for i, param_name in enumerate(param_names):
            config = set_nested_value(config, param_name, combo[i])
        configs.append(config)
    
    return configs


def generate_random_configs(base_config: Dict[str, Any], 
                           sweep_config: Dict[str, Any], 
                           num_configs: int) -> List[Dict[str, Any]]:
    """
    Generate configurations for a random search.
    
    Args:
        base_config: Base configuration dictionary
        sweep_config: Sweep configuration dictionary
        num_configs: Number of configurations to generate
        
    Returns:
        List of configuration dictionaries
    """
    # Extract parameters to sweep over
    param_dict = {}
    for param_path, param_config in sweep_config["parameters"].items():
        param_dict[param_path] = param_config["values"]
    
    # Generate random combinations
    configs = []
    for _ in range(num_configs):
        config = copy.deepcopy(base_config)
        for param_name, param_values in param_dict.items():
            random_value = random.choice(param_values)
            config = set_nested_value(config, param_name, random_value)
        configs.append(config)
    
    return configs


def generate_sweep_configs(base_config_path: str, 
                          sweep_config_path: str) -> Tuple[List[Dict[str, Any]], str]:
    """
    Generate configurations for a hyperparameter sweep.
    
    Args:
        base_config_path: Path to the base configuration file
        sweep_config_path: Path to the sweep configuration file
        
    Returns:
        List of configuration dictionaries and sweep name
    """
    # Load configurations
    base_config = load_config(base_config_path)
    sweep_config = load_config(sweep_config_path)
    
    # Get sweep strategy and number of runs
    strategy = sweep_config.get("strategy", "random")
    num_runs = sweep_config.get("num_runs", 1)
    sweep_name = sweep_config.get("sweep_name", "unnamed_sweep")
    
    # Generate configurations
    if strategy == "grid":
        configs = generate_grid_configs(base_config, sweep_config)
    else:  # random
        configs = generate_random_configs(base_config, sweep_config, num_runs)
    
    # Update run names to include sweep information
    for i, config in enumerate(configs):
        if "run" not in config:
            config["run"] = {}
        config["run"]["name"] = f"{sweep_name}_{i+1}"
    
    return configs, sweep_name 