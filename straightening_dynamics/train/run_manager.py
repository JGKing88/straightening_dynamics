import os
import uuid
import yaml
import pandas as pd
from pathlib import Path
import datetime
import shutil
import json
from typing import Dict, Any, Optional, Union


class RunManager:
    """
    Manages experiment runs, tracking, and model saving.
    """
    
    def __init__(self, base_save_dir: str = "saved_models"):
        """
        Initialize the RunManager.
        
        Args:
            base_save_dir: Base directory to save models and run information
        """
        self.base_save_dir = Path(base_save_dir)
        self.base_save_dir.mkdir(exist_ok=True, parents=True)
        
        # Create or load the runs dataframe
        self.runs_file = self.base_save_dir / "runs.csv"
        if self.runs_file.exists():
            self.runs_df = pd.read_csv(self.runs_file)
        else:
            # Create with minimal required columns - all other columns will be added dynamically
            self.runs_df = pd.DataFrame(columns=[
                "run_id", "run_name", "timestamp", "sweep_name"
            ])
    
    def generate_run_id(self) -> str:
        """
        Generate a unique run ID.
        
        Returns:
            A unique run ID string
        """
        return str(uuid.uuid4())[:8]
    
    def create_run_dir(self, run_id: str) -> Path:
        """
        Create a directory for the run.
        
        Args:
            run_id: Unique run ID
            
        Returns:
            Path to the run directory
        """
        run_dir = self.base_save_dir / run_id
        run_dir.mkdir(exist_ok=True, parents=True)
        return run_dir
    
    def save_config(self, config: Dict[str, Any], run_dir: Path) -> None:
        """
        Save the configuration to the run directory.
        
        Args:
            config: Configuration dictionary
            run_dir: Path to the run directory
        """
        config_path = run_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
    
    def register_run(self, 
                    config: Dict[str, Any], 
                    run_id: str, 
                    sweep_name: Optional[str] = None) -> None:
        """
        Register a run in the runs dataframe.
        
        Args:
            config: Configuration dictionary
            run_id: Unique run ID
            sweep_name: Name of the sweep if part of a sweep
        """
        # Extract basic information
        run_name = config.get("run", {}).get("name", "unnamed")
        
        # Create a new row with required fields
        new_row = {
            "run_id": run_id,
            "run_name": run_name,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "sweep_name": sweep_name
        }
        
        # Add all config parameters as flattened columns
        flat_config = self._flatten_dict(config)
        for key, value in flat_config.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                new_row[key] = value
        
        # Add to dataframe
        self.runs_df = pd.concat([self.runs_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Save the updated dataframe
        self.runs_df.to_csv(self.runs_file, index=False)
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
        """
        Flatten a nested dictionary.
        
        Args:
            d: Dictionary to flatten
            parent_key: Parent key for nested dictionaries
            sep: Separator for keys
            
        Returns:
            Flattened dictionary
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            else:
                items.append((new_key, v))
        return dict(items)
    
    def setup_run(self, config: Dict[str, Any], sweep_name: Optional[str] = None) -> str:
        """
        Set up a new run.
        
        Args:
            config: Configuration dictionary
            sweep_name: Name of the sweep if part of a sweep
            
        Returns:
            Run ID and run directory
        """
        run_id = self.generate_run_id()
        run_dir = self.create_run_dir(run_id)
        self.save_config(config, run_dir)
        self.register_run(config, run_id, sweep_name)
        return run_id
    
    def get_run_dir(self, run_id: str) -> Path:
        """
        Get the directory for a run.
        
        Args:
            run_id: Unique run ID
            
        Returns:
            Path to the run directory
        """
        return self.base_save_dir / run_id
    
    def get_runs_df(self) -> pd.DataFrame:
        """
        Get the runs dataframe.
        
        Returns:
            Dataframe of runs
        """
        return self.runs_df
    
    def get_run_config(self, run_id: str) -> Dict[str, Any]:
        """
        Get the configuration for a run.
        
        Args:
            run_id: Unique run ID
            
        Returns:
            Configuration dictionary
        """
        config_path = self.get_run_dir(run_id) / "config.yaml"
        with open(config_path, "r") as f:
            return yaml.safe_load(f)
    
    def update_metrics(self, run_id: str, metrics: Dict[str, Any]) -> None:
        """
        Update metrics for a run.
        
        Args:
            run_id: Unique run ID
            metrics: Dictionary of metrics to update
        """
        # Find the row with the given run_id
        if run_id in self.runs_df["run_id"].values:
            idx = self.runs_df[self.runs_df["run_id"] == run_id].index[0]
            
            # Update metrics
            for key, value in metrics.items():
                self.runs_df.loc[idx, key] = value
            
            # Save the updated dataframe
            self.runs_df.to_csv(self.runs_file, index=False)
        else:
            print(f"Warning: Run ID {run_id} not found in runs dataframe.")
    
    def get_best_run(self, metric: str, higher_is_better: bool = False, filter_dict: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Get the run ID of the best run based on a metric.
        
        Args:
            metric: Metric to use for comparison
            higher_is_better: Whether higher values of the metric are better
            filter_dict: Dictionary of filters to apply before finding the best run
            
        Returns:
            Run ID of the best run, or None if no runs match the criteria
        """
        df = self.runs_df.copy()
        
        # Apply filters if provided
        if filter_dict:
            for key, value in filter_dict.items():
                if key in df.columns:
                    df = df[df[key] == value]
        
        # Check if metric exists and there are runs after filtering
        if metric not in df.columns or len(df) == 0:
            return None
        
        # Find the best run
        if higher_is_better:
            best_idx = df[metric].idxmax()
        else:
            best_idx = df[metric].idxmin()
        
        return df.loc[best_idx, "run_id"] 