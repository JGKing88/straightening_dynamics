import argparse
import pandas as pd
from pathlib import Path
import re
from straightening_dynamics.train.run_manager import RunManager


def view_runs(base_save_dir="saved_models", filter_sweep=None, filter_dict=None, sort_by=None, ascending=True, 
              show_columns=None, hide_columns=None, limit=None):
    """
    View the runs dataframe.
    
    Args:
        base_save_dir: Base directory where models are saved
        filter_sweep: Filter by sweep name
        filter_dict: Dictionary of column-value pairs to filter by
        sort_by: Column to sort by
        ascending: Sort in ascending order
        show_columns: List of columns to show (if None, show all)
        hide_columns: List of columns to hide
        limit: Maximum number of rows to display
    """
    # Initialize run manager
    run_manager = RunManager(base_save_dir)
    
    # Get runs dataframe
    runs_df = run_manager.get_runs_df()
    
    # Filter by sweep name if specified
    if filter_sweep:
        runs_df = runs_df[runs_df["sweep_name"] == filter_sweep]
    
    # Apply additional filters if specified
    if filter_dict:
        for col, value in filter_dict.items():
            if col in runs_df.columns:
                # Handle regex patterns
                if isinstance(value, str) and (value.startswith('/') and value.endswith('/')):
                    pattern = value[1:-1]  # Remove the slashes
                    runs_df = runs_df[runs_df[col].astype(str).str.match(pattern)]
                else:
                    runs_df = runs_df[runs_df[col] == value]
    
    # Sort if specified
    if sort_by:
        if sort_by in runs_df.columns:
            runs_df = runs_df.sort_values(by=sort_by, ascending=ascending)
        else:
            print(f"Warning: Column '{sort_by}' not found for sorting.")
    
    # Select columns to display
    if show_columns:
        # Always include run_id and run_name
        if "run_id" not in show_columns:
            show_columns = ["run_id"] + show_columns
        if "run_name" not in show_columns:
            show_columns = ["run_name"] + show_columns
        
        # Filter to only existing columns
        show_columns = [col for col in show_columns if col in runs_df.columns]
        display_df = runs_df[show_columns]
    else:
        display_df = runs_df.copy()
    
    # Hide columns if specified
    if hide_columns:
        for col in hide_columns:
            if col in display_df.columns:
                display_df = display_df.drop(columns=[col])
    
    # Apply limit if specified
    if limit and limit > 0:
        display_df = display_df.head(limit)
    
    # Print the dataframe
    if len(display_df) == 0:
        print("No runs found matching the criteria.")
    else:
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        print(f"Found {len(display_df)} runs:")
        print(display_df)
        
        # Print summary statistics for numerical columns
        print("\nSummary Statistics:")
        numeric_cols = display_df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            print(display_df[numeric_cols].describe())
        else:
            print("No numeric columns to summarize.")
        
        # Print unique values for categorical columns
        print("\nUnique Values:")
        categorical_cols = ['model.name', 'data.dataset_size', 'run.init_type', 'sweep_name']
        categorical_cols = [col for col in categorical_cols if col in display_df.columns]
        
        if len(categorical_cols) > 0:
            for col in categorical_cols:
                unique_values = display_df[col].unique()
                print(f"{col}: {unique_values}")
        else:
            print("No categorical columns to summarize.")


def main():
    parser = argparse.ArgumentParser(description="View the runs dataframe")
    parser.add_argument("--base_save_dir", type=str, default="saved_models",
                        help="Base directory where models are saved")
    parser.add_argument("--filter_sweep", type=str, default=None,
                        help="Filter by sweep name")
    parser.add_argument("--filter", type=str, nargs="+", default=None,
                        help="Filter by column=value pairs (e.g., 'model.name=gpt2')")
    parser.add_argument("--sort_by", type=str, default=None,
                        help="Column to sort by")
    parser.add_argument("--descending", action="store_true",
                        help="Sort in descending order")
    parser.add_argument("--show", type=str, nargs="+", default=None,
                        help="Columns to show (space-separated)")
    parser.add_argument("--hide", type=str, nargs="+", default=None,
                        help="Columns to hide (space-separated)")
    parser.add_argument("--limit", type=int, default=None,
                        help="Maximum number of rows to display")
    parser.add_argument("--list_columns", action="store_true",
                        help="List all available columns and exit")
    args = parser.parse_args()
    
    # Initialize run manager to get dataframe
    run_manager = RunManager(args.base_save_dir)
    runs_df = run_manager.get_runs_df()
    
    # List columns if requested
    if args.list_columns:
        print("Available columns:")
        for col in sorted(runs_df.columns):
            print(f"  {col}")
        return
    
    # Parse filter arguments
    filter_dict = {}
    if args.filter:
        for filter_str in args.filter:
            if "=" in filter_str:
                col, val = filter_str.split("=", 1)
                # Convert string values to appropriate types
                if val.lower() == "true":
                    val = True
                elif val.lower() == "false":
                    val = False
                elif val.lower() == "none" or val.lower() == "null":
                    val = None
                elif val.isdigit():
                    val = int(val)
                elif re.match(r"^-?\d+\.\d+$", val):
                    val = float(val)
                filter_dict[col] = val
    
    view_runs(args.base_save_dir, args.filter_sweep, filter_dict, args.sort_by, 
              not args.descending, args.show, args.hide, args.limit)


if __name__ == "__main__":
    main() 