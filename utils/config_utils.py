# config_utils.py
"""
Utilities for loading configuration files and setting up experiment directories.
Requires PyYAML: pip install pyyaml
"""
import yaml
import os
import time

def load_config(config_path: str) -> dict:
    # ... (Implementation looks solid - handles file not found, YAML errors, empty file) ...
    # No immediate updates seem necessary here. It correctly loads a YAML file.
    print(f"Loading configuration from: {config_path}")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Configuration file not found at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        if config is None: # Handle empty file case
             print("Warning: Configuration file is empty.")
             return {}
        print("Configuration loaded successfully.")
        return config
    except yaml.YAMLError as e:
        print(f"Error parsing YAML file: {e}")
        raise e
    except Exception as e:
        print(f"Error reading configuration file: {e}")
        raise e


def setup_run_directory(config: dict, base_log_dir: str = "runs") -> tuple[str, str]:
    # ... (Implementation looks mostly okay) ...
    print("\nSetting up run directory...")
    try:
        # --- Customize this section based on your config keys ---
        # Using .get() provides defaults if keys are missing, making it more robust
        model_params = config.get('model_params', {})
        training_params = config.get('training_params', {})
        data_params = config.get('data_params', {})

        model_name = model_params.get('model_name', 'model')
        lr = training_params.get('learning_rate', 'LR')
        epochs = training_params.get('num_epochs', 'Epochs')
        bs = data_params.get('batch_size', 'BS')
        # Consider adding freeze_strategy if it varies significantly between runs
        freeze_strategy = model_params.get('freeze_strategy', 'freeze')
        # --- End Customize ---

        timestamp = time.strftime("%m%d%H%M%S")

        # Create a descriptive run name (kept the same format)
        run_name = f"{model_name}_{freeze_strategy}_lr{lr}_ep{epochs}_bs{bs}_{timestamp}"

        run_dir = os.path.join(base_log_dir, run_name)

        # Create base run directory
        os.makedirs(run_dir, exist_ok=True)

        # --> **UPDATE POINT**: Explicitly create subdirs expected by other utils <--
        # Ensure subdirectories for models, plots, and data exist
        os.makedirs(os.path.join(run_dir, "models"), exist_ok=True)  # For checkpoints
        os.makedirs(os.path.join(run_dir, "plots_data"), exist_ok=True) # For plots and saved data (history, cm)

        print(f"Run Name: {run_name}")
        print(f"Run Directory: {run_dir}")
        # Verify subdirectory creation (optional print)
        # print(f"  Models subdir created: {os.path.join(run_dir, 'models')}")
        # print(f"  Plots/Data subdir created: {os.path.join(run_dir, 'plots_data')}")
        print("Run directory setup complete.")
        return run_dir, run_name

    except KeyError as e: # Should be less likely now with .get()
        print(f"Error: Missing key in configuration required for naming the run: {e}")
        raise e
    except Exception as e:
        print(f"Error creating run directory: {e}")
        raise e

# --- Example Usage ---
# ... (Example usage (__main__) looks fine for demonstrating the functions) ...
# No updates needed in the example usage itself.