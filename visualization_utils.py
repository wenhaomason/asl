# visualization_utils.py
"""
Utilities for plotting training history and evaluation results,
and saving the corresponding data.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import json # For saving history dictionary
import torch # Needed for checking tensor type in history

def plot_and_save_history(history: dict, run_name: str, save_dir: str):
    """
    Plots learning curves (loss and accuracy) using matplotlib, saves the plot,
    and saves the history data dictionary to a JSON file.

    Args:
        history (dict): Dictionary containing 'train_loss', 'train_acc',
                          'val_loss', 'val_acc' lists (and potentially 'train_time', 'val_time').
        run_name (str): Unique name for the run (used for filenames/titles).
        save_dir (str): Directory where the plot and JSON history file will be saved.
                          This directory will be created if it doesn't exist.
    """
    print(f"\nProcessing history for run: {run_name}")
    print(f"  Saving plots and data to: {save_dir}")

    # --- Ensure save directory exists ---
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory {save_dir}: {e}")
        return # Cannot proceed without save directory

    # --- Plotting ---
    plot_filename = os.path.join(save_dir, f'{run_name}_learning_curves.png')
    try:
        if not all(k in history for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc']):
             raise ValueError("History dictionary is missing required keys ('train_loss', 'train_acc', 'val_loss', 'val_acc').")
        if not history['train_loss']: # Check if lists are empty
             raise ValueError("History lists are empty.")

        epochs = range(len(history['train_loss']))
        plt.figure(figsize=(14, 6)) # Slightly wider figure

        # Plot Training & Validation Loss
        plt.subplot(1, 2, 1)
        plt.plot(epochs, history['train_loss'], 'bo-', label='Training Loss', markersize=4)
        plt.plot(epochs, history['val_loss'], 'ro-', label='Validation Loss', markersize=4)
        plt.title(f'{run_name}\nTraining and Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.legend()
        plt.grid(True)
        plt.ylim(bottom=0) # Often helpful for loss

        # Plot Training & Validation Accuracy
        plt.subplot(1, 2, 2)
        plt.plot(epochs, history['train_acc'], 'bo-', label='Training Accuracy', markersize=4)
        plt.plot(epochs, history['val_acc'], 'ro-', label='Validation Accuracy', markersize=4)
        plt.title(f'{run_name}\nTraining and Validation Accuracy')
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 1.05) # Often helpful for accuracy

        plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout

        plt.savefig(plot_filename, dpi=150) # Increase dpi for better resolution
        print(f"  Learning curve plot saved successfully: {plot_filename}")
        plt.close() # Close the figure to free memory

    except ValueError as ve:
         print(f"Warning: Cannot plot history - {ve}")
    except Exception as e:
        print(f"Error generating or saving learning curve plot: {e}")
        if plt.fignum_exists(1): # Check if figure was created before error
             plt.close()


    # --- Save History Data to JSON ---
    history_filename = os.path.join(save_dir, f'{run_name}_history.json')
    try:
        # Convert tensors in history to native types before saving
        history_to_save = {}
        for key, values in history.items():
            if values: # Ensure list is not empty
                # Check if the first element is a tensor to decide if conversion is needed
                if isinstance(values[0], torch.Tensor):
                    history_to_save[key] = [v.item() for v in values]
                else:
                    history_to_save[key] = values # Assume native types already
            else:
                history_to_save[key] = [] # Keep empty lists as empty

        with open(history_filename, 'w') as f:
            json.dump(history_to_save, f, indent=4) # Use indent for readability
        print(f"  History data saved successfully: {history_filename}")
    except Exception as e:
        print(f"Error saving history data to JSON: {e}")


def plot_and_save_confusion_matrix(cm: np.ndarray,
                                   class_names: list[str],
                                   run_name: str,
                                   save_dir: str,
                                   figsize: tuple = (12, 10),
                                   normalize: bool = False,
                                   save_csv: bool = False): # Option to save as CSV
    """
    Plots and saves a confusion matrix heatmap using seaborn.
    Also saves the confusion matrix data as a .npy file and optionally as .csv.

    Args:
        cm (np.ndarray): Confusion matrix array.
        class_names (list[str]): List of class names for labels.
        run_name (str): Unique name for the run (used for filenames/titles).
        save_dir (str): Directory where plots and data files will be saved.
                        Will be created if it doesn't exist.
        figsize (tuple): Figure size for the plot.
        normalize (bool): Whether to normalize the CM for plotting percentages.
                          Note: The raw CM data is always saved.
        save_csv (bool): Whether to also save the confusion matrix as a CSV file.
    """
    print(f"\nProcessing confusion matrix for run: {run_name}")
    print(f"  Saving plots and data to: {save_dir}")

    # --- Input Validation ---
    if cm is None or cm.size == 0:
        print("Warning: Confusion matrix is empty or None. Cannot process.")
        return
    if len(class_names) != cm.shape[0]:
         print(f"Warning: Number of class names ({len(class_names)}) does not match confusion matrix dimension ({cm.shape[0]}). Plot labels may be incorrect.")
         # Fallback to generic labels if needed for plotting/saving headers
         class_names_fallback = [str(i) for i in range(cm.shape[0])]
         plot_class_names = class_names_fallback # Use fallback for plotting
    else:
         plot_class_names = class_names # Use provided names for plotting

    # --- Ensure save directory exists ---
    try:
        os.makedirs(save_dir, exist_ok=True)
    except Exception as e:
        print(f"Error creating save directory {save_dir}: {e}")
        return

    # --- Plotting ---
    suffix = "_norm" if normalize else ""
    plot_filename = os.path.join(save_dir, f'{run_name}_confusion_matrix{suffix}.png')
    try:
        if normalize:
            cm_plot_data = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-9) # Add epsilon for safety
            fmt = '.2f'
            plot_title = f'{run_name}\nNormalized Confusion Matrix'
        else:
            cm_plot_data = cm
            fmt = 'd'
            plot_title = f'{run_name}\nConfusion Matrix'

        plt.figure(figsize=figsize)
        sns.heatmap(cm_plot_data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=plot_class_names, yticklabels=plot_class_names,
                    annot_kws={"size": 8}) # Adjust font size if needed
        plt.title(plot_title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()

        plt.savefig(plot_filename, dpi=150)
        print(f"  Confusion matrix plot saved successfully: {plot_filename}")
        plt.close()

    except Exception as e:
        print(f"Error generating or saving confusion matrix plot: {e}")
        if plt.fignum_exists(1):
            plt.close()

    # --- Save Confusion Matrix Data (.npy) ---
    # Always save the raw (non-normalized) confusion matrix data
    cm_data_filename_npy = os.path.join(save_dir, f'{run_name}_confusion_matrix.npy')
    try:
        np.save(cm_data_filename_npy, cm)
        print(f"  Confusion matrix data saved successfully (Numpy format): {cm_data_filename_npy}")
    except Exception as e:
        print(f"Error saving confusion matrix data to .npy: {e}")

    # --- Save Confusion Matrix Data (.csv - Optional) ---
    if save_csv:
        cm_data_filename_csv = os.path.join(save_dir, f'{run_name}_confusion_matrix.csv')
        try:
            # Use savetxt; consider adding header/index if needed for clarity
            # Creating a header string
            header = ','.join(['Predicted_' + name for name in class_names]) # Header uses original names
            # Note: np.savetxt expects strings for header/comments. Add True Label index manually
            # This is a bit more manual if you want row headers
            # Alternative: Use pandas for easier CSV saving with headers/index
            # import pandas as pd
            # df = pd.DataFrame(cm, index=['True_'+name for name in class_names], columns=['Pred_'+name for name in class_names])
            # df.to_csv(cm_data_filename_csv)
            np.savetxt(cm_data_filename_csv, cm, delimiter=",", fmt='%d', header=header, comments='True Labels (Rows) vs Predicted Labels (Cols)\n')
            print(f"  Confusion matrix data saved successfully (CSV format): {cm_data_filename_csv}")
        except Exception as e:
            print(f"Error saving confusion matrix data to .csv: {e}")


# --- Example Usage (within your main script after getting metrics) ---
# if __name__ == "__main__":
#     # --- Dummy Data ---
#     dummy_history = {
#         'train_loss': [1.0, 0.5, 0.2], 'train_acc': [0.5, 0.7, 0.9],
#         'val_loss': [1.2, 0.6, 0.3], 'val_acc': [0.4, 0.65, 0.85]
#     }
#     dummy_cm = np.array([[10, 2, 0], [1, 12, 1], [0, 3, 9]])
#     dummy_class_names = ['Class A', 'Class B', 'Class C']
#     dummy_run_name = "test_run_123"
#     dummy_save_dir = os.path.join("experiment_results", dummy_run_name, "plots_and_data") # Save inside run dir

#     # --- Test Functions ---
#     plot_and_save_history(dummy_history, dummy_run_name, dummy_save_dir)
#     plot_and_save_confusion_matrix(dummy_cm, dummy_class_names, dummy_run_name, dummy_save_dir, save_csv=True)
#     plot_and_save_confusion_matrix(dummy_cm, dummy_class_names, dummy_run_name, dummy_save_dir, normalize=True)

#     print("\nCheck the 'experiment_results/test_run_123/plots_and_data' directory for saved files.")
#     # Optional: Clean up dummy directory
#     # import shutil
#     # if os.path.exists(os.path.dirname(dummy_save_dir)):
#     #      shutil.rmtree(os.path.dirname(dummy_save_dir))