# evaluation_utils.py
"""
Utilities for evaluating PyTorch models after training.
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from torch.utils.data import DataLoader

def evaluate_model(model: nn.Module,
                   test_loader: DataLoader,
                   device: torch.device,
                   criterion: nn.Module | None = None
                   ) -> tuple[float | None, float, np.ndarray, np.ndarray]:
    """
    Evaluates a trained model on a test dataset.

    Args:
        model (nn.Module): The trained model (should already be on the device).
        test_loader (DataLoader): DataLoader for the test set.
        device (torch.device): The device to perform evaluation on.
        criterion (nn.Module | None): The loss function used during training.
                                      If provided, calculates test loss. Defaults to None.

    Returns:
        tuple: A tuple containing:
            - test_loss (float | None): Average test loss (None if criterion is None).
            - test_accuracy (float): Overall test accuracy.
            - all_preds (np.ndarray): Numpy array of predicted labels for all samples.
            - all_labels (np.ndarray): Numpy array of true labels for all samples.
    """
    print(f"\nEvaluating model {model.__class__.__name__} on {device}...")
    model.eval()  # Set model to evaluation mode

    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []

    with torch.no_grad(): # Disable gradient calculations
        num_batches = len(test_loader)
        for i, (inputs, labels) in enumerate(test_loader):
            # Basic progress print
            if (i + 1) % max(1, (num_batches // 5)) == 0 or (i + 1) == num_batches:
                 print(f'\r  Evaluating Batch {i+1}/{num_batches}', end='')

            # Handle potential errors from custom datasets (e.g., failed image load)
            if inputs is None or labels is None or not isinstance(inputs, torch.Tensor):
                print(f"\nWarning: Skipping potentially corrupted batch {i+1}.")
                continue

            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            if criterion is not None:
                loss = criterion(outputs, labels)
                running_loss += loss.item() * inputs.size(0)

            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    print("\nEvaluation complete.") # Newline after progress indicator

    total_samples = len(test_loader.dataset)
    if isinstance(test_loader.dataset, torch.utils.data.ConcatDataset):
        # Handle ConcatDataset specifically if needed, though len() should work
        total_samples = len(test_loader.dataset)
    elif isinstance(test_loader.dataset, torch.utils.data.Subset):
        total_samples = len(test_loader.dataset)

    if total_samples == 0:
         print("Warning: Test dataset size is 0. Cannot calculate metrics.")
         return None, 0.0, np.array([]), np.array([])


    test_accuracy = running_corrects.float().item() / total_samples # Use item() for scalar
    test_loss = running_loss / total_samples if criterion is not None else None

    if test_loss is not None:
        print(f'Test Loss: {test_loss:.4f}')
    print(f'Test Accuracy: {test_accuracy:.4f}')

    return test_loss, test_accuracy, np.array(all_preds), np.array(all_labels)


def calculate_metrics(y_true: np.ndarray,
                        y_pred: np.ndarray,
                        class_names: list[str] | None = None
                       ) -> dict:
    """
    Calculates detailed classification metrics using sklearn.

    Args:
        y_true (np.ndarray): Numpy array of true labels.
        y_pred (np.ndarray): Numpy array of predicted labels.
        class_names (list[str] | None): List of class names corresponding to the
                                        label indices. Defaults to None.

    Returns:
        dict: A dictionary containing:
            - 'accuracy': Overall accuracy score.
            - 'classification_report': Detailed report (precision, recall, F1)
                                       as a dictionary.
            - 'confusion_matrix': Confusion matrix as a numpy array.
    """
    print("\nCalculating detailed metrics...")
    if len(y_true) == 0 or len(y_pred) == 0:
        print("Warning: Cannot calculate metrics with empty label/prediction arrays.")
        return {
            'accuracy': 0.0,
            'classification_report': {},
            'confusion_matrix': np.array([])
        }

    try:
        accuracy = accuracy_score(y_true, y_pred)
        report = classification_report(y_true, y_pred, target_names=class_names, output_dict=True, zero_division=0)
        cm = confusion_matrix(y_true, y_pred)

        metrics = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': cm
        }
        print("Metrics calculated successfully.")
        # Optionally print parts of the report here for quick view
        # print(f"Overall Accuracy: {accuracy:.4f}")
        # print(f"Weighted F1-Score: {report.get('weighted avg', {}).get('f1-score', 'N/A'):.4f}")

    except Exception as e:
        print(f"Error calculating metrics: {e}")
        metrics = {
            'accuracy': 0.0,
            'classification_report': {},
            'confusion_matrix': np.array([])
        }

    return metrics