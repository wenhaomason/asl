# ASL Project

This project implements a machine learning system for American Sign Language (ASL) alphabet recognition using deep learning models.

## Project Overview

The system uses convolutional neural networks to classify images of hand signs representing the ASL alphabet. It supports multiple model architectures including EfficientNet and ResNet, with configurable training parameters.

## Setup and Installation

### Prerequisites

- Python 3.10+
- PyTorch and TorchVision
- CUDA-compatible GPU (recommended for faster training)

### Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/ASL.git
   cd ASL
   ```

2. Set up a virtual environment (recommended):
   
   Using venv (Python's built-in virtual environment):
   ```
   python -m venv asl_env
   
   # Activate the environment
   # On Windows:
   asl_env\Scripts\activate
   # On macOS/Linux:
   source asl_env/bin/activate
   ```
   
   Or using Conda:
   ```
   conda create -n asl_env python=3.10
   conda activate asl_env
   ```

3. Install required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Dataset

The project uses the ASL Alphabet dataset, which contains images of hand signs representing the American Sign Language alphabet. The dataset will be automatically downloaded from Kaggle if not found locally.

## Running the Project

### Step 1: Configure Parameters

Open `main.py` and modify the configuration parameters if needed:

```python
# Data Configuration
DATA_DIR = 'data'   # Directory containing the dataset
BATCH_SIZE = 512    # Batch size for training and validation
IMG_SIZE = 224      # Image size for the model
TRAIN_RATIO = 0.7   # Ratio of training data
VAL_RATIO = 0.1     # Ratio of validation data
TEST_RATIO = 0.2    # Ratio of test data
NUM_WORKERS = 2     # Number of workers for data loading

# Model Configuration
MODEL_NAME = 'efficientnet_b0' # Options: 'efficientnet_b0' or 'resnet50'
UNFREEZE_LAYERS = 'C0'

# Training Configuration
LEARNING_RATE = 0.01                # Initial learning rate
LR_SCHEDULER = 'CosineAnnealingLR'  # Options: 'CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'
NUM_EPOCHS = 1                      # Number of epochs for training
```

### Step 2: Run the Training Script

Execute the main script to start training:

```
python main.py
```

The script will:
1. Download the dataset if not found locally
2. Load and preprocess the data
3. Initialize the model
4. Train the model for the specified number of epochs
5. Evaluate the model on the test set
6. Save results and visualizations

### Step 3: View Results

After training, results will be saved in the `results/<run_name>/` directory, including:
- Model checkpoints
- Training and validation metrics
- Confusion matrices
- TensorBoard logs

## Project Structure

### Files and Directories

- **main.py**: Main script for training and evaluating models
- **requirements.txt**: List of Python dependencies
- **utils/**: Utility modules
  - **data_utils.py**: Data loading and preprocessing utilities
  - **device_utils.py**: Helper functions for device configuration
  - **evaluation_utils.py**: Functions for model evaluation and metrics
  - **model_utils.py**: Model architecture definitions and configuration
  - **training_utils.py**: Training loop and scheduling utilities
  - **visualization_utils.py**: Functions for visualizing results
  - **config_utils.py**: Configuration management and parameter settings
- **results/**: Directory where training results are saved
- **runs/**: TensorBoard logs and model checkpoints directory

## Customization Options

### Model Selection

The project supports different model architectures. To use a different model, modify the `MODEL_NAME` parameter in `main.py`:

```python
# Options: 'efficientnet_b0', 'resnet50'
MODEL_NAME = 'resnet50'  
```

### Fine-tuning Strategy

You can control which layers of the pre-trained model to fine-tune using the `UNFREEZE_LAYERS` parameter:

- `C0`: Only train the classifier layer
- `C1`: Train the classifier layer and the last block
- `C2`: Train the classifier layer and the last two blocks
