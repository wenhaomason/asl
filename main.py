from data_utils import ASLDataLoader
from device_utils import get_device
from model_utils import get_model
from training_utils import Trainer
from evaluation_utils import evaluate_model, calculate_metrics
from visualization_utils import plot_and_save_history, plot_and_save_confusion_matrix
import torch
import os

from visualization_utils import plot_and_save_history

DATA_DIR = 'asl_alphabet'
BATCH_SIZE = 512
IMG_SIZE = 224
TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2
NUM_WORKERS = 2

MODEL_NAME = 'resnet50'
UNFREEZE_LAYERS = 'C1'

LEARNING_RATE = 0.01
LR_SCHEDULER = 'CosineAnnealingLR' # Options: 'CosineAnnealingLR', 'StepLR', 'ReduceLROnPlateau'
NUM_EPOCHS = 1

if __name__ == '__main__':
    dataset = ASLDataLoader(
        data_dir=DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        num_workers=NUM_WORKERS,
        random_seed=42)

    train_loader = dataset.get_train_loader()
    val_loader = dataset.get_val_loader()
    test_loader = dataset.get_test_loader()
    class_names = dataset.get_class_names()

    device = get_device()

    model = get_model(model_name=MODEL_NAME,
                      unfreeze_layers=UNFREEZE_LAYERS,
                      num_classes=len(class_names),
                      use_pretrained=True)

    model.to(device)
    params_to_update = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params_to_update, lr=LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss()
    if LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS, eta_min=1e-6)
    elif LR_SCHEDULER == 'StepLR':
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    elif LR_SCHEDULER == 'CosineAnnealingLR':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3)
    else:
        scheduler = None

    print("="*20)
    if UNFREEZE_LAYERS == 'C0':
        print("Fine Tuning: Classifier Layer")
    elif UNFREEZE_LAYERS == 'C1':
        print("Fine Tuning: Classifier Layer + Last Block")
    elif UNFREEZE_LAYERS == 'C2':
        print("Fine Tuning: Classifier Layer + Last 2 Blocks")
    print("="*20)

    trainer = Trainer(model=model,
                      unfreeze_layers=UNFREEZE_LAYERS,
                      optimizer=optimizer,
                      criterion=criterion,
                      learning_rate=LEARNING_RATE,
                      scheduler=scheduler,
                      device=device,
                      train_loader=train_loader,
                      val_loader=val_loader,
                      batch_size=BATCH_SIZE,
                      num_epochs=NUM_EPOCHS,
                      tensorboard_log=True)

    history, best_checkpoint = trainer.train()
    test_loss, test_accuracy, all_preds, all_labels = evaluate_model(
        model=model,
        test_loader=test_loader,
        criterion=criterion,
        device=device)
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_accuracy:.4f}")
    cm = calculate_metrics(all_preds, all_labels, class_names)


    save_dir = os.path.join('results', trainer.run_name, 'plots_and_data')
    plot_and_save_history(history, run_name=trainer.run_name, save_dir=save_dir)
    plot_and_save_confusion_matrix(cm, class_names, run_name=trainer.run_name, save_dir=save_dir)
