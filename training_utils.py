import torch
from torch.utils.tensorboard import SummaryWriter
import os
import time
import shutil


# noinspection PyTypeChecker
class Trainer:
    def __init__(self,
                 model,
                 unfreeze_layers,
                 optimizer,
                 criterion,
                 learning_rate=0.01,
                 scheduler=None,
                 device=torch.device("cpu"),
                 train_loader=None,
                 val_loader=None,
                 batch_size=512,
                 num_epochs=10,
                 start_epoch=0,
                 best_val_acc=0.0,
                 tensorboard_log=False):
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.start_epoch = start_epoch
        self.best_val_acc = best_val_acc
        self.time_stamp = time.strftime("%m%d%H%M%S")
        self.run_name = f"{self.model.__class__.__name__}_{unfreeze_layers}_LR{self.learning_rate}_{scheduler.__class__.__name__}_{self.num_epochs}epochs_BatchSize{self.batch_size}_{self.time_stamp}"
        self.tensorboard_log = tensorboard_log
        if self.tensorboard_log:
            self.log_dir = os.path.join("runs", self.run_name)
            self.model_dir = os.path.join(self.log_dir, "models")
            os.makedirs(self.model_dir, exist_ok=True)
            self.latest_checkpoint_path = os.path.join(self.model_dir, f'{self.run_name}_latest.pth')
            self.best_checkpoint_path = os.path.join(self.model_dir, f'{self.run_name}_best.pth')
            self.writer = SummaryWriter(log_dir=self.log_dir)
            print(f"TensorBoard log directory: {self.log_dir}")

    def train(self):
        history = {'train_loss': [], 'train_acc': [], 'train_time': [], 'val_loss': [], 'val_acc': [], 'val_time': []}
        print(f"Training {self.model.__class__.__name__} on {self.device}")
        self.model.to(self.device)

        since = time.time()
        for epoch in range(self.start_epoch, self.num_epochs):

            train_loss, train_acc, train_time = self._train_epoch(epoch)
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['train_time'].append(train_time)

            val_loss, val_acc, val_time, is_best = self._validate_epoch()
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['val_time'].append(val_time)

            # Update the learning rate
            current_lr = self.optimizer.param_groups[0]['lr']
            if self.scheduler:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()  # For CosineAnnealingLR, StepLR etc.

            # Log metrics to TensorBoard
            if self.tensorboard_log:
                self._log_metrics(epoch, train_loss, val_loss, train_acc, val_acc, train_time, val_time, current_lr)

            self._save_checkpoint(epoch, is_best)
        time_elapsed = time.time() - since
        print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
        print(f'Best val Acc: {self.best_val_acc:4f}')

        self.writer.close()
        return history, self.best_checkpoint_path


    def _train_epoch(self, epoch):
        print(f'Training epoch {epoch + 1}/{self.num_epochs}')
        self.model.train()
        running_loss = 0.0
        running_corrects = 0

        num_batches = len(self.train_loader)
        epoch_start_time = time.time()
        for i, (inputs, labels) in enumerate(self.train_loader):
            batch_start_time = time.time()
            inputs = inputs.to(self.device)
            labels = labels.to(self.device)

            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            _, predictions = torch.max(outputs, 1)
            # Backward pass and optimization
            loss.backward()
            self.optimizer.step()
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(predictions == labels.data)
            batch_end_time = time.time()
            batch_time = batch_end_time - batch_start_time
            bar_length = int((i + 1) / num_batches * 40)
            print(f'\rBatch {i + 1}/{num_batches} Loss: {loss.item():.4f}' + ' [' + '=' * bar_length + '>' + '-' * (40 - bar_length - 1) + f'] ({batch_time:.2f}s/batch)     ', end='')
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / len(self.train_loader.dataset)
        epoch_acc = running_corrects.float() / len(self.train_loader.dataset)
        print(f'\nEpoch {epoch + 1}/{self.num_epochs} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s')

        return epoch_loss, epoch_acc.item(), epoch_time

    def _validate_epoch(self):
        print('Validating')
        is_best = False
        self.model.eval()
        running_loss = 0.0
        running_corrects = 0

        num_batches = len(self.val_loader)
        epoch_start_time = time.time()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(self.val_loader):
                batch_start_time = time.time()
                inputs = inputs.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                _, predictions = torch.max(outputs, 1)
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(predictions == labels.data)
                batch_end_time = time.time()
                batch_time = batch_end_time - batch_start_time
                bar_length = int((i + 1) / num_batches * 40)
                print(f'\rBatch {i + 1}/{num_batches} Loss: {loss.item():.4f}' + ' [' + '=' * bar_length + '>' + '-' * (40 - bar_length - 1) + f'] ({batch_time:.2f}s/batch)     ', end='')
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time
        epoch_loss = running_loss / len(self.val_loader.dataset)
        epoch_acc = running_corrects.float() / len(self.val_loader.dataset)
        print(f'\nValidation Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} Time: {epoch_time//60:.0f}m {epoch_time%60:.0f}s')
        if epoch_acc > self.best_val_acc:
            print(f"  New best validation accuracy! ({epoch_acc:.4f} > {self.best_val_acc:.4f})")
            self.best_val_acc = epoch_acc.item() # Update best accuracy
            is_best = True

        return epoch_loss, epoch_acc.item(), epoch_time, is_best

    def _log_metrics(self, epoch, train_loss, val_loss, train_acc, val_acc, train_time, val_time, current_lr):
        # Log metrics to TensorBoard
        self.writer.add_scalar(f'{self.run_name}/Loss/train', train_loss, epoch)
        self.writer.add_scalar(f'{self.run_name}/Loss/val', val_loss, epoch)
        self.writer.add_scalar(f'{self.run_name}/Accuracy/train', train_acc, epoch)
        self.writer.add_scalar(f'{self.run_name}/Accuracy/val', val_acc, epoch)
        self.writer.add_scalar(f'{self.run_name}/Time/train', train_time, epoch)
        self.writer.add_scalar(f'{self.run_name}/Time/val', val_time, epoch)
        self.writer.add_scalar(f'{self.run_name}/LearningRate', current_lr, epoch)
        self.writer.flush()  # Ensure data is written to disk

    def _save_checkpoint(self, epoch, is_best):
        self.model.to('cpu') # Move model to CPU

        state = {
            'epoch': epoch, # Epoch just completed
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_acc': self.best_val_acc, # Store the current best accuracy
        }
        if self.scheduler is not None:
            state['scheduler_state_dict'] = self.scheduler.state_dict()

        # Save the latest checkpoint
        try:
            torch.save(state, self.latest_checkpoint_path)
            print(f"Epoch {epoch+1}: Latest checkpoint saved to '{self.latest_checkpoint_path}'")
        except Exception as e:
             print(f"ERROR saving latest checkpoint: {e}")

        # If this is the best model so far, copy the latest checkpoint file to best_checkpoint file
        if is_best:
            try:
                shutil.copyfile(self.latest_checkpoint_path, self.best_checkpoint_path)
                print(f"Epoch {epoch+1}: Best checkpoint updated and saved to '{self.best_checkpoint_path}'")
            except Exception as e:
                 print(f"ERROR copying best checkpoint: {e}")

        # Move model back to original device if moved to CPU earlier
        self.model.to(self.device)
