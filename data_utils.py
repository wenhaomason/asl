import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset, Dataset, ConcatDataset
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import kagglehub
import shutil

class OfficialTestDataset(Dataset):
    """
    Custom Dataset for loading images from the official test directory,
    where images are named like 'A_test.jpg', 'B_test.jpg', etc.
    """
    def __init__(self, test_dir, class_to_idx, transform=None):
        """
        Args:
            test_dir (string): Directory with all official test images.
            class_to_idx (dict): Mapping from class name to class index
                                 (obtained from the training ImageFolder).
            transform (callable, optional): Optional transform to apply.
        """
        self.test_dir = test_dir
        self.transform = transform
        self.class_to_idx = class_to_idx
        self.image_paths = []
        self.image_labels = []

        if not os.path.isdir(self.test_dir):
             print(f"ERROR: Official test directory not found at: {self.test_dir}")
             # Return an empty dataset state
             return

        print(f"Scanning official test directory: {self.test_dir}")
        try:
            filenames = [f for f in os.listdir(test_dir) if f.lower().endswith('.jpg') and '_test.' in f.lower()]
            if not filenames:
                 print(f"WARNING: No files matching '*_test.jpg' found in {test_dir}")

            for filename in filenames:
                class_name = filename.split('_')[0]
                if class_name in self.class_to_idx:
                    self.image_paths.append(os.path.join(test_dir, filename))
                    self.image_labels.append(self.class_to_idx[class_name])
                else:
                    print(f"Warning: Class name '{class_name}' from file '{filename}' not found in training classes. Skipping.")
            print(f"Found {len(self.image_paths)} images in official test directory.")

        except Exception as e:
             print(f"Error scanning official test directory {self.test_dir}: {e}")
             # Reset to empty state on error
             self.image_paths = []
             self.image_labels = []

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.image_labels[idx]
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image, label
        except Exception as e:
            print(f"Error loading/transforming image: {img_path}. Error: {e}")
            # Return None or handle error appropriately for your DataLoader's collate_fn
            # Returning None requires a custom collate_fn to filter them out.
            # For simplicity here, we might just return an empty tensor or raise error.
            # Let's try returning None and assume default collate might error later if not handled
            return None, None # Be aware this might cause issues if not handled downstream

class ASLDataLoader:
    def __init__(self,
                 data_dir,
                 batch_size=512,
                 img_size=224,
                 train_ratio=0.7,
                 val_ratio=0.1,
                 test_ratio=0.2,
                 num_workers=2,
                 random_seed=42):
        """
        Initializes and prepares DataLoaders for ASL dataset.

        Args:
            data_dir (str): Path to the main directory containing
                            'asl_alphabet_train' and 'asl_alphabet_test' subdirectories.
            batch_size (int): Batch size for DataLoaders.
            img_size (int): Target image size (height and width).
            train_ratio (float): Proportion of training data for the train set.
            val_ratio (float): Proportion of training data for the validation set.
            test_ratio (float): Proportion of training data for the custom test set.
            num_workers (int): Number of subprocesses for data loading.
            random_seed (int): Random seed for reproducible splits.
        """
        self.data_dir = data_dir
        if not os.path.isdir(self.data_dir):
            print(f"Data directory not found. Attempting to download dataset from Kaggle...")
            self._download_data()
            if not os.path.isdir(self.data_dir):
                raise FileNotFoundError(f"Failed to download or find data directory: {self.data_dir}")

        self.train_dir = os.path.join(data_dir, 'asl_alphabet_train/asl_alphabet_train')
        self.test_dir = os.path.join(data_dir, 'asl_alphabet_test/asl_alphabet_test')
        self.batch_size = batch_size
        self.img_size = img_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.num_workers = num_workers
        self.random_seed = random_seed

        if not abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-9:
            raise ValueError("Train, validation, and test ratios must sum to 1.0")

        self._define_transforms()
        self._setup_datasets_and_loaders()

    def _download_data(self):
        # Create a destination folder in the current directory
        dest_dir = os.path.join(os.getcwd(), self.data_dir)
        os.makedirs(dest_dir, exist_ok=True)
        # Download the dataset from Kaggle
        path = kagglehub.dataset_download("grassknoted/asl-alphabet")
        # Move the downloaded files to the destination folder
        for item in os.listdir(path):
            s = os.path.join(path, item)
            d = os.path.join(dest_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d, False, None)
            else:
                shutil.copy2(s, d)
        # Remove the original downloaded folder
        shutil.rmtree(path)
        # Print the path to the dataset files
        print("Path to dataset files:", dest_dir)

    def _define_transforms(self):
        """Defines training and validation/test transformations."""
        # Using typical ImageNet normalization values
        self.train_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomRotation(degrees=15),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        self.val_test_transforms = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        print("Data transforms defined.")

    def _setup_datasets_and_loaders(self):
        """Loads data, performs splits, and creates DataLoaders."""
        print("-" * 20)
        print("Setting up Datasets and DataLoaders...")

        # --- 1. Load info from Training Directory & Split ---
        try:
            full_dataset_info = datasets.ImageFolder(self.train_dir)
            self.class_names = full_dataset_info.classes
            self.class_to_idx = full_dataset_info.class_to_idx
            print(f"Found {len(full_dataset_info)} images in {len(self.class_names)} classes in {self.train_dir}")
            print(f"Classes: {self.class_names}")

            all_indices = list(range(len(full_dataset_info)))
            all_targets = full_dataset_info.targets

            train_indices, temp_indices, _, temp_targets = train_test_split(
                all_indices, all_targets, test_size=(self.val_ratio + self.test_ratio),
                random_state=self.random_seed, stratify=all_targets
            )

            # Avoid division by zero if val_ratio + test_ratio is very small or zero
            denominator = self.val_ratio + self.test_ratio
            if denominator < 1e-9:
                 if self.test_ratio > 1e-9:
                     # Only test set exists in the 'temp' split
                     val_indices = [] # No validation indices
                     test_indices = temp_indices
                 else:
                     # No validation or test set requested from split
                     val_indices = temp_indices # All remaining go to validation
                     test_indices = []
            else:
                relative_test_size = self.test_ratio / denominator
                val_indices, test_indices, _, _ = train_test_split(
                    temp_indices, temp_targets, test_size=relative_test_size,
                    random_state=self.random_seed, stratify=temp_targets
                )

            print(f"  Train indices: {len(train_indices)}")
            print(f"  Validation indices: {len(val_indices)}")
            print(f"  Custom Test indices: {len(test_indices)}")

        except FileNotFoundError:
            print(f"ERROR: Training directory not found at {self.train_dir}")
            raise
        except Exception as e:
            print(f"ERROR during training data loading/splitting: {e}")
            raise

        # --- 2. Create Train/Val/CustomTest Datasets (Subsets) ---
        try:
            train_dataset_transformed = datasets.ImageFolder(self.train_dir, transform=self.train_transforms)
            val_test_dataset_transformed = datasets.ImageFolder(self.train_dir, transform=self.val_test_transforms)

            self.train_data = Subset(train_dataset_transformed, train_indices)
            self.val_data = Subset(val_test_dataset_transformed, val_indices)
            self.custom_test_data = Subset(val_test_dataset_transformed, test_indices) if test_indices else None
            print("Created train, validation, and custom test Subsets.")
        except Exception as e:
            print(f"ERROR creating Subset datasets: {e}")
            raise

        # --- 3. Create Official Test Dataset ---
        try:
            self.official_test_data = OfficialTestDataset(self.test_dir, self.class_to_idx, transform=self.val_test_transforms)
            if len(self.official_test_data) == 0:
                 print("Warning: OfficialTestDataset loaded 0 images. Check path and contents.")
        except Exception as e:
            print(f"Warning: Failed to create OfficialTestDataset: {e}. Proceeding without it.")
            self.official_test_data = None

        # --- 4. Combine Test Datasets ---
        datasets_to_combine = []
        print("Combining test sets...")
        if self.custom_test_data is not None and len(self.custom_test_data) > 0:
            print(f"  Adding custom test set ({len(self.custom_test_data)} images)")
            datasets_to_combine.append(self.custom_test_data)
        else:
            print("  Custom test set is empty or None.")

        if self.official_test_data is not None and len(self.official_test_data) > 0:
            print(f"  Adding official test set ({len(self.official_test_data)} images)")
            datasets_to_combine.append(self.official_test_data)
        else:
            print("  Official test set is empty or None.")

        if not datasets_to_combine:
            print("Warning: No test data available to combine.")
            self.combined_test_data = None
        else:
            self.combined_test_data = ConcatDataset(datasets_to_combine)
            print(f"Combined test dataset created with {len(self.combined_test_data)} images.")

        # --- 5. Create DataLoaders ---
        # Determine device type for pin_memory check
        self.device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        use_pin_memory = self.device.type == 'cuda'

        self.train_loader = DataLoader(
            dataset=self.train_data, batch_size=self.batch_size, shuffle=True,
            num_workers=self.num_workers, pin_memory=use_pin_memory
        )
        self.val_loader = DataLoader(
            dataset=self.val_data, batch_size=self.batch_size, shuffle=False,
            num_workers=self.num_workers, pin_memory=use_pin_memory
        )
        self.combined_test_loader = None
        if self.combined_test_data:
            self.combined_test_loader = DataLoader(
                dataset=self.combined_test_data, batch_size=self.batch_size, shuffle=False,
                num_workers=self.num_workers, pin_memory=use_pin_memory
            )

        print("DataLoaders created.")

        # --- 6. Store Dataset Sizes ---
        self.dataset_sizes = {
            'train': len(self.train_data),
            'val': len(self.val_data),
            'custom_test': len(self.custom_test_data) if self.custom_test_data else 0,
            'official_test': len(self.official_test_data) if self.official_test_data else 0,
            'combined_test': len(self.combined_test_data) if self.combined_test_data else 0
        }
        print(f"Final Dataset sizes: {self.dataset_sizes}")
        print("-" * 20)


    # --- Public methods to access loaders and info ---
    def get_train_loader(self):
        return self.train_loader

    def get_val_loader(self):
        return self.val_loader

    def get_test_loader(self):
        """Returns the DataLoader for the combined (custom + official) test set."""
        if self.combined_test_loader is None:
             print("Warning: Combined test loader was not created (likely no test data found).")
        return self.combined_test_loader

    def get_class_names(self):
        return self.class_names

    def get_class_to_idx(self):
        return self.class_to_idx

    def get_dataset_sizes(self):
        return self.dataset_sizes

if __name__ == "__main__":
    # Test
    data_dir = "asl_alphabet"  # Adjust this path as needed
    asl_loader = ASLDataLoader(data_dir=data_dir, batch_size=32, img_size=224)

    print("Train Loader Size:", len(asl_loader.get_train_loader()))
    print("Validation Loader Size:", len(asl_loader.get_val_loader()))
    print("Combined Test Loader Size:", len(asl_loader.get_test_loader()) if asl_loader.get_test_loader() else "No test data")