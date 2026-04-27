import os
from pathlib import Path

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from character_cnn import CharacterCNN

# Download dataset
path = kagglehub.dataset_download("vaibhao/handwritten-characters")
train_dir = os.path.join(path, 'Train')
val_dir = os.path.join(path, 'Validation')

# Define Transforms matching the notebook's preprocessing
data_transforms = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((64, 64)),
    transforms.ToTensor(), # Automatically scales [0, 255] to [0, 1]
])

# Load full datasets
full_train_ds = datasets.ImageFolder(root=train_dir, transform=data_transforms)
full_val_ds = datasets.ImageFolder(root=val_dir, transform=data_transforms)

# Filter classes to match the 35 characters used in the reference notebook
non_chars = ["#", "$", "&", "@"]
class FilteredRemappedDataset(Dataset):
    def __init__(self, dataset, keep_classes, class_to_idx):
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.indices = [
            i for i, (_, label) in enumerate(dataset.imgs)
            if dataset.classes[label] in keep_classes
        ]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        image, original_label = self.dataset[self.indices[idx]]
        class_name = self.dataset.classes[original_label]
        remapped_label = self.class_to_idx[class_name]
        return image, remapped_label
keep_classes = [c for c in full_train_ds.classes if c not in non_chars]
class_to_idx = {class_name: idx for idx, class_name in enumerate(keep_classes)}

train_ds = FilteredRemappedDataset(full_train_ds, keep_classes, class_to_idx)
val_ds = FilteredRemappedDataset(full_val_ds, keep_classes, class_to_idx)

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "0"))

if MAX_TRAIN_SAMPLES > 0:
    train_ds = Subset(train_ds, range(min(MAX_TRAIN_SAMPLES, len(train_ds))))
if MAX_VAL_SAMPLES > 0:
    val_ds = Subset(val_ds, range(min(MAX_VAL_SAMPLES, len(val_ds))))

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

num_classes = len(keep_classes) # Results in 35

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = CharacterCNN(num_classes).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

def train(epochs=15):
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        print(f"Epoch {epoch+1} Loss: {total_loss/len(train_loader):.4f}")


def save_checkpoint(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": keep_classes,
            "num_classes": num_classes,
            "image_size": 64,
        },
        str(path),
    )
    print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    epochs = int(os.getenv("EPOCHS", "15"))
    checkpoint_path = Path(os.getenv("CHECKPOINT_PATH", "models/character_cnn.pt"))
    print(
        f"Training config: epochs={epochs}, batch_size={BATCH_SIZE}, "
        f"train_samples={len(train_ds)}, val_samples={len(val_ds)}, device={device}"
    )
    train(epochs=epochs)
    save_checkpoint(checkpoint_path)