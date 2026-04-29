import os
from pathlib import Path

import kagglehub
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from character_cnn import CharacterCNN

IMAGE_SIZE = 64
NON_CHARS = ["#", "$", "&", "@"]


class FilteredRemappedDataset(Dataset):
    def __init__(self, dataset: datasets.ImageFolder, keep_classes: list[str], class_to_idx: dict[str, int]):
        self.dataset = dataset
        self.class_to_idx = class_to_idx
        self.indices = [
            index
            for index, (_, label) in enumerate(dataset.imgs)
            if dataset.classes[label] in keep_classes
        ]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        image, original_label = self.dataset[self.indices[idx]]
        class_name = self.dataset.classes[original_label]
        remapped_label = self.class_to_idx[class_name]
        return image, remapped_label


def build_dataloaders(
    batch_size: int,
    max_train_samples: int,
    max_val_samples: int,
) -> tuple[DataLoader, DataLoader, list[str], int, int]:
    dataset_root = kagglehub.dataset_download("vaibhao/handwritten-characters")
    train_dir = os.path.join(dataset_root, "Train")
    val_dir = os.path.join(dataset_root, "Validation")

    data_transforms = transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
        ]
    )

    full_train_ds = datasets.ImageFolder(root=train_dir, transform=data_transforms)
    full_val_ds = datasets.ImageFolder(root=val_dir, transform=data_transforms)

    keep_classes = [class_name for class_name in full_train_ds.classes if class_name not in NON_CHARS]
    class_to_idx = {class_name: idx for idx, class_name in enumerate(keep_classes)}

    train_ds: Dataset = FilteredRemappedDataset(full_train_ds, keep_classes, class_to_idx)
    val_ds: Dataset = FilteredRemappedDataset(full_val_ds, keep_classes, class_to_idx)

    if max_train_samples > 0:
        train_ds = Subset(train_ds, range(min(max_train_samples, len(train_ds))))
    if max_val_samples > 0:
        val_ds = Subset(val_ds, range(min(max_val_samples, len(val_ds))))

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, keep_classes, len(train_ds), len(val_ds)


def build_model(num_classes: int) -> tuple[CharacterCNN, torch.device, nn.Module, optim.Optimizer]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CharacterCNN(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    return model, device, criterion, optimizer


def train(
    model: CharacterCNN,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epochs: int,
) -> None:
    for epoch_idx in range(epochs):
        model.train()
        epoch_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        print(f"Epoch {epoch_idx + 1} Loss: {epoch_loss / len(train_loader):.4f}")


def save_checkpoint(model: CharacterCNN, class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "num_classes": len(class_names),
            "image_size": IMAGE_SIZE,
        },
        str(path),
    )
    print(f"Saved checkpoint: {path}")


if __name__ == "__main__":
    epochs = int(os.getenv("EPOCHS", "15"))
    batch_size = int(os.getenv("BATCH_SIZE", "32"))
    #max_train_samples = int(os.getenv("MAX_TRAIN_SAMPLES", "0"))
    #max_val_samples = int(os.getenv("MAX_VAL_SAMPLES", "0"))
    max_train_samples = 50000
    max_val_samples = 25000
    checkpoint_path = Path(os.getenv("CHECKPOINT_PATH", "models/character_cnn.pt"))

    train_loader, _val_loader, keep_classes, train_size, val_size = build_dataloaders(
        batch_size=batch_size,
        max_train_samples=max_train_samples,
        max_val_samples=max_val_samples,
    )
    model, device, criterion, optimizer = build_model(num_classes=len(keep_classes))

    print(
        f"Training config: epochs={epochs}, batch_size={batch_size}, "
        f"train_samples={train_size}, val_samples={val_size}, device={device}"
    )
    train(
        model=model,
        train_loader=train_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )
    save_checkpoint(model=model, class_names=keep_classes, path=checkpoint_path)