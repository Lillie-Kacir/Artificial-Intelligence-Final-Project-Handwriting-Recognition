import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
import kagglehub
import os

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
def get_filtered_indices(dataset, ignore_list):
    indices = [i for i, (img_path, label) in enumerate(dataset.imgs) 
               if dataset.classes[label] not in ignore_list]
    return indices

train_indices = get_filtered_indices(full_train_ds, non_chars)
val_indices = get_filtered_indices(full_val_ds, non_chars)

train_loader = DataLoader(Subset(full_train_ds, train_indices), batch_size=32, shuffle=True)
val_loader = DataLoader(Subset(full_val_ds, val_indices), batch_size=32, shuffle=False)

num_classes = len(full_train_ds.classes) - len(non_chars) # Results in 35

class CharacterCNN(nn.Module):
    def __init__(self, num_classes):
        super(CharacterCNN, self).__init__()
        
        # Block 1: 32 filters
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        # Block 2: 64 filters
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        # Block 3: 128 filters
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        
        # Classifier
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 8, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.classifier(x)
        return x

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

train()