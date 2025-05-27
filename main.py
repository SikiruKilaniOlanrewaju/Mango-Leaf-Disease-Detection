print("Hello, World! This is the starting point for your mango leaf anthracnose detection project.")

import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision import models
import matplotlib
import torchvision

# Set dataset path
DATASET_PATH = 'Mango Leaf Disease Identification Dataset (MLDID)'
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Define transforms for training and validation
transform = transforms.Compose([
    transforms.Resize(IMG_SIZE),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Load datasets
train_dataset = datasets.ImageFolder(
    root=DATASET_PATH,
    transform=transform
)

# Split dataset into train, validation, and test
val_size = int(0.2 * len(train_dataset))
test_size = int(0.1 * len(train_dataset))
train_size = len(train_dataset) - val_size - test_size
train_subset, val_subset, test_subset = torch.utils.data.random_split(
    train_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(train_subset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_subset, batch_size=BATCH_SIZE, shuffle=False)
test_loader = DataLoader(test_subset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Classes found: {train_dataset.classes}")
print(f"Number of training samples: {len(train_subset)}")
print(f"Number of validation samples: {len(val_subset)}")

# Use a more advanced model (ResNet18)
model = models.resnet18(weights=None)
model.fc = nn.Linear(model.fc.in_features, len(train_dataset.classes))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop (5 epochs)
best_val_acc = 0.0
train_losses = []
val_accuracies = []
num_epochs = 5
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)
    print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

    # Validation after each epoch
    model.eval()
    correct = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
    val_acc = correct / len(val_subset)
    val_accuracies.append(val_acc)
    print(f"Epoch {epoch+1}, Validation Accuracy: {val_acc:.4f}")
    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), 'best_model.pth')
        print(f"Best model saved at epoch {epoch+1}")

# Test set evaluation
model.load_state_dict(torch.load('best_model.pth'))
model.eval()
test_correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += (preds == labels).sum().item()
        total += labels.size(0)
test_acc = test_correct / total
print(f"Test Accuracy: {test_acc:.4f}")

# Visualize training loss and validation accuracy
plt.figure(figsize=(10,4))
plt.subplot(1,2,1)
plt.plot(range(1, num_epochs+1), train_losses, marker='o')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.subplot(1,2,2)
plt.plot(range(1, num_epochs+1), val_accuracies, marker='o')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
