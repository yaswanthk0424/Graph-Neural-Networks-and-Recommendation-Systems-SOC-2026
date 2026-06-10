import torch
import torch.nn as nn
import torch.optim as optim

from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# STEP 1: LOAD MNIST DATASET

# Convert images to tensors
transform = transforms.ToTensor()

# Training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)

# Test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train=False,
    download=True,
    transform=transform
)

# Data loaders
train_loader = DataLoader(
    train_dataset,
    batch_size=64,
    shuffle=True
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False
)

# STEP 2: DEFINE CNN Model

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            # (1,28,28)
            nn.Conv2d(in_channels=1,out_channels=32,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(32,14,14) feature map
            
            nn.Conv2d(in_channels=32,out_channels=64,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),#(64,7,7) feature map
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*7*7,128),
            nn.ReLU(),
            nn.Linear(128,10),
        )

    def forward(self,x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# Create model
model = CNN()
# STEP 3: LOSS FUNCTION
criterion = nn.CrossEntropyLoss()
# STEP 4: OPTIMIZER
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)
# STEP 5: Training function
def train(model, loader, criterion, optimizer):
    # Enable training mode
    model.train()
    running_loss = 0
    for images, labels in loader:
        # Clear old gradients
        optimizer.zero_grad()
        # Forward pass
        outputs = model(images)
        # Compute loss
        loss = criterion(outputs, labels)
        # Backpropagation
        loss.backward()
        # Update weights
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(loader)


# STEP 5: Evaluation function
def evaluate(model, loader):
    # Evaluation mode
    model.eval()
    correct = 0
    total = 0
    # Disable gradient calculations
    with torch.no_grad():
        for images, labels in loader:
            outputs = model(images)
            # Choose class with highest score
            predictions = outputs.argmax(dim=1)
            total += labels.size(0)
            correct += (predictions == labels).sum().item()
    accuracy = 100 * correct / total
    return accuracy

# STEP 6: Train the model
practice_rounds = 5
for practice_round in range(practice_rounds):
    train_loss = train(
        model,
        train_loader,
        criterion,
        optimizer
    )
    accuracy = evaluate(
        model,
        test_loader
    )
    print(
        f"Practice round ({practice_round+1}/{practice_rounds}) "
        f"Loss: {train_loss:.4f} "
        f"Accuracy: {accuracy:.4f}%"
    )

# STEP 7: Final accuracy
final_accuracy = evaluate(
    model,
    test_loader
)
print("\nFinal Test Accuracy:", final_accuracy)