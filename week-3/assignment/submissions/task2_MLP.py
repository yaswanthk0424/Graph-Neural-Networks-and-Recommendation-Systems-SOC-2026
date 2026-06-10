# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import time # for training time

# Utilities for loading MNIST
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# STEP 1: Load and preprocess MNIST

# Convert images to tensors
# Pixel values become floats in [0,1]
transform = transforms.ToTensor()

# Training dataset
train_dataset = datasets.MNIST(
    root="./data",
    train=True,   # automatically fetches data from files named t10k...... in ./data(path relative to current folder of this file)
    download=True,
    transform=transform
)

# Test dataset
test_dataset = datasets.MNIST(
    root="./data",
    train = False,
    download=True,
    transform=transform
)

# DataLoader divides data in small batches
train_loader = DataLoader(
    train_dataset,
    batch_size=64,# nn.Linear() expects input to be (batch,freq)
    shuffle=True #self explanatory 
)

test_loader = DataLoader(
    test_dataset,
    batch_size=64,# nn.Linear() expects input to be (batch,freq)
    shuffle=False
)


# STEP 2: Define MLP Model

class MLP(nn.Module):   # inherited
    def __init__(self):
        super().__init__()      #using nn.Module's inbuilt constructor
    
        # network is an object of class nn.Sequential
        # following is the constructor
        self.network = nn.Sequential(
            # Convert 28x28 image into 784 values
            nn.Flatten(),
            # First hidden layer
            nn.Linear(784, 128),
            nn.ReLU(),
            # Second hidden layer
            nn.Linear(128, 64),
            nn.ReLU(),
            # Output layer
            # 10 outputs for digits 0-9s
            nn.Linear(64, 10)
        )
        # class nn.Sequential:
            # def __init__(self, *layers):        # constructor of nn.Sequential class
                                                  # layers is attribute of MLP.network class
        #         self.layers = list(layers)      # store all objects (for above one)

        #     def __call__(self, x):              # when we call MLP.network object as function this ex
        #         for layer in self.layers:       # chain them (2)
        #             x = layer(x)
        #         return x
        
        #nn.Module's   def __call__(self,x):
                          #internal stuff....
                          #return self.forward(x)
                          
    def forward(self, x):
        return self.network(x) #(2) runs


# Create model object
model = MLP()

# STEP 3:function calculating loss
criterion = nn.CrossEntropyLoss()

# Adam optimizer updates weights automatically
optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

# STEP 4: Training function
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
        # Backpropagation(chain rule of Calculus)
        loss.backward()#gradients are calculated and stored in .grad optrimiser uses these
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
practice_rounds = 30
start = time.time()
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
end = time.time()
# STEP 7: Final accuracy
final_accuracy = evaluate(
    model,
    test_loader
)
print("\nFinal Test Accuracy:", final_accuracy)
print(f"\nTraining time:{end-start:.3f}sec")
totalparameters = sum(p.numel() for p in model.parameters())
print("\nTotal Parameters:", totalparameters)
