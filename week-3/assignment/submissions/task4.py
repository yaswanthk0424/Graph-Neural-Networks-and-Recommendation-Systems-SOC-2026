# Import PyTorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import time
import torch
from torch.utils.data import TensorDataset, DataLoader

#datasets
x_train = torch.linspace(-10,10,1000).reshape(-1,1)
y_train = x_train*x_train

x_test = torch.linspace(-10,10,500).reshape(-1,1) # testing on trained data
y_test = x_test*x_test

train_dataset = TensorDataset(x_train,y_train)
test_dataset = TensorDataset(x_test,y_test)
   
train_dataloader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
) 
test_dataloader = DataLoader(
    test_dataset,
    batch_size = 32,
    shuffle = False
)

class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.network = nn.Sequential(
            nn.Linear(1,128),
            nn.ReLU(),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Linear(64,16),
            nn.ReLU(),
            nn.Linear(16,8),
            nn.ReLU(),
            nn.Linear(8,1)
        )
    def forward(self,x):
        return self.network(x)
       
model = MLP()
criterion = nn.MSELoss() # MeanSquareLoss is suitable for this problem(Regresession)
optimizer = optim.Adam(model.parameters(),lr=0.001)

def train(model,loader,criterion,optimiser):
    model.train()
    running_loss = 0
    for x,y in loader:
        optimizer.zero_grad()
        outputs = model(x)
        loss = criterion(outputs,y)
        loss.backward()
        optimiser.step()
        running_loss += loss.item()
        
    return running_loss/len(loader)

def evaluate(model, loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for x, y in loader:
            outputs = model(x)
            loss = criterion(outputs, y)
            total_loss += loss.item()
    return total_loss / len(loader)

practice_rounds = 50 # more practice
start = time.time()
for practice_round in range(practice_rounds):
    train_loss = train(
        model,
        train_dataloader,
        criterion,
        optimizer
    )
    test_MSE = evaluate(
        model,
        test_dataloader,
        criterion
    )
    print(
        f"Practice round ({practice_round+1}/{practice_rounds}) "
        f"Loss: {train_loss:.4f} "
        f"Test Mean Squared Error: {test_MSE:.4f}"
    )
end = time.time()


final_accuracy = evaluate(
    model,
    test_dataloader,
    criterion
)
print("\nFinal Test Error:", final_accuracy)
totalparameters = sum(p.numel() for p in model.parameters())
print(f"\ntotal training time:{end-start:.4f}sec")
print("\nTotal Parameters:", totalparameters)
for name, param in model.named_parameters():
    print(f"{name}: {param.data}")