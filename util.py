import math
from model import GNN_FULL_CLASS
import torch

def chunk_into_n(lst, n):
  size = math.ceil(len(lst) / n)
  return list(
    map(lambda x: lst[x * size:x * size + size],
    list(range(n)))
  )

def init_model(NO_MP, lr, wd):
  # Model
  NO_MP = NO_MP
  model = GNN_FULL_CLASS(NO_MP)

  # Optimizer
  LEARNING_RATE = lr
  optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=wd)

  # Criterion
  #criterion = torch.nn.MSELoss()
  criterion = torch.nn.L1Loss()
  return model, optimizer, criterion

def train(model, criterion, optimizer, loader):
    loss_sum = 0
    for batch in loader:
        # Forward pass and gradient descent
        labels = batch.y
        predictions = model(batch)
        loss = criterion(predictions[torch.isfinite(labels)], labels[torch.isfinite(labels)])

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
    return loss_sum/len(loader)



def evaluate(model, criterion, loader):
    loss_sum = 0
    with torch.no_grad():
        for batch in loader:
            # Forward pass
            labels = batch.y
            predictions = model(batch)
            loss = criterion(predictions[torch.isfinite(labels)], labels[torch.isfinite(labels)])
            loss_sum += loss.item()
    return loss_sum/len(loader)