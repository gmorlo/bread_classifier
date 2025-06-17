import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
                dataloader: torch.utils.DataLoader,
                loss_fn: torch.nn.Module,
                optimizer: torch.optim.Optimizer,
                device: torch.device) -> Tuple[float, float]:
    """Performs a single training epoch on the model.

    Turs on the model to training mode and runs through all required training steps.

    Args:
        model: A PyTorch model to train.
        dataloader: A PyTorch DataLoader containing training data.
        loss_fn: A PyTorch loss function to calculate the loss.
        optimizer: A PyTorch optimizer to update the model's parameters and to help minimize the loss function.
        device: A PyTorch device (e.g. "cuda" or "cpu") to compute on.

    Returns:
        A tuple containing the average training loss and accuracy for the epoch.
    """
    
    model.train() # set model to training mode

    train_loss, train_acc = 0, 0 # setup train loss and accuracy variables

    # loop through dataloader batches
    for batch, (X, y) in enumerate(dataloader): #without tqdm - it will be implemented in the main training loop
        # send data to target device
        X, y = X.to(device), y.to(device)

        # 1. Forward pass
        y_pred = model(X)

        # 2. Calculate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Calculate and accumulate accuracy metrics across all batches
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += torch.sum(y_pred_class == y).item() / len(y_pred)
    
    # Adjust train loss and accuracy to be averages
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc