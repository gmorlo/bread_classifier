import torch

from tqdm.auto import tqdm
from typing import Dict, List, Tuple

def train_step(model: torch.nn.Module,
                dataloader: torch.utils.data.DataLoader,
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

        # 2. Calculate and accumulate loss
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()

        # 3. Optimizer zero grad
        optimizer.zero_grad()

        # 4. Loss backward
        loss.backward()

        # 5. Optimizer step
        optimizer.step()

        # 6. Calculate and accumulate accuracy
        y_pred_class = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_acc += torch.sum(y_pred_class == y).item() / len(y_pred)
    
    # Adjust train loss and accuracy to be averages
    train_loss = train_loss / len(dataloader)
    train_acc = train_acc / len(dataloader)

    return train_loss, train_acc

def test_step(model: torch.nn.Module,
              dataloader: torch.utils.data.DataLoader,
              loss_fn: torch.nn.Module,
              device: torch.device) -> Tuple[float, float]:
    """Performs a single testing epoch on the model.

    Turns on the model to evaluation mode and runs through all required testing steps.

    Args:
        model: A PyTorch model to test.
        dataloader: A PyTorch DataLoader containing testing data.
        loss_fn: A PyTorch loss function to calculate the loss.
        device: A PyTorch device (e.g. "cuda" or "cpu") to compute on.
    
    Returns:
        A tuple containing testing loss and testing accuracy for the epoch.
    """

    model.eval() # set model to evaluation mode
    test_loss, test_acc = 0, 0 # setup test loss and accuracy variables

    # turn on inference mode to speed up testing
    with torch.inference_mode(): 
        # loop through dataloader batches
        for batch, (X, y) in enumerate(dataloader):
            # send data to target device
            X, y = X.to(device), y.to(device)

            # 1. Forward pass
            test_pred_logits = model(X)

            # 2. Calculate and accumulate loss
            loss = loss_fn(test_pred_logits, y)
            test_loss += loss.item()

            # 3. Calculate and accumulate accuracy
            test_pred_labels = torch.argmax(test_pred_logits, dim=1)
            test_acc += torch.sum(test_pred_labels == y).item() / len(test_pred_labels)

    # Adjust metrics to get averages loss and accuracy per batch
    test_loss = test_loss / len(dataloader)
    test_acc = test_acc / len(dataloader)

    return test_loss, test_acc

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader: torch.utils.data.DataLoader,
          loss_fn: torch.nn.Module,
          optimizer: torch.optim.Optimizer,
          epochs: int,
          device: torch.device) -> Dict[str, List[float]]:
    """Trains and tests a PyTorch model.

    """

    pass # TODO: Implement the main training loop with tqdm progress bar