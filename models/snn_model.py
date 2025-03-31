import torch
from torch import nn
import snntorch as snn
from torch.utils.data import DataLoader

def build_snn(input_size:int, hidden_size:int, output_size:int) -> nn.Module:
    """
    Build a Spiking Neural Network (SNN) model for time series data.

    Args:
        input_size (int): Number of input features.
        hidden_size (int): Number of hidden units.
        output_size (int): Number of output units.

    Returns:
        nn.Module: The SNN model.
    """
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        snn.Leaky(beta=0.9, init_hidden=True),
        nn.Linear(hidden_size, output_size),
        snn.Leaky(beta=0.9, init_hidden=True)
    )
    return model

def train_snn(model:nn.Module, train_loader:DataLoader, epochs:int=10) -> None:
    """
    Train the SNN model.

    Args:
        model (nn.Module): The SNN model to train.
        train_loader (DataLoader): DataLoader with time series training data.
        epochs (int): Number of training epochs.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    for epoch in range(epochs):
        for data, target in train_loader:
            optimizer.zero_grad()
            output, _ = model(data)
            loss = loss_fn(output, target)
            loss.backward()
            optimizer.step()
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item()}")

def predict_snn(model:nn.Module, input_data:torch.Tensor) -> torch.Tensor:
    """
    Make predictions with the trained SNN model.

    Args:
        model (nn.Module): The trained SNN model.
        input_data (torch.Tensor): Input time series data for prediction.

    Returns:
        torch.Tensor: The model's predictions.
    """
    with torch.no_grad():
        output, _ = model(input_data)
    return output

def evaluate_snn(model:nn.Module, test_loader:DataLoader) -> dict:
    """
    Evaluate the SNN model on test data.
    
    Args:
        model (nn.Module): The trained SNN model
        test_loader (DataLoader): DataLoader with test data
        
    Returns:
        dict: Dictionary with evaluation metrics
    """
    model.eval()
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    predictions = []
    actuals = []
    with torch.no_grad():
        for data, target in test_loader:
            output, _ = model(data)
            loss = loss_fn(output, target)
            total_loss += loss.item()
            predictions.append(output)
            actuals.append(target)

    predictions = torch.cat(predictions)
    actuals = torch.cat(actuals)
    mse = torch.mean((predictions - actuals) ** 2).item()
    mae = torch.mean(torch.abs(predictions - actuals)).item()
    return  {"loss":total_loss/len(test_loader),
              "mse":mse,
              "mae":mae}

def save_snn_checkpoint(model: nn.Module, path: str, optimizer: torch.optim.Optimizer = None, 
                       epoch: int = 0, metadata: dict = None) -> None:
    """
    Save a checkpoint of the SNN model.
    
    Args:
        model (nn.Module): The SNN model to save
        path (str): Path to save the checkpoint
        optimizer (torch.optim.Optimizer, optional): The optimizer state
        epoch (int): Current epoch number
        metadata (dict, optional): Additional metadata
    """
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }
    
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    
    if metadata:
        checkpoint.update(metadata)
    
    torch.save(checkpoint, path)