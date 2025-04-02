import torch
from torch import nn
import snntorch as snn
from snntorch import utils
from torch.utils.data import DataLoader
from typing import Tuple, List, Dict, Any

class RecurrentSNN(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, output_size:int, num_layers:int = 1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.input_fc = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=0.9, reset_mechanism="subtract")
        # Add more layers if num_layers > 1, managing states for each
        self.output_fc = nn.Linear(hidden_size, output_size)
        self.lif_out = snn.Leaky(beta=0.9, reset_mechanism = "subtract", output = True)
    
    def forward(self, x:torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Initialize hidden states at the start of a sequence
        mem_rec = [] # Record membrane potential state over time
        spk_rec = [] # Record spikes over time
        # If multi-layers then initialize states of all layers
        mem1 = self.lif1.init_leaky()
        mem_out = self.lif_out.init_leaky()
        # Loop over time steps 
        for step in range(x.size(1)):
            x_step = x[:, step, :]
            cur1 = self.input_fc(x_step)
            spk1, mem1 = self.lif1(cur1, mem1)
            # If multi-layer, pass spk1 to the next layer
            cur_out = self.output_fc(spk1) # Output based on spikes from hidden layer
            spk_out, mem_out = self.lif_out(cur_out, mem_out)
            mem_rec.append(mem_out) # Record output layer membrane potential
            spk_rec.append(spk_out) # Record output layer spikes
        # Stack outputs over time
        mem_rec = torch.stack(mem_rec, dim = 1) # [batch, time, output_size]
        spk_rec = torch.stack(spk_rec, dim = 1) # [batch, time, output_size]

        return mem_rec, spk_rec
    
def build_snn(input_size:int, hidden_size:int, output_size:int) -> nn.Module:
    model = RecurrentSNN(input_size, hidden_size, output_size)
    return model

def train_snn(model:nn.Module, train_loader:DataLoader, epochs:int=10, device:str="cuda" if
    torch.cuda.is_available() else "cpu") -> None:
    """Train the SNN model with sequence handling"""
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    model.to(device)
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for i, (data, targets) in enumerate(train_loader):
            data = data.to(device) # [batch, time, features]
            targets = targets.to(device)
            optimizer.zero_grad()
            # Forward pass through time, now handles timeloop
            mem_out, spk_out = model(data)
            loss = loss_fn(mem_out, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{epochs}, Average loss: {avg_loss:.6f}")

@torch.no_grad()
def predict_snn(model:nn.Module, input_data:torch.Tensor,
                device:str = "cuda" if torch.cuda.is_available() else "cpu") -> torch.Tensor:
    """Make prediction with SNN model"""
    model.eval()
    model.to(device)
    input_data = input_data.to(device) # [batch, time, features]
    mem_out, spk_out = model(input_data)

    return mem_out

@torch.no_grad()
def evaluate_snn(model:nn.Module, test_loader:DataLoader,
                 device:str = "cuda" if torch.cuda.is_available() else "cpu") -> dict:
    """Evaluate model on test data"""
    model.eval()
    model.to(device)
    loss_fn = nn.MSELoss()
    total_loss = 0.0
    all_predictions = []
    all_actuals = []
    for data, targets in test_loader:
        data = data.to(device)
        targets = targets.to(device)

        mem_out, spk_out = model(data)
        loss = loss_fn(mem_out, targets)
        total_loss += loss.item()

        all_predictions.append(mem_out.cpu())
        all_actuals.append(targets.cpu())

    predictions_tensor = torch.cat(all_predictions)
    actuals_tensor = torch.cat(all_actuals)

    mse = torch.mean((predictions_tensor - actuals_tensor) ** 2).item()
    mae = torch.mean(torch.abs(predictions_tensor - actuals_tensor)).item()

    return {"avg_loss":total_loss / len(test_loader),
            "mse":mse,
            "mae":mae}

def save_snn_checkpoint(model: nn.Module, path: str, optimizer: torch.optim.Optimizer = None,
                       epoch: int = 0, metadata: dict = None) -> None:
    """ Save a checkpoint of the SNN model. """
    # ... (implementation is likely correct) ...
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'epoch': epoch
    }
    if optimizer:
        checkpoint['optimizer_state_dict'] = optimizer.state_dict()
    if metadata:
        checkpoint.update(metadata)
    torch.save(checkpoint, path)
    print(f"SNN Checkpoint saved to {path}")