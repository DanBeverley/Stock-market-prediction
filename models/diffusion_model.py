import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict, Any, Optional
from torch.amp import GradScaler, autocast

class ResidualBlock(nn.Module):
    """
    Residual block with time embedding
    Uses GroupNorm with SiLU activation
    """
    def __init__(self, in_channels:int, out_channels:int, time_dim:int, num_groups:int = 8):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.GroupNorm(num_groups, in_channels),
            nn.SiLU(),
            nn.Conv1d(in_channels, out_channels, kernel_size = 3, padding = 1)
        )
        self.time_emb = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        self.conv2 = nn.Sequential(
            nn.GroupNorm(num_groups, out_channels),
            nn.SiLU(),
            nn.Conv1d(out_channels, out_channels, kernel_size = 3, padding = 1)
        )
        # Match input and output for residual connection
        if in_channels == out_channels:
            self.residual_connection = nn.Identity()
        else:
            self.residual_connection = nn.Conv1d(in_channels, out_channels, kernel_size = 1)
    
    def forward(self, x:torch.Tensor, t:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, in_channels, seq_len)
            t: Time embedding tensor (batch, time_dim)
        Returns:
            Output tensor (batch, out_channels, seq_len)
        """
        h = self.conv1(x)
        time_embed = self.time_emb(t).unsqueeze(-1) # (batch, out_channels, l)
        h = h + time_embed # Broadcast time embedding
        h = self.conv2(h)
        return h + self.residual_connection(x)

class AttentionBlock(nn.Module):
    """
    Self-attention for 1D sequences
    """
    def __init__(self, channels:int, num_heads:int = 4, num_groups:int = 8):
        super().__init__()
        self.num_heads = num_heads
        assert channels % num_heads == 0, f"Channels ({channels}) must be divisible by num_heads ({num_heads})"
        self.norm = nn.GroupNorm(num_groups, channels)
        self.qkv = nn.Conv1d(channels, channels * 3, kernel_size = 1)
        self.attention = nn.MultiheadAttention(channels, num_heads, batch_first = True)
        self.proj_out = nn.Conv1d(channels, channels, kernel_size = 1)

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor (batch, channels, seq_len)
        Returns:
            Output tensor (batch, channels, seq_len)
        """
        b, c, l = x.shape
        h = self.norm(x)
        qkv = self.qkv(h) # (b, c*3, l)
        # Split into Q, K, V
        q, k, v = qkv.chunk(3, dim = 1) # Each is (b, c, l)
        # Reshape for MultiheadAttention (batch, seq_len, channels)
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)

        # Apply attenion
        attn_output, _ = self.attention(q, k, v) # (b, l, c)
        # Reshape back and project
        attn_output = attn_output.permute(0, 2, 1) # (b, c, l)
        h = self.proj_out(attn_output)

        return x + h # Add residual connection

class Downsample(nn.Module):
    """Downsampling using strided convolution"""
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.Conv1d(channels, channels, kernel_size = 3, stride = 2, padding = 1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class Upsample(nn.Module):
    """Upsampling using ConvTranspose1d"""
    def __init__(self, channels:int):
        super().__init__()
        self.conv = nn.ConvTranspose1d(channels, channels, kernel_size=4, stride=2, padding=1)
    def forward(self, x:torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class SinusoidalPosEmb(nn.Module):
    """Sinusoidal Position Embedding for time steps"""
    def __init__(self, dim:int):
        super().__init__()
        self.dim = dim
    def forward(self, time:torch.Tensor) -> torch.Tensor:
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device)*-embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        # Check for odd dimensions
        if self.dim % 2 == 1:
           embeddings = torch.cat([embeddings, torch.zeros_like(embeddings[:, :1])], dim=-1)
        return embeddings
    
class UNET(nn.Module):
    def __init__(self, in_channels:int, model_channels:int = 64,
                 out_channels:Optional[int] = None, channel_mult:Tuple[int, ...] = (1,2,4,8),
                 sequence_length:Optional[int] = None,
                 num_res_blocks:int = 2, attention_resolutions: Tuple[int, ...] = (8,4),
                 time_dim_mult:int = 4, num_groups:int = 8, num_heads:int = 4):
        super().__init__()
        if out_channels is None:
            out_channels = in_channels
        self.model_channels = model_channels
        time_embed_dim = model_channels * time_dim_mult

        # Time embedding projection
        self.time_mlp = nn.Sequential(SinusoidalPosEmb(model_channels),
                                      nn.Linear(model_channels, time_embed_dim),
                                      nn.SiLU(),
                                      nn.Linear(time_embed_dim, time_embed_dim))
        
        # Initial Convolution
        self.init_conv = nn.Conv1d(in_channels, model_channels, kernel_size=3, padding=1)

        self.abs_attetion_resolutions = set()
        if sequence_length:
            print(f"Calculating absolute attention resolutions based on sequence_length={sequence_length}")
            self.abs_attetion_resolutions = set(attention_resolutions)
            print(f"Applying attention at sequnce lengths: {self.abs_attetion_resolutions}")
        else:
            print("Warning: sequence_length not provided to UNET constructor. Cannot calculate absolute attention resolutions")

        # Downsampling Path
        self.down_blocks = nn.ModuleList([])
        current_channels = model_channels
        levels = len(channel_mult)
        current_resolution = 1 # Assume initial relative resolution factor is 1
        current_seq_len = sequence_length
        for i in range(levels):
            level_channels = model_channels * channel_mult[i]
            is_last_level = (i == levels - 1)
            for _ in range(num_res_blocks):
                block = ResidualBlock(current_channels, level_channels, time_embed_dim, num_groups)
                self.down_blocks.append(block)
                current_channels = level_channels

                # Add Attention if specified resolution matches
                if current_seq_len in self.abs_attetion_resolutions:
                     print(f"Adding Attention at level {i}, channels {current_channels}, seq_len {current_seq_len}")
                     self.down_blocks.append(AttentionBlock(current_channels, num_heads, num_groups))
            # Downsample except for the last level
            if not is_last_level:
                self.down_blocks.append(Downsample(current_channels))
                current_resolution *= 2 # Resolution decrease (length halved)
                if current_seq_len:
                    current_seq_len //= 2
        
        # Middle Path
        self.middle_block = nn.Sequential(
            ResidualBlock(current_channels, current_channels, time_embed_dim, num_groups),
            AttentionBlock(current_channels, num_heads, num_groups),
            ResidualBlock(current_channels, current_channels, time_embed_dim, num_groups)
        )

        # Upsampling Path
        self.up_blocks = nn.ModuleList([])
        for i in reversed(range(levels)):
            level_channels = model_channels * channel_mult[i]
            is_first_level = (i == 0)

            for _ in range(num_res_blocks + 1): # +1 for skip connection handling
                # Input channels = current + skip connection channels
                skip_channels = model_channels * channel_mult[i]
                block = ResidualBlock(current_channels + skip_channels, level_channels, time_embed_dim, num_groups)
                self.up_blocks.append(block)
                current_channels = level_channels
                if current_seq_len in self.abs_attetion_resolutions:
                    print(f"Adding Attention at up-level {i}, channels {current_channels}, seq_len {current_seq_len}")
                    self.up_blocks.append(AttentionBlock(current_channels, num_heads, num_groups))
            # Upsample, except for the first layer, inner_most
            if not is_first_level:
                self.up_blocks.append(Upsample(current_channels))
                current_resolution //= 2 # Resolution increase, length doubled
        self.final_conv = nn.Sequential(
            nn.GroupNorm(num_groups, model_channels),
            nn.SiLU(),
            nn.Conv1d(model_channels, out_channels, kernel_size = 3, padding = 1)
        )
    
    def forward(self, x:torch.Tensor, time:torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input noisy sequence (batch, in_channels, seq_len)
            time: Timesteps (batch,)
        Returns:
            Predicted noise (batch, out_channels, seq_len)
        """
        #1. Time embedding
        t_emb = self.time_mlp(time)
        # 2. Initial Convolution
        h = self.init_conv(x)
        skips = [h] # Store activation for skip connections
        # 3. Downsampling
        for block in self.down_blocks:
            if isinstance(block, (ResidualBlock, AttentionBlock)):
                h = block(h, t_emb)
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else: # Downsample
                h = block(h)
            skips.append(h)
        # 4. Middle
        h = self.middle_block[0](h, t_emb)
        h = self.middle_block[1](h) # Attention doesn't need time emb directly
        h = self.middle_block[2](h, t_emb)
        # 5. Upsampling
        for block in self.up_blocks:
            if isinstance(block, ResidualBlock):
                # Concatenate with skip connection before the block
                skip = skips.pop()
                if h.shape[-1] != skip.shape[-1]:
                   target_len = skip.shape[-1]
                   h = F.interpolate(h, size=target_len, mode = "linear", align_corners=False)
                h = torch.cat([h, skip], dim = 1)
                h = block(h, t_emb) # Pass concatenated tensor and time emb
            elif isinstance(block, AttentionBlock):
                h = block(h)
            else: # Upsample
                h = block(h)
        # Ensure the last skip connection (from init_conv) used correctly
        h = torch.cat([h, skips.pop()], dim = 1)
        assert len(skips) == 0, "Skip connection mismatch"
        # 6.Final layer
        out = self.final_conv(h)
        return out
    
    def define_diffusion_complex(input_channels:int = 1, sequence_length:int = 64, model_channels:int = 64,
                                channel_mult:Tuple[int, ...] = (1,2,3,4),
                                num_res_blocks:int = 2, attention_resolutions:Tuple[int, ...] = (4,),
                                time_dim_mult:int=4, timesteps:int = 1000,
                                beta_schedule_type:str="cosine") -> Tuple[nn.Module, Dict[str, Any]]:
        """
        Define a diffusion model with the UNet.

        Args:
            input_channels: Number of input features per time step.
            sequence_length: The length of the time series sequences the model processes.
                            Needed to calculate relative attention resolutions.
            model_channels: Base channels for the U-Net.
            channel_mult: Channel multipliers for U-Net levels.
            num_res_blocks: Residual blocks per U-Net level.
            attention_resolutions: Relative sequence length factors at which to apply attention.
                                Example: if sequence_length=64, attention_resolutions=(4,) means apply attention
                                when the sequence length is downsampled to 64/4 = 16.
            time_dim_mult: Multiplier for time embedding dimension relative to model_channels.
            timesteps: Number of diffusion steps.
            beta_schedule_type: Type of noise schedule ('linear' or 'cosine').

        Returns:
            Tuple[nn.Module, Dict[str, Any]]: Model and parameters.
        """
        # Calculate absolute sequence lengths for attention
        base_downsample_factor = 2**(len(channel_mult) - 1) # Max downsampling factor
        abs_attention_resolutions = tuple(sequence_length // (2**res_factor) for res_factor in attention_resolutions)
        print(f"Applying attention at sequence lengths: {abs_attention_resolutions}")

        model = UNET(in_channels = input_channels,
                        model_channels = model_channels,
                        channel_mult = channel_mult,
                        num_res_blocks = num_res_blocks,
                        attention_resolutions=attention_resolutions,
                        sequence_length=sequence_length,
                        time_dim_mult=time_dim_mult
                        )
        params = {"beta_schedule":beta_schedule_type,
                  "timesteps":timesteps,
                  "input_channels":input_channels,
                  "time_dim":model_channels * time_dim_mult,
                  "sequence_length":sequence_length}
        if beta_schedule_type == "linear":
            params['beta_start'] = 0.0001
            params["beta_end"] = 0.02
        elif beta_schedule_type == "cosine":
            params["cosine_s"] = 0.008 # Default value from paper
        return model, params

def get_cosine_beta_schedule(timesteps:int, s:float = 0.008) -> torch.Tensor:
    """
    Cosine beta schedule as proposed in Improved DDPM paper.
    https://arxiv.org/abs/2102.09672
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype = torch.float64)
    alphas_cumprod = torch.cos(((x/timesteps) + s)/(1 + s)* math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:]/alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999) # Clamp values

def train_diffusion(model:nn.Module, dataloader:torch.utils.data.DataLoader,
                    params:Dict[str, Any], epochs:int = 100, lr:float=1e-4,
                    device:str = "cuda" if torch.cuda.is_available() else "cpu",
                    use_mixed_precision:bool = True) -> Dict[str, List[float]]:
    """
    Train a diffusion model on time series data.

    Args:
        model (nn.Module): The diffusion model (e.g., ComplexUNet instance).
        dataloader (DataLoader): DataLoader with training data (batch, channels, seq_len).
                                Ensure data is appropriately scaled (e.g., [-1, 1]).
        params (Dict[str, Any]): Diffusion parameters (from define_diffusion_complex).
        epochs (int): Number of training epochs.
        lr (float): Learning rate.
        device (str): Device to train on ('cuda' or 'cpu').
        use_mixed_precision (bool): Whether to use AMP for training.

    Returns:
        Dict[str, List[float]]: Training history (losses).
    """
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    timesteps = params["timesteps"]

    # Define beta scheduler
    if params["beta_schedule"] == "linear":
        betas = torch.linspace(params["beta_start"], params["beta_end"], timesteps, device = device)
    elif params["beta_schedule"] == "cosine":
        betas = get_cosine_beta_schedule(timesteps, s=params.get("cosine_s", 0.008)).to(device)
    else:
        raise ValueError(f"Unsupported beta schedule: {params['beta_schedule']}")
    
    # Pre-calculate diffusion constant
    alphas = 1. - betas
    alphas_cumprod = torch.cumprod(alphas, dim = 0)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod).to(device)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod).to(device)

    # Mixed-precision setup
    scaler = GradScaler(enabled=use_mixed_precision and device == "cuda")

    history = {"loss": []}
    print(f"Starting training on {device} with {'mixed precision' if use_mixed_precision and device == 'cuda' else 'standard precision'}")

    for epoch in range(epochs):
        model.train() # Ensure model is in training mode
        epoch_loss = 0.0
        num_batches = len(dataloader)

        for i, batch_data in enumerate(dataloader):
            x_start = batch_data.to(device)
            batch_size = x_start.shape[0]
            optimizer.zero_grad()

            # Sample t uniformly for each sample in the batch
            t = torch.randint(0, timesteps, (batch_size,), device=device).long()

            # Sample noise and apply forward diffusion process q(x_t | x_0)
            noise = torch.randn_like(x_start, device=device)

            # Get appropriate shapes for broadcasting constants
            sqrt_alpha_cumprod_t = sqrt_alphas_cumprod[t].view(batch_size, 1, 1)
            sqrt_one_minus_alpha_cumprod_t = sqrt_one_minus_alphas_cumprod[t].view(batch_size, 1, 1)

            x_noisy = sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

            # Mixed precision context
            with autocast(enabled=use_mixed_precision and device == "cuda"):
                # Predict the noise added (epsilon)
                predicted_noise = model(x_noisy, t)
                loss = F.mse_loss(predicted_noise, noise)
            
            # Backpropagation
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            if (i + 1) % 50 == 0:
               print(f"  Epoch {epoch+1}, Batch {i+1}/{num_batches}, Batch Loss: {loss.item():.6f}")
        
        avg_loss = epoch_loss/num_batches
        history["loss"].append(avg_loss)
        print(f"Epoch {epoch+1}/{epochs}, Average Loss: {avg_loss:.6f}")
    print("Training finished")
    return history

@torch.no_grad()

def sample_prediction(model:nn.Module, batch_size:int, params:Dict[str, Any],
                      sampling_timesteps:Optional[int] = None, eta:float = 0.0,
                      device:str="cuda" if torch.cuda.is_available() else "cpu",
                      show_progress:bool=True) -> torch.Tensor:
    """
    Generate samples using DDPM or DDIM sampler.

    Args:
        model (nn.Module): Trained diffusion model.
        batch_size (int): Number of samples to generate.
        params (Dict[str, Any]): Parameters used for training the model
                                 (must include timesteps, beta_schedule, sequence_length, input_channels).
        sampling_timesteps (Optional[int]): Number of steps for DDIM sampling. If None, uses all 'timesteps'.
        eta (float): DDIM parameter (0.0 for deterministic DDIM, 1.0 resembles DDPM).
        device (str): Device for sampling.
        show_progress (bool): Print progress updates.

    Returns:
        torch.Tensor: Generated samples (batch_size, channels, seq_len).
    """
    model.eval().to(device)

    train_timesteps = params["timesteps"]
    channels = params["input_channels"]
    seq_len = params["sequence_length"]

    if params["beta_schedule"] == "linear":
        betas = torch.linspace(params["beta_start"], params["beta_end"],
                               train_timesteps, device = device)
    elif params["beta_schedule"] == "cosine":
        betas = get_cosine_beta_schedule(train_timesteps, s = params.get("cosine_s", 0.008)).to(device)
    else:
        raise ValueError(f"Unsupported beta schedule: {params['beta_schedule']}")

    alphas = 1.-betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    # Needed for DDIM / DDPM sampling step
    sqrt_recip_alphas = torch.sqrt(1.0/alphas)
    sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
    sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
    posterior_variance = betas*(1.-F.pad(alphas_cumprod[:-1], (1.0), value=1.0)) / (1. - alphas_cumprod)

    # Setup sampling steps (DDIM)
    use_ddim = sampling_timesteps is not None and sampling_timesteps < train_timesteps
    if use_ddim:
        steps = torch.linspace(train_timesteps - 1, 0, sampling_timesteps, dtype = torch.long, device = device)
    else:
        steps = torch.arange(train_timesteps - 1, -1, -1, dtype=torch.long, device = device)
    print(f"Sampling using {'DDIM' if use_ddim else 'DDPM'} with {len(steps)} steps...")

    # Initial noise
    x = torch.randn((batch_size, channels, seq_len), device = device)
    # Reverse diffusion loop
    for i, t_val in enumerate(steps):
        t = torch.full((batch_size,), t_val, device = device, dtype = torch.long)

        # Get previous timestep index (for DDIM)
        prev_t_val = steps[i + 1] if i < len(steps) - 1 else -1
        alphas_cumprod_t = alphas_cumprod[t_val]
        alphas_cumprod_t_prev = alphas_cumprod[prev_t_val] if prev_t_val >= 0 else torch.tensor(1.0, device = device)

        predicted_noise = model(x, t)
        # pred_x0 = (x - sqrt_one_minus_alphas_cumprod[t_val].view(-1, 1, 1) * predicted_noise)/sqrt_alphas_cumprod[t_val].view(-1, 1, 1)
        # pred_x0 = torch.clamp(pred_x0, -1.0, -1.0)

        # Calculate coefficient for sampling step
        beta_t = betas[t_val]
        sqrt_one_minus_alphas_cumprod_t = sqrt_one_minus_alphas_cumprod[t_val].view(-1, 1, 1)
        sqrt_alphas_cumprod_t = sqrt_alphas_cumprod[t_val].view(-1, 1, 1)

        if use_ddim:
            # DDIM sampling step
            sigma_t = eta * torch.sqrt((1 - alphas_cumprod_t_prev)/(1 - alphas_cumprod_t)*(1-alphas_cumprod_t/alphas_cumprod_t_prev))
            noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)

            # Direction pointing to x_t
            dir_xt = torch.sqrt(1. - alphas_cumprod_t_prev - sigma_t**2)*predicted_noise
            x = torch.sqrt(alphas_cumprod_t_prev) * ((x - sqrt_one_minus_alphas_cumprod_t * predicted_noise) / sqrt_one_minus_alphas_cumprod_t) + dir_xt + sigma_t * noise
        else:
            # DDPM sampling step (eta=1 implicity)
            noise = torch.randn_like(x) if t_val > 0 else torch.zeros_like(x)
            term1 = sqrt_recip_alphas[t_val].view(-1, 1, 1) * (x - (betas[t_val].view(-1, 1, 1)*predicted_noise/sqrt_one_minus_alphas_cumprod_t))
            term2 = torch.sqrt(posterior_variance[t_val]).view(-1, 1, 1) * noise
            x = term1 + term2
        if show_progress and (i % (len(steps)//10) == 0 or i == len(steps) - 1):
            print(f"Sampling step {i + 1}/{len(steps)}(t={t_val})")
            # Potentially un-scale data here if it was scaled before training
            # Example: if scaled with MinMaxScaler to [-1, 1]
            # x = (x + 1) / 2 # Rescale to [0, 1]
            # x = scaler.inverse_transform(x.cpu().numpy().reshape(batch_size, -1)).reshape(batch_size, channels, seq_len) # If using sklearn scaler
            return x

