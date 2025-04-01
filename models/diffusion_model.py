import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from typing import Tuple, List, Dict, Any, Optional

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

        # Downsampling Path
        self.down_blocks = nn.ModuleList([])
        current_channels = model_channels
        levels = len(channel_mult)
        current_resolution = 1 # Assume initial relative resolution factor is 1
        for i in range(levels):
            level_channels = model_channels * channel_mult[i]
            is_last_level = (i == levels - 1)
            for _ in range(num_res_blocks):
                block = ResidualBlock(current_channels, level_channels, time_embed_dim, num_groups)
                self.down_blocks.append(block)
                current_channels = level_channels

                # Add Attention if specified resolution matches
                if current_resolution in attention_resolutions:
                     self.down_blocks.append(AttentionBlock(current_channels, num_heads, num_groups))
            # Downsample except for the last level
            if not is_last_level:
                self.down_blocks.append(Downsample(current_channels))
                current_resolution *= 2 # Resolution decrease (length halved)
        
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


