import math
import os
import random
from typing import Optional, Tuple, List, Dict

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class GRU_TCN_GLU_Block(nn.Module):
    """One block of the F(x) extractor: BiGRU -> TCN conv1d -> GLU gate. 

    This section is for feature extraction in nodes of source graph
    Paper: "Each block consists of a bidirectional GRU and a subsequent Temporal
    Convolutional Network (TCN), activated by the Gated Linear Unit (GLU)."
    """
    def __init__(self, in_dim: int, hidden_dim: int, conv_kernel: int = 3, dilation: int = 1):
        super().__init__()
        # BiGRU: input sequence length L, features in_dim -> hidden_dim
        self.bigru = nn.GRU(
            input_size=in_dim, 
            hidden_size=hidden_dim // 2,
            batch_first=True,
            bidirectional=True,
        )
        # TCN conv1d over the time axis
        self.conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim * 2,
            kernel_size=conv_kernel,
            padding=(conv_kernel - 1) // 2 * dilation,
            dilation=dilation,
        )
        self.glu = nn.GLU(dim=1)

    def forward(self, x: Tensor) -> Tensor:
        # x: (batch, L, in_dim)
        out, _ = self.bigru(x)  # (batch, L, hidden_dim)

        # Transpose for Conv1d: (batch, hidden_dim, L)
        out_t = out.transpose(1, 2) 
        conv_out = self.conv(out_t)  # (batch, 2*hidden_dim, L)

        # GLU reduces channels by half: (batch, hidden_dim, L)
        # GLU will split conv output into 2 halves = left * sigmoid(right)
        out = self.glu(conv_out) 
        
        # Transpose back: (batch, L, hidden_dim)
        return out.transpose(1, 2)


class StrokeNodeEncoder(nn.Module):
    def __init__(self, input_dim: int = 2, hidden_dim: int = 256, n_blocks: int = 4):
        """
        Implements Equation (3) to extract node features f_0^n. 
        Paper uses 4 blocks. 
        Note: Paper mentions hidden dim 256 for decoder and 400 for sub-graph attention.
        256 is a safe default for the encoder embedding C.
        """
        super().__init__()
        self. hidden_dim = hidden_dim
        self.blocks = nn.ModuleList()

        # Input projection to match hidden dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # Stacking the blocks with increasing dilation for larger receptive field
        for i in range(n_blocks):
            self.blocks.append(GRU_TCN_GLU_Block(hidden_dim, hidden_dim, dilation=2**i))

    def forward(self, strokes_points: List[Tensor]) -> Tensor:
        """
        Args:
            strokes_points: List of Tensors, where each tensor is (L_n, input_dim).
                            Each tensor represents one stroke's points.
        Returns:
            node_features: (Num_Strokes, hidden_dim)
        """
        # 1.  Prepare Batch Sequence F(x)
        # Concatenate all strokes into one long sequence as per Equation (3) context
        # "L is the input trajectory length"
        
        # lengths for later pooling
        lengths = [len(s) for s in strokes_points] 
        
        # Create (1, Total_L, input_dim) - Batch size 1
        x = torch.cat(strokes_points, dim=0). unsqueeze(0) 
        
        x = self.input_proj(x)

        # Pass through Blocks (BiGRU -> TCN -> GLU) with residual connections
        for block in self.blocks:
            x = x + block(x)  # Residual connection for better gradient flow 
        # Remove batch dim -> (Total_L, hidden_dim)
        sequence_features = x.squeeze(0)

        # 3. Apply Equation (3): Masking
        # f_0^n = F(x)k_n / ||k_n||_1  
        # equivalent to Average Pooling over the specific stroke points. 
        
        node_features = []
        start_idx = 0
        for length in lengths:
            end_idx = start_idx + length
            
            # Extract points for this stroke
            stroke_segment = sequence_features[start_idx:end_idx]
            
            # Average them to get the initial Node Feature f0^n
            node_emb = torch.mean(stroke_segment, dim=0)
            node_features.append(node_emb)
            
            start_idx = end_idx
            
        return torch.stack(node_features)  # (Num_Strokes, hidden_dim)