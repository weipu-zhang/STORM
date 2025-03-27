# Add parent directory to Python path
import sys
from pathlib import Path

parent_dir = Path(__file__).resolve().parent.parent
sys.path.append(str(parent_dir))

import gymnasium
import ale_py
import argparse
import cv2
import numpy as np
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
from tqdm import tqdm
from sub_models.world_models import WorldModel
from sub_models.constants import DEVICE

print("Device:", DEVICE.type)
# Dummy variables
obs = torch.randn(16, 64, 3, 64, 64).to(DEVICE)
action = torch.randint(0, 64, (16, 64)).to(
    DEVICE
)  # Ensure action contains valid integer indices
reward = torch.randn(16, 64).to(DEVICE)
termination = torch.randn(16, 64).to(DEVICE)

# Define world_model
print("Defining the world model")
world_model = WorldModel(
    in_channels=3,
    action_dim=64,
    transformer_max_length=64,
    transformer_hidden_dim=512,
    transformer_num_layers=2,
    transformer_num_heads=8,
).to(DEVICE)

# Test world model update
print("Testing world model update")
world_model.update(obs, action, reward, termination, logger=None)
