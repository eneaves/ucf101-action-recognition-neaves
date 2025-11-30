import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineSkeletonMLP(nn.Module):

    def __init__(self, num_keypoints: int, coord_dim: int, num_classes: int, hidden_dim: int = 256):
        super().__init__()
        input_dim = num_keypoints * coord_dim

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.mean(dim=1)

        # Flatten
        x = x.view(x.size(0), -1)  

        # MLP
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        logits = self.fc_out(x)  

        return logits
