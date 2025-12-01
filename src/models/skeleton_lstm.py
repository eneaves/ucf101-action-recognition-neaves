import torch
import torch.nn as nn


class SkeletonLSTMModel(nn.Module):

    def __init__(
        self,
        num_keypoints: int,
        coord_dim: int,
        num_classes: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.3,
    ):
        super().__init__()

        input_size = num_keypoints * coord_dim

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=True,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, V, C = x.shape

        # Flatten 
        x = x.view(B, T, V * C)

        # LSTM
        out, (h_n, c_n) = self.lstm(x)
        
        # h_n[-2] es forward de la última capa
        # h_n[-1] es backward de la última capa
        last_hidden = torch.cat((h_n[-2], h_n[-1]), dim=1)

        last_hidden = self.dropout(last_hidden)
        logits = self.fc(last_hidden) 
        return logits
