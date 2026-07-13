import torch
import torch.nn as nn
from torchvision import models

# This must match your DFModel architecture exactly
class DFModel(nn.Module):
    def __init__(self, num_classes=2, latent_dim=2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
        super(DFModel, self).__init__()
        model = models.resnext50_32x4d(pretrained=False)  # NO pretrained weights
        self.model = torch.nn.Sequential(*list(model.children())[:-2])
        self.lstm = torch.nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)
        self.linear1 = torch.nn.Linear(2048, num_classes)
        self.avgpool = torch.nn.AdaptiveAvgPool2d(1)
        self.dp = torch.nn.Dropout(0.4)

    def forward(self, x):
        x = x.unsqueeze(1)
        batch_size, seq_length, c, h, w = x.shape
        x = x.view(batch_size * seq_length, c, h, w)
        fmap = self.model(x)
        x = self.avgpool(fmap)
        x = x.view(batch_size, seq_length, 2048)
        x_lstm, _ = self.lstm(x, None)
        return fmap, self.dp(self.linear1(x_lstm[:, -1, :]))

# Create a random model compatible with your app
model = DFModel()

# Save it so your server.py can load it

torch.save(model.state_dict(), "best_model-v3.pt")




