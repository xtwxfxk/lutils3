# -*- coding: utf-8 -*-

import numpy as np
import torch
import torch.nn as nn

class Autoencoder(nn.Module):
    def __init__(self): # 1 1
        super(Autoencoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(1, 256),
            nn.BatchNorm1d(256),
            nn.Sigmoid(),
            nn.Linear(256, 128),
            nn.Sigmoid(),
            nn.Linear(128, 64),
            nn.Sigmoid(),
            nn.Linear(64, 32),
            nn.Sigmoid(),
            nn.Linear(32, 1),
            
            nn.BatchNorm1d(1),
        )
        
        self.decoder = nn.Sequential(
            nn.Linear(1, 32),
            nn.Sigmoid(),
            nn.Linear(32, 64),
            nn.Sigmoid(),
            nn.Linear(64, 128),
            nn.Sigmoid(),
            nn.Linear(128, 256),
            nn.Sigmoid(),
            nn.BatchNorm1d(256),
            nn.Linear(256, 1),
        )
        
    def forward(self, x):
        encoded = self.encoder(x)
        return self.decoder(encoded), encoded


class Normalizer():
    def __init__(self, f, device):
        self.device = device
        self.model = Autoencoder().to(device)
        self.model.load_state_dict(torch.load(f))
        self.model.eval()

    def fit_transform(self, x):
        return self.model.encoder(torch.tensor(x, dtype=torch.float32, device=self.device).reshape([-1, 1])).cpu().detach().numpy()

    def inverse_transform(self, x):
        return self.model.decoder(torch.tensor(x, dtype=torch.float32, device=self.device).reshape([-1, 1])).cpu().detach().numpy()