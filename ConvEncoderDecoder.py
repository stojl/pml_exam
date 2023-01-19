import torch.nn as nn
import torch

class ConvDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim 
        
        # setup the two linear transformations used
        
        self.l1 = nn.Sequential(
            nn.Linear(z_dim, 100),
            nn.ReLU(),
            nn.Linear(100, 400),
            nn.ReLU(),
            nn.Linear(400, 32 * 13 * 13),
            nn.ReLU()
        )
        
        self.l2 = nn.Sequential(
            nn.ConvTranspose2d(32, 32, 4),
            nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4),
            nn.LayerNorm((19, 19)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 4),
            nn.LayerNorm((22, 22)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 16, 4),
            nn.LayerNorm((25, 25)),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4),
            nn.Sigmoid()
        )
        
    def forward(self, z):
        
        z = self.l1(z)
        z = z.view(-1, 32, 13, 13)
        z = self.l2(z)
        return z.reshape(-1, 784)

class ConvEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        
        # setup the three linear transformations used
        
        self.c0 = nn.Conv2d(1, 16, 4)
        
        self.l1 = nn.Sequential(
            nn.Conv2d(16, 16, 4),
            nn.LayerNorm((22, 22)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 4),
            nn.LayerNorm((19, 19)),
            nn.ReLU(),
            nn.Conv2d(16, 32, 4),
            nn.LayerNorm((16, 16)),
            nn.ReLU(),
            nn.Conv2d(32, 32, 4),
            nn.LayerNorm((13, 13)),
            nn.ReLU()
        )
        
        self.l3 = nn.Sequential(
            nn.Linear(32 * 13 * 13, 400),
            nn.ReLU(),
            nn.Linear(400, 100),
            nn.ReLU()
        )
        
        self.o1 = nn.Linear(100, z_dim)
        self.o2 = nn.Linear(100, z_dim)
        
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.relu(self.c0(x))
        
        x = self.l1(x)
        x = x.view(-1, 32 * 13 * 13)
        x = self.l3(x)
        
        z_loc = self.o1(x)
        z_scale = torch.exp(self.o2(x))
        
        return z_loc, z_scale
