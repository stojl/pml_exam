import torch.nn as nn
import torch

class ConvDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim 
        
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 32 * 14 * 14)
        self.decConv1 = nn.ConvTranspose2d(32, 1, kernel_size=4, padding=1, stride=2)
        
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        z = self.relu(self.fc1(z))
        z = self.relu(self.fc2(z))
        z = z.reshape(-1, 32, 14, 14)
        z = self.sigmoid(self.decConv1(z))
        return z.reshape(-1, 784)

class ConvEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.z_dim = z_dim
        
        # setup the three linear transformations used
        self.encConv1 = nn.Conv2d(1, 32, kernel_size = 3, padding = 1, stride = 2) # OUTPUT SHAPE (32, 14, 14)
        self.f1a = nn.Linear(32 * 14 * 14, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.reshape(-1, 1, 28, 28)
        x = self.relu(self.encConv1(x))
        x = torch.flatten(x, 1)
        x = self.relu(self.f1a(x))
        z_loc = self.fc21(x)
        z_scale = torch.exp(self.fc22(x))
        
        return z_loc, z_scale
