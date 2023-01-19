from torch import nn
import torch

class LinearDecoder(nn.Module):
        def __init__(self, z_dim, h1_dim, h2_dim):
            super().__init__()
            
            self.z_dim = z_dim
            
            self.fc1 = nn.Linear(z_dim, h2_dim)
            self.fc1a = nn.Linear(h2_dim, h1_dim)
            self.fc2 = nn.Linear(h1_dim, 784)
            self.sdl = nn.Linear(h1_dim, 784)
               
            self.relu = nn.ReLU()
            self.sigmoid = nn.Sigmoid()
            
        def forward(self, z):
            h1 = self.relu(self.fc1(z))
            h1a = self.relu(self.fc1a(h1))
            
            img = self.sigmoid(self.fc2(h1a))
            sd = torch.exp(self.sdl(h1a))
            return img, sd

class LinearEncoder(nn.Module):
        def __init__(self, z_dim, h1_dim, h2_dim):
            super().__init__()
            
            self.z_dim = z_dim
            
            self.fc1 = nn.Linear(784, h1_dim)
            self.fc1a = nn.Linear(h1_dim, h2_dim)
            self.fc21 = nn.Linear(h2_dim, self.z_dim)
            self.fc22 = nn.Linear(h2_dim, self.z_dim)
            
            self.relu = nn.ReLU()
            
        def forward(self, x):
            x = x.reshape(-1, 784)
            
            h1 = self.relu(self.fc1(x))
            h1a = self.relu(self.fc1a(h1))
            
            z_mean = self.fc21(h1a)
            z_var = torch.exp(self.fc22(h1a))
            
            return z_mean, z_var