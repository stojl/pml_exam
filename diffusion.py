import torch
import torch.nn as nn
import torch.distributions as tdist

class DiffusionModel(nn.Module):
    def __init__(self, beta):
        super(DiffusionModel, self).__init__()
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta.to(self.device) # Decay schedule
        self.T = beta.shape[0]
        
        self.c11 = nn.Conv2d(1, 16, 3, padding=1)   # (1, 28, 28) -> (16, 28, 28)
        self.c12 = nn.Conv2d(16, 16, 3, padding=1)  # (16, 28, 28) -> (16, 28, 28)
        self.p1 = nn.MaxPool2d(2) # (16, 28, 28) -> (16, 14, 14)

        # BOTTOM
        self.c21 = nn.Conv2d(16, 32, 3, padding = 1) # (16, 14, 14) -> (32, 14, 14)
        self.c22 = nn.Conv2d(32, 32, 3, padding = 1) # (32, 14, 14) -> (32, 14, 14)
        self.p2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # (32, 14, 14) -> (32, 28, 28)
        
        # OUT
        self.c31 = nn.Conv2d(32, 16, 3, padding=1) # (32, 28, 28) -> (16, 28, 28)
        self.c32 = nn.Conv2d(16, 1, 3, padding=1) # (16, 28, 28) -> (1, 28, 28)
        
        self.relu = nn.ReLU()
        
    def alphabar(self, t):
        alpha = torch.zeros(t.shape[0], device=self.device)
        for idx, s in enumerate(t):
            alpha[idx] = torch.prod(1 - self.beta[0:s])
        return alpha
    
    def embedtime(self, t, embedding_size, dimensions):
        emb = torch.zeros((embedding_size // 2) * t.shape[0], device=self.device).reshape(t.shape[0], embedding_size // 2)
        emb2 = torch.zeros((embedding_size // 2) * t.shape[0], device=self.device).reshape(t.shape[0], embedding_size // 2)
        s = torch.linspace(2, embedding_size // 2, embedding_size // 2, device=self.device)
        for idx, samp in enumerate(t):
            emb[idx] = torch.sin(2 * s * torch.pi * samp / self.T)
            emb2[idx] = torch.cos(2 * s * torch.pi * samp / self.T)

        return torch.cat([emb, emb2], dim = 1).view(-1, embedding_size, 1, 1).repeat(1, 1, dimensions, dimensions)
    
    def sample_image(self):
        samples = torch.zeros((self.T + 1, 784), device=self.device)
        
        samples[self.T] = torch.randn(784, device=self.device)
        
        with torch.no_grad():
            for t in torch.arange(self.T - 1, 0, -1):
                beta = self.beta[t]
                alfa = 1 - beta
                alfabar = self.alphabar(torch.tensor([t + 1], device=self.device))
                z = torch.randn(784, device=self.device)
                xt = samples[t + 1]
                eps = self.forward(xt.view(1, 1, 28, 28), torch.tensor([t + 1], device=self.device)).view(784)
                samples[t] = (xt - beta * eps / (torch.sqrt(1 - alfabar))) / torch.sqrt(alfa) + beta * z
            
            beta = self.beta[0]
            alfa = 1 - beta
            alfabar  = self.alphabar(torch.tensor([1], device=self.device))
            xt = samples[1]
            eps = self.forward(xt.view(1, 1, 28, 28), torch.tensor([1], device=self.device)).view(784)
            samples[0] = (xt - beta * eps / (torch.sqrt(1 - alfabar))) / torch.sqrt(alfa)
        
        return samples
        
    def forward(self, x, t):
        x = self.c11(x)
        x = self.relu(x)
        x = self.c12(x) + self.embedtime(t, 16, 28)
        x = self.relu(self.p1(x))
        
        y = self.c21(x)
        y = self.relu(y)
        y = self.c22(y) + self.embedtime(t, 32, 14)
        y = self.relu(y)
        y = self.p2(y)
        y = self.relu(y)
        
        z = self.c31(y) + self.embedtime(t, 16, 28)
        z = self.c32(z)
        return z