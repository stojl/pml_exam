import torch
import torch.nn as nn
import torch.distributions as tdist
import torch.nn.functional as F

class DiffusionModel(nn.Module):
    def __init__(self, beta):
        super(DiffusionModel, self).__init__()
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta.to(self.device) # Decay schedule
        self.T = beta.shape[0]
        
        self.l1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.LayerNorm((28, 28)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LayerNorm((28, 28))
        )
        
        self.l2 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LayerNorm((28, 28)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LayerNorm((28, 28))
        )
        
        self.l3 = nn.Sequential(
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LayerNorm((28, 28)),
            nn.ReLU(),
            nn.Conv2d(16, 16, 3, padding=1),
            nn.LayerNorm((28, 28))
        )
        
        self.cout = nn.Conv2d(16, 1, 3, padding=1)
        
        self.relu = nn.ReLU()
        
    def alphabar(self, t):
        alpha = torch.zeros(t.shape[0], device=self.device)
        for idx, s in enumerate(t):
            alpha[idx] = torch.prod(1 - self.beta[0:s])
        return alpha
    
    def embedtime(self, t, embedding_size, dimensions):
        w = 1.0 / (10000 ** (torch.arange(0, embedding_size, 2, dtype=torch.float32, device=self.device) / embedding_size))
        y = torch.sin(t.view(t.shape[0], 1).repeat(1, embedding_size // 2) * w)
        x = torch.cos(t.view(t.shape[0], 1).repeat(1, embedding_size // 2) * w)
        return torch.cat([x, y], dim=1).view(-1, embedding_size, 1, 1).repeat(1, 1, dimensions, dimensions)
    
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
                samples[t] = (xt - beta * eps / (torch.sqrt(1 - alfabar))) / torch.sqrt(alfa) + torch.sqrt(beta) * z
            
            beta = self.beta[0]
            alfa = 1 - beta
            alfabar  = self.alphabar(torch.tensor([1], device=self.device))
            xt = samples[1]
            eps = self.forward(xt.view(1, 1, 28, 28), torch.tensor([1], device=self.device)).view(784)
            samples[0] = (xt - beta * eps / (torch.sqrt(1 - alfabar))) / torch.sqrt(alfa)
        
        return samples
        
    def forward(self, x, t):
        T = self.embedtime(t, 16, 28)
        
        x = self.l1(x) + T
        x = self.relu(x)
        
        x = self.l2(x) + T
        x = self.relu(x)
        
        x = self.l3(x) + T
        x = self.relu(x)
        
        x = self.cout(x)

        return x