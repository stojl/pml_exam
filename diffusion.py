import torch
import torch.nn as nn
import torch.distributions as tdist

class DiffusionModel(nn.Module):
    def __init__(self, beta):
        super(DiffusionModel, self).__init__()
        self.device = device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.beta = beta.to(self.device) # Decay schedule
        self.T = beta.shape[0]
        
        self.c1 = nn.Conv2d(1, 16, 3)
        self.c2 = nn.Conv2d(16, 32, 3)
        self.p1 = nn.MaxPool2d(2)
        self.c3 = nn.Conv2d(32, 64, 3)
        self.l1 = nn.Linear(64 * 10 * 10, 1000)
        self.l2 = nn.Linear(1000, 784)
        
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
                samples[t] = (xt - beta * eps / (torch.sqrt(1 - alfabar))) / torch.sqrt(alfa) + torch.sqrt(beta) * z
            
            beta = self.beta[0]
            alfa = 1 - beta
            alfabar  = self.alphabar(torch.tensor([1], device=self.device))
            xt = samples[1]
            eps = self.forward(xt.view(1, 1, 28, 28), torch.tensor([1], device=self.device)).view(784)
            samples[0] = (xt - beta * eps / (torch.sqrt(1 - alfabar))) / torch.sqrt(alfa)
        
        return samples
        
    def forward(self, x, t):
        x = self.c1(x) + self.embedtime(t, 16, 26)
        x = self.relu(x)
        x = self.c2(x) + self.embedtime(t, 32, 24)
        x = self.p1(x)
        x = self.relu(x)
        x = self.c3(x) + self.embedtime(t, 64, 10)
        x = self.relu(x)
        x = x.view(-1, 64 * 10 * 10)
        x = self.l1(x)
        x = self.relu(x)
        x = self.l2(x)
        
        return x.view(-1, 1, 28, 28)