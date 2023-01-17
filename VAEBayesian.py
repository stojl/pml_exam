from torch import nn
import torch
from torch.distributions import constraints
import pyro

import pyro.distributions as pdist

class BayesianVAE(nn.Module):
    def __init__(self, encoder, z_dim, h1_dim, h2_dim, sd, use_cuda=False):
        super().__init__()
        
        self.use_cuda = use_cuda
        
        if use_cuda:
            self.device = torch.device('cuda:0')
        else:
            self.device = torch.device('cpu')
        
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        
        self.z_dim = z_dim
        self.h1_dim = h1_dim
        self.h2_dim = h2_dim
        self.encoder = encoder
        self.sd = torch.tensor(sd, device = self.device)
        
        if use_cuda:
            self.cuda()
            
    def model(self, x):
        w1 = pyro.sample("w1", pdist.Normal(0, self.sd).expand([self.z_dim, self.h2_dim]).to_event(2))
        b1 = pyro.sample("b1", pdist.Normal(0, self.sd).expand([self.h2_dim]).to_event(1))
        
        w2 = pyro.sample("w2", pdist.Normal(0, self.sd).expand([self.h2_dim, self.h1_dim]).to_event(2))
        b2 = pyro.sample("b2", pdist.Normal(0, self.sd).expand([self.h1_dim]).to_event(1))
        
        w3 = pyro.sample("w3", pdist.Normal(0, self.sd).expand([self.h1_dim, 784]).to_event(2))
        b3 = pyro.sample("b3", pdist.Normal(0, self.sd).expand([784]).to_event(1))

        with pyro.plate("data", x.shape[0]):
            z_mean = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=self.device)
            z_var = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=self.device)
            
            z = pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            h1 = self.relu(z @ w1 + b1)
            h2 = self.relu(h1 @ w2 + b2)
            img = self.sigmoid(h2 @ w3 + b3)
            pyro.sample("obs", pdist.Bernoulli(img, validate_args = False).to_event(1), obs=x.reshape(-1, 784))
            
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        
        w1_sd = pyro.param("w1_sd", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        w1_mean = pyro.param("w1_mean", torch.tensor(1.0, device=self.device))
        b1_sd = pyro.param("b1_sd", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        b1_mean = pyro.param("b1_mean", torch.tensor(1.0, device=self.device))
        w1 = pyro.sample("w1", pdist.Normal(w1_mean, w1_sd).expand([self.z_dim, self.h2_dim]).to_event(2))
        b1 = pyro.sample("b1", pdist.Normal(b1_mean, b1_sd).expand([self.h2_dim]).to_event(1))
        
        w2_sd = pyro.param("w2_sd", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        w2_mean = pyro.param("w2_mean", torch.tensor(1.0, device=self.device))
        b2_sd = pyro.param("b2_sd", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        b2_mean = pyro.param("b2_mean", torch.tensor(1.0, device=self.device))
        w2 = pyro.sample("w2", pdist.Normal(w2_mean, w2_sd).expand([self.h2_dim, self.h1_dim]).to_event(2))
        b2 = pyro.sample("b2", pdist.Normal(b2_mean, b2_sd).expand([self.h1_dim]).to_event(1))
        
        w3_sd = pyro.param("w3_sd", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        w3_mean = pyro.param("w3_mean", torch.tensor(1.0, device=self.device))
        b3_sd = pyro.param("b3_sd", torch.tensor(1.0, device=self.device), constraint=constraints.positive)
        b3_mean = pyro.param("b3_mean", torch.tensor(1.0, device=self.device))
        w3 = pyro.sample("w3", pdist.Normal(w3_mean, w3_sd).expand([self.h1_dim, 784]).to_event(2))
        b3 = pyro.sample("b3", pdist.Normal(b3_mean, b3_sd).expand([784]).to_event(1))
        
        with pyro.plate("data", x.shape[0]):
            z_mean, z_var = self.encoder.forward(x)
            # Reparametrization trick in disguise
            pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            
    def decoder(self, z):
        w1_mean = pyro.param("w1_mean").item()
        w1_sd = pyro.param("w1_sd").item()
        b1_mean = pyro.param("b1_mean").item()
        b1_sd = pyro.param("b1_sd").item()
        
        w1 = torch.distributions.Normal(w1_mean, w1_sd).expand([self.h2_dim, self.z_dim]).sample()
        b1 = torch.distributions.Normal(b1_mean, b1_sd).expand([self.h2_dim]).sample()
        w1 = nn.Parameter(w1)
        b1 = nn.Parameter(b1)

        w2_mean = pyro.param("w2_mean").item()
        w2_sd = pyro.param("w2_sd").item()
        b2_mean = pyro.param("b2_mean").item()
        b2_sd = pyro.param("b2_sd").item()
        
        w2 = torch.distributions.Normal(w2_mean, w2_sd).expand([self.h1_dim, self.h2_dim]).sample()
        b2 = torch.distributions.Normal(b2_mean, b2_sd).expand([self.h1_dim]).sample()
        w2 = nn.Parameter(w2)
        b2 = nn.Parameter(b2)

        w3_mean = pyro.param("w3_mean").item()
        w3_sd = pyro.param("w3_sd").item()
        b3_mean = pyro.param("b3_mean").item()
        b3_sd = pyro.param("b3_sd").item()
        
        w3 = torch.distributions.Normal(w3_mean, w3_sd).expand([784, self.h1_dim]).sample()
        b3 = torch.distributions.Normal(b3_mean, b3_sd).expand([784]).sample()
        w3 = nn.Parameter(w3)
        b3 = nn.Parameter(b3)
        
        l1 = nn.Linear(self.z_dim, self.h2_dim)
        l1.weight = w1
        l1.bias = b1
        
        l2 = nn.Linear(self.h2_dim, self.h1_dim)
        l2.weight = w2
        l2.bias = b2
        
        l3 = nn.Linear(self.h1_dim, 784)
        l3.weight = w3
        l3.bias = b3
        
        z = self.relu(l1(z))
        z = self.relu(l2(z))
        z = self.sigmoid(l3(z))
        
        return z
            
    def reconstruct_img(self, x):
        z_mean, z_var = self.encoder(x)
        
        z = pdist.Normal(z_mean, z_var).sample()
        
        img = self.decoder.forward(z)
        
        return img