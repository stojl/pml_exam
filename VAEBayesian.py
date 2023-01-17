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
        self.sd = sd
        
        if use_cuda:
            self.cuda()
            
    def model(self, x):
        print(x.device)
        
        w1 = pyro.sample("w1", pdist.Normal(0, self.sd).expand([self.z_dim, self.h2_dim]).to_event(2))
        print(w1.device)
        b1 = pyro.sample("b1", pdist.Normal(0, self.sd).expand([self.h2_dim]).to_event(1))
        print(b1.device)
        
        w2 = pyro.sample("w2", pdist.Normal(0, self.sd).expand([self.h2_dim, self.h1_dim]).to_event(2))
        print(w2.device)
        b2 = pyro.sample("b2", pdist.Normal(0, self.sd).expand([self.h1_dim]).to_event(1))
        print(b2.device)
        
        w3 = pyro.sample("w3", pdist.Normal(0, self.sd).expand([self.h1_dim, 784]).to_event(2))
        print(w3.device)
        b3 = pyro.sample("b3", pdist.Normal(0, self.sd).expand([784]).to_event(1))
        print(b3.device)

        with pyro.plate("data", x.shape[0]):
            z_mean = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=self.device)
            print(z_mean.device)
            z_var = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=self.device)
            print(z_var.device)
            
            z = pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            print(z.device)
            h1 = self.relu(z @ w1 + b1)
            print(h1.device)
            h2 = self.relu(h1 @ w2 + b2)
            print(h2.device)
            img = self.sigmoid(h2 @ w3 + b3)
            print(img.device)
            pyro.sample("obs", pdist.Bernoulli(img, validate_args = False).to_event(1), obs=x.reshape(-1, 784))
            
    def guide(self, x):
        pyro.module("encoder", self.encoder)
        print(x.device)
        
        w1_map = pyro.param("w1_map", torch.tensor(0.0, device=self.device))
        print(w1_map.device)
        b1_map = pyro.param("b1_map", torch.tensor(1.5, device=self.device))
        print(b1_map.device)
        w1 = pyro.sample("w1", pdist.Delta(w1_map).expand([self.z_dim, self.h2_dim]).to_event(2))
        print(w1.device)
        b1 = pyro.sample("b1", pdist.Delta(b1_map).expand([self.h2_dim]).to_event(1))
        print(b1.device)
        
        w2_map = pyro.param("w2_map", torch.tensor(0.0, device=self.device))
        print(w2_map.device)
        b2_map = pyro.param("b2_map", torch.tensor(0.5, device=self.device))
        print(b2_map.device)
        w2 = pyro.sample("w2", pdist.Delta(w2_map).expand([self.h2_dim, self.h1_dim]).to_event(2))
        print(w2.device)
        b2 = pyro.sample("b2", pdist.Delta(b2_map).expand([self.h1_dim]).to_event(1))
        print(b2.device)
        
        w3_map = pyro.param("w3_map", torch.tensor(0.0, device=self.device))
        print(w3_map.device)
        b3_map = pyro.param("b3_map", torch.tensor(2.0, device=self.device))
        print(b3_map.device)
        w3 = pyro.sample("w3", pdist.Delta(w3_map).expand([self.h1_dim, 784]).to_event(2))
        print(w3.device)
        b3 = pyro.sample("b3", pdist.Delta(b3_map).expand([784]).to_event(1))
        print(b3.device)
        
        with pyro.plate("data", x.shape[0]):
            z_mean, z_var = self.encoder.forward(x)
            print(z_mean.device)
            print(z_var.device)
            # Reparametrization trick in disguise
            pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            
    def reconstruct_img(self, x):
        z_mean, z_var = self.encoder(x)
        
        z = pdist.Normal(z_mean, z_var).sample()
        
        img = self.decoder.forward(z)
        
        return img