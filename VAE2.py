from torch import nn
import torch

import pyro
import pyro.distributions as pdist

class VAE(nn.Module):
    def __init__(self, encoder, decoder, sd, use_cuda=False):
        super().__init__()
        
        assert(encoder.z_dim == decoder.z_dim)
        
        self.encoder = encoder
        self.decoder = decoder
        self.sd = sd
        
        self.use_cuda = use_cuda
        
        self.z_dim = encoder.z_dim
        
        if use_cuda:
            self.cuda()
            
    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_mean = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_var = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            
            z = pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            
            img = self.decoder.forward(z)
            
            pyro.sample("obs", pdist.Normal(loc = img, scale=torch.tensor([self.sd], device=x.device).repeat(784)).to_event(1), obs=x.reshape(-1, 784))
            
    def guide(self, x):
        
        pyro.module("encoder", self.encoder)
        
        with pyro.plate("data", x.shape[0]):
            z_mean, z_var = self.encoder.forward(x)
            pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            
    def reconstruct_img(self, x):
        z_mean, z_var = self.encoder(x)
        
        z = pdist.Normal(z_mean, z_var).sample()
        
        img = self.decoder.forward(z)
        
        return img
    
    def reconstruction_model(self, x):
        pyro.module('encoder', self.encoder)
        pyro.module('decoder', self.decoder)
        
        with pyro.iarange('data', x.shape[0]):
            z_mean, z_sd = self.encoder(x)
            
            base_dist = pdist.Normal(z_mean, z_sd).independent(1)
            Z = pyro.sample('z', base_dist)
            
            img = self.decoder(Z)
            return pyro.sample(
                'reconstruction', pdist.Normal(loc=img, scale=torch.tensor([self.sd], device=x.device).repeat(784)).independent(1)
            )