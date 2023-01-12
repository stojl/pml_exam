from torch import nn
import torch

import pyro
import pyro.distributions as pdist

class VAE(nn.Module):
    def __init__(self, encoder, decoder, use_cuda=False):
        super().__init__()
        
        assert(encoder.z_dim == decoder.z_dim)
        
        self.encoder = encoder
        self.decoder = decoder
        
        self.use_cuda = use_cuda
        
        self.z_dim = encoder.z_dim
        
        if use_cuda:
            self.cuda()
            
    def model(self, x):
        pyro.module("decoder", self.decoder)
        with pyro.plate("data", x.shape[0]):
            z_mean = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device="cuda:0")
            z_var = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device="cuda:0")
            
            z = pyro.sample("z", pdist.Normal(z_mean, z_var).to_event(1))
            
            img = self.decoder.forward(z)
            
            pyro.sample("obs", pdist.Bernoulli(img, validate_args = False).to_event(1), obs=x.reshape(-1, 784))
            
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