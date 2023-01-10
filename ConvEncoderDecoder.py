import torch.nn as nn
import torch

class ConvDecoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.zdim = z_dim
        # setup the two linear transformations used
        self.fc1 = nn.Linear(z_dim, hidden_dim)
        self.fc21 = nn.MaxUnpool2d(nn.ConvTranspose2d(hidden_dim, 784, kernel_size = 3), kernel_size = 3)
        # setup the non-linearities
        self.softplus = nn.Softplus()
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        hidden = self.softplus(self.fc1(z))
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        loc_img = self.sigmoid(self.fc21(hidden))
        return loc_img

class ConvEncoder(nn.Module):
    def __init__(self, z_dim, hidden_dim):
        super().__init__()
        self.zdim = z_dim
        # setup the three linear transformations used
        self.encConv1 = nn.MaxPool2d(nn.Conv2d(784, z_dim, kernel_size = 3), kernel_size = 3))
        self.fc21 = nn.Linear(hidden_dim, z_dim)
        self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        x = x.reshape(-1, 784)
        # then compute the hidden units
        hidden = self.softplus(self.fc1(x))
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc21(hidden)
        z_scale = torch.exp(self.fc22(hidden))
        return z_loc, z_scale
