# This is the source code for the second part of assignement 2 in the PML exam 2022/2023

import torch

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyro
import pyro.contrib.gp as gp
import pyro.distributions as dist
import arviz as az
import warnings
warnings.warn('ignore')

# Setup

def target_func(x):
  return torch.sin(20*x) + 2*torch.cos(14*x) - 2*torch.sin(6*x)

X = torch.tensor([-1, -1/2, 0, 1/2, 1])

y = target_func(X)

x_test = torch.linspace(-1, 1, 200)

# Minimization

for i in range(6):
    print(i)
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(input_dim = 1)
    kernel.variance = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))

    kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))
    gpmodel = gp.models.GPRegression(X, y, kernel=kernel, noise = torch.tensor(1e-4))
  
    noise = torch.tensor(1e-4)

    hmc_kernel = pyro.infer.NUTS(gpmodel.model)

    mcmc = pyro.infer.MCMC(hmc_kernel, num_samples=500, warmup_steps = 200)
    mcmc.run()

    hyperparam = mcmc.get_samples()
    x1 = hyperparam["kernel.lengthscale"]
    y1 = hyperparam["kernel.variance"]

    while(True):
        try:
            # Sample theta
            idx = torch.randint(0, 100, (1,)) 

            gpmodel.kernel.variance = y1[idx]

            gpmodel.kernel.lengthscale = x1[idx]

            loc, cov = gpmodel.forward(x_test, full_cov= True)

            cov = cov + noise.expand(x_test.shape[0]).diag()


            samples = dist.MultivariateNormal(loc, covariance_matrix=cov, validate_args = False).sample(sample_shape=(1,))
            min_value, min_idx = torch.min(samples[0], dim = 0, keepdim=False)
            xstar = x_test[min_idx.item()].reshape(1)
            if not (xstar in X):
                break
        except:
            print("Try again")

    mean_new = gpmodel.forward(x_test)[0] # Returns the mean of our posterior samples
    sd_new = gpmodel.forward(x_test)[1] # Returns the standard deviation of our posteriour samples
    fstar = target_func(xstar)
    
    
    max_var, max_idx = torch.max(sd_new, dim = 0, keepdim=False)

    xstar2 = x_test[max_idx.item()].reshape(1)

    fstar2 = target_func(xstar2)
    
    X = torch.cat((X, xstar, xstar2), 0)
    y = torch.cat((y, fstar, fstar2), 0)

    with torch.no_grad():
        f, ax = plt.subplots(1)
        sns.lineplot(x = x_test, y = mean_new, ax=ax, label = "$m(X^*)$")
        ax.fill_between(x_test, mean_new+2*sd_new, mean_new-2*sd_new, facecolor='blue', alpha=0.3)

        sns.lineplot(x = x_test, y = samples[0], label = "$f^*$", ax = ax)

        ax.scatter(xstar, min_value, marker='o', s = 50, color = "blue")
        ax.scatter(xstar2, samples[0][max_idx.item()].reshape(1), marker = 'o', s = 50 ,color = 'red')
        sns.lineplot(x = x_test, y = target_func(x_test), label = "$f$", ax = ax, color = "red")
        ax.legend(loc = "lower right", fontsize = 14)
        ax.set_xlabel("$X^*$", fontsize = 12, fontweight="bold")
        ax.set_ylabel("$y$", fontsize = 12, fontweight="bold")
        ax.set_title("Visualization for k = "+ str(i))
        plt.show()
  
# get estimate
ymin, idxmin = torch.min(y, dim = 0, keepdim=False)
argmin = X[idxmin]
argmin

