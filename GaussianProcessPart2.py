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

def target_func(x):
  return torch.sin(20*x) + 2*torch.cos(14*x) - 2*torch.sin(6*x)

X = torch.tensor([-1, -1/2, 0, 1/2, 1])

y = target_func(X)

x_test = torch.linspace(-1, 1, 200)

torch.manual_seed(102)

for i in range(11):
    print(i)
    pyro.clear_param_store()
    kernel = gp.kernels.RBF(input_dim=1)
    kernel.variance = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))

    kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))

    gpmodel = gp.models.GPRegression(X, y, kernel=kernel, noise = torch.tensor(1e-4))

    noise = torch.tensor(1e-4)

    hmc_kernel = pyro.infer.NUTS(gpmodel.model)

    mcmc = pyro.infer.MCMC(hmc_kernel, num_samples=200, warmup_steps = 200)
    mcmc.run()

    hyperparam = mcmc.get_samples()
    x1 = hyperparam["kernel.lengthscale"]
    y1 = hyperparam["kernel.variance"]

    while(True):
        try:
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

    X = torch.cat((X, xstar), 0)
    y = torch.cat((y, fstar), 0)
    
    with torch.no_grad():
        f, ax = plt.subplots(1)
        sns.lineplot(x = x_test, y = samples[0], label = "$f^*$", ax = ax)

        sns.lineplot(x = x_test, y = mean_new, ax=ax, label = "$m(X*)$")
        ax.fill_between(x_test, mean_new+2*sd_new, mean_new-2*sd_new, facecolor='blue', alpha=0.3)

        ax.scatter(xstar, min_value, marker='o', s=75)
        sns.lineplot(x = x_test, y = target_func(x_test), label = "$f$", ax = ax, color = "red")
        ax.legend(loc = "lower right", fontsize = 14)
        ax.set_title("Visualisation for k = "+ str(i), fontsize = 16)
        ax.set_xlabel("$X^*$", fontsize = 12, fontweight="bold")
        ax.set_ylabel("$y$", fontsize = 12, fontweight="bold")
        plt.show()
        
        
ymin, idxmin = torch.min(y, dim = 0, keepdim=False)
argmin = X[idxmin]
argmin
