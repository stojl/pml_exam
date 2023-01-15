import torch
import gpytorch
import pyro
from pyro.infer.mcmc import NUTS, MCMC, HMC

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import pyro.contrib.gp as gp
import pyro.distributions as dist
pyro.set_rng_seed(0)

def target_func(x):
  return torch.sin(20*x) + 2*torch.cos(14*x) - 2*torch.sin(6*x)

xData = torch.tensor([-1, -1/2, 0, 1/2, 1])

yData = target_func(xData)

kernel = gp.kernels.RBF(input_dim=1)

kernel.variance = pyro.nn.PyroSample(dist.Logno(torch.tensor(0.5), torch.tensor(1.5)))

kernel.lengthscale = pyro.nn.PyroSample(dist.Uniform(torch.tensor(1.0), torch.tensor(3.0)))

gpr = gp.models.GPRegression(xData, yData, kernel)

hmc_kernel = HMC(gpr.model)

mcmc = MCMC(hmc_kernel, num_samples=10)

mcmc.run()

ls_name = "kernel.lengthscale"

posterior_ls = mcmc.get_samples()[ls_name]

optimizer = torch.optim.Adam(gpr.parameters(), lr=0.01)

loss_fn = pyro.infer.TraceMeanField_ELBO().differentiable_loss

for i in range(1000):

    optimizer.zero_grad()

    loss = loss_fn(gpr.model, gpr.guide)  

    loss.backward()  

    optimizer.step()

gpr.mod


# Gammel kode


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, xData, yData, likelihood):
        super(ExactGPModel, self).__init__(xData, yData, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


from gpytorch.priors import LogNormalPrior, NormalPrior, UniformPrior
# Use a positive constraint instead of usual GreaterThan(1e-4) so that LogNormal has support over full range.
likelihood = gpytorch.likelihoods.GaussianLikelihood(noise_constraint=gpytorch.constraints.Positive())
model = ExactGPModel(xData, yData, likelihood)

model.mean_module.register_prior("mean_prior", UniformPrior(-1, 1), "constant")
model.covar_module.base_kernel.register_prior("lengthscale_prior", UniformPrior(0.01, 0.5), "lengthscale")
model.covar_module.register_prior("outputscale_prior", UniformPrior(1, 2), "outputscale")
likelihood.register_prior("noise_prior", UniformPrior(0.01, 0.5), "noise")

mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)



import os
smoke_test = ('CI' in os.environ)

num_samples = 2000
warmup_steps = 500

def pyro_model(x, y):
    with gpytorch.settings.fast_computations(False, False, False):
        sampled_model = model.pyro_sample_from_prior()
        output = sampled_model.likelihood(sampled_model(x))
        pyro.sample("obs", output, obs=y)
    return y

nuts_kernel = NUTS(pyro_model)
mcmc_run = MCMC(nuts_kernel, num_samples=num_samples, warmup_steps=warmup_steps)
mcmc_run.run(xData, yData)

model.pyro_load_from_samples(mcmc_run.get_samples())

model.eval()
test_x = torch.linspace(-1, 1, 101).unsqueeze(-1)
test_y = target_func(test_x)
expanded_test_x = test_x.unsqueeze(0).repeat(num_samples, 1, 1)
output = model(expanded_test_x)

np.mean(output.cov.detach().numpy(), axis = 0)


with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Plot training data as black stars
    ax.plot(xData.numpy(), yData.numpy(), 'k*', zorder=10)

        # Plot predictive means as blue line
    ax.plot(test_x.numpy(), np.mean(output.mean.detach().numpy(), axis = 0), 'b', linewidth=0.3)

    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    ax.legend(['Observed Data', 'Sampled Means'])

plt.show()
