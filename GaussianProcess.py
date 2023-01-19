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

# Setting up D

def target_func(x):
  return torch.sin(20*x) + 2*torch.cos(14*x) - 2*torch.sin(6*x)

X = torch.tensor([-1, -1/2, 0, 1/2, 1])

y = target_func(X)

# Initial MCMC sampling

pyro.clear_param_store()
kernel = gp.kernels.RBF(input_dim=1)
kernel.variance = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))
kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))
gpmodel = gp.models.GPRegression(X, y, kernel=kernel, noise = torch.tensor(1e-4))

hmc_kernel = pyro.infer.NUTS(gpmodel.model)

mcmc = pyro.infer.MCMC(hmc_kernel, num_samples=500, warmup_steps = 5)
    
mcmc.run()

# Sample from prior distributions of theta

lengthprior = torch.distributions.LogNormal(-1, 1)
varianceprior = torch.distributions.LogNormal(0, 2)
x1 = lengthprior.sample_n(500)
y1 = varianceprior.sample_n(500)

f, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
sns.scatterplot(x = x1, y = y1)
ax.set_xlabel("Lengthscale $\sigma_l^2$", fontsize = 14)
ax.set_ylabel("Variance $\sigma_s^2$", fontsize = 14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()

# Sample from posterior distribution of theta

hyperparam = mcmc.get_samples()
x1 = hyperparam["kernel.lengthscale"][0:499]
y1 = hyperparam["kernel.variance"][0:499]

f, ax = plt.subplots()
ax.set(xscale="log", yscale="log")
sns.scatterplot(x = x1, y = y1)
ax.set_xlabel("Lengthscale $\sigma_l^2$", fontsize = 14)
ax.set_ylabel("Variance $\sigma_s^2$", fontsize = 14)
plt.yticks(fontsize=12)
plt.xticks(fontsize=12)
plt.show()

# Plot mean and confidence intervals

with torch.no_grad():
    x_new = torch.linspace(-1, 1, 200)
    mean_new = gpmodel.forward(x_new)[0] # Returns the mean of our posterior samples
    sd_new = gpmodel.forward(x_new)[1] # Returns the standard deviation of our posteriour samples
    f, ax = plt.subplots(1)
    sns.lineplot(x = x_new, y = mean_new, ax=ax, label = "$m(x^*)$")
    sns.scatterplot(x = X, y = y, ax=ax, s = 75)
    ax.fill_between(x_new, mean_new+2*sd_new, mean_new-2*sd_new, facecolor='blue', alpha=0.4)
    sns.lineplot(x = x_new, y = target_func(x_new), ax = ax, label =  "$f$", color = "red")
    ax.legend(loc='lower right', fontsize = 12)
    ax.set_xlabel("$x^*$", fontsize = 14)
    ax.set_ylabel("$y$", fontsize = 14)
    plt.yticks([-5, -4, -3, -2, -1, 0, 1, 2, 3],fontsize=12)
    plt.xticks(fontsize=12)
    plt.show()
    
# arviz diagnostics

sum1 = az.summary(diag)
params =  az.extract(diag)

samples_kernel_lengthscale = params["kernel.lengthscale"]
samples_kernel_variance = params["kernel.variance"]

xs = np.arange(1, samples_kernel_variance.shape[0]+1, 1)

f, ax = plt.subplots()

#sns.lineplot(x = xs, y = samples_kernel_lengthscale, ax = ax)
sns.lineplot(x = xs, y = samples_kernel_variance, ax = ax)
ax.set_xlabel("Number of sample (after warmup)", fontsize = 12)
ax.set_ylabel("Value of $\sigma_v^2$", fontsize = 12)
plt.show()

# arviz hyperparameter tuning
warmup_array = np.array([25, 50, 100, 150, 300, 400, 500])
diag_array = np.empty([warmup_array.size, 6])
mean_calc_arrary = np.empty([4,4])


for i in range(warmup_array.size):
    print(str(i) + ": Start")
    for j in range(4):
        pyro.clear_param_store()
        kernel = gp.kernels.RBF(input_dim=1)
        kernel.variance = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(-1.0), torch.tensor(1.0)))
        kernel.lengthscale = pyro.nn.PyroSample(pyro.distributions.LogNormal(torch.tensor(0.0), torch.tensor(2.0)))
        gpmodel = gp.models.GPRegression(X, y, kernel=kernel, noise = torch.tensor(1e-4))

        hmc_kernel = pyro.infer.NUTS(gpmodel.model)

        mcmc = pyro.infer.MCMC(hmc_kernel, num_samples=1500, warmup_steps = warmup_array[i])
        mcmc.run()
        print(str(i) + ": past MCMC")
        
        diag = az.from_pyro(mcmc)
        params =  az.extract(diag)
        print(str(i) + ": past arviz")
        sum2 = az.summary(diag)
        mean_calc_arrary[j][0] = sum2["ess_bulk"][0]
        mean_calc_arrary[j][1] = sum2["ess_bulk"][1]
        mean_calc_arrary[j][2] = sum2["ess_tail"][0]
        mean_calc_arrary[j][3] = sum2["ess_tail"][1]

    temp_array = np.mean(mean_calc_arrary, axis = 0)
    
    diag_array[i][0] = temp_array[0]
    diag_array[i][1] = temp_array[1]
    diag_array[i][2] = temp_array[2]
    diag_array[i][3] = temp_array[3]

# And plotting of results

f, ax = plt.subplots(1, 1)

ax.plot(warmup_array, diag_array[:, 0], color="blue", label="Length_scale (ess_bulk)", linestyle="-")
ax.plot(warmup_array, diag_array[:, 1], color="red", label="Variance (ess_bulk)", linestyle="-")
ax.plot(warmup_array, diag_array[:, 2], color="green", label="Length_scale (ess_tail)", linestyle="-")
ax.plot(warmup_array, diag_array[:, 3], color="yellow", label="Variance (ess_tail)", linestyle="-")
ax.set_ylabel("Effective sampling size $(ess)$", fontsize = 12)
ax.set_xlabel("Number of warmup samples $(W)$", fontsize = 12)
ax.legend(loc='lower right')

plt.show()


