import torch as th
import torch.nn as nn
import torch.distributions as D

# class MixtureOfGaussians(D.Distribution):
#     def __init__(self, means, variances, log_weights):
#         assert len(means) == len(variances) == len(log_weights)
#         self.means = means
#         self.variances = variances
#         self.log_weights = log_weights
#         self.categorical = D.Categorical(logits=log_weights)
#         self.num_components = len(means)
#         self.component_distributions = D.Independent(D.Normal(self.means, self.variances),1)
#         self.mixture_distribution = D.MixtureSameFamily(self.categorical, self.component_distributions)

#     def sample(self, num_samples):
#         return self.mixture_distribution.sample(sample_shape=(num_samples,))

#     def log_prob(self, value):
#         return self.mixture_distribution.log_prob(value)

def mog_log_prob(samples,means,variances,log_weights):
    
    samples = samples.reshape(-1,1)
    means = means.reshape(1,-1)
    variances = variances.reshape(1,-1)
    log_weights = log_weights.reshape(1,-1)
    # [B,K]
    normal_log_probs = -0.5*(samples-means)**2/variances - 0.5*th.log(2*th.pi*variances)
    # [B]
    log_probs = th.logsumexp(normal_log_probs + log_weights, dim=1)
    return log_probs


if __name__ == "__main__":

    th.manual_seed(420)

    # Create a mixture of 1D Gaussians with 2 components
    means = th.tensor([0.0,1.0])
    variances = th.tensor([[1.0,0.5]])
    log_weights = th.log(th.tensor([0.6,0.4]))

    samples = 3*th.rand(10)
    print(samples)

    log_probs = mog_log_prob(samples,means,variances,log_weights)
    print(th.exp(log_probs))
