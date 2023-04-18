import torch as th
import torch.nn as nn
import torch.distributions as D

class MixtureOfGaussians(D.Distribution):
    def __init__(self, means, covs, log_weights):
        assert len(means) == len(covs) == len(log_weights)
        self.means = means
        self.covs = covs
        self.log_weights = log_weights
        self.categorical = D.Categorical(logits=log_weights)
        self.num_components = len(means)
        self.component_distributions = D.Independent(D.Normal(self.means, self.covs),1)
        self.mixture_distribution = D.MixtureSameFamily(self.categorical, self.component_distributions)

    def sample(self, num_samples):
        return self.mixture_distribution.sample(sample_shape=(num_samples,))

    def log_prob(self, value):
        return self.mixture_distribution.log_prob(value)


if __name__ == "__main__":

    th.manual_seed(420)

    # Create a mixture of Gaussians with 2 components
    means = th.tensor([[0.0, 0.0], [1.0, 1.0]])
    covs = th.tensor([[0.5, 0.1], [0.1, 0.5]])
    log_weights = th.log(th.tensor([0.6, 0.4]))
    mixture = MixtureOfGaussians(means, covs, log_weights)

    # Test sampling
    samples = mixture.sample(10)
    print(samples)

    # Test log probability calculation
    log_probs = mixture.log_prob(samples)
    print(th.exp(log_probs))