import torch
from torch import nn
from bayes.bnn.distributions import dist


class ELBOLoss(nn.Module):
    """Loss function for Bayesian models
    Combines the negative log-likelihood and the KL divergence.
    """
    def __init__(self, model, kl_factor, dataset_size):
        super(ELBOLoss, self).__init__()
        self.model = model
        self.kl_factor = kl_factor
        self.dataset_size = dataset_size

    def forward(self, output, target):
        nll = -dist.Categorical(logits=output).log_prob(target).mean()
        kl = torch.tensor(0.0, device=output.device)
        if self.kl_factor > 0:
            kl = sum(module.parameter_loss().sum() for module in self.model.modules() if hasattr(module, "parameter_loss")) / self.dataset_size
        loss = nll + kl * self.kl_factor
        
        return loss, nll, kl