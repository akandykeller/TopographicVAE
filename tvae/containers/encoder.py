import torch
import torch.nn as nn
from torch.distributions.kl import kl_divergence
from torch.distributions.normal import Normal
from tvae.utils.vis import plot_filters

class Encoder(nn.Module):
    def __init__(self, model, prior, n_params=1):
        super(Encoder, self).__init__()
        self.model = model
        self.prior = prior
        self.n_params = n_params

    def forward(self, x):
        raise NotImplementedError

    def plot_weights(self, name, wandb_on=True):
        name_w_idx = name + '_L{}'
        for idx, layer in enumerate(self.model):
            if isinstance(layer, nn.Conv2d):
                if idx == len(self.model) - 1:
                    s_dim = layer.weight.shape[0] // self.n_params
                    w = layer.weight.view(self.n_params, s_dim, *layer.weight.shape[1:])
                    for i in range(self.n_params):
                        plot_filters(w[i], name_w_idx.format(idx) + f'_p{i}', wandb_on=wandb_on)
                else:                   
                    plot_filters(layer.weight, name_w_idx.format(idx), wandb_on=wandb_on)

    def sample(self, x, n_samples, cap_dim):
        return self.prior.sample(torch.Size([n_samples, cap_dim**2, 1, 1])).to(x.device)

class Gaussian_Encoder(Encoder):
    def __init__(self, model, loc=0.0, scale=1.0, post_scale=1.0):
        super(Gaussian_Encoder, self).__init__(model, Normal(loc=loc, scale=scale), n_params=2)
        self.post_scale = post_scale

    def forward(self, x):
        mu_logvar = self.model(x)
        s_dim = mu_logvar.shape[1] // self.n_params
        mu = mu_logvar[:, :s_dim]
        logvar = mu_logvar[:, s_dim:]
        std = torch.exp(logvar / 2.0)

        q = Normal(mu, std)
        z = q.rsample() * self.post_scale

        log_q_z = q.log_prob(z)
        log_p_z = self.prior.log_prob(z)

        kl = kl_divergence(q, self.prior)
        return z, kl, log_q_z, log_p_z