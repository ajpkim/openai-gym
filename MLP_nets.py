import torch
import torch.nn as nn

from gym.spaces import Box, Discrete

# MLPs are a subset of DNNs that generally refers to vanilla fully connected feed-forward neural nets.

def build_MLP(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        activation_func = activation if i < len(layers) - 2 else output_activation
        layers += [nn.Linear(in_features=sizes[i], out_features=sizes[i+1]), activation_func()]
    MLP = nn.Sequential(*layers)
    return MLP

class CategoricalMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, activation=nn.ReLU):
        super(CategoricalMLP, self).__init__()
        sizes = [obs_dim] + hidden_sizes + [action_dim]
        self.logits_net = build_MLP(sizes, activation)

    def forward(self, obs):
        "Return probalitity distribution policy wrt given observations."
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)

    def action_log_probs(obs, actions):
        logits = self.logits_net(obs)
        pi = torch.distributions.Categorical(logits=logits)
        return pi.log_prob(actions)



class GaussianMLP(nn.Module):
    pass
