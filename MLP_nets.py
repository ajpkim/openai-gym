import pdb
import torch
import torch.nn as nn

from gym.spaces import Box, Discrete

# MLPs are a subset of DNNs that generally refers to vanilla fully connected feed-forward neural nets.

# Sergey Levine Actor-Critic lecture notes: http://rail.eecs.berkeley.edu/deeprlcourse-fa19/static/slides/lec-6.pdf


## TODO
# Fix the "str" obj not callable wrt activation_func() from config

def build_MLP(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    layers = []
    for i in range(len(sizes) - 1):
        activation_func = activation if i < len(layers) - 2 else output_activation
        layers += [nn.Linear(in_features=sizes[i], out_features=sizes[i+1]), activation_func()]
    MLP = nn.Sequential(*layers)
    return MLP

class CategoricalPolicyMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, activation=nn.ReLU):
        super(MLP_CategoricalPolicyNet, self).__init__()
        sizes = [obs_dim] + hidden_sizes + [action_dim]
        self.logits_net = build_MLP(sizes, activation)

    def forward(self, obs):
        "Return probability distribution policy wrt given observations."
        logits = self.logits_net(obs)
        return torch.distributions.Categorical(logits=logits)

    def action_log_probs(self, obs, actions):
        logits = self.logits_net(obs)
        pi = torch.distributions.Categorical(logits=logits)
        return pi.log_prob(actions)


# Continuous action spaces use gaussian distribution policy where the model output
# is the mean + std which defines the normal distribution to use for action selection (policy).

# Going to use state-dependent Gaussian distribution over continuous actions.
# To get action for state we generate mu and sigma for state and then sample
# from this gaussian distribution.

# For continuous action spaces we cannot assign a probability to every action because
# the space of possible actions is infinite. Instead, we learn the params of some
# distribution over actions

# Mu and sigma for policy pi are a function of the state.
# Mu can be any parameterized function (here we use a nn, and denote with theta_mu)
# Parameterized function sigma has one constraint, it must be positive.

# Sigma controls the degree of exploration and we typically initialize to be large.
# As learning progresses, we expect the variance to shrink and the policy to concentrate
# on the best action in each state


# Architecture possibilities:
# 1. Use 2 different networks to generate the mu and sigma for our
#    policy distribution. We can use softplus to ensure that sigma is positive
#    (softplus(x) = log(e**x + 1))
# 2. Have 2 units in final layer, one repr. mu and the other sigma
# 3. Have shared layers and distinct mu and sigma network heads (like AlphaConnect).


# Can use Tanh or something to squeeze mu into [-1,1]

# TODO
# Change the default activations

class GaussianPolicyMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, activation=nn.ReLU):
        super(GaussianPolicyMLP, self).__init__()
        sizes = [obs_dim] + hidden_sizes + [action_dim]
        self.mu_net = build_MLP(sizes, activation)
        self.log_std_net = build_MLP(sizes, activation)

    def forward(self, obs):
        mu = self.mu_net(obs)
        log_std = self.log_std_net(obs)
        sigma = torch.exp(log_std)
        return torch.distributions.Normal(mu, sigma)

    def action_log_probs(self, obs, actions):
        pi = self.forward(obs)
        return pi.log_prob(actions)


class ValueNetworkMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, activation=nn.ReLU):
        super(ValueNetworkMLP, self).__init__()
        sizes = [obs_dim] + hidden_sizes + [action_dim]
        self.value_net = build_mlp(sizes, activation)

    def forward(self, obs):
        v = self.value_net(obs)
        # TODO
        # TRY WITHOUT SQEEZE
        return torch.squeeze(v, -1)

class ActorCriticMLP(nn.Module):
    def __init__(self, obs_dim, hidden_sizes, action_dim, activation=nn.Tanh):
        supert(MLPActorCritic, self).__init__()

        ### Policy network depends on env (Categorical vs. Gaussian)
        pass

    def forward(self, obs):
        pass

    def action_log_probs(self, obs, actions):
        pass

    def act(self, obs):
        pass
    
