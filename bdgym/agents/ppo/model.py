"""The PPO Actor and Value NN """
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical


class PPOActor(nn.Module):
    """PPO Actor NN Model """

    def __init__(self,
                 input_dim,
                 hidden_sizes,
                 num_actions,
                 activation,
                 output_activation):
        super().__init__()
        layers = [nn.Linear(input_dim[0], hidden_sizes[0]), activation()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_sizes[-1], num_actions))
        layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x, act=None):
        """Forward pass in network """
        pi = self.get_pi(x)
        logp_a = None
        if act is not None:
            logp_a = pi.log_prob(act)
        return pi, logp_a

    def get_pi(self, obs):
        """Get policy """
        return Categorical(logits=self.net(obs))

    def step(self, obs, act):
        """
        Returns
        -------
        pi : a distribution over actions
        logp_a : log likelihood of given action 'act' under pi
        """
        pi = self.get_pi(obs)
        logp_a = pi.log_prob(act)
        return pi, logp_a


class PPOCritic(nn.Module):
    """PPO Critic NN Model """

    def __init__(self, input_dim, hidden_sizes, activation, output_activation):
        super().__init__()
        layers = [nn.Linear(input_dim[0], hidden_sizes[0]), activation()]
        for i in range(1, len(hidden_sizes)):
            layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            layers.append(activation())
        layers.append(nn.Linear(hidden_sizes[-1], 1))
        layers.append(output_activation())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        """Forward pass in network """
        # removes last dimension
        return torch.squeeze(self.net(x), -1)


class PPOActorCritic(nn.Module):
    """ PPO Actor-Critic NN model """

    def __init__(self, obs_dim, hidden_sizes, num_actions,
                 activation=nn.Tanh, output_activation=nn.Identity):
        super().__init__()
        self.actor = PPOActor(
            obs_dim, hidden_sizes, num_actions, activation, output_activation
        )
        self.critic = PPOCritic(
            obs_dim, hidden_sizes, activation, output_activation
        )

    def step(self, obs):
        """Forward pass returning a, v, logp_a """
        with torch.no_grad():
            pi = self.actor.get_pi(obs)
            a = pi.sample()
            logp_a = pi.log_prob(a)
            v = self.critic(obs)
        return a.cpu().numpy(), v.cpu().numpy(), logp_a.cpu().numpy()

    def act(self, obs):
        """Get action """
        with torch.no_grad():
            pi = self.actor.get_pi(obs)
            a = pi.sample()
            return a.cpu().numpy()

    def get_value(self, obs):
        """Get value """
        with torch.no_grad():
            return self.critic(obs).cpu().numpy()

    def save_ac(self, file_path):
        """Save the model """
        torch.save(self.state_dict(), file_path)

    def load_ac(self, file_path):
        """load the model """
        torch.load_state_dict(torch.load(file_path))
