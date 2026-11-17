import torch
import torch.nn.functional as F
from torch import nn
import math
import torch.distributions as td
from torch.distributions import MultivariateNormal


class Flow(nn.Module):
    # flow matching
    def __init__(self, state_dim, action_dim, max_action, flow_steps, device, hidden_dim=750):
        super(Flow, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim + 1, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.e3 = nn.Linear(hidden_dim, hidden_dim)
        self.e4 = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.relu = nn.ReLU()
        self.max_action = max_action
        self.flow_steps = flow_steps
        self.device = device

    def net(self, state, x_t, t):
        z = self.relu(self.e1(torch.cat([state, x_t, t], -1)))
        z = self.relu(self.e2(z))
        z = self.relu(self.e3(z))
        return self.e4(z)

    @torch.no_grad()
    def forward(self, state, noise=None):
        action = torch.randn(state.shape[0], self.action_dim, device=state.device, dtype=state.dtype).clamp(-1,1) if noise is None else noise

        ts = torch.arange(self.flow_steps, dtype=torch.float32) / self.flow_steps
        for i, j in enumerate(ts): # Euler method.
            t = torch.ones(state.size(0), 1, device=self.device) * j
            vels = self.net(state, action, t)
            action = action + vels / self.flow_steps

        return action if self.max_action is None else action.clamp(-self.max_action, self.max_action)

    def train_flow(self, state, action):
        x_0 = torch.randn_like(action).clamp(-1, 1)
        t = torch.rand(action.size(0), 1, device=self.device)
        x_t = (1 - t) * x_0 + t * action
        vel = action - x_0

        u = self.net(state, x_t, t)
        return F.mse_loss(u, vel)

    @torch.no_grad()
    def select_action(self, state):
        state = torch.FloatTensor(state.reshape(1, -1)).to(self.device)
        action = self.forward(state).cpu().data.numpy().flatten()
        return action


class Flow_one(nn.Module):
    # One-step flow matching
    def __init__(self, state_dim, action_dim, max_action, device, hidden_dim=256, init_w=None):
        super(Flow_one, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.e2 = nn.Linear(hidden_dim, hidden_dim)
        self.e3 = nn.Linear(hidden_dim, hidden_dim)
        self.e4 = nn.Linear(hidden_dim, action_dim)

        self.action_dim = action_dim
        self.relu = nn.ReLU()
        self.max_action = max_action
        self.device = device

        if init_w:
            weights_init_(self.e4, init_w=init_w)

    def net(self, state, x_0):
        z = self.relu(self.e1(torch.cat([state, x_0], -1)))
        z = self.relu(self.e2(z))
        z = self.relu(self.e3(z))

        return self.e4(z)

    def forward(self, state, noise=None):
        noise = torch.randn(state.shape[0], self.action_dim, device=state.device, dtype=state.dtype).clamp(-1,1) if noise is None else noise
        action = self.net(state, noise)

        return action if self.max_action is None else self.max_action * torch.tanh(action)


def weights_init_(m, init_w=3e-3):
    m.weight.data.uniform_(-init_w, init_w)
    m.bias.data.uniform_(-init_w, init_w)
