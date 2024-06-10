
import torch
import torch.nn as nn
from typing import Tuple
from collections import namedtuple
from collections import deque
import numpy.random as nr
import numpy as np

import gymnasium as gym

from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import copy

Tensor = torch.DoubleTensor
torch.set_default_dtype(torch.float64)
Transitions = namedtuple('Transitions', ['obs', 'action', 'reward', 'next_obs', 'done'])



class ReplayBuffer(nn.Module):
    def __init__(self, config):
        super().__init__()
        replay_buffer_size = config['replay_buffer_size']
        seed = config['seed']
        self.device = config['device']
        nr.seed(seed)

        self.replay_buffer_size = replay_buffer_size
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

    def append_memory(self,
                      obs,
                      action,
                      reward,
                      next_obs,
                      done: bool):
        self.obs.append(obs)
        self.action.append(action)
        self.reward.append(reward)
        self.next_obs.append(next_obs)
        self.done.append(done)

    def sample(self, batch_size):
        buffer_size = len(self.obs)

        idx = nr.choice(buffer_size,
                        size=min(buffer_size, batch_size),
                        replace=False)
        t = Transitions
        t.obs = torch.tensor(np.array([self.obs[i] for i in idx]), dtype=torch.float64).to(self.device)
        t.action = torch.tensor(np.array([self.action[i] for i in idx]), dtype=torch.float64).to(self.device)
        t.reward = torch.tensor(np.array([self.reward[i] for i in idx]), dtype=torch.float64).to(self.device)
        t.next_obs = torch.tensor(np.array([self.next_obs[i] for i in idx]), dtype=torch.float64).to(self.device)
        t.done = torch.tensor(np.array([self.done[i] for i in idx])).to(self.device)
        return t

    def clear(self):
        self.obs = deque([], maxlen=self.replay_buffer_size)
        self.action = deque([], maxlen=self.replay_buffer_size)
        self.reward = deque([], maxlen=self.replay_buffer_size)
        self.next_obs = deque([], maxlen=self.replay_buffer_size)
        self.done = deque([], maxlen=self.replay_buffer_size)

class ActorNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 feature_extractor,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(ActorNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.feature_extractor = feature_extractor

        n_neurons = (dim_obs,) + dims_hidden_neurons + (dim_action,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor):
        x = self.feature_extractor(obs)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        a = torch.tanh(self.output(x))
        return a

class QCriticNet(nn.Module):
    def __init__(self,
                 dim_obs: int,
                 dim_action: int,
                 feature_extractor,
                 dims_hidden_neurons: Tuple[int] = (64, 64)):
        super(QCriticNet, self).__init__()
        self.n_layers = len(dims_hidden_neurons)
        self.dim_action = dim_action
        self.feature_extractor = feature_extractor
        n_neurons = (dim_obs + dim_action,) + dims_hidden_neurons + (1,)
        for i, (dim_in, dim_out) in enumerate(zip(n_neurons[:-2], n_neurons[1:-1])):
            layer = nn.Linear(dim_in, dim_out).double()
            # nn.Linear: input: (batch_size, n_feature)
            #            output: (batch_size, n_output)
            torch.nn.init.xavier_uniform_(layer.weight)
            torch.nn.init.zeros_(layer.bias)
            exec('self.layer{} = layer'.format(i + 1))  # exec(str): execute a short program written in the str

        self.output = nn.Linear(n_neurons[-2], n_neurons[-1]).double()
        torch.nn.init.xavier_uniform_(self.output.weight)
        torch.nn.init.zeros_(self.output.bias)

    def forward(self, obs: torch.Tensor, action: torch.Tensor):
        x = torch.cat((self.feature_extractor(obs), action), dim=1)
        for i in range(self.n_layers):
            x = eval('torch.relu(self.layer{}(x))'.format(i + 1))
        return self.output(x)

class NatureCNN(nn.Module):
    """CNN from DQN nature paper."""
    def __init__(self, input_channels, state_dim):
        super(NatureCNN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc = nn.Linear(64 * 7 * 7, 512)
        self.out = nn.Linear(512, state_dim)

    def forward(self, x):
        x = x/255.0
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = torch.relu(self.fc(x))
        return self.out(x)

class DDPG(nn.Module):
    def __init__(self, config):
        super(DDPG,self).__init__()
        torch.manual_seed(config['seed'])
        self.config = config
        self.lr_actor = config['lr_actor']  # learning rate
        self.lr_critic = config['lr_critic']
        self.smooth = config['smooth']  # smoothing coefficient for target net
        self.discount = config['discount']  # discount factor
        self.batch_size = config['batch_size']  # mini batch size
        self.sig = config['sig']  # exploration noise

        self.dims_hidden_neurons = config['dims_hidden_neurons']
        self.dim_obs = config['dim_obs']
        self.dim_action = config['dim_action']

        self.device = config['device']
        
        self.buffer = ReplayBuffer(config)

        self.feature_extractor = NatureCNN(input_channels=4, state_dim=512).to(self.device)

        self.actor = ActorNet(dim_obs=self.dim_obs,
                              dim_action=self.dim_action,
                              feature_extractor=self.feature_extractor,
                              dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q = QCriticNet(dim_obs=self.dim_obs,
                            dim_action=self.dim_action,
                            feature_extractor=self.feature_extractor,
                            dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.actor_tar = ActorNet(dim_obs=self.dim_obs,
                                  dim_action=self.dim_action,
                                  feature_extractor=self.feature_extractor,
                                  dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)
        self.Q_tar = QCriticNet(dim_obs=self.dim_obs,
                                dim_action=self.dim_action,
                                feature_extractor=self.feature_extractor,
                                dims_hidden_neurons=self.dims_hidden_neurons).to(self.device)

        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.optimizer_Q = torch.optim.Adam(self.Q.parameters(), lr=self.lr_critic)

    def update(self, buffer):
        # sample from replay memory
        t = buffer.sample(self.batch_size)
        
        with torch.no_grad():
            next_actions = self.actor_tar(t.next_obs)
            tar_Q_val = self.Q_tar(t.next_obs, next_actions)
            tar_Q_val = t.reward + (~t.done) * self.discount * tar_Q_val

        curr_Q_val = self.Q(t.obs, t.action)
        criterion = nn.MSELoss()
        critic_loss = criterion(curr_Q_val, tar_Q_val)

        self.optimizer_Q.zero_grad()
        critic_loss.backward()
        self.optimizer_Q.step()

        actor_loss = -self.Q(t.obs, self.actor(t.obs)).mean()

        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()

        for target_param, param in zip(self.actor_tar.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.smooth * target_param.data + (1 - self.smooth) * param.data)
        for target_param, param in zip(self.Q_tar.parameters(), self.Q.parameters()):
            target_param.data.copy_(self.smooth * target_param.data + (1 - self.smooth) * param.data)

    def act_probabilistic(self, obs: torch.Tensor):
        self.actor.eval()
        exploration_noise = torch.normal(torch.zeros(size=(self.dim_action,)), self.sig).to(self.device)
        a = self.actor(obs) + exploration_noise
        self.actor.train()
        return a

    def act_deterministic(self, obs: torch.Tensor):
        self.actor.eval()
        a = self.actor(obs)
        self.actor.train()
        return a

    def train(self, env):
        steps = 0
        train_writer = SummaryWriter(log_dir='tensorboard/ddpg')
        obs = env.reset()
        ret = [0] * 8
        for timestep in tqdm(range(self.config['max_episode'])):
            obs_tensor = torch.tensor(obs).type(Tensor).to(self.device)
            action = self.act_probabilistic(obs_tensor).detach().cpu().numpy()
            next_obs, reward, done, _ = env.step((action > 0 ).astype(int))

            for i in range(8):  # Assuming done, action, reward have the same length
                ret[i] += reward[i]
                self.buffer.append_memory(obs=obs[i],
                                action=action[i],
                                reward=np.array([reward[i]]),
                                next_obs=next_obs[i],
                                done=np.array([done[i]])
                            )
                if done[i]:
                    print(f"Return {ret[i]}")
                    ret[i] = 0
            self.update(self.buffer)
            obs = copy.deepcopy(next_obs)
            # train_writer.add_scalar('Performance/episodic_return', ret, i_episode)
        env.close()
        train_writer.close()
