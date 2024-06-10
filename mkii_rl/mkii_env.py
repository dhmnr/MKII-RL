"""
Train an agent using Proximal Policy Optimization from Stable Baselines 3
"""

import gymnasium as gym
import numpy as np
import torch
from gymnasium.wrappers.time_limit import TimeLimit
from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)
from stable_baselines3.common.callbacks import CheckpointCallback
Tensor = torch.DoubleTensor


import retro


class StochasticFrameSkip(gym.Wrapper):
    def __init__(self, env, n, stickprob):
        gym.Wrapper.__init__(self, env)
        self.n = n
        self.stickprob = stickprob
        self.curac = None
        self.rng = np.random.RandomState()
        self.supports_want_render = hasattr(env, "supports_want_render")

    def reset(self, **kwargs):
        self.curac = None
        return self.env.reset(**kwargs)

    def step(self, ac):
        terminated = False
        truncated = False
        totrew = 0
        for i in range(self.n):
            # First step after reset, use action
            if self.curac is None:
                self.curac = ac
            # First substep, delay with probability=stickprob
            elif i == 0:
                if self.rng.rand() > self.stickprob:
                    self.curac = ac
            # Second substep, new action definitely kicks in
            elif i == 1:
                self.curac = ac
            if self.supports_want_render and i < self.n - 1:
                ob, rew, terminated, truncated, info = self.env.step(
                    self.curac,
                    want_render=False,
                )
            else:
                ob, rew, terminated, truncated, info = self.env.step(self.curac)
            totrew += rew
            if terminated or truncated:
                break
        return ob, totrew, terminated, truncated, info


def make_retro(*, game, state=None, max_episode_steps=4500, **kwargs):
    if state is None:
        state = retro.State.DEFAULT
    env = retro.make(game, state, **kwargs)
    env = StochasticFrameSkip(env, n=4, stickprob=0.25)
    if max_episode_steps is not None:
        env = TimeLimit(env, max_episode_steps=max_episode_steps)
    return env

def wrap_deepmind_retro(env):
    env = WarpFrame(env)
    env = ClipRewardEnv(env)
    return env

class MultiBinaryToBoxWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = self.env.action_space.n
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(n,), dtype=np.float32)

    def action(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        binary_action = (action > 0).astype(int)
        return binary_action

class MultiBinaryToDiscreteWrapper(gym.ActionWrapper):
    def __init__(self, env):
        super().__init__(env)
        n = self.env.action_space.n
        self.action_space = gym.spaces.Discrete(2 ** n)

    def action(self, action):
        binary_action = [int(x) for x in format(action, f'0{self.env.action_space.n}b')]
        return np.array(binary_action, dtype=int)

def init_env(game = "MortalKombatII-Genesis", state = retro.State.DEFAULT, scenario=None):
        env = make_retro(game=game, state=state, scenario=scenario, render_mode='rgb_array')
        env = wrap_deepmind_retro(env)
        return env

def make_env():
    return VecTransposeImage(VecFrameStack(SubprocVecEnv([init_env]*8), n_stack=4))

def init_box_env(game = "MortalKombatII-Genesis", state = retro.State.DEFAULT, scenario=None):
    env = make_retro(game=game, state=state, scenario=scenario, render_mode='rgb_array')
    env = wrap_deepmind_retro(env)
    env = MultiBinaryToBoxWrapper(env)
    return env

def make_box_env():
    return VecTransposeImage(VecFrameStack(SubprocVecEnv([init_box_env]*8), n_stack=4))

def init_discrete_env(game = "MortalKombatII-Genesis", state = retro.State.DEFAULT, scenario=None):
    env = make_retro(game=game, state=state, scenario=scenario, render_mode='rgb_array')
    env = wrap_deepmind_retro(env)
    env = MultiBinaryToDiscreteWrapper(env)
    return env

def make_discrete_env():
    return VecTransposeImage(VecFrameStack(SubprocVecEnv([init_discrete_env]*8), n_stack=4))

def record_video(model, device):
    def make_nenv():
        env = init_env()
        env = gym.wrappers.RecordVideo(env, 'video_folder_new')
        return env

    env = VecTransposeImage(VecFrameStack(DummyVecEnv([make_nenv]), 4))

    obs = env.reset()
    done = False

    # Run the policy until the episode is done
    while not done:
        obs_tensor = torch.tensor(obs).type(Tensor).to(device)
        action = model.act_deterministic(obs_tensor).detach().cpu().numpy()
        obs, rewards, done, info = env.step((action > 0 ).astype(int))

    # Close the environment
    env.close()
