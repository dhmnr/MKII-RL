from stable_baselines3 import DDPG
from ...mkii_env import make_box_env, init_box_env
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)
import numpy as np
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise


self_play = False
if self_play:
    pass
else:
    env = make_box_env()
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    model = DDPG(
        policy="CnnPolicy",
        env=env,
        action_noise=action_noise,
        learning_rate=lambda f: f * 2.5e-4,
        batch_size=32,
        gamma=0.99,
        verbose=1,
        tensorboard_log="./logs/ddpg/",
    )

    checkpoint_callback = CheckpointCallback(save_freq=5e4, save_path='./ckpts/ddpg/')
    # model.learn(
    #     total_timesteps=5_000_000,
    #     log_interval=1,
    #     callback=checkpoint_callback
    # )
    model = DDPG.load('./ckpts/ddpg/rl_model_4800000_steps.zip')

    def make_nenv():
        env = init_box_env()
        env = gym.wrappers.RecordVideo(env, 'video_folder/ddpg')
        return env
    # model = PPO.load('./ckpts/ppo/rl_model_29600000_steps.zip')
    env = VecTransposeImage(VecFrameStack(DummyVecEnv([make_nenv]), 4))
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    env.close()