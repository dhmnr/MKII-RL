from stable_baselines3 import PPO
from stable_baselines3.common.atari_wrappers import ClipRewardEnv, WarpFrame
from ...mkii_env import make_env, init_env
import gymnasium as gym
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import (
    SubprocVecEnv,
    VecFrameStack,
    VecTransposeImage,
    DummyVecEnv
)

self_play = False
if self_play:
    pass
else:
    env = make_env()
    model = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=lambda f: f * 2.5e-4,
        n_steps=128,
        batch_size=32,
        n_epochs=4,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.1,
        ent_coef=0.5,
        verbose=1,
        tensorboard_log="./logs/ppo/",
    )

    checkpoint_callback = CheckpointCallback(save_freq=5e4, save_path='./ckpts/ppo/')
    # model.learn(
    #     total_timesteps=10_000_000,
    #     log_interval=1,
    #     callback=checkpoint_callback
    # )

    def make_nenv():
        env = init_env()
        env = gym.wrappers.RecordVideo(env, 'video_folder')
        return env
    model = PPO.load('./ckpts/ppo/rl_model_29600000_steps.zip')
    env = VecTransposeImage(VecFrameStack(DummyVecEnv([make_nenv]), 4))
    obs = env.reset()
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, done, info = env.step(action)

    env.close()