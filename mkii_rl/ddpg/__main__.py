import os
import torch
import numpy as np


from ..mkii_env import make_env, record_video
from .ddpg import DDPG


Tensor = torch.DoubleTensor
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"]="0"
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
print('Found device at: {}'.format(device))


config = {
    'dim_obs': 512,
    'dim_action': 12,
    'dims_hidden_neurons': (512, 1024, 512),
    'lr_actor': 0.001,
    'lr_critic': 0.005,
    'smooth': 0.99,
    'discount': 0.92,
    'sig': 0.01,
    'batch_size': 64,
    'replay_buffer_size': 1000000,
    'seed': 1,
    'max_episode': 20000,
    'device':device
}

env = make_env()
ddpg = DDPG(config).to(device)


ddpg.train(env)
torch.save(ddpg, './ddpg_model.bin')

# ddpg = torch.load('./ddpg_model.bin')
record_video(ddpg, device)





