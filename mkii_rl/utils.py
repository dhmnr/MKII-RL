import torch
import os

def get_device():
    if torch.cuda.is_available():
        os.environ["CUDA_VISIBLE_DEVICES"]="0"
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    print('Found device at: {}'.format(device))
    return device


