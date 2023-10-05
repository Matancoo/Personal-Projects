import random
import matplotlib.pyplot as plt
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import DataLoader
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from torch.utils.data import Dataset
def sample_datapoints(a: int, b: int, samples_num: int, samples_dim: int):
    '''
    Function to generate points uniformly,
    inside a square, ranging from x = [−1,1], y = [−1,1]
    :param samples_num:
    :return: np.tensor
    '''
    return a + (b - a) * torch.rand(samples_num, samples_dim)
def stochastic_exp_s(t):
    '''
    Noise schedule function with random noise
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    noise = torch.randn(t.shape) * 0.0001  # Small amount of noise
    return torch.exp(8.1*(t - 1)) + noise

def exp_scheduler(t):
    '''
    Noise schedule function
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return torch.exp(5 * (t - 1))
def sigmoid_scheduler(t):
    '''
    Sigmoid noise schedule function
    This scheduler increases slowly at first, more quickly in the middle, and then slows again towards the end.
    This could be helpful when your model needs to adapt to a variety of noise levels throughout training
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return torch.sigmoid(t)
def sqrt_scheduler(t):
    '''
    Square root noise schedule function
    These schedulers can be useful if you want the amount of noise added to increase slowly.
    This might be appropriate for problems where adding too much noise early in the training process could be detrimental.
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return torch.sqrt(t)
def derived_scheduler(t):
    '''
    Noise schedule function derived according to t
    :param t: timestep
    :return: standard deviation of the noise added at each timestep
    '''
    return 5 * torch.exp(5 * (t - 1))
