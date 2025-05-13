#I downloaded the "female-female mixture, Original" and the "male-male mixture, Original" audio files
#from https://www.merl.com/research/highlights/deep-clustering to perform a quick forward-pass test to ensure the architecture is valid. 
#I placed them in a folder named "data".

import numpy as np
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

import torchaudio

import xml.etree.ElementTree as ET
import os
from os import listdir
from os.path import join

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

import time
import os

from architecture import get_pad_masks, Model

#all audio wil be capped at 11 seconds
audio_len = 11
expected_samplerate = 8000

def get_data():
    audios = []
    paddings = []
    for file in os.listdir('data'):
        audio, samplerate = torchaudio.load('data/{}'.format(file))
        assert samplerate == expected_samplerate
        
        padding = (samplerate*audio_len)-audio.shape[1]
        paddings.append(padding)
        
        audio = F.pad(audio, (0, padding))
        audio = audio.unsqueeze(0)
        audios.append(audio)
    #T.vstack(audios) will have the shape: [batch_size=2, 1, timesteps=88000]
    #paddings is a list containing the amount of padding added to each audio clips 
    return T.vstack(audios), paddings

data, pad_data = get_data()
initial_pad_mask, intrapads, interpads = get_pad_masks(data.shape[-1], pad_data)
model = Model(channels=256, input_timesteps=data.shape[-1], chunk_size=250)
output = model(data,intrapads,interpads)
