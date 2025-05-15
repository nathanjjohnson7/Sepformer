import numpy as np
import math

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import torchaudio

import matplotlib
import matplotlib.pyplot as plt

import os
import random

from dataloader import dataloader
from architecture import get_pad_masks, Model
from loss import si_snr_loss

train_dataloader = dataloader('train')
#input_timesteps equals the audio_len_seconds*expected_sample_rate of the dataloader
model = Model(channels = 256, input_timesteps = 88000, chunk_size = 250)
lr = 1.5e-4
optimizer = T.optim.Adam(model.parameters(), lr = lr)
max_norm = 5.0
training_steps = 10000
batch_size = 1
losses = []

def train():
    model.train()
    for i in range(training_steps):
        mix, mix_pad, clips, speaker_pads = next(train_dataloader)
        initial_pad_mask, intrapads, interpads = get_pad_masks(mix.shape[-1], mix_pad)
        out = model(mix,intrapads,interpads)
        
        #[num_speakers=2, batch_size=1, channels=1, timesteps] to [num_speakers=2, timesteps] 
        out = out.squeeze(1).squeeze(1)
        loss, loss_idx = si_snr_loss(out, clips, speaker_pads)
        
        optimizer.zero_grad()
        loss.backward()
        T.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()
        losses.append(loss)
        print("Training step: {}, loss: {}".format(i, loss))

if __name__ == ‘__main__’:
    train()
