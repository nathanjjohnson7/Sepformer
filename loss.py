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

#preds shape -> [num_speakers, timesteps]
#pad_data.shape -> [num_speakers]
def pad_mask_preds(preds, pad_data):
    #assert that there is a padding value for each speaker prediction
    #assert preds.shape[0] == len(pad_data)
    assert preds.shape[0] == pad_data.shape[0]
    
    #pad_data = T.tensor(pad_data) #shape -> [num_speakers]
    pad_data = pad_data.unsqueeze(1) #shape -> [num_speakers, 1]
    
    timestep_positions = T.arange(preds.shape[-1]).unsqueeze(0) #shape: [1, timesteps]
    
    mask = timestep_positions >= (preds.shape[-1]-pad_data)
    
    preds[mask] = 0
    
    return preds

#scale-invariant signal-to-noise ratio
#"SDR, Half-Baked or Well Done?" https://arxiv.org/pdf/1811.02508
#Basic Signal-Noise-Ratio calcuation:
#s -> target, s_hat -> prediction
#SNR = 10log_10((||s||^2)/(||s-s_hat||^2))
#s-s_hat is the noise. When we find the ratio between signal and noise we need to ensure that they are
#orthogonal; the noise shouldn't contain any of the signal, otherwise our ratio will be incorrect.
#To ensure that the s-s_hat is orthogonal to s, we orthogonally project s_hat onto s to get another
#vector that lies on the vector s, s_hat_proj.
#s_hat_proj is othogonal with s_hat_proj-s_hat
#signal = s_hat_proj, noise = s_hat_proj-s_hat
#s_hat_proj = ((s_hat.T * s)/(||s||^2))s

#the paper uses a batch size of 1, so our loss function will only work for batch sizes 1
#output of our model -> [num_speakers=2, batch_size=1, channels=1, timesteps]
#squeeze to [num_speakers=2, timesteps]
#targets should also be of shape [num_speakers, timesteps]
#we assume num_speakers = 2
#pad_data shape -> [num_speakers]
def si_snr_loss(preds, targs, pad_data = T.tensor([0,0]), clip_to_30_db=True):
    assert preds.shape[0] == 2
    assert preds.shape == targs.shape
    
    #the regions which are padded with zeros in the targets, are set to zero in the preds
    preds = pad_mask_preds(preds, pad_data)
    
    #permutation invariant training
    #speaker 1 and speaker 2 of the ouptut might not match speaker 1 and speaker 2 of the labels
    #we have to find the losses: [(speaker1, label1), (speaker2, label2)], 
    #[(speaker1, label2), (speaker2, label1)] and pick the smaller one
    #so we stack two preds and two targs, flipping the last targ, to get the correct pairings
    preds = T.vstack([preds,preds]) #[4, timesteps]
    targs = T.vstack([targs,T.flip(targs, dims=[0])]) #[4, timesteps]
    
    #project preds on target
    preds_proj = (T.sum(preds*targs, dim=1, keepdim=True)
                  /T.sum(targs*targs, dim=1, keepdim=True))*targs
    
    noise = preds_proj - preds
    
    #add eps to avoid division by zero
    eps = T.tensor(1e-8)
    ratio = (T.sum(preds_proj*preds_proj, dim=1, keepdim=True)
              / (T.sum(noise*noise, dim=1, keepdim=True)+eps))
  
    #ln(x)/ln(10) -> log_10(x)
    result = T.tensor(10) * T.log(ratio) / T.log(T.tensor(10.0))
    
    #according to the "Wavesplit: End-to-End Speech Separation by Speaker Clustering" paper,
    #we have to clip the result to 30dB to "limit the influence of the best training predictions"
    if clip_to_30_db:
        result = T.clamp(result, max=30.0)
    
    #result is the snr, a larger snr is desired so we negate so we can minimize
    result *= T.tensor(-1)
    result = result.squeeze().view(2,2) #[num_permutations, num_speakers]

    #sum the speaker losses for each permutation
    permutation_loss = T.sum(result, dim=1) #[num_permutations]

    #get permutation with lowest loss
    loss_idx = permutation_loss.argmin()
    loss = permutation_loss[loss_idx]
    
    #loss_idx of 0: [(speaker1, label1), (speaker2, label2)]
    #loss_idx of 1: [(speaker1, label2), (speaker2, label1)]
    return loss, loss_idx 
