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

def resample(audio, sample_rate, expected_sample_rate):
    resampler = T.Resample(orig_freq=sample_rate, new_freq=expected_sample_rate)
    audio = resampler(audio)
    return audio

#y: speaker 1, u: speaker 2
#SNR = 10log_10((||y||^2)/(||u||^2))

#expected_db = 10log_10((||y||^2)/(||u*gain||^2))

#||u||^2 = u_1^2 + u_2^2 ... + u_n^2
#||u*gain||^2 = (u_1*gain)^2 + (u_2*gain)^2 ... + (u_n*gain)^2
#||u*gain||^2 = (gain^2)u_1^2 + (gain^2)u_2^2 ... + (gain^2)u_n^2
#||u*gain||^2 = (gain^2)(u_1^2 + u_2^2 ... + u_n^2)
#||u*gain||^2 = (||u||^2)(gain^2)

#expected_db = 10log_10((||y||^2)/(||u||^2 * gain^2))
#expected_db/10 = ln((||y||^2)/(||u||^2 * gain^2))/log10
#log10 * expected_db/10 = ln((||y||^2)/(||u||^2 * gain^2))
#e^(log10 * expected_db/10) = (||y||^2)/(||u||^2 * gain^2)
#gain^2 = (||y||^2)/((||u||^2)*e^(log10 * expected_db/10))
#gain = root((||y||^2)/((||u||^2)*e^(log10 * expected_db/10)))

def get_gain_for_snr(y, u, expected_db):
    l2_norm_y = T.sum(y*y, dim=1, keepdim=True) 
    l2_norm_u = T.sum(u*u, dim=1, keepdim=True)
    gain = T.sqrt(l2_norm_y/(l2_norm_u*T.exp((expected_db/10)*T.log(T.tensor(10)))))
    return gain

def dataloader(foldername, audio_len_seconds=11, expected_sample_rate=8000):
    files = os.listdir(foldername)
    timesteps = expected_sample_rate*audio_len_seconds
    while True:
        file1 = random.choice(files)
        
        new_list = files.copy()
        new_list.remove(file1)
        
        file2 = random.choice(new_list)
        
        audio1, samplerate1 = torchaudio.load(os.path.join(foldername, file1))
        if samplerate1 != expected_sample_rate:
            audio1 = resample(audio1, samplerate1, expected_sample_rate)
        if audio1.shape[-1] >= timesteps:
            #extract a timestep-long clip from the audio, starting from a random point
            pos = random.randint(0, audio1.shape[-1]-timesteps)
            clip1 = audio1[:,pos:pos+timesteps]
            padding1 = T.tensor([0])
        else:
            #if the audio is shorter than the expected length, we take the whole audio as the clip
            # and pad to make for it
            clip1 = F.pad(audio1, (0, timesteps-audio1.shape[-1]))
            padding1 = T.tensor([timesteps-audio1.shape[-1]])
            
        audio2, samplerate2 = torchaudio.load(os.path.join(foldername, file2))
        if samplerate2 != expected_sample_rate:
            audio2 = resample(audio2, samplerate2, expected_sample_rate)
        if audio2.shape[-1] >= timesteps:
            pos = random.randint(0, audio2.shape[-1]-timesteps)
            clip2 = audio2[:,pos:pos+timesteps]
            padding2 = T.tensor([0])
        else:
            clip2 = F.pad(audio2, (0, timesteps-audio2.shape[-1]))
            padding2 = T.tensor([timesteps-audio2.shape[-1]])
        
        #pick a random signal to noise ratio (0-5), between our two clips
        rand_snr = T.tensor([random.random()*5]).unsqueeze(0)
        gain = get_gain_for_snr(clip1, clip2, rand_snr)
        mix = clip1+(clip2*gain)
        #renormalize between 0 and 1
        mix = mix / mix.abs().max(dim=1, keepdim=True)[0]
        
        speaker_pads = T.cat((padding1, padding2))
        #padding of the mix equals the lower padding of the two speakers
        mix_pad = T.min(speaker_pads, dim=0, keepdim=True)[0]
        
        clips = T.vstack((clip1, clip2)) #shape [2,timesteps]
        
        yield mix.unsqueeze(0), mix_pad, clips, speaker_pads
