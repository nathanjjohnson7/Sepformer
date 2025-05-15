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

class Chunking(nn.Module):
    def __init__(self, chunk_size, input_timesteps):
        super(Chunking, self).__init__()
        self.input_timesteps = input_timesteps
        self.chunk_size = chunk_size
        self.half_chunk = int(chunk_size/2)
        #the chunk overlap is 50%
        #how much padding do we need for sequence length to be divisible by chunk_size/2
        self.padding = int(self.half_chunk-(input_timesteps%self.half_chunk))
        if(self.padding == self.half_chunk):
            self.padding = 0
        
        self.cols = self.chunk_size
        self.rows = int(((self.input_timesteps+self.padding)/self.half_chunk)-1)
        #get the indexes with a chunked, overlapping pattern
        self.index = (T.arange(self.cols).unsqueeze(0) + 
                      self.half_chunk * T.arange(self.rows).unsqueeze(1)).mT
    def forward(self, x):
        #x input shape: [batch_size, channels, input_timesteps]
        assert x.shape[-1] == self.input_timesteps
        x = F.pad(x, (0, self.padding)) #shape: [batch_size, channels, chunkable_timesteps]
        
        return x[:,:,self.index] #shape: [batch_size, channels, chunk_size, num_chunks]

def get_pad_masks(seq_len, pad_data, kernel_size=16, stride=8, chunk_size=250):
    #seq_len: number of timesteps in the audio
    #pad_data: list with length equal to batch_size, each element denotes number of padding timesteps
    # for the associated datapoint
    
    pad_data = T.tensor(pad_data)
    
    #create a mask [batch_size, timesteps], where non-padding values are 1 and padding values are 0
    initial_pad_mask = T.ones((pad_data.shape[0], seq_len)) #[batch_size, timesteps]
    indices = T.arange(seq_len).view(1, seq_len)  # [1,timesteps]
    paddings_expanded = pad_data.view(pad_data.shape[0], 1)  # [batch_size,1]
    keep_mask = indices >= (seq_len - paddings_expanded) #[batch_size, timesteps]
    initial_pad_mask[keep_mask] = 0 #[batch_size, timesteps]
    
    initial_pad_mask = initial_pad_mask.unsqueeze(1) #[batch_size, 1, timesteps]
    
    #account for effect of encoder and chunking on paddings
    padding_conv = nn.Conv1d(
                    in_channels=1, 
                    out_channels=1, 
                    kernel_size=kernel_size, 
                    stride=stride, 
                    bias=False
                   )
    with T.no_grad():
        padding_conv.weight.fill_(1.0)
    conved_paddings = padding_conv(initial_pad_mask)
    conved_paddings = T.clamp(conved_paddings, min=0.0, max=1.0)
    c = Chunking(chunk_size=chunk_size, input_timesteps=conved_paddings.shape[-1])
    chunked_paddings = c(conved_paddings) #[batch_size, 1, chunk_size, num_chunks]
    
    chunked_paddings = chunked_paddings.squeeze() #[batch_size, chunk_size, num_chunks]
    chunked_paddings = T.permute(chunked_paddings, (0,2,1)) #[batch_size, num_chunks, chunk_size]
    
    #counts: [batch_size, num_chunks]
    #for each datapoint, for each chunk, how many padded positions are there?
    counts = (T.count_nonzero(chunked_paddings, dim=2)-chunked_paddings.shape[-1])*-1
    print(counts)
    
    #padding mask for the intra transformer [batch_size, num_chunks, chunk_size, chunk_size]
    intrapads = T.zeros((*counts.shape, chunk_size, chunk_size))
    # Create row and column masks
    row_idx = T.arange(chunk_size).view(1, 1, chunk_size, 1)  # [1,1,chunk_size,1]
    col_idx = T.arange(chunk_size).view(1, 1, 1, chunk_size)  # [1,1,1,chunk_size]
    counts_expanded = counts.view(*counts.shape, 1, 1)        # [batch_size,num_chunks,1,1]

    # mask that selects the last n rows and columns, where n denotes the number of padding values
    # in the chunk. We first select the non-padded values and use `==0` to get the padded values
    keep_mask = ((row_idx < (chunk_size - counts_expanded)) 
                 & (col_idx < (chunk_size - counts_expanded)))==0
    
    #set padding values to -inf
    intrapads[keep_mask] = -1e9 #float('-inf')
    
    #for the inter transformer, which attends across chunks, we pad out all chunks that are
    # completely filled iwth padding. So we count the chunks where the padded count 
    # equals the chunk_size
    #shape [batch_size]
    counts_inter = T.count_nonzero(counts==chunk_size, dim=1)
    print(counts_inter)
    
    num_chunks = counts.shape[-1]
    
    #[batch_size, num_chunks, num_chunks]
    interpads = T.zeros((*counts_inter.shape, num_chunks, num_chunks))
    
    # Create row and column masks
    row_idx = T.arange(num_chunks).view(1, num_chunks, 1)  # [1,num_chunks,1]
    col_idx = T.arange(num_chunks).view(1, 1, num_chunks)  # [1,1,num_chunks]
    c_expanded = counts_inter.view(*counts_inter.shape, 1, 1)  # [batch_size,1,1]

    # mask that selects the last n rows and columns, where n denotes the number of completely
    # padded chunks. We first select the non-padding chunks and use `==0` to get the padding chunks
    keep_mask = ((row_idx < (num_chunks - c_expanded)) 
                 & (col_idx < (num_chunks - c_expanded)))==0

    interpads[keep_mask] = -1e9 #float('-inf')
    interpads = interpads.unsqueeze(1) #[batch_size, 1, num_chunks, num_chunks]
    
    return initial_pad_mask, intrapads, interpads

class Encoder(nn.Module):
    def __init__(self, channels, stride=8):
        super(Encoder, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=channels, kernel_size=16, stride=stride)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        #x input shape: [batch_size, 1, seq_len]
        return self.relu(self.conv(x))

class Norm_Linear(nn.Module):
    def __init__(self, channels, timesteps):
        super(Norm_Linear, self).__init__()
        #we're applying global LayerNorm (applied over channels and timesteps)
        self.gLN = nn.LayerNorm((channels, timesteps))
        self.lin = nn.Linear(channels, channels)
    def forward(self, x):
        #x.shape: [batch_size, channels, timesteps]
        x = self.gLN(x)
        x = x.mT #shape: [batch_size, timesteps, channels]
        x = self.lin(x)
        return x.mT #shape: [batch_size, channels, timesteps]

class Multi_Head_Attention(nn.Module):
    def __init__(self, d_model, h=8):
        super(Multi_Head_Attention, self).__init__()
        self.h = h
        self.d_model = d_model
        self.q_linear = nn.Linear(self.d_model, self.d_model*self.h)
        self.k_linear = nn.Linear(self.d_model, self.d_model*self.h)
        self.v_linear = nn.Linear(self.d_model, self.d_model*self.h)
        
        self.softmax = nn.Softmax(dim=-1)
        
        self.final_linear = nn.Linear(self.d_model*self.h, self.d_model)
    def forward(self, x, mask=None):
        #print("input: ", x.shape)
        #intra input.shape -> [batch_size, num_chunks, chunk_size, channels]
        #inter input.shape -> [batch_size, channels, num_chunks, chunk_size]
        #the query, key and value inputs are assumed to be identical
        batch_size = x.shape[0]
        dim1 = x.shape[1]
        dim2 = x.shape[2]
        assert self.d_model == x.shape[3]
        
        queries = self.q_linear(x)
        keys = self.k_linear(x)
        values = self.v_linear(x)
        #print("queries: {}, keys: {}, values: {}".format(queries.shape, keys.shape, values.shape))
        
        #reshape to [h, batch_size, dim1, dim2, d_model]
        #intra -> [h, batch_size, num_chunks, chunk_size, channels]
        #inter -> [h, batch_size, channels, num_chunks, chunk_size]
        queries = queries.view(batch_size, dim1, dim2, self.d_model, self.h)
        queries = T.permute(queries, (4, 0, 1, 2, 3))
        #print("queries: ", queries.shape)
        
        #reshape to [h, batch_size, dim1, d_model, dim2]
        #intra -> [h, batch_size, num_chunks, channels, chunk_size]
        #inter -> [h, batch_size, channels, chunk_size, num_chunks]
        keys = keys.view(batch_size, dim1, dim2, self.d_model, self.h)
        keys = T.permute(keys, (4, 0, 1, 3, 2))
        #print("keys: ", keys.shape)
        
        #reshape to [h, batch_size, dim1, dim2, d_model]
        #intra -> [h, batch_size, num_chunks, chunk_size, channels]
        #inter -> [h, batch_size, channels, num_chunks, chunk_size]
        values = values.view(batch_size, dim1, dim2, self.d_model, self.h)
        values = T.permute(values, (4, 0, 1, 2, 3))
        #print("values: ", values.shape)
        
        multipled = T.matmul(queries, keys)
        scaled = multipled/math.sqrt(self.d_model)
        if mask!=None:
            scaled = scaled + mask
            
        #attention weights: [h, batch_size, dim1, dim2, dim2]
        #intra -> [h, batch_size, num_chunks, chunk_size, chunk_size] attending within each chunks
        #inter -> [h, batch_size, channels, num_chunks, num_chunks] attending between all chunks
        softmaxed = self.softmax(scaled)
        #print("attention_weights: ", softmaxed.shape)
                
        #final matmul shape: [h, batch_size, dim1, dim2, d_model]
        #intra -> [h, batch_size, num_chunks, chunk_size, channels]
        #inter -> [h, batch_size, channels, num_chunks, chunk_size]
        final_matmul = T.matmul(softmaxed, values)
        #print("final_matmul: ", final_matmul.shape)
        
        #new shape: [batch_size, dim1, dim2, d_model*h]
        #intra -> [batch_size, num_chunks, chunk_size, channels*h]
        #inter -> [batch_size, channels, num_chunks, chunk_size*h]
        final_matmul = T.permute(final_matmul, (1, 2, 3, 4, 0))
        reshaped = final_matmul.reshape(batch_size, dim1, dim2, self.d_model*self.h)
        #print("reshaped: ", reshaped.shape)
        
        output = self.final_linear(reshaped)
        #print("output: ", output.shape)
        
        #output: [batch_size, dim1, dim2, d_model]
        #intra -> [batch_size, num_chunks, chunk_size, channels]
        #inter -> [batch_size, channels, num_chunks, chunk_size]
        
        return output

class Feed_Forward(nn.Module):
    def __init__(self, d_model):
        super(Feed_Forward, self).__init__()
        self.d_model = d_model
        self.linear1 = nn.Linear(self.d_model, self.d_model*4)
        self.linear2 = nn.Linear(self.d_model*4, self.d_model)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        return self.linear2(self.relu(self.linear1(x)))

class Positional_Encoding(nn.Module):
    def __init__(self, max_len, d_model):
        super(Positional_Encoding, self).__init__()
        self.pe = T.zeros(max_len, d_model)
        self.pe[:, 0::2] = T.sin(T.arange(max_len).unsqueeze(1)/(10000**(T.arange(0, d_model, 2)/d_model)))
        self.pe[:, 1::2] = T.cos(T.arange(max_len).unsqueeze(1)/(10000**(T.arange(1, d_model, 2)/d_model)))
    def forward(self, x):
        x = x + self.pe[:x.size(1)]
        return x

class Intra_Transformer(nn.Module):
    def __init__(self, channels, chunk_size, num_chunks, k=8):
        super(Intra_Transformer, self).__init__()
        self.channels = channels
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.k = k
        self.pe = Positional_Encoding(max_len=chunk_size, d_model=channels)
        #we're applying global LayerNorm (applied over num_chunks, chunk_size, channels)
        self.gLN = nn.LayerNorm((num_chunks, chunk_size, channels))
        #self.gLN = nn.LayerNorm((channels, chunk_size, num_chunks))
        self.mha = Multi_Head_Attention(channels, h=8)
        self.ffw = Feed_Forward(channels)
    def forward(self, x, mask=None):
        #input shape: [batch_size, channels, chunk_size, num_chunks]
        #permute to:  [batch_size, num_chunks, chunk_size, channels]
        #we permute to the above shape to allow us to attend each chunk with itself
        x = T.permute(x, (0,3,2,1))
        
        batch_size = x.shape[0]
        
        #positionally encode
        #reshape to [batch_size*num_chunks, chunk_size, channels] for positional encoding
        x_prime = x.contiguous().view(batch_size*self.num_chunks, *x.shape[2:])
        x_prime = self.pe(x_prime)
        #reshape to [batch_size, num_chunks, chunk_size, channels]
        x_prime = x_prime.contiguous().view(batch_size, self.num_chunks, *x.shape[2:])
        
        
        for i in range(self.k):
            x_2_prime = self.gLN(x_prime)
            x_2_prime = self.mha(x_2_prime, mask=mask)
            x_2_prime += x_prime
            x_3_prime = self.gLN(x_2_prime)
            x_3_prime = self.ffw(x_3_prime)
            x_3_prime += x_2_prime
            x_prime = x_3_prime
            
        out = x_3_prime + x
        
        #permute from: [batch_size, num_chunks, chunk_size, channels]
        #to: [batch_size, channels, chunk_size, num_chunks]
        out = T.permute(out, (0, 3, 2, 1))
            
        return out

class Inter_Transformer(nn.Module):
    def __init__(self, channels, chunk_size, num_chunks, k=8):
        super(Inter_Transformer, self).__init__()
        self.channels = channels
        self.chunk_size = chunk_size
        self.num_chunks = num_chunks
        self.k = k
        self.pe = Positional_Encoding(max_len=num_chunks, d_model=chunk_size)
        #we're applying global LayerNorm (applied over channels, num_chunks, chunk_size)
        self.gLN = nn.LayerNorm((channels, num_chunks, chunk_size))
        self.mha = Multi_Head_Attention(chunk_size, h=8)
        self.ffw = Feed_Forward(chunk_size)
    def forward(self, x, mask=None):
        #input shape: [batch_size, channels, chunk_size, num_chunks]
        #permute to:  [batch_size, channels, num_chunks, chunk_size]
        #we permute to the above shape to allow us to attend each chunk with itself
        x = T.permute(x, (0,1,3,2))
        
        batch_size = x.shape[0]
        
        #positionally encode
        #reshape to [batch_size*channels, num_chunks, chunk_size] for positional encoding
        x_prime = x.contiguous().view(batch_size*self.channels, *x.shape[2:])
        x_prime = self.pe(x_prime)
        #reshape to [batch_size, channels, num_chunks, chunk_size]
        x_prime = x_prime.contiguous().view(batch_size, self.channels, *x.shape[2:])
        
        for i in range(self.k):
            x_2_prime = self.gLN(x_prime)
            x_2_prime = self.mha(x_2_prime, mask)
            x_2_prime += x_prime
            x_3_prime = self.gLN(x_2_prime)
            x_3_prime = self.ffw(x_3_prime)
            x_3_prime += x_2_prime
            x_prime = x_3_prime
            
        out = x_3_prime + x
        
        #permute from: [batch_size, channels, num_chunks, chunk_size]
        #to: [batch_size, channels, chunk_size, num_chunks]
        out = T.permute(out, (0, 1, 3, 2))
            
        return out

class Sepformer_Block(nn.Module):
    def __init__(self, channels, chunk_size, num_chunks, N=2):
        super(Sepformer_Block, self).__init__()
        self.N = N
        self.intra = Intra_Transformer(channels, chunk_size, num_chunks)
        self.inter = Inter_Transformer(channels, chunk_size, num_chunks)
    def forward(self, x, intrapad_mask=None, interpad_mask=None):
        for i in range(self.N):
            x = self.intra(x, mask=intrapad_mask)
            x = self.inter(x, mask=interpad_mask)
        
        return x

class PReLU_Linear(nn.Module):
    def __init__(self, channels, num_speakers=2):
        super(PReLU_Linear, self).__init__()
        self.channels = channels
        self.num_speakers = num_speakers
        self.prelu = nn.PReLU()
        self.lin = nn.Linear(channels, channels*num_speakers)
    def forward(self, x):
        #x input shape -> [batch_size, channels, chunk_size, num_chunks]
        batch_size = x.shape[0]
        chunk_size = x.shape[2]
        num_chunks = x.shape[3]
        #permute to -> [batch_size, chunk_size, num_chunks, channels]
        x = T.permute(x, (0, 2, 3, 1))
        x = self.prelu(x)
        x = self.lin(x)
        #output of linear shape -> [batch_size, chunk_size, num_chunks, channels*num_speakers]
        x = x.view(batch_size, chunk_size, num_chunks, self.channels, self.num_speakers)
        #permute to: [batch_size, channels, num_speakers, chunk_size, num_chunks]
        x = T.permute(x, (0, 3, 4, 1, 2))
        return x

class Overlap_Add(nn.Module):
    def __init__(self, chunk_size, input_timesteps):
        super(Overlap_Add, self).__init__()
        self.input_timesteps = input_timesteps
        self.chunk_size = chunk_size
        self.half_chunk = int(chunk_size/2)
        
        self.rows = self.chunk_size
        self.cols = int((self.input_timesteps/self.half_chunk)-1)
        #get the overlapping indices, flattened
        self.index = (T.arange(self.rows).unsqueeze(1) + 
                      self.half_chunk * T.arange(self.cols).unsqueeze(0)).flatten()
        self.index = self.index.view(1,1,1,*self.index.shape)
        self.out_len = self.rows + self.half_chunk * (self.cols - 1)
    def forward(self, x):
        #x input shape: [batch_size, channels, num_speakers, chunk_size, num_chunks]
        assert x.shape[-2] == self.rows
        assert x.shape[-1] == self.cols
        
        #[batch_size, channels, num_speakers, chunk_size*num_chunks]
        x = x.contiguous().view(*x.shape[:-2],x.shape[-2]*x.shape[-1])
        
        #shape: [batch_size, channels, num_speakers, seq_len]
        result = T.zeros((*x.shape[:-1], self.out_len), dtype=x.dtype)
        
        result.scatter_add_(3, self.index.repeat(*x.shape[:-1],1), x)
        
        return result

class FFN_ReLU(nn.Module):
    def __init__(self, channels):
        super(FFN_ReLU, self).__init__()
        self.linear1 = nn.Linear(channels, channels*2)
        self.linear2 = nn.Linear(channels*2, channels)
        self.relu = nn.ReLU(inplace = True)
    def forward(self, x):
        #x input shape: [batch_size, channels, num_speakers, seq_len]
        #permute to: [batch_size, num_speakers, seq_len, channels]
        x = T.permute(x, (0,2,3,1))
        
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        
        #permute to: [num_speakers, batch_size, channels, seq_len]
        x = T.permute(x, (1,0,3,2))
        
        return x

class Decoder(nn.Module):
    def __init__(self, in_channels, kernel_size=16, stride=8, num_speakers=2):
        super(Decoder, self).__init__()
        self.num_speakers = num_speakers
        self.convT = nn.ConvTranspose1d(
                        in_channels=in_channels, 
                        out_channels=1, 
                        kernel_size=kernel_size,
                        stride=stride
                     )
    def forward(self, masks, h):
        #masks input shape: [num_speakers, batch_size, channels, seq_len+padding_for_chunking]
        #h input shape: [batch_size, channels, seq_len]
        batch_size = h.shape[0]
        channels = h.shape[1]
        seq_len = h.shape[2]
        
        masks = masks[:,:,:,:h.shape[-1]]
        
        #element-wise multiply
        x = masks*h
        
        #collapse num_speakers into batch_size
        x = x.contiguous().view(self.num_speakers*batch_size, channels, seq_len)
        
        x = self.convT(x)
        
        #extract num_speakers from batch_size
        x = x.contiguous().view(self.num_speakers, batch_size, 1, x.shape[-1])
        
        #final shape: [num_speakers, batch_size, 1, seq_len]
        return x

#timesteps: number of timesteps after padding
class Model(nn.Module):
    def __init__(self, channels, input_timesteps, chunk_size, num_speakers=2):
        super(Model, self).__init__()
        self.encoder = Encoder(channels)
        
        kernel = 16
        stride = 8
        timesteps = int((input_timesteps-kernel)/stride) + 1
        
        self.norm_linear = Norm_Linear(channels, timesteps)
        self.chunking = Chunking(chunk_size, timesteps)
        self.sepformer = Sepformer_Block(
                            channels, 
                            chunk_size, 
                            self.chunking.rows #num_chunks
                         )
        self.prelu_linear = PReLU_Linear(channels, num_speakers)
        self.overlap_add = Overlap_Add(chunk_size, timesteps+self.chunking.padding)
        self.ffn_relu = FFN_ReLU(channels)
        self.decoder = Decoder(channels)
    def forward(self, x, intrapad_mask=None, interpad_mask=None):
        #print(x.shape)
        x = self.encoder(x)
        h = x
        #print("encoder {}".format(x.shape))
        x = self.norm_linear(x)
        #print("norm_linear {}".format(x.shape))
        x = self.chunking(x)
        #print("chunking {}".format(x.shape))
        x = self.sepformer(x, intrapad_mask, interpad_mask)
        #print("sepformer {}".format(x.shape))
        x = self.prelu_linear(x)
        #print("prelu_linear {}".format(x.shape))
        x = self.overlap_add(x)
        #print("overlap_add {}".format(x.shape))
        x = self.ffn_relu(x)
        #print("ffn_relu {}".format(x.shape))
        x = self.decoder(x,h)
        #print("decoder {}".format(x.shape))
        return x
