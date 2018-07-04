# -*- coding: utf-8 -*-

import numpy as np
import scipy.io.wavfile as wav

from scipy.linalg import hankel#, svd

def patch_matrix_build_circ(f, l):
    '''
    Build the patch matrix of f described in A Tale of 2 Bases.
    Input
    f: Nx1 signal.
    l: patch size.
    Output
    hankel: patch matrix of f.
    '''
    last_row = np.zeros(l, dtype=f.dtype)
    last_row[1:] = f[0:l-1]
    return hankel(f,last_row)

def patch_matrix_build_trcon(f, f2):
    '''
    Build the Hankel structured patch matrix of f under the truncated linear convolution setup.
    Input
    f: Nx1 signal.
    f2: (l-1)x1 start of next block
    Output
    hankel: patch matrix of f.
    '''
    last_row = np.zeros(np.size(f2)+1, dtype=f.dtype)
    last_row[1:] = f2
    return hankel(f,last_row)
    
def patch_matrix_stitch_trcon(H_old,H_new):
    l = np.size(H_new,axis=1)
    H_stitched = np.copy(H_new)
    for i in range(l-1):
        H_stitched[-(i+1),(i+1):] = H_old[-(i+1),(i+1):]
    return H_stitched

def patch_matrix_average(F):
    N, l = np.shape(F)
    F_aligned = np.zeros((N, l), dtype=F.dtype)
    F_aligned[:, 0] = F[:, 0]
    for i in range(1, l):
        F_aligned[:, i] = np.roll(F[:, i],i)
    return np.mean(F_aligned, axis=1)
    
def wav_block_import(file_name, block_size, delay=0):
    rate, f = wav.read(file_name)
    if np.size(np.shape(f)) > 1:
        f = np.squeeze(f[:,0])
    if delay > 0:
        f = np.concatenate((np.zeros(delay, dtype=f.dtype), f))
    N = np.size(f,0)
    block_number = int(np.ceil(N/float(block_size)))
    N_ext = block_number*block_size
#    print(N,block_number,N_ext,rate)
    
    f_ext = np.zeros(N_ext, dtype=f.dtype)
    f_ext[:N] = f[:]
    f_blocks =  np.reshape(f_ext,(block_number,block_size))
    return f_blocks, rate