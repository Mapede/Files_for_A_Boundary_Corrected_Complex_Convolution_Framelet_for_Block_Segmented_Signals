# -*- coding: utf-8 -*-
"""
Created on Fri Jan 26 13:26:24 2018

@author: mathias
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.linalg import svd
from scipy.signal import spectrogram
from pywt import threshold

import functions_article as fn

block_size = 800
patch_size = 256
block_num = 60
#rate = 16000

#t = np.linspace(0, block_size*block_num/float(rate), num=block_size*block_num, endpoint=False)
#
#f1 = 400.0
#f2 = 7950.0
#
#signal = np.sin(t*2*np.pi*f1) + 0.2*np.sin(t*2*np.pi*f2)
#signal = np.reshape(signal, (block_num, block_size))

signal, rate = fn.wav_block_import('timit90.wav', block_size)
signal = signal.astype(np.float64)
block_num = np.size(signal, axis=0)

prev_synth_patch_matrix = np.zeros((block_size,patch_size), dtype = signal.dtype)
signal_processed = np.zeros(np.shape(signal), dtype = signal.dtype)
for m in range(block_num): # Loop over blocks in signal
    # Analysis
    #   Patch Matrix
    f = signal[m]
    f2 = np.zeros(patch_size-1)
    
    if m < block_num - 1:
        f2 = signal[m+1,:patch_size-1]
        
    patch_matrix = fn.patch_matrix_build_trcon(f, f2)
    
    #   Coefficient matrix
    phi = svd(patch_matrix, False)[0]
    coeff_matrix = np.dot(np.conj(phi.T), patch_matrix)
    coeff_matrix = np.fft.fft(coeff_matrix, axis=1)

    
    # Synthesis
    synth_patch_matrix = np.fft.ifft(coeff_matrix, axis=1)
    synth_patch_matrix = np.dot(phi, synth_patch_matrix)

    #   Patch Matrix averaging
    stitched_patch_matrix = fn.patch_matrix_stitch_trcon(prev_synth_patch_matrix, synth_patch_matrix)
    synth_f = fn.patch_matrix_average(stitched_patch_matrix)
    signal_processed[m] = synth_f
    prev_synth_patch_matrix = synth_patch_matrix

clean_signal = np.reshape(signal, -1)
synth_signal = np.reshape(signal_processed, -1)

perfect_reconstruction = np.allclose(clean_signal[patch_size:], synth_signal[patch_size:])

print(perfect_reconstruction)

plt.figure()
#plt.plot(clean_signal[patch_size:],'k-', label='clean')
#plt.plot(synth_signal[patch_size:],'r--', label='trunc')
plt.plot(synth_signal[patch_size:]- clean_signal[patch_size:])
plt.legend()













