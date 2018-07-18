# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from scipy.io.wavfile import write
from scipy.linalg import svd
from scipy.signal import spectrogram
from pywt import threshold

import functions as fn

block_size = 1024
patch_size = 256
#block_num = 60
thr_val = 100000
#rate = 16000

#t = np.linspace(0, block_size*block_num/float(rate), num=block_size*block_num, endpoint=False)
#
#f1 = 400.0
#f2 = 7950.0
#
#signal = np.sin(t*2*np.pi*f1) + 0.2*np.sin(t*2*np.pi*f2)
#signal = np.reshape(signal, (block_num, block_size))

signal, rate = fn.wav_block_import('Example.wav', block_size)
block_num = np.size(signal, axis=0)

prev_synth_patch_matrix = np.zeros((block_size,patch_size), dtype = signal.dtype)
signal_processed = np.zeros(np.shape(signal), dtype = signal.dtype)
signal_processed_circ = np.zeros(np.shape(signal), dtype = signal.dtype)
for m in range(block_num): # Loop over blocks in signal
    # Analysis
    #   Patch Matrix
    f = signal[m]
    f2 = np.zeros(patch_size-1)
    
    if m < block_num - 1:
        f2 = signal[m+1,:patch_size-1]
        
    patch_matrix = fn.patch_matrix_build_trcon(f, f2)
    patch_matrix_circ = fn.patch_matrix_build_circ(f, patch_size)
    
    #   Coefficient matrix
    phi = svd(patch_matrix, False)[0]
    coeff_matrix = np.dot(np.conj(phi.T), patch_matrix)
    coeff_matrix = np.fft.fft(coeff_matrix, axis=1)
    
    phi_circ = svd(patch_matrix_circ, False)[0]
    coeff_matrix_circ = np.dot(np.conj(phi_circ.T), patch_matrix_circ)
    coeff_matrix_circ = np.fft.fft(coeff_matrix_circ, axis=1)
    
    # Processing
    processed_coeff_matrix = threshold(coeff_matrix, thr_val, mode='hard')
    processed_coeff_matrix_circ = threshold(coeff_matrix_circ, thr_val, mode='hard')
    
    # Synthesis
    synth_patch_matrix = np.fft.ifft(processed_coeff_matrix, axis=1)
    synth_patch_matrix = np.dot(phi, synth_patch_matrix)
    
    synth_patch_matrix_circ = np.fft.ifft(processed_coeff_matrix_circ, axis=1)
    synth_patch_matrix_circ = np.dot(phi_circ, synth_patch_matrix_circ)
    
    #   Patch Matrix averaging
    stitched_patch_matrix = fn.patch_matrix_stitch_trcon(prev_synth_patch_matrix, synth_patch_matrix)
    synth_f = fn.patch_matrix_average(stitched_patch_matrix)
    synth_f_circ = fn.patch_matrix_average(synth_patch_matrix_circ)
    signal_processed[m] = synth_f
    signal_processed_circ[m] = synth_f_circ
    prev_synth_patch_matrix = synth_patch_matrix

clean_signal = np.reshape(signal, -1)
synth_signal = np.reshape(signal_processed, -1)
synth_signal_circ = np.reshape(signal_processed_circ, -1)

fre, tim, spec_clean = spectrogram(clean_signal, rate, window='hann', mode='magnitude')
fre, tim, spec_btc = spectrogram(synth_signal, rate, window='hann', mode='magnitude')
fre, tim, spec_circ = spectrogram(synth_signal_circ, rate, window='hann', mode='magnitude')
F,T = np.meshgrid(fre,tim)

#%%
spec_bound = 20000

plt.figure()
plt.subplot(2,2,1)
plt.title('(A) Original - magnitude spectrum')
plt.pcolormesh(T.T,F.T,spec_clean, vmax=spec_bound)
plt.xlim(0,2.6)
plt.ylabel('frequency [Hz]')

plt.subplot(2,2,2)
plt.title('(B) Circular - magnitude spectrum')
plt.pcolormesh(T.T,F.T,spec_circ, vmax=spec_bound)
plt.xlim(0,2.6)

plt.subplot(2,2,3)
plt.title('(C) Boundary corrected - magnitude spectrum')
plt.pcolormesh(T.T,F.T,spec_btc, vmax=spec_bound)
plt.xlim(0,2.6)
plt.xlabel('time [s]')

plt.subplot(2,2,4)
plt.title('(D) Difference - magnitude spectrum')
plt.pcolormesh(T.T,F.T,np.abs(spec_circ-spec_btc), vmax=spec_bound)
plt.xlim(0,2.6)

#clean_signal_wav = (clean_signal/np.max(np.abs(clean_signal)))*32767
#synth_signal_wav = synth_signal/np.max(np.abs(synth_signal))*32767
#synth_signal_circ_wav = synth_signal_circ/np.max(np.abs(synth_signal_circ))*32767

clean_signal_wav = clean_signal
synth_signal_wav = synth_signal
synth_signal_circ_wav = synth_signal_circ
difference_signal_wav = synth_signal_circ-synth_signal#clean_signal

write('clean.wav',rate, clean_signal_wav.astype(np.int16))
write('synth.wav',rate, synth_signal_wav.astype(np.int16))
write('circ.wav',rate, synth_signal_circ_wav.astype(np.int16))
write('difference.wav',rate, difference_signal_wav.astype(np.int16))

#plt.figure()
#plt.plot(synth_signal_circ-synth_signal,'k-', label='clean')
#plt.plot(synth_signal,'r--', label='trunc')
#plt.plot(synth_signal_circ,'b--', label='circ')
#plt.legend()



















