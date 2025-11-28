#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:32:37 2025

@author: vdz11526
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tomoconsistency_tools_oriol as tc

file = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.nxs'

with h5py.File(file, 'r') as f:
    img_orig = f['/stack_object'][:,:,0:10]
    
method = 'fft_1d'
method = method.lower()

def wrapToPi(x):
    """
    Wrap values to the range [-pi, pi].
    """
    return (x + np.pi) % (2 * np.pi) - np.pi

def findresidues(phase):
    """
    Compute phase residues for 2D phase unwrapping.

    Parameters:
        phase (ndarray): Input phase (real or complex).

    Returns:
        ndarray: Residues array.
    """
    # If input is complex, take its phase angle
    if not np.isrealobj(phase):
        phase = np.angle(phase)

    # Compute residues using wrapped differences
    residues = wrapToPi(phase[1:, :-1, ...] - phase[:-1, :-1, ...])
    residues += wrapToPi(phase[1:, 1:, ...] - phase[1:, :-1, ...])
    residues += wrapToPi(phase[:-1, 1:, ...] - phase[1:, 1:, ...])
    residues += wrapToPi(phase[:-1, :-1, ...] - phase[:-1, 1:, ...])

    residues = residues / (2 * np.pi)
    return residues

def unwrap2D_fft(phase_diff, axis, boundary=None, step=0):
    """
    Perform 2D phase unwrapping using FFT-based integration.

    Parameters:
        phase_diff (ndarray): Input phase difference or complex array.
        axis (int): Axis along which to unwrap.
        empty_region (optional): Region to remove ramp (default: None).
        step (int): Step size for gradient calculation (default: 0).

    Returns:
        tuple: (phase, phase_diff, residues)
    """
    residues = []

    # Compute residues if requested and input is complex
    if not np.isrealobj(phase_diff):
        residues = np.abs(findresidues(phase_diff)) > 0.1

    # If input is complex, compute phase gradient
    if not np.isrealobj(phase_diff):
        phase_diff = tc.get_phase_gradient_1D(phase_diff, ax=axis, step=step)

    # Integrate to get phase
    phase = np.real(tc.get_img_int_1D(phase_diff, axis))

    # Remove ramp if empty_region provided and axis == 2
    if boundary is not None and axis == 1:  # MATLAB axis=2 → Python axis=1
        phase = tc.remove_sinogram_ramp(phase, boundary, -1)

    return phase, phase_diff, residues

sinogram = tc.get_phase_gradient_1D(img_orig,ax=1)

if method == 'fft_1d':
    # Unwrap the data by FFT along slices
    img_unwrapped = -unwrap2D_fft(sinogram, axis=1, boundary=None)[0]
elif method in ['none', 'diff']:
    # Do nothing
    pass
else:
    raise ValueError("Missing method")

#%%

plt.figure()
plt.subplot(1,2,1), plt.imshow(np.angle(img_orig[:,:,0]))
plt.subplot(1,2,2), plt.imshow(img_unwrapped[:,:,0])