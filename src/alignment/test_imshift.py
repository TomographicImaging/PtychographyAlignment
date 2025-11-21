#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:32:37 2025

@author: vdz11526
"""
import h5py
import numpy as np
import matplotlib.pyplot as plt

file = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.nxs'

apply_fft = True

method = 'circ'

# translation_test = np.zeros((10,2))
# translation_test[:,0] = 100 # x
# translation_test[:,1] = 200 # y

with h5py.File(file, 'r') as f:
    img_orig = f['/stack_object'][:,:,0:10]
    
# y = translation_test[:, 1]
# x = translation_test[:, 0]

ax = 0
shift = 100

from scipy.ndimage import map_coordinates

Nx, Ny, Nlayers = img_orig.shape # img_orig = img_orig.astype(np.float32)

if np.isscalar(shift):
    shift = np.full(Nlayers, shift)

img_out = np.copy(img_orig)

if method.lower() == 'circ':
    # Circular shift
    X = np.arange(Nx)
    Y = np.arange(Ny)
    for ii in range(Nlayers):
        if ax == 0:
            img_out[:, :, ii] = img_orig[np.roll(X, int(round(shift[ii]))), :, ii]
        if ax == 1:
            img_out[:, :, ii] = img_orig[:, np.roll(Y, int(round(shift[ii]))), ii]
        if ax != 0 and ax != 1:
            print(ax)
            raise Exception('"ax" is neither 0 nor 1')
    # for i in range(Npix[fixed_axis]):  
    #     slicer = [slice(None)] * img_orig.ndim
    #     slicer[fixed_axis] = i
    #     img_out[tuple(slicer)] = np.roll(img_orig[tuple(slicer)], int(round(shift[i])), axis=0)
else:
    # Interpolation-based shift
    order = {'nearest': 0, 'linear': 1, 'cubic': 3}.get(method.lower(), 1)
    for ii in range(Nlayers):
        coords_y, coords_x = np.meshgrid(np.arange(Ny), np.arange(Nx))
        if int(ax) == int(0):
            coords = np.array([coords_x - shift[ii], coords_y])
        if int(ax) == int(1):
            coords = np.array([coords_x, coords_y - shift[ii]])    
        if ax != 0 and ax != 1:
            print(ax)
            raise Exception('"ax" is neither 0 nor 1')
            
        img_out[:, :, ii] = map_coordinates(img_orig[:, :, ii], coords, order=order, mode='constant', cval=0.0)
        
    # for i in range(Npix[fixed_axis]):
    #     slicer = [slice(None)] * img_orig.ndim
    #     slicer[fixed_axis] = i
    #     coords = np.arange(img_orig.shape[0]) - shift[i]
    #     img_out[tuple(slicer)] = np.interp(coords, np.arange(img_orig.shape[0]), img_orig[tuple(slicer)], left=np.nan, right=np.nan)

# Move axis back to original position
# img_out = np.moveaxis(img_out, 0, ax)
    
plt.figure()
plt.subplot(1,2,1),plt.imshow(np.imag(img_orig[:,:,0]))
plt.subplot(1,2,2), plt.imshow(np.imag(img_out[:,:,0]))