# %%
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 10:32:37 2025

@author: vdz11526
"""
import sys
import os
sys.path.append(os.path.abspath(".."))
import h5py
import numpy as np
import matplotlib.pyplot as plt
import tomoconsistency_tools_oriol as tc
import tomoconsistency_tools_hannah as tch
# from VerticalAlignmentSwiss import VerticalAlignmentSwiss as va
# from utilities import utils_tomo
from scipy.signal import windows 
from scipy.ndimage import convolve
from scipy.ndimage import center_of_mass
import time
#%%
file = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.nxs'
data_key = '/stack_object'

with h5py.File(file, 'r') as f:
    img_orig = np.angle(f[data_key][:,:,:])

theta = np.linspace(0,np.pi,img_orig.shape[-1])

#%%

img_orig_grad = tc.get_phase_gradient_1D(img_orig,ax=1) #[83:-83,71:-71,:]

width_sinogram = img_orig_grad.shape[1]
high_pass_filter = 0.01
unwrap_data_method = 'fft_1d'

# include the effect of high pass filter into the weights 
size = np.maximum(3, int(np.ceil(high_pass_filter * width_sinogram)))
gauss_window = windows.gaussian(size, std = size/6)
hanning_window = windows.hann(3)
ker = gauss_window.reshape(-1,1) * hanning_window

# relevance weights -> remove effect of potential residues / phase jumps 
ker2 = ker[np.newaxis,:,:]
convolution_result = convolve((np.abs(img_orig_grad) > 2).astype(np.float32), ker2.astype(np.float32), mode = 'constant', cval = 0.0)
weights_find_shift = np.maximum(0,1-convolution_result)
# weights = windows.tukey(sinogram.shape[1], alpha= 0.2)

shift_total = np.zeros((img_orig_grad.shape[-1],2))

#%%
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss
# vertical alignment
vertical = False
if vertical:
    img_orig_unwrapped = tc.unwrap_data(img_orig_grad, 'fft_1d', boundary=None)
    
    # va= VerticalAlignmentSwiss(img_orig_unwrapped)
    # psi_shifted, mean = va.shift_psi_by_array_and_mean(psi, np.array([2,0,1]))
    
    img_orig_grad, weights_find_shift, shift = tc.projections_align_vertical(img_orig_unwrapped,img_orig_grad,weights_find_shift,x0=0,xf=-1,y0=50,yf=550)
    plt.figure()
    plt.subplot(121),plt.imshow(np.sum(img_orig_grad[:,:,:],axis=1))
    plt.subplot(122),plt.plot(shift)

#%%

# binning
sinogram = tc.imshift_generic(img_orig_grad, shift_total, Npix = None, affine_matrix = None, smooth = 0, 
                                      ROI = None, downsample = 1, interp_method = 'linear', interp_sign = 0)

weights_find_shift = tc.imshift_generic(weights_find_shift, shift_total, Npix = None, affine_matrix = None, smooth = 0, 
                                      ROI = None, downsample = 1, interp_method = 'linear', interp_sign = 0)

sinogram = tc.unwrap_data(sinogram, 'fft_1d', boundary=None)

# % ASTRA needs the reconstruction to be dividable by 32 othewise there
#     % will be artefacts in left corner 
#     Npix = ceil(Npix/par.binning);
#     if isscalar(Npix)
#         Npix = [Npix, Npix, Nlayers];
#     elseif length(Npix) == 2
#         Npix = [Npix, Nlayers];
#     end


#%%
iteration_no = 2

Nx = sinogram.shape[1]
Ny = sinogram.shape[0]
Nangles = sinogram.shape[2]

vol_geom, proj_geom = tch.init_astra(Nx, Ny, theta)

weights = np.ones(Nangles)

#%%   
#### tomoconsistency
center_reconstruction = True
plot_figures = True
for ii in range(iteration_no):
    t0 = time.time()
    # shift with imdeform_affine_fft
    sinogram_shifted = tch.imshift_fft(sinogram, shift_total)

    if plot_figures:
        plt.figure()
        plt.subplot(121),plt.imshow(sinogram[:,:,0])
        plt.subplot(122),plt.imshow(sinogram_shifted[:,:,0])

    # fbp (ASTRA needs shape Ny * Nangle * Nx)
    sinogram_shifted = sinogram_shifted.transpose((0, 2, 1)) # for astra
    
    rec = tch.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights)
    
    rec_mask = tch.apply_circular_mask(rec, 0.9)
    
    if plot_figures: 
        plt.figure()
        plt.subplot(131),plt.imshow(rec[:,:,rec.shape[2]//2])
        plt.subplot(132),plt.imshow(rec[:,rec.shape[1]//2,:])
        plt.subplot(133),plt.imshow(rec[rec.shape[0]//2,:,:])
    
    # centering 
    rec_center = tc.centering_reconstruction(rec_mask)
    
    if ii == 0:
        if center_reconstruction:
            rec_center_0 = [0,0]
        else:
            rec_center_0 = rec_center
        
    shift_rec = -0.5*(rec_center - rec_center_0)
    
    rec_mask = tc.imshift_fft(rec_mask,shift_rec[0],shift_rec[1])
    
    # get reprojection
    sinogram_model = tch.get_projections(rec, vol_geom, proj_geom)
    
    sinogram_shifted = sinogram_shifted.transpose((0, 2, 1))
    sinogram_model = sinogram_model.transpose((0, 2, 1))
    
    if plot_figures:
        plt.figure()
        plt.subplot(121),plt.imshow(sinogram_shifted[:,:,0])
        plt.subplot(122),plt.imshow(sinogram_model[:,:,0])
  
    MASS = np.median(sinogram_shifted * np.mean(abs(sinogram_shifted), axis=(0,1)))
    
    # sinogram_model is reprojected sinogram
    # sinogram is the original sino (also called "sinogram_shifted" in the MATLAB code)
    shift_upd, err = tc.find_optimal_shift(sinogram_model, sinogram_shifted, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
    step_relaxation = 0.01
    shift_upd = np.minimum(0.5, abs(shift_upd))*np.sign(shift_upd)*step_relaxation
    
    shift_total = shift_total + shift_upd
    if plot_figures:
        plt.figure()
        plt.plot(shift_total[:,0], 'r', label='Total x shift')
        plt.plot(shift_total[:,1], 'b', label='Total y shift')
        plt.plot(shift_upd[:,0], '--r', label='Latest x shift')
        plt.plot(shift_upd[:,1], '--b', label='Latest y shift')
        plt.legend()

    print(f'Iteration {str(ii)} time {time.time()-t0}')

if plot_figures is False: 
    plt.figure()
    plt.subplot(131),plt.imshow(rec_mask[:,:,250])
    plt.subplot(132),plt.imshow(rec_mask[:,250,:])
    plt.subplot(133),plt.imshow(rec_mask[250,:,:])
    plt.tight_layout()
    
