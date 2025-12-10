# %%
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
import tomoconsistency_tools_hannah as tch
from utilities import utils_tomo
from scipy.signal import windows 
from scipy.ndimage import convolve
from scipy.ndimage import center_of_mass
import time
# %%
# file = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.nxs'
folder = '/mnt/share/ALC_ptychography_alignment/Experimental/connor_wright'
file = folder + '/scan_275019_275199_tomo_complex.nxs'
data_key = '/entry1/data'
angle_key = '/entry1/rotation_angle'

img_orig, theta, _ = utils_tomo.load_data(file, data_key= data_key, angle_key = angle_key)
img_orig = img_orig.transpose((1,2,0))

# %%
# with h5py.File(file, 'r') as f:
#     img_orig = np.angle(f['/stack_object'][:,:,:])

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

sinogram = tc.unwrap_data(img_orig_grad, 'fft_1d', boundary=None)
#%%
# sinogram = np.real(img_orig)
# sinogram = sinogram[:,0:722,:]
shift_total = np.zeros((sinogram.shape[-1],2))

#### tomoconsistency
iteration_no = 1

Nx = sinogram.shape[1]
Ny = sinogram.shape[0]
Nangles = sinogram.shape[2]

vol_geom, proj_geom = tch.init_astra(Nx, Ny, theta)

weights = np.ones(Nangles)

plot_figures = True
shift_history = []
for ii in range(iteration_no):
    t0 = time.time()
    # shift with imdeform_affine_fft
    sinogram_shifted = tc.imshift_fft(sinogram, shift_total) #(sinogram, shift_total)
    
    if plot_figures:
        plt.figure()
        plt.subplot(121),plt.imshow(sinogram[:,:,0]), plt.title('Sinogram')
        plt.subplot(122),plt.imshow(sinogram_shifted[:,:,0]), plt.title('Sinogram shifted')
    
    # fbp (ASTRA needs shape Ny * Nangle * Nx)
    sinogram_shifted = sinogram_shifted.transpose((0, 2, 1)) # for astra

    rec = tch.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights)

    rec_mask = tch.apply_circular_mask(rec, 0.9)
    
    if plot_figures:
        plt.figure()
        plt.subplot(131),plt.imshow(rec_mask[:,:,250]), plt.xlabel('x'), plt.ylabel('y')
        plt.subplot(132),plt.imshow(rec_mask[:,250,:]), plt.xlabel('x'), plt.ylabel('z')
        plt.subplot(133),plt.imshow(rec_mask[250,:,:]), plt.xlabel('y'), plt.ylabel('z')
        plt.tight_layout()
    
    # centering 
    # eps = np.finfo(rec.dtype).ep
    # w = np.sqrt(np.maximum(0,rec)) + eps
    # [x,y] = center_of_mass(w)
    # mass = w.sum()
    
    # get reprojection
    sinogram_model = tch.get_projections(rec_mask, vol_geom, proj_geom)

    sinogram_shifted = sinogram_shifted.transpose((0,2,1))
    sinogram_model = sinogram_model.transpose((0,2,1))

    if plot_figures:
        plt.figure()
        plt.subplot(131),plt.imshow(sinogram_shifted[:,:,0]), plt.title('Sinogram')
        plt.subplot(132),plt.imshow(sinogram_model[:,:,0]), plt.title('Sinogram model')
        plt.subplot(133),plt.imshow(sinogram_shifted[:,:,0]-sinogram_model[:,:,0]), plt.title('Difference')
        plt.tight_layout()

    MASS = np.median(sinogram_shifted * np.mean(abs(sinogram_shifted), axis=(0,1)))
    # MASS = 0.0557
    
    # sinogram_model is reprojected sinogram
    # sinogram is the original sino (also called "sinogram_shifted" in the MATLAB code)
    shift_upd, err = tc.find_optimal_shift(sinogram_model, sinogram_shifted, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
    step_relaxation = 0.01
    shift_upd = np.minimum(0.5, abs(shift_upd))#*np.sign(shift_upd)*step_relaxation
    
    shift_total = shift_total + shift_upd
    shift_history.append(shift_upd)
    # if plot_figures:
    #     plt.figure()
    #     plt.plot(shift_total[:,0], 'r', label='Total x shift')
    #     plt.plot(shift_total[:,1], 'b', label='Total y shift')
    #     plt.plot(shift_upd[:,0], '--r', label='Latest x shift')
    #     plt.plot(shift_upd[:,1], '--b', label='Latest y shift')
    #     plt.legend()

    print(f'Iteration {str(ii)} time {time.time()-t0}')

# if plot_figures is False: 
#     plt.figure()
#     plt.subplot(131),plt.imshow(rec_mask[:,:,250])
#     plt.subplot(132),plt.imshow(rec_mask[:,250,:])
#     plt.subplot(133),plt.imshow(rec_mask[250,:,:])
#     plt.tight_layout()
plt.show()
# %%
shift_history = np.array(shift_history)
plt.figure(figsize=(10,5))
for i in range(iteration_no):
    plt.plot(shift_history[i, :, 0], color='blue', alpha=0.3, label='x')
    plt.plot(shift_history[i, :, 1], color='red', alpha=0.3, label='y')
    plt.xlabel("Pixel")
    plt.ylabel("Shift value")
plt.show()

    
#%% Test find optimal shift
resid_sino = tc.get_resid_sino(sinogram_model, sinogram, high_pass_filter)
dX = tc.get_img_grad_filtered(sinogram_model, axis=0, high_pass_filter=high_pass_filter, smooth_win=5)
fig, axs = plt.subplots(1,3, figsize=(10,5))
axs[0].imshow(resid_sino[:,:,0])
axs[1].imshow()


# fig, ax1 = plt.subplots()

# ax1.plot(x, 'o')

# ax2 = ax1.twinx()
# ax2.plot(shift[:,0], 'x--')

#%%


x = np.random.randint(20, size=sinogram.shape[-1]) - 10

sinogram_model = np.empty(sinogram.shape)
for p in range(sinogram.shape[-1]):
    sinogram_model[:,:,p] = np.roll(sinogram[:,:,p],x[p],axis=1)

# file2 = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/NiTi_Zifan_395414/NiTiZifan_395414.mat'
# with h5py.File(file2, 'r') as f:
#     img = f['/stack_object'][:,:,0:10]
    

# img_orig_grad = tc.get_phase_gradient_1D(img,ax=1) #[300:-1250,400:-1300]
# sinogram = tc.unwrap_data(img_orig_grad, 'fft_1d', boundary=None)

# plt.figure()
# plt.plot(x)


# method = 'fft_1d'
# method = method.lower()

# def wrapToPi(x):
#     """
#     Wrap values to the range [-pi, pi].
#     """
#     return (x + np.pi) % (2 * np.pi) - np.pi

# def findresidues(phase):
#     """
#     Compute phase residues for 2D phase unwrapping.

#     Parameters:
#         phase (ndarray): Input phase (real or complex).

#     Returns:
#         ndarray: Residues array.
#     """
#     # If input is complex, take its phase angle
#     if not np.isrealobj(phase):
#         phase = np.angle(phase)

#     # Compute residues using wrapped differences
#     residues = wrapToPi(phase[1:, :-1, ...] - phase[:-1, :-1, ...])
#     residues += wrapToPi(phase[1:, 1:, ...] - phase[1:, :-1, ...])
#     residues += wrapToPi(phase[:-1, 1:, ...] - phase[1:, 1:, ...])
#     residues += wrapToPi(phase[:-1, :-1, ...] - phase[:-1, 1:, ...])

#     residues = residues / (2 * np.pi)
#     return residues

# def unwrap2D_fft(phase_diff, axis, boundary=None, step=0):
#     """
#     Perform 2D phase unwrapping using FFT-based integration.

#     Parameters:
#         phase_diff (ndarray): Input phase difference or complex array.
#         axis (int): Axis along which to unwrap.
#         empty_region (optional): Region to remove ramp (default: None).
#         step (int): Step size for gradient calculation (default: 0).

#     Returns:
#         tuple: (phase, phase_diff, residues)
#     """
#     residues = []

#     # Compute residues if requested and input is complex
#     if not np.isrealobj(phase_diff):
#         residues = np.abs(findresidues(phase_diff)) > 0.1

#     # If input is complex, compute phase gradient
#     if not np.isrealobj(phase_diff):
#         phase_diff = tc.get_phase_gradient_1D(phase_diff, ax=axis, step=step)

#     # Integrate to get phase
#     phase = np.real(tc.get_img_int_1D(phase_diff, axis))

#     # Remove ramp if empty_region provided and axis == 2
#     if boundary is not None and axis == 1:  # MATLAB axis=2 → Python axis=1
#         phase = tc.remove_sinogram_ramp(phase, boundary, -1)

#     return phase, phase_diff, residues

# sinogram = tc.get_phase_gradient_1D(img_orig,ax=1)

# if method == 'fft_1d':
#     # Unwrap the data by FFT along slices
#     img_unwrapped = -unwrap2D_fft(sinogram, axis=1, boundary=None)[0]
# elif method in ['none', 'diff']:
#     # Do nothing
#     pass
# else:
#     raise ValueError("Missing method")
    


# plt.figure()
# plt.subplot(1,2,1), plt.imshow(np.angle(img_orig[:,:,0]))
# plt.subplot(1,2,2), plt.imshow(img_unwrapped[:,:,0])