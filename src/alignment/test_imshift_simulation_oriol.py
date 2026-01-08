# %%
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
from alignment.Alignment import VerticalAlignmentCrossCorrelation, HorizontalAlignmentCrossCorrelation, COMAlignment
# %%
folder = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/simulated_data/sphere_phantom_360_theta/'
file = folder + 'sphere_phantom_360_projections.npy'
data = np.load(file) # = [Nangles, Ny, Nx]

data = data.transpose((1,2,0)) # [Ny, Nx, Nangles]

Nx = data.shape[1]
Ny = data.shape[0]
Nangles = data.shape[2]


# %%
plt.figure(figsize=(10,5))
plt.subplot(131),plt.imshow(data[Nx//2,:,:]), plt.xlabel('Angles'), plt.ylabel('x')
plt.subplot(132),plt.imshow(data[:,Nx//2,:]), plt.xlabel('Angles'), plt.ylabel('y')
plt.subplot(133),plt.imshow(data[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
plt.tight_layout()

# %% Get the real shifts
delta_x = np.load(folder+'sphere_phantom_360_shifts_x.npy')
delta_y = np.load(folder+'sphere_phantom_360_shifts_y.npy')
theta = np.load(folder + 'sphere_phantom_360_theta.npy')
theta_rad = np.deg2rad(theta)

dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
weights = np.full(len(theta), dtheta, dtype=np.float32)

plt.figure()
plt.plot(theta, delta_x, label='x shifts')
plt.plot(theta, delta_y, label='y shifts')
plt.xlabel('Angles')
plt.ylabel('Pixels')
plt.legend()

# %% First check shifts can be applied correctly
shift_simulated = np.column_stack((delta_x, delta_y))
sinogram_shifted = tc.imshift_fft(data, shift_simulated) #(sinogram, shift_total)
plt.figure(figsize=(10,5))
plt.subplot(131),plt.imshow(sinogram_shifted[Nx//2,:,:]), plt.xlabel('Angles'), plt.ylabel('x')
plt.subplot(132),plt.imshow(sinogram_shifted[:,Nx//2,:]), plt.xlabel('Angles'), plt.ylabel('y')
plt.subplot(133),plt.imshow(sinogram_shifted[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
plt.tight_layout()

# %% and reconstructed
vol_geom, proj_geom = tch.init_astra(Nx, Ny, np.deg2rad(theta))
sinogram_shifted = sinogram_shifted.transpose((0, 2, 1)) # transpose for astra
rec = tch.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights)
sinogram_shifted = sinogram_shifted.transpose((0,2,1)) # transpose back

plt.figure()
plt.subplot(131),plt.imshow(rec[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
plt.subplot(132),plt.imshow(rec[:,Nx//2,:]), plt.xlabel('x'), plt.ylabel('z')
plt.subplot(133),plt.imshow(rec[Nx//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
plt.tight_layout()

# %% also try reconstructing the UNSHIFTED data using astra vector geometry with shifts

data = data.transpose((0, 2, 1)) # for astra
vol_geom, proj_geom = tch.init_astra_vec(Nx, Ny, theta_rad, shift_simulated)
rec = tch.FBP_astra(data, vol_geom, proj_geom, weights)
data = data.transpose((0,2,1)) # transpose back

plt.figure()
plt.subplot(131),plt.imshow(rec[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
plt.subplot(132),plt.imshow(rec[:,Nx//2,:]), plt.xlabel('x'), plt.ylabel('z')
plt.subplot(133),plt.imshow(rec[Nx//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
plt.tight_layout()

# %% and just check what the unshifted data looks like when reconstructed
sinogram = data.copy()
vol_geom, proj_geom = tch.init_astra(Nx, Ny, np.deg2rad(theta))
sinogram = sinogram.transpose((0, 2, 1)) # transpose for astra
rec = tch.FBP_astra(sinogram, vol_geom, proj_geom, weights)
sinogram = sinogram.transpose((0,2,1)) # transpose back


plt.figure()
plt.subplot(131),plt.imshow(rec[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
plt.subplot(132),plt.imshow(rec[:,Nx//2,:]), plt.xlabel('x'), plt.ylabel('z')
plt.subplot(133),plt.imshow(rec[Nx//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
plt.tight_layout()
# %% use cross-correlation to align the projections before tomo consistency
from skimage.registration import phase_cross_correlation as register_translation
from scipy import optimize, signal
projections = data.transpose((2,1,0))
quick_correlation = np.zeros((2,projections.shape[0]))
                            
for gag in range(1,projections.shape[0]):
    a = (projections[gag-1,:,:])
    b = (projections[gag,:,:])
    
    shift_ab = register_translation(a,b,upsample_factor=100)
    quick_correlation[1,gag] = shift_ab[0][1] + quick_correlation[1,gag-1]
    # quick_correlation[0,gag] = shift_ab[0][0] + quick_correlation[0,gag-1]

proj_sum = np.sum(projections,2)
xcor_ar = np.zeros_like(proj_sum)
a = np.gradient(proj_sum[0,:])
for i in range(projections.shape[0]):
    b = np.gradient(proj_sum[i,:])
    xcor = signal.correlate(a, b, mode='same')
    quick_correlation[0,i] = np.argmax(xcor)-(xcor.shape[0]/2)
    xcor_ar[i,:] = xcor

plt.figure(figsize=[10,3])
plt.subplot(121),plt.plot(theta, quick_correlation[0,:])
plt.subplot(121),plt.plot(theta, delta_x, '--')
plt.subplot(122),plt.plot(theta, quick_correlation[1,:])
plt.subplot(122),plt.plot(theta, delta_y, '--')


# %% Check physically shifting the correlated values

projections_out = np.zeros_like(projections)
quick_correlation = np.round(quick_correlation).astype(np.int32)

for i in range(projections.shape[0]):
#         print("Applying shift [%d,%d] to projection %d" %(shifts[0,i], shifts[1,i], i))
    projections_out[i] = np.roll(projections[i], 1*quick_correlation[0,i], axis=0)
    projections_out[i] = np.roll(projections[i], 1*quick_correlation[1,i], axis=1)

plt.figure(figsize=(10,3))
plt.subplot(121), plt.imshow(projections[:,Nx//2,:])
plt.subplot(122), plt.imshow(projections_out[:,Nx//2,:])

# %%
# sinogram = np.real(img_orig)
# sinogram = sinogram[:,0:722,:]

sinogram = data.copy()
weights_find_shift = np.ones_like(sinogram)
high_pass_filter = 0.001
unwrap_data_method = 'fft_1d'
shift_method = 'geometry' # physical or geometry

# apply vertical shifts
shift_simulated = np.column_stack((np.zeros(delta_x.shape), delta_y))
sinogram = tc.imshift_fft(sinogram, shift_simulated) #(sinogram, shift_total)

# shift_total = quick_correlation.T


# binning
binning = [4, 2, 1] 
max_iteration = 100
dtheta = (theta_rad[-1] - theta_rad[0]) / (len(theta_rad) - 1) if len(theta_rad) > 1 else 1.0
weights = np.full(len(theta_rad), dtheta, dtype=np.float32)
plot_figures = True
min_step_size = 0.05 #max_update * par.binning < par.min_step_size

Npix = []

optimal_shift = np.zeros((sinogram.shape[-1],2))

for b in range(len(binning)):
    
    optimal_shift, shift_history = tc.align_tomo_consistency_linear(sinogram, weights_find_shift, weights, theta, theta_rad, 
                                                                    Npix, optimal_shift, binning[b], high_pass_filter = high_pass_filter, unwrap_data_method = 'fft_1d')
    
    shift_history = np.array(shift_history)
    plt.figure(figsize=(10,5))
    
    # for i in range(ii):
    #     plt.plot(theta, shift_history[i, :, 0], color='blue', alpha=0.3, label='x')
    #     # plt.plot(theta, shift_history[i, :, 1], color='red', alpha=0.3, label='y')
    #     plt.xlabel("Angle")
    #     plt.ylabel("Shift value")
    # plt.plot()

    plt.plot(theta, delta_x)
    plt.plot(theta, optimal_shift[:,0])
    
    plt.ylabel('Horizontal shift value')
    plt.xlabel('Angle (deg)')
    plt.legend(['Simulated shifts', 'Calculated shifts'])
    plt.grid()



