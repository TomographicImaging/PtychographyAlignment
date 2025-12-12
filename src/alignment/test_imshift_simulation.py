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
# %%
folder = '/mnt/share/ALC_ptychography_alignment/simulations/'
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
weights = np.ones(Nangles)
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
dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
weights = np.full(len(theta), dtheta, dtype=np.float32)

data = data.transpose((0, 2, 1)) # for astra
vol_geom, proj_geom = tch.init_astra_vec(Nx, Ny, theta_rad, shift_simulated)
rec = tch.FBP_astra(data, vol_geom, proj_geom, weights)
data = data.transpose((0,2,1)) # transpose back

plt.figure()
plt.subplot(131),plt.imshow(rec[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
plt.subplot(132),plt.imshow(rec[:,Nx//2,:]), plt.xlabel('x'), plt.ylabel('z')
plt.subplot(133),plt.imshow(rec[Nx//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
plt.tight_layout()

# %%
# sinogram = np.real(img_orig)
# sinogram = sinogram[:,0:722,:]

vol_geom, proj_geom = tch.init_astra(Nx, Ny, np.deg2rad(theta))
sinogram = data.copy()
weights_find_shift = np.ones_like(sinogram)
high_pass_filter = 0.01
unwrap_data_method = 'fft_1d'
shift_total = np.zeros((sinogram.shape[-1],2))
#### tomoconsistency
iteration_no = 5
# weights = np.ones(Nangles)
dtheta = (theta_rad[-1] - theta_rad[0]) / (len(theta_rad) - 1) if len(theta_rad) > 1 else 1.0
weights = np.full(len(theta_rad), dtheta, dtype=np.float32)
plot_figures = True
shift_history = []

for ii in range(iteration_no):
    t0 = time.time()
    # shift with imdeform_affine_fft
    # sinogram_shifted = tc.imshift_fft(sinogram, shift_total) #(sinogram, shift_total)
    
    if plot_figures:
        plt.figure(figsize=(10,3))
        plt.subplot(121),plt.imshow(sinogram[:,:,0]), plt.title('Sinogram'), plt.colorbar()
        plt.subplot(122),plt.imshow(sinogram_shifted[:,:,0]), plt.title('Sinogram shifted'), plt.colorbar()
    
    # fbp (ASTRA needs shape Ny * Nangle * Nx)
    sinogram_shifted = sinogram_shifted.transpose((0, 2, 1)) # for astra
    vol_geom, proj_geom = tch.init_astra_vec(Nx, Ny, theta_rad, shift_total) # try applying shifts with astra vector geometry
    rec = tch.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights)

    rec = tch.apply_circular_mask(rec, 0.9)
    
    if plot_figures:
        plt.figure(figsize=(10,3))
        plt.suptitle('Shifted reconstruction')
        plt.subplot(131),plt.imshow(rec[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
        plt.subplot(132),plt.imshow(rec[:,Nx//2,:]), plt.xlabel('x'), plt.ylabel('z')
        plt.subplot(133),plt.imshow(rec[Nx//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
        plt.tight_layout()
    
    # centering 
    # eps = np.finfo(rec.dtype).ep
    # w = np.sqrt(np.maximum(0,rec)) + eps
    # [x,y] = center_of_mass(w)
    # mass = w.sum()
    
    # get reprojection
    sinogram_model = tch.get_projections(rec, vol_geom, proj_geom)

    sinogram_shifted = sinogram_shifted.transpose((0,2,1))
    sinogram_model = sinogram_model.transpose((0,2,1))

    if plot_figures:
        plt.figure(figsize=(10,3))
        plt.subplot(131),plt.imshow(sinogram_shifted[:,:,0]), plt.title('Sinogram'), plt.colorbar()
        plt.subplot(132),plt.imshow(sinogram_model[:,:,0]), plt.title('Sinogram model'), plt.colorbar()
        plt.subplot(133),plt.imshow(sinogram_shifted[:,:,0]-sinogram_model[:,:,0]), plt.title('Difference'), plt.colorbar()
        plt.tight_layout()

    MASS = np.median(sinogram_shifted * np.mean(abs(sinogram_shifted), axis=(0,1)))
    # MASS = 0.0557
    
    # sinogram_model is reprojected sinogram
    # sinogram is the original sino (also called "sinogram_shifted" in the MATLAB code)
    shift_upd, err = tc.find_optimal_shift(sinogram_model, sinogram_shifted, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
    step_relaxation = 0.01
    # shift_upd = np.minimum(0.5, abs(shift_upd))#*np.sign(shift_upd)*step_relaxation
    
    # shift_total = shift_total + shift_upd
    shift_history.append(shift_upd)
    # if plot_figures:
    #     plt.figure()
    #     plt.plot(shift_total[:,0], 'r', label='Total x shift')
    #     plt.plot(shift_total[:,1], 'b', label='Total y shift')
    #     plt.plot(shift_upd[:,0], '--r', label='Latest x shift')
    #     plt.plot(shift_upd[:,1], '--b', label='Latest y shift')
    #     plt.legend()

    print(f'Iteration {str(ii)} time {time.time()-t0}')
# %%
shift_history = np.array(shift_history)
plt.figure(figsize=(10,5))
for i in range(iteration_no):
    plt.plot(shift_history[i, :, 0], color='blue', alpha=0.3, label='x')
    plt.plot(shift_history[i, :, 1], color='red', alpha=0.3, label='y')
    plt.xlabel("Pixel")
    plt.ylabel("Shift value")
plt.show()
