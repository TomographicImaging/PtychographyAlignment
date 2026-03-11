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
folder = '/mnt/share/ALC_ptychography_alignment/simulations/'
file = folder + 'sphere_phantom_360_projections.npy'
data = np.load(file) # = [Nangles, Ny, Nx]

data = data.transpose((1,2,0)) # [Ny, Nx, Nangles]

Ny = data.shape[0]
Nx = data.shape[1]
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

dtheta = (theta_rad[-1] - theta_rad[0]) / (len(theta_rad) - 1) if len(theta_rad) > 1 else 1.0
weights = np.full(len(theta_rad), dtheta, dtype=np.float32)

# %%
### options
binning = 1 # max 4
center_reconstruction = True
ii = 0
high_pass_filter = 0.01
unwrap_data_method = 'none'
### 

sinogram = data.copy()
sinogram = sinogram.reshape(Ny//binning, binning, Nx//binning, binning, Nangles).mean(axis=(1, 3))
vol_geom, proj_geom = tch.init_astra(Nx, Ny, np.deg2rad(theta), binning)
shift_total = np.zeros((sinogram.shape[-1],2))


# %%
sinogram_astra = sinogram.transpose((0, 2, 1))
sinogram_shifted = tc.imshift_fft(sinogram, shift_total) #(sinogram, shift_total)
sinogram_astra = sinogram_shifted.transpose((0, 2, 1))
plt.figure(figsize=(10,3))
plt.subplot(121),plt.imshow(sinogram[:,sinogram.shape[1]//2,:]), plt.title('Sinogram'), plt.colorbar()
plt.subplot(122),plt.imshow(sinogram_shifted[:,sinogram_shifted.shape[1]//2,:]), plt.title('Sinogram shifted'), plt.colorbar()

rec = tch.FBP_astra(sinogram_astra, vol_geom, proj_geom, weights)

# rec = tch.apply_circular_mask(rec, 0.9)
plt.figure(figsize=(10,3))
plt.suptitle('Shifted reconstruction')
plt.subplot(131),plt.imshow(rec[:,:,rec.shape[2]//2]), plt.xlabel('z'), plt.ylabel('x')
plt.subplot(132),plt.imshow(rec[:,rec.shape[1]//2,:]), plt.xlabel('y'), plt.ylabel('x')
plt.subplot(133),plt.imshow(rec[rec.shape[0]//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
plt.tight_layout()

if center_reconstruction:
    # centering 
    rec_center = tc.centering_reconstruction(rec)
    
    if ii == 0:
        if center_reconstruction:
            rec_center_0 = [rec.shape[0]/2,rec.shape[2]/2]
        else:
            rec_center_0 = rec_center
        
    shift_rec = -0.5*(rec_center - rec_center_0)
    
    rec = tc.imshift_fft(rec, shift_rec[1], shift_rec[0])

sinogram_model_astra = tch.get_projections(rec, vol_geom, proj_geom)

sinogram_model = sinogram_model_astra.transpose((0,2,1))
# %%
shift_x = np.zeros(sinogram_model.shape[2], dtype=np.float32)
shift_y = np.zeros(sinogram_model.shape[2], dtype=np.float32)

resid_sino = tc.get_resid_sino(sinogram_model, sinogram_shifted, high_pass_filter)

if unwrap_data_method.lower() == 'none':
    resid_sino = tc.imfilter_high_pass_1d(resid_sino, ax=2, sigma=high_pass_filter, padding=0)
       
plt.figure(figsize=(10,3))
plt.subplot(131),plt.imshow(sinogram_model[:,sinogram_model.shape[1]//2,:]), plt.title('Sinogram model'), plt.colorbar()
plt.subplot(132),plt.imshow(sinogram_shifted[:,sinogram_shifted.shape[1]//2,:]), plt.title('Sinogram shifted'), plt.colorbar()
plt.subplot(133),plt.imshow(resid_sino[:,resid_sino.shape[1]//2,:]), plt.title('Resid sino'), plt.colorbar()

plt.figure(figsize=(10,3))
plt.subplot(131),plt.imshow(resid_sino[resid_sino.shape[0]//2,:,:]), plt.title('Resid sino'), plt.colorbar()
plt.subplot(132),plt.imshow(resid_sino[:,resid_sino.shape[1]//2,:]), plt.title('Resid sino'), plt.colorbar()
plt.subplot(133),plt.imshow(resid_sino[:,:,resid_sino.shape[2]//2]), plt.title('Resid sino'), plt.colorbar()


# %%



# Horizontal alignment 
# dX = tc.get_img_grad_filtered(sinogram_model, axis=0, high_pass_filter=high_pass_filter, smooth_win=5)
img = sinogram_model
smooth_win = 5
axis = 0
img = tc.smooth_edges(img, smooth_win, [1 + (axis % 2)])
plt.figure(figsize=(10,3))
plt.subplot(131),plt.imshow(img[img.shape[0]//2,:,:]), plt.title('img'), plt.colorbar()
plt.subplot(132),plt.imshow(img[:,img.shape[1]//2,:]), plt.title('img'), plt.colorbar()
plt.subplot(133),plt.imshow(img[:,:,img.shape[2]//2]), plt.title('img'), plt.colorbar()

is_real = np.isrealobj(img)
Np = img.shape

if axis == 0:  # Horizontal direction
    X = 2j * np.pi * (np.fft.fftshift(np.arange(Np[1]) / Np[1]) - 0.5)
    d_img = np.fft.fft(img, axis=1)
    plt.figure(figsize=(10,3))
    plt.subplot(131),plt.imshow(d_img.real[d_img.shape[0]//2,:,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(132),plt.imshow(d_img.real[:,d_img.shape[1]//2,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(133),plt.imshow(d_img.real[:,:,d_img.shape[2]//2]), plt.title('d_img'), plt.colorbar()
    d_img = d_img * X[np.newaxis,:,np.newaxis]  # Broadcasting works automatically
    plt.figure(figsize=(10,3))
    plt.subplot(131),plt.imshow(d_img.real[d_img.shape[0]//2,:,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(132),plt.imshow(d_img.real[:,d_img.shape[1]//2,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(133),plt.imshow(d_img.real[:,:,d_img.shape[2]//2]), plt.title('d_img'), plt.colorbar()
    # Apply high-pass filter along horizontal direction
    d_img = tc.imfilter_high_pass_1d(d_img, ax=1, sigma=high_pass_filter, padding=0, apply_fft=False)
    plt.figure(figsize=(10,3))
    plt.subplot(131),plt.imshow(d_img.real[d_img.shape[0]//2,:,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(132),plt.imshow(d_img.real[:,d_img.shape[1]//2,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(133),plt.imshow(d_img.real[:,:,d_img.shape[2]//2]), plt.title('d_img'), plt.colorbar()
    d_img = np.fft.ifft(d_img, axis=1)
    plt.figure(figsize=(10,3))
    plt.subplot(131),plt.imshow(d_img.real[d_img.shape[0]//2,:,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(132),plt.imshow(d_img.real[:,d_img.shape[1]//2,:]), plt.title('d_img'), plt.colorbar()
    plt.subplot(133),plt.imshow(d_img.real[:,:,d_img.shape[2]//2]), plt.title('d_img'), plt.colorbar()
# %%
if is_real:
    d_img = np.real(d_img)
dX = d_img
if unwrap_data_method.lower() == 'none':
    dX = tc.imfilter_high_pass_1d(dX, ax=2, sigma=high_pass_filter, padding=0)
plt.figure(figsize=(10,3))
plt.subplot(131),plt.imshow(dX[dX.shape[0]//2,:,:]), plt.title('dX'), plt.colorbar()
plt.subplot(132),plt.imshow(dX[:,dX.shape[1]//2,:]), plt.title('dX'), plt.colorbar()
plt.subplot(133),plt.imshow(dX[:,:,dX.shape[2]//2]), plt.title('dX'), plt.colorbar()

# %%
A = weights * dX * resid_sino
plt.figure(figsize=(10,3))
plt.subplot(131),plt.imshow(A[A.shape[0]//2,:,:]), plt.title('A'), plt.colorbar()
plt.subplot(132),plt.imshow(A[:,A.shape[1]//2,:]), plt.title('A'), plt.colorbar()
plt.subplot(133),plt.imshow(A[:,:,A.shape[2]//2]), plt.title('A'), plt.colorbar()
# %%
numerator = np.sum(weights * dX * resid_sino, axis=(0, 1))
plt.plot(numerator)

# %%
# if np.mean(numerator) < 0.01:
#     numerator[:] = 0
denominator = np.sum(weights * dX**2, axis=(0, 1)) # sum2 and mean 2????????????????
plt.plot(denominator, '--')

# denominator = np.sum(weights * dX**2, axis=(0, 2)) # sum2 and mean 2????????????????
# plt.plot(denominator, '--')

# denominator = np.sum(weights * dX**2, axis=(1, 2)) # sum2 and mean 2????????????????
# plt.plot(denominator, '--')

# %%
shift_x = -numerator / denominator
plt.plot(shift_x)

# # Vertical alignment
# if align_vertical:
#     dY = get_img_grad_filtered(sinogram_model, axis=1, high_pass_filter=high_pass_filter, smooth_win=5)
#     if unwrap_data_method.lower() == 'none':
#         dY = imfilter_high_pass_1d(dY, axis=0, high_pass_filter=high_pass_filter, pad=0)

#     numerator = np.sum(weights * dY * resid_sino, axis=(0, 1))
#     # if np.mean(numerator) < 0.01:
#     #     numerator[:] = 0
#     denominator = np.sum(weights * dY**2, axis=(0, 1))
#     shift_y = -numerator / denominator

# Combine shifts
shift = np.stack([shift_x, shift_y], axis=-1)
# %%
