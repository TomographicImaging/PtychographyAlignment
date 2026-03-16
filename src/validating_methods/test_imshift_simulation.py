# %%
import h5py
import numpy as np
import astra
import matplotlib.pyplot as plt
from archive import tomoconsistency_tools_oriol as tc
from archive import tomoconsistency_tools_hannah as tch
from utilities import utils_tomo
from scipy.signal import windows 
from scipy.ndimage import convolve
from scipy.ndimage import center_of_mass
import time
from alignment.Alignment import VerticalAlignmentCrossCorrelation, HorizontalAlignmentCrossCorrelation, COMAlignment
# %%
folder = '/mnt/share/ALC_ptychography_alignment/simulations/'
file = folder + 'sphere_phantom_360_projections_noy.npy'
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
delta_x = np.load(folder+'sphere_phantom_360_shifts_x_noy.npy')
delta_y = np.load(folder+'sphere_phantom_360_shifts_y_noy.npy')
theta = np.load(folder + 'sphere_phantom_360_theta_noy.npy')
theta_rad = np.deg2rad(theta)

dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
weights = np.full(len(theta), dtheta, dtype=np.float32)

plt.figure()
plt.plot(theta, delta_x, label='x shifts')
plt.plot(theta, delta_y, label='y shifts')
plt.xlabel('Angles')
plt.ylabel('Pixels')
plt.legend()

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


# %%
sinogram = data.copy()
sinogram = sinogram[10:290,:,:] # make it asymmetrical for diagnostic
Ny = sinogram.shape[0]
Nx = sinogram.shape[1]
Nangles = sinogram.shape[2]
binning = 4 # max 4
sinogram = sinogram.reshape(Ny//binning, binning, Nx//binning, binning, Nangles).mean(axis=(1, 3))

vol_geom, proj_geom = tch.init_astra(Nx//binning, Ny//binning, np.deg2rad(theta))

weights_find_shift = np.ones_like(sinogram)
high_pass_filter = 0.01
unwrap_data_method = 'none'
shift_method = 'physical' # physical or geometry
step_relaxation = 0.5

# shift_total = quick_correlation.T
shift_total = np.zeros((sinogram.shape[-1],2))

# %%   
iteration_no = 100
step_relaxation = 0.5
high_pass_filter = 0.01
min_step_size = 0.01
unwrap_data_method = 'none'
shift_method = 'physical' # physical or geometry

plot_figures = False # plot every iteration
center_reconstruction = True
limit_steps = True

bin_levels = [4, 2] 



optimal_shift = np.zeros((data.shape[-1],2))

dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
weights = np.full(len(theta), dtheta, dtype=np.float32)



for binning in bin_levels:

    sinogram = tc.imshift_generic(data, optimal_shift, Npix = None, affine_matrix = None, smooth = 5, 
                                        ROI = None, downsample = binning, interp_method = 'fft', interp_sign = 1)

    weights_find_shift = np.ones_like(data)
    weights_find_shift = tc.imshift_generic(weights_find_shift, optimal_shift, Npix = None, affine_matrix = None, smooth = 0, 
                                        ROI = None, downsample = binning, interp_method = 'linear', interp_sign = 1)

    optimal_shift, err, rec, sinogram_shifted = align_tomo_consistency_linear(sinogram, weights_find_shift, weights, theta, Nx, optimal_shift, binning,
                                  high_pass_filter, unwrap_data_method)



# %%
#### tomoconsistency
iteration_no = 1

dtheta = (theta_rad[-1] - theta_rad[0]) / (len(theta_rad) - 1) if len(theta_rad) > 1 else 1.0
weights = np.full(len(theta_rad), dtheta, dtype=np.float32)

plot_figures = True
shift_history = []
shift_history.append(shift_total)
center_reconstruction = True

sinogram_astra = sinogram.transpose((0, 2, 1))

for ii in range(iteration_no):
    t0 = time.time()

    if shift_method == 'physical':
        # shift with imdeform_affine_fft
        sinogram_shifted = sinogram
        sinogram_shifted = tc.imshift_fft(sinogram_shifted, shift_total) #(sinogram, shift_total)
        
        if unwrap_data_method is not 'none':
            sinogram_shifted = tc.unwrap_data(sinogram_shifted, 'fft_1d', boundary=None)
        sinogram_astra = sinogram_shifted.transpose((0, 2, 1))

        if plot_figures:
            plt.figure(figsize=(10,3))
            plt.subplot(121),plt.imshow(sinogram[:,sinogram.shape[1]//2,:]), plt.title('Sinogram'), plt.colorbar()
            plt.subplot(122),plt.imshow(sinogram_shifted[:,sinogram_shifted.shape[1]//2,:]), plt.title('Sinogram shifted'), plt.colorbar()

    elif shift_method == 'geometry':
        # need to work out how to do unwrap in this case, not being done for now
        vol_geom, proj_geom = tch.init_astra_vec(Nx//binning, Ny//binning, theta_rad, shift_total) # try applying shifts with astra vector geometry
    
    # fbp (ASTRA needs shape Ny * Nangle * Nx)
    rec = tch.FBP_astra(sinogram_astra, vol_geom, proj_geom, weights)

    rec = tch.apply_circular_mask(rec, 0.9)
    
    if plot_figures:
        plt.figure(figsize=(10,3))
        plt.suptitle('Shifted reconstruction')
        plt.subplot(131),plt.imshow(rec[:,:,rec.shape[2]//2]), plt.xlabel('z'), plt.ylabel('x')
        plt.subplot(132),plt.imshow(rec[:,rec.shape[1]//2,:]), plt.xlabel('y'), plt.ylabel('x')
        plt.subplot(133),plt.imshow(rec[rec.shape[0]//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
        plt.tight_layout()
    
    
    # centering
    if center_reconstruction:
        rec_center = tc.centering_reconstruction2(rec)
        print(rec_center)
        
        if ii == 0:
            rec_center_0 = [rec.shape[0]/2,rec.shape[2]/2]
        shift_rec = -(rec_center - rec_center_0)
        
        # for now don't shift y because it's actually shifting z
        rec = tc.imshift_fft(rec, 0, shift_rec[0])

        # check if shift has moved it to the centre correctly
        rec_center = tc.centering_reconstruction2(rec)
        print(rec_center)


    # centering 
    # eps = np.finfo(rec.dtype).ep
    # w = np.sqrt(np.maximum(0,rec)) + eps
    # [x,y] = center_of_mass(w)
    # mass = w.sum()
    
    # get reprojection
    sinogram_model_astra = tch.get_projections(rec, vol_geom, proj_geom)
    sinogram_model = sinogram_model_astra.transpose((0,2,1))

    if plot_figures:
        plt.figure(figsize=(10,3))
        plt.subplot(131),plt.imshow(sinogram_shifted[:,sinogram_shifted.shape[1]//2,:]), plt.title('Sinogram shifted'), plt.colorbar()
        plt.subplot(132),plt.imshow(sinogram_model[:,sinogram_model.shape[1]//2,:]), plt.title('Sinogram model'), plt.colorbar()
        plt.subplot(133),plt.imshow(sinogram_shifted[:,sinogram_shifted.shape[1]//2,:]-sinogram_model[:,sinogram_shifted.shape[1]//2,:]), plt.title('Difference'), plt.colorbar()
        plt.tight_layout()

    MASS = np.median(sinogram * np.mean(abs(sinogram), axis=(0,1)))
    # MASS = 0
    
    # sinogram_model is reprojected sinogram
    # sinogram is the original sino (also called "sinogram_shifted" in the MATLAB code)
    if shift_method == 'physical':
        shift_upd, err = tc.find_optimal_shift(sinogram_model, sinogram_shifted, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
    else:
        shift_upd, err = tc.find_optimal_shift(sinogram_model, sinogram, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
    shift_upd[:, 1] -= np.median(shift_upd[:, 1]) # vertical only
    max_step = min(np.quantile(abs(shift_upd), 0.99), 0.5); 
    shift_upd = np.minimum(max_step, abs(shift_upd))*np.sign(shift_upd)*step_relaxation
    shift_total = shift_total + shift_upd

    shift_history.append(shift_upd)
    print(f'Iteration {str(ii)} time {time.time()-t0}')

shift_history = np.array(shift_history)
plt.figure(figsize=(10,5))
for i in range(iteration_no):
    plt.plot(theta, shift_history[i, :, 0], color='blue', alpha=0.3, label='x')
    plt.plot(theta, shift_history[i, :, 1], color='red', alpha=0.3, label='y')
    plt.xlabel("Angle")
    plt.ylabel("Shift value")
# plt.plot()

plt.figure(figsize=(10,5))
plt.plot(theta, delta_x)
plt.plot(theta, shift_total[:,0])

plt.ylabel('Horizontal shift value')
plt.xlabel('Angle (deg)')
plt.legend(['Simulated shifts', 'Calculated shifts'])
plt.grid()


# %%
