# This file is a modified and translated version of code from the
# “cSAXS matlab package” https://zenodo.org/records/3539550 developed 
# by the CXS group https://www.psi.ch/en/sls/csaxs/software, at the Paul 
# Scherrer Institute (PSI), Switzerland.
#
# Original work:
# Copyright (c) 2017 Paul Scherrer Institute (http://www.psi.ch)
# Author: CXS group, PSI
#
# Licensed under the Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International (CC BY-NC-SA 4.0).
#
# This version:
# Copyright 2026 United Kingdom Research and Innovation (UKRI) and Diamond Light Source
# Modifications: Translated to Python, adapted and extended.
#
# This work is distributed under the same license as the original.
# You may not use this work for commercial purposes.
# You must give appropriate credit and indicate if changes were made.

# For publications using this software, please acknowledge:
# "This work uses a translated and adapted version of the
# 'cSAXS tomo package' developed by the CXS group,
# Paul Scherrer Institut, Switzerland." and cite  M. Odstrcil, M. Holler, J. Holler, 
# M. Guizar-Sicairos,"Alignment methods for nanotomography with deep sub-pixel accuracy", 
# Opt. Express, (2019).
#
# Full license text: https://creativecommons.org/licenses/by-nc-sa/4.0/

from dataclasses import dataclass
import astra
import numpy as np
import time
from scipy.optimize import fmin
import matplotlib.pyplot as plt
from IPython.display import clear_output
import warnings
import tomoconsistency_tools_oriol as tc

@dataclass
class TomoConsistencyConfig:
    max_iterations: int = 200
    step_relaxation: float = 0.5
    high_pass_filter: float = 0.01
    min_step_size: float = 0.01
    unwrap_data_method: str = 'fft_1d'
    plot_interactive: bool = True
    center_reconstruction: bool = True
    apply_mask: bool = False
    momentum_acceleration: bool = True
    align_horizontal: bool = True
    align_vertical: bool = False
    apply_positivity: bool = True

class TomoConsistencyAlignment:
    """
    tomo-consistency alignment code based on M. Odstrcil, M. Holler, J. Holler, M. Guizar-Sicairos, 
    'Alignment methods for nanotomography with deep sub-pixel accuracy', Optics Express, 2019
    """
    def __init__(self, config: TomoConsistencyConfig = None):
        self.config = config or TomoConsistencyConfig()

    def run_alignment(self, sinogram, theta, weights_find_shift, optimal_shift, binning):
        """
        Run tomo-consistency code. This code expects data in order [Ny, Nx, Nangles].

        Parameters
        ----------
        sinogram: ndarray
            Sinogram in data order [Ny, Nx, Nangles]
        theta: ndarray
            Array of angles in radians
        optimal_shift: ndarray
            The current best shift in Ny, Nx
        binning: int
            Curent binning value for sinogram
        """

        # transpose the sinogram to the [Ny, Nangles, Nx] order required by astra
        sinogram = sinogram.transpose((0, 2, 1))
        weights_find_shift = weights_find_shift.transpose((0, 2, 1))
        [Ny, Nangles, Nx] = sinogram.shape

        dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
        weights_fbp = np.full(len(theta), dtheta, dtype=np.float32)
        
        shift_total = optimal_shift / binning
        shift_history = []
        shift_velocity = np.zeros((Nangles, 2))

        # Phase unwrapping
        if self.config.unwrap_data_method is not None:
            if self.config.unwrap_data_method == 'fft_1d':
                sinogram_shifted = -tc.unwrap2D_fft(sinogram, axis=2, boundary=None)[0]
            else:
                raise ValueError("Supported unwrapping methods are None or 'fft_1d'")

        iterations = []
        maxvals = []
        for ii in range(self.config.max_iterations):
            t0 = time.time()

            # FBP
            vol_geom, proj_geom = self.init_astra_vec(Nx, Ny, theta, shift_total)
            rec = self.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights_fbp)

            # Mask
            if self.config.apply_mask: 
                rec = self.apply_circular_mask(rec, 0.9)

            # Remove negative values
            if self.config.apply_positivity:
                rec = np.maximum(0, rec)

            # Centering
            if self.config.center_reconstruction:
                rec_center = tc.centering_reconstruction(rec)
                # print(rec_center)
                
                if ii == 0:
                    rec_center_0 = [rec.shape[2]/2,rec.shape[1]/2]

                shift_rec = -0.5*(rec_center - rec_center_0)
                rec = tc.imshift_fft_2dax(rec, shift_rec[0], shift_rec[1], axis=(2,1))

                # debugging: check if shift has moved the rec to the centre correctly
                # rec_center = tc.centering_reconstruction(rec)
                # print(rec_center)
                        
            # Get reprojection
            sinogram_model = self.get_projections(rec, vol_geom, proj_geom)

            # Calculate optimal shift
            MASS = np.median(sinogram_shifted * np.mean(abs(sinogram_shifted), axis=(0,1)))
            shift_upd, err = tc.find_optimal_shift_ax(sinogram_model, sinogram_shifted, weights_find_shift, MASS, self.config.high_pass_filter, self.config.unwrap_data_method, 
                                                      align_horizontal=self.config.align_horizontal, align_vertical=self.config.align_vertical, axes=(0,2,1))
            
            # Limit the shift size and apply a step relaxation factor
            max_step = min(np.quantile(abs(shift_upd), 0.99), 0.5); 
            shift_upd = np.minimum(max_step, abs(shift_upd))*np.sign(shift_upd)*self.config.step_relaxation
            
            # Update shift history
            shift_history.append(shift_upd)

            # Use momentum to accelerate convergence
            if self.config.momentum_acceleration == True and ii > 2:
                shift_upd, shift_velocity = self.add_momentum_horizontal(shift_history, shift_velocity)

            # Apply a median shift in the vertical direction only
            shift_upd[:, 1] -= np.median(shift_upd[:, 1])

            shift_total = shift_total + shift_upd

            # Check the maximal step update and stop if it's below the stopping criterion
            max_update = np.quantile(abs(shift_upd), 0.995)
            
            if max_update*binning < self.config.min_step_size:
                break

            if self.config.plot_interactive:
                iterations.append(ii)
                maxvals.append(max_update)
                self.plot_alignment(shift_upd, shift_total, rec, theta, iterations, maxvals, binning)
            else:
                print(f"\rMaximum step update: {max_update*binning:.2f} px stopping criterion: {self.config.min_step_size:4.2g} px" , end="", flush=True)
            print(f'\rIteration {str(ii)} time {time.time()-t0:.2f} seconds', end="", flush=True)
        
        optimal_shift = shift_total*binning

        if self.config.plot_interactive is False:
            plt.figure(figsize=(10,5))
            for i in range(shift_history.shape[0]):
                plt.plot(i, np.quantile(abs(np.array(shift_history[i, :, :])), 0.995), 'ok')
                plt.ylabel("Maximum step update")
                plt.xlabel("Iteration")
            plt.show()

        return optimal_shift, err, rec, sinogram_shifted
    
    def plot_alignment(self, shift_upd, shift_total, rec, theta, iterations, maxvals, binning):
        
        clear_output(wait=True)  # clear previous figure
        fig, axs = plt.subplots(2, 3, figsize=(12, 7))

        plt.suptitle("Binning = " + str(binning))

        ax = axs[0,0]
        ax.plot(theta, shift_upd)
        ax.set_ylabel('Current shift')
        ax.set_xlabel('Angle (rad)')
        ax.legend(['Horizontal', 'Vertical'])
        ax.grid()

        ax = axs[0,1]
        ax.plot(theta, shift_total)
        ax.set_ylabel('Total shift')
        ax.set_xlabel('Angle (rad)')
        ax.legend(['Horizontal', 'Vertical'])
        ax.grid()

        ax = axs[0,2]
        ax.plot(iterations, maxvals, 'ok')
        ax.plot(ax.get_xlim(), [self.config.min_step_size/binning, self.config.min_step_size/binning], '--r')
        ax.set_ylabel('Maximum step update')
        ax.set_xlabel('Iteration')
        ax.legend(['Shift', 'Convergence criteria'])
        ax.grid()

        ax = axs[1,0]
        im=ax.imshow(rec[rec.shape[0]//2,:,:])
        fig.colorbar(im, ax=ax)
        ax.set_ylabel('Nx')
        ax.set_xlabel('Nx')

        ax = axs[1,1]
        im=ax.imshow(rec[:,rec.shape[1]//2,:])
        ax.set_ylabel('Nx')
        ax.set_xlabel('Ny')

        fig.colorbar(im, ax=ax)
        ax = axs[1,2]
        im=ax.imshow(rec[:,:,rec.shape[2]//2])
        fig.colorbar(im, ax=ax)
        ax.set_ylabel('Nx')
        ax.set_xlabel('Ny')

        plt.tight_layout()
        plt.show()

    def add_momentum_horizontal(self, shift_history, velocity_map):

        # get a subset of the shift history as a numpy array
        momentum_memory = 2
        shift_memory = np.stack(shift_history[-(momentum_memory+1):], axis=0)
        
        # this is the current shift we want to update with momentum
        shift = shift_memory[-1].copy()

        # only apply momentum in horizontal direction
        axis = 0 
        
        if np.all(shift[:, axis] == 0):
            return shift, velocity_map
        
        # find correlation between current shift and previous shifts in the horizontal direction
        C = np.zeros(momentum_memory)
        for ii in range(momentum_memory):
            prev_shift = shift_memory[ii][:, axis]
            if np.all(prev_shift == 0):
                C[ii] = 0
            else:
                # correlation coefficient
                C[ii] = np.corrcoef(shift[:, axis], prev_shift)[0,1]

        # minimise the difference between the current shift and the exponentially decayed previous shifts
        def loss(x):
            return np.linalg.norm(C - np.exp(-x * np.arange(momentum_memory, 0, -1)))

        decay = fmin(loss, 0.0, disp = False)[0]

        # scaling parameters
        alpha = 2.0
        gain = 0.5
        friction = np.clip(alpha * decay, 0, 1)

        # update velocity map
        velocity_map[:,axis] = (1 - friction) * velocity_map[:, axis] + shift[:, axis]

        # update shift using velocity map
        shift[:, axis] = (1 - gain) * shift[:, axis] + gain * velocity_map[:, axis]

        return shift, velocity_map
    
    def init_astra_vec(self, Nx, Ny, theta_rad, shifts, rot_center_x=0, rot_center_y=0):
        # Need to add COR

        delta_x = shifts[:,0]
        delta_y = shifts[:,1]
        vectors = np.zeros((len(theta_rad), 12), dtype=np.float32)
        du, dv = 1.0, 1.0 # detector pixel sizes - update this

        for i, th in enumerate(theta_rad):
            # ray direction
            vectors[i,0] =  np.sin(th)
            vectors[i,1] = -np.cos(th)
            vectors[i,2] =  0

            # u-direction (0,0)->(0,1)
            u_x = np.cos(th) * du
            u_y = np.sin(th) * du
            u_z = 0

            # v-direction (0,0)->(1,0)
            v_x = 0
            v_y = 0
            v_z = dv

            # detector center with shifts
            vectors[i,3] = delta_x[i] * u_x + delta_y[i] * v_x
            vectors[i,4] = delta_x[i] * u_y + delta_y[i] * v_y
            vectors[i,5] = delta_x[i] * u_z + delta_y[i] * v_z

            vectors[i,3] -= u_x * rot_center_x + v_x * rot_center_y
            vectors[i,4] -= u_y * rot_center_x + v_y * rot_center_y
            vectors[i,5] -= u_z * rot_center_x + v_z * rot_center_y

            # u-direction (0,0)->(0,1)
            vectors[i,6] = u_x
            vectors[i,7] = u_y
            vectors[i,8] = u_z

            # v-direction (0,0)->(1,0)
            vectors[i,9]  = 0
            vectors[i,10] = 0
            vectors[i,11] = dv

        proj_geom = astra.create_proj_geom('parallel3d_vec', 
            Ny,          
            Nx,          
            vectors
        )

        Nz = Nx

        vol_geom = astra.create_vol_geom(Nx, Nz, Ny)

        return vol_geom, proj_geom
    
    def FBP_astra(self, sinogram, vol_geom, proj_geom, weights):
      
        Ny, Nangles, Nx = sinogram.shape

        Fsize = max(64, 2**int(np.ceil(np.log2(2*Nx))))
        pad_left = (Fsize - Nx) // 2
        pad_right = Fsize - Nx - pad_left
        freqs = np.fft.fftfreq(Fsize)
        ramp = np.abs(freqs)

        filtered = np.zeros_like(sinogram)
        block_size = 2048
        for start in range(0, Ny, block_size):
            end = min(start + block_size, Ny)

            for iy in range(start, end):
                for ia in range(Nangles):
                    proj = sinogram[iy, ia, :]
                    proj_padded = np.pad(proj, (pad_left,pad_right), mode='constant')

                    F = np.fft.fft(proj_padded)
                    F *= ramp * weights[ia]
                    filtered_proj = np.real(np.fft.ifft(F))

                    filtered[iy, ia, :] = filtered_proj[pad_left:pad_left+Nx]
                
        filtered = filtered.astype(np.float32)

        sino_id = astra.data3d.create('-proj3d', proj_geom, filtered)
        vol_id  = astra.data3d.create('-vol',  vol_geom)

        cfg = astra.astra_dict('BP3D_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['ReconstructionDataId'] = vol_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        rec = astra.data3d.get(vol_id)

        astra.algorithm.delete(alg_id)
        astra.data3d.delete(sino_id)
        astra.data3d.delete(vol_id)

        return rec
    
    def get_projections(self, volume, vol_geom, proj_geom):

        vol_id = astra.data3d.create('-vol', vol_geom, volume)

        sino_id = astra.data3d.create('-proj3d', proj_geom)

        cfg = astra.astra_dict('FP3D_CUDA')
        cfg['ProjectionDataId'] = sino_id
        cfg['VolumeDataId'] = vol_id

        alg_id = astra.algorithm.create(cfg)
        astra.algorithm.run(alg_id)

        sino = astra.data3d.get(sino_id)

        # Cleanup
        astra.algorithm.delete(alg_id)
        astra.data3d.delete(vol_id)
        astra.data3d.delete(sino_id)

        return sino
    
    def apply_circular_mask(self, rec, radius=0.99, apodize=True):

        if rec.ndim == 2:
            H, W = rec.shape
            rec_reshaped = rec[None, ...] 
        elif rec.ndim == 3:
            _, H, W = rec.shape
            rec_reshaped = rec
        else:
            raise ValueError("rec must be 2D or 3D numpy array")

        if apodize == True:
            tomogram = np.zeros((H, W, 1), dtype=np.float32)
            _, mask = self.apply_3D_apodization(tomogram, rad_apod=0, axial_apod=0, radial_smooth=5)
        else:
            y_range = (H - 1) / 2
            x_range = (W - 1) / 2

            Y, X = np.ogrid[-y_range:y_range+1, -x_range:x_range+1]

            dist = np.sqrt(X**2 + Y**2)

            max_dim = max(H, W)
            radius_applied = radius * (max_dim / 2)

            r = np.sqrt(1 / np.pi)

            mask = (radius_applied - dist).clip(-r, r)
            mask *= (0.5 * np.pi) / r
            mask = np.sin(mask)
            mask = 0.5 + 0.5 * mask  

        masked = rec_reshaped * mask  

        if rec.ndim == 2:
            return masked[0]
        else:
            return masked
        
    def apply_3D_apodization(self, tomogram, rad_apod=None, axial_apod=None, radial_smooth=None):
        if tomogram.ndim != 3:
            raise ValueError("tomogram must have shape (rows, cols, layers)")
        rows, cols, layers = tomogram.shape
        if radial_smooth is None:
            radial_smooth = min(rows, cols) / 10.0
        circulo = None
        out = tomogram
        if rad_apod is not None:
            yt = np.arange(-np.floor(rows / 2.0), np.ceil(rows / 2.0), dtype=np.float32)
            xt = np.arange(-np.floor(cols / 2.0), np.ceil(cols / 2.0), dtype=np.float32)
            Y, X = np.meshgrid(yt, xt, indexing='ij')
            tappix = max(float(radial_smooth), 1.0)
            half_min = min(rows, cols) / 2.0
            zerorad = int(round(half_min - float(rad_apod) - tappix))
            taperfunc = self.radtap(X, Y, tappix, zerorad)
            circulo = np.float32(1.0 - taperfunc)
            out = out * circulo[:, :, np.newaxis]
        if axial_apod is not None and layers > 1:
            pad = max(0, int(round(layers - 2.0 * float(axial_apod))))
            filters = fract_hanning_pad(layers, layers, pad)
            filt = np.fft.ifftshift(filters[:, 0])
            out = out * filt[np.newaxis, np.newaxis, :]
        return out, circulo
    
    def radtap(self, X, Y, tappix, zerorad):
        tau = 2.0 * float(tappix)
        R = np.sqrt(X**2 + Y**2, dtype=np.float32)
        with np.errstate(invalid='ignore'):
            taper = 0.5 * (1.0 + np.cos(2.0 * np.pi * (R - zerorad - tau / 2.0) / tau))
        out = np.zeros_like(R, dtype=np.float32)
        mask_transition = R <= (zerorad + tau / 2.0)
        out[mask_transition] = taper[mask_transition]
        out[R > (zerorad + tau / 2.0)] = 1.0
        out[R < zerorad] = 0.0
        return out

def fract_hanning_pad(self, outputdim, filterdim=None, unmodsize=None):
    if filterdim is None and unmodsize is None:
        filterdim = outputdim
        unmodsize = 0
    elif filterdim is None or unmodsize is None:
        raise ValueError("Provide either only outputdim, or outputdim+filterdim+unmodsize.")
    outputdim = int(outputdim)
    filterdim = int(filterdim)
    unmodsize = int(unmodsize)
    if outputdim < unmodsize:
        raise ValueError("Output dimension must be smaller or equal to size of unmodulated window")
    if outputdim < filterdim:
        raise ValueError("Filter cannot be larger than output size")
    if unmodsize < 0:
        warnings.warn("Specified unmodsize < 0, setting unmodsize = 0")
        unmodsize = 0
    fd = filterdim
    N = np.arange(fd, dtype=np.float32)
    Nc, Nr = np.meshgrid(N, N, indexing='ij')
    if unmodsize == 0:
        inner = ((1.0 + np.cos(2.0 * np.pi * Nc / fd)) *
                 (1.0 + np.cos(2.0 * np.pi * Nr / fd))) / 4.0
    else:
        s = int(np.floor((unmodsize - 1) / 2.0))
        denom = float(fd + 1 - unmodsize)
        out_cols = 0.5 * (1.0 + np.cos(2.0 * np.pi * (Nc - s) / denom))
        if s > 0:
            out_cols[:, :s] = 1.0
        start_right_0b = s + fd + 2 - unmodsize
        if start_right_0b < fd:
            out_cols[:, start_right_0b:] = 1.0
        out_rows = 0.5 * (1.0 + np.cos(2.0 * np.pi * (Nr - s) / denom))
        if s > 0:
            out_rows[:s, :] = 1.0
        if start_right_0b < fd:
            out_rows[start_right_0b:, :] = 1.0
        inner = out_cols * out_rows
    inner = inner.astype(np.float32)
    out = np.zeros((outputdim, outputdim), dtype=np.float32)
    start_1b = int(np.round(outputdim / 2.0 + 1.0 - fd / 2.0))
    end_1b   = int(np.round(outputdim / 2.0 + 1.0 + fd / 2.0 - 1.0))
    start_0b = start_1b - 1
    end_excl = end_1b
    out[start_0b:end_excl, start_0b:end_excl] = np.fft.fftshift(inner)
    out = np.fft.fftshift(out)
    return out
