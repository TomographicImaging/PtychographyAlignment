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
import numpy as np
import time
from scipy.optimize import fmin
from scipy.signal.windows import tukey
import matplotlib.pyplot as plt
from IPython.display import clear_output

from utilities import phase_tools, recon_tools, shift_tools, sino_tools

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

        # taper weights at the edges to avoid edge artefacts dominating the shift estimate
        win = tukey(Nx, 0.2).reshape(1, 1, Nx)
        if Ny > 10 and self.config.align_vertical:
            win = tukey(Ny, 0.2).reshape(Ny, 1, 1) * win
        weights_find_shift = np.maximum(0, weights_find_shift * win)

        dtheta = (theta[-1] - theta[0]) / (len(theta) - 1) if len(theta) > 1 else 1.0
        weights_fbp = np.full(len(theta), dtheta, dtype=np.float32)
        
        shift_total = optimal_shift / binning
        shift_history = []
        shift_velocity = np.zeros((Nangles, 2))

        # Phase unwrapping
        if self.config.unwrap_data_method is not None:
            if self.config.unwrap_data_method == 'fft_1d':
                sinogram_shifted = -phase_tools.unwrap2D_fft(sinogram, axis=2, boundary=None)[0]
            else:
                raise ValueError("Supported unwrapping methods are None or 'fft_1d'")

        iterations = []
        maxvals = []
        plt.figure()
        for ii in range(self.config.max_iterations):
            t0 = time.time()

            # FBP
            vol_geom, proj_geom = recon_tools.init_astra_vec(Nx, Ny, theta, shift_total)
            rec = recon_tools.FBP_astra(sinogram_shifted, vol_geom, proj_geom, weights_fbp)

            # Mask
            if self.config.apply_mask: 
                rec = recon_tools.apply_circular_mask(rec, 0.9)

            # Remove negative values
            if self.config.apply_positivity:
                rec = np.maximum(0, rec)

            # Centering
            if self.config.center_reconstruction:
                rec_center = sino_tools.centering_reconstruction(rec)
                # print(rec_center)
                
                if ii == 0:
                    rec_center_0 = [rec.shape[2]/2,rec.shape[1]/2]

                shift_rec = -0.5*(rec_center - rec_center_0)
                rec = shift_tools.imshift_fft_2dax(rec, shift_rec[0], shift_rec[1], axis=(2,1))

                # debugging: check if shift has moved the rec to the centre correctly
                # rec_center = sino_tools.centering_reconstruction(rec)
                # print(rec_center)
                        
            # Get reprojection
            sinogram_model = recon_tools.get_projections(rec, vol_geom, proj_geom)

            # Calculate optimal shift
            MASS = np.median(sinogram_shifted * np.mean(abs(sinogram_shifted), axis=(0,1)))
            shift_upd, err = self.find_optimal_shift_ax(sinogram_model, sinogram_shifted, weights_find_shift, MASS, self.config.high_pass_filter, self.config.unwrap_data_method, 
                                                      align_horizontal=self.config.align_horizontal, align_vertical=self.config.align_vertical, axes=(0,2,1))
            

            shift_upd = np.minimum(0.5, np.abs(shift_upd)) * np.sign(shift_upd) * self.config.step_relaxation
            
            # Limit the shift size and apply a step relaxation factor
            #max_step = min(np.quantile(abs(shift_upd), 0.99), 0.5); 
            #shift_upd = np.minimum(max_step, abs(shift_upd))*np.sign(shift_upd)*self.config.step_relaxation
            
            # Update shift history
            shift_history.append(shift_upd) # reshape?
            # max_update = np.quantile(abs(shift_upd), 0.995)

            # Use momentum to accelerate convergence, but only once updates have mostly converged
            pre_momentum_max_update = np.quantile(np.abs(shift_upd[:, 0]), 0.995)
            if self.config.momentum_acceleration == True and ii > 2 and pre_momentum_max_update * binning < 0.5:
                
                shift_upd, shift_velocity = self.add_momentum_horizontal(shift_history, shift_velocity)

            # Apply a median shift in the vertical direction only
            shift_upd[:, 1] -= np.median(shift_upd[:, 1])
            
            max_step = np.minimum(np.quantile(abs(shift_upd), 0.99), 0.5)
            shift_upd = np.minimum(max_step, abs(shift_upd)) * np.sign(shift_upd) 

            shift_total = shift_total + shift_upd
            print('shift update shape:', shift_upd.shape)
            
            #position update smoothing?

            # Check the maximal step update and stop if it's below the stopping criterion
            # max_update = np.quantile(abs(shift_upd), 0.995)
            max_update = np.max(np.quantile(np.abs(shift_upd),0.995,axis=0))
            
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
            shift_history = np.array(shift_history)
            plt.figure(figsize=(10,5))
            for i in range(shift_history.shape[0]):
                plt.plot(i, np.quantile(abs(shift_history[i, :, :]), 0.995), 'ok')
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

    def find_optimal_shift_ax(self, sinogram_model, sinogram, weights, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False, axes=(0, 1, 2)):
        """
        Parameters
            axes Specify order of [Ny, Nx, Nangles] for sinogram_model and sinogram. Default matlab order is [Ny, Nx, Nangles] = (0, 1, 2), default astra order is [Ny, Nangles, Nx] = (0, 2, 1)"""

        Ny_ax = axes[0]
        Nx_ax = axes[1]
        Nangles_ax = axes[2]

        shift_x = np.zeros(sinogram_model.shape[Nangles_ax], dtype=np.float32)
        shift_y = np.zeros(sinogram_model.shape[Nangles_ax], dtype=np.float32)
        
        resid_sino = sinogram_model - sinogram
        # apply high pass filter to get rid of phase artefacts
        resid_sino = phase_tools.imfilter_high_pass_1d(resid_sino, Nx_ax, high_pass_filter)
        
        if unwrap_data_method.lower() == 'none':
            resid_sino = phase_tools.imfilter_high_pass_1d(resid_sino, ax=Nangles_ax, sigma=high_pass_filter, padding=0)
        
        # Horizontal alignment 
        if align_horizontal:      
            dX = phase_tools.get_img_grad_filtered_ax(sinogram_model, axis=Ny_ax, high_pass_filter=high_pass_filter, smooth_win=5, axes=axes)
            if unwrap_data_method.lower() == 'none':
                dX = phase_tools.imfilter_high_pass_1d(dX, ax=Nangles_ax, sigma=high_pass_filter, padding=0)
            
            numerator = np.sum(weights * dX * resid_sino, axis=(Ny_ax, Nx_ax))
            # if np.mean(numerator) < 0.01:
            #     numerator[:] = 0
            denominator = np.sum(weights * dX**2, axis=(Ny_ax, Nx_ax)) # sum2 and mean 2????????????????
            shift_x = -numerator / denominator

        
        # Vertical alignment
        if align_vertical:
            dY = phase_tools.get_img_grad_filtered_ax(sinogram_model, axis=Nx_ax, high_pass_filter=high_pass_filter, smooth_win=5, axes=axes)
            if unwrap_data_method.lower() == 'none':
                dY = phase_tools.imfilter_high_pass_1d(dY, ax=Nangles_ax, sigma=high_pass_filter, padding=0)

            numerator = np.sum(weights * dY * resid_sino, axis=(Nx_ax, Ny_ax))
            # if np.mean(numerator) < 0.01:
            #     numerator[:] = 0
            denominator = np.sum(weights * dY**2, axis=(Nx_ax, Ny_ax))
            shift_y = -numerator / denominator
        
        # Combine shifts
        shift = np.stack([shift_x, shift_y], axis=-1)

        # Check for NaNs
        if np.isnan(shift).any():
            print("Warning: Alignment failed, estimated shift is NaN")

        # Compute error
        err = np.sqrt(np.mean((weights * resid_sino)**2, axis=(Ny_ax, Nx_ax))) / MASS

        return shift, err

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
