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

import numpy as np
from scipy.signal.windows import tukey
from scipy.ndimage import map_coordinates, convolve
import scipy.signal as sig
from scipy.cluster.vq import kmeans2
from dataclasses import dataclass

import matplotlib.pyplot as plt
import tomoconsistency_tools_oriol as tc

@dataclass
class VerticalAlignmentConfig:
    data_filter: float = 0.01
    iterations: int = 1000


class VerticalAlignment:
    def __init__(self, config: VerticalAlignmentConfig = None):
        self.config = config or VerticalAlignmentConfig()

    def run_alignment(self, img, residuals, angles):
        """
        Runs vertical alignment, expects an image of shape (Ny, Nx, Nangles)
        
        Parameters
        ----------
        img: ndarray
            Phase unwrapped image of shape (Ny, Nx, Nangles)
        residuals: Boolean ndarray
            Boolean array mask  of residual values from the phase unwrapping. Must match img shape (Ny, Nx, Nangles)
        angles: ndarray
            Array of angles in radians

        """
        sinogram = np.sum(img,1) # get a vertical sinogram
        weight = ~(np.sum(residuals,1)>0) # valid pixels

        Ny, Nangles = sinogram.shape
        shift_Y = np.zeros((Nangles))

        sinogram = tc.remove_linear_ramp(sinogram)

        # remove edges
        sinogram = sinogram[1:Ny-1]
        weight = weight[1:Ny-1]

        # get an initial estimate of the Y shifts from cross-correlation
        shift_Y, _, weight = self.cross_correlation_estimate(sinogram, weight, angles)
        shift_Y = shift_Y - (max(shift_Y)+min(shift_Y))/2
        
        plt.figure(figsize=(8,4))
        plt.subplot(121),plt.plot(angles, shift_Y)
        plt.title('Shift from cross correlation estimate')
        plt.ylabel('Vertical shift (pixels)'), plt.xlabel('Angle (rad)')
        plt.grid()
        
        # vertical mass fluctuation
        refinement_loops = 3
        for ii in np.arange(refinement_loops):

            sinogram_shifted =  np.squeeze(tc.imshift_fft_ax(np.expand_dims(sinogram,1), np.squeeze(shift_Y),0))
            weights_shifted =  np.squeeze(tc.imshift_linear_ax(np.expand_dims(weight,1), np.squeeze(shift_Y), 0,'nearest'))
            
            offset = int(np.round(np.max(np.abs(shift_Y))))
            sinogram_shifted = sinogram_shifted[offset:sinogram.shape[0]-offset]
            weights_shifted = weights_shifted[offset:sinogram.shape[0]-offset]

            shift_update, sinogram_shifted, weights_shifted = self.linear_iterative_refinement(sinogram_shifted, weights_shifted)
            shift_Y = shift_Y + shift_update

        shift_Y = shift_Y - np.median(shift_Y)

        plt.subplot(122),plt.plot(angles, shift_Y)
        plt.title('Shift from vertical mass alignment')
        plt.ylabel('Vertical shift (pixels)'), plt.xlabel('Angle (rad)')
        plt.grid()
        plt.tight_layout()

        return shift_Y

    def linear_iterative_refinement(self, sinogram0, weights0):
        """
        Vertical mass fluctuation method with iterative refinement. 
        Expects a sinogram of shape (Ny, Nangles)

        """

        Nlayers, Nangles = sinogram0.shape
        total_shift_Y = np.zeros(Nangles)

        # initial filtering
        sinogram0 = tc.remove_linear_ramp(sinogram0)
        sinogram0 = tc.imfilter_high_pass_1d(sinogram0, 0, self.config.data_filter, Nlayers // 2)
        sinogram0 = sinogram0 * tukey(Nlayers, 0.1)[:, None]
        sinogram0 = self.fill_gaps_1D(sinogram0, ~weights0.astype(bool),20)

        X = np.arange(1, Nlayers + 1)
        Y = np.arange(1, Nangles + 1)
        X, Y = np.meshgrid(X, Y, indexing="ij")

        weights = map_coordinates(weights0, [X.ravel(), Y.ravel()], order=3, mode='nearest').reshape(weights0.shape)

        relax_step = 0.9
        err = []
        for i in range(self.config.iterations):
            # shift by current  guess
            sinogram = tc.imshift_fft_ax(sinogram0, total_shift_Y,ax=1)

            sinogram = tc.imfilter_high_pass_1d(sinogram, 0, self.config.data_filter, Nlayers // 2)
            sinogram = sinogram * tukey(Nlayers, 0.2)[:, None]

            m_sinogram = (np.sum(sinogram * weights, axis=1)/ (np.sum(weights, axis=1) + 1e-3))
            md_sinogram = self.get_img_grad_conv(m_sinogram, 2, 0)

            DY = m_sinogram[:, None] - sinogram

            numerator = np.sum(
                weights * (DY * md_sinogram[:, None]),
                axis=0,
            )

            denominator = np.sum(
                weights * (md_sinogram[:, None] ** 2),
                axis=0,
            )

            shift_Y = -numerator / denominator

            shift_Y = (
                relax_step
                * np.minimum(0.5, np.abs(shift_Y))
                * np.sign(shift_Y)
            )

            total_shift_Y = total_shift_Y + shift_Y

            current_err = np.mean(weights * DY**2)
            err.append(current_err)

            if (i > 2
                and err[i - 1] < err[i]
                or np.max(np.abs(shift_Y)) < 1e-2):
                break

        return total_shift_Y, sinogram, weights

    def cross_correlation_estimate(self, sinogram, weight, angles):
        """
        Estimate shifts from a 1D cross correlation. Expects a sinogram with data
        order (Ny, Nangles)
        """
        Nlayers, Nangles = sinogram.shape

        angle_sort = np.argsort(angles)

        sinogram = sinogram[:, angle_sort]
        weight = weight[:, angle_sort]

        # cluster sinogram rows to find a representative subset of data
        D = kmeans2(sinogram, 1)[0]
        ind = np.argsort(D)

        # initial filtering
        sinogram = tc.imfilter_high_pass_1d(
            sinogram, 0, self.config.data_filter, Nlayers // 2
        )

        window = sig.windows.tukey(Nlayers, alpha=0.1)
        sinogram = sinogram * window[:, None]

        # remove outliers
        q_low = np.quantile(sinogram, 0.01)
        q_high = np.quantile(sinogram, 0.99)

        weight_sinogram = (
            (sinogram > q_low)
            & (sinogram < q_high)
            & weight.astype(bool)
        )
        sinogram = np.clip(sinogram, q_low, q_high)
        sinogram = self.fill_gaps_1D(sinogram, ~weight_sinogram.astype(bool), 20)

        # take the median from the represenative subset as a reference
        sinogram_reference = np.median(sinogram[:, ind[0,0:5]],axis=1)
        sinogram_reference = np.expand_dims(sinogram_reference,1)

        # calculate cross correlation between sinogram and reference
        shift_Y = -self.find_xcorr_shift_1D(sinogram,sinogram_reference)

        # apply a median filter to the shifts to avoid jumps
        medfilt_win = 3
        mshift_Y = sig.medfilt(shift_Y, kernel_size=medfilt_win)

        medfilt_resid = shift_Y - mshift_Y

        q_low = np.quantile(medfilt_resid, 0.001)
        q_high = np.quantile(medfilt_resid, 0.999)
        medfilt_resid = np.clip(
            medfilt_resid,
            2 * q_low,
            2 * q_high,
        )
        shift_Y = mshift_Y + medfilt_resid

        # update weights
        weight = weight & weight_sinogram

        # return to original order
        scan_sort = np.argsort(angle_sort)
        shift_Y = shift_Y[scan_sort]
        weight = weight[:, scan_sort]
        sinogram = sinogram[:, scan_sort]

        return shift_Y, sinogram, weight
    
    def fill_gaps_1D(self, sinogram, mask_sinogram, Niter=20):
        """
        Removes masked values
        """

        if not np.any(mask_sinogram):
            return sinogram

        valid_values = sinogram[~mask_sinogram]
        q_low = np.quantile(valid_values, 0.01)
        q_high = np.quantile(valid_values, 0.99)

        for _ in range(Niter):

            U, S, Vt = np.linalg.svd(sinogram, full_matrices=False)

            sinogram_filt = U @ np.diag(S) @ Vt

            sinogram = (
                sinogram * (~mask_sinogram)
                + sinogram_filt * mask_sinogram
            )

            sinogram = np.clip(sinogram, q_low, q_high)

        return sinogram
    
    def find_xcorr_shift_1D(self, o1, o2):
        """
        Cross-correlation based shift estimation between two arrays along the 0th axis.
        """

        ax = 0

        max_shift = o1.shape[ax] / 3

        Npix = o1.shape
        shape = [1] * o1.ndim
        shape[ax] = Npix[ax]

        win = sig.windows.tukey(Npix[ax])
        win = win.reshape(shape)
        o1 = o1 * win
        o2 = o2 * win

        o1 = np.fft.fft(o1, axis=ax)
        o2 = np.fft.fft(o2, axis=ax)

        xcorrmat = np.abs(np.fft.ifft(o1 * np.conj(o2), axis=ax))

        # roll shift
        xcorrmat = np.roll(xcorrmat, Npix[ax] // 2, axis=ax)

        # Limit search range
        center = Npix[ax] // 2
        low = int(np.ceil(center - max_shift))
        high = int(np.ceil(center + max_shift))

        mask = np.zeros_like(xcorrmat, dtype=bool)
        slicer = [slice(None)] * o1.ndim
        slicer[ax] = slice(low, high)
        mask[tuple(slicer)] = True
        xcorrmat[~mask] = 0

        WIN = 10
        kernel = np.ones((WIN, 1)) if o1.ndim == 2 else np.ones(WIN)

        peak_mask = (xcorrmat == np.max(xcorrmat, axis=ax, keepdims=True))
        peak_mask = convolve(peak_mask.astype(float), kernel, mode='constant') > 0

        xcorrmat[~peak_mask] = 0

        # Normalize
        maxval = np.max(xcorrmat, axis=ax, keepdims=True)
        maxval[maxval == 0] = 1
        xcorrmat = (xcorrmat / maxval) ** 4

        mass = np.sum(xcorrmat, axis=ax)

        grid = np.arange(Npix[ax]).reshape(shape)

        shift = (np.sum(xcorrmat * grid, axis=ax) / (mass + 1e-12) - center - 1)

        return shift
    
    def get_img_grad_conv(self, img, win_size, axis=None):
        """
        Get gradient by convolution method
        """

        if not np.isrealobj(img):
            raise ValueError("Not implemented for complex input")

        ker = self.get_kernel(win_size)

        # If no axis specified → default to horizontal (MATLAB default was axis=2)
        if axis is None:
            axis = 1

        if np.isscalar(axis):
            axis = [axis]

        outputs = []

        for ax in axis:
            # Build kernel shape for broadcasting
            shape = [1] * img.ndim
            shape[ax] = len(ker)
            k = ker.reshape(shape)

            grad = sig.convolve(img, k, mode="same")
            outputs.append(grad)

        if len(outputs) == 1:
            return outputs[0]

        return tuple(outputs)

    def get_kernel(self, win_size):
        N = max(9, 2 * win_size + 1)

        grid = 2j * np.pi * (
            np.fft.fftshift(np.arange(N) / N) - 0.5
        )

        ker = -np.real(
            np.fft.fftshift(np.fft.fft(grid))
        ) / N

        center = N // 2
        half = int(np.ceil(win_size))

        ker = ker[center - half : center + half + 1]

        return ker.astype(np.float32)