#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Oriol
"""

# import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import fft

# image1 = '/dls/i13-1/data/2025/cm40629-1/processing/ptycho-tomo_alignment/testing_FRC/_393797_0_20250206-143245.hdf'
# image2 = '/dls/i13-1/data/2025/cm40629-1/processing/ptycho-tomo_alignment/testing_FRC/_393798_0_20250206-143244.hdf'
# data_key = '/entry_1/process_1/output_1/object_phase'

# with h5py.File(image1,'r') as data_file: 
#     data_shape = data_file[str(data_key)].shape
#     print('raw data shape:', data_shape)

#     image1_array=np.array(data_file[str(data_key)][0,0,0,0,0,:,0:-1])

# with h5py.File(image2,'r') as data_file: 
#     data_shape = data_file[str(data_key)].shape
#     print('raw data shape:', data_shape)

#     image2_array=np.array(data_file[str(data_key)][0,0,0,0,0,:,:])
    
# plt.figure()
# plt.imshow(image1_array)

def fourier_ring_correlation(image1_array, image2_array):

    image1_fft = fft.fftshift(fft.fft2(image1_array))
    image2_fft = fft.fftshift(fft.fft2(image2_array))
    nx = image1_fft.shape[0]
    ny = image1_fft.shape[1]

    # plt.figure()
    # plt.imshow(abs(np.log(image1_fft)))

    correlation = image1_fft*np.conjugate(image2_fft)
    f1 = abs(image1_fft)**2
    f2 = abs(image2_fft)**2

    # plt.figure()
    # plt.imshow(abs(np.log(correlation)))

    frequencies = fft.fftshift(fft.fftfreq(image1_fft.shape[0], d=3e-8))
    condition = (frequencies > 0)
    frequencies_pos = frequencies[condition]

    x0 = nx//2
    y0 = ny//2

    x1 = np.linspace(-x0,x0,nx)
    y1 = np.linspace(-y0,y0,ny)
    xx, yy = np.meshgrid(x1,y1)

    coordinate_matrix = np.dstack((xx, yy))
    distance_matrix = np.sqrt(coordinate_matrix[:,:,0]**2 + coordinate_matrix[:,:,1]**2)
    distance_matrix_flat = distance_matrix.ravel()  # flatten

    n_rings = len(frequencies_pos)
    frc = np.empty(n_rings)
    fsigma = np.empty(n_rings)
    f7 = np.empty(n_rings)

    for ii in range(n_rings):
        condition = (distance_matrix_flat > ii-10) & (distance_matrix_flat < ii+10)
        correlation_summed = np.sum(correlation.ravel()[condition])
        f1_summed = np.sum(f1.ravel()[condition])
        f2_summed = np.sum(f2.ravel()[condition])

        frc[ii] = np.real(correlation_summed)/np.sqrt(f1_summed*f2_summed)
        fsigma[ii] = 2/np.sqrt(np.sum(condition)/2)


    f7[:] = frc[0]*(1/7)

    d = frc - fsigma
    minimum = np.min(abs(d))
    spatial_resolution_fsigma = (1/frequencies_pos[np.where(abs(d) == minimum)[0]])*1e9

    d7 = frc - f7
    minimum = np.min(abs(d7))
    spatial_resolution_f7 = (1/frequencies_pos[np.where(abs(d7) == minimum)[0]])*1e9

    plt.figure()
    plt.title('Fourier ring correlation')
    plt.xlabel('Spatial frequencies')
    plt.ylabel('Correlation value')
    plt.plot(frequencies_pos[0:n_rings],frc)
    plt.plot(frequencies_pos[0:n_rings],fsigma)
    plt.plot(frequencies_pos[0:n_rings],f7)
    plt.legend(['FRC', '2sigma', '1/7th'])

    print("Spatial resolution is roughly ", spatial_resolution_fsigma[0], " - ", spatial_resolution_f7[0], " nm.")

def fourier_shell_correlation(image1_array, image2_array):

    image1_fft = fft.fftshift(fft.fftn(image1_array))
    image2_fft = fft.fftshift(fft.fftn(image2_array))
    
    nz, ny, nx = image1_fft.shape

    correlation = image1_fft * np.conjugate(image2_fft)
    f1 = abs(image1_fft)**2
    f2 = abs(image2_fft)**2

    frequencies = fft.fftshift(fft.fftfreq(nx, d=3e-8))
    condition = (frequencies > 0)
    frequencies_pos = frequencies[condition]

    x0, y0, z0 = nx//2, ny//2, nz//2
    x1 = np.linspace(-x0, x0, nx)
    y1 = np.linspace(-y0, y0, ny)
    z1 = np.linspace(-z0, z0, nz)
    xx, yy, zz = np.meshgrid(x1, y1, z1, indexing='ij')
    distance_matrix_flat = np.sqrt(xx**2 + yy**2 + zz**2).ravel()

    n_rings = len(frequencies_pos)
    fsc = np.empty(n_rings)
    fsigma = np.empty(n_rings)
    f7 = np.empty(n_rings)

    correlation_flat = correlation.ravel()
    f1_flat = f1.ravel()
    f2_flat = f2.ravel()

    for ii in range(n_rings):
        condition = (distance_matrix_flat > ii-10) & (distance_matrix_flat < ii+10)
        correlation_summed = np.sum(correlation_flat[condition])
        f1_summed = np.sum(f1_flat[condition])
        f2_summed = np.sum(f2_flat[condition])

        fsc[ii] = np.real(correlation_summed) / np.sqrt(f1_summed * f2_summed)
        fsigma[ii] = 2 / np.sqrt(np.sum(condition) / 2)

    f7[:] = fsc[0] * (1/7)

    d = fsc - fsigma
    minimum = np.min(abs(d))
    spatial_resolution_fsigma = (1/frequencies_pos[np.where(abs(d) == minimum)[0]]) * 1e9

    d7 = fsc - f7
    minimum = np.min(abs(d7))
    spatial_resolution_f7 = (1/frequencies_pos[np.where(abs(d7) == minimum)[0]]) * 1e9

    plt.figure()
    plt.title('Fourier shell correlation')
    plt.xlabel('Spatial frequencies')
    plt.ylabel('Correlation value')
    plt.plot(frequencies_pos[0:n_rings], fsc)
    plt.plot(frequencies_pos[0:n_rings], fsigma)
    plt.plot(frequencies_pos[0:n_rings], f7)
    plt.legend(['FSC', '2sigma', '1/7th'])

    print("Spatial resolution is roughly", spatial_resolution_fsigma[0], "-", spatial_resolution_f7[0], "nm.")