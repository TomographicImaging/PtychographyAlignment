#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 28 16:43:56 2025

@author: vdz11526
"""

import h5py
import numpy as np
import hdf5plugin
# import scipy.io
import time
import glob
import re

tic = time.time()

from src.config.user_paths import DATA_ROOT
directory = DATA_ROOT / "Experimental" / "NiTi_Zifan_396963"
phase_filename = directory / "tomo_ptycho_396963_0_699_phase.nxs"
mod_filename = directory / "tomo_ptycho_396963_0_699_modulus.nxs"
filename = 'NiTiZifan_396963.mat'

from config.paths import NiTi_Zifan_data_key, NiTi_Zifan_angle_key
data_key = NiTi_Zifan_data_key
angle_key = NiTi_Zifan_angle_key

with h5py.File(phase_filename,'r') as data_file: 
    data_shape = data_file[str(data_key)].shape
    print('raw data shape:', data_shape)

    projections_phase=np.array(data_file[str(data_key)][:,:,:])

    angles_phase = np.array(data_file[angle_key][:], dtype=np.float64)
    print('angles shape:', angles_phase.shape)

with h5py.File(mod_filename,'r') as data_file: 
    data_shape = data_file[str(data_key)].shape
    print('raw data shape:', data_shape)

    projections_mod=np.array(data_file[str(data_key)][:,:,:])

print('convert to complex valued array...')
projections = projections_mod * ( np.cos(projections_phase) + 1j * np.sin(projections_phase) )

projections = np.transpose(projections, (1, 2, 0))

structured_dict = {
    'stack_object': np.array(projections, dtype=np.complex64),
    'theta': angles_phase,
    }

print('save...')
with h5py.File(directory + filename, 'w') as f:
    for key, value in structured_dict.items():
        f.create_dataset(key, data=value)

# scipy.io.savemat(directory + 'NiTiZifan_ordered.mat', structured_dict)

toc = time.time()
print('Elapsed time ' + str((toc - tic)/60) + ' minutes.')


#%%

# tic = time.time()

# directory = '/dls/i13-1/data/2025/cm40629-3/processing/Oriol/386939/'
# phase_filename = directory + 'tomo_ptycho_386939_0_719_phase.nxs'
# mod_filename = directory + 'tomo_ptycho_386939_0_719_modulus.nxs'
# filename = 'FBrun_386939.mat'

# data_files = sorted(glob.glob(directory + '_386939*.hdf'))
# print(data_files[0])
# total_no_files = len(data_files)
# no_projections = 720

# angles_phase = np.array(np.arange(0,180,180/no_projections), dtype=np.float64)
# object_key = '/entry_1/process_1/output_1/object'

# projections = np.empty((2235, 2022, total_no_files), dtype=np.complex64)
# matches = np.empty(total_no_files, dtype=np.int32)

# for i in range(total_no_files):
    
#     match = re.findall(r'386939_(\d+)_', str(data_files[i]))
#     matches[i] = int(match[0])
    
# sorted_indices = np.argsort(matches)
# sorted_matches = matches[sorted_indices]

# sorted_data_files = np.array(data_files)[sorted_indices]

# for i in range(total_no_files):
    
#     if i%50 == 0:
#         print(sorted_data_files[i])
#         print(str(i) + ' out of ' + str(total_no_files))
    
#     with h5py.File(sorted_data_files[i],'r') as data_file: 
#         projections[:,:,i]=np.array(data_file[object_key][0,0,0,0,0,0:2235,0:2022], dtype=np.complex64)
        
 
# final_angles = angles_phase[sorted_matches]

# structured_dict = {
#     'stack_object': np.array(projections, dtype=np.complex64),
#     'theta': final_angles,
#     }

# print('save...')
# with h5py.File(directory + filename, 'w') as f:
#     for key, value in structured_dict.items():
#         f.create_dataset(key, data=value)

# toc = time.time()
# print('Elapsed time ' + str((toc - tic)/60) + ' minutes.')

