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

file = '/dls/i13-1/data/2025/cm40629-5/processing/ptycho-tomo_alignment/connor_wright/275019_275199_tomo.nxs'

with h5py.File(file, 'r') as f:
    img_orig = f['/stack_object'][:,:,0:10]
    
sino = tc.unwrap_data(img_orig, 'fft_1d', None)