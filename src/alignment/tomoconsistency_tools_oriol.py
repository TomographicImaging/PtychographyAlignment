#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:38:37 2025

@author: vdz11526
"""

# GET_PHASE_GRADIENT_1D get 1D gradient of phase of an image stack. 
# Accepts either complex image or just phase 
# 
# [d_img] = get_phase_gradient_1D(img, ax=2, step=0, shift=0)
#
# Inputs
#     **img     - stack of complex valued input images 
# *optional*
#     **ax      - axis of derivative, default = 2
#     **step    - step used to calculate the central difference, default=0 (analytic expression)
#     **shift   - perform shift and gradient calculation in single step 
#
# *returns*
#     d_img - phase gradient array
    
def get_phase_gradient_1D(img, ax=2, step=0.5, shift=0):

    # import utils.*
    # import math.*
    import numpy as np
    from scipy.ndimage import pad

    if np.isreal(img):
        img = np.exp(1j*img)

    np.testing.assert_allclose(step >= 0, 'Difference step has to be > 0')
    
    # suppress edge issues if phase ramp is not subtracted / there is no
    # air around sample 
    
    pad_distance = 8
    pad_widths = np.roll([pad_distance,0,0], ax-1)
    pad_config = [(w,w) for w in pad_widths]
    img = np.pad(img,pad_config,mode = 'symmetric')
    
    img = smooth_edges(img, pad_distance, ax); # this is from their utils

    if step == 0:
        # analytic formula (sensitive to noise) but faster 
        img = img ./ (abs(img) + eps); 
        d_img = get_img_grad(img, ax);  % img is assumed to be complex 
        d_img = imag(conj(img).*d_img);
    else
        d_img = angle( imshift_fft_ax(img,-step+shift,ax) .* conj( imshift_fft_ax(img,step+shift,ax)))/(2*step);
    end
    % remove padding 
    ind = circshift({pad_distance:size(d_img,ax)-pad_distance-1,':', ':'},ax-1);
    d_img = d_img(ind{:}); 
    

    return d_img