#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 17:38:37 2025

@author: vdz11526
"""

def gausswin(L, a=2.5):
    '''
    Generates Gaussian window
    ----------
    L : int
        window size 
    a : float, optional
        Related to the variance. The default is 2.5.

    Returns
    -------
    w : array
        Gaussian 

    '''
    
    import numpy as np
    
    N = L - 1
    x = np.arange(0, L) - N/2
    w = np.exp(-0.5 * (a * x / (N/2))**2) # Gaussian
    
    return w

def smooth_edges(img, win_size=5, dims=(0,1)):
    '''
    SMOOTH_EDGES takes stack of 2D images and smooths boundaries to avoid sharp edge artefacts during imshift_fft 
    
    img = smooth_edges(img, win_size, dims)
    
    Inputs:
         **img - 2D stacked array, smoothing is done along first two dimensions 
         **win_size - size of the smoothing region, default is 3 
         **dims - list of dimensions along which will by smoothing done 
    Outputs: 
         ++img - smoothed array 
     '''
     
    import numpy as np
    from scipy.ndimage import convolve
    # from scipy.signal import windows

    try:
        Npix = img.shape
        for i in dims: 
            if Npix[i] <= 2 * win_size:
                continue
            win_size = max(win_size,3)
            
            # Get indices of edge regions
            edge_indices = list(range(Npix[i] - win_size, Npix[i])) + list(range(win_size)) 
            slicer = [slice(None)] * img.ndim
            slicer[i] = edge_indices
            img_tmp = img[tuple(slicer)]
            
            # Create Gaussian kernel
            ker_size = [1] * img.ndim
            ker_size[i] = win_size
            kernel= gausswin(win_size, 2.5).reshape(ker_size)
            
            # Smooth across image edges
            img_tmp = convolve(img_tmp, kernel, mode='constant', cval=0.0)
            
            # Normalise to avoid boundary issues
            boundary_shape = [1] * img.ndim
            boundary_shape[i] = len(edge_indices)
            norm = convolve(np.ones(boundary_shape), kernel, mode='constant', cval=0.0)
            img_tmp = img_tmp/norm
            
            # Assign smoothed values back
            img[tuple(slicer)] = img_tmp
             
    except Exception as err:
        print(f"Warning: Smooth edges failed: {err}")
                 
    return img

def fft_partial(x, fft_axis, split_axis, split=None, inverse=False):
    
    
def get_img_grad(img, axis=None, split=1):
    '''
    Parameters
    ----------
    img : TYPE
        DESCRIPTION.
    axis : TYPE, optional
        DESCRIPTION. The default is None.
    split : TYPE, optional
        DESCRIPTION. The default is 1.

    Returns
    -------
    dX : TYPE
        DESCRIPTION.
    dY : TYPE
        DESCRIPTION.

    '''
    
    # Check if image is real
    is_real = np.isrealobj(img)
    Np = img.shape

    dX = None
    dY = None

    if axis is None or 2 in axis:
        # Compute frequency vector for X-axis
        X = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[1]//2, np.ceil(Np[1]/2))) / Np[1]
        # Apply partial FFT and multiply by frequency
        dX = fft_partial(img, 2, 1, split, False) * X
        # Apply inverse partial FFT
        dX = fft_partial(dX, 2, 1, split, True)
        if is_real:
            dX = np.real(dX)

    if axis is not None and (1 in axis or dX is None):
        # Compute frequency vector for Y-axis
        Y = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[0]//2, np.ceil(Np[0]/2))) / Np[0]
        # Apply partial FFT and multiply by frequency
        dY = fft_partial(img, 1, 2, split, False) * Y[:, np.newaxis]
        # Apply inverse partial FFT
        dY = fft_partial(dY, 1, 2, split, True)
        if is_real:
            dY = np.real(dY)
        if axis is None or len(axis) == 1:
            dX = dY

    return dX, dY

     
def get_phase_gradient_1D(img, ax=2, step=0.5, shift=0):
    '''
    GET_PHASE_GRADIENT_1D get 1D gradient of phase of an image stack. 
    Accepts either complex image or just phase 
    
    [d_img] = get_phase_gradient_1D(img, ax=2, step=0, shift=0)
    
    Inputs
        **img     - stack of complex valued input images 
    *optional*
        **ax      - axis of derivative, default = 2
        **step    - step used to calculate the central difference, default=0 (analytic expression)
        **shift   - perform shift and gradient calculation in single step 
    
    *returns*
        d_img - phase gradient array
    '''
    
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
        img = img / (abs(img) + eps); 
        d_img = get_img_grad(img, ax); # img is assumed to be complex 
        d_img = np.imag(np.conj(img)*d_img);
    else
        d_img = angle( imshift_fft_ax(img,-step+shift,ax) .* conj( imshift_fft_ax(img,step+shift,ax)))/(2*step);
    end
    % remove padding 
    ind = circshift({pad_distance:size(d_img,ax)-pad_distance-1,':', ':'},ax-1);
    d_img = d_img(ind{:}); 
    

#     return d_img