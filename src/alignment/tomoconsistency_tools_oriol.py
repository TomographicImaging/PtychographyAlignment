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

def fft_partial(x, fft_axis, split_axis, split=1, inverse=False):
    
    import numpy as np
    
    # Estimate memory and determine split if not provided
    if split is None:
        raise Exception('This part of the code is not ready yet!')
        # mem_req = x.size * 8 * np.log2(x.shape[fft_axis])
        # if mem_req < 50e9:  # Assume at least 50 GB RAM available
        #     split = 1
        # else:
        #     avail_mem = utils.check_available_memory() * 1e6  # Placeholder
        #     split = int(np.ceil(mem_req / avail_mem))

    # If no splitting needed, apply FFT directly
    if split == 1:
        if inverse:
            return np.fft.ifft(x, axis=fft_axis)
        else:
            return np.fft.fft(x, axis=fft_axis)

    # Split and process in chunks
    Np = x.shape
    chunk_size = int(np.ceil(Np[split_axis] / split))

    slices = [slice(None)] * x.ndim
    for i in range(split):
        start = i * chunk_size
        end = min(Np[split_axis], (i + 1) * chunk_size)
        slices[split_axis] = slice(start, end)
        x_chunk = x[tuple(slices)]
        if inverse:
            x_chunk = np.fft.ifft(x_chunk, axis=fft_axis)
        else:
            x_chunk = np.fft.fft(x_chunk, axis=fft_axis)
        x[tuple(slices)] = x_chunk

    return x

def ifft_partial(x,fft_axis,split_axis, split = 1):

    x = fft_partial(x, fft_axis, split_axis, split, inverse=True)
    
    return x
    
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
    import numpy as np
    
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


def imshift_fft_ax(img, shift, ax, apply_fft=True):
    '''
    % IMSHIFT_FFT_AX  will apply subpixel shift that can be different for each 
% frame along one dimension only 
% If apply_fft == false, then images will be assumed to be in fourier space 
%
% Inputs:
%   **img - inputs ndim array to be shifted along ax-th dimension
%   **ax  - axis along which the array will be shifted 
%   **shift - Nx1 vector of shifts, positive direction is up
%   **apply_fft = false - if the img is already after fft, default is false
% *returns*: 
%   ++img - shifted image  / volume 
    '''
    import numpy as np
    
    if np.all(np.asarray(shift) == 0):
        return img

    is_real = np.isrealobj(img)
    Npix = img.shape

    # Determine shape for broadcasting
    if img.ndim == 3:
        Np = [1, 1, Npix[2]]
    else:
        Np = list(Npix)
        Np[ax] = 1

    Ng = [1, 1, 1]
    if ax >= img.ndim:
        Npix = list(Npix)
        Npix.append(1)
    Ng[ax] = Npix[ax]

    # Expand scalar shift to full shape
    if np.isscalar(shift):
        shift = np.full(Np, shift)

    # Create frequency grid
    grid = np.fft.ifftshift(np.arange(-Npix[ax] // 2, int(np.ceil(Npix[ax] / 2)))) / Npix[ax]

    # Compute phase shift
    X = np.exp(-2j * np.pi * np.reshape(shift, Np) * np.reshape(grid, Ng))

    # Apply FFT if requested
    if apply_fft:
        img = fft_partial(img, fft_axis=ax, split_axis=(1 + ax % img.ndim))

    # Apply phase shift
    img = img * X

    # Apply inverse FFT if requested
    if apply_fft:
        img = ifft_partial(img, fft_axis=ax, split_axis=(1 + ax % img.ndim))

    if is_real:
        img = np.real(img)

    return img

     
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
    
    if np.isreal(img.all()):
        img = np.exp(1j*img)

    # np.testing.assert_array_less(0, step, err_msg='Difference step has to be > 0') # it should be less or equal, but I couldn't find the right np.testing.assert
    
    # suppress edge issues if phase ramp is not subtracted / there is no
    # air around sample 
    
    pad_distance = 8
    pad_widths = np.roll([pad_distance,0,0], ax-1)
    pad_config = [(w,w) for w in pad_widths]
    img = np.pad(img,pad_config,mode = 'symmetric')
    
    img = smooth_edges(img, pad_distance, ax); # this is from their utils

    if step == 0:
        # analytic formula (sensitive to noise) but faster 
        img = img / (abs(img) + np.finfo(float).eps); 
        d_img = get_img_grad(img, ax); # img is assumed to be complex 
        d_img = np.imag(np.conj(img)*d_img);
    else:
        d_img = np.angle(imshift_fft_ax(img,-step+shift,ax) * np.conj(imshift_fft_ax(img,step+shift,ax)))/(2*step);
    
    # remove padding 
        
    # Create slicing indices
    ind = [slice(None)] * d_img.ndim
    ind[ax] = slice(pad_distance, d_img.shape[ax] - pad_distance - 1)
    
    # Apply circular shift to the list of slices
    ind = np.roll(ind, ax - 1)

    d_img = d_img[tuple(ind)]
    

    return d_img