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

def smooth_edges(img, win_size=5, dims=[0,1]):
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


def imshift_fft(img, x, y=None, apply_fft=True, weights=None):
    """
    Apply subpixel shift using Fourier domain phase multiplication.

    Parameters:
        img (ndarray): Input image stack (can be complex).
        x (float or ndarray): Shift in x-direction (columns) or Nx2 array.
        y (float or ndarray): Shift in y-direction (rows).
        apply_fft (bool): If False, assume img is already in Fourier space.
        weights (ndarray or None): Optional weighting array to reduce noise.

    Returns:
        ndarray: Shifted image stack.
    """
    import numpy as np
    
    # Handle input arguments
    if y is None:
        y = x[:, 1]
        x = x[:, 0]

    if weights is not None and not np.isscalar(weights):
        eps_ = 1e2 * np.finfo(img.dtype).eps
        weights = np.maximum(eps_, weights)

    if np.all(x == 0) and np.all(y == 0):
        return img

    # Apply weights before FFT if provided
    if weights is not None and not np.isscalar(weights) and apply_fft:
        img = img * weights

    # Optimize for single-axis shifts
    if np.all(x == 0):
        return imshift_fft_ax(img, y, ax=0, apply_fft=apply_fft)
    elif np.all(y == 0):
        return imshift_fft_ax(img, x, ax=1, apply_fft=apply_fft)

    # Full 2D FFT-based shift
    real_img = np.isrealobj(img)
    Np = img.shape

    if apply_fft:
        img = np.fft.fft2(img, axes=(0,1)) # this was math.fft2_partial

    # Compute phase shift for x-axis
    xgrid = np.fft.ifftshift(np.arange(-Np[1] // 2, int(np.ceil(Np[1] / 2)))) / Np[1]
    X = np.exp(-2j * np.pi * np.reshape(x[:, None] * xgrid, (1, Np[1], -1)))
    img = img * X

    # Compute phase shift for y-axis
    ygrid = np.fft.ifftshift(np.arange(-Np[0] // 2, int(np.ceil(Np[0] / 2)))) / Np[0]
    Y = np.exp(-2j * np.pi * np.reshape(y[:, None] * ygrid, (Np[0], 1, -1)))
    img = img * Y

    if apply_fft:
        img = np.fft.ifft2(img, axes=(0,1)) # this was math.ifft2_partial

    if real_img:
        img = np.real(img)

    # Adjust weights after shift if provided
    if weights is not None and not np.isscalar(weights) and apply_fft:
        weights_shifted = imshift_fft(weights, x, y)
        img = img / weights_shifted

    return img

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


def imshift_linear(img, x, y=None, method='linear'):
    """
    Apply a shift to an image or stack of images using linear interpolation.
    
    Parameters:
        img (ndarray): Input image or stack of images (Nx, Ny, Nlayers).
        x (float or ndarray): Shift in x-direction (columns) or array of shifts per frame.
        y (float or ndarray): Shift in y-direction (rows) or array of shifts per frame.
        method (str): Interpolation method: 'nearest', 'linear', 'cubic', or 'circ'.
    
    Returns:
        ndarray: Shifted image or stack of images.
    """
    
    import numpy as np
    from scipy.ndimage import map_coordinates
    
    if y is None:
        # If y not provided, assume x contains pairs
        y = x[:, 1]
        x = x[:, 0]

    if np.all(np.asarray(x) == 0) and np.all(np.asarray(y) == 0):
        return img

    real_img = np.isrealobj(img)
    Nx, Ny, Nlayers = img.shape

    # Expand scalar shifts to arrays
    if np.isscalar(x):
        x = np.full(Nlayers, x)
    if np.isscalar(y):
        y = np.full(Nlayers, y)

    if method.lower() == 'circ':
        # Circular shift
        X = np.arange(Nx)
        Y = np.arange(Ny)
        for ii in range(Nlayers):
            img[:, :, ii] = img[np.roll(X, int(round(y[ii]))), np.roll(Y, int(round(x[ii]))), ii]
    else:
        # Interpolation-based shift
        order = {'nearest': 0, 'linear': 1, 'cubic': 3}.get(method.lower(), 1)
        for ii in range(Nlayers):
            coords_y, coords_x = np.meshgrid(np.arange(Ny), np.arange(Nx))
            coords = np.array([coords_x - y[ii], coords_y - x[ii]])
            img[:, :, ii] = map_coordinates(img[:, :, ii], coords, order=order, mode='constant', cval=0.0)

    if real_img:
        img = np.real(img)

    return img

import numpy as np

def imshift_linear_ax(img, shift, ax, method='linear', extrap_val=np.nan):
    """
    Apply a shift along a specific axis for each frame using linear interpolation.

    Parameters:
        img (ndarray): Input image or stack of images.
        shift (float or ndarray): Shift or vector of shifts for each frame.
        ax (int): Axis along which the shift will be performed.
        method (str): Interpolation method: 'nearest', 'linear', 'cubic', or 'circ'.
        extrap_val (float): Value for missing regions after interpolation (default: NaN).

    Returns:
        ndarray: Shifted image or stack of images.
    """
        
    if np.all(np.asarray(shift) == 0):
        return img

    Npix = img.shape
    img = img.astype(np.float32)

    # Bring the shifting axis to the front
    img = np.moveaxis(img, ax, 0)
    img_out = np.copy(img)

    fixed_axis = (ax + 1) % img.ndim  # Equivalent to MATLAB's ax_0

    if method.lower() == 'circ':
        # Circular shift
        for i in range(Npix[fixed_axis]):
            slicer = [slice(None)] * img.ndim
            slicer[fixed_axis] = i
            img_out[tuple(slicer)] = np.roll(img[tuple(slicer)], int(round(shift[i])), axis=0)
    else:
        # Interpolation-based shift
        order = {'nearest': 0, 'linear': 1, 'cubic': 3}.get(method.lower(), 1)
        for i in range(Npix[fixed_axis]):
            slicer = [slice(None)] * img.ndim
            slicer[fixed_axis] = i
            coords = np.arange(img.shape[0]) - shift[i]
            img_out[tuple(slicer)] = np.interp(coords, np.arange(img.shape[0]), img[tuple(slicer)], left=extrap_val, right=extrap_val)

    # Move axis back to original position
    img_out = np.moveaxis(img_out, 0, ax)

    return img_out

     
def get_phase_gradient_1D(img, ax=1, step=0.5, shift=0):
    '''
    GET_PHASE_GRADIENT_1D get 1D gradient of phase of an image stack. 
    Accepts either complex image or just phase 
    
    [d_img] = get_phase_gradient_1D(img, ax=2, step=0, shift=0)
    
    Inputs
        **img     - stack of complex valued input images 
    *optional*
        **ax      - axis of derivative, default = 1
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
    
    img = smooth_edges(img, pad_distance, [ax]) # this is from their utils

    if step == 0:
        # analytic formula (sensitive to noise) but faster 
        img = img / (abs(img) + np.finfo(float).eps)
        d_img = get_img_grad(img, ax) # img is assumed to be complex 
        d_img = np.imag(np.conj(img)*d_img)
    else:
        d_img = np.angle(imshift_fft_ax(img,-step+shift,ax) * np.conj(imshift_fft_ax(img,step+shift,ax)))/(2*step)
    
    # remove padding 
        
    # Create slicing indices
    ind = [slice(None)] * d_img.ndim
    ind[ax] = slice(pad_distance, d_img.shape[ax] - pad_distance - 1)
    
    # Apply circular shift to the list of slices
    ind = np.roll(ind, ax - 1)

    d_img = d_img[tuple(ind)]
    

    return d_img