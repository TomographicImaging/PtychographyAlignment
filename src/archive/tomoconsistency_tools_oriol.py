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
    
# def get_img_grad(img, axis=None, split=1):
#     '''
#     Parameters
#     ----------
#     img : TYPE
#         DESCRIPTION.
#     axis : TYPE, optional
#         DESCRIPTION. The default is None.
#     split : TYPE, optional
#         DESCRIPTION. The default is 1.

#     Returns
#     -------
#     dX : TYPE
#         DESCRIPTION.
#     dY : TYPE
#         DESCRIPTION.

#     '''
#     import numpy as np
    
#     # Check if image is real
#     is_real = np.isrealobj(img)
#     Np = img.shape

#     dX = None
#     dY = None
    
#     axis = np.array(axis) # check if this works
    
#     if axis is None or 2 in axis:
#         # Compute frequency vector for X-axis
#         X = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[1]//2, np.ceil(Np[1]/2))) / Np[1]
#         # Apply partial FFT and multiply by frequency
#         dX = fft_partial(img, 1, 1, split, False) * X
#         # Apply inverse partial FFT
#         dX = fft_partial(dX, 1, 1, split, True)
#         if is_real:
#             dX = np.real(dX)

#     if axis is not None and (1 in axis or dX is None):
#         # Compute frequency vector for Y-axis
#         Y = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[0]//2, np.ceil(Np[0]/2))) / Np[0]
#         # Apply partial FFT and multiply by frequency
#         dY = fft_partial(img, 0, 2, split, False) * Y[:, np.newaxis, np.newaxis]
#         # Apply inverse partial FFT
#         dY = fft_partial(dY, 0, 2, split, True)
#         if is_real:
#             dY = np.real(dY)
#         if axis is None or axis.size == 1:
#             dX = dY

#     return dX, dY

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
    
    if axis is None:
        axis = np.array((0,1))
    else:
        axis = np.array(axis) 

    if 1 in axis:
        # Compute frequency vector for X-axis
        X = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[1]//2, np.ceil(Np[1]/2))) / Np[1]
        # Apply partial FFT and multiply by frequency
        dX = fft_partial(img, 1, 1, split, False)
        shape = [1] * img.ndim
        shape[1] = Np[1]
        dX = dX * np.broadcast_to(X.reshape(shape),Np)
        # Apply inverse partial FFT
        dX = fft_partial(dX, 1, 1, split, True)
        if is_real:
            dX = np.real(dX)

    if 0 in axis:
        # Compute frequency vector for Y-axis
        Y = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[0]//2, np.ceil(Np[0]/2))) / Np[0]
        # Apply partial FFT and multiply by frequency
        dY = fft_partial(img, 0, 2, split, False)
        shape = [1] * img.ndim
        shape[0] = Np[0]
        dY = dY * np.broadcast_to(Y.reshape(shape),Np)
        # Apply inverse partial FFT
        dY = fft_partial(dY, 0, 2, split, True)
        if is_real:
            dY = np.real(dY)
            
    if dX is None:
        dX = dY


    return dX, dY

def imshift_fft_2dax(img, x, y=None, axis=(1, 0), apply_fft=True, weights=None):
    """
    Apply subpixel shift using Fourier domain phase multiplication.
    img can be in any order - specify the axes to apply shifts with the axis parameter

    Parameters:
        img (ndarray): Input image stack (can be complex).
        x (float or ndarray): Shift in x-direction (columns) or Nx2 array.
        y (float or ndarray): Shift in y-direction (rows).
        apply_fft (bool): If False, assume img is already in Fourier space.
        weights (ndarray or None): Optional weighting array to reduce noise.
        axis (tuple): Tuple of axes (ax_x, ax_y) to apply shifts x and y, default 
            x is applied to 1 and y is applied to 0.

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
        return imshift_fft_ax(img, y, ax=axis[1], apply_fft=apply_fft)
    elif np.all(y == 0):
        return imshift_fft_ax(img, x, ax=axis[0], apply_fft=apply_fft)

    # Full 2D FFT-based shift
    real_img = np.isrealobj(img)
    Np = img.shape

    if apply_fft:
        img = np.fft.fft2(img, axes=axis) # this was math.fft2_partial

    all_axes = set(range(len(Np)))
    ax_z = list(all_axes - set(axis))[0]

    # Compute phase shift for x-axis
    ax_x = axis[0]
    Ng = [1] * img.ndim
    Ng[ax_x] = Np[ax_x]

    if np.isscalar(x):
        shift = np.full(Np, x)
    else:
        shape = [1] * img.ndim
        shape[ax_z] = len(x)
        shift = np.broadcast_to(x.reshape(shape), Np)

    xgrid = np.fft.ifftshift(np.arange(-(Np[ax_x]//2), int(np.ceil(Np[ax_x]/2)))) / Np[ax_x]
    xgrid = xgrid.reshape(Ng)
    X = np.exp(-2j * np.pi * shift * xgrid)
    img *= X

    # Compute phase shift for y-axis
    ax_y = axis[1]
    Ng = [1] * img.ndim
    Ng[ax_y] = Np[ax_y]
    
    if np.isscalar(y):
        shift = np.full(Np, y)
    else:
        shape = [1] * img.ndim
        shape[ax_z] = len(x)
        shift = np.broadcast_to(y.reshape(shape), Np)
    
    ygrid = np.fft.ifftshift(np.arange(-(Np[ax_y]//2), int(np.ceil(Np[ax_y]/2)))) / Np[ax_y]
    ygrid = ygrid.reshape(Ng)
    Y = np.exp(-2j * np.pi * shift * ygrid)
    img *= Y

    if apply_fft:
        img = np.fft.ifft2(img, axes=axis) # this was math.ifft2_partial

    if real_img:
        img = np.real(img)

    # Adjust weights after shift if provided
    if weights is not None and not np.isscalar(weights) and apply_fft:
        weights_shifted = imshift_fft(weights, x, y)
        img = img / weights_shifted

    return img

    

def imshift_fft_transposed(img, x, y=None, apply_fft=True, weights=None):
    """
    Transposed version of imshift_fft for data in order [Ny, Nangles, Nx]
    y is applied to axis 0 [Ny] and x is applied to axis 2 [Nx]

    Parameters:
        img (ndarray): Input image stack (Ny, Nangles, Nx).
        x (float or ndarray): Shift in x-direction (columns) or Nx2 array.
        y (float or ndarray): Shift in y-direction (rows).
        apply_fft (bool): If False, assume img is already in Fourier space.
        weights (ndarray or None): Optional weighting array to reduce noise.
    
    Returns:
        ndarray: Shifted image stack
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
        return imshift_fft_ax(img, x, ax=2, apply_fft=apply_fft)

    # Full 2D FFT-based shift
    real_img = np.isrealobj(img)
    Np = img.shape

    if apply_fft:
        img = np.fft.fft2(img, axes=(0,2)) # this was math.fft2_partial

    # Compute phase shift for x-axis
    Ng = [1, 1, Np[2]]
    # shift = np.full(Np, x)
    shift = np.broadcast_to(x[None,:,None], Np)
    xgrid = np.fft.ifftshift(np.arange(-(Np[2] // 2), int(np.ceil(Np[2] / 2)))) / Np[2]
    X = np.exp(-2j * np.pi * np.reshape(shift, Np) * np.reshape(xgrid, Ng))
    img = img * X

    # Compute phase shift for y-axis
    Ng = [Np[0], 1, 1]
    # shift = np.full(Np, y)
    shift = np.broadcast_to(y[None,:,None], Np)
    ygrid = np.fft.ifftshift(np.arange(-(Np[0] // 2), int(np.ceil(Np[0] / 2)))) / Np[0]
    Y = np.exp(-2j * np.pi * np.reshape(shift, Np) * np.reshape(ygrid, Ng))
    img = img * Y

    if apply_fft:
        img = np.fft.ifft2(img, axes=(0,2)) # this was math.ifft2_partial

    if real_img:
        img = np.real(img)

    # Adjust weights after shift if provided
    if weights is not None and not np.isscalar(weights) and apply_fft:
        weights_shifted = imshift_fft(weights, x, y)
        img = img / weights_shifted

    return img
    

def imshift_fft(img, x, y=None, apply_fft=True, weights=None):
    """
    Apply subpixel shift using Fourier domain phase multiplication.
    img is in order [Ny, Nx, Nangles]
    y is applied to axis 0 [Ny] and x is applied to axis 1 [Nx]

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
    Ng = [1, Np[1], 1]
    shift = np.full(Np, x)
    xgrid = np.fft.ifftshift(np.arange(-(Np[1] // 2), int(np.ceil(Np[1] / 2)))) / Np[1]
    X = np.exp(-2j * np.pi * np.reshape(shift, Np) * np.reshape(xgrid, Ng))
    img = img * X

    # Compute phase shift for y-axis
    Ng = [Np[0], 1, 1]
    shift = np.full(Np, y)
    ygrid = np.fft.ifftshift(np.arange(-(Np[0] // 2), int(np.ceil(Np[0] / 2)))) / Np[0]
    Y = np.exp(-2j * np.pi * np.reshape(shift, Np) * np.reshape(ygrid, Ng))
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

    Ng = [1] * img.ndim
    Ng[ax] = Npix[ax]

    # Expand scalar shift to full shape
    if np.isscalar(shift):
        shift = np.full(Np, shift)
        shift = np.reshape(shift, Np)
    else:
        ax_len_shift = np.where(np.array(img.shape) == len(shift))[0][0]
        shape = [1] * img.ndim
        shape[ax_len_shift] = len(shift)
        shift = np.broadcast_to(shift.reshape(shape), Npix)

    # Create frequency grid
    grid = np.fft.ifftshift(np.arange(-(Npix[ax] // 2), int(np.ceil(Npix[ax] / 2)))) / Npix[ax]

    # Compute phase shift
    X = np.exp(-2j * np.pi * shift * np.reshape(grid, Ng))

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
        
    img_f = np.zeros(img.shape, dtype=img.dtype)
    
    if method.lower() == 'circ':
        # Circular shift
        X = np.arange(Nx)
        Y = np.arange(Ny)
        for ii in range(Nlayers):
            img_f[:, :, ii] = img[np.roll(X, int(round(y[ii]))), np.roll(Y, int(round(x[ii]))), ii]
    else:
        # Interpolation-based shift
        order = {'nearest': 0, 'linear': 1, 'cubic': 3}.get(method.lower(), 1)
        for ii in range(Nlayers):
            coords_y, coords_x = np.meshgrid(np.arange(Ny), np.arange(Nx))
            coords = np.array([coords_x - y[ii], coords_y - x[ii]])
            img_f[:, :, ii] = map_coordinates(img[:, :, ii], coords, order=order, mode='constant', cval=0.0)

    if real_img:
        img_f = np.real(img_f)

    return img_f

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
    from scipy.ndimage import map_coordinates
     
    if np.all(np.asarray(shift) == 0):
        return img

    Nx, Ny, Nlayers = img.shape
    
    if np.isscalar(shift):
        shift = np.full(Nlayers, shift)
        
    img_f = np.copy(img)

    if method.lower() == 'circ':
        # Circular shift
        X = np.arange(Nx)
        Y = np.arange(Ny)
        for ii in range(Nlayers):
            if ax == 0:
                img_f[:, :, ii] = img[np.roll(X, int(round(shift[ii]))), :, ii]
            if ax == 1:
                img_f[:, :, ii] = img[:, np.roll(Y, int(round(shift[ii]))), ii]
            if ax != 0 and ax != 1:
                print(ax)
                raise Exception('"ax" is neither 0 nor 1')
    else:
        # Interpolation-based shift
        order = {'nearest': 0, 'linear': 1, 'cubic': 3}.get(method.lower(), 1)
        for ii in range(Nlayers):
            coords_y, coords_x = np.meshgrid(np.arange(Ny), np.arange(Nx))
            if int(ax) == int(0):
                coords = np.array([coords_x - shift[ii], coords_y])
            if int(ax) == int(1):
                coords = np.array([coords_x, coords_y - shift[ii]])    
            if ax != 0 and ax != 1:
                print(ax)
                raise Exception('"ax" is neither 0 nor 1')
                
            img_f[:, :, ii] = map_coordinates(img[:, :, ii], coords, order=order, mode='constant', cval=0.0)
            
    return img_f


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
        d_img = get_img_grad(img, ax)[0] # img is assumed to be complex 
        d_img = np.imag(np.conj(img)*d_img)
    else:
        d_img = np.angle(imshift_fft_ax(img,-step+shift,ax) * np.conj(imshift_fft_ax(img,step+shift,ax)))/(2*step)
    
    # remove padding 
        
    # Create slicing indices
    ind = [slice(None)] * d_img.ndim
    ind[0] = slice(pad_distance, d_img.shape[ax-1] - pad_distance - 1)
    
    # Apply circular shift to the list of slices
    ind = np.roll(ind, ax - 1)

    d_img = d_img[tuple(ind)]
    

    return d_img


def wrapToPi(x):
    """
    Wrap values to the range [-pi, pi].
    """
    return (x + np.pi) % (2 * np.pi) - np.pi


def findresidues(phase):
    """
    Compute phase residues for 2D phase unwrapping.

    Parameters:
        phase (ndarray): Input phase (real or complex).

    Returns:
        ndarray: Residues array.
    """
    # If input is complex, take its phase angle
    if not np.isrealobj(phase):
        phase = np.angle(phase)

    # Compute residues using wrapped differences
    residues = wrapToPi(phase[1:, :-1, ...] - phase[:-1, :-1, ...])
    residues += wrapToPi(phase[1:, 1:, ...] - phase[1:, :-1, ...])
    residues += wrapToPi(phase[:-1, 1:, ...] - phase[1:, 1:, ...])
    residues += wrapToPi(phase[:-1, :-1, ...] - phase[:-1, 1:, ...])

    residues = residues / (2 * np.pi)
    return residues


def get_img_int_1D(grad_array, ax=0):
    """
    Use FFT to integrate the image along the selected axis.
    Can be used for phase unwrapping.

    Parameters:
        grad_array (ndarray): Phase gradient array.
        ax (int): Axis along which to integrate (default: 0).

    Returns:
        ndarray: Integrated image.
    """
    Np = grad_array.shape

    if ax == 1:  # MATLAB axis=2 → Python axis=1
        grad_array_fft = np.fft.fft(grad_array, axis=1)
        xgrid = np.fft.ifftshift(np.arange(-(Np[1] // 2), int(np.ceil(Np[1] / 2)))) / Np[1]

        # Integration filter
        X = np.exp(2j * np.pi * xgrid)
        X = X / (2j * np.pi * xgrid)
        X[0] = 0  # Avoid division by zero

        shape = [1] * grad_array.ndim
        shape[1] = Np[1]

        # Apply filter and inverse FFT
        integer = grad_array_fft * np.broadcast_to(X.reshape(shape),Np)
        integer = np.fft.ifft(integer, axis=1)

    elif ax == 0:  # MATLAB axis=1 → Python axis=0
        grad_array_fft = np.fft.fft(grad_array, axis=0)
        ygrid = np.fft.ifftshift(np.arange(-(Np[0] // 2), int(np.ceil(Np[0] / 2)))) / Np[0]

        # Integration filter
        Y = np.exp(2j * np.pi * ygrid)
        Y = Y / (2j * np.pi * ygrid)
        Y[0] = 0

        shape = [1] * grad_array.ndim
        shape[0] = Np[0]

        # Apply filter and inverse FFT
        integer = grad_array_fft *  np.broadcast_to(Y.reshape(shape),Np)
        integer = np.fft.ifft(integer, axis=0)

    elif ax == 2:
        grad_array_fft = np.fft.fft(grad_array, axis=2)
        zgrid = np.fft.ifftshift(np.arange(-(Np[2] // 2), int(np.ceil(Np[2] / 2)))) / Np[2]

        # Integration filter
        Z = np.exp(2j * np.pi * zgrid)
        Z = Z / (2j * np.pi * zgrid)
        Z[0] = 0

        shape = [1] * grad_array.ndim
        shape[2] = Np[2]

        # Apply filter and inverse FFT
        integer = grad_array_fft * np.broadcast_to(Z.reshape(shape),Np)
        integer = np.fft.ifft(integer, axis=2)

    else:
        raise ValueError("Non-implemented dimension")

    return integer

def remove_sinogram_ramp(sinogram, air_gap, polyfit_order=-1):
    """
    Remove phase ramp/offset from an unwrapped sinogram using air regions.

    Parameters:
        sinogram (ndarray): Unwrapped projections (Nlayers x width x ...).
        air_gap (list or tuple): [left_gap, right_gap] in pixels.
        polyfit_order (int): 
            -1 = subtract linear offset from each row separately
             0 = remove constant offset
             1 = remove 2D plane ramp

    Returns:
        ndarray: Sinogram after ramp removal.
    """
    air_gap = np.ceil(air_gap).astype(int)
    Nlayers, width_sinogram = sinogram.shape[:2]
    ax = np.arange(width_sinogram)

    # Masks for air regions
    mask_left = ax <= air_gap[0]
    mask_right = ax >= width_sinogram - air_gap[min(len(air_gap)-1, 1)]

    air_values = []

    for mask in [mask_left, mask_right]:
        # Average values in air region
        avg_vals = np.sum(sinogram[:, mask], axis=1) / np.sum(mask)

        if polyfit_order == 0:
            # Constant offset
            avg_vals = np.mean(avg_vals)
        elif polyfit_order == 1:
            # Fit a 2D plane iteratively
            ramp = np.linspace(-1, 1, Nlayers)
            weight = np.ones_like(avg_vals)
            for _ in range(10):
                plane_fit = (np.mean(weight * avg_vals) / np.mean(weight) +
                             np.mean(weight * avg_vals * ramp) / np.mean(weight * ramp**2) * ramp)
                deviation = 5 * np.median(np.abs(avg_vals - plane_fit))
                weight = 1 / (1 + (np.abs(avg_vals - plane_fit) / deviation)**2)
            avg_vals = plane_fit

        air_values.append(avg_vals)

    # Interpolate ramp between left and right air regions
    ramp = np.interp(ax, [0, width_sinogram - 1], np.vstack(air_values).T)
    ramp = np.expand_dims(ramp, axis=0)  # Match dimensions for broadcasting

    # Remove ramp
    sinogram = sinogram - ramp

    return sinogram


def remove_sinogram_ramp_3D(sinogram, air_gap, polyfit_order=1):
    """
    Remove phase ramp/offset from a 3D unwrapped sinogram using air regions.

    Parameters:
        sinogram (ndarray): 3D array of shape (Nlayers, width, Nprojections)
        air_gap (list or tuple): [left_gap, right_gap] in pixels
        polyfit_order (int):
             0 = remove constant offset using air regions
             1 = remove 2D plane fit along rows using air regions

    Returns:
        ndarray: Sinogram after ramp removal
    """
    sinogram = np.asarray(sinogram)
    Nlayers, width, Nproj = sinogram.shape
    air_gap = np.ceil(air_gap).astype(int)

    ax = np.arange(width)
    mask_left = ax < air_gap[0]
    mask_right = ax >= width - air_gap[min(len(air_gap)-1, 1)]

    # average values in air regions (left and right)
    left_vals = np.mean(sinogram[:, mask_left, :], axis=1)  # shape: (Nlayers, Nproj)
    right_vals = np.mean(sinogram[:, mask_right, :], axis=1)

    if polyfit_order == 0:
        left_vals[:] = np.mean(left_vals, axis=0)
        right_vals[:] = np.mean(right_vals, axis=0)
    elif polyfit_order == 1:
        ramp = np.linspace(-1, 1, Nlayers)[:, None]
        for vals in [left_vals, right_vals]:
            weight = np.ones_like(vals)
            for _ in range(10):
                plane_fit = (np.sum(weight * vals, axis=0, keepdims=True) / np.sum(weight, axis=0, keepdims=True) +
                             np.sum(weight * vals * ramp, axis=0, keepdims=True) / np.sum(weight * ramp**2, axis=0, keepdims=True) * ramp)
                deviation = 5 * np.median(np.abs(vals - plane_fit), axis=0, keepdims=True)
                weight = 1 / (1 + (np.abs(vals - plane_fit)/deviation)**2)
            vals[:] = plane_fit

    # linear interpolation along width (axis=1)
    interp = np.linspace(0, 1, width)[None, :, None]  # shape: (1, width, 1)
    ramp_vals = left_vals[:, None, :] * (1 - interp) + right_vals[:, None, :] * interp
    # ramp_vals shape now is (Nlayers, width, Nproj), matches sinogram

    # subtract ramp
    sinogram = sinogram - ramp_vals
    return sinogram


def unwrap2D_fft(phase_diff, axis, boundary=None, step=0):
    """
    Perform 2D phase unwrapping using FFT-based integration.

    Parameters:
        phase_diff (ndarray): Input phase difference or complex array.
        axis (int): Axis along which to unwrap.
        empty_region (optional): Region to remove ramp (default: None).
        step (int): Step size for gradient calculation (default: 0).

    Returns:
        tuple: (phase, phase_diff, residues)
    """
    residues = []

    # Compute residues if requested and input is complex
    if not np.isrealobj(phase_diff):
        residues = np.abs(findresidues(phase_diff)) > 0.1

    # If input is complex, compute phase gradient
    if not np.isrealobj(phase_diff):
        phase_diff = get_phase_gradient_1D(phase_diff, ax=axis, step=step)

    # Integrate to get phase
    phase = np.real(get_img_int_1D(phase_diff, axis))

    # Remove ramp if empty_region provided and axis == 2
    if boundary is not None:  # MATLAB axis=2 → Python axis=1
        if axis is not 1:
            raise ValueError("Boundary removal is only implemented for axis=1") 
        else:
            phase = remove_sinogram_ramp(phase, boundary, -1)

    return phase, phase_diff, residues

def unwrap2D_fft2(img, empty_region=None, step=0, weights=1, polyfit_order=1):
    if weights == 1:
        weights = np.ones((img.shape[0], img.shape[1], 1))
    weights = np.clip(weights, 0, 1).astype(np.float32)

    residues = np.abs(findresidues(img))*weights[1:,1:]> 0.1

    phase = 0
    Niters = 5
    for iter in np.arange(Niters):
        if iter == 0:
            img_resid = img
        else:
            img_resid = img*np.exp(-1j*phase)

            a_resid, c_factor = stabilize_phase(img_resid)
            if np.all(np.abs(weights* (a_resid) )<2):
                phase = phase + weights*(a_resid-c_factor)
                phase = remove_sinogram_ramp_3D(phase, empty_region, 1)
                break

        phase = phase + unwrap2D_fft2_iteration(img_resid, weights)
        phase = remove_sinogram_ramp_3D(phase, empty_region, 1)

    return phase.real, residues

def unwrap2D_fft2_iteration(img, weights):

    img = weights * img / (np.abs(img) + 1e-12)
    pad_y, pad_x = 64, 64
    img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0,0)), mode="symmetric")
    img = smooth_edges(img, 5, [0, 1])

    [d_X, d_Y] = get_phase_gradient_2D(img, step=0, padding=0)
    
    phase = np.real(get_img_int_2D(d_X, d_Y))

    # remove padding
    phase = phase[pad_y:-pad_y, pad_x:-pad_x]
    return phase

def get_phase_gradient_2D(img, step=0, padding=0):
    if padding > 0:
        img = np.pad(img, ((padding, padding), (padding, padding),(0,0)),
                     mode="symmetric")

        img = smooth_edges(img, padding, [1, 2])

    if step == 0:

        [d_X, d_Y] = get_img_grad(img)

        d_X = np.imag(np.conj(img) * d_X)
        d_Y = np.imag(np.conj(img) * d_Y)

    else:
        raise NotImplementedError("Finite difference method is not implemented for 2D gradients yet")

    if padding > 0:
        d_X = d_X[padding:-padding, padding:-padding]
        d_Y = d_Y[padding:-padding, padding:-padding]

    return d_X.real, d_Y.real

def get_img_int_2D(dX, dY):
    Ny, Nx, _ = dX.shape

    fD = np.fft.fft2(dX + 1j * dY, axes=(0,1))

    xgrid = np.fft.ifftshift(np.arange(-Nx//2, np.ceil(Nx/2))) / Nx
    ygrid = np.fft.ifftshift(np.arange(-Ny//2, np.ceil(Ny/2))) / Ny
    X, Y = np.meshgrid(xgrid, ygrid)
    filter = np.exp(2j * np.pi * (X + Y))
    filter = filter/ (2j * np.pi * (X + 1j * Y))

    filter[0, 0] = 1

    integral_hat = fD * filter[:, :, None]
    integral = np.fft.ifft2(integral_hat, axes=(0,1))

    return integral

def stabilize_phase(img_orig, img_ref=None, weights=None, remove_ramp=True, normalize_amplitude=False):
    if img_ref is None:
        img_ref = 1

    if weights is None:
        weights = 1

    M0, N0 = img_orig.shape[:2]


    img = img_orig.copy()

    phase_diff = img_ref * np.conj(img)

    M, N = img.shape[:2]

    xramp = np.pi * np.linspace(-1, 1, M)[:, None, None]
    yramp = np.pi * np.linspace(-1, 1, N)[None, :, None]


    x = 0
    y = 0
    c_offset = 0

    gamma = np.mean(phase_diff * weights, axis=(0,1)) / np.mean(weights)
    gamma = gamma / np.abs(gamma)

    if np.any(np.isnan(gamma)):
        gamma = 1

    gamma_x = None
    gamma_y = None

    if remove_ramp:
        phase_diff = phase_diff * np.conj(gamma)

        # linearize
        phase_diff = np.angle(phase_diff) * weights

        gamma_x = (
            np.mean(phase_diff * xramp, axis=(0,1))
            / np.mean(weights * np.abs(xramp) ** 2)
        )

        gamma_y = (
            np.mean(phase_diff * yramp, axis=(0,1))
            / np.mean(weights * np.abs(yramp) ** 2)
        )

        gamma_x = gamma_x / M
        gamma_y = gamma_y / N

        # full resolution ramps
        xramp_full = np.pi * np.linspace(-1, 1, M0)[:, None, None]
        yramp_full = np.pi * np.linspace(-1, 1, N0)[None, :, None]

        c_offset = np.angle(gamma) + xramp_full*(gamma_x*M0) + yramp_full*(gamma_y*M0)
        img_out = img_orig*np.exp(1j*c_offset)

    else:
        img_out = img_orig * gamma
        c_offset = np.angle(gamma)

    if normalize_amplitude:
        mean_amp = (
            np.mean(weights * img_out) / np.mean(weights)
        )
        img_out = img_out / mean_amp

    return img_out, c_offset


def unwrap_data(sinogram, method, boundary):
    """
    Auxiliary function to perform data unwrapping.
    See unwrap2D_fft for detailed help.

    Parameters:
        sinogram (ndarray): Input data.
        method (str): Unwrapping method ('fft_1d', 'none', 'diff').
        boundary: Boundary condition passed to unwrap2D_fft.

    Returns:
        ndarray: Unwrapped sinogram.
    """
    
    method = method.lower()

    if method == 'fft_1d':
        # Unwrap the data by FFT along slices
        sinogram = -unwrap2D_fft(sinogram, axis=1, boundary=boundary)[0]
    elif method in ['none', 'diff']:
        # Do nothing
        pass
    else:
        raise ValueError("Missing method")

    return sinogram

def imfilter_high_pass_1d(img, ax, sigma, padding=0, apply_fft=True):
    """
    Apply a high-pass filter along a given axis using FFT.
    
    Parameters:
    img : ndarray
        Input image (N-dimensional).
    ax : int
        Axis along which to apply the filter.
    sigma : float
        Filtering intensity in [0, 1]. If sigma <= 0, derivative filter is used.
    padding : int, optional
        Padding size along the axis to avoid edge artifacts (default=0).
    apply_fft : bool, optional
        If True, assume img is in real space and apply FFT (default=True).
    
    Returns:
    img : ndarray
        High-pass filtered image.
    """

    Ndims = img.ndim
    padding = int(np.ceil(padding))

    # Symmetric padding along the specified axis
    if padding > 0:
        pad_width = [(0, 0)] * Ndims
        pad_width[ax] = (padding, padding)
        img = np.pad(img, pad_width, mode='symmetric')

    Npix = img.shape
    shape = [1] * Ndims
    shape[ax] = Npix[ax]
    is_real = np.isrealobj(img)

    # FFT along the specified axis
    if apply_fft:
        img = np.fft.fft(img, axis=ax)

    # Frequency grid
    x = np.arange(-Npix[ax] // 2, Npix[ax] // 2) / Npix[ax]
    x = x.reshape(shape)

    # Adjust sigma for resolution independence
    sigma = 256 / (Npix[ax] - 2 * padding) * sigma

    # Construct spectral filter
    if sigma == 0:
        # Derivative filter
        freq = np.fft.fftshift(np.arange(Npix[ax]) / Npix[ax] - 0.5)
        spectral_filter = 2j * np.pi * freq.reshape(shape)
    else:
        spectral_filter = np.fft.fftshift(np.exp(1.0 / (-(x ** 2) / (sigma ** 2))))

    # Apply filter
    img = img * spectral_filter

    # Inverse FFT if needed
    if apply_fft:
        img = np.fft.ifft(img, axis=ax)
    if is_real:
        img = np.real(img)

    # Crop padding
    if padding > 0:
        slicer = [slice(None)] * Ndims
        slicer[ax] = slice(padding, Npix[ax] - padding)
        img = img[tuple(slicer)]

    return img

def get_img_grad_filtered_ax(img, axis, high_pass_filter, smooth_win, axes=(0, 1, 2)):
    """
    Compute filtered image gradient along specified axis.
    
    Parameters:
    img : ndarray
        Input image.
    axis : int
        Axis for gradient 
    high_pass_filter : float
        Filter strength.
    smooth_win : int
        Window size for edge smoothing.
    Parameters
        axes Specify order of [Ny, Nx, Nangles] for img. Default matlab order is [Ny, Nx, Nangles] = (0, 1, 2), default astra order is [Ny, Nangles, Nx] = (0, 2, 1)
    
    Returns:
    d_img : ndarray
        Filtered gradient image.
    """

    horizontal_axis = axes[0]
    vertical_axis = axes[1]

    

    if axis == horizontal_axis:  # Horizontal direction
        # Smooth edges to avoid jumps
        img = smooth_edges(img, smooth_win, [vertical_axis])
        is_real = np.isrealobj(img)
        Np = img.shape
        X = 2j * np.pi * (np.fft.fftshift(np.arange(Np[vertical_axis]) / Np[vertical_axis]) - 0.5)
        d_img = np.fft.fft(img, axis=vertical_axis)
        shape = [1] * img.ndim
        shape[vertical_axis] = Np[vertical_axis]
        d_img = d_img*np.broadcast_to(X.reshape(shape), Np)
        # Apply high-pass filter along horizontal direction
        d_img = imfilter_high_pass_1d(d_img, ax=vertical_axis, sigma=high_pass_filter, padding=0, apply_fft=False)
        d_img = np.fft.ifft(d_img, axis=vertical_axis)

    elif axis == vertical_axis:  # Vertical direction
        # Smooth edges to avoid jumps
        img = smooth_edges(img, smooth_win, [horizontal_axis])
        is_real = np.isrealobj(img)
        Np = img.shape
        X = 2j * np.pi * (np.fft.fftshift(np.arange(Np[horizontal_axis]) / Np[horizontal_axis]) - 0.5)
        d_img = np.fft.fft(img, axis=horizontal_axis)
        shape = [1] * img.ndim
        shape[horizontal_axis] = Np[horizontal_axis]
        d_img = d_img*np.broadcast_to(X.reshape(shape), Np)
        # Apply high-pass filter along horizontal direction
        d_img = imfilter_high_pass_1d(d_img, ax=horizontal_axis, sigma=high_pass_filter, padding=0, apply_fft=False)
        d_img = np.fft.ifft(d_img, axis=horizontal_axis)

    if is_real:
        d_img = np.real(d_img)

    return d_img

def get_img_grad_filtered(img, axis, high_pass_filter, smooth_win):
    """
    Compute filtered image gradient along specified axis.
    
    Parameters:
    img : ndarray
        Input image.
    axis : int
        Axis for gradient (1 for horizontal, 2 for vertical in MATLAB terms).
    high_pass_filter : float
        Filter strength.
    smooth_win : int
        Window size for edge smoothing.
    
    Returns:
    d_img : ndarray
        Filtered gradient image.
    """

    # Smooth edges to avoid jumps
    img = smooth_edges(img, smooth_win, [1 + (axis % 2)])
    is_real = np.isrealobj(img)
    Np = img.shape

    if axis == 0:  # Horizontal direction
        X = 2j * np.pi * (np.fft.fftshift(np.arange(Np[1]) / Np[1]) - 0.5)
        d_img = np.fft.fft(img, axis=1)
        d_img = d_img * X[np.newaxis,:,np.newaxis]  # Broadcasting works automatically
        # Apply high-pass filter along horizontal direction
        d_img = imfilter_high_pass_1d(d_img, ax=1, sigma=high_pass_filter, padding=0, apply_fft=False)
        d_img = np.fft.ifft(d_img, axis=1)

    elif axis == 1:  # Vertical direction
        X = 2j * np.pi * (np.fft.fftshift(np.arange(Np[0]) / Np[0]) - 0.5)
        d_img = np.fft.fft2(img)
        d_img = d_img * X[:, np.newaxis, np.newaxis]  # Column vector broadcasting
        # Apply high-pass filter along horizontal direction
        d_img = imfilter_high_pass_1d(d_img, ax=1, sigma=high_pass_filter, padding=0, apply_fft=False)
        d_img = np.fft.ifft2(d_img)

    if is_real:
        d_img = np.real(d_img)

    return d_img

def get_resid_sino(sinogram_model, sinogram, high_pass_filter):
    
    resid_sino = sinogram_model - sinogram
    
    # apply high pass filter to get rid of phase artefacts
    resid_sino = imfilter_high_pass_1d(resid_sino, 1, high_pass_filter)
    
    return resid_sino

def find_optimal_shift_ax(sinogram_model, sinogram, weights, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False, axes=(0, 1, 2)):
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
    resid_sino = imfilter_high_pass_1d(resid_sino, Nx_ax, high_pass_filter)
    
    if unwrap_data_method.lower() == 'none':
        resid_sino = imfilter_high_pass_1d(resid_sino, ax=Nangles_ax, sigma=high_pass_filter, padding=0)
       
    # Horizontal alignment 
    if align_horizontal:      
        dX = get_img_grad_filtered_ax(sinogram_model, axis=Ny_ax, high_pass_filter=high_pass_filter, smooth_win=5, axes=axes)
        if unwrap_data_method.lower() == 'none':
            dX = imfilter_high_pass_1d(dX, ax=Nangles_ax, sigma=high_pass_filter, padding=0)
        
        numerator = np.sum(weights * dX * resid_sino, axis=(Ny_ax, Nx_ax))
        # if np.mean(numerator) < 0.01:
        #     numerator[:] = 0
        denominator = np.sum(weights * dX**2, axis=(Ny_ax, Nx_ax)) # sum2 and mean 2????????????????
        shift_x = -numerator / denominator

    
    # Vertical alignment
    if align_vertical:
        dY = get_img_grad_filtered_ax(sinogram_model, axis=Nx_ax, high_pass_filter=high_pass_filter, smooth_win=5, axes=axes)
        if unwrap_data_method.lower() == 'none':
            dY = imfilter_high_pass_1d(dY, ax=Nangles_ax, sigma=high_pass_filter, padding=0)

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

def find_optimal_shift(sinogram_model, sinogram, weights, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False):
    
    shift_x = np.zeros(sinogram_model.shape[2], dtype=np.float32)
    shift_y = np.zeros(sinogram_model.shape[2], dtype=np.float32)
    
    resid_sino = get_resid_sino(sinogram_model, sinogram, high_pass_filter)
    
    if unwrap_data_method.lower() == 'none':
        resid_sino = imfilter_high_pass_1d(resid_sino, ax=2, sigma=high_pass_filter, padding=0)
       
    # Horizontal alignment 
    if align_horizontal:      
        dX = get_img_grad_filtered(sinogram_model, axis=0, high_pass_filter=high_pass_filter, smooth_win=5)
        if unwrap_data_method.lower() == 'none':
            dX = imfilter_high_pass_1d(dX, ax=2, sigma=high_pass_filter, padding=0)
        
        numerator = np.sum(weights * dX * resid_sino, axis=(0, 1))
        # if np.mean(numerator) < 0.01:
        #     numerator[:] = 0
        denominator = np.sum(weights * dX**2, axis=(0, 1)) # sum2 and mean 2????????????????
        shift_x = -numerator / denominator

    
    # Vertical alignment
    if align_vertical:
        dY = get_img_grad_filtered(sinogram_model, axis=1, high_pass_filter=high_pass_filter, smooth_win=5)
        if unwrap_data_method.lower() == 'none':
            dY = imfilter_high_pass_1d(dY, ax=0, sigma=high_pass_filter, padding=0)

        numerator = np.sum(weights * dY * resid_sino, axis=(0, 1))
        # if np.mean(numerator) < 0.01:
        #     numerator[:] = 0
        denominator = np.sum(weights * dY**2, axis=(0, 1))
        shift_y = -numerator / denominator
    
    # Combine shifts
    shift = np.stack([shift_x, shift_y], axis=-1)

    # Check for NaNs
    if np.isnan(shift).any():
        print("Warning: Alignment failed, estimated shift is NaN")

    # Compute error
    err = np.sqrt(np.mean((weights * resid_sino)**2, axis=(0, 1))) / MASS

    return shift, err

###################################################################################################################################################
############################################################ imshift_generic ######################################################################
###################################################################################################################################################

import numpy as np
from scipy.ndimage import gaussian_filter, zoom
from typing import Optional, Tuple

def crop_pad(img: np.ndarray, outsize, fill=0):
    """
    Adjust the size of an image by zero-padding or cropping, centered.

    Parameters:
        img (ndarray): Input image (2D or multi-dimensional).
        outsize (tuple): Desired output size (rows, cols).
        fill (scalar): Value to fill padded regions (default = 0).

    Returns:
        ndarray: Cropped or padded image.
    """
    Nin = img.shape
    if outsize is None or (Nin[0] == outsize[0] and Nin[1] == outsize[1]):
        return img  # No change needed

    Nout = outsize[:2]

    # Initialize output array with fill value
    imout_shape = (Nout[0], Nout[1]) + Nin[2:]
    imout = np.full(imout_shape, fill, dtype=img.dtype)

    # Compute centers
    center_in = np.array(Nin[:2]) // 2
    center_out = np.array(Nout) // 2

    # Compute start and end indices for input and output
    start_out = np.maximum(center_out - center_in, 0)
    end_out = np.minimum(start_out + Nin[0:2], Nout)

    start_in = np.maximum(center_in - center_out, 0)
    end_in = np.minimum(start_in + Nout, Nin[0:2])

    # Copy overlapping region
    imout[start_out[0]:end_out[0], start_out[1]:end_out[1], ...] = \
        img[start_in[0]:end_in[0], start_in[1]:end_in[1], ...]

    # Preserve complex type if needed
    if np.iscomplexobj(img):
        imout = imout.astype(complex)

    return imout

def interpolateFT(im: np.ndarray, outsize):
    """
    Interpolates a 2D image using Fourier transform (Dirichlet interpolation).
    Zero-pads or crops the Fourier spectrum to match the desired output size.

    Parameters:
        im (ndarray): Input complex or real 2D array.
        outsize (tuple): Desired output size (rows, cols).

    Returns:
        ndarray: Interpolated image (complex).
    """
    Nin = im.shape
    Nout = outsize

    # Compute centered FFT
    imFT = np.fft.fftshift(np.fft.fft2(im))

    # Crop or pad to new size
    imFT_resized = crop_pad(imFT, Nout)

    # Inverse FFT and scale
    imout = np.fft.ifft2(np.fft.ifftshift(imFT_resized)) * (Nout[0] * Nout[1]) / (Nin[0] * Nin[1])

    return imout


def interpolate_linear(img: np.ndarray, size_out, method='linear'):
    """
    Rescale a 2D image or stack of 2D images using interpolation.
    
    Parameters:
        img (ndarray): 2D image or stack (H x W x Nlayers).
        size_out (tuple): Target size (height, width).
        method (str): 'linear' (default), 'cubic', or 'nearest'.
    
    Returns:
        ndarray: Rescaled image stack.
    """
    Nx, Ny = img.shape[:2]
    Nlayers = img.shape[2] if img.ndim == 3 else 1

    # If size is unchanged, return original
    if (Nx, Ny) == tuple(size_out[:2]):
        return img

    # Map MATLAB method names to SciPy order
    method_map = {'nearest': 0, 'linear': 1, 'cubic': 3}
    order = method_map.get(method, 1)

    # Compute zoom factors
    zoom_factors = (size_out[0] / Nx, size_out[1] / Ny)

    if Nlayers > 1:
        img_out = np.zeros((int(size_out[0]), int(size_out[1]), int(Nlayers)), dtype=img.dtype)
        for i in range(Nlayers):
            img_out[:, :, i] = zoom(img[:, :, i], zoom_factors, order=order)
    else:
        img_out = zoom(img, zoom_factors, order=order)

    return img_out


def interpolateFT_centered(img: np.ndarray, Np_new, interp_sign: int):
    """
    Perform FT interpolation of a 2D or 3D stack using FFT so that the center of mass
    is preserved after resolution change. Critical for subpixel-accurate up/down sampling.

    Parameters:
        img (ndarray): Input image (2D or 3D stack).
        Np_new (tuple): Desired output size (rows, cols).
        interp_sign (int): +1 or -1, determines extra 0.5 px shift direction.

    Returns:
        ndarray: Interpolated image (complex or real).
    """
    Np = img.shape
    Np_new = (Np_new[0] + 2, Np_new[1] + 2)  # Add padding for boundary handling
    is_real = np.isrealobj(img)

    # Compute scaling factor to preserve average intensity
    scale = np.prod((np.array(Np_new)-2))/np.prod(np.array(Np)[0:2])
    downsample = int(np.ceil(np.sqrt(1 / scale)))

    # Pad symmetrically to reduce boundary artifacts
    img = np.pad(img, ((downsample, downsample), (downsample, downsample)) + ((0, 0),) * (img.ndim - 2),
                 mode='symmetric')

    # Forward FFT
    img_ft = np.fft.fft2(img, axes=(1,0))

    # Apply ±0.5 px shift in Fourier space
    img_ft = imshift_fft(img_ft, -0.5 * interp_sign, -0.5 * interp_sign, apply_fft=False)

    # Crop/pad in Fourier space
    img_ft = np.fft.ifftshift(crop_pad(np.fft.fftshift(img_ft), Np_new))

    # Apply opposite ±0.5 px shift after cropping
    img_ft = imshift_fft(img_ft, 0.5 * interp_sign, 0.5 * interp_sign, apply_fft=False)

    # Inverse FFT
    img_out = np.fft.ifft2(img_ft, axes=(1,0))

    # Scale to keep average constant
    img_out *= scale

    # Remove padding
    img_out = img_out[1:-1, 1:-1, ...]

    # Preserve real type if original was real
    if is_real:
        img_out = np.real(img_out)

    return img_out


def imshift_generic(img: np.ndarray,
                    shift: np.ndarray,
                    Npix: Optional[Tuple[int, int]] = None,
                    affine_matrix=None,  # Not implemented yet
                    smooth: int = 0,
                    ROI: Optional[Tuple[slice, slice]] = None,
                    downsample: int = 1,
                    interp_method: str = 'linear',
                    interp_sign: int = 0) -> np.ndarray:
    """
    Python equivalent of MATLAB's imshift_generic.
    
    Parameters:
        img (ndarray): 2D image or stack.
        shift (ndarray): Nx2 array of shifts.
        Npix (tuple): Target size for upsampling (None -> no upsampling).
        affine_matrix: Not implemented.
        smooth (int): Pixels to smooth around edges before shifting.
        ROI (tuple): Region of interest as slices.
        downsample (int): Downsampling factor (1 -> no downsampling).
        interp_method (str): 'linear' or 'fft'.
        interp_sign (int): Sign for FFT-based interpolation.
    
    Returns:
        ndarray: Shifted and processed image.
    """
    # Convert uint8 to float
    if img.dtype == np.uint8:
        img = img.astype(np.float32) / 255.0

    # Upsample if needed
    if Npix is not None and (img.shape[0] != Npix[0] or img.shape[1] != Npix[1]):
        if interp_method == 'linear':
            img = interpolate_linear(img, Npix)
        elif interp_method == 'fft':
            img = interpolateFT(img, Npix)

    is_real = np.isrealobj(img)

    # Apply shift if non-zero
    if np.any(shift != 0):
        smooth_axis = 2 - np.argmax(np.any(shift != 0, axis=0))
        img = smooth_edges(img, smooth, [smooth_axis])
        if img.ndim > 2:
            if interp_method == 'linear':
                img = imshift_linear(img, shift)
            elif interp_method == 'fft':
                img = imshift_fft(img, shift)

    # Crop ROI
    if ROI is not None:
        img = img[ROI[0], ROI[1], ...]

    Np = img.shape

    # Downsample with interpolation
    if downsample > 1:
        # img = gaussian_filter(img, sigma=[downsample, downsample, 0])
        img = imgaussfilt3_conv(img, filter_size=[downsample, downsample, 0])
        # Correct boundary effects
        correction = imgaussfilt3_conv(np.ones(Np[:2], dtype=img.dtype), filter_size=[downsample, downsample, 0])
        img /= correction[..., None] if img.ndim > 2 else correction
        target_size = tuple((np.ceil(np.array(Np[:2]) / downsample / 2) * 2).astype(int))
        if interp_method == 'linear':
            img = interpolate_linear(img, target_size)
        elif interp_method == 'fft':
            img = interpolateFT_centered(smooth_edges(img, 2 * downsample), target_size, interp_sign)

    if is_real:
        img = np.real(img)

    return img

def imgaussfilt3_conv(img, filter_size):
    from scipy.signal import convolve

    last_fs = None
    ker = None
    ndim = img.ndim
    for ax in range(ndim):
        fs = filter_size[min(len(filter_size)-1, ax)]
        if fs == 0:
            continue
        if last_fs != fs:
            ker = get_kernel(fs, img.dtype)
            last_fs = fs

        shape = [1] * ndim
        shape[ax] = ker.size
        ker_ax = ker.reshape(shape)

        img = convolve(img, ker_ax, mode="same")

    return img

def get_kernel(filter_size, dtype):
    grid = np.arange(-int(np.ceil(2*filter_size)),
                      int(np.ceil(2*filter_size)) + 1) / filter_size
    ker = np.exp(-grid**2)
    ker /= ker.sum()
    return ker.astype(dtype)

def xcor(in1, in2):
    from scipy import signal
    
    xcor = signal.correlate(in1, in2, mode='same')
    shift = np.argmax(xcor)-(xcor.shape[0]/2)
    return shift, xcor

def remove_linear_ramp(proj_sum):
    # auxiliary function to subtract linear ramp from sinogram 
    # it is important to avoid edge ringing and other artefacts when FFT
    # filtering is applied on the 2D array 
    
    Nlayers= proj_sum.shape[0]
    Nedge = 5; # number of averaged  edge layers 
    top = np.mean(proj_sum[0:Nedge,:],axis=0)
    bottom = np.mean(proj_sum[-Nedge:-1,:],axis=0)
    
    # xp = np.array([0,Nlayers],dtype=float)
    # fp = np.array([top,bottom])
    x = np.arange(1, Nlayers+1, dtype=float)
    
    frac = (x / float(Nlayers))[:, None]        # (Nlayers, 1) for broadcasting
    ramp = top[None, :] + (bottom - top)[None, :] * frac

    # ramp = np.interp(x,xp,fp) 
    proj_sum =  proj_sum - ramp
    
    return proj_sum

def get_y_shifts(projections):
    proj_sum = np.sum(projections,1)
    # proj_sum = remove_linear_ramp(proj_sum)
    
    shifts = np.zeros(projections.shape[-1])
    xcor_ar = np.zeros_like(proj_sum)
    a = np.gradient(proj_sum[:,0])
    for i in range(projections.shape[-1]):
        b = np.gradient(proj_sum[:,i])
        shifts[i], xcor_ar[:,i] = xcor(a,b) #
    return shifts

def apply_y_shifts(projections, trans):
    projections_shifted = np.zeros(projections.shape)
    for i in range(projections.shape[-1]):
        shift = int(trans[i])
        projections_shifted[:,:,i] = np.roll(projections[:,:,i],shift, axis=0)
    return projections_shifted

def projections_align_vertical(projections,projections_grad,weights,x0=0,xf=-1,y0=0,yf=-1):
    y_trans = get_y_shifts(projections[y0:yf,x0:xf,:])
    projections_shifted = apply_y_shifts(projections_grad, y_trans)

    min_trans = int(np.floor(np.amin(y_trans)))
    max_trans = int(np.ceil(np.amax(y_trans)))

    if min_trans < 0:
        y_to = min_trans
    else:
        y_to = -1
    
    if max_trans > 0:
        y_from = max_trans
    else:
        y_from = 0

    print("y_from", y_from)
    print("y_to", y_to)

    projections_shifted = projections_shifted[y_from:y_to,:,:]
    weights = weights[y_from:y_to,:,:]
    return projections_shifted, weights, y_trans

def find_cor(projections,first=0,last=-1):
    
    from scipy.ndimage import center_of_mass
    
    eps = np.finfo(projections.dtype).eps
    w = np.sqrt(np.maximum(0,projections)) + eps
    x = []
    
    # 0 deg
    com = center_of_mass(w[:,first,:])
    x.append(com[1])
    
    # 180 deg
    com = center_of_mass(w[:,last,:])
    x.append(com[1])
    
    cor = projections.shape[2]/2 - (x[0] + (x[1] - x[0])/2)
    
    return cor

def centering_reconstruction(rec):
    
    from scipy.ndimage import center_of_mass
    
    eps = np.finfo(rec.dtype).eps
    w = np.sqrt(np.maximum(0,rec)) + eps
    x = []
    y = []
    rec_center = np.zeros((2))
    for l in range(w.shape[0]):
        com = center_of_mass(w[l,:,:])
        x.append(com[1])
        y.append(com[0])
    mass = np.sum(w,axis=(1,2))
    rec_center[0] = np.mean(x*mass) / np.mean(mass)
    rec_center[1] = np.mean(y*mass) / np.mean(mass)
    
    return rec_center

def centering_reconstruction2(rec):    
    from scipy.ndimage import center_of_mass
    
    eps = np.finfo(rec.dtype).eps
    w = np.sqrt(np.maximum(0,rec)) + eps
    x = []
    y = []
    rec_center = np.zeros((2))
    for l in range(w.shape[2]):
        com = center_of_mass(w[:,:,l])
        x.append(com[0])
        y.append(com[1])
    mass = np.sum(w,axis=(0,1))
    rec_center[0] = np.mean(x*mass) / np.mean(mass)
    rec_center[1] = np.mean(y*mass) / np.mean(mass)
    
    return rec_center

def align_tomo_consistency_linear(sinogram, weights_find_shift, weights, theta_rad, Npix, optimal_shift, binning, 
                                  high_pass_filter = 0.0001, unwrapping = True, unwrap_data_method = 'fft_1d', max_iteration=100, shift_method = 'geometry',
                                  plot_figures = True, min_step_size = 0.05, center_reconstruction = False):
    
    import tomoconsistency_tools_hannah as tch
    import time
    import matplotlib.pyplot as plt

    print('Binning down by ', str(binning))
    
    sinogram = imshift_generic(sinogram, optimal_shift, Npix = None, affine_matrix = None, smooth = 0, 
                                          ROI = None, downsample = binning, interp_method = 'linear', interp_sign = 0)
    
    weights_find_shift = imshift_generic(weights_find_shift, optimal_shift, Npix = None, affine_matrix = None, smooth = 0, 
                                          ROI = None, downsample = binning, interp_method = 'linear', interp_sign = 0)
    
    if unwrapping:
        sinogram = unwrap_data(sinogram, 'fft_1d', boundary=None)
       
    #### tomoconsistency
    shift_history = []
    shift_history.append(optimal_shift)
        
    Nx = sinogram.shape[1]
    Ny = sinogram.shape[0]
    
    shift_total = np.zeros((sinogram.shape[-1],2))
    
    

    for ii in range(max_iteration):
        
        print('-- iteration ', str(ii))
        t0 = time.time()
    
        if shift_method == 'physical':
            # shift with imdeform_affine_fft
            sinogram_shifted = imshift_fft(sinogram, shift_total) #(sinogram, shift_total)
            sinogram_astra = sinogram_shifted.transpose((0, 2, 1))
            
            vol_geom, proj_geom = tch.init_astra(Nx, Ny, theta_rad)
            
            if plot_figures:
                plt.figure(figsize=(10,3))
                plt.subplot(121),plt.imshow(sinogram[:,Nx//2,:]), plt.title('Sinogram'), plt.colorbar()
                plt.subplot(122),plt.imshow(sinogram_shifted[:,Nx//2,:]), plt.title('Sinogram shifted'), plt.colorbar()
        
        
        elif shift_method == 'geometry':
            vol_geom, proj_geom = init_astra_vec(Nx, Ny, theta_rad, shift_total) # try applying shifts with astra vector geometry
            sinogram_astra = sinogram.transpose((0, 2, 1))
        
        cor = find_cor(sinogram_astra, first=5)
        sinogram_astra = np.roll(sinogram_astra,int(cor),axis=2)
        
        # fbp (ASTRA needs shape Ny * Nangle * Nx)
        rec = tch.FBP_astra(sinogram_astra, vol_geom, proj_geom, weights)
    
        rec = tch.apply_circular_mask(rec, 0.9)
        rec = np.maximum(0,rec)
        
        if plot_figures:
            plt.figure(figsize=(10,3))
            plt.suptitle('Shifted reconstruction')
            plt.subplot(131),plt.imshow(rec[:,:,Nx//2]), plt.xlabel('x'), plt.ylabel('y')
            plt.subplot(132),plt.imshow(rec[:,Nx//2,:]), plt.xlabel('x'), plt.ylabel('z')
            plt.subplot(133),plt.imshow(rec[Nx//2,:,:]), plt.xlabel('y'), plt.ylabel('z')
            plt.tight_layout()
        
        if center_reconstruction:
            # centering 
            rec_center = centering_reconstruction(rec)
            
            if ii == 0:
                if center_reconstruction:
                    rec_center_0 = [0,0]
                else:
                    rec_center_0 = rec_center
                
            shift_rec = -0.5*(rec_center - rec_center_0)
            
            rec = imshift_fft(rec,shift_rec[0],shift_rec[1])

        
        # get reprojection
        sinogram_model_astra = tch.get_projections(rec, vol_geom, proj_geom)
    
        sinogram_model = sinogram_model_astra.transpose((0,2,1))
        sinogram_astra = sinogram_astra.transpose((0,2,1))
        # if plot_figures:
        #     plt.figure(figsize=(10,3))
        #     plt.subplot(131),plt.imshow(sinogram[:,Nx//2,:]), plt.title('Sinogram'), plt.colorbar()
        #     plt.subplot(132),plt.imshow(sinogram_model[:,Nx//2,:]), plt.title('Sinogram model'), plt.colorbar()
        #     plt.subplot(133),plt.imshow(sinogram[:,Nx//2,:]-sinogram_model[:,Nx//2,:]), plt.title('Difference'), plt.colorbar()
        #     plt.tight_layout()
    
        MASS = np.median(sinogram * np.mean(abs(sinogram), axis=(0,1)))
    
        # MASS = 0
        
        # sinogram_model is reprojected sinogram
        # sinogram is the original sino (also called "sinogram_shifted" in the MATLAB code)
        shift_upd, err = find_optimal_shift(sinogram_model, sinogram_astra, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
        # shift_upd, err = find_optimal_shift(sinogram_model, sinogram, weights_find_shift, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False)
        step_relaxation = 0.01
        # shift_upd = np.minimum(0.5, abs(shift_upd))#*np.sign(shift_upd)*step_relaxation
        shift_total = shift_total + shift_upd
        
        shift_history.append(shift_upd * binning)
        
        max_update = np.quantile(np.abs(shift_upd[:, :]), 0.995, axis=0).max()
        print('   max update ', str(max_update * binning))
        print(f'   iteration {str(ii)} time {time.time()-t0}')
        
        if max_update * binning < min_step_size:
            break
        
    optimal_shift = optimal_shift + shift_total * binning
        
    return optimal_shift, shift_history

def init_astra_vec(Nx, Ny, theta_rad, shifts):
    # Need to add COR
    import astra
    
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

    vol_geom = astra.create_vol_geom(Nx, Nx, Ny)

    return vol_geom, proj_geom

# def center(X, use_shift=True):
#     """
#     Find center of mass of matrix X and optionally calculate variance.
    
#     Parameters
#     ----------
#     X : ndarray
#         2D array (image) or stacked images (N x M).
#     use_shift : bool, optional
#         If True, CoM is calculated relative to the image center. Default = True.
    
#     Returns
#     -------
#     pos_x : float
#         Center of mass in x-direction.
#     pos_y : float
#         Center of mass in y-direction.
#     mass : float
#         Total sum of X.
#     mu : ndarray, shape (2,)
#         [pos_x, pos_y].
#     sigma : ndarray, shape (2, 2)
#         Covariance matrix [[xx, xy], [yx, yy]].
#     """
    
#     # Ensure X is a NumPy array
#     X = np.asarray(X, dtype=float)
#     X = X.transpose((1,2,0))
#     N, M = X.shape[:2]

#     # Total mass
#     mass = np.sum(X, axis=(0,1))

#     # Coordinate grids
#     xgrid = np.arange(1, M + 1)      # 1..M
#     ygrid = np.arange(1, N + 1)      # 1..N

#     # Center of mass
#     pos_x = np.sum(np.sum(X, axis=0) * xgrid[:,np.newaxis], axis=0) / mass
#     pos_y = np.sum(np.sum(X, axis=1) * ygrid[:,np.newaxis], axis=0) / mass

#     mu = np.array([pos_x, pos_y])
    
#     # # Variance (sigma)
#     # dx = xgrid[None,:] - pos_x[:,None]
#     # dy = ygrid[None,:] - pos_y[:,None]
#     # pos_xx = np.sum(np.sum(X, axis=0) * dx**2) / mass
#     # pos_yy = np.sum(np.sum(X, axis=1) * dy**2) / mass

#     # # Cross terms: compute using matrix multiplication
#     # pos_xy = np.sum(X @ dx[:, None] * dy[None, :]) / mass
#     # pos_yx = pos_xy  # symmetric for real-valued images

#     # sigma = np.array([[pos_xx, pos_xy],
#     #                   [pos_yx, pos_yy]])

#     # Apply shift if requested
#     if use_shift:
#         pos_x -= M / 2 + 0.5
#         pos_y -= N / 2 + 0.5

#     return pos_x, pos_y, mass, mu
    
    


