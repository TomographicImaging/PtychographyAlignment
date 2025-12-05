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
    
    axis = np.array(axis) # check if this works
    
    if axis is None or 2 in axis:
        # Compute frequency vector for X-axis
        X = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[1]//2, np.ceil(Np[1]/2))) / Np[1]
        # Apply partial FFT and multiply by frequency
        dX = fft_partial(img, 1, 1, split, False) * X
        # Apply inverse partial FFT
        dX = fft_partial(dX, 1, 1, split, True)
        if is_real:
            dX = np.real(dX)

    if axis is not None and (1 in axis or dX is None):
        # Compute frequency vector for Y-axis
        Y = 2j * np.pi * np.fft.ifftshift(np.arange(-Np[0]//2, np.ceil(Np[0]/2))) / Np[0]
        # Apply partial FFT and multiply by frequency
        dY = fft_partial(img, 0, 2, split, False) * Y[:, np.newaxis, np.newaxis]
        # Apply inverse partial FFT
        dY = fft_partial(dY, 0, 2, split, True)
        if is_real:
            dY = np.real(dY)
        if axis is None or axis.size == 1:
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
    Ng = [1, Np[1], 1]
    shift = np.full(Np, x)
    xgrid = np.fft.ifftshift(np.arange(-Np[1] // 2, int(np.ceil(Np[1] / 2)))) / Np[1]
    X = np.exp(-2j * np.pi * np.reshape(shift, Np) * np.reshape(xgrid, Ng))
    img = img * X

    # Compute phase shift for y-axis
    Ng = [Np[0], 1, 1]
    shift = np.full(Np, y)
    ygrid = np.fft.ifftshift(np.arange(-Np[0] // 2, int(np.ceil(Np[0] / 2)))) / Np[0]
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
        xgrid = np.fft.ifftshift(np.arange(-Np[1] // 2 + 1, int(np.ceil(Np[1] / 2)))) / Np[1]

        # Integration filter
        X = np.exp(2j * np.pi * xgrid)
        X = X / (2j * np.pi * xgrid)
        X[0] = 0  # Avoid division by zero

        # Apply filter and inverse FFT
        integer = grad_array_fft * X[np.newaxis, :, np.newaxis]
        integer = np.fft.ifft(integer, axis=1)

    elif ax == 0:  # MATLAB axis=1 → Python axis=0
        grad_array_fft = np.fft.fft(grad_array, axis=0)
        ygrid = np.fft.ifftshift(np.arange(-Np[0] // 2, int(np.ceil(Np[0] / 2)))) / Np[0]

        # Integration filter
        Y = np.exp(2j * np.pi * ygrid)
        Y = Y / (2j * np.pi * ygrid)
        Y[0] = 0

        # Apply filter and inverse FFT
        integer = grad_array_fft * Y[:, np.newaxis, np.newaxis]
        integer = np.fft.ifft(integer, axis=0)

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
    if boundary is not None and axis == 1:  # MATLAB axis=2 → Python axis=1
        phase = remove_sinogram_ramp(phase, boundary, -1)

    return phase, phase_diff, residues


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

def find_optimal_shift(sinogram_model, sinogram, weights, MASS, high_pass_filter, unwrap_data_method, align_horizontal=True, align_vertical=False):
    
    shift_x = np.zeros(sinogram_model.shape[2], dtype=np.float32)
    shift_y = np.zeros(sinogram_model.shape[2], dtype=np.float32)
    
    resid_sino = get_resid_sino(sinogram_model, sinogram, high_pass_filter)
    
    if unwrap_data_method.lower() == 'none':
        resid_sino = imfilter_high_pass_1d(resid_sino, axis=2, high_pass_filter=high_pass_filter, pad=0)
       
    # Horizontal alignment 
    if align_horizontal:      
        dX = get_img_grad_filtered(sinogram_model, axis=0, high_pass_filter=high_pass_filter, smooth_win=5)
        if unwrap_data_method.lower() == 'none':
            dX = imfilter_high_pass_1d(dX, axis=1, high_pass_filter=high_pass_filter, pad=0)
        
        numerator = np.sum(weights * dX * resid_sino, axis=(0, 1))
        if np.mean(numerator) < 0.01:
            numerator[:] = 0
        denominator = np.sum(weights * dX**2, axis=(0, 1)) # sum2 and mean 2????????????????
        shift_x = -numerator / denominator

    
    # Vertical alignment
    if align_vertical:
        dY = get_img_grad_filtered(sinogram_model, axis=1, high_pass_filter=high_pass_filter, smooth_win=5)
        if unwrap_data_method.lower() == 'none':
            dY = imfilter_high_pass_1d(dY, axis=0, high_pass_filter=high_pass_filter, pad=0)

        numerator = np.sum(weights * dY * resid_sino, axis=(0, 1))
        if np.mean(numerator) < 0.01:
            numerator[:] = 0
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
    scale = (Np_new[0] * Np_new[1]) / (Np[0] * Np[1])
    downsample = int(np.ceil(np.sqrt(1 / scale)))

    # Pad symmetrically to reduce boundary artifacts
    img = np.pad(img, ((downsample, downsample), (downsample, downsample)) + ((0, 0),) * (img.ndim - 2),
                 mode='symmetric')

    # Forward FFT
    img_ft = np.fft.fft2(img)

    # Apply ±0.5 px shift in Fourier space
    img_ft = imshift_fft(img_ft, -0.5 * interp_sign, -0.5 * interp_sign, fourier=True)

    # Crop/pad in Fourier space
    img_ft = np.fft.ifftshift(crop_pad(np.fft.fftshift(img_ft), Np_new))

    # Apply opposite ±0.5 px shift after cropping
    img_ft = imshift_fft(img_ft, 0.5 * interp_sign, 0.5 * interp_sign, fourier=True)

    # Inverse FFT
    img_out = np.fft.ifft2(img_ft)

    # Scale to keep average constant
    img_out *= scale

    # Remove padding
    img_out = img_out[downsample:-downsample, downsample:-downsample, ...]

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
        img = smooth_edges(img, smooth, smooth_axis)
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
        img = gaussian_filter(img, sigma=[downsample, downsample, 0])
        # Correct boundary effects
        correction = gaussian_filter(np.ones(Np[:2], dtype=img.dtype), sigma=[downsample, downsample])
        img /= correction[..., None] if img.ndim > 2 else correction
        target_size = tuple(np.ceil(np.array(Np[:2]) / downsample / 2) * 2)
        if interp_method == 'linear':
            img = interpolate_linear(img, target_size)
        elif interp_method == 'fft':
            img = interpolateFT_centered(smooth_edges(img, 2 * downsample), target_size, interp_sign)

    if is_real:
        img = np.real(img)

    return img




