import numpy as np
from scipy.ndimage import zoom, map_coordinates
from scipy.signal import convolve
from typing import Optional, Tuple
from . import sino_tools

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
        img = sino_tools.smooth_edges(img, smooth, [smooth_axis])
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
            img = interpolateFT_centered(sino_tools.smooth_edges(img, 2 * downsample), target_size, interp_sign)

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

def imgaussfilt3_conv(img, filter_size):

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

def get_kernel(filter_size, dtype):
    grid = np.arange(-int(np.ceil(2*filter_size)),
                      int(np.ceil(2*filter_size)) + 1) / filter_size
    ker = np.exp(-grid**2)
    ker /= ker.sum()
    return ker.astype(dtype)

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