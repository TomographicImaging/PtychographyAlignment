import numpy as np
from . import shift_tools, sino_tools

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
    
    if axis is None:
        axis = np.array((0,1))
    else:
        axis = np.array(axis) 

    if 1 in axis:
        # Compute frequency vector for X-axis
        X = 2j * np.pi * np.fft.ifftshift(np.arange(-(Np[1]//2), np.ceil(Np[1]/2))) / Np[1]
        # Apply partial FFT and multiply by frequency
        dX = shift_tools.fft_partial(img, 1, 1, split, False)
        shape = [1] * img.ndim
        shape[1] = Np[1]
        dX = dX * np.broadcast_to(X.reshape(shape),Np)
        # Apply inverse partial FFT
        dX = shift_tools.fft_partial(dX, 1, 1, split, True)
        if is_real:
            dX = np.real(dX)

    if 0 in axis:
        # Compute frequency vector for Y-axis
        Y = 2j * np.pi * np.fft.ifftshift(np.arange(-(Np[0]//2), np.ceil(Np[0]/2))) / Np[0]
        # Apply partial FFT and multiply by frequency
        dY = shift_tools.fft_partial(img, 0, 2, split, False)
        shape = [1] * img.ndim
        shape[0] = Np[0]
        dY = dY * np.broadcast_to(Y.reshape(shape),Np)
        # Apply inverse partial FFT
        dY = shift_tools.fft_partial(dY, 0, 2, split, True)
        if is_real:
            dY = np.real(dY)
            
    if dX is None:
        dX = dY


    return dX, dY

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
    img = sino_tools.smooth_edges(img, smooth_win, [1 + (axis % 2)])
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
        img = sino_tools.smooth_edges(img, smooth_win, [vertical_axis])
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
        img = sino_tools.smooth_edges(img, smooth_win, [horizontal_axis])
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
    
    if np.isreal(img).all():
        img = np.exp(1j*img)

    # np.testing.assert_array_less(0, step, err_msg='Difference step has to be > 0') # it should be less or equal, but I couldn't find the right np.testing.assert
    
    # suppress edge issues if phase ramp is not subtracted / there is no
    # air around sample 
    
    pad_distance = 8
    if ax == 1:
        pad_widths = [0,pad_distance,0]
    elif ax == 0:
        pad_widths = [pad_distance,0,0]
    elif ax == 2:
        pad_widths = [0,0,pad_distance]
    pad_config = [(w,w) for w in pad_widths]
    img = np.pad(img,pad_config,mode = 'symmetric')
    
    img = sino_tools.smooth_edges(img, pad_distance, [ax]) # this is from their utils

    if step == 0:
        # analytic formula (sensitive to noise) but faster 
        img = img / (abs(img) + np.finfo(float).eps)
        d_img = get_img_grad(img, ax)[0] # img is assumed to be complex 
        d_img = np.imag(np.conj(img)*d_img)
    else:
        d_img = np.angle(shift_tools.imshift_fft_ax(img,-step+shift,ax) * np.conj(shift_tools.imshift_fft_ax(img,step+shift,ax)))/(2*step)
    
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
        if axis != 1:
            raise ValueError("Boundary removal is only implemented for axis=1") 
        else:
            phase = remove_sinogram_ramp(phase, boundary, -1)

    return phase, phase_diff, residues

def unwrap2D_fft2(img, empty_region=None, step=0, weights=1, polyfit_order=1):
    """
    Iterative 2D phase unwrapping using FFT-based integration.

    Unwraps the phase of a complex image over up to 5 iterations, removing
    residual ramps after each step. Convergence is checked by testing whether
    the weighted residual phase is everywhere smaller than 2 radians.

    Parameters
    ----------
    img : np.ndarray, shape (Ny, Nx, 1) or (Ny, Nx, Nz)
        Complex input image(s) whose phase is to be unwrapped.
    empty_region : array-like or None, optional
        Region descriptor passed to remove_sinogram_ramp_3D to identify
        background pixels used for ramp removal. If None, no region is used.
    step : int, optional
        Gradient computation method. Currently only 0 (analytic) is supported.
        Default is 0.
    weights : np.ndarray or int, optional
        Float weight map of shape (Ny, Nx, 1), clipped to [0, 1]. Pixels with
        low weight (e.g. low amplitude) contribute less to the unwrapping.
        Default is 1 (uniform weights).
    polyfit_order : int, optional
        Order of polynomial used in ramp removal. Default is 1.

    Returns
    -------
    phase : np.ndarray, shape (Ny, Nx, ...)
        Unwrapped phase in radians (real part).
    residues : np.ndarray of bool, shape (Ny-1, Nx-1, ...)
        Boolean map of phase residues (locations where unwrapping is
        ambiguous), thresholded at 0.1 after weighting.
    """
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
    """
    Single iteration of FFT-based 2D phase unwrapping.

    Normalises the complex image by its amplitude, symmetrically pads it to
    reduce edge discontinuities, computes the phase gradient, and integrates
    back to obtain the unwrapped phase.

    Parameters
    ----------
    img : np.ndarray, shape (Ny, Nx, 1) or (Ny, Nx, Nz)
        Complex input image(s) to unwrap in this iteration.
    weights : np.ndarray, shape (Ny, Nx, 1)
        Weight map in [0, 1] applied to the normalised image before gradient
        computation.

    Returns
    -------
    phase : np.ndarray, shape (Ny, Nx, ...)
        Real unwrapped phase estimate for this iteration, with padding removed.
    """


    img = weights * img / (np.abs(img) + 1e-12)
    pad_y, pad_x = 64, 64
    img = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0,0)), mode="symmetric")
    img = sino_tools.smooth_edges(img, 5, [0, 1])

    [d_X, d_Y] = get_phase_gradient_2D(img, step=0, padding=0)
    
    phase = np.real(get_img_int_2D(d_X, d_Y))

    # remove padding
    phase = phase[pad_y:-pad_y, pad_x:-pad_x]
    return phase

def get_phase_gradient_2D(img, step=0, padding=0):
    """
    Compute the 2D phase gradient of a complex image.

    Calculates the x and y phase gradient components as the imaginary part of
    conj(img) * grad(img), optionally with symmetric padding and edge smoothing
    to reduce boundary artefacts.

    Parameters
    ----------
    img : np.ndarray, shape (Ny, Nx, 1) or (Ny, Nx, Nz)
        Complex input image(s).
    step : int, optional
        Gradient method. 0 uses the analytic (FFT-based) gradient via
        get_img_grad. Other values raise NotImplementedError. Default is 0.
    padding : int, optional
        Number of pixels to symmetrically pad the image on each side before
        computing gradients. Padding is removed before returning. Default is 0.

    Returns
    -------
    d_X : np.ndarray, shape (Ny, Nx, ...)
        Real x-component of the phase gradient.
    d_Y : np.ndarray, shape (Ny, Nx, ...)
        Real y-component of the phase gradient.

    Raises
    ------
    NotImplementedError
        If step != 0.
    """
    if padding > 0:
        img = np.pad(img, ((padding, padding), (padding, padding),(0,0)),
                     mode="symmetric")

        img = sino_tools.smooth_edges(img, padding, [1, 2])

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
    """
    Integrate 2D phase gradients to recover the phase via FFT.

    Treats (dX + i*dY) as the complex gradient of an analytic signal and
    recovers the integral using a frequency-domain filter derived from the
    finite-difference shift theorem.

    Parameters
    ----------
    dX : np.ndarray, shape (Ny, Nx, Nchannels)
        x-component of the phase gradient.
    dY : np.ndarray, shape (Ny, Nx, Nchannels)
        y-component of the phase gradient.

    Returns
    -------
    integral : np.ndarray, shape (Ny, Nx, Nchannels), complex
        Integrated phase. Take the real part for the unwrapped phase.

    """
    Ny, Nx, _ = dX.shape

    fD = np.fft.fft2(dX + 1j * dY, axes=(0,1))

    xgrid = np.fft.ifftshift(np.arange(-(Nx//2), np.ceil(Nx/2))) / Nx
    ygrid = np.fft.ifftshift(np.arange(-(Ny//2), np.ceil(Ny/2))) / Ny
    X, Y = np.meshgrid(xgrid, ygrid)
    filter = np.exp(2j * np.pi * (X + Y))
    filter = filter/ (2j * np.pi * (X + 1j * Y))

    filter[0, 0] = 1

    integral_hat = fD * filter[:, :, None]
    integral = np.fft.ifft2(integral_hat, axes=(0,1))

    return integral

def stabilize_phase(img_orig, img_ref=None, weights=None, remove_ramp=True, normalize_amplitude=False):
    """
    Align the global phase (and optionally linear ramp) of a complex image to a reference.

    Estimates and removes a constant phase offset and, if requested, linear
    x/y phase ramps by minimising the weighted phase difference between
    img_orig and img_ref.

    Parameters
    ----------
    img_orig : np.ndarray, shape (Ny, Nx, ...), complex
        Input complex image to be stabilized.
    img_ref : np.ndarray or None, optional
        Complex reference image. If None, a uniform reference of 1 is used,
        effectively just removing the mean phase. Default is None.
    weights : np.ndarray or None, optional
        Real weight map for the phase difference estimation. If None, uniform
        weights of 1 are used. Default is None.
    remove_ramp : bool, optional
        If True, estimate and remove a linear (x + y) phase ramp in addition
        to the constant offset. Default is True.
    normalize_amplitude : bool, optional
        If True, normalise the amplitude of the output image to 1 after phase
        correction. Default is False.

    Returns
    -------
    img_out : np.ndarray, same shape and dtype as img_orig
        Phase-stabilized complex image.
    c_offset : np.ndarray or float
        Phase correction applied to img_orig; a scalar if remove_ramp=False,
        or an array of shape (Ny, Nx, 1) containing the full ramp + offset if
        remove_ramp=True.
    """
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

