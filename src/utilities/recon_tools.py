import numpy as np
import astra
import warnings

def init_astra(Nx, Ny, theta_rad):
    """
    Initialise astra projection and volume geometry
    
    Parameters
    ----------
    Nx: int
        Number of pixels in the x (horizontal) direction
    Ny: int
        Number of pixels in the y (vertical) direction
    theta_rad: ndarray
        Array of projection angles in radians
    """
    Nz = Nx
    vol_geom = astra.create_vol_geom(Nx, Nz, Ny)

    # Projection geometry (3D parallel beam)
    proj_geom = astra.create_proj_geom(
        'parallel3d',
        1,  # detector pixel size in x
        1,  # detector pixel size in y   
        Ny,          
        Nx,          
        theta_rad
    )

    return vol_geom, proj_geom

def init_astra_vec(Nx, Ny, theta_rad, shifts, rot_center_x=0, rot_center_y=0):
    """
    Initialise astra projection and volume geometry using the vector geometry definition
    The projections are defined as an array of vectors which allows non-uniform shifts to
    be defined per-projection. Input an array of x and y shifts to be applied to each 
    projection plus an additional rotation centre offset on top to be applied in x and y.
    
    Using this method we don't need to physically shift the projection images - it is used 
    in the TomoConsistencyAlignment method. 
    
    Parameters
    ----------
    Nx: int
        Number of pixels in the x (horizontal) direction
    Ny: int
        Number of pixels in the y (vertical) direction
    theta_rad: ndarray
        Array of projection angles in radians
    shifts: ndarray
        Array of shape (Nangles, 2) containing x and y shifts in pixels
    rot_center_x: float
        Rotation centre offset in x. Defined as the offset in pixels from the center 0
    rot_center_y: float
        Rotation centre offset in y. Defined as the offset in pixels from the center 0
    """

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

        vectors[i,3] -= u_x * rot_center_x + v_x * rot_center_y
        vectors[i,4] -= u_y * rot_center_x + v_y * rot_center_y
        vectors[i,5] -= u_z * rot_center_x + v_z * rot_center_y

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

    Nz = Nx

    vol_geom = astra.create_vol_geom(Nx, Nz, Ny)

    return vol_geom, proj_geom

def FBP_astra(sinogram, vol_geom, proj_geom, weights):
    """
    Custom astra FBP call. 
    Applies a weighted ramp filter to each projection in the sinogram via FFT, then
    performs back projection using ASTRA's BP3D_CUDA algorithm.

    Parameters
    ----------
    sinogram : np.ndarray, shape (Ny, Nangles, Nx)
        3D sinogram array where Ny is the number of detector rows, Nangles is the
        number of projection angles, and Nx is the number of detector columns.
    vol_geom : dict
        ASTRA volume geometry dictionary defining the reconstruction volume,
        as created by astra.create_vol_geom().
    proj_geom : dict
        ASTRA projection geometry dictionary defining the acquisition geometry,
        as created by astra.create_proj_geom().
    weights : np.ndarray, shape (Nangles,)
        Per-angle scaling factors applied to the ramp filter in the frequency domain.
        Can encode angular sampling intervals or other angle-dependent corrections

    Returns
    -------
    rec : np.ndarray
        3D reconstructed volume retrieved from ASTRA, with shape determined by vol_geom.

    """

    
    Ny, Nangles, Nx = sinogram.shape

    Fsize = max(64, 2**int(np.ceil(np.log2(2*Nx))))
    pad_left = (Fsize - Nx) // 2
    pad_right = Fsize - Nx - pad_left
    freqs = np.fft.fftfreq(Fsize)
    ramp = np.abs(freqs)

    filtered = np.zeros_like(sinogram)
    block_size = 2048
    for start in range(0, Ny, block_size):
        end = min(start + block_size, Ny)

        for iy in range(start, end):
            for ia in range(Nangles):
                proj = sinogram[iy, ia, :]
                proj_padded = np.pad(proj, (pad_left,pad_right), mode='constant')

                F = np.fft.fft(proj_padded)
                F *= ramp * weights[ia]
                filtered_proj = np.real(np.fft.ifft(F))

                filtered[iy, ia, :] = filtered_proj[pad_left:pad_left+Nx]
            
    filtered = filtered.astype(np.float32)

    sino_id = astra.data3d.create('-proj3d', proj_geom, filtered)
    vol_id  = astra.data3d.create('-vol',  vol_geom)

    cfg = astra.astra_dict('BP3D_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['ReconstructionDataId'] = vol_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    rec = astra.data3d.get(vol_id)

    astra.algorithm.delete(alg_id)
    astra.data3d.delete(sino_id)
    astra.data3d.delete(vol_id)

    return rec

def get_projections(volume, vol_geom, proj_geom):
    """
    Compute 3D forward projections of a volume using ASTRA's FP3D_CUDA projector.

    Parameters
    ----------
    volume : np.ndarray
        3D volume array to project, with shape consistent with vol_geom.
    vol_geom : dict
        ASTRA volume geometry dictionary defining the volume dimensions and voxel
        size, as created by astra.create_vol_geom().
    proj_geom : dict
        ASTRA projection geometry dictionary defining the acquisition geometry,
        as created by astra.create_proj_geom().

    Returns
    -------
    sino : np.ndarray, shape (Ny, Nangles, Nx)
        3D sinogram of forward projections, where Ny is the number of detector
        rows, Nangles is the number of projection angles, and Nx is the number
        of detector columns.
    """

    vol_id = astra.data3d.create('-vol', vol_geom, volume)

    sino_id = astra.data3d.create('-proj3d', proj_geom)

    cfg = astra.astra_dict('FP3D_CUDA')
    cfg['ProjectionDataId'] = sino_id
    cfg['VolumeDataId'] = vol_id

    alg_id = astra.algorithm.create(cfg)
    astra.algorithm.run(alg_id)

    sino = astra.data3d.get(sino_id)

    # Cleanup
    astra.algorithm.delete(alg_id)
    astra.data3d.delete(vol_id)
    astra.data3d.delete(sino_id)

    return sino

def apply_circular_mask(rec, radius=0.99, apodize=True):
    """
    Apply a circular mask (with optional apodization) to a 2D or 3D reconstruction.

    Parameters
    ----------
    rec : np.ndarray
        2D array of shape (H, W) or 3D array of shape (Nz, H, W) to be masked.
    radius : float, optional
        Fractional radius of the circular mask relative to half the largest
        image dimension. Only used when apodize=False. Default is 0.99.
    apodize : bool, optional
        If True, use apply_3D_apodization() to generate the mask with default
        radial smoothing. If False, build a smooth sinusoidal roll-off mask
        controlled by radius. Default is True.

    Returns
    -------
    masked : np.ndarray
        Array of the same shape as rec with the circular mask applied.
    """

    if rec.ndim == 2:
        H, W = rec.shape
        rec_reshaped = rec[None, ...] 
    elif rec.ndim == 3:
        _, H, W = rec.shape
        rec_reshaped = rec
    else:
        raise ValueError("rec must be 2D or 3D numpy array")

    if apodize == True:
        tomogram = np.zeros((H, W, 1), dtype=np.float32)
        _, mask = apply_3D_apodization(tomogram, rad_apod=0, axial_apod=0, radial_smooth=5)
    else:
        y_range = (H - 1) / 2
        x_range = (W - 1) / 2

        Y, X = np.ogrid[-y_range:y_range+1, -x_range:x_range+1]

        dist = np.sqrt(X**2 + Y**2)

        max_dim = max(H, W)
        radius_applied = radius * (max_dim / 2)

        r = np.sqrt(1 / np.pi)

        mask = (radius_applied - dist).clip(-r, r)
        mask *= (0.5 * np.pi) / r
        mask = np.sin(mask)
        mask = 0.5 + 0.5 * mask  

    masked = rec_reshaped * mask  

    if rec.ndim == 2:
        return masked[0]
    else:
        return masked
    
def apply_3D_apodization(tomogram, rad_apod=None, axial_apod=None, radial_smooth=None):
    """
    Apply radial and/or axial apodization to a 3D tomogram to suppress edge artefacts.

    Parameters
    ----------
    tomogram : np.ndarray, shape (rows, cols, layers)
        3D array to apodize.
    rad_apod : float or None, optional
        Radius (in pixels) of the unapodized central disc. Pixels beyond
        (min(rows, cols)/2 - rad_apod - radial_smooth) are smoothly tapered
        to zero. If None, no radial apodization is applied.
    axial_apod : float or None, optional
        Half-width (in layers) of the unapodized axial region. Layers outside
        this range are tapered using a fractional Hanning window. If None,
        no axial apodization is applied.
    radial_smooth : float or None, optional
        Transition width (in pixels) for the radial taper. Defaults to
        min(rows, cols) / 10 if not provided.

    Returns
    -------
    out : np.ndarray, shape (rows, cols, layers)
        Apodized tomogram.
    circulo : np.ndarray or None, shape (rows, cols)
        2D radial mask applied to each layer, or None if rad_apod is None.

    """
    if tomogram.ndim != 3:
        raise ValueError("tomogram must have shape (rows, cols, layers)")
    rows, cols, layers = tomogram.shape
    if radial_smooth is None:
        radial_smooth = min(rows, cols) / 10.0
    circulo = None
    out = tomogram
    if rad_apod is not None:
        yt = np.arange(-np.floor(rows / 2.0), np.ceil(rows / 2.0), dtype=np.float32)
        xt = np.arange(-np.floor(cols / 2.0), np.ceil(cols / 2.0), dtype=np.float32)
        Y, X = np.meshgrid(yt, xt, indexing='ij')
        tappix = max(float(radial_smooth), 1.0)
        half_min = min(rows, cols) / 2.0
        zerorad = int(round(half_min - float(rad_apod) - tappix))
        taperfunc = radtap(X, Y, tappix, zerorad)
        circulo = np.float32(1.0 - taperfunc)
        out = out * circulo[:, :, np.newaxis]
    if axial_apod is not None and layers > 1:
        pad = max(0, int(round(layers - 2.0 * float(axial_apod))))
        filters = fract_hanning_pad(layers, layers, pad)
        filt = np.fft.ifftshift(filters[:, 0])
        out = out * filt[np.newaxis, np.newaxis, :]
    return out, circulo

def radtap(X, Y, tappix, zerorad):
    """
    Compute a 2D radial taper (roll-off) function on a meshgrid.

    Produces a smooth transition from 0 (inside) to 1 (outside) using a
    raised cosine over a transition annulus of width 2 * tappix.

    Parameters
    ----------
    X : np.ndarray
        2D array of x-coordinates, typically from np.meshgrid.
    Y : np.ndarray
        2D array of y-coordinates, typically from np.meshgrid.
    tappix : float
        Half-width (in pixels) of the cosine transition zone.
    zerorad : int
        Inner radius (in pixels) below which the taper is zero.

    Returns
    -------
    out : np.ndarray, shape matching X and Y
        Float32 taper array with values in [0, 1]; 0 inside zerorad,
        1 outside the transition zone, and a smooth cosine roll-off in between.
    """
    tau = 2.0 * float(tappix)
    R = np.sqrt(X**2 + Y**2, dtype=np.float32)
    with np.errstate(invalid='ignore'):
        taper = 0.5 * (1.0 + np.cos(2.0 * np.pi * (R - zerorad - tau / 2.0) / tau))
    out = np.zeros_like(R, dtype=np.float32)
    mask_transition = R <= (zerorad + tau / 2.0)
    out[mask_transition] = taper[mask_transition]
    out[R > (zerorad + tau / 2.0)] = 1.0
    out[R < zerorad] = 0.0
    return out

def fract_hanning_pad(outputdim, filterdim=None, unmodsize=None):
    """
    Construct a 2D fractional Hanning window, zero-padded to a specified output size.

    The window has a central flat (unmodulated) region of size unmodsize and a
    raised-cosine taper towards the edges within the filterdim x filterdim support,
    embedded in a zero-padded output of size outputdim x outputdim.

    Parameters
    ----------
    outputdim : int
        Size of the square output array.
    filterdim : int or None, optional
        Size of the Hanning window support. Must be <= outputdim.
        If None, both filterdim and unmodsize must be None, in which case
        filterdim defaults to outputdim and unmodsize to 0.
    unmodsize : int or None, optional
        Number of central samples left unmodulated (flat top). Must be
        >= 0 and <= filterdim. If 0, a standard 2D Hanning window is used.

    Returns
    -------
    out : np.ndarray, shape (outputdim, outputdim)
        float32 array containing the fftshifted fractional Hanning window
        embedded in zero-padding.

    """
    if filterdim is None and unmodsize is None:
        filterdim = outputdim
        unmodsize = 0
    elif filterdim is None or unmodsize is None:
        raise ValueError("Provide either only outputdim, or outputdim+filterdim+unmodsize.")
    outputdim = int(outputdim)
    filterdim = int(filterdim)
    unmodsize = int(unmodsize)
    if outputdim < unmodsize:
        raise ValueError("Output dimension must be smaller or equal to size of unmodulated window")
    if outputdim < filterdim:
        raise ValueError("Filter cannot be larger than output size")
    if unmodsize < 0:
        warnings.warn("Specified unmodsize < 0, setting unmodsize = 0")
        unmodsize = 0
    fd = filterdim
    N = np.arange(fd, dtype=np.float32)
    Nc, Nr = np.meshgrid(N, N, indexing='ij')
    if unmodsize == 0:
        inner = ((1.0 + np.cos(2.0 * np.pi * Nc / fd)) *
                (1.0 + np.cos(2.0 * np.pi * Nr / fd))) / 4.0
    else:
        s = int(np.floor((unmodsize - 1) / 2.0))
        denom = float(fd + 1 - unmodsize)
        out_cols = 0.5 * (1.0 + np.cos(2.0 * np.pi * (Nc - s) / denom))
        if s > 0:
            out_cols[:, :s] = 1.0
        start_right_0b = s + fd + 2 - unmodsize
        if start_right_0b < fd:
            out_cols[:, start_right_0b:] = 1.0
        out_rows = 0.5 * (1.0 + np.cos(2.0 * np.pi * (Nr - s) / denom))
        if s > 0:
            out_rows[:s, :] = 1.0
        if start_right_0b < fd:
            out_rows[start_right_0b:, :] = 1.0
        inner = out_cols * out_rows
    inner = inner.astype(np.float32)
    out = np.zeros((outputdim, outputdim), dtype=np.float32)
    start_1b = int(np.round(outputdim / 2.0 + 1.0 - fd / 2.0))
    end_1b   = int(np.round(outputdim / 2.0 + 1.0 + fd / 2.0 - 1.0))
    start_0b = start_1b - 1
    end_excl = end_1b
    out[start_0b:end_excl, start_0b:end_excl] = np.fft.fftshift(inner)
    out = np.fft.fftshift(out)
    return out