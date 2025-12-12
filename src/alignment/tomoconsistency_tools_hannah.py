import numpy as np
import astra
import matplotlib.pyplot as plt
from numpy.fft import fft2, ifft2, fft, ifft, ifftshift
import warnings

def init_astra(Nx, Ny, angles ):
    # TO-DO: add lamino angles, tilt angles, pixel scaling, rotation centre, skewness

    # Volume
    Nz = Ny
    vol_geom = astra.create_vol_geom(Nx, Ny, Nz)

    # Projection geometry (3D parallel beam)
    proj_geom = astra.create_proj_geom(
        'parallel3d',
        1.0,  # detector pixel size in x
        1.0,  # detector pixel size in y   
        Ny,          
        Nx,          
        angles
    )

    return vol_geom, proj_geom

def init_astra_vec(Nx, Ny, theta_rad, shifts):
    # Need to add COR

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

    vol_geom = astra.create_vol_geom(Nx, Ny, Ny)

    return vol_geom, proj_geom

def ram_lak_filter(N, d=1.0):
    """
    Ram-Lak filter for FBP
    N : number of detector pixels (projection width)
    d : frequency scaling (default=1.0)
    Returns a 1D filter of length N
    """
    # Next power of 2 for FFT efficiency
    n = max(64, 2**int(np.ceil(np.log2(2*N))))
    
    # Ramp filter
    filt = np.zeros(n//2 + 1)
    filt[1:] = 2 * np.arange(1, n//2 + 1) / n
    
    # Symmetrize for real FFT
    filt = np.concatenate([filt, filt[-2:0:-1]])
    
    return filt

def FBP_astra(sinogram, vol_geom, proj_geom, weights):
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

def make_grid(N):
    start = -int(np.floor(N/2))
    end   = int(np.ceil(N/2)) - 1
    return ifftshift(np.arange(start, end + 1)) / N


def imshift_fft(img, shifts, apply_fft=True):
    """
    img:    (Ny, Nx, Nangles)
    shifts: (Nangles, 2)  # [dx, dy] for each angle
    """

    Ny, Nx, Nang = img.shape
    dx = shifts[:, 0]
    dy = shifts[:, 1]

    if np.all(dx == 0) and np.all(dy == 0):
        return img.copy()

    real_input = np.isrealobj(img)

    if apply_fft:
        F = fft2(img, axes=(0, 1))
    else:
        F = img.copy()

    xgrid = make_grid(Nx)
    ygrid = make_grid(Ny)

    phase_x = np.exp(-2j * np.pi * xgrid[None, :, None] * dx[None, None, :])
    phase_y = np.exp(-2j * np.pi * ygrid[:, None, None] * dy[None, None, :])
    phase = phase_x * phase_y

    F_shifted = F * phase

    if apply_fft:
        out = ifft2(F_shifted, axes=(0, 1))
    else:
        out = F_shifted

    if real_input:
        out = out.real

    return out


def resample_fft_1d(F, scale):
    n = F.shape[0]
    new_n = int(round(n * scale))

    if new_n == n:
        return F

    if new_n > n:
        pad = (new_n - n) // 2
        return np.pad(F, ((pad, pad),), mode='constant')
    else:
        cut = (n - new_n) // 2
        return F[cut:cut + new_n]


def imdeform_affine_fft(img, affine_matrix=None, shift=None):
    out = img.copy()


    if shift is not None:
        out = imshift_fft(out, shift)


    if affine_matrix is not None:

        if affine_matrix.shape == (2, 2):
            A = affine_matrix
            U, S, Vt = np.linalg.svd(A)
            sx, sy = S
            rotation = np.arctan2(U[1, 0], U[0, 0])
            shear = Vt[0, 1] / Vt[0, 0]
        else:
            sx, sy = affine_matrix[:, 0]
            shear = affine_matrix[:, 3]
            rotation = affine_matrix[:, 2]

        if abs(sx - 1) > 1e-5 or abs(sy - 1) > 1e-5:

            F = np.fft.fft2(out)

            Fy = np.fft.fft(out, axis=0)
            Fx = np.fft.fft(out, axis=1)

            Fy2 = np.apply_along_axis(resample_fft_1d, 0, Fy, sy)
            Fx2 = np.apply_along_axis(resample_fft_1d, 1, Fx, sx)

            out = np.real(np.fft.ifft2(Fx2))

        if abs(shear) > 1e-5:
            ny, nx = out.shape
            x = np.arange(nx)

            F = fft(out, axis=1)
            phase = np.exp(-2j * np.pi * shear * x / nx)
            out = np.real(ifft(F * phase, axis=1))

        if abs(rotation) > 1e-5:
            t = np.tan(rotation / 2)

            ny, nx = out.shape
            x = np.arange(nx)
            F = fft(out, axis=1)
            phase = np.exp(-2j * np.pi * t * x / nx)
            out = np.real(ifft(F * phase, axis=1))

            y = np.arange(ny)
            F = fft(out, axis=0)
            phase = np.exp(-2j * np.pi * np.sin(rotation) * y / ny)[:, None]
            out = np.real(ifft(F * phase, axis=0))

            F = fft(out, axis=1)
            phase = np.exp(-2j * np.pi * t * x / nx)
            out = np.real(ifft(F * phase, axis=1))

    return out


from scipy.ndimage import uniform_filter1d

def plot_alignment(rec, sinogram_shifted, weights_shifted, err,
                   shift_upd, shift_total, angles, valid_angles, iter, binning):

    Nlayers, _, Nangles = sinogram_shifted.shape

    if not plt.fignum_exists(5464):
        plt.figure(5464, figsize=(12, 8))
    else:
        plt.figure(5464)

    mid = rec[:, :, rec.shape[2] // 2]
    q_lo, q_hi = np.quantile(mid, [0.01, 0.999])
    rng = [q_lo, q_hi]

    plt.clf()
    plt.subplots_adjust(wspace=0.25, hspace=0.25)

    plt.subplot(2, 3, 1)
    sino_slice = sinogram_shifted[Nlayers // 2, :, :].T

    hp = sino_slice - uniform_filter1d(sino_slice, size=5, axis=0)
    q1, q2 = np.quantile(hp, [0.01, 0.99])
    plt.imshow(hp, vmin=q1, vmax=q2, cmap='bone', aspect='auto')
    plt.title(f'High-pass filtered shifted sinogram\nCurrent downsampling: {binning}x')
    plt.axis('off')

    # if par.showsorted:
    #     xaxis = angles
    #     xlab = 'Angle [deg]'
    # else:
    xaxis = np.arange(Nangles)
    xlab = '# projection'

    plt.subplot(2, 3, 2)
    plt.plot(xaxis, shift_upd[:, 0] * binning, '.-r')
    plt.plot(xaxis, shift_upd[:, 1] * binning, '.-b')
    plt.grid(True)
    plt.legend(['horiz', 'vert'])
    plt.title('Current position update')
    plt.xlim([min(xaxis), max(xaxis)])
    plt.ylabel('Shift  x downsampling [px]')
    plt.xlabel(xlab)

    plt.subplot(2, 3, 3)
    plt.plot(xaxis, shift_total[:, 0] * binning, '.-r')
    plt.plot(xaxis, shift_total[:, 1] * binning, '.-b')
    plt.title('Total position update')
    plt.legend(['horiz', 'vert'])
    plt.ylabel('Shift x downsampling [px]')
    plt.xlim([min(xaxis), max(xaxis)])
    plt.xlabel(xlab)
    plt.grid(True)

    plt.subplot(2, 3, 6)
    plt.plot(xaxis[valid_angles], err[iter, valid_angles], 'k.')
    plt.plot(xaxis[~valid_angles], err[iter, ~valid_angles], 'r.')
    if np.any(~valid_angles):
        plt.legend(['errors', 'ignored'])
    plt.title('Current error')
    plt.grid(True)
    plt.xlim([min(xaxis), max(xaxis)])
    plt.xlabel(xlab)

    plt.subplot(2, 3, 5)
    plt.plot(err)
    plt.plot(np.mean(err, axis=1), 'k', linewidth=3)
    plt.grid(True)
    plt.xlim([1, iter + 1])
    plt.xscale('log')
    plt.yscale('log')
    plt.title('MSE evolution')
    plt.xlabel('Iteration')
    plt.ylabel('Mean square error')

    plt.subplot(2, 3, 4)
    midz = rec[:, :, rec.shape[2] // 2]
    q_lo, q_hi = np.quantile(midz, [0.01, 0.999])
    plt.imshow(midz.T, cmap='bone', vmin=q_lo, vmax=q_hi, origin='lower')
    plt.axis('image')
    plt.xticks([])
    plt.yticks([])
    plt.title('Current reconstruction')

    plt.draw()
    plt.pause(0.001)


import numpy as np

def apply_circular_mask(rec, radius=0.99, apodize=True):

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


def radtap(X, Y, tappix, zerorad):
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

def apply_3D_apodization(tomogram, rad_apod=None, axial_apod=None, radial_smooth=None):
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
