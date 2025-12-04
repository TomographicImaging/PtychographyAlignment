import numpy as np
import astra
import matplotlib.pyplot as plt

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

                filtered[iy, ia, :] = filtered_proj[pad_left:pad_right+Nx]
            
    filtered = filtered.astype(np.float32)

    sino_id = astra.data3d.create('-sino', proj_geom, filtered)
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

    sino_id = astra.data3d.create('-sino', proj_geom)

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

from numpy.fft import fft2, ifft2, fft, ifft


def fft_shift(img, shift):
    dy, dx = shift
    ny, nx = img.shape

    ky = np.fft.fftfreq(ny)
    kx = np.fft.fftfreq(nx)
    Kx, Ky = np.meshgrid(kx, ky)

    phase = np.exp(-2j * np.pi * (dx * Kx + dy * Ky))
    return np.real(ifft2(fft2(img) * phase))


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
        out = fft_shift(out, shift)


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
