from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import uniform_filter1d, gaussian_filter
from scipy.ndimage import convolve as convolve_ndimage
from scipy.signal import convolve as convolve_signal
from scipy.signal.windows import tukey
from skimage.registration import phase_cross_correlation
from utilities import shift_tools
# from xml import utils


@dataclass
class CrossCorrelationConfig:
    upsample_factor: int = 100
    plot_correlation: bool = True


@dataclass
class MatlabStyleCrossCorrelationConfig:
    binning: int = 1
    max_iter: int = 1
    filter_pos: float = 50.0
    filter_data: float = 0.05
    roi: tuple | None = None
    upsample_factor: int = 100
    plot_correlation: bool = True


class CrossCorrelationAlignment:
    """
    Basic cross-correlation alignment code. This could be replaced with a different method
    """
    def __init__(self, config: CrossCorrelationConfig = None):
        self.config = config or CrossCorrelationConfig()

    def run_alignment(self, sinogram, theta):
        """
        Run cross-correlation code. This code expects data in order [Ny, Nx, Nangles].

        Parameters
        ----------
        sinogram: ndarray
            Sinogram in data order [Ny, Nx, Nangles]
        theta: ndarray
            Array of angles in radians
        optimal_shift: ndarray
            The current best shift in Ny, Nx
        binning: int
            Curent binning value for sinogram
        """
        [Ny, Nx, Nangles] = sinogram.shape

        shift_correlation = np.zeros((Nangles,2))
                            
        for gag in range(1, Nangles):
            a = (sinogram[:,:,gag-1])
            b = (sinogram[:,:,gag])
            
            shift_ab = phase_cross_correlation(a,b,upsample_factor=self.config.upsample_factor)
            shift_correlation[gag,0] = shift_ab[0][1] + shift_correlation[gag-1,0]
            shift_correlation[gag,1] = shift_ab[0][0] + shift_correlation[gag-1,1]

        shift_correlation[:,0] = shift_correlation[:,0] - np.median(shift_correlation[:,0])
        shift_correlation[:,1] = shift_correlation[:,1] - np.median(shift_correlation[:,1])

        if self.config.plot_correlation:
            plt.figure(figsize=[10,3])
            plt.subplot(121),plt.plot(theta, shift_correlation[:,1], label='x-correlation')
            plt.ylabel('Vertical shift (pixels)'), plt.xlabel('Angle (rad)')
            plt.grid()
            plt.subplot(122),plt.plot(theta, shift_correlation[:,0], label='x-correlation')
            plt.ylabel('Horizontal shift (pixels)'), plt.xlabel('Angle (rad)')
            plt.grid()

        return shift_correlation


class MatlabStyleCrossCorrelationAlignment:
    """
    Matlab-inspired cross-correlation alignment routine.

    This follows the logic of align_tomo_Xcorr.m more closely than the simpler
    implementation above: it builds a variation field, optionally bins/ROI-crops,
    filters the Fourier domain, computes relative shifts between neighboring
    projections, accumulates the drift, removes slow drift, and returns the final
    total shift.
    """

    def __init__(self, config: MatlabStyleCrossCorrelationConfig = None):
        self.config = config or MatlabStyleCrossCorrelationConfig()

    def run_alignment(
        self,
        object_0,
        theta,
        roi=None,
        binning=None,
        max_iter=None,
        filter_pos=None,
        filter_data=None,
        illum_sum=None,
    ):
        """
        Matlab-inspired cross-correlation alignment based on align_tomo_Xcorr.m.

        This routine sorts projections by angle before computing relative shifts,
        then applies the same drift-removal pattern used by the Matlab code. The
        output is finally mapped back to the original acquisition order.

        The convention used here is total_shift[:,0] = horizontal shift,
        total_shift[:,1] = vertical shift, matching the Matlab plotting/usage.
        """
        object_0 = np.asarray(object_0)
        if np.isrealobj(object_0):
            raise ValueError('Complex object expected.')

        if roi is None:
            roi = (slice(None), slice(None))
        if binning is None:
            binning = self.config.binning
        if max_iter is None:
            max_iter = self.config.max_iter
        if filter_pos is None:
            filter_pos = self.config.filter_pos
        if filter_data is None:
            filter_data = self.config.filter_data

        theta = np.asarray(theta, dtype=float)
        theta_order = np.argsort(theta)
        inv_order = np.argsort(theta_order)
        theta_sorted = theta[theta_order]
        object_sorted = object_0[:, :, theta_order]

        y_slice, x_slice = roi
        object_roi = object_0[y_slice, x_slice, :]
        Ny, Nx, Nangles = object_roi.shape

        if illum_sum is None:
            illum_sum = np.ones((Ny, Nx), dtype=float)

        weights = illum_sum[y_slice, x_slice] / (illum_sum[y_slice, x_slice] + 1e-1 * np.max(illum_sum))

        variation = self._get_variation_field(object_roi, binning, weights)
        variation = np.real(variation)
        total_shift_sorted = np.zeros((Nangles, 2), dtype=float)
        filter_pos = Nangles/4

        for _ in range(max_iter):
            fvar = self._filtered_fft(variation, total_shift_sorted, filter_data)

            frame_ref = fvar[:,:,theta_order]
            frame_align = fvar[:,:,np.roll(theta_order,-1)]
            relative_shifts = find_shift_fast_2D(frame_ref, frame_align, sigma=filter_data, apply_fft=False, method='full_range')

            # Matlab uses circshift(relative_shifts,1).
            relative_shifts = np.roll(relative_shifts, 1, axis=0)

            # Robust magnitude cap for a single iteration, matching the Matlab idea.
            # Clipping policy will be finalized later; for now this keeps the update bounded.
            max_shift = [max(10.0, (3.0 * np.median(np.abs(relative_shifts - np.median(relative_shifts,axis=0)),axis=0))[0]),
                         max(10.0, (3.0 * np.median(np.abs(relative_shifts - np.median(relative_shifts,axis=0)),axis=0))[1])]
            relative_shifts = np.minimum(max_shift, np.abs(relative_shifts)) * np.sign(relative_shifts)

            relative_shifts -= np.mean(relative_shifts,axis=0)
            cum_shift = np.cumsum(relative_shifts, axis=0)
            cum_shift -= np.mean(cum_shift, axis=0, keepdims=True)

            smooth = int(np.ceil(filter_pos / 2.0) * 2 + 1)
            kernel = np.ones(smooth, dtype=float) / smooth
            if np.isfinite(filter_pos) and filter_pos > 0:
                for axis in range(2):
                    # Matlab uses convolutions with a box filter and a normalizing denominator.
                    smoothed = np.convolve(cum_shift[:, axis], kernel, mode='same')
                    denom = np.convolve(np.ones(Nangles), kernel, mode='same')
                    cum_shift[:, axis] -= smoothed / denom

            total_shift_sorted += cum_shift

            # Matlab then clips total_shift based on MAD to suppress outliers.
            # This is kept as a comment for a later dedicated clipping step.
            mad_total = np.median(np.abs(total_shift_sorted - np.median(total_shift_sorted, axis=0, keepdims=True)), axis=0)
            clip_limit = 6.0 * mad_total
            total_shift_sorted = np.minimum(np.abs(total_shift_sorted), clip_limit) * np.sign(total_shift_sorted)

            if np.max(np.abs(cum_shift)) < 1e-6:
                print(f'Converged after {_+1} iterations.')
                break

        total_shift = total_shift_sorted[inv_order,:]
        variation_aligned = shift_tools.imshift_fft(variation, total_shift)

        total_shift = np.round(total_shift * binning)

        if self.config.plot_correlation:
            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            plt.plot(theta_sorted, total_shift[:, 1], label='matlab-style xcorr')
            plt.ylabel('Vertical shift (pixels)')
            plt.xlabel('Angle (rad)')
            plt.grid(True)
            plt.subplot(122)
            plt.plot(theta_sorted, total_shift[:, 0], label='matlab-style xcorr')
            plt.ylabel('Horizontal shift (pixels)')
            plt.xlabel('Angle (rad)')
            plt.grid(True)

        return total_shift, variation, variation_aligned

    @staticmethod
    def _get_variation_field(object_roi, binning, weights):
        """
        Build the variation field used by the Matlab routine.
        This is analogous to the local gradient magnitude in align_tomo_Xcorr.m.
        """
        dX = convolve_ndimage(object_roi, [[[-1],[1]]], mode='constant', cval=0.0)
        dY = convolve_ndimage(object_roi, [[[-1]],[[1]]], mode='constant', cval=0.0)

        variation = np.sqrt(np.abs(dX) ** 2 + np.abs(dY) ** 2)
        variation = variation * np.abs(object_roi)

        variation[[0, -1], :, :] = variation[[1, -2], :, :]
        variation[:, [0, -1], :] = variation[:, [1, -2], :]

        if weights is not None:
            weights = np.asarray(weights, dtype=float)
            varwei = variation * weights[:, :, None]
            mean_variation = np.mean(np.mean(varwei,axis=0, keepdims=1), axis=1, keepdims=1) / np.mean(weights)
            varmwei = (variation - mean_variation) ** 2 * weights[:, :, None]
            dev_variation = np.sqrt(np.mean(np.mean(varmwei,axis=0,keepdims=1),axis=1,keepdims=1)) / np.mean(weights)
            variation = np.minimum(variation, mean_variation + dev_variation)

        if binning > 1:
            sigma = (2 * binning, 2 * binning, 0)
            variation = gaussian_filter(variation, sigma=sigma, truncate=2.0, mode='constant',cval=0.0)
            boundary_correction = gaussian_filter(np.ones_like(variation[:, :, 0]), sigma=(2 * binning, 2 * binning), truncate=2.0, mode='constant',cval=0.0)
            variation = variation / boundary_correction[:, :, None]
            variation = variation[::binning, ::binning, :]

        return variation

    @staticmethod
    def _apply_fourier_shift(image, shift):
        """Apply a subpixel shift with a Fourier-domain phase ramp."""
        ny, nx = image.shape
        y = np.fft.fftfreq(ny)
        x = np.fft.fftfreq(nx)
        Y, X = np.meshgrid(y, x, indexing='ij')
        phase = np.exp(-2j * np.pi * (Y * shift[0] + X * shift[1]))
        return np.fft.ifft2(np.fft.fft2(image) * phase).real

    def _filtered_fft(self, img, total_shift, filter_data):
        """Mirror the Matlab filtered_FFT logic: window, shift, and high-pass in Fourier space."""

        ny, nx, _ = img.shape
        img = shift_tools.imshift_fft(img, total_shift)

        window = np.outer(tukey(ny, alpha=0.3), tukey(nx, alpha=0.3))

        img = img - np.mean(img)
        img *= window[:, :, None]

        img = np.fft.fft2(img,axes=(0,1))                
        if filter_data > 0:
            yy, xx = np.meshgrid(np.arange(-nx//2,nx//2), np.arange(-ny//2,ny//2), indexing='ij')
            radius_sq = xx ** 2 + yy ** 2
            mean_dim = np.mean([ny, nx])
            spectral_filter = np.fft.fftshift(np.exp(1/(-(radius_sq / ((mean_dim * filter_data) ** 2))))).T
            filtered = img * spectral_filter[:,:,None]
        else:
            filtered = img

        return filtered

    @staticmethod
    def _apply_shift_to_3d(img, total_shift):
        """Apply the accumulated shift back to the 3D data array."""
        out = np.empty_like(img, dtype=np.complex128)
        for i in range(img.shape[2]):
            y_shift = int(np.round(total_shift[i, 0]))
            x_shift = int(np.round(total_shift[i, 1]))
            out[:, :, i] = np.roll(np.roll(img[:, :, i], y_shift, axis=0), x_shift, axis=1)
        return out.real


def find_shift_fast_2D(o1, o2, sigma=0.0, apply_fft=True, method='full_range'):
    """
    Matlab-equivalent of utils.find_shift_fast_2D.

    This takes two 2D arrays (or 3D stacks with an image index along axis 2) and
    estimates the 2D displacement between them using a Fourier-domain
    cross-correlation, then extracts the subpixel peak location using a
    center-of-mass around the correlation maximum.

    Parameters
    ----------
    o1, o2 : ndarray
        Arrays of shape (nx, ny) or (nx, ny, nframes).
    sigma : float
        Spectral low-frequency suppression strength. sigma <= 0 disables it.
    apply_fft : bool
        If True, assume the inputs are real-space images and Fourier-transform them
        before correlation. If False, inputs are already in Fourier space.
    method : {'full_range', 'limited_range'}
        Matlab method used by the original code.
    """
    o1 = np.asarray(o1)
    o2 = np.asarray(o2)

    if o1.shape != o2.shape:
        raise ValueError(f'o1 and o2 must have the same shape; got {o1.shape} and {o2.shape}.')

    if apply_fft:
        nx, ny = o1.shape[:2]
        spatial_filter = np.outer(tukey(nx, alpha=0.5), tukey(ny, alpha=0.5))
        o1 = o1 * spatial_filter[:, :, None] if o1.ndim == 3 else o1 * spatial_filter
        o2 = o2 * spatial_filter[:, :, None] if o2.ndim == 3 else o2 * spatial_filter
        o1 = np.fft.fft2(o1)
        o2 = np.fft.fft2(o2)

    nx, ny = o1.shape[:2]

    if sigma > 0:
        yy, xx = np.meshgrid(np.arange(-nx // 2, nx // 2), np.arange(-ny // 2, ny // 2), indexing='ij')
        x = xx / nx
        y = yy / ny
        spectral_filter = np.fft.fftshift(np.exp(1.0 / (-(x ** 2 + y ** 2) / sigma ** 2)))
        if o1.ndim == 2:
            o1 = o1 * spectral_filter
            o2 = o2 * spectral_filter
        else:
            o1 = o1 * spectral_filter[:, :, None]
            o2 = o2 * spectral_filter[:, :, None]

    xcorr = np.fft.fftshift(np.abs(np.fft.ifft2(o1 * np.conj(o2), axes=(0,1))), axes=(0,1))

    if method == 'full_range':

        win = 5
        kernel = np.ones((win, win, 1), dtype=float)

        mask = np.equal(xcorr, np.max(xcorr,axis=(0,1),keepdims=True)).astype(int)
        mask = convolve_signal(mask, kernel, mode='same') > 0.1

        xcorr = np.where(mask, xcorr, np.nan)
        xcorr = np.maximum(0.0, xcorr - np.nanmin(xcorr,axis=(0,1),keepdims=True))
        xcorr = np.where(mask, xcorr, 0.0)
        xcorr = np.power(xcorr / np.max(xcorr,axis=(0,1),keepdims=True), 2)
        xcorr = np.maximum(0.0, xcorr - 0.5) ** 2

        mass = np.sum(xcorr, axis=(0,1))
        if np.sum(mass) == 0:
            print('Warning: No mass in cross-correlation; returning zero shift.')
            return np.array([0.0, 0.0], dtype=float)
        
        x_idx = np.arange(xcorr.shape[1])
        y_idx = np.arange(xcorr.shape[0])

        x_center = np.sum(np.sum(xcorr, axis=0, keepdims=True) * x_idx[None, :, None],axis=1) / mass - (xcorr.shape[1] // 2) - 1
        y_center = np.sum(np.sum(xcorr, axis=1, keepdims=True) * y_idx[:, None, None],axis=0) / mass - (xcorr.shape[0] // 2) - 1

        shift = np.array([x_center[0,:], y_center[0,:]], dtype=float).T

    elif method == 'limited_range':
        mxcorr = np.mean(xcorr, axis=2)
        y0, x0 = np.unravel_index(np.argmax(mxcorr), mxcorr.shape)
        max_shift = 10
        max_shift_x = min(nx // 2 - 1, max_shift)
        max_shift_y = min(ny // 2 - 1, max_shift)
        yrange = np.arange(y0 - max_shift_y, y0 + max_shift_y + 1)
        xrange = np.arange(x0 - max_shift_x, x0 + max_shift_x + 1)
        xcorr = xcorr[np.clip(yrange, 0, nx - 1)[:, None], np.clip(xrange, 0, ny - 1)[None, :], :]
        maxv = np.max(xcorr)
        xcorr = xcorr / maxv if maxv > 0 else xcorr
        xcorr = np.maximum(0.0, xcorr - 0.5) ** 2
        mass = np.sum(xcorr)
        y_idx = np.arange(xcorr.shape[0])
        x_idx = np.arange(xcorr.shape[1])
        y_center = np.sum(np.sum(xcorr, axis=1)[:, None] * y_idx[:, None] / mass) - (xcorr.shape[0] // 2) - 1
        x_center = np.sum(np.sum(xcorr, axis=0)[None, :] * x_idx[None, :] / mass) - (xcorr.shape[1] // 2) - 1
        shift = np.array([x_center, y_center], dtype=float) + np.array([x0, y0]) - np.array([ny // 2, nx // 2]) - 1

    else:
        raise ValueError("method must be 'full_range' or 'limited_range'")

    if np.any(np.isnan(shift)):
        raise FloatingPointError('NaN shift encountered in find_shift_fast_2D')

    return shift
