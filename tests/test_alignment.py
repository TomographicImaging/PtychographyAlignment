# %%
from skimage.registration import phase_cross_correlation
from scipy import signal, ndimage
import numpy as np

import utilities as utils
from config.paths import Connor_Wright_filepath, Connor_Wright_data_key, Connor_Wright_angle_key
from alignment import CrossCorrelationAlignment, CrossCorrelationConfig, TomoConsistencyAlignment, TomoConsistencyConfig, VerticalAlignment, VerticalAlignmentConfig
# %%
def setup():
    """Load data and perform initial cropping"""
    img, theta, _ = utils.utils_tomo.load_data(Connor_Wright_filepath, data_key= Connor_Wright_data_key, angle_key = Connor_Wright_angle_key)
    img = img.transpose((1,2,0)) # transpose to shape [Ny, Nx, Nangles]
    img = img[:,:,1:]
    theta = theta[1:]
    
    return img, theta

def setup_tomoconsistency(img):

    [Ny, Nx, Nangles] = img.shape
    Ny_start = 83
    Ny_stop = Ny-Ny_start

    Nx_start = 66
    Nx_stop = Nx-Nx_start

    # get the phase gradient
    img_grad = utils.phase_tools.get_phase_gradient_1D(img, ax=1)[Ny_start:Ny_stop,Nx_start:Nx_stop,:] # apply on the full image then re-crop
    [Ny, Nx, Nangles] = img_grad.shape

    high_pass_filter = 0.01
    size = np.maximum(3, int(np.ceil(high_pass_filter * Nx)))
    gauss_window = signal.windows.gaussian(size, std = size/6)
    hanning_window = signal.windows.hann(3)
    ker = gauss_window.reshape(-1,1) * hanning_window

    ker2 = ker[np.newaxis,:,:]
    convolution_result = ndimage.convolve((np.abs(img_grad) > 2).astype(np.float32), ker2.astype(np.float32), mode = 'constant', cval = 0.0)
    weights_find_shift = np.maximum(0,1-convolution_result)

    # further cropping
    vert_range = (31, Ny-33)

    ROI = (
        slice(int(vert_range[0]), int(vert_range[1])),
        slice(None)
    )

    return img_grad, weights_find_shift, convolution_result, ROI


def test_cross_correlation_alignment():
    """
    This is a functional test of the cross-correlation alignment using the Connor Wright dataset.
    """
    img, theta = setup()

    [Ny, Nx, Nangles] = img.shape
    Ny_start = 83
    Ny_stop = Ny-Ny_start

    Nx_start = 66
    Nx_stop = Nx-Nx_start

    config = CrossCorrelationConfig(plot_correlation=False)
    xcorr = CrossCorrelationAlignment(config)
    shift_correlation = xcorr.run_alignment(img[Ny_start:Ny_stop,Nx_start:Nx_stop,:], theta)

    shift = np.loadtxt('shift_correlation.csv', delimiter=',', skiprows=1)

    np.testing.assert_array_almost_equal(shift_correlation, shift)

def test_vertical_alignment(crop=True):
    """
    This is a functional test of the vertical alignment using the Connor Wright dataset.
    It takes a couple of minutes to run.
    """
    img, theta = setup()

    [Ny, Nx, Nangles] = img.shape
    Ny_start = 83
    Ny_stop = Ny-Ny_start

    Nx_start = 66
    Nx_stop = Nx-Nx_start

    phase, residuals = utils.phase_tools.unwrap2D_fft2(img[Ny_start:Ny_stop,Nx_start:Nx_stop,:], empty_region=(50,50))
    phase=phase.real
    config = VerticalAlignmentConfig(
        data_filter = 0.01,
        iterations = 1000,
        plot_alignment=False)
    va = VerticalAlignment(config)
    shift_vertical = va.run_alignment(phase, residuals, theta)

    shift = np.loadtxt('shift_vertical.csv', delimiter=',', skiprows=1)

    np.testing.assert_array_almost_equal(shift_vertical, shift)

def test_tomoconsistency_alignment():
    """
    This is a functional test of the tomoconsistency alignment using the Connor Wright dataset.
    It requires the image to be shifted initially with values from the cross correlation and vertical alignment 
    It takes approximately 2 minutes to run
    """
    # load data
    img, theta = setup()
    [Ny, Nx, Nangles] = img.shape
    Ny_start = 83
    Ny_stop = Ny-Ny_start

    Nx_start = 66
    Nx_stop = Nx-Nx_start

    # cross correlation
    config = CrossCorrelationConfig(plot_correlation=False)
    xcorr = CrossCorrelationAlignment(config)
    shift_correlation = xcorr.run_alignment(img[Ny_start:Ny_stop,Nx_start:Nx_stop,:], theta)
    
    shift_total = np.zeros((Nangles,2))
    shift_total[:,0] = shift_total[:,0] + shift_correlation[:,0]

    # vertical alignment
    phase, residuals = utils.phase_tools.unwrap2D_fft2(img[Ny_start:Ny_stop,Nx_start:Nx_stop,:], empty_region=(50,50))
    phase=phase.real
    config = VerticalAlignmentConfig(
        data_filter = 0.01,
        iterations = 1000,
        plot_alignment=False)
    va = VerticalAlignment(config)
    shift_vertical = va.run_alignment(phase, residuals, theta)

    shift_total[:,1] = shift_total[:,1] + shift_vertical

    # shift image
    for m in range(shift_total.shape[0]):
        img[:,:,m] = np.roll(img[:,:,m],(int(shift_total[m,0]), int(shift_total[m,1])),axis=(1,0)) 
    
    # tomoconsistency alignment
    img_grad, weights_find_shift, convolution_result, ROI = setup_tomoconsistency(img)
    
    config = TomoConsistencyConfig(
        max_iterations = 200,
        step_relaxation = 0.5,
        high_pass_filter = 0.01,
        min_step_size = 0.01,
        unwrap_data_method = 'fft_1d',
        plot_interactive = False,
        center_reconstruction = True,
        apply_mask = False,
        momentum_acceleration = True,
        align_horizontal = True,
        align_vertical = False,
        apply_positivity = True)

    aligner = TomoConsistencyAlignment(config)

    binning = 8
    sinogram = utils.shift_tools.imshift_generic(img_grad, shift=np.zeros((Nangles,2)), Npix = None, affine_matrix = None, smooth = 5, 
                                        ROI = ROI, downsample = binning, interp_method = 'fft', interp_sign = 1)

    weights_find_shift = np.maximum(0,1-convolution_result)
    weights_find_shift = utils.shift_tools.imshift_generic(weights_find_shift, shift=np.zeros((Nangles,2)), Npix = None, affine_matrix = None, smooth = 0, 
                                        ROI = ROI, downsample = binning, interp_method = 'linear', interp_sign = 1)
    
    shift_tomoconsistency = np.zeros((Nangles,2))
    shift_tomoconsistency, _, _, _ = aligner.run_alignment(sinogram, theta, weights_find_shift, shift_tomoconsistency, binning)

    shift = np.loadtxt('shift_tomoconsistency.csv', delimiter=',', skiprows=1)

    np.testing.assert_array_almost_equal(shift_tomoconsistency, shift)
    