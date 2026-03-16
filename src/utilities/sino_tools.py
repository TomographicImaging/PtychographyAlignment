import numpy as np
from scipy.ndimage import convolve, center_of_mass

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

def centering_reconstruction(rec):
    """
    Estimate the 2D centre of mass of a 3D reconstruction volume.

    Computes a mass-weighted mean of per-slice centres of mass across all
    slices, using the square root of the (non-negative) voxel intensities
    as the mass distribution.

    Parameters
    ----------
    rec : np.ndarray, shape (Nz, Ny, Nx)
        3D reconstruction volume. Negative values are clamped to zero before
        computing the mass.

    Returns
    -------
    rec_center : np.ndarray, shape (2,)
        Mass-weighted centre of the volume as (x, y) pixel coordinates,
        where x corresponds to the column axis and y to the row axis.
    """    
    
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