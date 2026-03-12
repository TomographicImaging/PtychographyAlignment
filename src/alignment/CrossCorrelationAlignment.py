from dataclasses import dataclass
import numpy as np
import matplotlib.pyplot as plt
from skimage.registration import phase_cross_correlation

@dataclass
class CrossCorrelationConfig:
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