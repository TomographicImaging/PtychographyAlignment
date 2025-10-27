# author: Oriol
# Danica only wrote the classes and tidied code
#--------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
import utilities.utils_used as utils_used
from scipy.ndimage import gaussian_filter1d, measurements
from scipy import optimize, signal
from skimage.registration import phase_cross_correlation as register_translation


class BaseClassHorizontalAlignment():
    """
    This class should (will) contain basic methods useful to all alignment procedures.
    The way the shifts are calculated is, on the other hand, determined in the specific class.
    """
    
    def apply_projection_shifts(self, projections_in, shifts, pad=0):
        """
        This method uses the shifts calculated before to "move" the projections
        
        Parameters
        ----------
        projections_in : 3D image to be aligned
        shifts : shifts required in pixels
        pad : zero-padding the projections_in

        Returns
        -------
        projections_out : aligned projections
        """
        projections_out = np.zeros_like(projections_in)
        projections_pad = np.copy(utils_used.pad(projections_in, (pad,0)))
        shifts = np.round(shifts).astype(np.int32)

        for i in range(projections_in.shape[0]):
    #         print("Applying shift [%d,%d] to projection %d" %(shifts[0,i], shifts[1,i], i))
            projections_out[i] = np.roll(projections_pad[i], -1*shifts[0,i], axis=0)[:,pad:-pad]
            projections_out[i] = np.roll(projections_pad[i], -1*shifts[1,i], axis=1)[:,pad:-pad]

            #if i == 0:
            #    plt.figure()
            #    plt.imshow(projections_out[i])
            #    plt.show()
        
    #     min_trans_x = int(np.floor(np.amin(shifts[1,:])))
    #     max_trans_x = int(np.ceil(np.amax(shifts[1,:])))
    #     if min_trans_x < 0:
    #         x_to = min_trans_x
    #     else:
    #         x_to = -1
        
    #     if max_trans_x > 0:
    #         x_from = max_trans_x
    #     else:
    #         x_from = 0
            
            
    #     min_trans_y = int(np.floor(np.amin(shifts[0,:])))
    #     max_trans_y = int(np.ceil(np.amax(shifts[0,:])))
    #     if min_trans_y < 0:
    #         y_to = min_trans_y
    #     else:
    #         y_to = -1
        
    #     if max_trans_y > 0:
    #         y_from = max_trans_y
    #     else:
    #         y_from = 0

    #     projections_out = projections_out[:,y_from:y_to,x_from:x_to]
        
        return projections_out


class VerticalAlignmentCrossCorrelation():
    def __init__(self, projections):
        """Given projections, calculates the vertical alignment and applies it to them.
        It stores the result in an attribute."""
        self.projections_aligned, self.vertical_shifts = self.projection_align_vertical(projections)

    def xcor(self, in1, in2):
        """
        Calculates the cross correlation between two N-dimensional arrays. They
        could be 2D images or 1D arrays. The shift is expressed as the maximum 
        value of the correlation from the centre of the image.
        
        Parameters
        ----------
        in1, in2 : 2D images
        
        Returns
        -------
        shift : shift in pixels according to the cross correlation
        xcor : result of cross correlation between images
        """
        xcor = signal.correlate(in1, in2, mode='same')
        shift = np.argmax(xcor)-(xcor.shape[0]/2)
        return shift, xcor

    def get_y_shifts(self, projections):
        """
        Calculates the y-shifts using the cross correlation method. 
        Sums the values along the second axis, then calculates the gradients.
        Cross correlates the gradients.
        
        Parameters
        ----------
        projections : 3D array with size Nx x Ny x Ntheta
        
        Returns
        -------
        shifts : in the y direction
        """
        proj_sum = np.sum(projections,2)
        shifts = np.zeros(projections.shape[0])
        xcor_ar = np.zeros_like(proj_sum)
        a = np.gradient(proj_sum[0,:])
        for i in range(projections.shape[0]):
            b = np.gradient(proj_sum[i,:])
            shifts[i], xcor_ar[i,:] = self.xcor(a,b) #
        return shifts

    def apply_y_shifts(self, projections, trans):
        """
        Given a 3D image, applies the yshifts in each image as stored in trans.
        
        Parameters
        ----------
        projections : 3D image
        trans : 1D array of shifts

        Returns
        -------
        projections : in place
        """
        for i in range(projections.shape[0]):
            shift = int(trans[i])
            projections[i,:,:] = np.roll(projections[i,:,:],shift, axis=0)
        return projections

    def projection_align_vertical(self, projections):
        """
        The full pipeline to perform the vertical alignment.
        Caclulates the yshifts, applies them. Crops the result.
        
        Parameters
        ----------
        projections : 3D image

        Returns
        -------
        projections : cropped 3D image in place
        y_trans : y shifts calculated and applied
        """
        y_trans = self.get_y_shifts(projections)
        projections = self.apply_y_shifts(projections, y_trans)

        min_trans = int(np.floor(np.amin(y_trans)))
        max_trans = int(np.ceil(np.amax(y_trans)))

        if min_trans < 0:
            y_to = min_trans
        else:
            y_to = -1
        
        if max_trans > 0:
            y_from = max_trans
        else:
            y_from = 0

        print("y_from", y_from)
        print("y_to", y_to)

        projections = projections[:,y_from:y_to,:]
        return projections, y_trans


class HorizontalAlignmentCrossCorrelation(BaseClassHorizontalAlignment):
    def __init__(self, projections):
        """
        Saves the projections as an attribute.
        Performs the horizontal alignment with cross-correlation and saves the resulting 3D image 
        as an attribute.
        """
        self.projections = projections
        self.correction_rough = self.horizontal_alignment()

    def calculate_correlation(self):
        """
        This method uses cross correlation between images to calculate the shifts in
        the horizontal direction. The cross correlation method is taken from "skimage".
        
        Returns
        -------
        quick_xcorrelation : the shift in the horizontal direction.
        """
        projections = self.projections
        quick_xcorrelation = np.zeros((2,projections.shape[0]))
                                    
        for gag in range(1,projections.shape[0]):
            a = (projections[gag-1,:,:])
            b = (projections[gag,:,:])
            
            shift_ab = register_translation(a,b,upsample_factor=100)
            quick_xcorrelation[1,gag] = shift_ab[0][1] + quick_xcorrelation[1,gag-1]

        plt.figure(figsize=[4,4])
        plt.plot(quick_xcorrelation[1,:])
        
        return quick_xcorrelation
    
    def horizontal_alignment(self):
        """
        Applies the shifts calculated by "calculate_correlation" to the projections attribute.

        Returns
        -------
        correction_rough : projections after rough horizontal alignment from cross correlation
        """
        projections = self.projections
        quick_xcorrelation = self.calculate_correlation()
        correction_rough = self.apply_projection_shifts(projections, -gaussian_filter1d(quick_xcorrelation,10), pad=0)
        
        plt.imshow(projections[:,projections.shape[1]//2,:])
        plt.figure(),plt.imshow(correction_rough[:,projections.shape[1]//2,:])
        
        return correction_rough


class COMAlignment(BaseClassHorizontalAlignment):

    def __init__(self, projections, angles):
        
        """
        Defines attributes, aligns the projections.

        Parameters
        ----------
        projections : 3D stack of projections (usually after vertical alignment)
        angles : array of acquisition angles
        """
        self.projections = projections
        self.angles = angles
        self.projections_fitted = self.alignment()

    def sin_func(self, x, a, w, b, c):
        """Fit function. Used by COMAlignment."""
        return a * np.sin(w*x + b) + c

    def fit_com_to_sin(self, projections, angles, blur_filter = 2):
        """
        Calculates the shifts by a centre of mass method. The COM is calculated using "scipy".
        This requires a sum over one axis. Performs a fit to a sinusoidal function
        and a gaussian filter. Applies the shifts.

        Parameters 
        ----------
        projections : 3D stack of projections Nx Ny Ntheta
        angles : array of acquisition angles, must match ntheta
        blur_filter : amount of gaussian blurring to be done on the shifts

        Returns
        -------
        projections_out : projections after COM alignment
        shifts : COM shifts
        """
        projections -= np.amin(projections)
        summed = np.sum(projections[:,:,:],1)
        
        com = np.zeros([2,projections.shape[0]])
        shifts = np.zeros([2,projections.shape[0]])
        for i in range(projections.shape[0]):
            com[:,i] = measurements.center_of_mass(summed[i,:])
        
        com_min = np.min(com[1,:])
        com_max = np.max(com[1,:])
        com_range = com_max - com_min
        com_mid = com_min + com_range//2
        
        params, params_covariance = optimize.curve_fit(self.sin_func, angles, com[1,:], p0=[com_range, 1, 0, com_mid])
        ideal = self.sin_func(self.angles, params[0], params[1], params[2], params[3])
        
        #ideal = (np.cos(angles)*com_range) + com_mid
        
        shifts[1,:] = ideal[:] - com[1,:]
        shifts[1,:] = shifts[1,:] - shifts[1,0]
        #shifts[1,:] = shifts[1,:]*overall_shift/np.max(abs(shifts[1,:]))
        #shifts[1,:] = shifts[1,:]#*overall_shift/(abs(shifts[1,-1]))

        #print("shifts[-1,:] ", shifts[-1,:])
        
        shifts[1,:] = gaussian_filter1d(shifts[1,:],blur_filter)

        plt.figure()
        plt.plot(com[1,:],'bo')
        plt.plot(ideal)
        plt.plot(shifts[1,:],'r+')
        #plt.legend(handles=['com','ideal','shifts'])

        projections_out = self.apply_projection_shifts(projections, shifts, pad=1)
        return projections_out, shifts

    def alignment(self):
        """
        Calculate the centre of mass (COM) for each projection of the scan. The COM should follow a sinusoidal motion.
        Then fit a sinusoid and shift the projections as required to fit that sinusoid.
        
        Returns
        -------
        projections_fitted : projections after centre of mass alignment
        
        """
        projections = self.projections
        angles = self.angles

        projections_fitted, shifts_fitted = self.fit_com_to_sin(projections,angles, blur_filter = 20)

        plt.figure(figsize=[5,5])
        plt.plot(shifts_fitted[1,:])

        plt.figure(figsize=[7,5])
        plt.subplot(1,3,1), plt.imshow(projections_fitted[:,int(projections_fitted.shape[1]//2),:])
        plt.subplot(1,3,2), plt.imshow(projections[:,int(projections.shape[1]//2),:])
        plt.subplot(1,3,3), plt.imshow(projections[:,int(projections.shape[1]//2),:]-projections_fitted[:,int(projections_fitted.shape[1]/2),:])

        return projections_fitted
