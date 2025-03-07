import numpy as np
import matplotlib.pyplot as plt
import utils_tomo as utils
from scipy.ndimage import gaussian_filter
from scipy import optimize
from scipy import signal
from skimage.registration import phase_cross_correlation as register_translation

def sin_func(x, a, w, b, c):
    return a * np.sin(w*x + b) + c

def fit_com_to_sin(projections,angles,blur_filter = 2):
    
    #plt.figure()
    
    projections -= np.amin(projections)
    summed = np.sum(projections[:,:,:],1)
    
    com = np.zeros([2,projections.shape[0]])
    shifts = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = utils.ndimage.measurements.center_of_mass(summed[i,:])
    
    com_min = np.min(com[1,:])
    com_max = np.max(com[1,:])
    com_range = com_max - com_min
    com_mid = com_min + com_range//2
    
    params, params_covariance = optimize.curve_fit(sin_func, angles, com[1,:], p0=[com_range, 1, 0, com_mid])
    ideal = sin_func(angles_reduced, params[0], params[1], params[2], params[3])
    
    #ideal = (np.cos(angles)*com_range) + com_mid
    
    shifts[1,:] = ideal[:] - com[1,:]
    shifts[1,:] = shifts[1,:] - shifts[1,0]
    #shifts[1,:] = shifts[1,:]*overall_shift/np.max(abs(shifts[1,:]))
    #shifts[1,:] = shifts[1,:]#*overall_shift/(abs(shifts[1,-1]))

    #print("shifts[-1,:] ", shifts[-1,:])
    
    shifts[1,:] = scipy.ndimage.gaussian_filter1d(shifts[1,:],blur_filter)

    plt.figure()
    plt.plot(com[1,:],'bo')
    plt.plot(ideal)
    plt.plot(shifts[1,:],'r+')
    #plt.legend(handles=['com','ideal','shifts'])

    projections_out = apply_reprojection_shifts(projections, shifts, pad=1)
    return projections_out, shifts



def calculate_com_projs(projections):
    #plt.figure()
    
    projections -= np.amin(projections)
    summed = np.sum(projections[:,:,:],1)
    
    com = np.zeros([2,projections.shape[0]])
    shifts = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = utils.ndimage.measurements.center_of_mass(summed[i:i+1,:])
    
    return com

def calculate_com_projs_oriol(projections):
    #plt.figure()
    
    projections -= np.amin(projections)
    summed = np.sum(projections[:,:,:],1)
    
    com = np.zeros([2,projections.shape[0]])
    shifts = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = utils.ndimage.measurements.center_of_mass((summed[i:i+1,:]))
    
    return com

class VerticalAlignment():
    def __init__(self, projections):
        self.projections_aligned = self.projection_align_vertical(projections)

    def xcor(self, in1, in2):
        xcor = signal.correlate(in1, in2, mode='same')
        shift = np.argmax(xcor)-(xcor.shape[0]/2)
        return shift, xcor

    def get_y_shifts(self, projections):
        proj_sum = np.sum(projections,2)
        shifts = np.zeros(projections.shape[0])
        xcor_ar = np.zeros_like(proj_sum)
        a = np.gradient(proj_sum[0,:])
        for i in range(projections.shape[0]):
            b = np.gradient(proj_sum[i,:])
            shifts[i], xcor_ar[i,:] = self.xcor(a,b) #
        return shifts

    def apply_y_shifts(self, projections, trans):
        for i in range(projections.shape[0]):
            shift = int(trans[i])
            projections[i,:,:] = np.roll(projections[i,:,:],shift, axis=0)
        return projections

    def projection_align_vertical(self, projections):
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
        return projections

# This function gives the shifts between projections and reprojections in terms of x and y
def get_reprojection_shifts(projections, reprojections, sub_sample=1, sigma=[4,20]):
    shifts = np.zeros([2, projections.shape[0]], dtype = np.float32)
    
    border = 20
    
    sig_high = sigma[1]/sub_sample
    sig_low = sigma[0]/sub_sample
    
    projections = projections[:,::sub_sample,border:-border:sub_sample]
    reprojections = reprojections[:,::sub_sample,border:-border:sub_sample]
    
#     projections = blur(projections, sig_low) - blur(projections, sig_high)
#     reprojections = blur(reprojections, sig_low) - blur(reprojections, sig_high)
    
#     utils.plot_3_projections(projections)
#     utils.plot_sinogram(projections)
#     utils.plot_3_projections(reprojections)
#     utils.plot_sinogram(reprojections)
    
    for i in range(projections.shape[0]):
        image_a = projections[i]
        image_b = reprojections[i]

        if i == 0:
            plt.figure()
            plt.subplot(1,2,1); plt.imshow(image_a)
            plt.subplot(1,2,2); plt.imshow(image_b)
            plt.show()
        
        shift, error, diffphase = register_translation(image_a, image_b)
        
        # print("frame_shift:", shift)
        
        shifts[0,i] = shift[0]
        shifts[1,i] = shift[1]
    shifts *= sub_sample
    return shifts

# This function gives the shifts between projections and reprojections in terms of x and y
def get_reprojection_shifts_com(projections, reprojections, sub_sample=1, sigma=[4,20]):
    shifts = np.zeros([1, projections.shape[0]], dtype = np.float32)
    
    border = 20
    
    sig_high = sigma[1]/sub_sample
    sig_low = sigma[0]/sub_sample
    
    projections = projections[:,::sub_sample,border:-border:sub_sample]
    reprojections = reprojections[:,::sub_sample,border:-border:sub_sample]
    
    for i in range(projections.shape[0]):
        image_a = projections[i]
        image_b = reprojections[i]

        if i == 0:
            plt.figure()
            plt.subplot(1,2,1); plt.imshow(image_a)
            plt.subplot(1,2,2); plt.imshow(image_b)
            plt.show()
        
        #shift, error, diffphase = register_translation(image_a, image_b)
        
        for i in range(projections.shape[0]):
            a = np.sum(image_a,0)
            b = np.sum(image_b,0)

            coma = utils.ndimage.measurements.center_of_mass(a)
            comb = utils.ndimage.measurements.center_of_mass(b)
            
            shifts[0,i] = coma[0]-comb[0]
            
        # print("frame_shift:", shift)
        
        #shifts[0,i] = shift[0]
        #shifts[1,i] = shift[1]
    #shifts *= sub_sample
    return shifts



# Alignment in the x-direction with cross-correlation
class HorizontalAlignment():
    def __init__(self, projections_aligned_vert_fix2):
        self.projections_aligned_vert_fix2 = projections_aligned_vert_fix2
        self.correction_rough = self.horizontal_alignment()


    def calculate_correlation(self):
        projections_aligned_vert_fix2 = self.projections_aligned_vert_fix2
        quick_xcorrelation = np.zeros((2,projections_aligned_vert_fix2.shape[0]))
                                    
        for gag in range(1,projections_aligned_vert_fix2.shape[0]):
            a = (projections_aligned_vert_fix2[gag-1,:,:])#np.sum(projections[i,50:-50,250:750],0)
            b = (projections_aligned_vert_fix2[gag,:,:])#np.sum(reprojections2[i,50:-50,250:750],0)
            
            shift_ab = register_translation(a,b,upsample_factor=100)
            quick_xcorrelation[1,gag] = shift_ab[0][1] + quick_xcorrelation[1,gag-1]

        plt.figure(figsize=[4,4])
        plt.plot(quick_xcorrelation[1,:])
        return quick_xcorrelation
    
    # This function uses the shifts calculated before to "move" the projections
    def apply_reprojection_shifts(self, projections_in, shifts, pad=0):
        print("apply_reprojection_shifts")
        projections_out = np.zeros_like(projections_in)
        projections_pad = np.copy(utils.pad(projections_in, (pad,0)))
        shifts = np.round(shifts).astype(np.int32)
        
    #     print("projections_in.shape", projections_in.shape)
    #     print("projections_out.shape", projections_out.shape)
    #     print("projections.shape", projections.shape)

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

    # apply the shifts from cross-correlation to the projections
    def horizontal_alignment(self):
        projections_aligned_vert_fix2 = self.projections_aligned_vert_fix2
        quick_xcorrelation = self.calculate_correlation()
        import scipy
        correction_rough = self.apply_reprojection_shifts(projections_aligned_vert_fix2, -scipy.ndimage.gaussian_filter1d(quick_xcorrelation,10), pad=1)
        plt.imshow(projections_aligned_vert_fix2[:,200,:])
        plt.figure(),plt.imshow(correction_rough[:,200,:])
        return correction_rough
