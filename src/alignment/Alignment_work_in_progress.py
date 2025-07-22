# Author: Oriol
# These methods are only used in the "work in progress" section of Oriol's original code.
# -----------------------------------------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import  measurements
from skimage.registration import phase_cross_correlation as register_translation

def sin_func(x, a, w, b, c):
    """Used by COMAlignment and work in progress."""
    return a * np.sin(w*x + b) + c

def calculate_com_projs(projections):
    #plt.figure()
    
    projections -= np.amin(projections)
    summed = np.sum(projections[:,:,:],1)
    
    com = np.zeros([2,projections.shape[0]])
    shifts = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = measurements.center_of_mass(summed[i:i+1,:])
    
    return com

def calculate_com_projs_oriol(projections):
    #plt.figure()
    
    projections -= np.amin(projections)
    summed = np.sum(projections[:,:,:],1)
    
    com = np.zeros([2,projections.shape[0]])
    shifts = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = measurements.center_of_mass((summed[i:i+1,:]))
    
    return com

def get_reprojection_shifts(projections, reprojections, sub_sample=1, sigma=[4,20]):
    """This function gives the shifts between projections and reprojections in terms of x and y"""
    shifts = np.zeros([2, projections.shape[0]], dtype = np.float32)
    
    border = 20
    
    sig_high = sigma[1]/sub_sample
    sig_low = sigma[0]/sub_sample
    
    projections = projections[:,::sub_sample,border:-border:sub_sample]
    reprojections = reprojections[:,::sub_sample,border:-border:sub_sample]
    
#     projections = blur(projections, sig_low) - blur(projections, sig_high)
#     reprojections = blur(reprojections, sig_low) - blur(reprojections, sig_high)
    
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

def get_reprojection_shifts_com(projections, reprojections, sub_sample=1, sigma=[4,20]):
    """This function gives the shifts between projections and reprojections in terms of x and y"""
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

            coma = measurements.center_of_mass(a)
            comb = measurements.center_of_mass(b)
            
            shifts[0,i] = coma[0]-comb[0]
            
        # print("frame_shift:", shift)
        
        #shifts[0,i] = shift[0]
        #shifts[1,i] = shift[1]
    #shifts *= sub_sample
    return shifts
