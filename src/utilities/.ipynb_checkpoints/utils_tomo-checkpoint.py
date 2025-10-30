# author: Oriol
#
# This file contains extra methods, the ones are used in our
# repository have been moved out in other files.
#-------------------------------------------------------------------

import sys
sys.path.append('/dls_sw/apps/tomopy/tomopy/src')

import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')
logger = logging.getLogger()
logger.setLevel(logging.CRITICAL)

import h5py
import numpy as np
import glob

sys.path.append('/dls_sw/apps/pycho/ptyrex_3.4.0/')
#import ptyrex

from skimage.restoration import unwrap

import h5py
try:
    import tomopy
    from tomopy.recon.rotation import find_center
    from tomopy import recon
except ImportError as e:
    print(e)
import numpy as np
from matplotlib import pyplot as plt
#from scipy import math

try:
    from skimage import data
    from skimage.registration import phase_cross_correlation as register_translation
    #from skimage.feature.register_translation import _upsampled_dft
    from scipy.ndimage import fourier_shift
except ImportError as e:
    print(e)
#import cv2
    
try:
    from recast_tomo import FP2D_ASTRA
    from recast_tomo import FBP2D_ASTRA
    from recast_tomo import SIRT2D_ASTRA
except ImportError as e:
    print(e)
    
from scipy import ndimage
from scipy import signal
import os
import glob
import datetime
import time

#from clusterMPI import rotateRegisterShiftMPI
from multiprocessing import Process


try:
    from mpi4py import MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nprocs = comm.Get_size()
except ImportError:
    rank = 0
    nprocs = 1
import utilities.utils_used as utils_used
from CT_reconstruction import TomoRecon
#import utils

def check_crop(proj_dir, proj_centre = None, proj_shape = None):
    all_hdf_files = glob.glob(proj_dir + '*.hdf')
    all_hdf_files.sort()
    n_files = len(all_hdf_files)
    
    hdf_file_list = []
    projections = []
    
    for proj_id in [0, int((n_files-1)/2), n_files-1]:
#         print(proj_id)
        filename = glob.glob(proj_dir + '*_%d_*.hdf' % proj_id)[0]
#         print(filename)
        hdf_file_list.append(filename)
    
    for hdf_file in hdf_file_list:
#         print("Current file: %s" %hdf_file)
        with h5py.File(hdf_file,'r') as f:
            if proj_centre == None:
                projections.append(np.copy(np.array(f['entry_1/process_1/output_1/object_phase'])))
            else:
                array = np.array(f['entry_1/process_1/output_1/object_phase'])
                projections.append(np.copy(ptyrex.core.toolbox.cut2(array, proj_shape, proj_centre)))

    proj_count = 0
    plt.figure(figsize=(10,5))
    for proj in projections:
        plt.subplot(1,3,proj_count+1)
        plt.imshow(np.squeeze(proj))
        proj_count += 1
    plt.show()

def plot_projection(projections, idx):
    plt.figure(figsize=[10,10])
    plt.imshow(projections[idx,:,:], aspect='auto', cmap='gray')
    plt.show()

def plot_3_projections(projections):
    ind = range(np.size(projections, 0))
    ind_mid = int(np.median(ind))

    proj_1 = projections[0, :, :]
    proj_2 = projections[ind_mid, :, :]
    proj_3 = projections[-1]

    plt.figure(figsize=[15,10])
    plt.subplot(1,3,1)
    plt.imshow(proj_1, cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(proj_2, cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(proj_3, cmap='gray')
    plt.show()
    
def plot_sinogram(projections, y_slice=None):
    if y_slice == None:
        y_slice = int(projections.shape[1]/2)
    plt.figure(figsize=[5,5])
    plt.imshow(projections[:,y_slice,:], aspect='auto', cmap='gray')
    plt.show()
    
def plot_sinogram_sum(projections):
    plt.figure(figsize=[5,5])
    plt.imshow(np.sum(projections[:,:,:],2), aspect='auto', cmap='gray')
    plt.show()

def get_sinogram_com(projections):
    projections -= np.amin(projections)
    com = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = ndimage.measurements.center_of_mass(projections[i,:,:])
    return com

def plot_sinogram_com(projections):
    com = get_sinogram_com(projections)
    plt.figure(figsize=[5,5])
    plt.subplot(2,1,1)
    plt.plot(com[0,:])
    plt.subplot(2,1,2)
    plt.plot(com[1,:])
    plt.show()

def plot_sinogram_ft_com(projections):
    projections_ft = np.abs(np.fft.fftshift(np.fft.fft2( np.fft.fftshift(projections, axes=(-1,-2)) ), axes=(-1,-2)))
    com = get_sinogram_com(projections_ft)
    plt.figure(figsize=[5,5])
    plt.subplot(2,1,1)
    plt.plot(com[0,:])
    plt.subplot(2,1,2)
    plt.plot(com[1,:])
    plt.show()

def shift_sinogram_ft(projections, shifts):
    shift = np.zeros([1,1,2])
    proj_out = np.zeros_like(projections)
    for i in range(projections.shape[0]):
        shift[:,:,0] = shifts[0][i]
        shift[:,:,1] = shifts[1][i]
        proj_out[i] = ptyrex.core.toolbox.shift_ft_gen(projections[i],shift)
    return proj_out

def plot_sinogram_sum_com(projections):
    plt.figure(figsize=[5,5])
    projections -= np.amin(projections)
    summed = np.sum(projections[:,:,:],2)
    com = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = ndimage.measurements.center_of_mass(summed[i:i+1,:])
    plt.subplot(2,1,1)
    plt.plot(com[0,:])
    plt.subplot(2,1,2)
    plt.plot(com[1,:])
    diff = com[1,:] - np.mean(com[1,:])
    #plt.plot(diff)
    plt.show()
    
def plot_sinogram_sum(projections):
    plt.figure(figsize=[5,5])
    summed = np.sum(projections[:,:,:],1)
    com = np.zeros([2,projections.shape[0]])
    for i in range(projections.shape[0]):
        com[:,i] = ndimage.measurements.center_of_mass(summed[i:i+1,:])
    plt.subplot(2,1,1)
    plt.plot(com[1,:])
    plt.subplot(2,1,2)
    diff = com[1,:] - np.mean(com[1,:])
    plt.plot(diff)
    plt.show()

def plot_trans():
    import matplotlib.pyplot as plt
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(trans[0,:])
    plt.subplot(2,1,2)
    plt.plot(trans[1,:])
    plt.show()
    
def plot_volume(volume, x_slice = None, y_slice=None, z_slice=None):
    if x_slice == None:
        x_slice = int(volume.shape[0]/2)
    if y_slice == None:
        y_slice = int(volume.shape[1]/2)
    if z_slice == None:
        z_slice = int(volume.shape[2]/2)
    plt.figure(figsize=[15,5])
    plt.subplot(1,3,1)
    plt.imshow(volume[x_slice,:,:], aspect='auto', cmap='gray')
    plt.subplot(1,3,2)
    plt.imshow(volume[:,y_slice,:], aspect='auto', cmap='gray')
    plt.subplot(1,3,3)
    plt.imshow(volume[:,:,z_slice], aspect='auto', cmap='gray')
    plt.show()
    
def reconstruct(projections,angles,algorithm,CoR,iterations=1):
    if algorithm=='FBP':
        volume=FBP2D_ASTRA(projections,angles,CoR)
    elif algorithm == 'SIRT':
        volume=SIRT2D_ASTRA(projections,angles,CoR,iterations)
    elif algorithm == 'GRIDREC':
        volume = tomopy.recon(projections, angles, algorithm='gridrec', center=CoR)
    return volume
    
def get_reprojections(volume, angles):
    reprojections = FP2D_ASTRA(volume, angles)
    return reprojections
    
def get_translations(projections, reprojections):
    shifts = np.zeros([2, projections.shape[0]])
    for i in range(projections.shape[0]):
        shifts[:,i], error, diffphase = register_translation(projections[i,:,:], reprojections[i,:,:], upsample_factor=100)
    return shifts

def get_rmse(projections, reprojections):
    diff = np.abs(projections - reprojections)**2
    mean = np.mean(diff,axis = (-2,-1))
    return np.sqrt(mean)
    
def apply_translations(projections, trans):
    shifted_projections = np.zeros_like(projections)
    for i in range(projections.shape[0]):
        M1 = np.float32([[1,0,-trans[1,i]],[0,1,-trans[0,i]]])
        shifted_projections[i,:,:] = cv2.warpAffine(projections[i,:,:],M1,(projections.shape[2],projections.shape[1]))
    return shifted_projections

def normProjection(img,fromX,toX,fromY,toY,mean):
    newMean=np.mean(img[fromY:toY,fromX:toX])
    diff=mean-newMean
    imgNorm=img+np.ones(np.shape(img))*diff
    return imgNorm

def normTomo(projections,fromX,toX,fromY,toY):
    # projections_sums = np.sum(projections,(1,2))
    # projections_norms = np.mean(projections_sums)/projections_sums
    # for i in range(projections.shape[0]):
    #     projections[i,:,:] *= projections_norms[i]
    
#     mean_values=projections[:,fromY:toY,fromX:toX]
    mean_values = np.mean(projections[:,fromY:toY,fromX:toX], axis=(1,2), keepdims=True)
    np.subtract(projections, mean_values, out=projections)
    
def normTomo_modulus(projections,fromX,toX,fromY,toY):
    # projections_sums = np.sum(projections,(1,2))
    # projections_norms = np.mean(projections_sums)/projections_sums
    # for i in range(projections.shape[0]):
    #     projections[i,:,:] *= projections_norms[i]
    
#     mean_values=projections[:,fromY:toY,fromX:toX]
    mean_values = np.mean(projections[:,fromY:toY,fromX:toX], axis=(1,2), keepdims=True)
    np.divide(projections, mean_values, out=projections)

def smoothData(dataSmall):
    a,b,c=np.shape(dataSmall)
    test=np.zeros(np.shape(dataSmall))
    kernel = np.ones((15,15),np.float32)/25
    for m in range(b):
        test[:,m,:]=cv2.filter2D(dataSmall[:,m,:],-1,kernel)
    return test

def rotateMatrix(rec,M,cols,rows,height):
    for l in range (height):
        dst = cv2.warpAffine(rec[:,l,:],M,(cols,rows))
        rec[:,l,:]=dst
    return rec

def myRec(obj,continueLoop,pathTot,dataFolder):  
    ### recursive function to look for the data database
    temp=None
    i=1
    tempPath=''
    for name, value in obj.items():
        if continueLoop:
            #check if the object is a group
            if isinstance(obj[name], h5py.Group):
                tempPath='/'+name
                if len(obj[name])>0:
                    continueLoop,temp,tempPath= myRec(obj[name],continueLoop,tempPath,dataFolder)
                else:
                    continue
            else:
                test=obj[name]
                temp1='/'+dataFolder
                if temp1 in test.name:
                    continueLoop=False
                    tempPath=pathTot+'/'+name
                    return continueLoop,test.name,tempPath
            i=i+1
        if (i-1)>len(obj.items()):
            tempPath=''
    pathTot=pathTot+tempPath
    return continueLoop,temp, pathTot

def fullPath(folder,fileNr,year=''):
    if year=='':
        now = datetime.datetime.now()
        year=str(now.year)
    else:
        year=str(year)
        directory='/dls/i13-1/data/'+year+'/'+folder+'/processing/tomo/'+str(fileNr)+'/'
    return directory

def get_shift(img1,img2):
    img1 = np.float32(img1)
    img2 = np.float32(img2)
    mapper = cv2.reg_MapperGradShift()
    mappPyr = cv2.reg_MapperPyramid(mapper)

    resMap = mappPyr.calculate(img1, img2)
    mapShift = cv2.reg.MapTypeCaster_toShift(resMap)
    shift_out = mapShift.getShift()

    return shift_out

def xcor(in1, in2):
    xcor = signal.correlate(in1, in2, mode='same')
    shift = np.argmax(xcor)-(xcor.shape[0]/2)
    return shift, xcor

# def get_x_shifts(projections):
#     proj_sum = np.sum(projections,1)
#     shifts = np.zeros(projections.shape[0])
#     xcor_ar = np.zeros_like(proj_sum)
#     for i in range(projections.shape[0]):
#         if i > 0:
#             a = np.gradient(projections[i-1,:,:], axis=1)
#             b = np.gradient(projections[i,:,:], axis=1)
#             shift, error, diffphase = register_translation(a, b, 1)
# #             print shift
#             shifts[i] = shift[1]
#     return shifts

def get_x_shifts(projections):
    proj_sum = np.sum(projections,1)
    shifts = np.zeros(projections.shape[0])
    xcor_ar = np.zeros_like(proj_sum)
    a = np.gradient(proj_sum[0,:])
    for i in range(projections.shape[0]):
        b = np.gradient(proj_sum[i,:])
        shifts[i],xcor_ar[i,:] = xcor(a,b)
    return shifts

def apply_x_shifts(projections, trans):
    for i in range(projections.shape[0]):
        shift = int(trans[i])
        projections[i,:,:] = np.roll(projections[i,:,:],shift, axis=1)
    return projections

def get_y_shifts(projections):
    proj_sum = np.sum(projections,2)
    shifts = np.zeros(projections.shape[0])
    xcor_ar = np.zeros_like(proj_sum)
    a = np.gradient(proj_sum[0,:])
    for i in range(projections.shape[0]):
        b = np.gradient(proj_sum[i,:])
        shifts[i],xcor_ar[i,:] = xcor(a,b)
    return shifts

def get_y_shifts_com(projections):
    proj_sum = np.sum(projections,2)
    shifts = np.zeros(projections.shape[0])
    xcor_ar = np.zeros_like(proj_sum)
#     a = np.gradient(proj_sum[0,:])
    a = ndimage.center_of_mass(proj_sum[0,:])
    for i in range(projections.shape[0]):
#         b = np.gradient(proj_sum[i,:])
        b = ndimage.center_of_mass(proj_sum[i,:])
#         shifts[i],xcor_ar[i,:] = xcor(a,b)
        shifts[i] = b-a
    return shifts

def apply_y_shifts(projections, trans):
    for i in range(projections.shape[0]):
        shift = int(trans[i])
        projections[i,:,:] = np.roll(projections[i,:,:],shift, axis=0)
    return projections

def recon_align(projections, angles, algorithm = 'GRIDREC', iters = 1):
    projections_aligned = np.copy(projections)
    
    trans = np.zeros([2, projections_aligned.shape[0]])
    
    for n in range(iters):
        print(n)
    #     alpha = 1#(float(n_iterations)-float(n))/n_iterations
    #     print alpha
#         centre_of_rotation=utils_used.findCentre(projections[0,:,:],projections[-1,:,:])
        
        # Get volume
        volume = TomoRecon.tomo_recon(projections_aligned,angles,algorithm=algorithm,iterations=iters)
        # Generate simulated projections
        reprojections = get_reprojections(volume, angles)
        # Get translations
        trans += get_translations(projections_aligned[:,crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]], reprojections[:,crop_y[0]:crop_y[1], crop_x[0]:crop_x[1]])
        # Update original projections
        projections_aligned[:] = projections_aligned[:]*(1-alpha) + apply_translations(projections, trans)*(alpha)
        # Record error
        rmse = np.sqrt(np.mean(np.abs(projections_aligned - projections)**2))
        print(rmse)
    return volume, projections_aligned

        
def tomo_align(projections_in, angles, iters=1, subsample=1):
    projections = np.copy(projections_in)
    trans = np.zeros([2, projections.shape[0]])
    
    for n in range(iters):
        print("Iteration %d" %(n+1))
        alpha = 1#(float(n_iterations)-float(n))/n_iterations
    #     print alpha

        centre_of_rotation=utils_used.findCentre(projections)
        volume = TomoRecon.tomo_recon(projections, angles, centre=centre_of_rotation, algorithm='GRIDREC', iterations=1)
        
        reprojections = get_reprojections(volume, angles)
        
#        trans += get_translations(projections, reprojections)
        trans += get_reprojection_shifts(projections, reprojections, subsample)
        projections[:] = projections*(1-alpha) + apply_projection_shifts(projections_in, trans)*(alpha)

        rmse = np.sqrt(np.mean(np.abs(projections - projections_in)**2))
        print("Error (RMSE):", rmse)
    return volume, projections

"""
def make_tomoNX(raw_dir, proj_dir, out_path, proj_type = 'phase', proj_centre=None, proj_shape=None):
    all_hdf_files = glob.glob(proj_dir + '/*.hdf')
    all_hdf_files.sort()

    hdf_file = all_hdf_files[0]
    print("Taking probe shape from: %s"%hdf_file)
    with h5py.File(hdf_file,'r', swmr=True) as f:
        probe_shape = np.shape(f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])
        object_shape = np.shape(f['/entry_1/process_1/output_1/object_modulus'][:,:,:,0,0,:,:])
    
    if proj_shape == None:
        proj_shape = np.array(object_shape[-2:]) - 40
    
    if proj_centre == None:
        proj_centre = proj_shape // 2
    
    print("raw_dir:", raw_dir)
    print("proj_dir:", proj_dir)
    print("proj_centre:", proj_centre)
    print("proj_shape:", proj_shape)
    
    with h5py.File(out_path, 'w') as tomo_nx:
        print('Writing tomoNX file..')
        
        theta_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/rotation_angle', (len(all_hdf_files),), 'f')
        data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', (len(all_hdf_files), proj_shape[0], proj_shape[1]), 'f')

        probe_out_shape = (len(all_hdf_files), probe_shape[0]*probe_shape[1], probe_shape[-2], probe_shape[-1])
        probe_out_shape_single = (1, probe_shape[0]*probe_shape[1], probe_shape[-2], probe_shape[-1])
        probe_modulus_nx = tomo_nx.create_dataset('/entry1/pty_entry/probe/modulus', probe_out_shape, 'f')
        probe_phase_nx = tomo_nx.create_dataset('/entry1/pty_entry/probe/phase', probe_out_shape, 'f')

        #idx_nx = tomo_nx.create_dataset('/entry1/tomo_entry/instrument/detector/image_key', (len(all_hdf_files),), 'd')        
        idx = 0

        for hdf_file in all_hdf_files:
            hdf_file_split = hdf_file.split('/')
            hdf_file_split = hdf_file_split[-1].split('_')
            #print(hdf_file_split)
            tomo_id = hdf_file_split[1]
            proj_id = hdf_file_split[2]

            #print(tomo_id, proj_id)
            #idx_nx[idx] = idx        

            print(hdf_file)

            zebra_filename = '%s/pty_tomo.h5' %(raw_dir)
            with h5py.File(zebra_filename,'r') as f:
                theta_nx[idx] = np.mean(np.array(f['/data/scan'][int(proj_id),:,0]))
                #print('tomo_id:', tomo_id, '- proj_id:', proj_id, '- theta:', np.array(theta_nx[idx]), '- idx:', idx)
                
                
            with h5py.File(hdf_file,'r', swmr=True) as f:
                if proj_type == 'phase':
                    # print('Putting proj %s, angle %f, at slice %d' %(proj_id, idx_list[idx]))
                    data_nx[idx, :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_phase']), proj_shape, proj_centre)
                elif proj_type == 'modulus':
                    data_nx[idx, :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_modulus']), proj_shape, proj_centre)
                else:
                    print("ERROR: proj_type must be 'phase' or 'modulus'")
                probe_modulus_nx[idx, ...] = np.array((f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])).reshape(probe_out_shape_single)
                probe_phase_nx[idx, ...] = np.array((f['/entry_1/process_1/output_1/probe_phase'][:,:,:,0,0,:,:])).reshape(probe_out_shape_single)
                
            idx += 1

        # print("theta_nx", theta_nx)
        sort_idx = list(np.argsort(theta_nx))
        # print("sort_idx", sort_idx)

        # idx_list = list(np.argsort(sort_idx))
        # idx = 0
        # print("idx_list", idx_list)
        
        # data_nx = data_nx[idx_list]
        # temp_array = np.copy(data_nx)
        
        # for i in range(data_nx.shape[0]):
        #     data_nx[i] = temp_array[idx_list[i]]
        
        # for hdf_file in all_hdf_files:
        #     hdf_file_split = hdf_file.split('/')
        #     hdf_file_split = hdf_file_split[-1].split('_')
        #     #print(hdf_file_split)
        #     tomo_id = hdf_file_split[1]
        #     proj_id = hdf_file_split[2]

        #     with h5py.File(hdf_file,'r', swmr=True) as f:
        #         if proj_type == 'phase':
        #             print('Putting proj %s, angle %f, at slice %d' %(proj_id, idx_list[idx]))
        #             data_nx[idx_list[idx], :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_phase']), proj_shape, proj_centre)
        #         elif proj_type == 'modulus':
        #             data_nx[idx_list[idx], :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_modulus']), proj_shape, proj_centre)
        #         else:
        #             print("ERROR: proj_type must be 'phase' or 'modulus'")
                
        #         #probe_entry[idx,:,:] = np.sum(np.array((f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])),(0,1)).squeeze()
                
        #         probe_modulus_nx[sort_idx[idx], ...] = np.array((f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])).reshape(probe_out_shape_single)
        #         probe_phase_nx[sort_idx[idx], ...] = np.array((f['/entry_1/process_1/output_1/probe_phase'][:,:,:,0,0,:,:])).reshape(probe_out_shape_single)


            # idx += 1
"""

def make_tomoNX(raw_dir, proj_dir, out_path, proj_type = 'phase', proj_centre=None, proj_shape=None):
    all_hdf_files = glob.glob(proj_dir + '/*.hdf')
    all_hdf_files.sort()

    hdf_file = all_hdf_files[0]
    print("Taking probe shape from: %s"%hdf_file)
    with h5py.File(hdf_file,'r', swmr=True) as f:
        probe_shape = np.shape(f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])
        object_shape = np.shape(f['/entry_1/process_1/output_1/object_modulus'][:,:,:,0,0,:,:])
    
    if proj_shape == None:
        proj_shape = np.array(object_shape[-2:]) - 40
    
    if proj_centre == None:
        proj_centre = proj_shape // 2
    
    print("raw_dir:", raw_dir)
    print("proj_dir:", proj_dir)
    print("proj_centre:", proj_centre)
    print("proj_shape:", proj_shape)
    
    with h5py.File(out_path, 'w') as tomo_nx:
        print('Writing tomoNX file..')
        
        theta_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/rotation_angle', (len(all_hdf_files),), 'f')
        data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', (len(all_hdf_files), proj_shape[0], proj_shape[1]), 'f')

        probe_out_shape = (len(all_hdf_files), probe_shape[0]*probe_shape[1], probe_shape[-2], probe_shape[-1])
        probe_out_shape_single = (1, probe_shape[0]*probe_shape[1], probe_shape[-2], probe_shape[-1])
        probe_modulus_nx = tomo_nx.create_dataset('/entry1/pty_entry/probe/modulus', probe_out_shape, 'f')
        probe_phase_nx = tomo_nx.create_dataset('/entry1/pty_entry/probe/phase', probe_out_shape, 'f')

        idx_nx = tomo_nx.create_dataset('/entry1/tomo_entry/instrument/detector/image_key', (len(all_hdf_files),), 'd')        
        idx = 0

        for hdf_file in all_hdf_files:
            hdf_file_split = hdf_file.split('/')
            hdf_file_split = hdf_file_split[-1].split('_')
            #print(hdf_file_split)
            tomo_id = hdf_file_split[1]
            proj_id = hdf_file_split[2]

            #print(tomo_id, proj_id)
            idx_nx[idx] = idx        

            print(hdf_file)

            with h5py.File(hdf_file,'r', swmr=True) as f:
                if proj_type == 'phase':
                    data_nx[idx, :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_phase']), proj_shape, proj_centre)
                elif proj_type == 'modulus':
                    data_nx[idx, :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_modulus']), proj_shape, proj_centre)
                else:
                    print("ERROR: proj_type must be 'phase' or 'modulus'")
                
                #probe_entry[idx,:,:] = np.sum(np.array((f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])),(0,1)).squeeze()
                
                probe_modulus_nx[idx, ...] = np.array((f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])).reshape(probe_out_shape_single)
                probe_phase_nx[idx, ...] = np.array((f['/entry_1/process_1/output_1/probe_phase'][:,:,:,0,0,:,:])).reshape(probe_out_shape_single)

            zebra_filename = '%s/pty_tomo.h5' %(raw_dir)
            with h5py.File(zebra_filename,'r') as f:
                theta_nx[idx] = np.mean(np.array(f['/data/scan'][int(proj_id),:,0]))
                #print('tomo_id:', tomo_id, '- proj_id:', proj_id, '- theta:', np.array(theta_nx[idx]), '- idx:', idx)

            idx += 1

 
def make_tomoNX_vds(raw_dir, proj_dir, out_path, proj_centre=None, proj_shape=None):
    all_hdf_files = glob.glob(proj_dir + '/*.hdf')
    all_hdf_files.sort()

    hdf_file = all_hdf_files[0]
    print("Taking probe shape from: %s"%hdf_file)
    with h5py.File(hdf_file,'r', swmr=True) as f:
        probe_shape = np.shape(f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])
        object_shape = np.shape(f['/entry_1/process_1/output_1/object_modulus'][:,:,:,0,0,:,:])
    
    if proj_shape == None:
        proj_shape = np.array(object_shape[-2:]) - 10
    
    if proj_centre == None:
        proj_centre = proj_shape // 2
    
    print("raw_dir:", raw_dir)
    print("proj_dir:", proj_dir)
    print("proj_centre:", proj_centre)
    print("proj_shape:", proj_shape)
    
#    theta_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/rotation_angle', (len(all_hdf_files),), 'f')
#    data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', (len(all_hdf_files), proj_shape[0], proj_shape[1]), 'f')
#    idx_nx = tomo_nx.create_dataset('/entry1/tomo_entry/instrument/detector/image_key', (len(all_hdf_files),), 'd')   

    layout_data = h5py.VirtualLayout((len(all_hdf_files), proj_shape[0], proj_shape[1]),dtype=np.float32)
    layout_probe = h5py.VirtualLayout((len(all_hdf_files), probe_shape[0], proj_shape[-2], proj_shape[-1]),dtype=np.float32)
    layout_theta = h5py.VirtualLayout((len(all_hdf_files),),dtype=np.float32)

    vsource_list = []
    for hdf_file in all_hdf_files:
        hdf_file_split = hdf_file.split('/')
        hdf_file_split = hdf_file_split[-1].split('_')
        #print(hdf_file_split)
        tomo_id = hdf_file_split[1]
        proj_id = hdf_file_split[2]

        #print(tomo_id, proj_id)
#            idx_nx[idx] = idx        

        print(hdf_file)

        vsource_data_list.append(h5py.VirtualSource(hdf_file, 'entry_1/process_1/output_1/object_phase'))
        zebra_filename = '%s/pty_tomo.h5' %(raw_dir)
        vsource_theta_list.append(h5py.VirtualSource(zebra_filename, '/data/scan'))
                
                
#    with h5py.File(out_path, 'w') as tomo_nx:
#        print('Writing tomoNX file..')
              
    idx = 0

    for i in len(vsource_data_list):
        idx_nx[idx] = idx   
        layout_data[idx, :, :] = vsource_data_list[i][crop_from[0]:crop_to[0], crop_from[1]:crop_to[1]]
        layout_theta[idx] = vsource_theta_list[i][int(proj_id),:,0].mean()
        idx += 1
            
    with h5py.File(out_path, 'w', libver='latest') as f:        
        f.create_virtual_dataset('/entry1/tomo_entry/data/data', layout_data, fillvalue=0)
        f.create_virtual_dataset('/entry1/tomo_entry/data/rotation_angle', layout_theta, fillvalue=0)

def make_tomoNX_mpi(raw_dir, proj_dir, out_path, proj_centre=None, proj_shape=None):
    all_hdf_files = glob.glob(proj_dir + '/*.hdf')
    all_hdf_files.sort()

    hdf_file = all_hdf_files[0]
    print("Taking probe shape from: %s"%hdf_file)
    with h5py.File(hdf_file,'r', swmr=True) as f:
        probe_shape = np.shape(f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])
        object_shape = np.shape(f['/entry_1/process_1/output_1/object_modulus'][:,:,:,0,0,:,:])
    
    if proj_shape == None:
        proj_shape = np.array(object_shape[-2:]) - 10
    
    if proj_centre == None:
        proj_centre = proj_shape // 2
    
    print("raw_dir:", raw_dir)
    print("proj_dir:", proj_dir)
    print("proj_centre:", proj_centre)
    print("proj_shape:", proj_shape)
    
    with h5py.File(out_path, 'w') as tomo_nx:
        print('Writing tomoNX file..')
        
        theta_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/rotation_angle', (len(all_hdf_files),), 'f')
        data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', (len(all_hdf_files), proj_shape[0], proj_shape[1]), 'f')
        idx_nx = tomo_nx.create_dataset('/entry1/tomo_entry/instrument/detector/image_key', (len(all_hdf_files),), 'd')        
        idx = 0

        zebra_filename = '%s/pty_tomo.h5' %(raw_dir)
        with h5py.File(zebra_filename,'r') as f:
           theta = np.mean(np.array(f['/data/scan'][:,:,0]))

    """
        for hdf_file in all_hdf_files:
            
            def get_output_from_pty(pty_file, raw_file):
                
            hdf_file_split = hdf_file.split('/')
            hdf_file_split = hdf_file_split[-1].split('_')
            #print(hdf_file_split)
            tomo_id = hdf_file_split[1]
            proj_id = hdf_file_split[2]

            #print(tomo_id, proj_id)
            idx_nx[idx] = idx        

            print(hdf_file)

            with h5py.File(hdf_file,'r', swmr=True) as f:
                data_nx[idx, :, :] = ptyrex.core.toolbox.cut2(np.array(f['entry_1/process_1/output_1/object_phase']), proj_shape, proj_centre)
#                 probe_entry[idx,:,:] = np.sum(np.array((f['/entry_1/process_1/output_1/probe_modulus'][:,:,:,0,0,:,:])),(0,1)).squeeze()

            
            theta_nx[idx] = theta[proj_id]
                #print('tomo_id:', tomo_id, '- proj_id:', proj_id, '- theta:', np.array(theta_nx[idx]), '- idx:', idx)

            idx += 1
    """

def sort_tomoNX(out_path):
    with h5py.File(out_path, 'r+', swmr=True) as tomo_nx:
        print('Sorting tomoNX file..')

        theta_nx = tomo_nx['/entry1/tomo_entry/data/rotation_angle']
        data_nx = tomo_nx['/entry1/tomo_entry/data/data']
        idx_nx = tomo_nx['/entry1/tomo_entry/instrument/detector/image_key']
        
        probe_entry = tomo_nx['/entry1/pty_entry/probe/modulus']

        theta_ar = np.array(theta_nx)
        data_ar = np.array(data_nx)
        idx_ar = np.array(idx_nx)
        probe_ar = np.array(probe_entry)

        sort_idx = list(np.argsort(theta_ar))

        theta_ar = theta_ar[sort_idx]
        idx_ar = idx_ar[sort_idx]
        data_ar = data_ar[sort_idx, :, :]
        probe_ar = probe_ar[sort_idx, :, :]

        del tomo_nx['/entry1/tomo_entry/data/rotation_angle']
        del tomo_nx['/entry1/tomo_entry/data/data']
        del tomo_nx['/entry1/tomo_entry/instrument/detector/image_key']
        del tomo_nx['/entry1/pty_entry/probe/modulus']

        theta_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/rotation_angle', data = theta_ar)
        data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', data = data_ar)
        idx_nx = tomo_nx.create_dataset('/entry1/tomo_entry/instrument/detector/image_key', data = idx_ar)
        
        probe_entry = tomo_nx.create_dataset('/entry1/pty_entry/probe/modulus', data = probe_ar)


"""
def sort_tomoNX(out_path):
    with h5py.File(out_path, 'r+') as tomo_nx:
        print('Sorting tomoNX file..')

        theta_nx = tomo_nx['/entry1/tomo_entry/data/rotation_angle']
        data_nx = tomo_nx['/entry1/tomo_entry/data/data']
        idx_nx = tomo_nx['/entry1/tomo_entry/instrument/detector/image_key']
        
        probe_entry = tomo_nx['/entry1/pty_entry/probe/modulus']

        # copying to memory
        theta_ar = np.array(theta_nx)
        
        #data_ar = np.array(data_nx)
        #idx_ar = np.array(idx_nx)
        #probe_ar = np.array(probe_entry)

        sort_idx = list(np.argsort(theta_ar))

        count = 0
        for i in sort_idx:
            theta_nx[count] = theta_nx[i]
            idx_nx[count] = idx_nx[i]
            data_nx[count,:,:] = data_nx[i, :, :]
            probe_entry[count,:,:] = probe_entry[i, :, :]

        #del tomo_nx['/entry1/tomo_entry/data/rotation_angle']
        #del tomo_nx['/entry1/tomo_entry/data/data']
        #del tomo_nx['/entry1/tomo_entry/instrument/detector/image_key']
        #del tomo_nx['/entry1/pty_entry/probe/modulus']

        #theta_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/rotation_angle', data = theta_ar)
        #data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', data = data_ar)
        #idx_nx = tomo_nx.create_dataset('/entry1/tomo_entry/instrument/detector/image_key', data = idx_ar)
        
        #probe_entry = tomo_nx.create_dataset('/entry1/pty_entry/probe/modulus', data = probe_ar)
"""

def plot_extreme_angles(tomo_dir):
    with h5py.File(tomo_dir, 'r') as f:
        mid = np.int(np.round(np.shape(f['entry1/tomo_entry/data/data'])[0]/2))-1
        last = np.int(np.round(np.shape(f['entry1/tomo_entry/data/data'])[0]))-1
        index = [0,mid,last]
        projections = f['entry1/tomo_entry/data/data'][index,:,:]
    
    plt.figure()
    plt.subplot(1,3,1)
    plt.imshow(projections[0,:,:])
    plt.subplot(1,3,2)
    plt.imshow(projections[1,:,:])
    plt.subplot(1,3,3)
    plt.imshow(projections[2,:,:])
    plt.show()


def unwrap_scipy(phase_stack):
    for i in range(phase_stack.shape[0]):
        phase_stack[i,...] = unwrap.unwrap_phase(phase_stack[i,...])

def unwrap_stack_mpi(phase_stack, method='scipy'):
    try:
        from mpi4py import MPI
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        nprocs = comm.Get_size()
    except ImportError:
        rank = 0
        nprocs = 1
    
    if rank == 0:
        n_frames = phase_stack.shape[0]
        frames_per_rank = int(np.ceil(n_frames/nprocs))
        
        current_idx_start = np.arange(0, n_frames, frames_per_rank)[rank]
        
        phase_stack_list = []
        for i in current_idx_start:
            phase_stack_list.append(phase_stack[current_idx_start:current_idx_start+frames_per_rank,...])
        
    comm.scatter(phase_stack_list, root=0)
    
    unwrap_scipy(phase_stack_list)
    
    phase_stack_list = comm.gather(phase_stack_list, root=0)
    
    if rank == 0:
        counter = 0
        for i in current_idx_start:
            phase_stack[current_idx_start:current_idx_start+frames_per_rank,...] = phase_stack_list[counter]
            counter += 1
    print("phase_stack.shape",phase_stack.shape)
    
    
def unwrap_tomoNX(out_path):
    with h5py.File(out_path, 'r+') as tomo_nx:
        print('Unwrapping tomoNX file..')

        data_nx = tomo_nx['/entry1/tomo_entry/data/data']
        data_ar = np.array(data_nx)

        wrap_list=[]
        unwrap_list=[]
        for i in range(data_ar.shape[0]):
            if 1:#np.amax(np.diff(data_ar[i,:,:])) >= 6.27:
                data_ar[i,:,:] = unwrap.unwrap_phase(data_ar[i,:,:])
                wrap_list.append(i)
            else:
                unwrap_list.append(i)

        tomo_nx.create_dataset('/entry1/tomo_entry/data/data_unwrapped', data = data_ar)
        print(len(wrap_list),"wrapped projections")
        print(len(unwrap_list),"non-wrapped projections")
        
def unwrap_tomoNX_mpi(out_path):
    with h5py.File(out_path, 'r+', driver='mpio', comm=MPI.COMM_WORLD) as tomo_nx:
        print('Opening tomoNX file..')
        data_nx = tomo_nx['/entry1/tomo_entry/data/data']
        print('Filename:', out_path)
        print('Data shape', data_nx.shape)
        tomo_nx.create_dataset('/entry1/tomo_entry/data/data_unwrapped', data = data_nx)
        
        n_frames = data_nx.shape[0]
        frames_per_rank = int(np.ceil(n_frames/nprocs))
        
        current_idx_start = np.arange(0, n_frames, frames_per_rank)
        i = current_idx_start[rank]
        data_nx[i:i+frames_per_rank,...] = unwrap_scipy(data_nx[i:i+frames_per_rank,...])
            
            
#    phase_stack_list = []
#    for i in current_idx_start:
#        data_nx = tomo_nx['/entry1/tomo_entry/data/data']
#        phase_stack_list.append(data_nx[i:i+frames_per_rank,...])
#    
#    print('Process %d, phase_stack_list length: %d' %(rank, len(phase_stack_list)))
#
#    print('Scattering jobs..')
#    comm.scatter(phase_stack_list, root=0)
#
#    print('Process %d, phase_stack_list length: %d' %(rank, len(phase_stack_list)))
#
#    print('Executing job on rank %d' %rank)
#    unwrap_scipy(phase_stack_list)
#
#    print('Gathering jobs..')
#    phase_stack_list = comm.gather(phase_stack_list, root=0)
#    
#    if rank == 0:
#        counter = 0
#        for i in current_idx_start:
#            print('Moving results into the output array..')
#            data_nx[i:i+frames_per_rank,...] = phase_stack_list[counter]
#            counter += 1
#    
#    with h5py.File(out_path, 'r+') as tomo_nx:
#        print('Saving tomoNX file..')
#
#        tomo_nx.create_dataset('/entry1/tomo_entry/data/data_unwrapped', data = data_nx)


def drift_correction(data_ar, vertical=0.0, horizontal=0.0, tilt=0.0):
    n_proj = data_ar.shape[0]

    vert_shift_delta = vertical / n_proj
    horz_shift_delta = horizontal / n_proj
    tilt_shift_delta = tilt / n_proj

    print(n_proj, "projections")
    print("vert, horz, and tilt delta shifts:")
    print(vert_shift_delta)
    print(horz_shift_delta)
    print(tilt_shift_delta)

    count = 0
    for i in range(0, n_proj):
        shift_vert = i * vert_shift_delta
        shift_horz = i * horz_shift_delta
        shift_tilt = i * tilt_shift_delta
        
        data_ar[i:i+1, :, :] = apply_y_shifts(data_ar[i:i+1, :, :], [shift_vert])
        data_ar[i:i+1, :, :] = apply_x_shifts(data_ar[i:i+1, :, :], [shift_horz])

        
        
#def drift_correction(data_ar, vertical=0.0, horizontal=0.0, tilt=0.0):
#    n_proj = data_ar.shape[0]
#
#    vert_shift_delta = vertical / n_proj
#    horz_shift_delta = horizontal / n_proj
#    tilt_shift_delta = tilt / n_proj
#
#    print(n_proj, "projections")
#    print("vert, horz, and tilt delta shifts:")
#    print(vert_shift_delta)
#    print(horz_shift_delta)
#    print(tilt_shift_delta)
#
#    count = 0
#    for i in range(0, n_proj):
#        shift_vert = i * vert_shift_delta
#        shift_horz = i * horz_shift_delta
#        shift_tilt = i * tilt_shift_delta
#
#        shift_lateral = np.float32([[1, 0, shift_horz], [0, 1, shift_vert]])
#        data_ar[i, :, :] = cv2.warpAffine(data_ar[i, :, :], shift_lateral, (data_ar.shape[2], data_ar.shape[1]))
#
#        axes = (obj.shape[2], obj.shape[1])
#        M = cv2.getRotationMatrix2D((data_ar.shape[2] / 2, data_ar.shape[1] / 2), shift_tilt, 1)
#        data_ar[i, :, :] = cv2.warpAffine(data_ar[i, :, :], M, axes)

def remove_frames(data_ar, angles, rem_list):
    keep_list = []
    for i in range(data_ar.shape[0]):
        if i not in rem_list:
            keep_list.append(i)

    data_ar = data_ar[keep_list, ...]
    angles = angles[keep_list,...]
    
    return data_ar, angles

def remove_failed_frames_tomoNX(out_path):
    with h5py.File(out_path, 'r+') as tomo_nx:
        print('Removing erroneous projections from tomoNX file..')

        data_nx = tomo_nx['/entry1/tomo_entry/data/data']
        data_ar = np.array(data_nx)
        del tomo_nx['/entry1/tomo_entry/data/data']
        tomo_nx.create_dataset('/entry1/tomo_entry/data/data_unwrapped', data = data_ar)

        keep_list = []
        rem_list = []
        for i in range(data_ar.shape[0]):
#             print(i)
            if np.amax(np.diff(data_ar[i,:,:])) < 6.27:
                keep_list.append(i)
            else:
                print("Remove frame", i)
                rem_list.append(i)

        print(len(keep_list),"passed the test")
        print(len(rem_list),"failed the test")
        data_ar = data_ar[keep_list, :, :]
        data_nx = tomo_nx.create_dataset('/entry1/tomo_entry/data/data', data = data_ar)

def get_projection_shifts(projections, sub_sample=1):
    shifts = np.zeros([2, projections.shape[0]])
    for i in range(projections.shape[0]):
        if i > 0:
            image_a = projections[i-1,::sub_sample,::sub_sample]
            image_b = projections[i,::sub_sample,::sub_sample]
    
            shift, error, diffphase = register_translation(image_a, image_b, upsample_factor=1)
            
            shifts[0,i] = shift[0] + shifts[0,i-1]
            shifts[1,i] = shift[1] + shifts[1,i-1]
#        if i > 0:
#            shifts[0,i] = shift[0] + shifts[0,i-1]
#            shifts[1,i] = shift[1] + shifts[1,i-1]
#        else:
#            shifts[0,i] = shift[0]
#            shifts[1,i] = shift[1]
#         print("Shift at image %d is:" %i, shifts[:,i])
    shifts*=sub_sample
    return shifts

def get_reprojection_shifts(projections, reprojections, sub_sample=1):
    shifts = np.zeros([2, projections.shape[0]])
    for i in range(projections.shape[0]):
        
        image_a = projections[i,::sub_sample,::sub_sample]
        image_b = reprojections[i,::sub_sample,::sub_sample]

        shift, error, diffphase = register_translation(image_a, image_b, upsample_factor=1)
        
        shifts[0,i] = shift[0]
        shifts[1,i] = shift[1]
#         print("Shift at image %d is:" %i, shifts[:,i])
    shifts*=sub_sample
    return shifts
        
def apply_projection_shifts(projections_in, shifts):
    projections_out = np.zeros_like(projections_in)
    shifts = np.round(shifts).astype(np.int32)
    for i in range(projections_in.shape[0]):
        projections_out[i] = np.roll(projections_in[i], shifts[0,i], axis=0)
        projections_out[i] = np.roll(projections_in[i], shifts[1,i], axis=1)
    return projections_out

def projection_align_vertical(projections):
    #plt.figure(figsize=(10,10))
    #plt.subplot(2,2,1)
    #plt.imshow(np.sum(projections,2))
    #plt.subplot(2,2,2)
    #plt.imshow(np.sum(projections,1))

    y_trans = get_y_shifts(projections)
    projections = apply_y_shifts(projections, y_trans)

    #plt.subplot(2,2,3)
    #plt.imshow(np.sum(projections,2))
    #plt.subplot(2,2,4)
    #plt.imshow(np.sum(projections,1))
    #plt.show()

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

def plot_shifts(shifts):
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(shifts[0])
    plt.subplot(2,1,2)
    plt.plot(shifts[1])
    
def load_data(nxsfileName, data_key= '/entry1/tomo_entry/data/data_unwrapped', angle_key = '/entry1/tomo_entry/data/rotation_angle', probe_key = '/entry1/pty_entry/probe/', y_idx=[0,None], x_idx=[0,None], angle_idx=[0,None,1]):


    data_key_name = 'data'
    with h5py.File(nxsfileName,'r') as data_file: 
        data_shape = data_file[str(data_key)].shape
        print('raw data shape:', data_shape)


        projections=np.array(data_file[str(data_key)][angle_idx[0]:angle_idx[1]:angle_idx[2],y_idx[0]:y_idx[1]:1,x_idx[0]:x_idx[1]:1])#*(-1)
        np.nan_to_num(projections, copy=False)

        angles = np.array(data_file[angle_key][angle_idx[0]:angle_idx[1]:angle_idx[2]])
        #projections -= np.amin(projections)
        angles -= np.amin(angles)

        angles_rad = angles * np.pi/180.0

        #projections = np.pad(projections, ((0,0), (0,0), (50,50)), 'edge')

        probes = None
        if probe_key in data_file.keys():
            probes = np.array(data_file[probe_key+'modulus']) * np.exp(1j* np.array(data_file[probe_key+'phase']))

        print('projections shape:', projections.shape)
        print('angles shape:', angles_rad.shape)
    return projections, angles_rad, probes


def remove_stripes(projs, angles, threshold= 0.02):
    mid_1 = projs.shape[1]//2
    mid_2 = projs.shape[2]//2
    line = np.abs(np.sum(projs[:,mid_1,:], axis = 1))

    cutoff = np.mean(line)+np.mean(line)*threshold
    good_projs = np.where(line<cutoff)
    projs_out = projs[np.array(good_projs),:,:][0]
    angles_out = angles[np.array(good_projs)][0]
    return projs_out, angles_out