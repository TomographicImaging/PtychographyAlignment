import numpy as np
from viewer.OpenViewer import OpenViewer
from CT_reconstruction.TomoRecon import get_volume
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss

# Load the saved array
#projections_in = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_simulation_360_projections.npy")
#np.log(projections_in, out=projections_in)
#np.negative(projections_in,out=projections_in)
#print(projections_in[100,150:200,150:200])
#print(projections_in.shape)
#delta_x_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_delta_x_360.npy")
#delta_y_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_delta_y_360.npy")
#print(delta_y_sim)
#OpenViewer(projections_in)

# start = 0
# stop = 180 *np.pi/180
# step = 0.5 *np.pi/180
# angles = np.arange(start, stop, step)
#print(angles)
#print(len(angles))
angles = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_angles_361.npy")
angles = angles *np.pi/180
#print(angles)
#volume = get_volume(projections_in, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
#OpenViewer(volume)

# # Load the saved array
#projections_yjitter = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_yjitter_simulation_361_projections.npy")
# # Load the saved array
projections_xyjitter = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_jitter_simulation_361_projections.npy")

# np.log(projections_jitter, out=projections_jitter)
# np.negative(projections_jitter,out=projections_jitter)
# delta_x_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_jitter_delta_x_360.npy")
delta_y_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_jitter_delta_y_361.npy")
delta_y_sim = 0.5*delta_y_sim #this is unknown why
# #print(delta_x_sim)
print(delta_y_sim)
#OpenViewer(projections_yjitter)
OpenViewer(projections_xyjitter)
# def apply_x_shifts(projections, trans):
#     for th in range(projections.shape[0]):
#         shift = int(trans[th])
#         projections[th,:,:] = np.roll(projections[th,:,:],shift, axis=1)
#     return projections

def apply_y_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:],shift, axis=0)
    return projections

from scipy import signal
def xcorf(in1, in2):
    xcor = signal.correlate(in1, in2, mode='same')
    shift = np.argmax(xcor)-(xcor.shape[0]/2)
    return shift, xcor

def get_y_shifts(projections):
    proj_sum = np.sum(projections,2)
    shifts = np.zeros(projections.shape[0])
    xcor_ar = np.zeros_like(proj_sum)
    a = np.gradient(proj_sum[0,:])
    for i in range(projections.shape[0]):
        b = np.gradient(proj_sum[i,:])
        shifts[i], xcor_ar[i,:] = xcorf(a,b) #
    return shifts

yshifts = get_y_shifts(projections_xyjitter)
diff =delta_y_sim - yshifts
print("yshifts are",yshifts)
print("diff are",diff)
projections_aligned = apply_y_shifts(projections_xyjitter,yshifts)
OpenViewer(projections_aligned)
# #OpenViewer(projections_jitter)
# #print(delta_x)
# print("simulation delta y",delta_y_sim)
#volume_initial = get_volume(projections_yjitter, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
#OpenViewer(volume_initial)
#volume_aligned = get_volume(projections_aligned, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
#OpenViewer(volume_aligned)
# swap_xy =False
# va= VerticalAlignmentSwiss(projections_yjitter, max_shift=20, iterations=5,swap_xy =swap_xy)
# # # #va= VerticalAlignmentSwiss(projections_in, max_shift=50, swap_xy =swap_xy)
# if swap_xy ==True:
#     np.save(r"src\delta_x_1D.npy", va.delta_y_1D_final)
# else:
#     np.save(r"src\delta_y_1D.npy", va.delta_y_1D_final)

#delta_y_1D_final = np.load(r"src\delta_y_1D.npy")
#projections_aligned = apply_y_shifts(projections_yjitter,-delta_y_1D_final)
#OpenViewer(projections_aligned)
