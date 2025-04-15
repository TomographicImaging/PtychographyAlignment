import numpy as np
from viewer.OpenViewer import OpenViewer
from CT_reconstruction.TomoRecon import get_volume
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss

# Load the saved array
projections_in = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_simulation_360_projections.npy")
np.log(projections_in, out=projections_in)
np.negative(projections_in,out=projections_in)
#print(projections_in[100,150:200,150:200])
#print(projections_in.shape)
delta_x_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_delta_x_360.npy")
delta_y_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_delta_y_360.npy")
#print(delta_y_sim)
#OpenViewer(projections_in)

start = 0
stop = 180 *np.pi/180
step = 0.5 *np.pi/180
angles = np.arange(start, stop, step)
#print(angles)
#print(len(angles))


#volume = get_volume(projections_in, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
#OpenViewer(volume)

# Load the saved array
projections_jitter = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_jitter_simulation_360_projections.npy")
np.log(projections_jitter, out=projections_jitter)
np.negative(projections_jitter,out=projections_jitter)
delta_x_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_jitter_delta_x_360.npy")
delta_y_sim = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250415\sphere_phantom_jitter_delta_y_360.npy")
print(delta_x_sim)
print(delta_y_sim)
#OpenViewer(projections_jitter)
#print(delta_x)
#print(delta_y)
#volume_initial = get_volume(projections_jitter, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
#OpenViewer(volume_initial)
swap_xy =False
va= VerticalAlignmentSwiss(projections_jitter, max_shift=60, swap_xy =swap_xy)
#va= VerticalAlignmentSwiss(projections_in, max_shift=50, swap_xy =swap_xy)
if swap_xy ==True:
    np.save(r"src\delta_x_1D.npy", va.delta_y_1D_final)
else:
    np.save(r"src\delta_y_1D.npy", va.delta_y_1D_final)