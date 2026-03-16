import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from viewer.OpenViewer import OpenViewer
from validating_methods.TomoRecon import get_volume
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss
from config.paths import angles_path, projections_yjitter_path, projections_xyjitter_path, delta_x_sim_path, delta_y_sim_path

## Load the angles
angles = np.load(angles_path)
angles = angles *np.pi/180
print("angles", angles)

## Load the saved array
projections_yjitter = np.load(projections_yjitter_path)
OpenViewer(projections_yjitter)

## Load the simulated shifts
delta_x_sim = np.load(delta_x_sim_path)
delta_y_sim = np.load(delta_y_sim_path)
## divide the y shifts (this is unknown why)
delta_y_sim = 0.5*delta_y_sim 
print(delta_x_sim, "deltaxsim")
print(delta_y_sim, "deltaysim")

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


## this esult is not actually very good. The method can be improved.
va= VerticalAlignmentSwiss(projections_yjitter, roi_range = None, max_shift=20, iterations=5,swap_xy = False,plotting = True, saving=False)
va.run_alignment()
delta_y_1D_final = va.delta_y_1D_final
projections_aligned = apply_y_shifts(projections_yjitter,-delta_y_1D_final)
OpenViewer(projections_aligned)
diff =delta_y_sim - delta_y_1D_final
print("yshifts are", delta_y_1D_final)
print("diff are",diff)