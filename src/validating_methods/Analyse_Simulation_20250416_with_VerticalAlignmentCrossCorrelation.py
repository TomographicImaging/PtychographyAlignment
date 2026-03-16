import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from viewer.OpenViewer import OpenViewer
from validating_methods.TomoRecon import get_volume
from alignment.Alignment import VerticalAlignmentCrossCorrelation
from config.paths import angles_path, projections_yjitter_path, projections_xyjitter_path, delta_x_sim_path, delta_y_sim_path

## Load the angles
angles = np.load(angles_path)
angles = angles *np.pi/180
print("angles", angles)

## Load the saved array
projections_xyjitter = np.load(projections_xyjitter_path)
OpenViewer(projections_xyjitter)

## optional extra operations on data
# np.log(projections_jitter, out=projections_jitter)
# np.negative(projections_jitter,out=projections_jitter)

## Load the simulated shifts
delta_x_sim = np.load(delta_x_sim_path)
delta_y_sim = np.load(delta_y_sim_path)
## divide the y shifts (this is unknown why)
delta_y_sim = 0.5*delta_y_sim 
print(delta_x_sim, "deltaxsim")
print(delta_y_sim, "deltaysim")

# align 
vacc = VerticalAlignmentCrossCorrelation(projections_xyjitter)
projections_aligned, yshifts = vacc.projections_aligned, vacc.vertical_shifts
diff =delta_y_sim - yshifts
print("yshifts are",yshifts)
print("diff are",diff)
OpenViewer(projections_aligned)

## apply tomographic reconstruction
OpenViewer(projections_xyjitter)
volume_initial = get_volume(projections_xyjitter, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
OpenViewer(volume_initial)
volume_aligned = get_volume(projections_aligned, angles, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
OpenViewer(volume_aligned)

## uncomment to do the same on y shifted data
# projections_yjitter = np.load(projections_yjitter_path)
# OpenViewer(projections_yjitter)