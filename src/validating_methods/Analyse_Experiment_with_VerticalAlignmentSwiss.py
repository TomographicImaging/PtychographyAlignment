import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
from viewer.OpenViewer import OpenViewer
from CT_reconstruction.TomoRecon import get_volume
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss

# Load the saved array from the pollen_Volpe data (see "src\pipeline\pipeline_based_on_original_notebook.ipynb")
projections = np.load(r"data/experimental/data/projections_reduced.npy")
print("angle, vertical, horizontal = ",projections.shape)
angles_reduced = np.load(r"data/experimental/data/angles_reduced.npy")
print(angles_reduced)
OpenViewer(projections)

def apply_y_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:],shift, axis=0)
    return projections

## alignment
va= VerticalAlignmentSwiss(projections, roi_range = (100,600), max_shift=50, iterations=1, swap_xy = False, plotting = True, saving=False)
va.run_alignment()
delta_y_1D_final = va.delta_y_1D_final
projections_aligned = apply_y_shifts(projections,-delta_y_1D_final)
OpenViewer(projections_aligned)
print("yshifts are", delta_y_1D_final)

## uncomment for tomographic reconstruction
# volume_initial = get_volume(projections, angles_reduced, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
# OpenViewer(volume_initial)
# volume_aligned = get_volume(projections_aligned, angles_reduced, centre=None, pad=20, algorithm='GRIDREC', iterations=1)
# OpenViewer(volume_aligned)
