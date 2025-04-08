# reconstruct now
import numpy as np
from CT_reconstruction.TomoRecon import get_volume
from viewer.OpenViewer import OpenViewer

def apply_y_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:],shift, axis=0)
    return projections
# Load the saved array
projections_reduced = np.load(r"src\projections_reduced.npy")
angles_reduced = np.load(r"src\angles_reduced.npy")
delta_y_1D = np.load(r"src\delta_y_1D_fullNth_1it.npy")

volume_initial = get_volume(projections_reduced, angles_reduced, centre=None, pad=200, algorithm='GRIDREC', iterations=1)
OpenViewer(projections_reduced)

projections_aligned_vert = apply_y_shifts(projections_reduced,delta_y_1D)
OpenViewer(projections_aligned_vert)
OpenViewer(volume_initial)
volume_final = get_volume(projections_aligned_vert, angles_reduced, centre=None, pad=200, algorithm='GRIDREC', iterations=1)
OpenViewer(volume_initial)
OpenViewer(volume_final)