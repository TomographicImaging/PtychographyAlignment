# reconstruct now
import numpy as np
from CT_reconstruction.TomoRecon import get_volume
from viewer.OpenViewer import OpenViewer

def apply_y_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:],shift, axis=0)
    return projections

def apply_x_shifts(projections, trans):
    for th in range(projections.shape[0]):
        shift = int(trans[th])
        projections[th,:,:] = np.roll(projections[th,:,:],shift, axis=1)
    return projections

def crop(projections, crop_window):
    projections_crop = projections[:,crop_window[0]:crop_window[1],crop_window[2]:crop_window[3]]
    return projections_crop

# Load the saved array
projections_in = np.load(r"src\projections_in.npy")
good_projs = np.load(r"src\good_projs.npy")
projections_reduced_not_cropped = projections_in[good_projs,:,:][0]
print("projections_reduced_not_cropped shape is",projections_reduced_not_cropped.shape)
OpenViewer(projections_reduced_not_cropped)
delta_x_1D = np.load(r"src\delta_x_1D_fullNth_1it.npy")
#print(np.max(delta_x_1D))
#print(delta_x_1D)
projections_aligned_horiz = apply_x_shifts(projections_reduced_not_cropped,-delta_x_1D)
OpenViewer(projections_aligned_horiz[1:,:,:])
delta_y_1D = np.load(r"src\delta_y_1D_fullNth_1it.npy")
projections_aligned_horiz_and_vert = apply_y_shifts(projections_aligned_horiz,-delta_y_1D)
OpenViewer(projections_aligned_horiz_and_vert)
crop_window = [700,1430,1300,2300] # [y1, y2, x1, x2] where y is vertical and x is horizontal
projections_crop = crop(projections_aligned_horiz_and_vert, crop_window)
OpenViewer(projections_crop[1:,:,:])
projections_reduced = np.load(r"src\projections_reduced.npy")
angles_reduced = np.load(r"src\angles_reduced.npy")






volume_initial = get_volume(projections_reduced, angles_reduced, centre=None, pad=200, algorithm='GRIDREC', iterations=1)
OpenViewer(volume_initial)



# 
# OpenViewer(volume_initial)
# #volume_final = get_volume(projections_aligned_vert, angles_reduced, centre=None, pad=200, algorithm='GRIDREC', iterations=1)
# #OpenViewer(volume_final)
volume_horiz = get_volume(projections_crop[1:,:,:], angles_reduced[1:], centre=None, pad=200, algorithm='GRIDREC', iterations=1)


OpenViewer(volume_horiz)

