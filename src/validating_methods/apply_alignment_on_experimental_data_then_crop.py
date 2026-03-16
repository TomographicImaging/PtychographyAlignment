# Load the saved arrays from the pollen data (see "src\pipeline\pipeline_based_on_original_notebook.ipynb")
# This script assumes that partial results (".npy") are saved in the folder "data/experimental/data/"
# The scope of the script is to apply the shift calculated with the alignment method in "src\validating_methods\Analyse_Experiment_with_VerticalAlignmentSwiss.py".
# The alignment is performed in both x and y. The result is cropped only after alignment. 
# We assess the results by reconstructing the data and looking with the viewer.
# ----------------------------------------------------------------------------------------------------------------

import sys, os
project_root = os.path.abspath(os.path.join(os.getcwd(), '.'))
print(project_root)
sys.path.append(project_root)

import numpy as np
from validating_methods.TomoRecon import get_volume
from src.viewer.OpenViewer import OpenViewer

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

def alignxy():
    projections_in = np.load(r"data/experimental/data/projections_in.npy")
    good_projs = np.load(r"data/experimental/data/good_projs.npy")
    projections_reduced_not_cropped = projections_in[good_projs,:,:][0]
    print("projections_reduced_not_cropped shape is",projections_reduced_not_cropped.shape)
    OpenViewer(projections_reduced_not_cropped)
    delta_x_1D = np.load(r"data/experimental/data/delta_x_1D_fullNth_1it.npy")
    #print(np.max(delta_x_1D))
    #print(delta_x_1D)
    projections_aligned_horiz = apply_x_shifts(projections_reduced_not_cropped,-delta_x_1D)
    OpenViewer(projections_aligned_horiz[1:,:,:])
    delta_y_1D = np.load(r"data/experimental/data/delta_y_1D_fullNth_1it.npy")
    projections_aligned_horiz_and_vert = apply_y_shifts(projections_aligned_horiz,-delta_y_1D)
    OpenViewer(projections_aligned_horiz_and_vert)
    # np.save(r"data/experimental/data/projections_aligned_horiz_and_vert.npy", projections_aligned_horiz_and_vert)
    return projections_aligned_horiz_and_vert

def crop_data(projections_aligned_horiz_and_vert):
    crop_window = [700,1430,1300,2300] # [y1, y2, x1, x2] where y is vertical and x is horizontal
    projections_crop = crop(projections_aligned_horiz_and_vert, crop_window)
    OpenViewer(projections_crop[1:,:,:])
    return projections_crop

def plot_intitial_data():
    angles_reduced = np.load(r"data/experimental/data/angles_reduced.npy")
    projections_reduced = np.load(r"data/experimental/data/projections_reduced.npy")
    volume_initial = get_volume(projections_reduced, angles_reduced, centre=None, pad=200, algorithm='GRIDREC', iterations=1)
    OpenViewer(volume_initial)

def plot_final_data(projections_crop):
    angles_reduced = np.load(r"data/experimental/data/angles_reduced.npy")
    # OpenViewer(volume_initial)
    # #volume_final = get_volume(projections_aligned_vert, angles_reduced, centre=None, pad=200, algorithm='GRIDREC', iterations=1)
    # #OpenViewer(volume_final)
    volume_horiz = get_volume(projections_crop[1:,:,:], angles_reduced[1:], centre=None, pad=200, algorithm='GRIDREC', iterations=1)
    OpenViewer(volume_horiz)

projections_aligned_horiz_and_vert = alignxy()
projections_crop = crop_data(projections_aligned_horiz_and_vert)
plot_intitial_data()
plot_final_data(projections_crop)