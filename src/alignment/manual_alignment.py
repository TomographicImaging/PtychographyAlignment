# This file contains the manual alignment procedure from "src\pipeline\pipeline_based_on_original_notebook.ipynb"
# The idea was to attempt to generalise the method. WIP
#----------------------------------------------------------------------------------------------------------------

import matplotlib.pyplot as plt
import numpy as np
# Load the saved array
projections_aligned_vert = np.load("projections_aligned_vert.npy")

def get_subset(projections_aligned_vert, p1, p2):
    # Manual trial-and-error step to fix any large misalignments
    subset = projections_aligned_vert[p1:p2,projections_aligned_vert.shape[1]//2,:]
    return subset


def plot_subset(subset_, title = ''):
    plt.figure(figsize=[10,10])
    plt.imshow(subset_)
    plt.title(title)
    plt.show()

###-----------------------------------------------------------------------------------------------------------------
proj1 = 500
proj2 = 550
x1 = 800
x2 = 1400

subset = get_subset(projections_aligned_vert,proj1,proj2)
plot_subset(subset[:,x1:x2], 'No correction')

def correct_subset(subset1,misaligned_proj_,rough_correction_,error_):
    subset1[misaligned_proj_:proj2-proj1,:] = np.roll(subset[misaligned_proj_:proj2-proj1,:],rough_correction_ - error_,axis=1)
    return subset1

misaligned_proj = 21 # with respect to proj1
rough_correction = 10 # pixels
error = 5 # pixels

subset1 = np.copy(subset)
subset1 = correct_subset(subset1, 21, 10, error)
print(subset1)
print(subset.shape)
# Manual trial-and-error step to fix any large misalignments

print(subset1[:,x1:x2].shape)
plot_subset(subset1[:,x1:x2],'Correction of '+str(rough_correction - error))

subset2 = np.copy(subset)
subset2 = correct_subset(subset2, 21, 10, 0)
plot_subset(subset2[:,x1:x2], 'Correction of '+str(rough_correction))

subset3 = np.copy(subset)
subset3 = correct_subset(subset3, 21, 10, -error)
plot_subset(subset3[:,x1:x2], 'Correction of '+str(rough_correction + error))

###-----------------------------------------------------------------------------------------------------------------
# vertical correction 
# Manual trial-and-error step to fix any large misalignments
correction_value = 10
misaligned_projection = proj1+misaligned_proj
projections_aligned_vert_fix = np.copy(projections_aligned_vert)
projections_aligned_vert_fix[misaligned_projection:-1,:,:] = np.roll(projections_aligned_vert[misaligned_projection:-1,:,:],correction_value,axis = 2)
plt.figure(figsize=[10,10])
plt.subplot(1,2,1), plt.imshow(projections_aligned_vert_fix[:,projections_aligned_vert_fix.shape[1]//2,:]), plt.title('Manually corrected')
plt.subplot(1,2,2), plt.imshow(projections_aligned_vert[:,projections_aligned_vert.shape[1]//2,:]), plt.title('Uncorrected')
plt.show()


###-----------------------------------------------------------------------------------------------------------------
# do any other manual corrections that you might need e.g. by copy-pasting the cells from above
# Manual trial-and-error step to fix any large misalignments (part 2)
proj1 = 700
proj2 = 800
x1 = 800
x2 = 1500
subset = projections_aligned_vert[proj1:proj2,projections_aligned_vert.shape[1]//2,x1:x2]
plot_subset(subset)

# Manual trial-and-error step to fix any large misalignments
misaligned_proj = 32 # with respect to proj1
rough_correction = 790 # pixels
error = 10 # pixels

subset = projections_aligned_vert[proj1:proj2,projections_aligned_vert.shape[1]//2,:]
plot_subset(subset[:,x1:x2], 'No correction')

subset1 = np.copy(subset)
subset1[misaligned_proj,:] = np.roll(subset[misaligned_proj,:],rough_correction - error,axis=0)
plot_subset(subset1[:,x1:x2], 'Correction of '+str(rough_correction - error))

subset2 = np.copy(subset)
subset2[misaligned_proj,:] = np.roll(subset[misaligned_proj,:],rough_correction,axis=0)
plot_subset(subset2[:,x1:x2], 'Correction of '+str(rough_correction))

subset3 = np.copy(subset)
subset3[misaligned_proj,:] = np.roll(subset[misaligned_proj,:],rough_correction + error,axis=0)
plot_subset(subset3[:,x1:x2], 'Correction of '+str(rough_correction + error))

###-----------------------------------------------------------------------------------------------------------------

# Manual trial-and-error step to fix any large misalignments
correction_value = 790

misaligned_projection = proj1+misaligned_proj

projections_aligned_vert_fix2 = np.copy(projections_aligned_vert_fix)
projections_aligned_vert_fix2[misaligned_projection,:,:] = np.roll(projections_aligned_vert_fix[misaligned_projection,:,:],correction_value,axis = 1)

plt.figure(figsize=[10,10])
plt.subplot(1,2,1), plt.imshow(projections_aligned_vert_fix2[:,projections_aligned_vert_fix2.shape[1]//2,:]), plt.title('Manually corrected')
plt.subplot(1,2,2), plt.imshow(projections_aligned_vert[:,projections_aligned_vert_fix.shape[1]//2,:]), plt.title('Uncorrected')
plt.show()
