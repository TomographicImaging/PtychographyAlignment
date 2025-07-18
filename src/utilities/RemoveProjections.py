import numpy as np

def remove_stripes(projs, angles, threshold= 0.02):
    """
    Removes stripes caused by beam dumping every ~10 minutesmid_1 = projs.shape[1]//2"""
    mid_2 = projs.shape[2]//2
    line = np.abs(np.sum(projs[:,mid_1,:], axis = 1))

    cutoff = np.mean(line)+np.mean(line)*threshold
    good_projs = np.where(line<cutoff)
    projs_out = projs[np.array(good_projs),:,:][0]
    angles_out = angles[np.array(good_projs)][0]
    
    return projs_out, angles_out


def remove_stripes_oriol(projs, angles, threshold= 0.3):
    """
    The difference between this function below and the one above is that we normalise the sinogram before the stripe removal
    and that we use both the maximum and the median to determine which lines to remove"""

    mid_1 = projs.shape[1]//2
    pp = np.zeros_like(projs[:,0,:])
    projs_sum_sino = np.sum(projs[:,mid_1:mid_1+10,:],axis=1)
    for i in range(projs.shape[0]):
        pp[i,:] = projs_sum_sino[i,:]/np.mean(projs[i,0:100])
    
    line = np.abs(np.sum(pp[:,:], axis = 1))

    median = np.median(line)
    maximum = np.max(line)

    cutoff = median + (maximum-median)*threshold
    good_projs = np.where(line<cutoff)
    projs_out = projs[np.array(good_projs),:,:][0]
    angles_out = angles[np.array(good_projs)][0]
    
    return projs_out, angles_out, good_projs