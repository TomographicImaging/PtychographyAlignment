import numpy as np

class RemoveProjections():
    def __init__(self, projections, angles, threshold = 0.3):
        self.projections = projections
        self.angles = angles
        self.threshold = threshold
        projections_out, angles_out, good_projections = self.remove_stripes(projections, angles, threshold)
        
        return projections_out, angles_out, good_projections
    
    
    def remove_stripes(projections, angles, threshold):
        """
        The difference between this function below and the one above is that we normalise the sinogram before the stripe removal
        and that we use both the maximum and the median to determine which lines to remove
        
        Parameters
        ----------
        projections : 3D stack of projections
        angles : array of theta angles of acquisition
        threshold : threshold determined by user to establish the cutoff
        
        Returns
        -------
        projections_out : the projections that we want to keep
        angles_out : the corresponding angles
        good_projections : indices corresponding to the projections_out
        
        """
    
        mid_1 = projections.shape[1]//2
        pp = np.zeros_like(projections[:,0,:])
        projs_sum_sino = np.sum(projections[:,mid_1:mid_1+10,:],axis=1)
        for i in range(projections.shape[0]):
            pp[i,:] = projs_sum_sino[i,:]/np.mean(projections[i,0:100])
        
        line = np.abs(np.sum(pp[:,:], axis = 1))
    
        median = np.median(line)
        maximum = np.max(line)
        cutoff = median + (maximum-median)*threshold
        
        good_projections = np.where(line<cutoff)
        projections_out = projections[np.array(good_projections),:,:][0]
        angles_out = angles[np.array(good_projections)][0]
        
        return projections_out, angles_out, good_projections
    
    
    # def remove_stripes(projs, angles, threshold= 0.02):
    #     """
    #     Removes stripes caused by beam dumping every ~10 minutes"""
    #     mid_1 = projs.shape[1]//2
    #     mid_2 = projs.shape[2]//2
    #     line = np.abs(np.sum(projs[:,mid_1,:], axis = 1))
    
    #     cutoff = np.mean(line)+np.mean(line)*threshold
    #     good_projs = np.where(line<cutoff)
    #     projs_out = projs[np.array(good_projs),:,:][0]
    #     angles_out = angles[np.array(good_projs)][0]
        
    #     return projs_out, angles_out