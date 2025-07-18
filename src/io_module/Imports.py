import h5py
import hdf5plugin #necessary to import nxs files
#from hd5plugin import Blosc
import time
import utilities.utils_tomo as utils 

class ImportData():
    def __init__(self, filepath, data_key, angle_key):
        """
        Imports a hdf5 file and saves the projections, angles and probes as attributes.
        Times loading.
        
        Parameters
        ----------
        filepath: path of the hdf5 file
        data_key: key for the reconstructed ptycho-tomography data
        angle_key: corresponding angles for the tomography
        """

        with h5py.File(filepath, "r") as f:
            print(f.keys())

        tic = time.time()
        self.projections_raw, self.angles, self.probes = utils.load_data(filepath, data_key, 
                                                        angle_key, 
                                                        angle_idx=[0,None,1])
        toc = time.time()

        print('Angles go from '+str(self.angles[0])+' to '+str(self.angles[-1])+'.')
        print('Loading dataset took '+str(toc-tic)+' seconds.')

    def get_projections_raw(self):
        """Returns projections."""
        return self.projections_raw
    
    def get_angles(self):
        """Returns angles."""
        return self.angles

    def get_probes(self):
        """Returns probes"""
        return self.probes 

