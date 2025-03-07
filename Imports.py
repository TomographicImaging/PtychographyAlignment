import h5py
import time
import utils_tomo as utils 

class ImportData():
    def __init__(self, filepath):

        with h5py.File(filepath, "r") as f:
            print(f.keys())

        # Upload the dataset. This step will take a while (~minutes) depending on the size of the dataset. 
        tic = time.time()
        self.projections_raw, self.angles, self.probes = utils.load_data(filepath, data_key='/entry1/tomo_entry/data/data', 
                                                        angle_key = '/entry1/tomo_entry/data/rotation_angle', 
                                                        angle_idx=[0,None,1]) #data_key=''/entry/data/data'
        toc = time.time()

        print('Angles go from '+str(self.angles[0])+' to '+str(self.angles[-1])+'.')
        print('Loading dataset took '+str(toc-tic)+' seconds.')

    def get_projections_raw(self):
        return self.projections_raw
    
    def get_angles(self):
        return self.angles

    def get_probes(self):
        return self.probes 


def test_import_data():
    ptytomofile = 'C:/Users/zvm34551/Coding_environment/DATA/Ptychography/pty_tomo_NX.h5'
    ImportData(ptytomofile)

#test_import_data()
