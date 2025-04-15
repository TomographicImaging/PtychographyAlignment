from skimage.restoration import unwrap
from tests.TestData import TestData

class Unwrap():
    def __init__(self, projections_in, parallel = True):
        self.parallel = parallel
        self.unwrap(projections_in)

    def unwrap(self, phase_stack):
        if self.parallel == True:
            from joblib import Parallel, delayed
            # Run in parallel, passing only necessary slices
            Parallel(n_jobs=-1, prefer="threads")(delayed(unwrap_slice)(slice_data) for slice_data in phase_stack)

        else:
            # original method
            #for i in range(phase_stack.shape[0]):
            #    phase_stack[i,...] = unwrap.unwrap_phase(phase_stack[i,...])
            
            #there is a 3d method implemented already
            phase_stack = unwrap.unwrap_phase(phase_stack)
            
def unwrap_slice(slice_data):
    # Unwrap only the slice data in place
    slice_data[:] = unwrap.unwrap_phase(slice_data)


def test_unwrap():
    from viewer.OpenViewer import OpenViewer
    data = TestData().data
    import numpy 
    numpy.set_printoptions(suppress=True)
    print(data[0,0])
    print(data.shape)
    OpenViewer(data)
    Unwrap(data)
    OpenViewer(data)

    from io_module.Imports import ImportData
    ptytomofile = 'C:/Users/zvm34551/Coding_environment/DATA/Ptychography/pty_tomo_NX.h5'
    data = ImportData(ptytomofile)
    projections_raw= data.get_projections_raw()
    OpenViewer(projections_raw)
    Unwrap(projections_raw)
    OpenViewer(projections_raw)
    

#test_unwrap()