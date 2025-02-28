from skimage.restoration import unwrap
from OpenViewer import OpenViewer
# phase unwrapping. This takes a while, so just comment it if you don't need it. 

class Unwrap():
    def __init__(self, projections_in, parallel = True):
        self.parallel = parallel
        self.unwrap_scipy(projections_in)

    def open_viewer():
        OpenViewer(projections_in)

    def unwrap_scipy(self, phase_stack):
        if self.parallel == True:
            # Create a list of slices to pass to the executor
            slices = [phase_stack[i, ...] for i in range(phase_stack.shape[0])]
            from concurrent.futures import ProcessPoolExecutor
            # Parallelize using ProcessPoolExecutor
            with ProcessPoolExecutor() as executor:
                # Map the unwrap_slice function to each slice index
                # Map the unwrap_slice function to each slice
                results = executor.map(unwrap_slice, range(phase_stack.shape[0]), slices)
        else:
            for i in range(phase_stack.shape[0]):
                # Extract the slice and pass it to unwrap_slice
                unwrap_slice(phase_stack[i, ...])

def unwrap_slice(slice_data):
    # Unwrap only the slice data
    slice_data = unwrap.unwrap_phase(slice_data)

def test_unwrap():
    from Imports import ImportData
    ptytomofile = 'C:/Users/zvm34551/Coding_environment/DATA/Ptychography/pty_tomo_NX.h5'
    data = ImportData(ptytomofile)
    projections_raw= data.get_projections_raw()
    import numpy
    #projections_in = numpy.amin(projections_raw) - projections_raw
    Unwrap(projections_raw)

test_unwrap()