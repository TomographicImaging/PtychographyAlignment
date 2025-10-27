from skimage.restoration import unwrap

class Unwrap():
    def __init__(self, projections_in, parallel = True, sliced = True):
        self.parallel = parallel
        self.sliced = sliced
        self.unwrap(projections_in)

    def unwrap(self, phase_stack):
        """
        Unwraps a 3D stack of wrapped phase data.

        Behavior depends on the configuration of `self.parallel` and `self.sliced`:
        
        - If `self.parallel` is True: runs slice-by-slice unwrapping in parallel using joblib.
        - If `self.parallel` is False and `self.sliced` is True: unwraps each 2D slice sequentially in place.
        - Otherwise: applies native 3D unwrapping using `skimage.restoration.unwrap_phase`.

        Parameters
        ----------
        phase_stack : ndarray
            A 3D NumPy array of shape (Z, Y, X) representing unwrapped phase data.
        """
        if self.parallel == True:
            from joblib import Parallel, delayed
            # Run in parallel, passing only necessary slices
            Parallel(n_jobs=-1, prefer="threads")(delayed(unwrap_slice)(slice_data) for slice_data in phase_stack)

        else:
            if self.sliced == True:
                for i in range(phase_stack.shape[0]):
                    phase_stack[i,...] = unwrap.unwrap_phase(phase_stack[i,...])
            else:
                phase_stack[:] = unwrap.unwrap_phase(phase_stack)
            
def unwrap_slice(slice_data):
    """
    Unwraps a single 2D slice of wrapped phase data in place.

    This function modifies the input slice by applying `skimage.restoration.unwrap_phase`
    and writing the result back into the original array.

    Parameters
    ----------
    slice_data : ndarray
        A 2D NumPy array representing a single unwrapped phase slice.
    """
    slice_data[:] = unwrap.unwrap_phase(slice_data)

