import numpy as np
import itertools
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss

def _create_data():
    """# Define dimensions
    # (slices, height, width)
    # Create a 1D array from -2π to 2π
    # Repeat the 1D array across the 3D volume (nx x ny x nz)
    """
    (nx, ny, nth) = (2, 3, 4)  
    #phase_1d = np.linspace(-1* np.pi, 1 * np.pi, nth)
    phase_1d = np.linspace(0, 3, nth)
    
    phase_stack = np.tile(phase_1d, (nx, ny, 1))
    return phase_stack

def _create_psi():
    A =np.array([[[0,0,0],[1,1,1],[2,2,2]],[[10,10,10],[11,11,11],[12,12,12]],[[100,100,100],[101,101,101],[102,102,102]]])
    psi = A[:,:,0]
    return psi


def test_shift_psi_by_array_and_mean():
    psi = _create_psi()
    projections = _create_data()
    va= VerticalAlignmentSwiss(projections)
    psi_shifted, mean = va.shift_psi_by_array_and_mean(psi, np.array([2,0,1]))
    psi_shfted_expected = np.array([[2, 0, 1], [10, 11,12], [101, 102, 100]])
    mean_expected = np.array([37.66666667, 37.66666667, 37.66666667])
    assert np.array_equal(psi_shifted, psi_shfted_expected)
    assert np.allclose(mean, mean_expected)

def _test_shift_psi_by_array(self, psi):
    """WIP.
    This method was useful to test on real experimental data. On the simulated data it is fairly ueless."""
    Nth, Ny = psi.shape
    delta_y_1D = np.zeros(Nth, dtype=int)  # Initialize shifts to zero
    for theta in range(Nth//2):  # Process each projection separately
        delta_y_1D[theta] = 200
    # Compute the average projection over all θ, using the current shifts
    shifted_projections, avg = shift_psi_by_array(psi, delta_y_1D)
    #print("avg is ",avg)
    self.plot_1D(avg,"Average over the projection angle theta",self.align, "Average")
    self.plot_array(np.transpose(shifted_projections), f"psi_theta({self.align}) shifted",  'projection #', self.align)

def test_compute_error():
    """Simulates psi and calculates the error. Compares it to the expected result."""
    error_expected = 18206.0
    psi = _create_psi()
    Nth, Ny = psi.shape
    delta_y_1D = np.zeros(Nth, dtype=int)
    projections = _create_data()
    va= VerticalAlignmentSwiss(projections)
    error = va.compute_error(psi, delta_y_1D, 0, 1) 
    assert np.array_equal(error, error_expected) 

def test_class_flags_with_alignment(request):
    """
    Initialises the class and tests the run_alignment method for a mixture of
    flag values. Deletes the test file if saving is True.
    Includes the option to visualise the plots from the --view parser.
    """
    if request.config.getoption("--view"):
        flags = {
            "swap_xy": [True, False],
            "saving": [True, False],
            "plotting": [True, False]
        }
    else:
        flags = {
            "swap_xy": [True, False],
            "saving": [True, False],
            "plotting": [False]
        }

    projections = _create_data()
    for combo in itertools.product(*flags.values()):
        kwargs = dict(zip(flags.keys(), combo))
        va= VerticalAlignmentSwiss(projections, **kwargs)
        va.run_alignment()
        import os
        if va.swap_xy == True:
            file_path = "delta_x_1D.npy"
        else:
            file_path = "delta_y_1D.npy"
        if os.path.exists(file_path):
            os.remove(file_path)
            print(f"Deleted test file: {file_path}.")

def test_class_parameters_with_alignment():
    """Initialises the class and tests the run_alignment method for a mixture of max_shift and iterations."""
    projections = _create_data()
    parameters = {
            "max_shift": [2,5,10],
            "iterations": [1,2,4,5,10,20]
        }
    for combo in itertools.product(*parameters.values()):
        kwargs = dict(zip(parameters.keys(), combo))
        va= VerticalAlignmentSwiss(projections, **kwargs)
        va.run_alignment()
