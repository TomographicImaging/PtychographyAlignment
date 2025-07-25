import numpy as np
from alignment.VerticalAlignmentSwiss import VerticalAlignmentSwiss

def create_data():# Define dimensions
    (nx, ny, nth) = (2, 3, 4)  # (slices, height, width)
    # Create a 1D array from -2π to 2π
    #phase_1d = np.linspace(-1* np.pi, 1 * np.pi, nth)
    phase_1d = np.linspace(0, 3, nth)
    # Repeat the 1D array across the 3D volume (nx x ny x nz)
    phase_stack = np.tile(phase_1d, (nx, ny, 1))
    return phase_stack

def create_psi():
    A =np.array([[[0,0,0],[1,1,1],[2,2,2]],[[10,10,10],[11,11,11],[12,12,12]],[[100,100,100],[101,101,101],[102,102,102]]])
    psi = A[:,:,0]
    return psi

projections = create_data()
vas = VerticalAlignmentSwiss(projections, max_shift = 50, iterations = 1, swap_xy = False, plotting = True)

def _test_shift_psi_by_array_and_mean():
    psi = create_psi()
    psi_shifted, mean = vas.shift_psi_by_array_and_mean(psi, np.array([2,0,1]))
    psi_shfted_expected = np.array([[2, 0, 1], [10, 11,12], [101, 102, 100]])
    mean_expected = np.array([37.66666667, 37.66666667, 37.66666667])
    assert np.array_equal(psi_shifted, psi_shfted_expected)
    assert np.allclose(mean, mean_expected)

def _test_shift_psi_by_array_and_mean2(self, psi):
    Nth, Ny = psi.shape
    delta_y_1D = np.zeros(Nth, dtype=int)  # Initialize shifts to zero
    for theta in range(Nth//2):  # Process each projection separately
        delta_y_1D[theta] = 200
    # Compute the mean projection over all θ, using the current shifts
    shifted_projections, mean = shift_psi_by_array_and_mean(psi, delta_y_1D)
    self.plot_1D(mean,"Mean over the projection angle theta",self.align, "mean")
    self.plot_array(np.transpose(shifted_projections), f"psi_theta({self.align}) shifted",  'projection #', self.align)

def _test_compute_error(self, psi):
    Nth, Ny = psi.shape
    delta_y_1D = np.zeros(Nth, dtype=int)  # Initialize shifts to zero
    error = self.compute_error(psi, delta_y_1D, 0, 1) 
    print("error is ", error)

def _test_alignment():
    va= VerticalAlignmentSwiss(projections)
    va.run_alignment()
    # if va.swap_xy ==True:
    #     np.save(r"src\delta_x_1D.npy", va.delta_y_1D_final)
    # else:
    #     np.save(r"src\delta_y_1D.npy", va.delta_y_1D_final)

_test_alignment()
_test_shift_psi_by_array_and_mean()
#test_shift_psi_by_array_and_mean(psi)
#test_compute_error(psi)