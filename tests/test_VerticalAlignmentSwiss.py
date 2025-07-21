import numpy as np
from alignment.VerticalAlignmentSwiss import shift_psi_by_array

def test_shift_psi_by_array():
    A =np.array([[[0,0,0],[1,1,1],[2,2,2]],[[10,10,10],[11,11,11],[12,12,12]],[[100,100,100],[101,101,101],[102,102,102]]])
    psi = A[:,:,0]
    psi_shifted, mean = shift_psi_by_array(psi, np.array([2,0,1]))
    psi_shfted_expected = np.array([[2, 0, 1], [10, 11,12], [101, 102, 100]])
    mean_expected = np.array([37.66666667, 37.66666667, 37.66666667])
    assert np.array_equal(psi_shifted, psi_shfted_expected)
    assert np.allclose(mean, mean_expected)