import numpy as np
from utilities.utils_tomo import apply_y_shifts

def test_apply_y_shifts():
    A =np.array([[[0,0,0],[1,1,1],[2,2,2]],[[10,10,10],[11,11,11],[12,12,12]],[[100,100,100],[101,101,101],[102,102,102]]])
    B=apply_y_shifts(A, [2,0,1])
    B_expected = np.array([[[1, 1, 1],[2,2,2],[0,0,0]],[[10,10,10],[11,11,11],[12,12,12]],[[102,102,102],[100,100,100],[101,101,101]]])
    assert np.array_equal(B, B_expected)
