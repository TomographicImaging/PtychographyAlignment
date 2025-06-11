print("herllo")
import numpy as np

A =np.array([[[0,0,0],[1,1,1],[2,2,2]],[[10,10,10],[11,11,11],[12,12,12]],[[100,100,100],[101,101,101],[102,102,102]]])
#print(A)

def apply_y_shifts(projections, trans):
    for th in range(projections.shape[0]):
        print(th)
        shift = int(trans[th])
        print(shift)
        projections[th,:,:] = np.roll(projections[th,:,:],-shift, axis=0)
    return projections


def shift_psi_by_array(psi, shifts_1D):
    "shift psi by 1d array of y shifts and calculate mean for this, over angle"
    Nth, Ny = psi.shape  # Get dimensions
    # Compute the average projection over all θ, using the current shifts
    shifted_psi = np.array([np.roll(psi[theta, :], -shifts_1D[theta]) for theta in range(Nth)])
    avg_psi = np.mean(shifted_psi, axis=0)  # Compute mean along θ
    return shifted_psi, avg_psi

B=apply_y_shifts(A, [2,0,1])
print(B)

A =np.array([[[0,0,0],[1,1,1],[2,2,2]],[[10,10,10],[11,11,11],[12,12,12]],[[100,100,100],[101,101,101],[102,102,102]]])

print(A[:,:,0])
C, D = shift_psi_by_array(A[:,:,0], np.array([2,0,1]))
print(C)