import numpy as np
from joblib import Parallel, delayed
a = np.array([[0,0,0],[1,1,1],[2,2,2]])
b = np.array([a,a,a])
#print(b)

def func(arr2d):
    arr2d += arr2d

#func(b)
print(b)
Parallel(n_jobs=-1, prefer="threads")(delayed(func)(arr) for arr in b)
print(b)

# Example values
def wrap_phase_arg(phase):
    return np.angle(np.exp(1j * phase), deg=False)
phases = np.array([0, np.pi, -np.pi, 2*np.pi, -2*np.pi, np.pi/2, -np.pi/2])
wrapped = wrap_phase_arg(phases)

print("Original:", phases)
print("Wrapped:", wrapped)