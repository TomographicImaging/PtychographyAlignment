from scipy.io import savemat
import numpy as np
# angles = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_angles_361.npy")
# projections_xyjitter = np.load(r"c:\Users\zvm34551\Coding_environment\DATA\Ptychography\Simulations\20250416\sphere_phantom_jitter_simulation_361_projections.npy")
# savemat("sphere_phantom_angles_361.mat", {"angles": angles})
# savemat("projections_xyjitter.mat", {"projections": projections_xyjitter})
angles_reduced = np.load(r"data\experimental\data\angles_reduced.npy")
projections_reduced = np.load(r"data\experimental\data\projections_reduced.npy")
savemat("projections_reduced.mat", {"projections": projections_reduced})
savemat("angles_reduced.mat", {"angles": angles_reduced})