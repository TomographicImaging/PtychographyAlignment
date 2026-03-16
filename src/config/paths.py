try:
    from .user_paths import DATA_ROOT
except ImportError:
    raise RuntimeError("Set your DATA_ROOT in src.congif.user_paths.py")

#experimental data from Diamond

#pollen_filepath = DATA_ROOT / "Experimental" / "pollen_Volpe" / "pty_tomo_NX.h5"
pollen_filepath = DATA_ROOT / "pollen" / "pty_tomo_NX.h5"
pollen_data_key = '/entry1/tomo_entry/data/data'
pollen_angle_key = '/entry1/tomo_entry/data/rotation_angle'

NiTi_filepath = DATA_ROOT / "Experimental" / "NiTi_Zifan" / "tomo_ptycho_394043_0_499_phase.nxs"
NiTi_data_key='/entry/data/data'
NiTi_angle_key ='/entry/data/rotation_angle'

TiAlloy_filepath = DATA_ROOT / "Experimental" / "TiAlloy_Kuda" / "projectionsSorted_phase.h5"
TiAlloy_unwrapped_filepath = DATA_ROOT / "Experimental" / "TiAlloy_Kuda" / "projectionsSorted_unwrapped_phase.h5"
TiAlloy_data_key='/data'
TiAlloy_angle_key ='/entry/data/rotation_angle'

# battery data
battery_filepath = DATA_ROOT / "Experimental" / "battery" / "scan_275019_275199_tomo_complex.nxs"
battery_data_key = '/entry1/data'
battery_angle_key = '/entry1/rotation_angle'

# simulations 20250416
angles_path = DATA_ROOT / "simulations" / "sphere_phantom_angles_361.npy"
projections_yjitter_path = DATA_ROOT / "simulations" / "sphere_phantom_yjitter_simulation_361_projections.npy"
projections_xyjitter_path = DATA_ROOT / "simulations" / "sphere_phantom_jitter_simulation_361_projections.npy"
delta_x_sim_path = DATA_ROOT / "simulations" / "sphere_phantom_jitter_delta_x_361.npy"
delta_y_sim_path  = DATA_ROOT / "simulations" / "sphere_phantom_jitter_delta_y_361.npy"

