from io_module.Imports import ImportData

ptytomofile = 'C:/Users/zvm34551/Coding_environment/DATA/Ptychography//Experimental/pollen_Volpe/pty_tomo_NX.h5'
    
def test_import_data():
    """Tests that the import of the file specified runs successfully."""
    ImportData(ptytomofile, data_key='/entry1/tomo_entry/data/data', angle_key = '/entry1/tomo_entry/data/rotation_angle')