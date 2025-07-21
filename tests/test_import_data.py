from io_module.Imports import ImportData
from config.paths import pollen_Volpe_filepath, pollen_Volpe_data_key, pollen_Volpe_angle_key
 
def test_import_data():
    """Tests that the import of the file specified runs successfully."""
    ImportData(pollen_Volpe_filepath, data_key= pollen_Volpe_data_key, angle_key = pollen_Volpe_angle_key)