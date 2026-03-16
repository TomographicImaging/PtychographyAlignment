import pytest

def test_open_viewer(request):
    """Tests that the viewer works correctly. It is skipped if --view not set."""
    
    if not request.config.getoption("--view"):
        pytest.skip("Skipping test_open_viewer because --view was not set.")
    
    from viewer.OpenViewer import OpenViewer
    from io_module.Imports import ImportData
    from config.paths import pollen_filepath, pollen_data_key, pollen_angle_key

    data = ImportData(pollen_filepath, data_key= pollen_data_key, angle_key = pollen_angle_key)
    projections_raw= data.get_projections_raw()
    OpenViewer(projections_raw)


