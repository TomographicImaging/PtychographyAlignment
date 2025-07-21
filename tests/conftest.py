def pytest_addoption(parser):
    parser.addoption(
        "--viewer", action="store_true", default=False, help="Enable image viewer during tests"
    )

def pytest_configure(config):
    config.viewer_enabled = config.getoption("--viewer")