def pytest_addoption(parser):
    parser.addoption(
        "--view", action="store_true", default=False, help="Enable image viewer during tests"
    )

def pytest_configure(config):
    config.viewer_enabled = config.getoption("--view")