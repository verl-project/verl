import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--runproviders", action="store_true", default=False, help="run provider tests"
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "providers: mark test as requires providers api keys to run"
    )


def pytest_collection_modifyitems(config, items):
    if config.getoption("--runproviders"):
        # --runproviders given in cli: do not skip slow tests
        return
    skip_providers = pytest.mark.skip(reason="need --runproviders option to run")
    for item in items:
        if "providers" in item.keywords:
            item.add_marker(skip_providers)
