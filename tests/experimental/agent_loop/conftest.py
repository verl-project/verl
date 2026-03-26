import pytest


def pytest_configure(config):
    config.addinivalue_line("markers", "vllm_omni: requires the vllm-omni package")


def pytest_collection_modifyitems(config, items):
    try:
        import vllm_omni  # noqa: F401
    except ImportError:
        skip = pytest.mark.skip(reason="vllm-omni not installed")
        for item in items:
            if "vllm_omni" in item.keywords:
                item.add_marker(skip)
