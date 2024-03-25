import pytest


@pytest.fixture(scope="session", autouse=True)
def global_setup():
    pass
