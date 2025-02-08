import pytest
from data.utils import read_corpus


@pytest.fixture(scope="module")
def test_corpus():
    return read_corpus("data/tinyshakespeare.txt")