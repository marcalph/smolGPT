import pytest
from models.tokenizers import CharTokenizer
from data.utils import read_corpus
from hypothesis import given, strategies as st, settings, HealthCheck


@pytest.fixture
def test_corpus():
    return read_corpus("data/tinyshakespeare.txt")

# we dont care that the test_corpus fixture is not reset between hypothesis sample generation
# we dont use hypothesis for now because char tokenization depends on input corpus
@settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(test_string = st.text(min_size=0, max_size=100))
def test_tokenizer(test_corpus, test_string):
    # arrange
    tokenizer = CharTokenizer()
    tokenizer.tokenize(test_corpus)
    default_string = "hii"
    # act
    default_tokenized = tokenizer.encode(default_string)
    # assert
    assert default_tokenized == [46, 47, 47]
    assert tokenizer.decode(tokenizer.encode(default_string)) == default_string
