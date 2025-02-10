from model.tokenizing import CharTokenizer
from hypothesis import given, strategies as st, settings, HealthCheck
import torch


def test_tokenizer(test_corpus):
    # arrange
    tokenizer = CharTokenizer()
    tokenizer.read(test_corpus)
    default_string = "Hello, World!"
    # act
    default_tokenized = tokenizer.encode(default_string)
    # assert
    assert torch.equal(default_tokenized, torch.Tensor([20, 43, 50, 50, 53, 6, 1, 35, 53, 56, 50, 42, 2]))
    assert tokenizer.decode(tokenizer.encode(default_string)) == default_string


@given(corpus_alphabet=st.data())
def test_tokenizer_random(test_corpus, corpus_alphabet):
    # arrange
    tokenizer = CharTokenizer()
    tokenizer.read(test_corpus)
    text_strat = st.text(alphabet=tokenizer.vocab)
    random_str = corpus_alphabet.draw(text_strat)
    # act/assert
    assert tokenizer.decode(tokenizer.encode(random_str)) == random_str


def test_tokenizer_empty(test_corpus):
    # arrange
    tokenizer = CharTokenizer()
    tokenizer.read(test_corpus)
    empty_string = ""
    # act
    default_tokenized = tokenizer.encode(empty_string)
    # assert
    assert torch.equal(default_tokenized, torch.Tensor([]))
    assert tokenizer.decode(tokenizer.encode(empty_string)) == empty_string
