import pytest
from model.training import Splitter
from model.tokenizing import CharTokenizer
from hypothesis import given, strategies as st
import torch
def test_splitter(test_corpus):
    # arrange
    splitter = Splitter()
    tokenizer = CharTokenizer()
    tokenizer.read(test_corpus)
    tokenized_corpus = tokenizer.encode(test_corpus)
    # act
    data, val= splitter.sequential_split(tokenized_corpus).values()
    # assert
    assert len(data)+len(val) == len(tokenized_corpus)
    assert len(data) == int(len(tokenized_corpus)*splitter.train_sz)


@given(corpus_alphabet=st.data(), train_sz=st.floats(min_value=0., max_value=1.))
def test_splitter_random(test_corpus, corpus_alphabet, train_sz):
    # arrange
    splitter = Splitter(train_sz=train_sz)
    tokenizer = CharTokenizer()
    tokenizer.read(test_corpus)
    text_strat = st.text(min_size=1, alphabet=tokenizer.vocab)
    random_corpus = corpus_alphabet.draw(text_strat)
    tokenized_corpus = tokenizer.encode(random_corpus)
    # act
    data, val= splitter.sequential_split(tokenized_corpus).values()
    # assert
    assert len(data)+len(val) == len(tokenized_corpus)
    assert len(data) == int(len(tokenized_corpus)*splitter.train_sz)
    assert torch.equal(torch.cat((data, val), dim=0), tokenized_corpus)