import logging
import torch


logger = logging.getLogger(__name__)


class Tokenizer:
    def __init__(self):
        self.toktoi = None
        self.itotok = None
        self.vocab_sz = None
        self.vocab = None

    def encode(self, text: str) -> torch.Tensor:
        return torch.tensor([self.toktoi[token] for token in text], dtype=torch.long)

    def decode(self, tokens: torch.Tensor) -> str:
        return "".join([self.itotok[tokenid.item()] for tokenid in tokens])


# TODO(marcalph): implement BPE/SentencePiece tokenizers
class CharTokenizer(Tokenizer):
    def __init__(self):
        super().__init__()

    def read(self, corpus: str) -> list:
        logger.info(f"Tokenization corpus lenght in characters: {len(corpus)}")
        tokenized_corpus = sorted(list(set(corpus)))
        self.toktoi = {c: i for i, c in enumerate(tokenized_corpus)}
        self.itotok = {i: c for c, i in self.toktoi.items()}
        self.vocab_sz = len(self.toktoi)
        self.vocab = "".join(tokenized_corpus)
