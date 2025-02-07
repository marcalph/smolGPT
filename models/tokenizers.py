import logging
logger = logging.getLogger(__name__)


class Tokenizer:
  def __init__(self):
    self.toktoi = None
    self.itotok = None
    self.vocab_sz = None

  def encode(self, text: str) -> list:
    return [self.toktoi[token] for token in text]
  
  def decode(self, tokens: list) -> str:
    return ''.join([self.itotok[token] for token in tokens])


# TODO(marcalph): implement BPE/SentencePiece tokenizers
class CharTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()
  
  def tokenize(self, corpus: str) -> list:
    logger.info(f"Tokenization corpus lenght in characters: {len(corpus)}")
    tokenized_corpus = sorted(list(set(corpus)))
    self.toktoi = {c: i for i, c in enumerate(tokenized_corpus)}
    self.itotok = {i: c for c, i in self.toktoi.items()}
    self.vocab_sz = len(self.toktoi)