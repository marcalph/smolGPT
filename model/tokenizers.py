
class Tokenizer:
  def __init__(self):
    self.toktoi = {}
    self.itotok = {}
    self.vocab_sz = 0

  def encode(self, text: str) -> list:
    tokens = self.tokenize(text)
    return [self.toktoi[token] for token in tokens]
  
  def decode(self, tokens: list) -> str:
    return ''.join([self.itotok[token] for token in tokens])


# TODO(marcalph): implement BPE/SentencePiece tokenizers
class CharTokenizer(Tokenizer):
  def __init__(self):
    super().__init__()
  
  def tokenize(self, corpus: str) -> list:
    tokenized_corpus = sorted(list(set(corpus)))
    self.toktoi = {c: i for i, c in enumerate(tokenized_corpus)}
    self.itotok = {i: c for c, i in self.toktoi.items()}
    self.vocab_sz = len(self.toktoi)