from data.utils import read_corpus
from pathlib import Path
from model.tokenize import CharTokenizer
import logging
from model.training import Splitter, make_batches, estimate_loss
from model.architecture import BigramLM
import torch

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

#hparams
batch_sz  = 32
context_sz = 8
max_steps = 20000
eval_interval = 1000
lr = 1e-2
device = torch.device("cpu")#torch.device("mps") if torch.mps.is_available() else torch.device("cpu")
eval_steps = 200


if __name__ == "__main__":
    corpus = read_corpus(Path("./data/tinyshakespeare.txt"))
    tokenizer = CharTokenizer()
    tokenizer.read(corpus)
    tokenized_corpus = tokenizer.encode(corpus)

    context_sz = 16
    splitter = Splitter()
    train_data, val_data = splitter.sequential_split(tokenized_corpus).values()

    m = BigramLM(tokenizer.vocab_sz, tokenizer.vocab_sz)
    m = m.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3) # 3e-4
    for step in range(max_steps):
        xb, yb = next(make_batches(train_data, device, 64, 24))
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % eval_interval == 0:
            train_loss = estimate_loss(m, train_data, eval_steps, batch_sz, context_sz)
            val_loss = estimate_loss(m, val_data, eval_steps, batch_sz, context_sz)
            print(f"Step: {step}, Train loss: {train_loss}, Val loss: {val_loss}")
        if step % 5000 == 0:
            print(tokenizer.decode(m.generate(torch.zeros((1,1), dtype=torch.long).to(device), 100)[0]))
    print(xb.shape)