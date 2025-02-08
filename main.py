from data.utils import read_corpus
from pathlib import Path
from models.tokenize import CharTokenizer
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


if __name__ == "__main__":
    # Read the corpus
    corpus = read_corpus(Path("./data/tinyshakespeare.txt"))
    # Tokenize the corpus
    tokenizer = CharTokenizer()
    tokens = tokenizer.tokenize(corpus)
    print(tokenizer.encode("Hello, World!"))
    
