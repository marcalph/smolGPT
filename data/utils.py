from pathlib import Path

def read_corpus(file_path: Path) -> str:
    with open(file_path, 'r') as f:
        return f.read()
