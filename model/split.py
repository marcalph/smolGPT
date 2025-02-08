from dataclasses import dataclass

@dataclass
class Splitter:
    train_sz: float = 0.9

    def sequential_split(self, data):
        n = int(len(data)*self.train_sz)
        return {"train": data[:n], "val": data[n:]}
    