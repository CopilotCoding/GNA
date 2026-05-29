"""
data/ptb.py — PTB word-level dataset, GPU-resident.
Zero CPU involvement per batch after initial load.
"""
import urllib.request
from pathlib import Path
import torch


PTB_URLS = {
    "train": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.train.txt",
    "valid": "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.valid.txt",
    "test":  "https://raw.githubusercontent.com/wojzaremba/lstm/master/data/ptb.test.txt",
}


def download(data_dir: str = "./ptb_data") -> Path:
    p = Path(data_dir)
    p.mkdir(parents=True, exist_ok=True)
    for split, url in PTB_URLS.items():
        f = p / f"ptb.{split}.txt"
        if not f.exists():
            print(f"  Downloading PTB {split}...")
            urllib.request.urlretrieve(url, f)
    return p


def build_vocab(path: Path) -> dict:
    words, seen = ["<unk>"], {"<unk>"}
    with open(path, encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                if w not in seen:
                    seen.add(w)
                    words.append(w)
    return {w: i for i, w in enumerate(words)}


def tokenize(path: Path, w2i: dict) -> torch.Tensor:
    ids, unk = [], w2i["<unk>"]
    with open(path, encoding="utf-8") as f:
        for line in f:
            for w in line.split():
                ids.append(w2i.get(w, unk))
            ids.append(w2i.get("<eos>", unk))
    return torch.tensor(ids, dtype=torch.long)


class GPUDataset:
    """Entire dataset lives on GPU. Batches are pure index ops — no transfers."""
    def __init__(self, tokens: torch.Tensor, seq_len: int, device: torch.device):
        n = (len(tokens) - 1) // seq_len * seq_len
        tok = tokens[:n + 1].to(device)
        self.x      = tok[:n].view(-1, seq_len)
        self.y      = tok[1:n + 1].view(-1, seq_len)
        self.n_seqs = self.x.shape[0]
        self.device = device

    def iter(self, batch_size: int, shuffle: bool = True):
        idx = torch.randperm(self.n_seqs, device=self.device) if shuffle \
              else torch.arange(self.n_seqs, device=self.device)
        for start in range(0, self.n_seqs - batch_size + 1, batch_size):
            b = idx[start:start + batch_size]
            yield self.x[b], self.y[b]

    def __len__(self): return self.n_seqs


def load(seq_len: int = 128, device: torch.device = torch.device("cuda"),
         data_dir: str = "./ptb_data"):
    p = download(data_dir)
    w2i = build_vocab(p / "ptb.train.txt")
    train = GPUDataset(tokenize(p / "ptb.train.txt", w2i), seq_len, device)
    valid = GPUDataset(tokenize(p / "ptb.valid.txt", w2i), seq_len, device)
    return train, valid, len(w2i)
