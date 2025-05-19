# train_tokenizer.py
#!/usr/bin/env python3
import argparse
from pathlib import Path
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.pre_tokenizers import ByteLevel

def train(input_dir: Path, vocab_size: int, output_dir: Path) -> None:
    files = [str(p) for p in input_dir.glob("*.txt")]
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    tok.normalizer = Sequence([NFKC(), Lowercase()])
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    tok.train(files=files, trainer=trainer)
    output_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_dir / "tokenizer.json"))

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_size", type=int, required=True)
    p.add_argument("--output_dir", type=Path, required=True)
    args = p.parse_args()

    inp = Path("combined_data")

    train(inp, args.vocab_size, args.output_dir)

if __name__ == "__main__":
    import os
    from tqdm import tqdm

    inp = Path("combined_data")

    for vocab_size in tqdm(range(1000, 90000, 5000)):
        folder = Path('./tokenizers/size_variation_{}'.format(vocab_size))
        train(inp, vocab_size, folder)
