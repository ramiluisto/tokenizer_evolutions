# train_tokenizer_optimized.py
#!/usr/bin/env python3
from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.normalizers import Sequence, NFKC, Lowercase
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.trainers import BpeTrainer
from tqdm import tqdm


# --------------------------------------------------------------------------- #
# Pre-tokenise once and (optionally) cache to disk
# --------------------------------------------------------------------------- #
def load_or_create_corpus(
    input_dir: Path,
    cache_path: Path | None = None,
) -> List[List[str]]:
    if cache_path and cache_path.exists():
        with cache_path.open("rb") as f:
            return pickle.load(f)

    normalizer = Sequence([NFKC(), Lowercase()])
    pre_tok = ByteLevel(add_prefix_space=False)

    corpus: List[List[str]] = []
    for path in tqdm(sorted(input_dir.glob("*.txt")), desc="Pre-tokenising"):
        with path.open(encoding="utf-8") as fh:
            for raw in fh:
                raw = raw.strip()
                if not raw:
                    continue
                text = normalizer.normalize_str(raw)
                tokens = [tok for tok, _ in pre_tok.pre_tokenize_str(text)]
                if tokens:
                    corpus.append(tokens)

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with cache_path.open("wb") as f:
            pickle.dump(corpus, f)

    return corpus


# --------------------------------------------------------------------------- #
# Train a single tokenizer from an *already tokenised* iterator
# --------------------------------------------------------------------------- #
def train_from_corpus(
    corpus: Iterable[List[str]],
    vocab_size: int,
    output_dir: Path,
) -> None:
    tok = Tokenizer(BPE(unk_token="[UNK]"))
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=2,
        special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"],
    )
    # length is a small speed hint; it isn't mandatory
    tok.train_from_iterator(corpus, trainer=trainer, length=len(corpus))

    output_dir.mkdir(parents=True, exist_ok=True)
    tok.save(str(output_dir / "tokenizer.json"))


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--vocab_start", type=int, default=85_000)
    p.add_argument("--vocab_end", type=int, default=200_000)
    p.add_argument("--vocab_step", type=int, default=15_000)
    p.add_argument("--input_dir", type=Path, default=Path("data/wikipedia"))
    p.add_argument("--cache", type=Path, default=Path("cache/wikipedia.pickle"))
    p.add_argument("--out_root", type=Path, default=Path("tokenizers/wikipedia"))
    args = p.parse_args()

    # One-shot preprocessing -------------------------------------------------- #
    corpus = load_or_create_corpus(args.input_dir, args.cache)

    # Train all vocab sizes --------------------------------------------------- #
    sizes = range(args.vocab_start, args.vocab_end, args.vocab_step)
    for vsize in tqdm(sizes, desc="Training tokenizers"):
        out_dir = args.out_root / f"size_{vsize}"
        train_from_corpus(corpus, vsize, out_dir)


if __name__ == "__main__":
    main()
