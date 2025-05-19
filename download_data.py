# download_data.py
#!/usr/bin/env python3
import nltk
from pathlib import Path
from datasets import load_dataset

def download_gutenberg(target: Path) -> None:
    nltk.download("gutenberg")
    from nltk.corpus import gutenberg
    target.mkdir(parents=True, exist_ok=True)
    for fid in gutenberg.fileids():
        text = gutenberg.raw(fid)
        (target / f"{fid}.txt").write_text(text, encoding="utf-8")

def download_wikipedia(target: Path) -> None:
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    target.mkdir(parents=True, exist_ok=True)
    out = target / "wikitext2_train.txt"
    with out.open("w", encoding="utf-8") as fh:
        for line in ds["text"]:
            fh.write(line + "\n")

def main() -> None:
    base = Path.cwd()
    download_gutenberg(base / "Gutenberg_data")
    download_wikipedia(base / "wikipedia_data")

if __name__ == "__main__":
    main()
