#!/usr/bin/env python3
import nltk
from pathlib import Path
from datasets import load_dataset
import shutil
import random
import argparse

# Target size in MB for each dataset
MAX_DATASET_SIZE_MB = 80  # Roughly 480MB total across all 6 datasets
FAST_MODE = False  # Global flag to control download size

def download_gutenberg(target: Path) -> None:
    """Download a sample of Gutenberg corpus texts."""
    nltk.download("gutenberg")
    from nltk.corpus import gutenberg
    target.mkdir(parents=True, exist_ok=True)
    
    # Sample a subset of files instead of all
    fileids = gutenberg.fileids()
    # Take even fewer in fast mode
    sample_size = 5 if FAST_MODE else 10
    sampled_fileids = random.sample(fileids, min(len(fileids), sample_size))
    
    for fid in sampled_fileids:
        text = gutenberg.raw(fid)
        (target / f"{fid}.txt").write_text(text, encoding="utf-8")
        
    print(f"Downloaded {len(sampled_fileids)} Gutenberg texts")

def download_wikipedia(target: Path) -> None:
    """Download a small sample of Wikipedia text from a more reliable source."""
    print("Using alternative method for Wikipedia data...")
    
    try:
        # Try using wikipedia dataset instead of wikitext
        ds = load_dataset("wikipedia", "20220301.en", split="train[:500]")
        target.mkdir(parents=True, exist_ok=True)
        out = target / "wikipedia_sample.txt"
        
        # Limit to 10MB of data
        max_bytes = 10 * 1024 * 1024
        current_bytes = 0
        count = 0
        
        with out.open("w", encoding="utf-8") as fh:
            for item in ds:
                # Only take title and first paragraph to save space
                title = f"## {item['title']}"
                # Split text and get first paragraph safely
                paragraphs = item['text'].split('\n\n')
                first_paragraph = paragraphs[0] if paragraphs else ""
                
                # Write with proper newlines
                text = title + "\n\n" + first_paragraph + "\n\n"
                text_bytes = len(text.encode('utf-8'))
                current_bytes += text_bytes
                
                fh.write(text)
                count += 1
                
                # Stop after reasonable size or 500 articles
                if current_bytes >= max_bytes or count >= 500:
                    break
                    
        print(f"Successfully downloaded {count} Wikipedia articles ({current_bytes / (1024*1024):.2f} MB)")
            
    except Exception as e:
        print(f"Error with wikipedia dataset: {e}")
        
        # Fallback to creating a small synthetic sample if everything fails
        target.mkdir(parents=True, exist_ok=True)
        out = target / "wikipedia_sample.txt"
        
        with out.open("w", encoding="utf-8") as fh:
            fh.write("# Sample Wikipedia Content\n\n")
            fh.write("This is a placeholder for Wikipedia content since the download failed.\n")
            fh.write("You may want to manually download a Wikipedia sample later.\n")
            
        print("Created fallback Wikipedia placeholder file")

def download_openwebtext(target: Path) -> None:
    """Download a very small sample of OpenWebText."""
    # Skip in fast mode as this is large
    if FAST_MODE:
        print("Skipping OpenWebText in fast mode")
        return
        
    # Stream a very small portion to avoid large downloads
    ds = load_dataset("openwebtext", split="train[:0.5%]", streaming=True)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "openwebtext.txt"
    
    # Limit to ~100MB worth of data
    max_bytes = 100 * 1024 * 1024
    current_bytes = 0
    
    with out.open("w", encoding="utf-8") as fh:
        for i, sample in enumerate(ds):
            text = sample["text"] + "\n\n"
            text_bytes = len(text.encode('utf-8'))
            current_bytes += text_bytes
            
            fh.write(text)
            
            # Stop after ~500 samples or size limit
            if i >= 500 or current_bytes >= max_bytes:
                break
                
    print(f"Downloaded OpenWebText sample ({current_bytes/(1024*1024):.2f} MB)")

def download_bookcorpus(target: Path) -> None:
    """Download a small sample of BookCorpus."""
    # Skip in fast mode as this is large
    if FAST_MODE:
        print("Skipping BookCorpus in fast mode")
        return
        
    # Take a very small slice of the bookcorpus
    ds = load_dataset("bookcorpus", split="train[:0.5%]")
    target.mkdir(parents=True, exist_ok=True)
    out = target / "bookcorpus.txt"
    
    with out.open("w", encoding="utf-8") as fh:
        # Limit to 500 samples
        sample_limit = 250 if FAST_MODE else 500
        for i, sample in enumerate(ds):
            if i >= sample_limit:
                break
            fh.write(sample["text"] + "\n\n")
            
    print(f"Downloaded {sample_limit} BookCorpus samples")

def download_cc_news(target: Path) -> None:
    """Download a small sample of CC News."""
    # Take a very small slice
    slice_size = "train[:0.5%]" if FAST_MODE else "train[:1%]"
    ds = load_dataset("cc_news", split=slice_size)
    target.mkdir(parents=True, exist_ok=True)
    out = target / "cc_news.txt"
    
    with out.open("w", encoding="utf-8") as fh:
        # Limit to fewer samples in fast mode
        sample_limit = 250 if FAST_MODE else 500
        for i, sample in enumerate(ds):
            if i >= sample_limit:
                break
            fh.write(sample["text"] + "\n\n")
            
    print(f"Downloaded {sample_limit} CC News samples")

def download_scientific_papers(target: Path) -> None:
    """Download a small sample of scientific papers."""
    # Fewer papers in fast mode
    paper_counts = {"arxiv": 100, "pubmed": 100} if FAST_MODE else {"arxiv": 200, "pubmed": 200}
    
    for subset in ("arxiv", "pubmed"):
        # Take a very small slice
        slice_size = "train[:0.2%]" if FAST_MODE else "train[:0.5%]"
        ds = load_dataset("scientific_papers", subset, split=slice_size)
        target.mkdir(parents=True, exist_ok=True)
        out = target / f"scientific_{subset}.txt"
        
        with out.open("w", encoding="utf-8") as fh:
            for i, sample in enumerate(ds):
                if i >= paper_counts[subset]:
                    break
                # Only take abstract to save space
                text = sample["abstract"] if "abstract" in sample else ""
                if not FAST_MODE and "section_names" in sample and "section_texts" in sample:
                    for section_name, section_text in zip(sample["section_names"], sample["section_texts"]):
                        if "introduction" in section_name.lower() or "background" in section_name.lower():
                            text += "\n\n" + section_text
                            break
                fh.write(text + "\n\n")
                
    print(f"Downloaded {sum(paper_counts.values())} scientific papers")

def clear_cache() -> None:
    """Clear the HuggingFace datasets cache to resolve potential conflicts."""
    from datasets import config
    cache_dir = Path(config.HF_DATASETS_CACHE)
    if cache_dir.exists():
        print(f"Clearing datasets cache at {cache_dir}")
        shutil.rmtree(cache_dir, ignore_errors=True)

def check_size(path: Path) -> float:
    """Check total size of directory in MB."""
    if not path.exists():
        return 0
    total_size = sum(f.stat().st_size for f in path.glob('**/*') if f.is_file())
    return total_size / (1024 * 1024)  # Convert to MB

def main() -> None:
    global FAST_MODE
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Download sample text data for tokenizer training")
    parser.add_argument("--fast", action="store_true", help="Use fast mode with minimal downloads")
    parser.add_argument("--output", type=str, default="combined_data", help="Output directory")
    args = parser.parse_args()
    
    FAST_MODE = args.fast
    if FAST_MODE:
        print("Running in FAST MODE - downloads will be minimal")
    
    base = Path.cwd() / args.output
    base.mkdir(parents=True, exist_ok=True)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    functions = [
        download_gutenberg,
        download_wikipedia,
        download_cc_news, 
        download_scientific_papers
    ]
    
    # Only download these larger datasets if not in fast mode
    if not FAST_MODE:
        functions.extend([download_openwebtext, download_bookcorpus])
    
    for function in functions:
        print(f"\nDownloading {function.__name__}...")
        try:
            function(base)
        except Exception as e:
            print(f"Error downloading {function.__name__}: {e}")

        # Check current size and stop if we exceed 500MB (or 100MB in fast mode)
        size_limit = 100 if FAST_MODE else 500
        current_size = check_size(base)
        print(f"Current data size: {current_size:.2f}MB")
        if current_size > size_limit:
            print(f"Stopping download because combined data size exceeds {size_limit}MB")
            break
    
    print(f"\nDownload complete! Total data size: {check_size(base):.2f}MB")
    print(f"Data saved to: {base.absolute()}")

if __name__ == "__main__":
    main()
