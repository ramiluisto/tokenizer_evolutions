#!/usr/bin/env python3
from pathlib import Path
from datasets import load_dataset, config
import argparse
import os
import shutil
import time

def download_wikipedia_dataset(output_dir: Path, num_articles: int = None, dry_run: bool = False):
    """
    Download the Wikipedia dataset and save to output directory.
    
    Args:
        output_dir: Path where to save the Wikipedia articles
        num_articles: Number of articles to save (None for all)
        dry_run: If True, only print information without saving files
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Print cache location
    cache_dir = Path(config.HF_DATASETS_CACHE)
    dataset_path = cache_dir / "wikipedia/20220301.en/2.0.0/d41137e149b2ea90eead07e7e3f805119a8c22dd1d5b61651af8e3e3ee736001"
    
    print(f"HuggingFace datasets cache: {cache_dir}")
    print(f"Expected Wikipedia dataset path: {dataset_path}")
    
    if dataset_path.exists():
        print(f"✅ Cached Wikipedia dataset found at {dataset_path}")
    else:
        print("❌ Cached Wikipedia dataset not found, will download (approx 20GB)")
    
    # Load dataset
    start_time = time.time()
    print(f"Loading Wikipedia dataset...")
    ds = load_dataset("wikipedia", "20220301.en", split="train")
    load_time = time.time() - start_time
    print(f"Dataset loaded in {load_time:.2f} seconds")
    
    # Print dataset statistics
    print(f"Dataset contains {len(ds):,} articles")
    
    if dry_run:
        print("Dry run mode - exiting without saving articles")
        return
    
    # Set number of articles to process
    total_articles = len(ds) if num_articles is None else min(num_articles, len(ds))
    print(f"Will process {total_articles:,} articles")
    
    # Save each article
    start_time = time.time()
    batch_size = 10000  # Process in batches to show progress
    saved_size = 0
    
    for i in range(0, total_articles, batch_size):
        batch_end = min(i + batch_size, total_articles)
        print(f"Processing articles {i:,} to {batch_end:,}...")
        
        # Process a batch of articles
        for j in range(i, batch_end):
            article = ds[j]
            
            # Create a safe filename from the title
            safe_title = "".join(c if c.isalnum() else "_" for c in article["title"])
            safe_title = safe_title[:100]  # Limit filename length
            
            # Save article with title and text
            article_path = output_dir / f"{j:08d}_{safe_title}.txt"
            content = f"# {article['title']}\n\n{article['text']}"
            
            # Calculate size and write the file
            content_bytes = content.encode('utf-8')
            saved_size += len(content_bytes)
            article_path.write_bytes(content_bytes)
        
        # Print progress
        elapsed = time.time() - start_time
        articles_per_sec = (batch_end - i) / elapsed if elapsed > 0 else 0
        saved_mb = saved_size / (1024 * 1024)
        
        print(f"Processed {batch_end:,} articles ({articles_per_sec:.1f} articles/sec), saved {saved_mb:.2f}MB")
        start_time = time.time()
        saved_size = 0

def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia dataset and save to disk")
    parser.add_argument("--output", type=str, default="./data/wikipedia", 
                        help="Output directory to save Wikipedia articles")
    parser.add_argument("--articles", type=int, default=None,
                        help="Number of articles to save (default: all)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Only check cache and load dataset without saving files")
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    
    print(f"Wikipedia Dataset Downloader")
    print(f"============================")
    print(f"Output directory: {output_dir.absolute()}")
    
    download_wikipedia_dataset(output_dir, args.articles, args.dry_run)
    
    print("\nDownload complete!")
    if not args.dry_run:
        # Count files and calculate total size
        num_files = len(list(output_dir.glob("*.txt")))
        total_size_bytes = sum(f.stat().st_size for f in output_dir.glob("*.txt"))
        total_size_mb = total_size_bytes / (1024 * 1024)
        print(f"Saved {num_files:,} articles ({total_size_mb:.2f}MB) to {output_dir.absolute()}")

if __name__ == "__main__":
    main() 