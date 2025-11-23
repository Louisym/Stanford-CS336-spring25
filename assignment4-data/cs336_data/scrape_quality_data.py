#!/usr/bin/env python3
"""
Script to scrape Wikipedia reference URLs and save them in WARC format
for training a quality classifier.
"""

import random
import subprocess
import sys
from pathlib import Path

def sample_urls(input_file: str, output_file: str, sample_size: int = 1000):
    """
    Sample URLs from the Wikipedia extracted URLs file.
    
    Args:
        input_file: Path to the compressed Wikipedia URLs file
        output_file: Path to save the sampled URLs
        sample_size: Number of URLs to sample
    """
    print(f"Sampling {sample_size} URLs from {input_file}...")
    
    # First, count total lines to get sampling probability
    count_cmd = ["gunzip", "-c", input_file]
    count_process = subprocess.Popen(count_cmd, stdout=subprocess.PIPE)
    total_lines = sum(1 for _ in count_process.stdout)
    count_process.wait()
    
    print(f"Total URLs available: {total_lines:,}")
    
    # Use reservoir sampling for large files
    sampled_urls = []
    cmd = ["gunzip", "-c", input_file]
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, text=True)
    
    for i, line in enumerate(process.stdout):
        url = line.strip()
        if not url or url.startswith('#'):
            continue
            
        if len(sampled_urls) < sample_size:
            sampled_urls.append(url)
        else:
            # Reservoir sampling: replace with probability sample_size / (i+1)
            j = random.randint(0, i)
            if j < sample_size:
                sampled_urls[j] = url
        
        if (i + 1) % 100000 == 0:
            print(f"Processed {i+1:,} URLs...")
    
    process.wait()
    
    print(f"Sampled {len(sampled_urls)} URLs")
    
    # Write sampled URLs to file
    with open(output_file, 'w') as f:
        for url in sampled_urls:
            f.write(url + '\n')
    
    print(f"Saved sampled URLs to {output_file}")
    return len(sampled_urls)

def scrape_to_warc(urls_file: str, output_warc: str, timeout: int = 10):
    """
    Scrape URLs and save to WARC format using wget.
    
    Args:
        urls_file: File containing URLs to scrape
        output_warc: Output WARC file path
        timeout: Timeout for each request in seconds
    """
    print(f"Scraping URLs to WARC format: {output_warc}")
    
    # Create data directory to avoid polluting main workspace
    data_work_dir = Path("data")
    data_work_dir.mkdir(exist_ok=True)
    
    # Use wget to scrape URLs and save in WARC format
    cmd = [
        "wget",
        "--timeout", str(timeout),
        "--tries", "1",  # Only try once per URL
        "--user-agent", "Mozilla/5.0 (compatible; WikiQualityBot/1.0)",
        "--input-file", str(urls_file),
        "--warc-file", str(output_warc).replace('.warc.gz', ''),
        "--no-directories",  # Don't create directory structure
        "--output-file", "/dev/null",  # Suppress wget output
        "--quiet"
    ]
    
    print("Running wget command...")
    print(f"Command: {' '.join(cmd)}")
    print(f"Working directory: {data_work_dir}")
    
    try:
        # Run wget in the data directory to avoid polluting main workspace
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, cwd=data_work_dir)  # 1 hour timeout
        if result.returncode == 0:
            print("✅ Scraping completed successfully!")
        else:
            print(f"⚠️ Scraping completed with some errors (return code: {result.returncode})")
            if result.stderr:
                print(f"Errors: {result.stderr}")
    except subprocess.TimeoutExpired:
        print("❌ Scraping timed out after 1 hour")
    except Exception as e:
        print(f"❌ Error during scraping: {e}")

def main():
    # Configuration
    data_dir = Path("cs336_data/quality_classifier_data")
    wiki_urls_file = data_dir / "enwiki-20240420-extracted_urls.txt.gz"
    sampled_urls_file = data_dir / "sampled_positive_urls.txt"
    output_warc = data_dir / "sampled_positive_urls.warc.gz"
    sample_size = 1000  # Start with a small sample
    
    # Check if input file exists
    if not Path(wiki_urls_file).exists():
        print(f"❌ Input file {wiki_urls_file} not found!")
        print("Please make sure you have downloaded the Wikipedia URLs file.")
        sys.exit(1)
    
    # Set random seed for reproducibility
    random.seed(42)
    
    # Step 1: Sample URLs
    print("=== Step 1: Sampling URLs ===")
    actual_sample_size = sample_urls(wiki_urls_file, sampled_urls_file, sample_size)
    
    if actual_sample_size == 0:
        print("❌ No URLs were sampled!")
        sys.exit(1)
    
    # Step 2: Scrape to WARC
    print("\n=== Step 2: Scraping to WARC ===")
    scrape_to_warc(sampled_urls_file, output_warc)
    
    # Check if output file was created
    expected_warc = output_warc
    if Path(expected_warc).exists():
        size = Path(expected_warc).stat().st_size
        print(f"✅ WARC file created: {expected_warc} ({size:,} bytes)")
    else:
        print(f"❌ WARC file not found: {expected_warc}")
    
    print("\n=== Complete ===")
    print(f"You can now use {expected_warc} as positive samples for training your quality classifier.")

if __name__ == "__main__":
    main()