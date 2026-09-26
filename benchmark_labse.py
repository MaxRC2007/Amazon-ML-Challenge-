import time
import argparse
import pandas as pd
from sentence_transformers import SentenceTransformer

def main():
    parser = argparse.ArgumentParser(description="Benchmark LaBSE encoding speed.")
    parser.add_argument("--batch-size", type=int, default=512, help="Batch size for SentenceTransformer encoding.")
    parser.add_argument("--num-samples", type=int, default=50000, help="Number of samples to benchmark on.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on (cuda or cpu).")
    args = parser.parse_args()
    
    print(f"Loading sentence-transformers/LaBSE on {args.device}...")
    try:
        model = SentenceTransformer("sentence-transformers/LaBSE", device=args.device)
    except Exception as e:
        print(f"Failed to load model: {e}")
        print("Please ensure sentence-transformers is installed: pip install sentence-transformers")
        return

    print(f"Loading {args.num_samples} samples from dataset/train/train_source2.tsv...")
    try:
        df = pd.read_csv("dataset/train/train_source2.tsv", sep="\t", nrows=args.num_samples)
        texts = df["business_name"].fillna("").astype(str).tolist()
    except Exception as e:
        print(f"Could not load data: {e}")
        # fallback to synthetic
        print("Using synthetic strings for benchmark.")
        texts = ["Example business name string for benchmarking"] * args.num_samples

    print(f"\nBenchmarking encoding of {len(texts)} names with batch size {args.batch_size}...")
    start_time = time.time()
    embeddings = model.encode(texts, batch_size=args.batch_size, show_progress_bar=True)
    end_time = time.time()
    
    elapsed_time = end_time - start_time
    time_per_100k = (elapsed_time / len(texts)) * 100000
    
    # 5M in S2/S3 train + 10M in S2/S3 test = ~15M total items to embed
    total_corpus_size = 15_000_000
    estimated_total_time = (elapsed_time / len(texts)) * total_corpus_size
    
    print("\n" + "="*50)
    print(f"BENCHMARK RESULTS (Batch Size: {args.batch_size})")
    print("="*50)
    print(f"Time for {len(texts):,} rows:    {elapsed_time:.2f} seconds")
    print(f"Extrapolated time for 100k:  {time_per_100k:.2f} seconds")
    print(f"Extrapolated time for 15M (Full Corpus):")
    print(f"  -> {estimated_total_time / 60:.2f} minutes")
    print(f"  -> {estimated_total_time / 3600:.2f} hours")
    print("="*50)
    
    if estimated_total_time / 3600 > 1.5:
        print("\nWARNING: Full corpus embedding is projected to take > 1.5 hours.")
        print("If time is tight, consider using gated mode:")
        print("  --run-embedding --embed-threshold 10")
    else:
        print("\nSUCCESS: Full corpus embedding looks well within bounds. Run with:")
        print("  --run-embedding --embed-threshold -1")

if __name__ == "__main__":
    main()
