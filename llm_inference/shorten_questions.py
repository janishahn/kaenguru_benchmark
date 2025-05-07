#!/usr/bin/env python3

import argparse
import pandas as pd
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description='Randomly sample entries from a parquet file')
    parser.add_argument('input_file', help='Path to input parquet file')
    parser.add_argument('num_samples', type=int, help='Number of entries to sample')
    parser.add_argument('output_dir', help='Relative path to output directory')
    
    args = parser.parse_args()
    
    # Read the parquet file
    df = pd.read_parquet(args.input_file)
    
    # Randomly sample the specified number of entries
    sampled_df = df.sample(n=min(args.num_samples, len(df)), random_state=42)
    
    # Create output directory if it doesn't exist
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate output filename based on input filename
    input_filename = Path(args.input_file).stem
    output_file = output_dir / f"{input_filename}_sampled_{args.num_samples}.parquet"
    
    # Save the sampled dataframe
    sampled_df.to_parquet(output_file)
    print(f"Saved {len(sampled_df)} samples to {output_file}")

if __name__ == "__main__":
    main()
