import pandas as pd
import os

def chunk_and_save_df(src_df, chunk_size, out_dir, prefix="chunk"):
    os.makedirs(out_dir, exist_ok=True)
    n = len(src_df)
    for i, start in enumerate(range(0, n, chunk_size)):
        chunk = src_df.iloc[start:start + chunk_size]
        out_fp = os.path.join(out_dir, f"{prefix}_{i}.pkl")
        chunk.to_pickle(out_fp)
        print(f"Saved {out_fp} ({len(chunk)} rows)")


if __name__ == "__main__":
    src_fp = "output/inference_outputs/ikohsc16/predictions.pkl"
    
    # Example usage
    chunk_size = 10000
    chunk_and_save_df(pd.read_pickle(src_fp), chunk_size, "output/inference_outputs/ikohsc16/chunks")