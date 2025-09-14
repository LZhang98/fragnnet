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
    #src_fp = "output/inference_outputs/ikohsc16/chunks/chunk_11.pkl"
    
    # Example usage
    chunk_size = 10_000
    #chunk_and_save_df(pd.read_pickle(src_fp), chunk_size, "output/inference_outputs/ikohsc16/chunks_2")
    #df1 = pd.read_pickle("output/inference_outputs/ikohsc16/chunks/chunk_12.pkl")
    #print(df1.tail(10))
    #df2 = pd.read_pickle("output/inference_outputs/ikohsc16/chunks/chunk_13.pkl")
    #print(df2.head(10))
    #df3 = pd.read_pickle("output/inference_outputs/ikohsc16/chunks/chunk_16.pkl")
    #print(df3.head(10))
    #print(df3.shape)
    num_done = 0
    existing_spec_ids = set()
    for i in range(28):
        src_fp = f"output/inference_outputs/ikohsc16/chunks/chunk_{i}.pkl"
        print(f"Loading predictions from {src_fp}")
        existing_df = pd.read_pickle(src_fp)
        existing_spec_ids = existing_spec_ids.union(set(existing_df['spec_id']))
        #num_done += len(existing_spec_ids)
        # = spec_df[~spec_df['spec_id'].isin(existing_spec_ids)]
    print(f"Number of already done spec_id: {len(existing_spec_ids)}")