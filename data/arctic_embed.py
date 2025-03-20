import polars as pl
import numpy as np
import os
import torch
from sentence_transformers import SentenceTransformer
import sys

def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]
    df = pl.read_csv(input_file).get_column("text").to_list()
    # Define a simpler local path (IMPORTANT: THIS IS TO FIX A BUG WHERE MODEL NAME INCLUDE ".")
    local_model_path = "local_arctic"

    # Check if model already exists locally, if not download and save it
    if not os.path.exists(local_model_path):
        print(f"Downloading model to {local_model_path}...")
        original_model = SentenceTransformer("Snowflake/snowflake-arctic-embed-l-v2.0", trust_remote_code=True, model_kwargs={ "torch_dtype": torch.float32})
        original_model.save(local_model_path)
        print("Model saved locally")
        del original_model

    # Then in your main function, use the local path
    model = SentenceTransformer(local_model_path, trust_remote_code=False, model_kwargs={ "torch_dtype": torch.float32 })
    model.max_seq_length = 1024
    torch.cuda.empty_cache()
    pool = model.start_multi_process_pool(["cuda:0", "cuda:1", "cuda:2", "cuda:3"])
    embeddings = l2_normalize(model.encode_multi_process(df, pool, batch_size=48, show_progress_bar=True)).tolist()
    model.stop_multi_process_pool(pool)

    # Create a DataFrame with an 'id' column (starting from 0) and an 'embedding' column.
    df_embeddings = pl.DataFrame({
        "id": list(range(len(embeddings))),
        "embedding": embeddings
    })

    # Save the DataFrame to a Parquet file.
    df_embeddings.write_parquet(output_file)


if __name__ == '__main__':
    main()