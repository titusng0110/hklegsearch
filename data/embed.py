import polars as pl
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from tqdm import tqdm  # Import tqdm for progress bars

# Read CSV with header columns 'id' and 'text'.
df = pl.read_csv("output.csv")

# Initialize your model.
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

batch_size = 48
results = []  # This will hold dictionaries of id and embedding.

# Process the DataFrame in batches with a progress bar.
for start in tqdm(range(0, df.height, batch_size), desc="Processing batches"):
    batch_df = df[start : start + batch_size]
    texts = batch_df["text"].to_list()

    # Encode the batch of text.
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )["dense_vecs"].astype(np.float32)
    
    # Loop over the results from the current batch.
    for id_val, emb in zip(batch_df["id"].to_list(), embeddings):
        results.append({
            "id": int(id_val),  # ensure id is an integer
            "embedding": emb
        })

# Convert the results to a Polars DataFrame with two columns: id and embedding.
results_df = pl.DataFrame(results)

# Write the DataFrame to a Parquet file.
results_df.write_parquet("embeddings.parquet")