import polars as pl
import numpy as np
from FlagEmbedding import BGEM3FlagModel

def l2_normalize(vectors):
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / (norms + 1e-10)

# Read CSV, but only extract the 'text' column since 'id' is redundant.
df = pl.read_csv("corpus.csv").get_column("text").to_list()

# Initialize the embedding model.
model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)

embeddings = l2_normalize(model.encode(
    df,
    batch_size=48,
    max_length=1024,
    return_dense=True,
    return_sparse=False,
    return_colbert_vecs=False
)["dense_vecs"]).tolist()
    
# Create a DataFrame with an 'id' column (starting from 0) and an 'embedding' column.
df_embeddings = pl.DataFrame({
    "id": list(range(len(embeddings))),
    "embedding": embeddings
})

# Save the DataFrame to a Parquet file.
df_embeddings.write_parquet("embeddings.parquet")