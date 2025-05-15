import faiss
import numpy as np
import polars as pl
import sys

TOP_K=100

def main():
    name = sys.argv[1]
    print("Loading embeddings for linq")
    embeddings1 = np.stack(pl.read_parquet(f"./linq_{name}.parquet")["embedding"].to_numpy()).astype(np.float32)
    print("Loaded embeddings for linq")
    print("Building index for linq")
    cpu_index1 = faiss.index_factory(embeddings1.shape[-1], "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    cpu_index1.hnsw.efConstruction = TOP_K*4
    cpu_index1.add(embeddings1)
    print("Builded index for linq")
    print("Saving index for linq")
    faiss.write_index(cpu_index1, f"./linq_{name}.faiss")
    print("Saved index for linq")
    del embeddings1
    del cpu_index1

    print("Loading embeddings for inf_big")
    embeddings2 = np.stack(pl.read_parquet(f"./inf_big_{name}.parquet")["embedding"].to_numpy()).astype(np.float32)
    print("Loaded embeddings for inf_big")
    print("Building index for inf_big")
    cpu_index2 = faiss.index_factory(embeddings2.shape[-1], "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    cpu_index2.hnsw.efConstruction = TOP_K*4
    cpu_index2.add(embeddings2)
    print("Builded index for inf_big")
    print("Saving index for inf_big")
    faiss.write_index(cpu_index2, f"./inf_big_{name}.faiss")
    print("Saved index for inf_big")
    del embeddings2
    del cpu_index2

    print("Loading embeddings for inf_small")
    embeddings3 = np.stack(pl.read_parquet(f"./inf_small_{name}.parquet")["embedding"].to_numpy()).astype(np.float32)
    print("Loaded embeddings for inf_small")
    print("Building index for inf_small")
    cpu_index3 = faiss.index_factory(embeddings3.shape[-1], "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    cpu_index3.hnsw.efConstruction = TOP_K*4
    cpu_index3.add(embeddings3)
    print("Builded index for inf_small")
    print("Saving index for inf_small")
    faiss.write_index(cpu_index3, f"./inf_small_{name}.faiss")
    print("Saved index for inf_small")
    del embeddings3
    del cpu_index3

    print("Loading embeddings for arctic")
    embeddings4 = np.stack(pl.read_parquet(f"./arctic_{name}.parquet")["embedding"].to_numpy()).astype(np.float32)
    print("Loaded embeddings for arctic")
    print("Building index for arctic")
    cpu_index4 = faiss.index_factory(embeddings4.shape[-1], "HNSW32,Flat", faiss.METRIC_INNER_PRODUCT)
    cpu_index4.hnsw.efConstruction = TOP_K*4
    cpu_index4.add(embeddings4)
    print("Builded index for arctic")
    print("Saving index for arctic")
    faiss.write_index(cpu_index4, f"./arctic_{name}.faiss")
    print("Saved index for arctic")
    del embeddings4
    del cpu_index4


if __name__ == '__main__':
    main()