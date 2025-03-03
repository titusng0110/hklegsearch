import threading
import queue
import traceback
from flask import Flask, request, jsonify, abort, send_file

# ------------------------
# Global constants
# ------------------------
MAX_PAYLOAD_LENGTH = 2048         # Maximum characters allowed in the payload
MAX_QUEUE_LENGTH = 24000          # Reject new requests if the job queue reaches this length
BATCH_SIZE = 24                   # Process up to 24 items in a batch for the GPU worker

# ------------------------
# Global queues for workers
# ------------------------
gpu_queue = queue.Queue()         # Queue for initial embedding creation (GPU)
cpu_queue = queue.Queue()         # Queue for FAISS search operations (CPU)
rerank_queue = queue.Queue()      # Queue for reranking operations (GPU)

# ------------------------
# Job class to hold task state
# ------------------------
class Job:
    def __init__(self, payload):
        self.payload = payload
        self.embedding1 = None     # Embedding from model1 (M3)
        self.embedding2 = None     # Embedding from model2 (GTE)
        self.doc_indices = None    # Doc indices returned from FAISS search
        self.doc_ids = None        # Reranked doc IDs
        self.texts = None          # Final texts corresponding to each doc id
        self.exception = None      # Exception info if an error occurs during processing
        self.event = threading.Event()  # Signals job completion

# ------------------------
# Processing Functions
# ------------------------
import polars as pl
from FlagEmbedding import BGEM3FlagModel
from sentence_transformers import SentenceTransformer
from FlagEmbedding import LayerWiseFlagLLMReranker
import torch
import faiss
import numpy as np

corpus = None
model1 = None
model2 = None
reranker = None
cpu_index1 = None
cpu_index2 = None


def init():
    global corpus, model1, model2, reranker, cpu_index1, cpu_index2
    # Load corpus texts from CSV.
    corpus = pl.read_csv("../data/corpus.csv").get_column("text").to_list()
    
    # Initialize the embedding model.
    model1 = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True, devices=["cuda:0"])
    model2 = SentenceTransformer("Alibaba-NLP/gte-Qwen2-1.5B-instruct", trust_remote_code=True, device="cuda:1", model_kwargs = {"torch_dtype": torch.float16})
    model2.max_seq_length = 1024
    reranker = LayerWiseFlagLLMReranker('BAAI/bge-reranker-v2-minicpm-layerwise', devices=["cuda:2"], use_fp16=True)
    
    # Load precomputed embeddings from a parquet file.
    embeddings1 = np.stack(pl.read_parquet("../data/m3_embeddings.parquet")["embedding"].to_numpy()).astype(np.float32)
    embeddings2 = np.stack(pl.read_parquet("../data/gte_embeddings.parquet")["embedding"].to_numpy()).astype(np.float32)
    
    # Initialize FAISS index on CPU.
    cpu_index1 = faiss.index_factory(embeddings1.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)
    cpu_index2 = faiss.index_factory(embeddings2.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)
    cpu_index1.add(embeddings1)
    cpu_index2.add(embeddings2)
    del embeddings1, embeddings2

def create_m3_embedding(payloads):
    """Creates normalized embeddings for a batch of payloads using M3 model."""
    def l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    q_embeddings = model1.encode(
        payloads,
        batch_size=BATCH_SIZE,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )["dense_vecs"]
    return l2_normalize(q_embeddings)

def create_gte_embedding(payloads):
    """Creates normalized embeddings for a batch of payloads using GTE model."""
    def l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    q_embeddings = model2.encode(
        payloads,
        prompt_name="query",
        batch_size=BATCH_SIZE,
        show_progress_bar=False
    )
    return l2_normalize(q_embeddings)
    
def faiss_search(q_embedding1, q_embedding2):
    """Perform a FAISS search to retrieve top similar document indices."""
    scores1, indices1 = cpu_index1.search(q_embedding1.astype(np.float32), k=40)
    scores2, indices2 = cpu_index2.search(q_embedding2.astype(np.float32), k=40)
    return np.unique(np.concatenate([indices1, indices2], axis=1), axis=1)

def rerank(query, doc_indices):
    """Rerank the document indices using a reranker model."""
    scores = reranker.compute_score([[query, corpus[doc_id]] for doc_id in doc_indices], cutoff_layers=[28])
    return [doc_id for _, doc_id in sorted(zip(scores, doc_indices), reverse=True)]

def get_texts(doc_ids):
    """Retrieve texts from the corpus given a list/array of doc ids."""
    return [corpus[doc_id] for doc_id in doc_ids]

# ------------------------
# Worker Thread Functions
# ------------------------

# GPU worker for creating embeddings
def embedding_worker():
    while True:
        batch = []
        try:
            # Block until at least one job is available
            job = gpu_queue.get(timeout=0.02)
            batch.append(job)
        except queue.Empty:
            continue

        # Gather additional jobs up to BATCH_SIZE without blocking
        while len(batch) < BATCH_SIZE:
            try:
                job = gpu_queue.get_nowait()
                batch.append(job)
            except queue.Empty:
                break

        payloads = [job.payload for job in batch]
        
        # Create embeddings with model1
        try:
            embeddings1 = create_m3_embedding(payloads)
            for job, embedding in zip(batch, embeddings1):
                job.embedding1 = embedding
        except Exception:
            error_info = traceback.format_exc()
            for job in batch:
                job.exception = error_info
                job.event.set()
            continue
            
        # Create embeddings with model2
        try:
            embeddings2 = create_gte_embedding(payloads)
            for job, embedding in zip(batch, embeddings2):
                job.embedding2 = embedding
        except Exception:
            error_info = traceback.format_exc()
            for job in batch:
                job.exception = error_info
                job.event.set()
            continue
            
        # Send to CPU queue for FAISS search
        for job in batch:
            cpu_queue.put(job)

# CPU worker for FAISS search
def faiss_worker():
    while True:
        try:
            # Get one job at a time to process
            job = cpu_queue.get(timeout=0.02)
            
            try:
                # Reshape single embeddings for FAISS search
                embedding1 = job.embedding1.reshape(1, -1)
                embedding2 = job.embedding2.reshape(1, -1)
                
                # Perform FAISS search
                indices = faiss_search(embedding1, embedding2)[0]  # Get the first (only) result
                job.doc_indices = indices
                
                # Send to reranker queue
                rerank_queue.put(job)
            except Exception:
                job.exception = traceback.format_exc()
                job.event.set()
                
        except queue.Empty:
            continue

# GPU worker for reranking - optimized version
def rerank_worker():
    while True:
        try:
            # Get one job at a time to process immediately, without batching
            job = rerank_queue.get(timeout=0.02)
            
            try:
                # Rerank the documents
                reranked_indices = rerank(job.payload, job.doc_indices)
                job.doc_ids = reranked_indices
                
                # Get the final texts
                job.texts = get_texts(reranked_indices)
            except Exception:
                job.exception = traceback.format_exc()
            finally:
                # Signal completion
                job.event.set()
                
        except queue.Empty:
            continue

# ------------------------
# Flask App Setup for the API Server
# ------------------------
app = Flask(__name__)

@app.route('/', methods=['GET'])
def serve_index():
    # Serve index.html from the same directory as this script.
    return send_file('index.html')

@app.route('/api/', methods=['GET'])
def handle_request():
    # Ensure only GET methods are accepted.
    if request.method != 'GET':
        abort(405, description="Method Not Allowed")

    # Retrieve the payload from the query parameters.
    payload = request.args.get("payload")
    if payload is None:
        abort(400, description="Missing 'payload' query parameter")
    if len(payload) > MAX_PAYLOAD_LENGTH:
        abort(400, description="Payload exceeds maximum allowed length")

    # Check for queue overload before accepting a new job.
    if gpu_queue.qsize() >= MAX_QUEUE_LENGTH:
        abort(503, description="Server busy due to queue overload")

    # Create a job and enqueue it in the GPU queue.
    job = Job(payload)
    gpu_queue.put(job)

    # Wait until processing is complete (with a timeout safeguard).
    job.event.wait(timeout=60)  # Wait up to 60 seconds.
    if not job.event.is_set():
        abort(504, description="Processing timed out")

    if job.exception is not None:
        abort(500, description=f"Processing error: {job.exception}")

    # Return the final texts as JSON.
    return jsonify({"texts": job.texts})

# ------------------------
# Background Threads Starter
# ------------------------
threads_started = False

def start_background_threads():
    global threads_started
    if not threads_started:
        # Start embedding worker (GPU)
        t_embedding = threading.Thread(target=embedding_worker, daemon=True)
        t_embedding.start()
        
        # Start FAISS worker (CPU)
        t_faiss = threading.Thread(target=faiss_worker, daemon=True)
        t_faiss.start()
        
        # Start reranking worker (GPU)
        t_rerank = threading.Thread(target=rerank_worker, daemon=True)
        t_rerank.start()

        threads_started = True
        print("Background threads started.")

# Start background threads before handling the first request.
with app.app_context():
    init()
    start_background_threads()

# ------------------------
# Standalone Server Execution
# ------------------------
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=30000, threaded=True)