import threading
import queue
import traceback
from flask import Flask, request, jsonify, abort, send_file

# ------------------------
# Global constants
# ------------------------
MAX_PAYLOAD_LENGTH = 2048         # Maximum characters allowed in the payload
MAX_QUEUE_LENGTH = 2400          # Reject new requests if the job queue reaches this length
BATCH_SIZE = 6                   # Number of payloads to process in a batch
TOP_K = 100                        # Number of top results to rerank
FINAL_TOP_K = 10                  # Number of final results to return
RRF_K = 60                        # Constant for Reciprocal Rank Fusion

# ------------------------
# Global queues for workers
# ------------------------
embed_queue = queue.Queue()         # Queue for initial embedding creation (GPU)
faiss_queue = queue.Queue()         # Queue for FAISS search operations (CPU)

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
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np

corpus = None
model1 = None
model2 = None
cpu_index1 = None
cpu_index2 = None


def init():
    global corpus, model1, model2, cpu_index1, cpu_index2
    # Load corpus texts from CSV.
    corpus = pl.read_csv("../data/corpus.csv").get_column("text").to_list()
    
    # Initialize the embedding model.
    model1 = SentenceTransformer("../data/local_linq", trust_remote_code=False, device="cuda:3", model_kwargs={ "torch_dtype": torch.float16 })
    model1.max_seq_length = 1024
    model1.encode(["Hello"])  # Warm-up
    model2 = SentenceTransformer("../data/local_inf_big", trust_remote_code=False, device="cuda:3", model_kwargs={ "torch_dtype": torch.bfloat16 })
    model2.max_seq_length = 1024
    model2.encode(["Hello"])  # Warm-up
    
    # Initialize FAISS index on CPU.
    cpu_index1 = faiss.read_index("../data/linq_index.faiss")
    cpu_index2 = faiss.read_index("../data/inf_big_index.faiss")
    cpu_index1.hnsw.efSearch = TOP_K*8
    cpu_index2.hnsw.efSearch = TOP_K*8
    

def create_linq_embedding(payloads):
    """Creates normalized embeddings for a batch of payloads using GTE model."""
    def l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    task = 'Given a question, retrieve Wikipedia passages that answer the question'
    prompt = f"Instruct: {task}\nQuery: "
    q_embeddings = model1.encode(
        payloads,
        prompt=prompt,
        show_progress_bar=False
    )
    return l2_normalize(q_embeddings)

def create_inf_embedding(payloads):
    """Creates normalized embeddings for a batch of payloads using GTE model."""
    def l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    q_embeddings = model2.encode(
        payloads,
        prompt_name="query",
        show_progress_bar=False
    )
    return l2_normalize(q_embeddings)
    
def faiss_search(q_embeddings1, q_embeddings2):
    """Perform a FAISS search to retrieve top similar document indices."""
    scores1, indices1 = cpu_index1.search(q_embeddings1.astype(np.float32), k=TOP_K)
    scores2, indices2 = cpu_index2.search(q_embeddings2.astype(np.float32), k=TOP_K)
    return scores1, indices1, scores2, indices2

def RRF(indices1, indices2):
    """
    Implements Reciprocal Rank Fusion to combine results from two different embeddings.
    
    Args:
        scores1: Similarity scores from first model
        indices1: Document indices from first model search
        scores2: Similarity scores from second model
        indices2: Document indices from second model search
        k: Constant in the RRF formula (typically 60)
        
    Returns:
        Tuple of (combined_scores, combined_indices) containing the fused results
    """
    # Get the first row for each result (since we're processing one query at a time)
    indices1 = indices1[0]
    indices2 = indices2[0]
    
    # Create a mapping of document ID to its rank position
    ranks1 = {doc_id: rank for rank, doc_id in enumerate(indices1)}
    ranks2 = {doc_id: rank for rank, doc_id in enumerate(indices2)}
    
    # Collect all unique document IDs from both result sets
    all_doc_ids = set(indices1) | set(indices2)
    
    # Calculate RRF score for each document
    rrf_scores = {}
    for doc_id in all_doc_ids:
        # Get the rank of the document in each list (or assign a large value if not found)
        rank1 = ranks1.get(doc_id, len(indices1) + 1)  # +1 to ensure it's not 0
        rank2 = ranks2.get(doc_id, len(indices2) + 1)
        
        # Calculate RRF score: the sum of reciprocal ranks with a constant k
        rrf_score = 1/(RRF_K + rank1) + 1/(RRF_K + rank2)
        rrf_scores[doc_id] = rrf_score
    
    # Sort documents by RRF score in descending order
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    
    # Take only the top FINAL_TOP_K documents
    top_docs = sorted_docs[:FINAL_TOP_K]
    
    # Extract document IDs and their scores
    combined_indices = np.array([[doc_id for doc_id, _ in top_docs]])
    combined_scores = np.array([[score for _, score in top_docs]])
    
    return combined_scores, combined_indices


# ------------------------
# Worker Thread Functions
# ------------------------

# GPU worker for creating embeddings (without batching)
def embedding_worker():
    while True:
        batch = []
        try:
            # Block until at least one job is available
            job = embed_queue.get(timeout=0.02)
            batch.append(job)
        except queue.Empty:
            continue

        # Gather additional jobs up to BATCH_SIZE without blocking
        while len(batch) < BATCH_SIZE:
            try:
                job = embed_queue.get_nowait()
                batch.append(job)
            except queue.Empty:
                break

        payloads = [job.payload for job in batch]
        
        # Create embeddings
        try:
            embeddings1 = create_linq_embedding(payloads)
            for job, embedding in zip(batch, embeddings1):
                job.embedding1 = embedding
            embeddings2 = create_inf_embedding(payloads)
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
            faiss_queue.put(job)

# CPU worker for FAISS search
def faiss_worker():
    while True:
        try:
            # Get one job at a time to process
            job = faiss_queue.get(timeout=0.02)
            
            try:
                # Reshape single embeddings for FAISS search
                embedding1 = job.embedding1.reshape(1, -1)
                embedding2 = job.embedding2.reshape(1, -1)
                
                # Perform FAISS search
                scores1, indices1, scores2, indices2 = faiss_search(embedding1, embedding2)
                
                # Apply RRF to fuse the results
                rrf_scores, rrf_indices = RRF(indices1, indices2)
                
                # Store the fused indices
                job.doc_indices = rrf_indices[0][:FINAL_TOP_K]  # Get the first row
                
                # Get the final text results
                job.texts = [corpus[idx] for idx in job.doc_indices]
                
            except Exception:
                error_info = traceback.format_exc()
                job.exception = error_info
            finally:
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
    if embed_queue.qsize() >= MAX_QUEUE_LENGTH or faiss_queue.qsize() >= MAX_QUEUE_LENGTH:
        abort(503, description="Server busy due to queue overload")

    # Create a job and enqueue it in the GPU queue.
    job = Job(payload)
    embed_queue.put(job)

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
    app.run(debug=False, host='127.0.0.1', port=30000, threaded=True)