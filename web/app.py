import time
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
# Global queue for GPU worker
# ------------------------
gpu_queue = queue.Queue()

# ------------------------
# Job class to hold task state (GPU-based processing)
# ------------------------
class Job:
    def __init__(self, payload):
        self.payload = payload
        self.doc_ids = None      # Doc IDs returned from FAISS search
        self.texts = None        # Final texts corresponding to each doc id
        self.exception = None    # Exception info if an error occurs during processing
        self.event = threading.Event()  # Signals job completion

# ------------------------
# Processing Functions
# ------------------------
import polars as pl
from FlagEmbedding import BGEM3FlagModel
import faiss
import numpy as np

corpus = None
model = None
gpu_index = None

def init():
    global corpus, model, gpu_index
    # Load corpus texts from CSV.
    corpus = pl.read_csv("../data/corpus.csv").get_column("text").to_list()
    
    # Initialize the embedding model.
    model = BGEM3FlagModel('BAAI/bge-m3', use_fp16=True)
    
    # Load precomputed embeddings from a parquet file.
    embeddings = np.stack(pl.read_parquet("../data/embeddings.parquet")["embedding"].to_numpy()).astype(np.float32)
    
    # Initialize FAISS index on GPU.
    resource = faiss.StandardGpuResources()
    cpu_index = faiss.index_factory(embeddings.shape[-1], "Flat", faiss.METRIC_INNER_PRODUCT)
    gpu_index = faiss.index_cpu_to_gpu(resource, 0, cpu_index)
    gpu_index.train(embeddings)
    gpu_index.add(embeddings)
    del embeddings

def create_embedding(payloads):
    """Creates normalized embeddings for a batch of payloads."""
    def l2_normalize(vectors):
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        return vectors / (norms + 1e-10)
    q_embeddings = model.encode(
        payloads,
        batch_size=BATCH_SIZE,
        max_length=1024,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )["dense_vecs"]
    return l2_normalize(q_embeddings)

def faiss_search(q_embeddings):
    """Perform a FAISS search to retrieve top 20 similar document indices."""
    scores, indices = gpu_index.search(q_embeddings.astype(np.float32), k=20)
    return indices

def get_texts(doc_ids):
    """Retrieve texts from the corpus given a list/array of doc ids."""
    return [corpus[doc_id] for doc_id in doc_ids]

# ------------------------
# Worker Thread Function (GPU-only worker)
# ------------------------
def gpu_worker():
    while True:
        batch = []
        try:
            # Block until at least one job is available.
            job = gpu_queue.get(timeout=0.05)
            batch.append(job)
        except queue.Empty:
            continue

        # Gather additional jobs up to BATCH_SIZE without blocking.
        while len(batch) < BATCH_SIZE:
            try:
                job = gpu_queue.get_nowait()
                batch.append(job)
            except queue.Empty:
                break

        payloads = [job.payload for job in batch]
        try:
            embeddings = create_embedding(payloads)
        except Exception:
            error_info = traceback.format_exc()
            for job in batch:
                job.exception = error_info
                job.event.set()
            continue

        try:
            indices_batch = faiss_search(embeddings)
        except Exception:
            error_info = traceback.format_exc()
            for job in batch:
                job.exception = error_info
                job.event.set()
            continue

        for job, indices in zip(batch, indices_batch):
            try:
                job.doc_ids = indices
                job.texts = get_texts(indices)
            except Exception:
                job.exception = traceback.format_exc()
            finally:
                job.event.set()

# ------------------------
# Status Monitor Function
# ------------------------
def status_printer():
    """
    Periodically prints the GPU queue size every 10 seconds.
    """
    while True:
        print("----- Status Monitor -----")
        print("GPU Queue Size:", gpu_queue.qsize())
        print("--------------------------")
        time.sleep(10)

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
    job.event.wait(timeout=10)  # Wait up to 10 seconds.
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
        t_gpu = threading.Thread(target=gpu_worker, daemon=True)
        t_gpu.start()

        t_monitor = threading.Thread(target=status_printer, daemon=True)
        t_monitor.start()

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