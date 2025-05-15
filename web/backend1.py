import threading
import queue
import traceback
from flask import Flask, request, jsonify, abort
from flask_cors import CORS

# ------------------------
# Global constants
# ------------------------
MAX_PAYLOAD_LENGTH = 2048
MAX_QUEUE_LENGTH = 2400
BATCH_SIZE = 6
TOP_K = 100
FINAL_TOP_K = 20
RRF_K = 60

# ------------------------
# Global queues for workers
# ------------------------
embed_queue = queue.Queue()
faiss_queue = queue.Queue()

# ------------------------
# Job class to hold task state
# ------------------------
class Job:
    def __init__(self, payload, query_type):
        self.payload = payload
        self.query_type = query_type  # "leg" or "clic"
        self.embedding1 = None
        self.embedding2 = None
        self.doc_indices = None
        self.texts = None
        self.exception = None
        self.event = threading.Event()

# ------------------------
# Processing Functions
# ------------------------
import polars as pl
from sentence_transformers import SentenceTransformer
import torch
import faiss
import numpy as np

# Placeholders for loaded resources
corpus_leg = None
corpus_clic = None
model1 = None
model2 = None
cpu_index_linq_leg = None
cpu_index_inf_leg = None
cpu_index_linq_clic = None
cpu_index_inf_clic = None

def init():
    global corpus_leg, corpus_clic
    global model1, model2
    global cpu_index_linq_leg, cpu_index_inf_leg
    global cpu_index_linq_clic, cpu_index_inf_clic

    # Load corpora
    corpus_leg = pl.read_csv("../data/corpus_leg.csv").get_column("text").to_list()
    corpus_clic = pl.read_csv("../data/corpus_clic.csv").get_column("text").to_list()

    # Load embedding models
    model1 = SentenceTransformer(
        "../data/local_linq",
        trust_remote_code=False,
        device="cuda:3",
        model_kwargs={"torch_dtype": torch.float16}
    )
    model1.max_seq_length = 1024
    model1.encode(["warm up"])

    model2 = SentenceTransformer(
        "../data/local_inf_big",
        trust_remote_code=False,
        device="cuda:3",
        model_kwargs={"torch_dtype": torch.bfloat16}
    )
    model2.max_seq_length = 1024
    model2.encode(["warm up"])

    # Load FAISS indexes for LEG
    cpu_index_linq_leg = faiss.read_index("../data/linq_leg.faiss")
    cpu_index_inf_leg  = faiss.read_index("../data/inf_big_leg.faiss")
    cpu_index_linq_leg.hnsw.efSearch = TOP_K * 8
    cpu_index_inf_leg.hnsw.efSearch  = TOP_K * 8

    # Load FAISS indexes for CLIC
    cpu_index_linq_clic = faiss.read_index("../data/linq_clic.faiss")
    cpu_index_inf_clic  = faiss.read_index("../data/inf_big_clic.faiss")
    cpu_index_linq_clic.hnsw.efSearch = TOP_K * 8
    cpu_index_inf_clic.hnsw.efSearch  = TOP_K * 8

def create_linq_embedding(payloads):
    def l2_normalize(v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)
    prompt = "Instruct: Given a question, retrieve Wikipedia passages that answer the question\nQuery: "
    embs = model1.encode(payloads, prompt=prompt, show_progress_bar=False)
    return l2_normalize(embs)

def create_inf_embedding(payloads):
    def l2_normalize(v):
        norms = np.linalg.norm(v, axis=1, keepdims=True)
        return v / (norms + 1e-10)
    embs = model2.encode(payloads, prompt_name="query", show_progress_bar=False)
    return l2_normalize(embs)

def faiss_search(q1, q2, idx1, idx2):
    s1, i1 = idx1.search(q1.astype(np.float32), k=TOP_K)
    s2, i2 = idx2.search(q2.astype(np.float32), k=TOP_K)
    return s1, i1, s2, i2

def RRF(indices1, indices2):
    indices1 = indices1[0]
    indices2 = indices2[0]
    ranks1 = {doc: rank for rank, doc in enumerate(indices1)}
    ranks2 = {doc: rank for rank, doc in enumerate(indices2)}
    all_docs = set(indices1) | set(indices2)
    rrf_scores = {}
    for doc in all_docs:
        r1 = ranks1.get(doc, len(indices1) + 1)
        r2 = ranks2.get(doc, len(indices2) + 1)
        rrf_scores[doc] = 1/(RRF_K + r1) + 1/(RRF_K + r2)
    sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:FINAL_TOP_K]
    combined_indices = np.array([[doc for doc, _ in sorted_docs]])
    combined_scores  = np.array([[score for _, score in sorted_docs]])
    return combined_scores, combined_indices

# ------------------------
# Worker Threads
# ------------------------
def embedding_worker():
    while True:
        batch = []
        try:
            batch.append(embed_queue.get(timeout=0.02))
        except queue.Empty:
            continue
        while len(batch) < BATCH_SIZE:
            try:
                batch.append(embed_queue.get_nowait())
            except queue.Empty:
                break

        payloads = [job.payload for job in batch]
        try:
            embs1 = create_linq_embedding(payloads)
            for job, emb in zip(batch, embs1):
                job.embedding1 = emb
            embs2 = create_inf_embedding(payloads)
            for job, emb in zip(batch, embs2):
                job.embedding2 = emb
        except Exception:
            err = traceback.format_exc()
            for job in batch:
                job.exception = err
                job.event.set()
            continue

        for job in batch:
            faiss_queue.put(job)

def faiss_worker():
    while True:
        try:
            job = faiss_queue.get(timeout=0.02)
        except queue.Empty:
            continue

        try:
            e1 = job.embedding1.reshape(1, -1)
            e2 = job.embedding2.reshape(1, -1)
            # Select indexes & corpus
            if job.query_type == "leg":
                idx1, idx2, corpus = cpu_index_linq_leg, cpu_index_inf_leg, corpus_leg
            else:  # "clic"
                idx1, idx2, corpus = cpu_index_linq_clic, cpu_index_inf_clic, corpus_clic

            s1, i1, s2, i2 = faiss_search(e1, e2, idx1, idx2)
            rrf_scores, rrf_indices = RRF(i1, i2)
            top_idxs = rrf_indices[0][:FINAL_TOP_K]
            job.doc_indices = top_idxs
            job.texts = [corpus[i] for i in top_idxs]
        except Exception:
            job.exception = traceback.format_exc()
        finally:
            job.event.set()

# ------------------------
# Flask App
# ------------------------
app = Flask(__name__)
CORS(app)

@app.route('/api/', methods=['GET'])
def handle_request():
    payload = request.args.get("payload")
    qtype   = request.args.get("type")
    if payload is None:
        abort(400, description="Missing 'payload' parameter")
    if len(payload) > MAX_PAYLOAD_LENGTH:
        abort(400, description="Payload too long")
    if qtype not in ("leg", "clic"):
        abort(400, description="Invalid 'type'; must be 'leg' or 'clic'")
    if embed_queue.qsize() >= MAX_QUEUE_LENGTH or faiss_queue.qsize() >= MAX_QUEUE_LENGTH:
        abort(503, description="Server busy")

    job = Job(payload, qtype)
    embed_queue.put(job)

    if not job.event.wait(timeout=60):
        abort(504, description="Processing timed out")
    if job.exception:
        abort(500, description=f"Error during processing: {job.exception}")

    return jsonify({"texts": job.texts})

# ------------------------
# Startup
# ------------------------
threads_started = False
def start_background_threads():
    global threads_started
    if threads_started:
        return
    threading.Thread(target=embedding_worker, daemon=True).start()
    threading.Thread(target=faiss_worker, daemon=True).start()
    threads_started = True
    print("Background threads started")

with app.app_context():
    init()
    start_background_threads()

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=30000, threaded=True)