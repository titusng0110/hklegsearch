import time
import threading
import queue
import traceback
from flask import Flask, request, jsonify, abort, send_file

# ------------------------
# Global constants
# ------------------------
MAX_PAYLOAD_LENGTH = 2048         # Maximum characters allowed in the payload
MIN_REQUEST_INTERVAL = 0.1        # Seconds per IP between requests (rate limiting)
MAX_QUEUE_LENGTH = 32000          # Reject new requests if any queue reaches this length
BATCH_SIZE = 32                   # Process up to 32 items per batch for GPU queues
MAX_REQUESTS_PER_HOUR = 10000     # Maximum requests allowed per IP in a 1-hour window
BAN_DURATION = 86400              # Ban duration in seconds (24 hours)

# ------------------------
# Global queues for workers
# ------------------------
gpu_queue_embedding = queue.Queue()
gpu_queue_faiss = queue.Queue()
cpu_queue = queue.Queue()

# ------------------------
# Global dictionaries for rate limiting and banning (protected by a lock)
# ------------------------
ip_last_request = {}    # Tracks last request time for each IP (for minimum interval)
ip_request_counter = {} # Tracks count and window start time for each IP
banned_ips = {}         # Maps ip -> ban expiration time (timestamp)
ip_lock = threading.Lock()

# ------------------------
# Helper function to extract client IP reliably.
# ------------------------
def get_client_ip():
    """Attempt to extract client IP using X-Forwarded-For header if available."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        # Use the first IP in the header
        ip = forwarded.split(",")[0].strip()
    else:
        ip = request.remote_addr
    return ip

# ------------------------
# Job class to hold task state
# ------------------------
class Job:
    def __init__(self, ip, payload):
        self.ip = ip
        self.payload = payload
        self.embedding = None       # Set after GPU embedding processing
        self.doc_ids = None         # Set after FAISS search (20 items expected)
        self.texts = [None] * 20    # To hold the final texts for each doc id
        self.pending = 20           # Number of remaining CPU tasks
        self.event = threading.Event()  # Signals job completion
        self.lock = threading.Lock()     # To safely update job state

# ------------------------
# Dummy Functions to Simulate Processing Steps
# ------------------------
def dummy_create_embedding(payload: str):
    """Simulate GPU-based embedding creation (typically using PyTorch/TensorFlow)."""
    time.sleep(0.01)
    return "embedding_of_" + payload

def dummy_faiss_search(embedding):
    """Simulate a FAISS GPU search that returns a list of 20 document ids."""
    time.sleep(0.01)
    base = hash(embedding) % 10000
    return [base + i for i in range(20)]

def dummy_get_text(doc_id):
    """Simulate retrieving text for a document id (e.g. a SQLite lookup)."""
    time.sleep(0.005)
    return f"Document text for id {doc_id}"

# ------------------------
# Worker Thread Functions
# ------------------------
def gpu_worker():
    """
    One GPU worker thread that processes items from one of the two GPU queues.
    Inspects both queues and processes a batch (up to BATCH_SIZE) from the one with more pending items.
    """
    while True:
        try:
            a_size = gpu_queue_embedding.qsize()
            b_size = gpu_queue_faiss.qsize()

            # If both GPU queues are empty, sleep briefly.
            if a_size == 0 and b_size == 0:
                time.sleep(0.001)
                continue

            # Process the queue with more work.
            if a_size >= b_size:
                batch = []
                for _ in range(min(BATCH_SIZE, a_size)):
                    try:
                        job = gpu_queue_embedding.get_nowait()
                        batch.append(job)
                    except queue.Empty:
                        break
                # Process each job: compute embedding and enqueue into the FAISS queue.
                for job in batch:
                    job.embedding = dummy_create_embedding(job.payload)
                    gpu_queue_faiss.put(job)
            else:
                batch = []
                for _ in range(min(BATCH_SIZE, b_size)):
                    try:
                        job = gpu_queue_faiss.get_nowait()
                        batch.append(job)
                    except queue.Empty:
                        break
                # Process each job: perform FAISS search and enqueue CPU tasks.
                for job in batch:
                    job.doc_ids = dummy_faiss_search(job.embedding)
                    for idx, doc_id in enumerate(job.doc_ids):
                        cpu_queue.put((job, idx, doc_id))
        except Exception as e:
            # Catch all exceptions to avoid killing the thread; log the error.
            print("Exception in gpu_worker:", traceback.format_exc())
            time.sleep(0.01)

def cpu_worker():
    """
    A CPU worker thread that processes CPU tasks one by one.
    Each task retrieves a document text (simulated) and updates the job state.
    """
    while True:
        try:
            try:
                job_item = cpu_queue.get(timeout=0.01)
            except queue.Empty:
                time.sleep(0.001)
                continue
            job_obj, index, doc_id = job_item
            text = dummy_get_text(doc_id)
            with job_obj.lock:
                job_obj.texts[index] = text
                job_obj.pending -= 1
                if job_obj.pending == 0:
                    job_obj.event.set()
        except Exception as e:
            print("Exception in cpu_worker:", traceback.format_exc())
            time.sleep(0.01)

def status_printer():
    """
    Periodically prints queue sizes and IP statistics every 10 seconds.
    """
    while True:
        with ip_lock:
            print("----- Status Monitor -----")
            print("GPU Embedding Queue Size:", gpu_queue_embedding.qsize())
            print("GPU FAISS Queue Size:    ", gpu_queue_faiss.qsize())
            print("CPU Queue Size:          ", cpu_queue.qsize())
            print("Active IPs:", len(ip_last_request))
            print("Banned IPs:", len(banned_ips))
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
    ip = get_client_ip()
    now = time.time()

    # Rate limiting and banning logic.
    with ip_lock:
        # Check if IP is banned.
        ban_expiry = banned_ips.get(ip)
        if ban_expiry and now < ban_expiry:
            abort(403, description="Your IP is banned for 24 hours due to excessive requests.")
        elif ban_expiry and now >= ban_expiry:
            # Ban period is over; remove ban and reset counter.
            del banned_ips[ip]
            ip_request_counter.pop(ip, None)

        # Check minimum request interval.
        last_time = ip_last_request.get(ip, 0)
        if now - last_time < MIN_REQUEST_INTERVAL:
            abort(429, description="Too Many Requests: Only one request every 0.1 seconds allowed")
        ip_last_request[ip] = now

        # Update the per-IP request counter for the sliding 1-hour window.
        info = ip_request_counter.get(ip)
        if info is None:
            ip_request_counter[ip] = {"count": 1, "window_start": now}
        else:
            if now - info["window_start"] < 3600:
                info["count"] += 1
            else:
                # Reset the counter and window.
                info["window_start"] = now
                info["count"] = 1

            if info["count"] > MAX_REQUESTS_PER_HOUR:
                banned_ips[ip] = now + BAN_DURATION
                abort(403, description="Your IP is banned for 24 hours due to excessive requests.")

    # Ensure only GET methods are accepted.
    if request.method != 'GET':
        abort(405, description="Method Not Allowed")

    # Retrieve the payload from the query string.
    payload = request.args.get("payload")
    if payload is None:
        abort(400, description="Missing 'payload' query parameter")
    if len(payload) > MAX_PAYLOAD_LENGTH:
        abort(400, description="Payload exceeds maximum allowed length")

    # Check for queue overload before accepting new jobs.
    if (gpu_queue_embedding.qsize() >= MAX_QUEUE_LENGTH or
        gpu_queue_faiss.qsize() >= MAX_QUEUE_LENGTH or
        cpu_queue.qsize() >= MAX_QUEUE_LENGTH):
        abort(503, description="Server busy due to queue overload")

    # Create a job and enqueue it in the GPU embedding queue.
    job = Job(ip, payload)
    gpu_queue_embedding.put(job)

    # Wait until processing is complete (with a timeout safeguard).
    job.event.wait(timeout=10)  # Wait up to 10 seconds.
    if not job.event.is_set():
        abort(504, description="Processing timed out")

    # Return the final texts as JSON.
    return jsonify({"texts": job.texts})

# ------------------------
# Background Threads Starter
# ------------------------
threads_started = False

def start_background_threads():
    """Start the GPU, CPU, and monitor threads once per process."""
    global threads_started
    if not threads_started:
        t_gpu = threading.Thread(target=gpu_worker, daemon=True)
        t_gpu.start()

        t_cpu = threading.Thread(target=cpu_worker, daemon=True)
        t_cpu.start()

        t_monitor = threading.Thread(target=status_printer, daemon=True)
        t_monitor.start()

        threads_started = True
        print("Background threads started.")

# Start background threads before handling the first request.
with app.app_context():
    start_background_threads()

# ------------------------
# Standalone Server Execution
# ------------------------
if __name__ == '__main__':
    # In case you want to run the app standalone for development.
    start_background_threads()
    
    # HTTPS: Provide paths to your certificate and key files.
    ssl_context = ('cert.pem', 'key.pem')
    app.run(debug=False, host='0.0.0.0', port=30000, threaded=True, ssl_context=ssl_context)
