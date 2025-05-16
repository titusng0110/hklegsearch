import os
import uuid
import json
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from openai import AzureOpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

# Initialize OpenAI client
client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT")
)
deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")
RETRIEVE_API_URL = "http://127.0.0.1:30000/api/"

# In-memory session store
sessions = {}

def prompt1(interview: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent and logical lawyer providing legal advice on Hong Kong law. Hong Kong law is very similar to UK law."
        },
        {
            "role": "user",
            "content": f"""Here is an interview between you and the client:
{interview}
Here are the Questions:
1) Is the client the plaintiff or the defendant or seeking advice for potential scenario?
2) What are the areas of law (at most three) related to the case?
3) If the client is the plaintiff, what are all their respective potential remedies available to the client? If the client is the defendant, what are all their respective potential claims against the client?
If you do not have enough information to answer the Questions, or it is obvious the client has not finished telling their situation, ask the client for key facts that will help you answer the Questions. Ask only one question at a time so as not to overwhelm the client. Don't press the client if he/she doesn't know the answer to your question. Start with "Information Incomplete:\n" and output a question with the format as Example 1 and 2.
Example 1: Information Incomplete:
Can you tell me more about why your employer fired you?
Example 2: Information Incomplete:
What is the timeline of who lived in the flat and who made payments to the mortgage of the flat?
If you have enough information to answer the Questions, output your answer with the format as Example 3 and 4, start with "Information Complete:\n":
Example 3:
Information Complete:
The client is the plaintiff.
Area of law #1: contract law
Potential remedy: damages, rescission, termination
Area of law #2: tort law
Potential remedy: damages
Area of law #3: employment law
Potential remedy: damages
Example 4:
Information Complete:
The client is the defendant.
Area of law #1: land law
Potential claims: equitable interest
Area of law #2: trust law
Potential claims: equitable interest"""
        }
    ]
    resp = client.chat.completions.create(model=deployment, messages=messages, temperature=0.4, top_p=0.9)
    return resp.choices[0].message.content

def prompt2(interview: str, response1: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an AI assistant specializing in generating query strings for vector search."
        },
        {
            "role": "user",
            "content": f"""{interview}
{response1}
Based on the areas of law identified, generate five query strings to search a legal database. The legal database has two options "clic" and "leg". "clic" is a community legal information database of these topics: 
Alternative Dispute Resolution (ADR)
Anti-discrimination
Bankruptcy, Individual Voluntary Arrangement, Companies Winding Up
Business and Commerce
Civil Case
Common Traffic Offences
Competition Law
Consumer Complaints
Defamation
DIY Residential Tenancy Agreement
Employment Disputes
Enduring Powers of Attorney
Family, Matrimonial and Cohabitation
Freedom of Assembly, Procession, Demonstration (Public Order)
Hong Kong Legal System
Immigration
Insurance
Intellectual Property
Landlord & Tenant
Legal Aid
Maintenance and Safety of Property
Medical Treatment, Consent and Withdrawal
Medical Negligence
Personal Data Privacy
Personal Injuries
Police and Crime
Probate (Wills & Estate)
Protection for Investors and Structured Products
Redevelopment and Acquisition of Property
Sale and Purchase of Property
Sexual Offences
Taxation
Judicial Review
"leg" contains all Hong Kong legislations in effect. Here are some questions for thought. What are the legal doctrine related to the current situation and the potential remedy/claim? What are some legal issues worth exploring? What might be specific about Hong Kong law that you will want to search?
The query strings generated shall encapsule the semantic meaning of the question you want to search. Expand abbreviations (CICT becomes Common Intention Constructive Trust) and remove specific information (Peter becomes Landlord, truck becomes vehicle) to enhance the effect of vector embeddings. As the database only contains Hong Kong law, do NOT include "Hong Kong" in your query. Output in json format, e.g.:
```json
[
    {{"query": "Business and Commerce: What are the requirements for forming a private company?", "option": "clic"}},
    {{"query": "Probate (Wills & Estate): How to make a will?", "option": "clic"}},
    {{"query": "Civil Case: How to start a proceeding in the Small Claims Tribunal?", "option": "clic"}},
    {{"query": "What are the elements of drink driving?", "option": "leg"}},
    {{"query": "When does ownership of goods pass from seller to buyer?", "option": "leg"}}
]
```"""
        }
    ]
    resp = client.chat.completions.create(model=deployment, messages=messages, temperature=0.4, top_p=0.9)
    return resp.choices[0].message.content

def prompt3(interview: str, response1: str, search_results: str) -> str:
    messages = [
        {
            "role": "system",
            "content": "You are an intelligent and logical AI legal assistant providing legal advice on Hong Kong law."
        },
        {
            "role": "user",
            "content": f"""Interview:
{interview}
Initial Analysis:
{response1}
Legal Sources:
{search_results}
Based only on the information given (legal sources might be irrelevant, be careful), write a detailed legal advice directly to the client in complete sentences. You must mention the relevant facts, a thorough explanation, and next steps the client should take. In your explanation, mention what the law is, apply the law to the facts, and explain step by step the analysis to the layman client. Avoid legal jargon, explain as if the client is a high school student. Sign yourself as "CLIC-Chat 3.0". Do not address yourself as a lawyer or a legal professional. Do not provide further assistance other than the legal advice. Do not add any disclaimers."""
        }
    ]
    resp = client.chat.completions.create(model=deployment, messages=messages, temperature=0.4, top_p=0.9)
    return resp.choices[0].message.content

def call_retrieve_api(query_text: str, option: str, number: int = 5):
    resp = requests.get(
        RETRIEVE_API_URL,
        params={"payload": query_text, "type": option, "number": str(number)},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("texts", [])

# Flask app
app = Flask(__name__)
CORS(app)

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    message = data.get("message")
    if not message:
        return jsonify(error="`message` is required"), 400

    # New session
    if not session_id:
        session_id = uuid.uuid4().hex
        sessions[session_id] = {
            "interview": "You: Hello, how can I help you?\n",
            "response1": None
        }
    if session_id not in sessions:
        return jsonify(error="Invalid session_id"), 400

    state = sessions[session_id]
    # Append client message
    state["interview"] += f"Client: {message}\n"
    full_resp = prompt1(state["interview"])

    if full_resp.startswith("Information Complete:"):
        # Final analysis
        analysis = full_resp.replace("Information Complete:", "").strip()
        state["response1"] = analysis
        return jsonify(session_id=session_id, reply=analysis, complete=True)
    else:
        # Need more info
        followup = full_resp.replace("Information Incomplete:", "").strip()
        state["interview"] += f"You: {followup}\n"
        return jsonify(session_id=session_id, reply=followup, complete=False)

@app.route("/queries", methods=["POST"])
def queries():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    if not session_id or session_id not in sessions:
        return jsonify(error="Invalid or missing session_id"), 400
    state = sessions[session_id]
    if not state.get("response1"):
        return jsonify(error="Chat phase not complete"), 400

    raw = prompt2(state["interview"], state["response1"])
    # strip markdown fences if any
    clean = raw.strip()
    if clean.startswith("```"):
        # Remove first line
        clean = "\n".join(clean.split("\n")[1:])
    clean = clean.rstrip("```").strip()
    try:
        queries = json.loads(clean)
    except Exception as e:
        return jsonify(error=f"Failed to parse queries JSON: {e}", raw=raw), 500

    state["queries"] = queries
    return jsonify(queries=queries)

@app.route("/search", methods=["POST"])
def search():
    data = request.get_json(force=True)
    queries = data.get("queries")
    if not isinstance(queries, list):
        return jsonify(error="`queries` must be a list"), 400

    all_texts = []
    for q in queries:
        texts = call_retrieve_api(q["query"], q["option"])
        all_texts.append(texts)

    return jsonify(search_results=all_texts)

@app.route("/advice", methods=["POST"])
def advice():
    data = request.get_json(force=True)
    session_id = data.get("session_id")
    search_results = data.get("search_results")
    if not session_id or session_id not in sessions:
        return jsonify(error="Invalid or missing session_id"), 400
    if search_results is None:
        return jsonify(error="Missing `search_results`"), 400

    state = sessions[session_id]
    # Format search_results as JSON string for the prompt
    sr_text = json.dumps(search_results, ensure_ascii=False, indent=2)
    final_advice = prompt3(state["interview"], state["response1"], sr_text)
    return jsonify(advice=final_advice)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=30001, threaded=True)