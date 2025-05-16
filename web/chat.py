import json
import requests
from openai import AzureOpenAI
from dotenv import load_dotenv
import os

load_dotenv(override=True)

client = AzureOpenAI(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_API_ENDPOINT")
)
deployment = os.getenv("AZURE_OPENAI_API_DEPLOYMENT_NAME")

RETRIEVE_API_URL = "http://127.0.0.1:30000/api/"

# Map our query "option" to retrieve.py's expected qtype
OPTION_TO_QTYPE = {
    "cases": "leg",        # Hong Kong court judgments
    "ordinances": "clic"   # Hong Kong ordinances
}

def prompt1(interview):
    messages = [
        {"role": "system", "content": "You are an intelligent and logical lawyer providing legal advice on Hong Kong law. Hong Kong law is very similar to UK law."},
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
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages
    )
    return completion.choices[0].message.content

def prompt2(interview, response1):
    messages = [
        {"role": "system", "content": "You are an AI assistant specializing in generating query strings for vector search."},
        {
            "role": "user",
            "content": f"""{interview}
{response1}
Based on the areas of law identified, generate five query strings to search a legal database. The legal database has two options "cases" and "ordinances". "cases" contain all Hong Kong court judgments. "ordinances" contain all Hong Kong Ordinances in effect. Here are some questions for thought. What are the legal doctrine related to the current situation and the potential remedy/claim? What are some legal issues worth exploring? What might be specific about Hong Kong law that you will want to search?
The query strings generated shall encapsule the semantic meaning of the question you want to search. Expand abbreviations (CICT becomes Common Intention Constructive Trust) and remove specific information (Peter becomes Landlord, truck becomes vehicle) to enhance the effect of vector embeddings. As the database only contains Hong Kong law, do NOT include "Hong Kong" in your query. Output in json format, e.g.:
```json
[
    {{"query": "your query 1", "option": "cases"}},
    {{"query": "your query 2", "option": "cases"}},
    {{"query": "your query 3", "option": "ordinances"}},
    {{"query": "your query 4", "option": "ordinances"}},
    {{"query": "your query 5", "option": "ordinances"}}
]
```"""
        }
    ]
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages
    )
    return completion.choices[0].message.content

def prompt3(interview, response1, search_results):
    messages = [
        {"role": "system", "content": "You are an intelligent and logical AI legal assistant providing legal advice on Hong Kong law."},
        {
            "role": "user",
            "content": f"""Interview:
{interview}
Initial Analysis:
{response1}
Legal Sources:
{search_results}
Based only on the information given (legal sources might be irrelevant, be careful), write a detailed legal advice directly to the client in complete sentences. You must mention the relevant facts, a thorough explanation, and next steps the client should take. In your explanation, mention what the law is, analyze the law to the facts, and explain step by step the analysis to the layman client. Avoid legal jargon, explain as if the client is a high school student."""
        }
    ]
    completion = client.chat.completions.create(
        model=deployment,
        messages=messages
    )
    return completion.choices[0].message.content

def call_retrieve_api(query_text, option):
    """Call retrieve.py's Flask API and return the list of retrieved texts."""
    qtype = OPTION_TO_QTYPE.get(option)
    if not qtype:
        raise ValueError(f"Unknown option '{option}' for retrieval")
    resp = requests.get(
        RETRIEVE_API_URL,
        params={"payload": query_text, "type": qtype, "number": 5},
        timeout=30
    )
    resp.raise_for_status()
    data = resp.json()
    # data is {"texts": [...]} 
    return data["texts"]

if __name__ == "__main__":
    # Start initial greeting
    interview = "You: Hello, how can I help you?\n"
    print("GPT: Hello, how can I help you?")
    response1 = ""

    # Phase 1: interactive Q&A until "Information Complete"
    while True:
        client_input = input("Client: ")
        interview += f"Client: {client_input}\n"
        full_resp = prompt1(interview)
        if full_resp.startswith("Information Complete:"):
            # Strip the header and break
            response1 = full_resp.replace("Information Complete:", "").strip()
            break
        else:
            # Ask the follow-up question
            followup = full_resp.replace("Information Incomplete:", "").strip()
            print(f"GPT: {followup}")
            interview += f"You: {followup}\n"

    print("\n--- Full Interview ---")
    print(interview)
    print("\n--- Initial Analysis ---")
    print(response1)

    # Phase 2: generate queries
    raw_qs = prompt2(interview, response1)
    print("\n--- Generated Queries (raw) ---")
    print(raw_qs)

    # Load JSON array of {query, option}
    queries = json.loads(raw_qs.replace("```json", "").replace("```", ""))
    # Phase 3: retrieve for each query via HTTP
    all_texts = []
    for q in queries:
        texts = call_retrieve_api(q["query"], q["option"])
        all_texts.append(texts)

    # Flatten or keep nested depending on prompt3 expectation
    search_results = json.dumps(all_texts, ensure_ascii=False, indent=2)
    print("\n--- Retrieved Texts ---")
    print(search_results)

    # Phase 4: final advice
    response3 = prompt3(interview, response1, search_results)
    print("\nGPT (Legal Advice):")
    print(response3)