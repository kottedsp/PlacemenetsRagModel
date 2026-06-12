"""
=============================================================================
  PLACEMENT RAG — Flask API Backend
  Run: python3 rag_api.py
  Serves on: http://localhost:5000
=============================================================================
  pip install flask flask-cors
=============================================================================
"""

import os
import warnings
warnings.filterwarnings("ignore")

from collections import Counter
from flask import Flask, request, jsonify
from flask_cors import CORS

from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaLLM

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

app = Flask(__name__)
CORS(app)  # allow frontend to call the API

# ── Data ──────────────────────────────────────────────────────────────────

RAW_DATA = [
    {"company":"Amazon","role":"SDE","year":2025,"rounds":["Online Assessment","Technical","HR"],"topics":["DSA","Graphs","DP"],"difficulty":"Medium","cgpa_cutoff":7.5,"backlog_policy":"No active backlogs allowed","experience":"Focused on graph traversal and DP with optimization questions.","result":"Selected"},
    {"company":"Microsoft","role":"SDE","year":2025,"rounds":["Online Assessment","Technical","HR"],"topics":["DSA","System Design"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Coding plus basic system design discussion.","result":"Selected"},
    {"company":"Google","role":"SWE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DSA","Algorithms"],"difficulty":"Hard","cgpa_cutoff":8.0,"backlog_policy":"Strictly no backlogs","experience":"Hard problem solving with edge cases.","result":"Rejected"},
    {"company":"Meta","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DSA","Trees"],"difficulty":"Hard","cgpa_cutoff":8.0,"backlog_policy":"No active backlogs allowed","experience":"Tree problems and recursion-heavy questions.","result":"Rejected"},
    {"company":"Apple","role":"SDE","year":2025,"rounds":["Technical","HR"],"topics":["DSA","OS"],"difficulty":"Hard","cgpa_cutoff":8.0,"backlog_policy":"Strictly no backlogs","experience":"OS concepts and tricky coding problems.","result":"Rejected"},
    {"company":"Oracle","role":"SDE","year":2025,"rounds":["Online Assessment","Technical","HR"],"topics":["DBMS","SQL"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"SQL queries and DBMS concepts tested.","result":"Selected"},
    {"company":"Salesforce","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DSA","OOP"],"difficulty":"Medium","cgpa_cutoff":7.5,"backlog_policy":"No active backlogs allowed","experience":"Object-oriented concepts and coding.","result":"Selected"},
    {"company":"ServiceNow","role":"SDE","year":2025,"rounds":["Online Assessment","Technical","HR"],"topics":["DSA","Java"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Java coding and DSA questions.","result":"Selected"},
    {"company":"Uber","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["Graphs","DP"],"difficulty":"Hard","cgpa_cutoff":7.5,"backlog_policy":"Strictly no backlogs","experience":"Graph and DP heavy coding rounds.","result":"Rejected"},
    {"company":"Swiggy","role":"SDE","year":2025,"rounds":["Technical","HR"],"topics":["DSA","System Design"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"Up to 1 backlog allowed","experience":"Backend design and coding.","result":"Selected"},
    {"company":"Zomato","role":"SDE","year":2025,"rounds":["Technical","HR"],"topics":["DSA","LLD"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"Up to 1 backlog allowed","experience":"Low-level design and coding.","result":"Selected"},
    {"company":"Flipkart","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["Arrays","Trees"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Tree traversal and array optimization.","result":"Selected"},
    {"company":"Meesho","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DSA","Graphs"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Graph traversal problems.","result":"Selected"},
    {"company":"Paytm","role":"SDE","year":2025,"rounds":["Technical","HR"],"topics":["DSA","System Design"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Backlogs must be cleared before joining","experience":"Backend concepts and coding.","result":"Selected"},
    {"company":"Razorpay","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DSA","DP"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Dynamic programming problems.","result":"Selected"},
    {"company":"Adobe","role":"SDE","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DP","DSA"],"difficulty":"Hard","cgpa_cutoff":7.5,"backlog_policy":"No active backlogs allowed","experience":"Hard DP questions.","result":"Selected"},
    {"company":"Goldman Sachs","role":"Analyst","year":2025,"rounds":["Online Assessment","Technical","HR"],"topics":["Math","DSA"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Math-heavy coding.","result":"Selected"},
    {"company":"JPMorgan Chase","role":"Analyst","year":2025,"rounds":["Online Assessment","Technical"],"topics":["DSA","Java"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Java coding and DSA.","result":"Selected"},
    {"company":"Morgan Stanley","role":"Analyst","year":2025,"rounds":["Online Assessment","Technical"],"topics":["Arrays","DP"],"difficulty":"Medium","cgpa_cutoff":7.5,"backlog_policy":"No active backlogs allowed","experience":"Array and DP problems.","result":"Selected"},
    {"company":"Deloitte","role":"Analyst","year":2025,"rounds":["Aptitude","HR"],"topics":["Aptitude","Case Study"],"difficulty":"Easy","cgpa_cutoff":6.5,"backlog_policy":"Up to 2 backlogs allowed","experience":"Case study and reasoning.","result":"Selected"},
    {"company":"Accenture","role":"ASE","year":2025,"rounds":["Aptitude","Technical","HR"],"topics":["OOP","DBMS"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 1 backlog allowed","experience":"OOP and DBMS basics.","result":"Selected"},
    {"company":"Infosys","role":"SE","year":2025,"rounds":["Aptitude","HR"],"topics":["Aptitude","DBMS"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"Aptitude and DBMS basics.","result":"Selected"},
    {"company":"TCS","role":"Digital","year":2025,"rounds":["Aptitude","Technical"],"topics":["DSA","OOP"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 1 backlog allowed","experience":"Basic DSA and OOP.","result":"Selected"},
    {"company":"Wipro","role":"Project Engineer","year":2025,"rounds":["Aptitude","Technical"],"topics":["Java","SQL"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"Java and SQL basics.","result":"Selected"},
    {"company":"Cognizant","role":"Programmer Analyst","year":2025,"rounds":["Aptitude","HR"],"topics":["Aptitude","Coding"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"Easy coding and aptitude.","result":"Selected"},
    {"company":"Capgemini","role":"Analyst","year":2025,"rounds":["Aptitude","Technical"],"topics":["Pseudo","SQL"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"Pseudo code and SQL.","result":"Selected"},
    {"company":"IBM","role":"Associate Developer","year":2025,"rounds":["Technical","HR"],"topics":["DSA","Cloud"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Coding and cloud basics.","result":"Selected"},
    {"company":"HCL","role":"Software Engineer","year":2025,"rounds":["Aptitude","Technical"],"topics":["SQL","OS"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"SQL and OS basics.","result":"Selected"},
    {"company":"Tech Mahindra","role":"SE","year":2025,"rounds":["Aptitude","HR"],"topics":["Java","OOP"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"Java and OOP basics.","result":"Selected"},
    {"company":"SAP Labs","role":"Developer","year":2025,"rounds":["Technical","HR"],"topics":["Java","DSA"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Java and DSA coding.","result":"Selected"},
    {"company":"Zoho","role":"Developer","year":2025,"rounds":["Technical"],"topics":["DSA","C++"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"C++ coding and logic problems.","result":"Selected"},
    {"company":"Freshworks","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","JavaScript"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Frontend and DSA.","result":"Selected"},
    {"company":"InMobi","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","Graphs"],"difficulty":"Hard","cgpa_cutoff":7.5,"backlog_policy":"No active backlogs allowed","experience":"Graph problems.","result":"Rejected"},
    {"company":"Oyo","role":"SDE","year":2025,"rounds":["Technical","HR"],"topics":["DSA","System Design"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Backend and coding.","result":"Selected"},
    {"company":"Byju's","role":"Engineer","year":2025,"rounds":["Technical"],"topics":["DSA","Arrays"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Array problems.","result":"Selected"},
    {"company":"PhonePe","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","System Design"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"System design basics.","result":"Selected"},
    {"company":"Cred","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","DP"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"DP problems.","result":"Selected"},
    {"company":"Nykaa","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","SQL"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"SQL + coding.","result":"Selected"},
    {"company":"BigBasket","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","Trees"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Tree problems.","result":"Selected"},
    {"company":"Delhivery","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","Graphs"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Graph traversal.","result":"Selected"},
    {"company":"Gainsight","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","Java"],"difficulty":"Medium","cgpa_cutoff":7.0,"backlog_policy":"No active backlogs allowed","experience":"Java coding.","result":"Selected"},
    {"company":"Darwinbox","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","OOP"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"OOP + coding.","result":"Selected"},
    {"company":"HighRadius","role":"SDE","year":2025,"rounds":["Technical"],"topics":["DSA","Arrays"],"difficulty":"Medium","cgpa_cutoff":6.5,"backlog_policy":"Up to 1 backlog allowed","experience":"Array problems.","result":"Selected"},
    {"company":"ValueLabs","role":"SDE","year":2025,"rounds":["Technical"],"topics":["Java","SQL"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"Java and SQL basics.","result":"Selected"},
    {"company":"Cyient","role":"Engineer","year":2025,"rounds":["Technical"],"topics":["C","DSA"],"difficulty":"Easy","cgpa_cutoff":6.0,"backlog_policy":"Up to 2 backlogs allowed","experience":"C programming basics.","result":"Selected"},
]

# ── RAG Setup ─────────────────────────────────────────────────────────────

FAISS_INDEX_PATH = os.path.expanduser("~/placement_faiss_index")

def convert_record_to_text(r):
    rounds_str = ", ".join(r["rounds"])
    topics_str = ", ".join(r["topics"])
    result_verb = "was selected" if r["result"] == "Selected" else "was rejected"
    return (
        f"Company: {r['company']}\nRole: {r['role']}\nYear: {r['year']}\n"
        f"Outcome: {r['result']}\nDifficulty: {r['difficulty']}\n"
        f"Interview Rounds: {rounds_str}\nTopics Covered: {topics_str}\n"
        f"CGPA Cutoff: {r['cgpa_cutoff']}\nBacklog Policy: {r['backlog_policy']}\n"
        f"Experience: {r['experience']}\n"
        f"Summary: A candidate applying to {r['company']} for {r['role']} "
        f"{result_verb} in {r['year']}. Rounds: {rounds_str}. Topics: {topics_str}. "
        f"Difficulty: {r['difficulty']}. Min CGPA: {r['cgpa_cutoff']}. "
        f"Backlog policy: {r['backlog_policy']}."
    )

def init_rag():
    print("[RAG] Initialising...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    docs = [Document(page_content=convert_record_to_text(r), metadata={
        "company": r["company"], "role": r["role"], "difficulty": r["difficulty"],
        "cgpa_cutoff": r["cgpa_cutoff"], "backlog_policy": r["backlog_policy"],
        "result": r["result"], "topics": r["topics"],
    }) for r in RAW_DATA]

    if os.path.exists(FAISS_INDEX_PATH):
        vs = FAISS.load_local(FAISS_INDEX_PATH, embeddings, allow_dangerous_deserialization=True)
    else:
        vs = FAISS.from_documents(docs, embeddings)
        vs.save_local(FAISS_INDEX_PATH)

    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 6, "fetch_k": 15, "lambda_mult": 0.7},
    )
    llm = OllamaLLM(model="llama3", temperature=0.1)

    PROMPT_TEMPLATE = """You are a placement advisor with access to verified campus recruitment records.

STRICT RULES:
1. Answer ONLY using the CONTEXT provided below.
2. If the context does not contain enough information, say: "I don't have enough data in my records to answer this."
3. Do NOT guess or use outside knowledge.
4. Always mention the company name for any specific data point.
5. For eligibility questions, always state CGPA cutoff AND backlog policy.
6. Be concise and structured. Use bullet points where helpful.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:"""

    prompt = PromptTemplate(input_variables=["context", "question"], template=PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n---\n\n".join(doc.page_content for doc in docs)

    retrieve = RunnableParallel(
        context=retriever | format_docs,
        question=RunnablePassthrough(),
        source_docs=retriever,
    )

    def answer_with_sources(inputs):
        answer = (prompt | llm | StrOutputParser()).invoke({
            "context": inputs["context"],
            "question": inputs["question"],
        })
        sources = list(dict.fromkeys(doc.metadata["company"] for doc in inputs["source_docs"]))
        return {"answer": answer, "sources": sources}

    chain = retrieve | answer_with_sources
    print("[RAG] Ready!")
    return chain

# Initialise once at startup
rag_chain = init_rag()

# ── API Routes ────────────────────────────────────────────────────────────

@app.route("/api/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "No question provided"}), 400
    try:
        result = rag_chain.invoke(question)
        return jsonify({"answer": result["answer"], "sources": result["sources"]})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/eligibility", methods=["POST"])
def eligibility():
    data = request.get_json()
    cgpa = float(data.get("cgpa", 0))
    has_backlog = bool(data.get("has_backlog", False))

    eligible = []
    for r in RAW_DATA:
        cgpa_ok = cgpa >= r["cgpa_cutoff"]
        policy = r["backlog_policy"].lower()
        backlog_ok = (
            not has_backlog
            or "up to 1" in policy
            or "up to 2" in policy
            or "cleared before joining" in policy
        )
        if cgpa_ok and backlog_ok:
            eligible.append({
                "company": r["company"], "role": r["role"],
                "difficulty": r["difficulty"], "cgpa_cutoff": r["cgpa_cutoff"],
                "backlog_policy": r["backlog_policy"], "topics": r["topics"],
                "rounds": len(r["rounds"]),
            })

    eligible.sort(key=lambda x: x["cgpa_cutoff"], reverse=True)
    return jsonify({"count": len(eligible), "companies": eligible})


@app.route("/api/stats", methods=["GET"])
def stats():
    all_topics = [t for r in RAW_DATA for t in r["topics"]]
    topic_counts = Counter(all_topics).most_common(10)
    diff_counts = Counter(r["difficulty"] for r in RAW_DATA)
    selected = sum(1 for r in RAW_DATA if r["result"] == "Selected")
    return jsonify({
        "total": len(RAW_DATA),
        "selected": selected,
        "top_topics": [{"topic": t, "count": c} for t, c in topic_counts],
        "difficulty": dict(diff_counts),
    })


@app.route("/api/companies", methods=["GET"])
def companies():
    return jsonify([{
        "company": r["company"], "role": r["role"],
        "difficulty": r["difficulty"], "cgpa_cutoff": r["cgpa_cutoff"],
        "topics": r["topics"], "result": r["result"],
        "rounds": r["rounds"],
    } for r in RAW_DATA])


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    print("\n" + "="*50)
    print("  Placement RAG API — http://localhost:5000")
    print("="*50 + "\n")
    app.run(debug=False, port=5000)
