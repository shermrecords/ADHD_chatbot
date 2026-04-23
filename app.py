import json
import os
from pathlib import Path

import chromadb
from dotenv import load_dotenv
from flask import Flask, jsonify, render_template, request
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

CHROMA_PATH = Path(os.getenv("CHROMA_PATH", "chroma_db"))
COLLECTION_NAME = "adhd_claims"
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
DEFAULT_TOP_K = 4
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")

DEBUG = True

SYSTEM_PROMPT = """You are an ADHD psychoeducation assistant.

Your role:
- explain ADHD clearly and naturally to a patient
- base your answer on the provided evidence when evidence is available

Style:
- speak like a clinician explaining things in plain language
- avoid academic phrasing unless the user explicitly asks for studies
- sound conversational, calm, and helpful

Behavior:
- answer the question directly in the first sentence
- then explain briefly in 1–2 short paragraphs
- include nuance where needed
- if the user asks for studies, use the provided citations when possible
- do not invent studies or facts
- do not mention claims, retrieval, or internal system details

Safety:
- do not invent facts
- if evidence is limited or mixed, say so plainly
- do not imply diagnosis or certainty from a brief description alone
"""

collection = None
embed_model = None
genai_client = None
initialized = False


def startup():
    global collection, embed_model, genai_client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    genai_client = genai.Client(api_key=api_key)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    collection = client.get_collection(COLLECTION_NAME)

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)


def ensure_started():
    global initialized
    if not initialized:
        if DEBUG:
            print("[DEBUG] Running startup()")
        startup()
        initialized = True


@app.route("/health")
def health():
    return jsonify({"ok": True})


def user_wants_sources(question: str) -> bool:
    triggers = [
        "source", "sources", "citation", "citations",
        "study", "studies", "paper", "papers",
        "what studies", "what papers", "show sources",
        "support this", "what supports this",
        "where did that come from", "what is that based on",
        "what's that based on", "research"
    ]
    q = question.lower()
    return any(t in q for t in triggers)


def retrieve(question: str):
    top_k = 3 if user_wants_sources(question) else DEFAULT_TOP_K

    query_embedding = embed_model.encode(
        question,
        normalize_embeddings=True
    ).tolist()

    return collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas"]
    )


def parse_citations(value):
    if value is None:
        return []

    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]

    if isinstance(value, str):
        value = value.strip()
        if not value:
            return []

        try:
            loaded = json.loads(value)
            if isinstance(loaded, list):
                return [str(v).strip() for v in loaded if str(v).strip()]
        except Exception:
            pass

        return [value]

    return [str(value).strip()]


def build_evidence_block(results):
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    blocks = []
    source_rows = []

    for i, (doc, meta) in enumerate(zip(docs, metas), start=1):
        title = meta.get("paper_title", "") or meta.get("paper_id", "Unknown source")
        year = meta.get("year", "")

        blocks.append(
            f"""[{i}]
Source: {title} ({year})
Claim: {doc}"""
        )

        source_rows.append({
            "paper_title": title,
            "year": year,
        })

    return "\n\n".join(blocks), source_rows


def build_prompt(question: str, evidence_block: str) -> str:
    return f"""Question:
{question}

Evidence:
{evidence_block}

Write a patient-friendly answer.

Requirements:
- answer the question directly
- you MUST use the evidence above
- explain clearly in 1–2 short paragraphs
- do not imply diagnosis
"""


def extract_response_text(response) -> str:
    try:
        if response.text:
            return response.text.strip()
    except Exception:
        pass

    parts = []
    for candidate in getattr(response, "candidates", []) or []:
        content = getattr(candidate, "content", None)
        if not content:
            continue
        for part in getattr(content, "parts", []) or []:
            if hasattr(part, "text") and part.text:
                parts.append(part.text)

    return "\n".join(parts).strip()


def generate_answer(prompt: str) -> str:
    response = genai_client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.2,
            max_output_tokens=800,
        ),
    )
    return extract_response_text(response)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        ensure_started()

        data = request.get_json(force=True)
        question = (data.get("message") or "").strip()

        if not question:
            return jsonify({"error": "Message is required."}), 400

        results = retrieve(question)
        evidence_block, source_rows = build_evidence_block(results)
        prompt = build_prompt(question, evidence_block)

        answer = generate_answer(prompt)

        return jsonify({
            "answer": answer,
            "show_sources": False,
            "sources": [],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    ensure_started()
    port = int(os.getenv("PORT", 5050))
    app.run(debug=True, host="0.0.0.0", port=port)