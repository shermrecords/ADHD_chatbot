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


def startup():
    global collection, embed_model, genai_client

    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("Missing GEMINI_API_KEY")

    genai_client = genai.Client(api_key=api_key)

    client = chromadb.PersistentClient(path=str(CHROMA_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception as e:
        raise RuntimeError(
            f"Could not find Chroma collection '{COLLECTION_NAME}' at {CHROMA_PATH.resolve()}"
        ) from e

    embed_model = SentenceTransformer(EMBED_MODEL_NAME)


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
        population = meta.get("population", "")
        article_type = meta.get("article_type", "")
        evidence_strength = meta.get("evidence_strength", "")
        citations = parse_citations(meta.get("citations"))

        citation_text = "; ".join(citations[:3]) if citations else "None provided"

        blocks.append(
            f"""[{i}]
Source paper: {title} ({year})
Population: {population}
Article type: {article_type}
Evidence strength: {evidence_strength}
Main claim: {doc}
Supporting citations: {citation_text}"""
        )

        source_rows.append({
            "paper_title": title,
            "year": year,
            "claim_text": doc,
            "citations": citations,
        })

    return "\n\n".join(blocks), source_rows


def build_prompt(question: str, evidence_block: str) -> str:
    if user_wants_sources(question):
        return f"""Question:
{question}

Evidence:
{evidence_block}

Write a clear answer based only on the evidence.

Requirements:
- answer the question directly
- name the most relevant studies or reviews if they are provided
- if a meta-analysis or review is present, mention it first
- prefer the most direct evidence for the specific question
- do not give only a generic summary if specific studies are available
- do not rely on tangential sources if a more direct study is present
- mention key studies in plain language (author + year if possible)
- do not fabricate anything
- keep tone natural and understandable
"""

    return f"""Question:
{question}

Evidence:
{evidence_block}

Write a patient-friendly answer.



Requirements:

- answer the question directly in the first sentence

- you MUST use the information from the evidence above

- include at least one specific idea or finding from the evidence

- do not ignore the evidence or answer only from general knowledge

- explain the idea in natural, conversational language

- include real-life impact when possible

- vary wording based on the question

- write 1–2 short paragraphs

- do not imply diagnosis or certainty

"""


def unique_sources(source_rows):
    seen = set()
    output = []

    for row in source_rows:
        key = (row["paper_title"], row["year"])
        if key in seen:
            continue
        seen.add(key)
        output.append({
            "paper_title": row["paper_title"],
            "year": row["year"],
        })

    return output


def fallback_answer(question: str = "", source_rows=None):
    q = question.lower()

    if "organized" in q or "waste time" in q or "stay organized" in q:
        return (
            "Difficulty staying organized and managing time can be related to ADHD, "
            "but it is not specific to ADHD. A lot of people experience this for different reasons. "
            "When it is ADHD, those patterns tend to be ongoing and show up across multiple areas of life, "
            "such as work, daily responsibilities, routines, and follow-through."
        )

    if "focus on things i enjoy" in q or "focus on things i like" in q:
        return (
            "That can happen with ADHD, but it does not automatically mean ADHD. "
            "Many people with ADHD find that their attention is less consistent across different kinds of tasks, "
            "especially when something feels boring, repetitive, or hard to organize."
        )

    if user_wants_sources(question):
        return (
            "There is evidence suggesting that ADHD often persists into adulthood, "
            "including findings from review papers and meta-analyses, although the degree of persistence varies."
        )

    return (
        "ADHD can affect attention, organization, and impulse control in everyday life. "
        "In adults, this can show up as difficulty managing tasks, staying organized, or following through consistently."
    )


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


def looks_incomplete(text: str) -> bool:
    if not text:
        return True

    t = text.strip()

    # only reject VERY short answers
    if len(t) < 25:
        return True

    # obvious truncation cases only
    bad_endings = (
        "but", "and", "or", "because", "which", "that",
        "if", "when", "while"
    )

    lower = t.lower()
    if any(lower.endswith(e) for e in bad_endings):
        return True

    # broken trailing characters
    if t.endswith(("'", "’", '"')):
        return True

    return False


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

    if DEBUG:
        print("\n[DEBUG Gemini response]")
        print(response)

    return extract_response_text(response)


def generate_plain_answer(question: str) -> str:
    prompt = f"""Question:
{question}

Write a patient-friendly ADHD psychoeducation answer.

Requirements:
- answer directly in the first sentence
- finish the answer completely
- write 2 short paragraphs
- be supportive and conversational
- do not imply diagnosis or certainty
- explain that these experiences can happen for different reasons
- if ADHD is relevant, explain how it might contribute in plain language
- include one or two concrete everyday examples
"""

    response = genai_client.models.generate_content(
        model=GEMINI_MODEL_NAME,
        contents=prompt,
        config=types.GenerateContentConfig(
            system_instruction=SYSTEM_PROMPT,
            temperature=0.3,
            max_output_tokens=900,
        ),
    )

    if DEBUG:
        print("\n[DEBUG Gemini plain retry response]")
        print(response)

    return extract_response_text(response)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json(force=True)
        question = (data.get("message") or "").strip()

        if not question:
            return jsonify({"error": "Message is required."}), 400

        results = retrieve(question)
        if DEBUG:

            print("\n[DEBUG retrieved docs]")

            docs = results.get("documents", [[]])[0]

            metas = results.get("metadatas", [[]])[0]

            for i, (doc, meta) in enumerate(zip(docs, metas), start=1):

                title = meta.get("paper_title", "") or meta.get("paper_id", "Unknown")

                year = meta.get("year", "")

                print(f"{i}. {title} ({year})")

                print(f"   {doc}\n")

            print("-----\n")
        evidence_block, source_rows = build_evidence_block(results)
        prompt = build_prompt(question, evidence_block)

        answer = ""

        try:
            answer = generate_answer(prompt)
            if DEBUG:
                print("\n[DEBUG final answer repr]")
                print(repr(answer))
                print("[DEBUG final answer length]")
                print(len(answer))
                print()
        except Exception as e:
            if DEBUG:
                print("[DEBUG Gemini RAG error]", repr(e))
            answer = ""

        if looks_incomplete(answer):
            if DEBUG:
                print("[DEBUG] RAG answer incomplete -> trying plain retry")
            try:
                answer = generate_plain_answer(question)
                if DEBUG:
                    print("\n[DEBUG plain retry answer repr]")
                    print(repr(answer))
                    print("[DEBUG plain retry answer length]")
                    print(len(answer))
                    print()
            except Exception as e:
                if DEBUG:
                    print("[DEBUG Gemini plain retry error]", repr(e))
                answer = ""

        if looks_incomplete(answer):
            if DEBUG:
                print("[DEBUG] Using fallback answer")
            answer = fallback_answer(question, source_rows)

        wants_sources = user_wants_sources(question)

        return jsonify({
            "answer": answer,
            "show_sources": wants_sources,
            "sources": unique_sources(source_rows) if wants_sources else [],
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


startup()

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)