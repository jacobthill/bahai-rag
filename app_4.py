import os
import json
import shutil
import sys
import re
from flask import Flask, request, jsonify, render_template
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
import chromadb

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# --- Configuration ---
DATA_FOLDER = 'data'
ALLOWED_EXTENSIONS = {'jsonl', 'json'}
DB_DIRECTORY = "chroma_db_rag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("FATAL: GOOGLE_API_KEY not found in .env file.")
    sys.exit(1)

app = Flask(__name__)

# --- Global Variables ---
rag_components = {
    "retriever": None,
    "all_data": {}
}

# --- Utility Functions ---
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_all_data():
    """Loads all data from the local data folder into memory with greedy mapping."""
    all_data = {}
    if not os.path.exists(DATA_FOLDER):
        os.makedirs(DATA_FOLDER, exist_ok=True)
        return

    print(f"[*] Loading structured JSON files from '{DATA_FOLDER}'...")
    for filename in os.listdir(DATA_FOLDER):
        if not allowed_file(filename) or filename == "temp_docs.json":
            continue

        file_path = os.path.join(DATA_FOLDER, filename)
        file_key = filename.rsplit('.', 1)[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                if filename.endswith(".jsonl"):
                    data_list = [json.loads(line) for line in f if line.strip()]
                    # Greedy capture of annotation files
                    if file_key.startswith('annotations'):
                        if 'annotations' not in all_data: all_data['annotations'] = []
                        all_data['annotations'].extend(data_list)
                    else:
                        all_data[file_key] = data_list
                elif filename.endswith(".json"):
                     all_data[file_key] = json.load(f)
            except Exception as e:
                print(f" [!] Error loading {filename}: {e}")

    # Map lists to IDs (Greedy ID normalization)
    for key in list(all_data.keys()):
        if key != 'annotations' and isinstance(all_data[key], list):
            new_map = {}
            for item in all_data[key]:
                # Ensure we only map items that actually have an ID to avoid null keys
                if isinstance(item, dict) and item.get('id'):
                    new_map[item['id']] = item
            all_data[key] = new_map

    rag_components['all_data'] = all_data
    print(f"[+] Successfully loaded {len(all_data.keys())} data categories into RAM.")

def build_rag_index():
    try:
        load_all_data()
        all_data = rag_components['all_data']
        if not all_data: return False, "No data to index."

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(DB_DIRECTORY) and os.listdir(DB_DIRECTORY):
            print(f"[*] Loading existing vector index from '{DB_DIRECTORY}'...")
            client = chromadb.PersistentClient(path=DB_DIRECTORY)
            vector_store = Chroma(client=client, collection_name="rag_collection", embedding_function=embeddings)
            rag_components["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})
            return True, "Existing index loaded."

        print(f"[*] Building new index from '{DATA_FOLDER}'...")
        rich_documents = []

        # Greedy Manifestation Retrieval for Vector Store
        all_manifestations = {}
        for key, value in all_data.items():
            if 'manifestation' in key.lower() and isinstance(value, dict):
                all_manifestations.update(value)

        for m_id, m in all_manifestations.items():
            if isinstance(m, dict) and m.get('text'):
                rich_documents.append({
                    "text": f"Manifestation: {m_id}\n{m.get('text', '')}",
                    "metadata": {"manifestation_id": m_id}
                })

        if not rich_documents:
            return False, "No manifestation text found to index."

        temp_path = os.path.join(DATA_FOLDER, "temp_docs.json")
        with open(temp_path, 'w', encoding='utf-8') as f: json.dump(rich_documents, f)
        docs = JSONLoader(file_path=temp_path, jq_schema='.[].text', text_content=False).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

        client = chromadb.PersistentClient(path=DB_DIRECTORY)
        vector_store = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory=DB_DIRECTORY, collection_name="rag_collection")
        rag_components["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})
        os.remove(temp_path)
        return True, "New index built."
    except Exception as e:
        return False, f"Index build failed: {str(e)}"

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    if not query: return jsonify({"error": "Query empty"}), 400

    all_data = rag_components['all_data']

    # 1. Entity Extraction (Case-Insensitive)
    mentioned_ids = set(re.findall(r'\b(?:person|group|event|place|subject|work)_[a-zA-Z0-9_]+\b', query, re.IGNORECASE))
    discovery_ids = {i.lower() for i in mentioned_ids if not i.lower().startswith('work_')}

    works_to_process = {}

    # 2. Greedy Discovery Path (Scans all data categories for matching Work IDs)
    if discovery_ids:
        print(f"[*] Greedy searching for connections to: {discovery_ids}")
        for key, data_map in all_data.items():
            if 'works' in key.lower() and isinstance(data_map, dict):
                for w_id, w in data_map.items():
                    # Safely handle Author ID
                    author_data = w.get('author') or {}
                    author_id = author_data.get('id', "").lower() if isinstance(author_data, dict) else ""

                    # Safely handle Recipient IDs (List or Dict)
                    recipient_raw = w.get('recipient') or []
                    if isinstance(recipient_raw, dict): recipient_raw = [recipient_raw]
                    recipient_ids = {r.get('id', "").lower() for r in recipient_raw if isinstance(r, dict)}

                    # Merge for subset check
                    all_rel = recipient_ids
                    if author_id: all_rel.add(author_id)

                    if discovery_ids.issubset(all_rel):
                        w_copy = w.copy()
                        w_copy['id'] = w_id
                        works_to_process[w_id] = w_copy

    # 3. Greedy Deep Dive into Expressions, Manifestations & Annotations
    manifestation_texts = []
    annotation_mentions = []

    # Pre-collect all Expressions and Manifestations across all file sources
    all_expressions = []
    for k, v in all_data.items():
        if 'expression' in k.lower() and isinstance(v, dict):
            all_expressions.extend(v.values())

    all_manifestations = {}
    for k, v in all_data.items():
        if 'manifestation' in k.lower() and isinstance(v, dict):
            all_manifestations.update(v)

    # Global annotations list
    all_annotations = all_data.get('annotations', [])

    for w_id, w in works_to_process.items():
        # A. Traverse Expressions to find Manifestation Text
        rel_exprs = [e for e in all_expressions if e.get('work_id') == w_id]
        for expr in rel_exprs:
            # Check both manifestation list and canonical link
            m_ids = [m.get('manifestation_id') for m in expr.get('manifestations', []) if m.get('manifestation_id')]
            if expr.get('canonical_manifestation_id'):
                m_ids.append(expr.get('canonical_manifestation_id'))

            for m_id in set(m_ids):
                manif = all_manifestations.get(m_id)
                if manif and manif.get('text'):
                    manifestation_texts.append(f"Work: {w_id} | Manifestation: {m_id}\nContent: {manif['text']}")

        # B. Search Annotations for Entities mentioned in this Work
        rel_annots = [a for a in all_annotations if a.get('work_id') == w_id]
        for annot in rel_annots:
            ent = annot.get('entity_id')
            label = annot.get('label') or ent
            role = annot.get('role') or "Mentioned"
            if ent:
                annotation_mentions.append(f"- {label} ({ent}) as {role} in {w_id}")

    # 4. Fallback to Vector Search
    semantic_context = "\n\n".join(manifestation_texts)
    if not manifestation_texts:
        print("[*] Structured path yielded zero text. Falling back to Vector Search.")
        docs = rag_components["retriever"].invoke(query)
        semantic_context = "\n\n".join([d.page_content for d in docs])

    # 5. Final Synthesis
    struct_sum = "\n".join([f"- {w.get('title') or 'Untitled'} (ID: {w_id})" for w_id, w in works_to_process.items()])
    annot_sum = "\n".join(sorted(list(set(annotation_mentions)))) # Sorted and unique

    prompt = f"""Use the following context to answer:
    STRUCTURED WORKS IDENTIFIED:
    {struct_sum or "None"}

    TEXT CONTEXT (PRIMARY EVIDENCE):
    {semantic_context or "No relevant manifestation text found."}

    ENTITY ANNOTATIONS (PEOPLE, PLACES, EVENTS MENTIONED):
    {annot_sum or "No specific entity mentions found in annotation records."}

    QUERY: {query}
    """

    model = genai.GenerativeModel('models/gemini-pro-latest')
    response = model.generate_content(prompt)
    return jsonify({"status": "success", "answer": response.text})

if __name__ == '__main__':
    print("\n" + "="*50 + "\nBOOTING RAG SYSTEM\n" + "="*50)
    success, msg = build_rag_index()
    print(f"[*] {msg}")
    print("="*50 + "\n")
    app.run(debug=True, port=5005)
