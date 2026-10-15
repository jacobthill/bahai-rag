import os
import json
import shutil
import sys
import re
from flask import Flask, request, jsonify, render_template
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
import chromadb

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# --- Configuration ---
DATA_FOLDER = '../bahai-data/data'
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

    print(f"[*] Loading structured files from '{DATA_FOLDER}'...")
    for filename in os.listdir(DATA_FOLDER):
        if not allowed_file(filename) or filename == "temp_docs.json":
            continue

        file_path = os.path.join(DATA_FOLDER, filename)
        file_key = filename.rsplit('.', 1)[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            try:
                if filename.endswith(".jsonl"):
                    data_list = [json.loads(line) for line in f if line.strip()]
                    # GREEDY ANNOTATION LOADING: Captures 'annotations_groups_with_offsets'
                    if 'annotation' in file_key.lower():
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
                if isinstance(item, dict) and item.get('id'):
                    new_map[str(item['id']).lower()] = item
            all_data[key] = new_map

    rag_components['all_data'] = all_data
    print(f"[+] Loaded {len(all_data.keys())} categories. Annotations count: {len(all_data.get('annotations', []))}")

def build_rag_index():
    try:
        load_all_data()
        all_data = rag_components['all_data']
        if not all_data: return False, "No data to index."

        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(DB_DIRECTORY) and os.listdir(DB_DIRECTORY):
            print(f"[*] Loading existing vector index...")
            client = chromadb.PersistentClient(path=DB_DIRECTORY)
            vector_store = Chroma(client=client, collection_name="rag_collection", embedding_function=embeddings)
            rag_components["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})
            return True, "Existing index loaded."

        print(f"[*] Building new index...")
        rich_documents = []

        # Collect Manifestations for Vector Store
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
            return False, "No text found for indexing."

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

    # 1. Extraction: Get all entity/work IDs from the query
    id_pattern = r'\b(?:person|group|event|place|subject|work|circular_instruction)_[a-zA-Z0-9_]+\b'
    mentioned_ids = set(re.findall(id_pattern, query, re.IGNORECASE))
    search_ids = {i.lower() for i in mentioned_ids}
    print(f"search_ids: {search_ids}")

    # 2. Discovery: Find everything linked to these IDs via Annotations
    works_to_process = {}
    manifestation_ids_to_load = set()
    annotation_mentions = []

    all_annotations = all_data.get('annotations', [])

    # PIVOT: Use your specific JSON schema (body -> value and target_manifestation)
    for annot in all_annotations:
        # Based on your data: {"body": {"value": "circular_instruction_..."}}
        body = annot.get('body', {})
        annot_value = str(body.get('value', "")).lower()

        # Based on your data: "target_manifestation": "manifestation_..."
        target_m_id = str(annot.get('target_manifestation', "")).lower()

        # If the ID in the 'body' matches one of our search IDs
        if annot_value in search_ids:
            if target_m_id:
                manifestation_ids_to_load.add(target_m_id)

            # Capture for synthesis
            label = body.get('label') or annot_value
            role = annot.get('role') or "Tagged Entity"
            m_type = "HISTORICAL MARKER" if ('date' in annot_value or 'event' in annot_value) else "ENTITY"
            annotation_mentions.append(f"- [{m_type}] {label} ({annot_value}) linked to {target_m_id}")

            # Optional: If you want to find the Work associated with this manifestation
            # (Requires a reverse lookup in your expressions data)
            for k, v in all_data.items():
                if 'expression' in k.lower() and isinstance(v, dict):
                    for expr in v.values():
                        # Check if this manifestation is part of this expression/work
                        m_ids = [str(m.get('manifestation_id')).lower() for m in expr.get('manifestations', [])]
                        if target_m_id in m_ids or str(expr.get('canonical_manifestation_id')).lower() == target_m_id:
                            w_id = str(expr.get('work_id')).lower()
                            if w_id in all_data.get('works', {}):
                                works_to_process[w_id] = all_data['works'][w_id]

    # 3. Deep Dive: Collect the actual Manifestation Texts
    manifestation_texts = []
    all_manifestations = {}
    for k, v in all_data.items():
        if 'manifestation' in k.lower() and isinstance(v, dict):
            all_manifestations.update(v)

    for m_id in manifestation_ids_to_load:
        manif = all_manifestations.get(m_id)
        if manif and manif.get('text'):
            manifestation_texts.append(f"Direct Match | Manifestation: {m_id}\nContent: {manif['text']}")

    # 4. Fallback: If structured links failed, try Vector Search
    semantic_context = "\n\n".join(manifestation_texts)
    if not manifestation_texts and not annotation_mentions:
        print("[*] No structured links found. Falling back to Vector Search.")
        docs = rag_components["retriever"].invoke(query)
        semantic_context = "\n\n".join([d.page_content for d in docs])

    # 5. Synthesis: Prepare prompt for Gemini
    struct_sum = "\n".join([f"- {v.get('title', 'Untitled')} ({k})" for k, v in works_to_process.items()])
    annot_sum = "\n".join(sorted(list(set(annotation_mentions))))

    # YOUR PRINTS
    print(f"works_to_process: {list(works_to_process.keys())}")
    print(f"manifestation_ids_to_load: {manifestation_ids_to_load}")
    print(f"annotation_mentions: {annotation_mentions}")

    prompt = f"""You are a precise historical researcher. Use the provided context to answer the query.

### INSTRUCTIONS:
1. Provide a detailed summary.
2. For every major claim, provide a DIRECT QUOTE from the 'TEXT EVIDENCE' section below.
3. Cite the Source ID (e.g., work_123 or manifestation_xyz) for every quote.
4. If the text is in Persian/Arabic, provide the original quote followed by a brief English explanation.

### STRUCTURED WORKS INVOLVED:
{struct_sum or "None identified."}

### TEXT EVIDENCE FROM MANIFESTATIONS:
{semantic_context or "No relevant manifestation text found."}

### ENTITY ANNOTATIONS:
{annot_sum or "No specific entity links found."}

### QUERY:
{query}
"""

    model = genai.GenerativeModel('models/gemini-pro-latest')
    try:
        response = model.generate_content(prompt)
        return jsonify({
            "status": "success",
            "answer": response.text,
            "debug": {
                "works_found": list(works_to_process.keys()),
                "manifestations_found": list(manifestation_ids_to_load)
            }
        })
    except Exception as e:
        return jsonify({"error": f"Model generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n" + "="*50 + "\nBOOTING RAG SYSTEM\n" + "="*50)
    success, msg = build_rag_index()
    print(f"[*] {msg}")
    print("="*50 + "\n")
    app.run(debug=True, port=5005)
