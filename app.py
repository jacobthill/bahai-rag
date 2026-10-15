"""
RAG app (formerly app_8). Changes from app_7:
- Mention windows: instead of a manifestation's full text, the prompt gets
  WINDOW_CHARS of context on each side of every annotated mention.
- Page citations: an annotation may carry an optional "page" field (e.g. "151",
  "43–4", "214n47") and a manifestation an optional "citation" field (e.g.
  "Shahvar et al., The Baha'is of Iran, Transcaspia and the Caucasus, vol. 1").
  When present, they are shown in the prompt and the model cites "<citation>, p. <page>".
- Extra data folders: files in DATA_FOLDERS are merged, so copyrighted volumes
  can live in a separate folder (e.g. ../bahai-data/data/copyright). Missing folders are skipped.
"""
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
from google import genai
import chromadb

# Load environment variables
load_dotenv()

# Disable ChromaDB telemetry
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# --- Configuration ---
DATA_FOLDERS = ['../bahai-data/data', '../bahai-data/data/copyright']
ALLOWED_EXTENSIONS = {'jsonl', 'json'}
DB_DIRECTORY = "chroma_db_rag"
WINDOW_CHARS = 1500  # Context kept on each side of an annotated mention
GEMINI_MODEL = "gemini-pro-latest"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

if GEMINI_API_KEY:
    gemini_client = genai.Client(api_key=GEMINI_API_KEY)
else:
    print("FATAL: GEMINI_API_KEY not found in .env file.")
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
    """Loads all data from the data folders into memory with greedy mapping.
    Files with the same name in different folders are merged, not replaced."""
    all_data = {}

    for data_folder in DATA_FOLDERS:
        if not os.path.isdir(data_folder):
            continue

        print(f"[*] Loading structured files from '{data_folder}'...")
        for filename in os.listdir(data_folder):
            if not allowed_file(filename) or filename == "temp_docs.json":
                continue

            file_path = os.path.join(data_folder, filename)
            file_key = filename.rsplit('.', 1)[0]

            with open(file_path, 'r', encoding='utf-8') as f:
                try:
                    if filename.endswith(".jsonl"):
                        data_list = [json.loads(line) for line in f if line.strip()]
                    else:
                        data_list = json.load(f)
                except Exception as e:
                    print(f" [!] Error loading {file_path}: {e}")
                    continue

            # GREEDY ANNOTATION LOADING: Captures 'annotations_groups_with_offsets'
            if 'annotation' in file_key.lower():
                all_data.setdefault('annotations', []).extend(data_list)
            elif isinstance(data_list, list) and isinstance(all_data.get(file_key), list):
                all_data[file_key].extend(data_list)
            else:
                all_data[file_key] = data_list

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

        temp_path = os.path.join(DATA_FOLDERS[0], "temp_docs.json")
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

# --- Mention Windows ---
def flat(s):
    """Collapses whitespace so selector context can be compared to the text."""
    return " ".join(str(s or "").split())

def locate_mention(text, selector):
    """Finds the (start, end) of an annotation's exact quote in the original text.
    Whitespace may differ; prefix/suffix pick the right one among repeats. Returns None if absent."""
    exact = flat(selector.get('exact'))
    if not exact:
        return None
    pattern = r'\s+'.join(re.escape(word) for word in exact.split(' '))
    prefix, suffix = flat(selector.get('prefix')), flat(selector.get('suffix'))

    best, best_score = None, -1
    for m in re.finditer(pattern, text):
        score = 0
        if prefix and flat(text[max(0, m.start() - len(prefix) * 2):m.start()]).endswith(prefix): score += 1
        if suffix and flat(text[m.end():m.end() + len(suffix) * 2]).startswith(suffix): score += 1
        if score > best_score:
            best, best_score = (m.start(), m.end()), score
            if score == 2: break
    return best

def mention_excerpts(text, mentions):
    """Cuts text down to windows around each mention, merging overlaps.
    mentions: list of ((start, end), page). Returns list of (excerpt, pages)."""
    windows = []
    for (start, end), page in sorted(mentions):
        lo, hi = max(0, start - WINDOW_CHARS), min(len(text), end + WINDOW_CHARS)
        if windows and lo <= windows[-1][1]:
            windows[-1][1] = max(windows[-1][1], hi)
            if page: windows[-1][2].append(page)
        else:
            windows.append([lo, hi, [page] if page else []])

    excerpts = []
    for lo, hi, pages in windows:
        excerpt = ("[…] " if lo > 0 else "") + text[lo:hi].strip() + (" […]" if hi < len(text) else "")
        excerpts.append((excerpt, list(dict.fromkeys(pages))))
    return excerpts

# --- Routes ---
@app.route('/')
def index():
    return render_template('index.html')

def build_prompt(query):
    """Retrieves context for a query and returns (prompt, debug_info). No LLM call."""
    all_data = rag_components['all_data']

    # 1. Extraction: Get all entity/work IDs from the query
    id_pattern = r'\b(?:person|group|event|place|subject|work|circular_instruction)_[a-zA-Z0-9_]+\b'
    mentioned_ids = set(re.findall(id_pattern, query, re.IGNORECASE))
    search_ids = {i.lower() for i in mentioned_ids}
    print(f"search_ids: {search_ids}")

    all_manifestations = {}
    for k, v in all_data.items():
        if 'manifestation' in k.lower() and isinstance(v, dict):
            all_manifestations.update(v)

    def source_label(m_id):
        """Human-readable citation for a manifestation, falling back to its ID."""
        return (all_manifestations.get(m_id) or {}).get('citation') or m_id

    # 2. Discovery: Find everything linked to these IDs via Annotations
    works_to_process = {}
    manifestation_ids_to_load = set()
    annotation_mentions = []
    mentions_by_manifestation = {}  # m_id -> list of ((start, end), page)

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
            page = annot.get('page')
            if target_m_id:
                manifestation_ids_to_load.add(target_m_id)
                manif = all_manifestations.get(target_m_id)
                span = locate_mention(manif['text'], annot.get('target_selector') or {}) if manif and manif.get('text') else None
                mentions_by_manifestation.setdefault(target_m_id, [])
                if span:
                    mentions_by_manifestation[target_m_id].append((span, page))

            # Capture for synthesis
            label = body.get('label') or annot_value
            m_type = "HISTORICAL MARKER" if ('date' in annot_value or 'event' in annot_value) else "ENTITY"
            where = f"{source_label(target_m_id)}, p. {page}" if page else target_m_id
            annotation_mentions.append(f"- [{m_type}] {label} ({annot_value}) linked to {where}")

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

    # 3. Deep Dive: Collect windows of text around each mention
    manifestation_texts = []
    unlocated = []
    for m_id in sorted(manifestation_ids_to_load):
        manif = all_manifestations.get(m_id)
        if not (manif and manif.get('text')):
            continue
        text = manif['text']
        mentions = mentions_by_manifestation.get(m_id) or []
        if not mentions:
            # Quote not found in the text: fall back to the opening of the manifestation
            unlocated.append(m_id)
            mentions = [((0, 0), None)]
        for excerpt, pages in mention_excerpts(text, mentions):
            source = source_label(m_id)
            header = f"Source: {source}, p. {', '.join(pages)} | Manifestation: {m_id}" if pages else f"Source: {source} | Manifestation: {m_id}"
            manifestation_texts.append(f"{header}\nContent: {excerpt}")

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
    if unlocated:
        print(f"[!] Mention quote not found, used opening text instead: {unlocated}")

    prompt = f"""You are a precise historical researcher. Use the provided context to answer the query.

### INSTRUCTIONS:
1. Provide a detailed summary.
2. For every major claim, provide a DIRECT QUOTE from the 'TEXT EVIDENCE' section below.
3. Cite the source for every quote. If the evidence gives a Source and page, cite it as "Source, p. N"
   (e.g., Shahvar et al., vol. 1, p. 151). Otherwise cite the Source ID (e.g., work_123 or manifestation_xyz).
4. If the text is in Persian/Arabic, provide the original quote followed by a brief English explanation.
5. TEXT EVIDENCE contains excerpts around each mention ([…] marks omitted text), not always whole texts.

### STRUCTURED WORKS INVOLVED:
{struct_sum or "None identified."}

### TEXT EVIDENCE FROM MANIFESTATIONS:
{semantic_context or "No relevant manifestation text found."}

### ENTITY ANNOTATIONS:
{annot_sum or "No specific entity links found."}

### QUERY:
{query}
"""
    print(f"prompt length: {len(prompt):,} chars")

    debug = {
        "works_found": list(works_to_process.keys()),
        "manifestations_found": list(manifestation_ids_to_load),
        "excerpts": len(manifestation_texts),
        "mentions_not_located": unlocated,
        "prompt_chars": len(prompt),
    }
    return prompt, debug

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query', '')
    if not query: return jsonify({"error": "Query empty"}), 400

    prompt, debug = build_prompt(query)

    try:
        response = gemini_client.models.generate_content(model=GEMINI_MODEL, contents=prompt)
        return jsonify({
            "status": "success",
            "answer": response.text,
            "debug": debug
        })
    except Exception as e:
        return jsonify({"error": f"Model generation failed: {str(e)}"}), 500

if __name__ == '__main__':
    print("\n" + "="*50 + "\nBOOTING RAG SYSTEM\n" + "="*50)
    success, msg = build_rag_index()
    print(f"[*] {msg}")
    print("="*50 + "\n")
    app.run(debug=True, port=5005)
