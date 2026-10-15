"""
This script creates a local RAG (Retrieval-Augmented Generation) system
using Flask, LangChain, ChromaDB, and the Google Gemini API.
It uses a local model for embeddings and the Gemini API for generation directly.
This version implements a structured-first hybrid search strategy and includes debug endpoints.
"""
import os
import json
import shutil
import sys
import re
from flask import Flask, request, jsonify, render_template
from werkzeug.utils import secure_filename
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import JSONLoader
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from dotenv import load_dotenv
from tqdm import tqdm
import google.generativeai as genai
import chromadb

# Load environment variables from .env file
load_dotenv()

# Forcefully disable ChromaDB telemetry to prevent errors and clean up console output.
os.environ['ANONYMIZED_TELEMETRY'] = 'False'

# --- Configuration ---
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'jsonl', 'json'}
DB_DIRECTORY = "chroma_db_rag"
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Configure the Gemini client
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
else:
    print("FATAL: GOOGLE_API_KEY not found in .env file. Exiting.")
    sys.exit(1)

# --- Flask App Initialization ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Global Variables ---
rag_components = {
    "retriever": None,
    "all_data": {} # Will hold all loaded, structured data in memory
}

# --- Utility Functions ---
def allowed_file(filename):
    """Checks if the file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def json_object_hook_for_duplicates(ordered_pairs):
    """
    A custom object hook for json.loads that raises an error on duplicate keys.
    """
    d = {}
    for key, value in ordered_pairs:
        if key in d:
            raise ValueError(f"Duplicate key found: '{key}'")
        else:
            d[key] = value
    return d

def load_all_data():
    """Loads all data from the uploads folder into memory."""
    all_data = {}
    for filename in os.listdir(UPLOAD_FOLDER):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        file_key = filename.rsplit('.', 1)[0]

        with open(file_path, 'r', encoding='utf-8') as f:
            if filename.endswith(".jsonl"):
                data_list = [json.loads(line) for line in f if line.strip()]
                if file_key.startswith('annotations_'):
                    if 'annotations' not in all_data:
                        all_data['annotations'] = []
                    all_data['annotations'].extend(data_list)
                else:
                    all_data[file_key] = data_list
            elif filename.endswith(".json"):
                 data_content = json.load(f)
                 all_data[file_key] = data_content
            # --- REMOVED test print statement ---

    problematic_files = set()
    for key in all_data:
        if key != 'annotations' and isinstance(all_data[key], list):
            new_map, has_missing_id = {}, False
            for item in all_data[key]:
                if 'id' in item: new_map[item['id']] = item
                else: has_missing_id = True
            if has_missing_id: problematic_files.add(key)
            all_data[key] = new_map

    if problematic_files:
        raise ValueError(f"Entries with missing 'id' found in: {', '.join(problematic_files)}.")

    rag_components['all_data'] = all_data
    print("All structured data loaded into memory.")

# --- Core RAG Logic ---
def build_rag_index():
    try:
        load_all_data()
        all_data = rag_components['all_data']

        # DEBUG PRINTS
        print(f"DEBUG: Keys found in all_data: {list(all_data.keys())}")
        for key in all_data:
            print(f"DEBUG: Found {len(all_data[key])} entries in {key}")
        # DEBUG END

        print("Creating rich documents from all available metadata...")
        rich_documents = []
        manifestations = all_data.get('manifestations') or all_data.get('manifestations_canonical') or {}

        def get_display_name_from_dict(item_dict, item_key):
            if not (isinstance(item_dict, dict) and 'id' in item_dict): return None
            entity_id = item_dict.get('id') or ''
            entity_label = item_dict.get('label')
            entity_type_key = 'people'
            if 'group' in item_key or entity_id.startswith('group_'): entity_type_key = 'groups'
            elif 'place' in item_key or entity_id.startswith('place_'): entity_type_key = 'places'
            entity_record = all_data.get(entity_type_key, {}).get(entity_id)
            return (entity_record.get('name') if entity_record else None) or entity_label

        for manif_id, manif in manifestations.items():
            context_parts = []
            found_expression = next((expr for expr in all_data.get('expressions', {}).values() if any(m.get('manifestation_id') == manif_id for m in expr.get('manifestations', []))), None)
            work = all_data.get('works', {}).get(found_expression.get('work_id')) if found_expression else None
            combined_metadata = {**(work or {}), **(found_expression or {}), **manif}

            for key, value in combined_metadata.items():
                if key in ['id', 'manifestations', 'text'] or not value: continue # --- FIX: Also ignore 'text' key here
                if isinstance(value, list):
                    names = [get_display_name_from_dict(item, key) for item in value if isinstance(item, dict)]
                    if names: context_parts.append(f"{key.replace('_', ' ').title()}: {', '.join(filter(None, names))}")
                elif isinstance(value, dict) and 'id' in value:
                    name = get_display_name_from_dict(value, key)
                    if name: context_parts.append(f"{key.replace('_', ' ').title()}: {name}")
                elif key.endswith('_id'):
                    entity_type, entity_id = key.replace('_id','').rstrip('s') + 's', value
                    record = all_data.get(entity_type, {}).get(entity_id)
                    if record:
                        name = record.get('name') or record.get('title')
                        if name: context_parts.append(f"{key.replace('_id', '').title()}: {name}")
                elif isinstance(value, (str, int, float, bool)):
                         context_parts.append(f"{key.replace('_', ' ').title()}: {value}")

            # --- FIX: Use correct annotation keys ('target_manifestation' and 'body.label') ---
            annotations_for_manif = [ann for ann in all_data.get('annotations', []) if ann.get('target_manifestation') == manif_id]
            related_entities = [ann.get('body', {}).get('label') for ann in annotations_for_manif if ann.get('body', {}).get('label')]
            if related_entities: context_parts.append(f"Related To: {', '.join(sorted(list(set(filter(None, related_entities)))))}")

            context = "\n".join(sorted(context_parts))
            # --- FIX: Changed 'text_content' to 'text' ---
            full_text = f"{context}\n---\n{manif.get('text', '')}"
            rich_documents.append({"text": full_text, "metadata": {"manifestation_id": manif_id, "work_title": (work.get('title') if work else 'N/A')}})

        temp_json_path = os.path.join(UPLOAD_FOLDER, "temp_docs.json")
        with open(temp_json_path, 'w', encoding='utf-8') as f: json.dump(rich_documents, f)

        docs = JSONLoader(file_path=temp_json_path, jq_schema='.[].text', text_content=False).load()
        splits = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)

        print("Initializing local embedding model...")
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        if os.path.exists(DB_DIRECTORY):
            print(f"Removing old database directory: {DB_DIRECTORY}")
            shutil.rmtree(DB_DIRECTORY)
        print("Initializing new ChromaDB vector store...")

        client = chromadb.PersistentClient(path=DB_DIRECTORY)
        collection = client.get_or_create_collection(name="rag_collection")
        vector_store = Chroma(client=client, collection_name="rag_collection", embedding_function=embeddings)

        batch_size = 4000
        for i in tqdm(range(0, len(splits), batch_size), desc="Embedding Documents in Batches"):
            vector_store.add_documents(documents=splits[i:i + batch_size])

        rag_components["retriever"] = vector_store.as_retriever(search_kwargs={"k": 5})
        os.remove(temp_json_path)
        return True, f"Index built successfully with {len(splits)} document chunks."
    except Exception as e:
        print(f"Error building index: {e}")
        return False, str(e)


# --- Flask Routes ---
@app.route('/')
def index(): return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_files():
    files = request.files.getlist('files')
    if not files or all(f.filename == '' for f in files):
        return jsonify({"status": "error", "message": "No files selected"}), 400

    validated_files_content = []
    print("\n--- Starting File Upload and Validation ---")

    for file in files:
        if not (file and file.filename and allowed_file(file.filename)):
            if file and file.filename: return jsonify({"status": "error", "message": f"File type not allowed for {file.filename}."}), 400
            continue

        filename = secure_filename(file.filename)
        print(f"Processing '{filename}'...")

        try:
            file_content = file.read().decode('utf-8')
            if filename.endswith('.jsonl'):
                lines = file_content.strip().split('\n')
                for i, line in enumerate(lines):
                    if not line: continue
                    try:
                        json.loads(line, object_pairs_hook=json_object_hook_for_duplicates)
                    except ValueError as e:
                        try:
                            record = json.loads(line)
                            record_id = record.get('id', 'ID not found')
                            raise ValueError(f"{e} in record with id='{record_id}' on line {i+1}")
                        except json.JSONDecodeError:
                            raise ValueError(f"Invalid JSON on line {i+1}: {line}")
            else:  # .json
                try:
                    data = json.loads(file_content)
                    if isinstance(data, list):
                        for i, record in enumerate(data):
                            record_str = json.dumps(record)
                            try:
                                json.loads(record_str, object_pairs_hook=json_object_hook_for_duplicates)
                            except ValueError as e:
                                record_id = record.get('id', 'ID not found')
                                raise ValueError(f"{e} in record with id='{record_id}' (at list index {i})")
                    else:
                        json.loads(file_content, object_pairs_hook=json_object_hook_for_duplicates)
                except ValueError as e:
                    raise e
                except json.JSONDecodeError as e:
                    raise ValueError(f"File contains invalid JSON. {e}")

            validated_files_content.append((filename, file_content))
            print(f"'{filename}' validated successfully.")
        except ValueError as e: # Catches duplicate key error
            return jsonify({"status": "error", "message": f"Validation Error in '{filename}': {e}"}), 400
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error parsing '{filename}': {e}"}), 500

    try:
        print("\n--- All files validated. Saving... ---")
        if os.path.exists(UPLOAD_FOLDER): shutil.rmtree(UPLOAD_FOLDER)
        os.makedirs(UPLOAD_FOLDER)
        for filename, content in validated_files_content:
            with open(os.path.join(UPLOAD_FOLDER, filename), 'w', encoding='utf-8') as f: f.write(content)
        print(f"Successfully saved {len(validated_files_content)} files.")
    except Exception as e:
        return jsonify({"status": "error", "message": "Server error during file save."}), 500

    return jsonify({"status": "success", "message": f"{len(validated_files_content)} files uploaded."})

@app.route('/index', methods=['POST'])
def create_index():
    success, message = build_rag_index()
    return jsonify({"status": "success", "message": message}) if success else jsonify({"status": "error", "message": f"Failed to build index: {message}"}), 500

def find_related_manifestation_ids(entity_id, all_data, debug=False):
    manifestation_ids = set()
    debug_info = {"entity_id": entity_id, "related_works": [], "related_annotations": []}

    works, expressions, manifestations = all_data.get('works', {}), all_data.get('expressions', {}), all_data.get('manifestations', {})

    work_to_expressions = {}
    for expr_id, expr in expressions.items():
        work_id = expr.get('work_id')
        if work_id:
            if work_id not in work_to_expressions:
                work_to_expressions[work_id] = []
            work_to_expressions[work_id].append(expr_id)

    expression_to_manifestations = {expr_id: [m.get('manifestation_id') for m in expr.get('manifestations', []) if m.get('manifestation_id')] for expr_id, expr in expressions.items()}

    for work_id, work in works.items():
        is_related = False
        author_info = work.get('author')
        if isinstance(author_info, dict) and author_info.get('id') == entity_id:
            is_related = True

        recipient_info = work.get('recipient')
        if isinstance(recipient_info, list):
            if any(isinstance(r, dict) and r.get('id') == entity_id for r in recipient_info):
                is_related = True

        if is_related:
            work_debug_info = {
                "work_id": work_id,
                "work_title": work.get('title'),
                "author": work.get('author'),
                "recipient": work.get('recipient'),
                "expressions": []
            }
            expression_ids = work_to_expressions.get(work_id, [])
            if not expression_ids and debug: work_debug_info["note"] = "No expressions found with this work_id."
            for expr_id in expression_ids:
                manifs = expression_to_manifestations.get(expr_id, [])
                manifestation_ids.update(manifs)
                if debug: work_debug_info["expressions"].append({"expression_id": expr_id, "found_manifestations": manifs})
            if debug: debug_info["related_works"].append(work_debug_info)

    for ann in all_data.get('annotations', []):
        # --- FIX: Use correct annotation keys ('body.value' and 'target_manifestation') ---
        target_entity_id = ann.get('body', {}).get('value')
        target_manif_id = ann.get('target_manifestation')
        if target_entity_id == entity_id and target_manif_id in manifestations:
            manifestation_ids.add(target_manif_id)
            if debug: debug_info["related_annotations"].append({"annotation_id": ann.get('id'), "manifestation_id": target_manif_id})

    if debug:
        debug_info["total_unique_manifestations_found"] = len(manifestation_ids)
        debug_info["final_manifestation_id_list"] = sorted(list(manifestation_ids))
        return debug_info

    return manifestation_ids

def get_entity_details(entity_id, all_data):
    """Helper function to find and get details for any entity ID."""
    for entity_type in ['people', 'places', 'groups', 'subjects', 'events', 'works']:
        entity_map = all_data.get(entity_type, {})
        if entity_id in entity_map:
            return entity_map[entity_id]
    return None

@app.route('/query', methods=['POST'])
def handle_query():
    data = request.get_json()
    query = data.get('query')
    if not query: return jsonify({"status": "error", "message": "Query cannot be empty"}), 400
    if not rag_components.get("retriever") or not rag_components.get("all_data"):
        return jsonify({"status": "error", "message": "RAG system not initialized."}), 500

    try:
        all_data = rag_components['all_data']
        structured_summary = ""
        metadata_context = ""
        semantic_context = ""
        sources = []

        # --- 1. Entity Extraction ---
        mentioned_entity_ids = set(re.findall(r'\b(?:person|group|event|place|subject|work)_[a-zA-Z0-9_]+\b', query))
        print(f"\n--- 1. Entity Extraction ---")
        print(f"Found {len(mentioned_entity_ids)} entity IDs in query: {mentioned_entity_ids}")

        # --- 2. Structured-First Retrieval ---
        # ---- 2.A. Structured Search
        print("\n--- 2.A. Structured Retrieval: Sorting Entities ---")

        # These will hold the *IDs*
        lookup_work_ids = set()
        discovery_entity_ids = set()

        # 1. Sort all extracted IDs into two buckets
        for entity_id in mentioned_entity_ids:
            if entity_id.startswith('work_'):
                lookup_work_ids.add(entity_id)
            else:
                # Assumes person, place, group, etc. are for discovery
                discovery_entity_ids.add(entity_id)

        print(f"Found {len(lookup_work_ids)} work IDs for direct lookup.")
        print(f"Found {len(discovery_entity_ids)} entities for discovery.")

        # This will hold the final list of WORK OBJECTS for the deep dive
        # We use a dict {work_id: work_object} to automatically handle duplicates
        works_to_process = {}

        # 2. Path A: Lookup Path (if work_... IDs were found)
        if lookup_work_ids:
            print("--- 2.A.1. Lookup Path: Getting work objects by ID ---")
            works_map = all_data.get('works', {})
            for work_id in lookup_work_ids:
                if work_id in works_map:
                    # Get the work object from our in-memory data
                    work_obj = works_map[work_id].copy()
                    work_obj['id'] = work_id # Ensure the ID is in the object for Step 2.B
                    works_to_process[work_id] = work_obj
                else:
                    print(f"Warning: Looked-up work_id '{work_id}' not found in works data.")

        # 3. Path B: Discovery Path (if >= 2 discovery entities were found)
        if len(discovery_entity_ids) >= 2:
            print("--- 2.A.2. Discovery Path: Finding connecting works ---")
            works_map = all_data.get('works', {})

            # Count how many new works we find
            new_discoveries = 0
            for work_id, work in works_map.items():
                # Avoid re-processing a work we already found via lookup
                if work_id in works_to_process:
                    continue

                related_ids = set()
                author_info = work.get('author')
                if isinstance(author_info, dict) and author_info.get('id'):
                    related_ids.add(author_info['id'])

                recipient_info = work.get('recipient')
                if isinstance(recipient_info, list):
                    for r in recipient_info:
                        if isinstance(r, dict) and r.get('id'):
                            related_ids.add(r['id'])

                # Check if all discovery entities are a subset of this work's relations
                if discovery_entity_ids.issubset(related_ids):
                    work_with_id = work.copy()
                    work_with_id['id'] = work_id
                    works_to_process[work_id] = work_with_id
                    new_discoveries += 1

            print(f"Discovery path found {new_discoveries} new connecting works.")

        elif discovery_entity_ids: # If 1 discovery entity was found
             print("--- 2.A.2. Discovery Path: Skipped (need >= 2 non-work entities to find connections) ---")
        else: # If 0 discovery entities were found
             print("--- 2.A.2. Discovery Path: No discovery entities provided. ---")

        # 4. Finalize the list of work objects for Step 2.B
        # The next block (Step 2.B) expects a list variable named `matching_works`
        matching_works = list(works_to_process.values())

        if matching_works:
            print(f"--- Total works to process in Deep Dive: {len(matching_works)} ---")

        # --- 2.B. Deep Dive Retrieval (IF works were found) ---
        if matching_works:
            print("\n--- 2.B. Deep Dive: Traversing Data Graph ---")
            works_info = []
            manifestation_texts = []
            annotation_details = {} # Use dict to group by manifestation

            # Build lookups
            expressions_map = all_data.get('expressions', {})
            manifestations_map = all_data.get('manifestations', {})
            annotations_list = all_data.get('annotations', [])

            work_to_exprs = {}
            for expr_id, expr in expressions_map.items():
                work_id = expr.get('work_id')
                if work_id:
                    if work_id not in work_to_exprs: work_to_exprs[work_id] = []
                    work_to_exprs[work_id].append(expr)

            for work in matching_works:
                work_id = work.get('id')
                work_title = work.get('title', 'Untitled')

                # Add to structured summary
                recipients_labels = [r.get('label') for r in work.get('recipient', []) if isinstance(r, dict)] if isinstance(work.get('recipient'), list) else []
                works_info.append(f"- Title: {work_title} (ID: {work_id}), From: {work.get('author', {}).get('label', 'Unknown')}, To: {', '.join(filter(None, recipients_labels)) or 'Unknown'}")

                print(f"  Traversing Work: {work_title} (ID: {work_id})")

                found_expressions = work_to_exprs.get(work_id, [])
                if not found_expressions:
                    print(f"    No expressions found for this work.")
                    continue

                for expr in found_expressions:
                    expr_id = expr.get('id', 'Unknown Expr')
                    manifestation_ids_to_process = []

                    # 1. Find Manifestations
                    canon_manif_id = expr.get('canonical_manifestation_id')
                    if canon_manif_id:
                        print(f"    Found Canonical Manifestation: {canon_manif_id}")
                        manifestation_ids_to_process.append(canon_manif_id)
                    else:
                        linked_manif_ids = [m.get('manifestation_id') for m in expr.get('manifestations', []) if m.get('manifestation_id')]
                        print(f"    No canonical. Found {len(linked_manif_ids)} linked manifestations: {linked_manif_ids}")
                        manifestation_ids_to_process.extend(linked_manif_ids)

                    if not manifestation_ids_to_process:
                        print(f"    No manifestations found for expression {expr_id}.")

                    # 2. Get Text and Annotations for each Manifestation
                    for manif_id in set(manifestation_ids_to_process): # Use set to avoid duplicates
                        if manif_id not in manifestations_map:
                            print(f"    Manifestation {manif_id} listed but not found in manifestations.json. Skipping.")
                            continue

                        manif = manifestations_map[manif_id]

                        # Get Text
                        if manif.get('text'):
                            print(f"      Got text for manifestation {manif_id}.")
                            manifestation_texts.append(f"Text for Work '{work_title}' (from manifestation {manif_id}):\n{manif.get('text')}")
                            sources.append({"content": "Full text loaded from metadata", "metadata": {"work_id": work_id, "expression_id": expr_id, "manifestation_id": manif_id}})
                        else:
                            print(f"      Manifestation {manif_id} found, but 'text' field is empty.")

                        # Get Annotations
                        if manif_id not in annotation_details: # Initialize list for this manif
                            annotation_details[manif_id] = []

                        found_annotations = [ann for ann in annotations_list if ann.get('target_manifestation') == manif_id]
                        print(f"      Found {len(found_annotations)} annotations for manifestation {manif_id}.")

                        for ann in found_annotations:
                            target_id = ann.get('body', {}).get('value')
                            if not target_id: continue

                            target_entity = get_entity_details(target_id, all_data)
                            if target_entity:
                                # 1. Determine the entity type from its ID string first.
                                entity_type = target_id.split('_')[0]
                                # 2. Use conditional logic to find the correct name field.
                                if entity_type == 'work':
                                    entity_name = target_entity.get('title', 'Unknown Entity')
                                else:
                                    entity_name = target_entity.get('preferred_label', {}).get('value', 'Unknown Entity')

                                annotation_details[manif_id].append(f"{entity_type.title()}: {entity_name} (ID: {target_id})")

            # --- 2.C. Assemble Contexts ---
            if works_info:
                structured_summary = f"Found {len(works_info)} works directly connecting the mentioned entities:\n" + "\n".join(works_info)

            if manifestation_texts:
                semantic_context = "\n\n" + "\n\n--- [End of Text] ---\n\n".join(manifestation_texts)

            if annotation_details:
                metadata_context_parts = ["\n\n--- Related Metadata from Annotations ---"]
                for manif_id, details in annotation_details.items():
                    if details: # Only add if there are details
                        metadata_context_parts.append(f"\nManifestation {manif_id} is related to:")
                        metadata_context_parts.extend([f"- {d}" for d in sorted(list(set(details)))])
                if len(metadata_context_parts) > 1: # Check if we actually added any details
                    metadata_context = "\n".join(metadata_context_parts)

            print("--- Deep Dive Complete ---")
            print(f"Structured Summary: {structured_summary}")
            print(f"Metadata Context: {metadata_context}")
            print(f"Semantic Context: Loaded {len(manifestation_texts)} texts.")

        # --- 3. Fallback: Vector Search (IF no structured results) ---
        if not matching_works:
            print("\n--- 3. Fallback: Running Standard Vector Search ---")
            docs = rag_components["retriever"].invoke(query)
            sources = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
            semantic_context = "\n\n".join([doc.page_content for doc in docs])
            if not docs:
                 print("--- Standard Vector Search found no documents. ---")
            else:
                 print(f"--- Standard Vector Search found {len(docs)} chunks. ---")

        # --- 4. Synthesis ---
        print("\n--- 4. Synthesizing Answer ---")
        prompt = f"""You are an expert research assistant. Your task is to answer the user's question based *ONLY* on the three sources of information provided below.

        **CRITICAL RULES:**
        1.  **DO NOT use any external or pre-existing knowledge.** Your entire answer must be derived *exclusively* from the "Structured Summary," "Metadata Context," and "Unstructured Text Context."
        2.  When stating a fact, date, or event, you *must* cite which source it came from (e.g., "According to the unstructured text...", "The metadata context links this to...").
        3.  If the provided context does not contain the answer, state that "Based on the provided documents, I cannot answer that question."
        4.  Do not invent dates or connections. If the context mentions an event but not its date, state that the date is unknown.

        ---
        Source 1: Structured Summary
        {structured_summary or "No direct matches found in metadata."}
        ---
        Source 2: Metadata Context
        {metadata_context or "No related metadata from annotations was found."}
        ---
        Source 3: Unstructured Text Context
        {semantic_context or "No relevant unstructured text was found."}
        ---

        Question: {query}

        Answer (derived *only* from the sources above):"""

        model = genai.GenerativeModel('models/gemini-pro-latest')
        response = model.generate_content(prompt)
        return jsonify({"status": "success", "answer": response.text, "sources": sources})

    except Exception as e:
        print(f"Error during query: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route('/debug-entity/<entity_id>')
def debug_entity(entity_id):
    if not rag_components.get("all_data"): return jsonify({"error": "Data not loaded."}), 404
    all_data = rag_components['all_data']
    entity_info = next((entity for entity_type in ['people', 'places', 'groups', 'subjects', 'events'] for entity_id_key, entity in all_data.get(entity_type, {}).items() if entity_id_key == entity_id), {})
    debug_report = find_related_manifestation_ids(entity_id, all_data, debug=True)
    debug_report["entity_details"] = entity_info
    return jsonify(debug_report)

@app.route('/debug-work/<work_id>')
def debug_work(work_id):
    if not rag_components.get("all_data"):
        return jsonify({"error": "Data not loaded. Please build the index first."}), 404
    work_data = rag_components.get("all_data", {}).get("works", {}).get(work_id)
    if not work_data:
        return jsonify({"error": f"Work with ID '{work_id}' not found in loaded data."}), 404
    return jsonify(work_data)

if __name__ == '__main__':
    app.run(debug=True, port=5001)
