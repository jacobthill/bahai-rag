# bahai-rag

A research tool for asking questions of the Bahá'í corpus and its annotations. It answers from the entities, works, manifestations, and annotations in [bahai-data](https://github.com/jacobthill/bahai-data), and cites its sources.

This is a deliberately simple, single-file Flask app meant for iterating on research questions, not a production system. The design it works toward (entity resolution, graph traversal, data tiers) is described in [docs/graphrag.md](docs/graphrag.md).

## Setup

The app expects bahai-data checked out as a sibling folder:

```
bahai-github/
├── bahai-data/   # data source (read-only for this app)
└── bahai-rag/    # this repo
```

1. Create the environment and install dependencies:
   ```bash
   uv venv --python 3.12
   uv pip install -r requirements.txt
   ```
2. Copy `.env.example` to `.env` and add your keys. `.env` is git-ignored.
   - `GEMINI_API_KEY`: required; used to generate answers.
   - `OPENAI_API_KEY`: only used by the evaluation scripts.

## Running

```bash
.venv/bin/python app.py
```

Then open http://127.0.0.1:5005.

On the first run the app builds a vector index of all manifestation texts in `chroma_db_rag/`. This is slow and the folder is large (about 1.8 GB). Later runs reuse it. The index is not updated when bahai-data changes; delete `chroma_db_rag/` to rebuild it.

## Asking questions

Name entities by their bahai-data IDs in the question, for example:

> Tell me about the correspondence between person_bahaullah and person_varqa.

IDs are recognized for `person_`, `place_`, `group_`, `event_`, `subject_`, `work_`, and `circular_instruction_`. Names alone are not resolved to IDs yet.

For each ID, the app:

1. Finds every annotation tagging that entity and the manifestation it points to.
2. Takes a window of text (`WINDOW_CHARS`, 1,500 characters on each side) around each mention, rather than the whole manifestation.
3. Sends the excerpts, the related works, and the annotation list to Gemini (`GEMINI_MODEL`), which answers with quotes and citations.

If the question contains no IDs that match annotations, the app falls back to semantic search over the vector index.

## Data conventions

All data is read from `../bahai-data/data`. Two optional fields improve citations:

- `page` on an annotation, e.g. `"151"`, `"43–4"`, `"214n47"`
- `citation` on a manifestation, e.g. `"Shahvar et al., The Baha'is of Iran, Transcaspia and the Caucasus, vol. 1"`

When both are present, sources are cited as "Shahvar et al., vol. 1, p. 151" instead of by manifestation ID.

Copyrighted texts can be kept in `../bahai-data/data/copyright/`. Files there are loaded and merged with the main data files of the same name. The folder is optional.

## Evaluation

`evaluation/` contains scripts that score answers with [RAGAS](https://docs.ragas.io/) (faithfulness and answer relevancy), comparing this system with NotebookLM. The scored answer data in `evaluation/data/` quotes the corpus and is not published in this repo.

## History

`app_1.py` through `app_7.py` are earlier versions, removed from the working tree but kept in git history (commit `02dfa25`). `app_1.py` includes features not carried forward, such as finding works by author and recipient.
