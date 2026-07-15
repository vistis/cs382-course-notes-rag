# RAG-Based AI Search System for Course Notes

This is a project for CS 382: Search Engine and Information Retrieval. It is a Retrieval-Augmented Generation based Search System for course notes, powered by a Large Language Model (using Google Gemini) to provide an overview of the results in a human-readable manner. LLM responses are based on the provided datasets only and cite facts from the documents within.

Built upon the Final Project Starter Pack provided by the instructor, which uses

- TXT loader only
- Naive word-count chunker
- TF-IDF based vectorization stored in memory (no data store)
- In-memory search powered by Cosine Similarity only for retrieval
- Extractive answer only; no LLM responses
- `Streamlit` as the interface

This project made the following changes:

- Upgraded the loader to also support PDF and MD (markdown) with metadata extraction
- Changed to a token-aware chunking strategy using OpenAI's TikToken
- Use SentenceTransformer with model `nomic-ai/nomic-embed-text-v1.5` as the embedder
- Use CrossEncoder with model `cross-encoder/ms-marco-MiniLM-L-6-v2` as the re-ranker
- Implemented bi-encoding + cross-encoding chunk retrieval
- Store vectors in an actual datastore using `ChromaDB` and use it to search for chunks against query
- Added LLM responses with `gemini-flash-lite` as the default with experimental local fallback using `llama.cpp` (model: `Phi-4-mini-instruct-Q4_K_M`)
- Graceful error handling such as falling back from LLM to basic extractive response when there is no internet or no API key was set
- Added dynamic dataset selection to simulate changing the course. On change, it will index the new dataset and switch to it. If the dataset is already built, it will simply switch to it, improving performance
- Added score threshold option when searching
- Added performance metrics in terms of latency/time-taken when loading and indexing dataset and generating response

The detailed architecture and pipeline is detailed below.

## Code Overview

The repository contains all the necessary Python scripts. Here is what each one does:

- `app.py`: is the Streamlit script. It provides the interface for the RAG system. It also checks server internet connection to determine if LLM with required API call (such as Google Gemini) is available to handle fallback, find dataset locations, download required models, etc.
- `rag/ingest.py`: handles loading documents and splitting them into chunks using the text splitter (TikToken).
- `rag/embed_store.py`: cached as a Streamlit resource to provide a persistent vector store across queries. It is a ChromaDB instance that store vectors of chunks loaded by `ingest.py`, embedded using SentenceTransformer, and handles switching dataset and chunk retrieval by querying the determined ChromaDB collection with an embedded query text. Results are re-ranked using `CrossEncoder`.
- `rag/generate.py`: generates the answer based on the retrieved chunks from `embed_store.py`. It has two modes - extractive, and LLM. Extractive return the chunks as they are as answer while LLM provide an overview response based on the retrieved chunks.

Models for `SentenceTransformer` and `llama.cpp` will be stored in the `model` folder and a dataset refers to a folder of documents inside the `data` folder.

## Architecture

![](architecture.webp)

There are two discrete flows: loading and indexing documents from dataset and searching + generating answer.

The architecture uses bi-encoder + cross-encoder. Bi-encoder quickly retrieve a large amount of matching chunks by embedding the query then use its vector to search for close chunk vectors, which are then re-ranked by the cross-encoder that scores by predicting the query against each chunk. This approach provides the speed of bi-encoder and accuracy of cross-encoder.

### Loading and Indexing

The loading and indexing of documents trigger only when the selected dataset (course in this case) have been been processed yet. `ingest.py` loads the documents from the selected dataset and use TikToken to split the loaded document texts into chunks. TikToken is loaded using LangChain, and was chosen because it is performant and preserve the text formatting.

A token-aware chunking strategy is used because the chunks will be sent to LLMs which operate on the basis of token window. A chunk size and overlap amount was tuned such that chunks provide enough context and can relate to each other (such as chunks of the same document) so that the LLM can understand better.

The chunks will then be forwarded to `embed_store.py` by `app.py` to be stored in the data store (ChromaDB). The chunks are first embedded using `nomic-ai/nomic-embed-text-v1.5` loaded via Sentence Transformers. The model allows for a larger context window compared to the tradition `all-MiniLM-L6-v2` while still being small enough, being run locally, and not too heavy.

The result is vectors that is then stored inside a vector store which is ChromaDB in this case. ChromaDB was chosen primarily for familiarity, but also because it has more development resources available. It is competent enough for this application.

Each dataset is stored in its own discrete collection. In case the selected dataset is already processed, the system will simply switch between the collections.

### Querying


User's question is passed to `embed_store.py` to be embedded into vectors which is then used to query the vector store. ChromaDB tries its best to find chunks which has its vector representation nearest to the vectors of the query.

From ChromaDB, it retrieves 5 times the number of chunks requested by top k as candidates. The candidates are then re-ranked using the cross encoder to predict the score and sort highest to lowest. Out all the candidates only the top k are returned back.

The retrieved chunks then sent to `generate.py` such that an actual answer can be displayed in the interface. The answer can be generated using external LLM (Google Gemini), local LLM (llama.cpp) or just extractive, depending on the search settings. If requested for external error, in the case of errors such as missing API keys or no server internet connection, it falls back to the local LLM. If the local LLM is not available, it falls back to extractive mode.

## Setup

### Using the Source Code

Requirements:

- Python (>=3.14)
- Internet Connection (for initial startup to download models)

It is recommended to take advantage of Python's virtual environment. Activate by running the following inside the project root directory:

```bash
python -m venv .venv
source .venv/bin/activate # For bash
```

Install the Python dependencies

```bash
# For GPU support; heavier
pip install -r requirements.txt

# CPU packages only; a lot lighter
pip install -r requirements-cpu.txt
```

The repository also provides sample datasets of course materials/notes of Paragon International University. They are stored in Github LFS. To get them, assuming the local source code copy comes from a `git clone`, make sure to have Git LFS installed on your system and run `git lfs pull`. Refer to other guides on how to use Git LFS.

Additional dataset can be provided as well. Simply categorize the documents (support only `.txt`, `.pdf`, and `.md`) into subfolders in the `data` folder. The subfolders should be 1 level deep only (i.e. `data/course/<course note files>`).

Make a copy of `.env.example` as `.env` and put your own Google Gemini API key.

Run the application by starting the Streamlit folder

```bash
streamlit run app.py
```

The command line interface (CLI) will provide a link to view the Streamlit interface (usually `localhost:8501`)

Optionally adjust Streamlit settings in `.streamlit/config.toml`. If there is any change, including changing `data` folder content, it is safer to restart the Streamlit server completely.

### Using the Docker Image

Requirements:

- Docker (Compose is optional)
- Internet Connection (for initial startup to download models)

To build a local image, in the source code root directory, run

```bash
docker build -t course-notes-rags:local .
```

Run the image with

```bash
docker run -p 8501:8501 --dns 9.9.9.9 -e PYTHONUNBUFFERED=1 -e GOOGLE_API_KEY=<your_gemini_api_key> -v /path/to/dataset:/app/data:ro,z -v course-notes-rag-model:/app/model:z course-notes-rags:local
```

or copy `docker-compose.yml.example` as `docker-compose.yml` and `.env.example` as `.env`. Then add your API key in `.env`. Edit `docker-compose.yml`, change the image of the web service to your local build tag (e.g. `course-notes-rags:local`) and run `docker compose up -d`. The data will be loaded from the `data` folder like normal.

Then follow the URL printed in the CLI (usually `localhost:8501`)

Pre-built images are also available. Replace the image tags with `ghcr.io/vistis/course-notes-rag:main`.

### Demo Server

> Demo server is temporary and might no longer be available.

A fully self-hosted demo server is also available, which was setup using the Docker Compose method. Visit [this link](https://course-notes-rag.entervise.net).

## Known Limitations

- Only support `.txt`, `.pdf`, and `.md`. Cannot read other common course notes format such as `.pptx` and `.docx`.
- Only extract text from documents. Course notes often contain images and the system cannot parse the text from images, thus introducing information gap. A good example is in the "AI: Evolution and Trends" document where the AI History section is completely encoded as images, thus the RAG will not return anything when asked about the history of AI.
- Local LLM is severely unfit. It is only an experimentation trying to generate answer locally, but it only takes 2048 tokens max, does not understand instruction well, and very prone to hallucination. It is kept as an experimental option and fallback for LLM answer mode.
- Text extraction is not aware of the text structure, such as when text is in table for example. It misses that context.
- Document metadata is extracted from the document properties which are often misleading.
- Once a dataset is processed, changing the actual folder content will not prompt a reload and reindex.
