# KernelGPT

Knowledge-Based Operating Systems Assistant. KernelGPT uses Streamlit, pypdf, ChromaDB, and **local Ollama models** for both embeddings and answer generation. It does not need an OpenAI API key or API credits.

## Architecture

**Indexing:** PDF -> page-by-page text extraction -> cleanup and overlapping chunks -> local Ollama embeddings -> persistent ChromaDB records with source and page metadata.

**Question answering:** question -> local Ollama embedding -> relevant ChromaDB passages -> local Ollama chat model -> answer with document and page citations.

Retrieved passages are evidence, not instructions. If the indexed notes do not support an answer, KernelGPT should say so. Scanned PDFs need OCR; the current version reports pages with no extractable text.

## Requirements

- Windows 10 or later and Python 3.11 or 3.12.
- Ollama for Windows, installed from https://ollama.com/download/windows.
- Enough free disk space for local models. The default embedding model is about 622 MB and the default chat model about 2.5 GB; the models also need working memory while running.

Ollama's `embeddinggemma` model requires Ollama 0.11.10 or later. Model sizes and availability may change; see https://ollama.com/library/embeddinggemma and https://ollama.com/library/qwen3.

## Windows setup

1. Install and start Ollama for Windows. Its local service should be available at `http://localhost:11434`.
2. In PowerShell, go to the project folder and download the two models:

```powershell
cd C:\Users\rites\Documents\Codex\KERNELGPT
ollama pull embeddinggemma
ollama pull qwen3:4b
ollama list
```

3. Install the Python dependencies in the project's virtual environment:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

If PowerShell blocks activation, run `Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass` in that terminal, then activate again. If `.venv` has not been created yet, create it with `py -3.12 -m venv .venv` first (or use Python 3.11).

4. Set the local model options. If `.env` does not exist, create it from the template; do not overwrite an existing `.env`:

```powershell
if (-not (Test-Path .env)) { Copy-Item .env.example .env }
notepad .env
```

The defaults work with Ollama running on this computer. Change `OLLAMA_HOST`, `CHAT_MODEL`, or `EMBEDDING_MODEL` in `.env` only if you use a different address or model. No API key is needed.

## Current implementation

- `config.py` loads Ollama, chunking, retrieval, and storage settings.
- `ingest.py` extracts page-aware chunks, creates local embeddings, and saves them in ChromaDB.
- `rag.py` retrieves relevant passages and generates an answer with a local chat model and verified source metadata.
- `data/` is for PDFs; `chroma_db/` holds the persistent vector database.

The Streamlit interface (`app.py`) is a later milestone. To ingest a PDF from Python after Ollama and both models are ready:

```powershell
python -c "from ingest import ingest_pdf; print(ingest_pdf('data/your_notes.pdf'))"
```

## Settings

| Variable | Default | Purpose |
|---|---|---|
| `OLLAMA_HOST` | `http://localhost:11434` | Local Ollama service address |
| `CHAT_MODEL` | `qwen3:4b` | Local answer model |
| `EMBEDDING_MODEL` | `embeddinggemma` | Local semantic search model |
| `CHUNK_SIZE` | `1000` | Maximum chunk size in characters |
| `CHUNK_OVERLAP` | `150` | Overlap between neighboring chunks |
| `TOP_K` | `4` | Maximum passages considered per question |
| `MAX_COSINE_DISTANCE` | `0.55` | Retrieval relevance cutoff |
| `MAX_PDF_SIZE_MB` | `50` | Maximum PDF size |

## Troubleshooting

- **ollama is not recognized:** install Ollama for Windows, then open a new PowerShell window.
- **Connection refused at localhost:11434:** start the Ollama app and check `ollama list`.
- **Model not found:** run `ollama pull embeddinggemma` and `ollama pull qwen3:4b`.
- **No text extracted:** the PDF may be scanned and require OCR, or it may be encrypted/corrupt.
- **Slow responses or out-of-memory errors:** try a smaller chat model such as `qwen3:1.7b` and set `CHAT_MODEL=qwen3:1.7b` in `.env`.
- **No relevant passages:** confirm the PDF was indexed with the same embedding model currently configured. Changing embedding models requires re-indexing documents because their vector dimensions and spaces may differ.

## Next step

Install Ollama and pull the two models above. Then index `data/1.pdf`, verify local retrieval and citations, and continue to the Streamlit interface milestone.

### If PowerShell blocks virtual environment activation

You can skip activation entirely and call the virtual environment's Python directly:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run app.py
```
## Knowledge graph

Open the **Knowledge Graph** tab in the Streamlit app to explore concepts detected in indexed Operating Systems passages. Document nodes link to concepts found in their passages; concept-to-concept links indicate terms that appeared together in a passage. Use the focus selector to inspect one concept and its connected sources. The graph is derived locally from the ChromaDB collection using a small, transparent Operating Systems vocabulary; it does not require a hosted graph service or LLM extraction.

The graph is interactive: drag nodes to rearrange them, scroll to zoom, and click a node to show its description and connections. Install the updated dependencies with `python -m pip install -r requirements.txt`, then restart Streamlit.

## Chat history and response speed

Conversations are saved locally in `chat_history.sqlite3`. Select a saved conversation in the sidebar to reopen it, start a new chat, or delete the current conversation. Model requests keep Ollama models loaded for ten minutes between turns. Retrieval defaults to three passages and generation is capped at 384 new tokens to reduce latency; adjust `TOP_K` or the generation limit in `rag.py` if you prefer longer, more detailed answers.
# Multimodal input

KernelGPT can index photos and videos alongside PDFs. Install dependencies, pull a local vision model, and set `VISION_MODEL` in `.env` (default `qwen3-vl:2b`). Current Ollama Qwen3-VL builds require Ollama 0.12.7 or later. Photo descriptions are stored as searchable passages; videos are sampled every five seconds (up to 30 frames), with timestamps retained in citations. Video decoding uses the `imageio-ffmpeg` dependency.

```powershell
ollama pull qwen3-vl:2b
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use **Add photos or videos** in the sidebar, then **Index photos and videos**. Supported files include PNG/JPG/WEBP/BMP images and MP4/MOV/MKV/AVI/WEBM/M4V videos. The vision model must be installed locally before media indexing.
