"""Streamlit interface for KernelGPT's local RAG assistant."""
from __future__ import annotations

import streamlit as st
from ollama import Client

from config import CHAT_MODEL, EMBEDDING_MODEL, OLLAMA_HOST, VISION_MODEL
from ingest import get_collection, indexed_documents, ingest_pdf
from knowledge_graph import build_knowledge_graph, graphviz_source
from multimedia import ingest_media
from chat_history import add_message, create_chat, delete_chat, list_chats, load_messages
from rag import answer_question

st.set_page_config(page_title="KernelGPT", page_icon="🧠", layout="wide")
st.title("KernelGPT")
st.caption("Knowledge-Based Operating Systems Assistant")

if "active_chat_id" not in st.session_state:
    st.session_state.active_chat_id = create_chat()
if "messages" not in st.session_state:
    st.session_state.messages = load_messages(st.session_state.active_chat_id)

collection = get_collection()
with st.sidebar:
    st.header("Chat history")
    if st.button("＋ New chat", use_container_width=True):
        st.session_state.active_chat_id = create_chat()
        st.session_state.messages = []
        st.rerun()
    saved_chats = list_chats()
    if saved_chats:
        chat_ids = [row[0] for row in saved_chats]
        labels = {row[0]: f"{row[1]} · {row[2][:10]}" for row in saved_chats}
        current = st.session_state.active_chat_id
        selected = st.selectbox("Saved conversations", chat_ids,
            index=chat_ids.index(current) if current in chat_ids else 0,
            format_func=lambda chat_id: labels[chat_id], label_visibility="collapsed")
        if selected != st.session_state.active_chat_id:
            st.session_state.active_chat_id = selected
            st.session_state.messages = load_messages(selected)
            st.rerun()
        if st.button("Delete this conversation", use_container_width=True):
            delete_chat(selected)
            remaining = list_chats()
            st.session_state.active_chat_id = remaining[0][0] if remaining else create_chat()
            st.session_state.messages = load_messages(st.session_state.active_chat_id)
            st.rerun()
    st.divider()
    st.header("Knowledge base")
    try:
        available_models = {item.model for item in Client(host=OLLAMA_HOST).list().models}
        available_models.update(name.removesuffix(":latest") for name in list(available_models))
        missing = [model for model in (CHAT_MODEL, EMBEDDING_MODEL) if model not in available_models]
        if missing:
            st.warning("Ollama is running, but these models are missing: " + ", ".join(missing))
        else:
            st.success("Ollama is ready")
    except Exception as exc:
        st.error("Ollama is not responding. Start the Ollama app and try again.")
        st.caption(str(exc)[:250])

    st.caption(f"{collection.count()} indexed passage(s)")
    documents = indexed_documents(collection)
    if documents:
        st.subheader("Indexed documents")
        for document in documents:
            st.write(f"📄 {document['source']} · {document['chunks']} chunks")
    else:
        st.info("Upload an Operating Systems PDF to get started.")

    uploads = st.file_uploader(
        "Add PDF documents", type=["pdf"], accept_multiple_files=True,
        help="Text-based PDFs up to the configured size limit are supported. Scanned PDFs need OCR.",
    )
    if st.button("Index uploaded PDFs", type="primary", disabled=not uploads):
        progress = st.progress(0)
        for index, uploaded in enumerate(uploads, start=1):
            try:
                with st.spinner(f"Indexing {uploaded.name}..."):
                    count, empty_pages = ingest_pdf(uploaded, filename=uploaded.name, collection=collection)
                st.success(f"{uploaded.name}: indexed {count} passage(s).")
                if empty_pages:
                    st.warning(f"Pages without extractable text (possibly scanned): {empty_pages}")
            except Exception as exc:
                st.error(f"Could not index {uploaded.name}: {exc}")
            progress.progress(index / len(uploads))
        st.rerun()

    media_uploads = st.file_uploader(
        "Add photos or videos", type=["png", "jpg", "jpeg", "webp", "bmp", "mp4", "mov", "mkv", "avi", "webm", "m4v"],
        accept_multiple_files=True,
        help=f"Photos and sampled video frames are described by the local vision model {VISION_MODEL} and indexed with timestamps.",
        key="media_uploader",
    )
    if st.button("Index photos and videos", disabled=not media_uploads):
        for uploaded in media_uploads:
            try:
                with st.spinner(f"Analyzing {uploaded.name} with {VISION_MODEL}..."):
                    count = ingest_media(uploaded, filename=uploaded.name, collection=collection)
                st.success(f"{uploaded.name}: indexed {count} visual description(s).")
            except Exception as exc:
                st.error(f"Could not index {uploaded.name}: {exc}")
        st.rerun()

    if st.button("Clear chat"):
        delete_chat(st.session_state.active_chat_id)
        st.session_state.active_chat_id = create_chat()
        st.session_state.messages = []
        st.rerun()

chat_tab, graph_tab = st.tabs(["Chat", "Knowledge Graph"])

with chat_tab:
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask a question about your Operating Systems notes")
    if question:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": question})
        add_message(st.session_state.active_chat_id, "user", question)
        with st.chat_message("user"):
            st.markdown(question)
        with st.chat_message("assistant"):
            with st.spinner("Searching your notes and generating an answer..."):
                try:
                    result = answer_question(question, history=history, collection=collection)
                    st.markdown(result.answer)
                    st.session_state.messages.append({"role": "assistant", "content": result.answer})
                    add_message(st.session_state.active_chat_id, "assistant", result.answer)
                except Exception as exc:
                    message = f"I couldn't complete that request: {exc}"
                    st.error(message)
                    st.session_state.messages.append({"role": "assistant", "content": message})
                    add_message(st.session_state.active_chat_id, "assistant", message)

with graph_tab:
    st.subheader("Explore your indexed knowledge")
    st.caption("Concepts are detected locally from indexed passage text. Document links show where each concept appears; concept links indicate terms found together in a passage.")
    if collection.count() == 0:
        st.info("Index a PDF first. Its concepts and source links will appear here.")
    else:
        graph = build_knowledge_graph(collection)
        concepts = sorted(graph["concept_counts"], key=lambda term: term.lower())
        if not concepts:
            st.info("No known Operating Systems concepts were detected in the indexed passages yet.")
        else:
            focus_label = st.selectbox("Focus the graph on a concept", ["All concepts", *[term.title() for term in concepts]])
            focus = None if focus_label == "All concepts" else next(term for term in concepts if term.title() == focus_label)
            visible_nodes = graph["nodes"]
            visible_edges = graph["edges"]
            if focus:
                focus_id = f"concept:{focus}"
                keep = {focus_id}
                for edge in visible_edges:
                    if edge["source"] == focus_id or edge["target"] == focus_id:
                        keep.update((edge["source"], edge["target"]))
                visible_nodes = [node for node in visible_nodes if node["id"] in keep]
                visible_edges = [edge for edge in visible_edges if edge["source"] in keep and edge["target"] in keep]
            node_by_id = {node["id"]: node for node in visible_nodes}
            try:
                from streamlit_agraph import agraph, Config, Edge, Node
                a_nodes = [Node(id=node["id"], label=node["label"],
                    title=node["detail"], size=27 if node["kind"] == "concept" else 20,
                    color="#dcefe4" if node["kind"] == "concept" else "#dce8fb",
                    shape="dot" if node["kind"] == "concept" else "box") for node in visible_nodes]
                a_edges = [Edge(source=edge["source"], target=edge["target"],
                    label=edge["label"], color="#95a3b3", font={"size": 10}) for edge in visible_edges]
                config = Config(width="100%", height=650, directed=False, physics=True,
                    hierarchical=False, nodeHighlightBehavior=True, highlightColor="#f3b562",
                    collapsible=False)
                selected = agraph(nodes=a_nodes, edges=a_edges, config=config)
                st.caption("Drag nodes to arrange the graph; scroll to zoom; click a node to inspect it.")
                st.caption(f"{len(visible_nodes)} visible nodes · {len(visible_edges)} visible links · {graph['passages']} indexed passages")
                if selected in node_by_id:
                    node = node_by_id[selected]
                    st.markdown(f"**Selected {node['kind']}:** {node['label']}")
                    st.caption(node["detail"])
                    connected = [edge for edge in visible_edges if edge["source"] == selected or edge["target"] == selected]
                    if connected:
                        st.markdown("**Connections**")
                        for edge in connected[:30]:
                            other_id = edge["target"] if edge["source"] == selected else edge["source"]
                            st.write(f"{node_by_id[other_id]['label']} - {edge['label']}")
            except ImportError:
                st.warning("Install streamlit-agraph from requirements.txt and restart to enable clicking and dragging.")
                st.graphviz_chart(graphviz_source(graph, focus), use_container_width=True)
