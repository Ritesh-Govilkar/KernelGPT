"""Build a lightweight, inspectable concept graph from indexed RAG passages."""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from typing import Any

# A transparent, domain vocabulary keeps graph extraction local and explainable.
# Terms are deliberately OS-focused; the graph describes concepts mentioned in
# retrieved/indexed text and does not claim to be an ontology or a fact checker.
CONCEPTS = [
    "operating system", "process", "thread", "scheduling", "cpu scheduling",
    "context switch", "process synchronization", "critical section", "semaphore",
    "mutex", "deadlock", "memory management", "virtual memory", "paging",
    "page replacement", "segmentation", "page table", "address translation",
    "file system", "directory", "file allocation", "disk scheduling", "disk",
    "input output", "device management", "system call", "kernel", "user mode",
    "kernel mode", "interrupt", "system architecture", "multiprogramming",
    "multithreading", "concurrency", "parallelism", "protection", "security",
    "distributed system", "real time system", "linux", "windows", "unix",
    "first come first served", "shortest job first", "round robin", "priority scheduling",
    "bankers algorithm", "thrashing", "page fault", "cache", "buffering",
]


def _mentions(text: str) -> set[str]:
    normalized = re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()
    padded = f" {normalized} "
    return {
        concept for concept in CONCEPTS
        if f" {re.sub(r'[^a-z0-9]+', ' ', concept).strip()} " in padded
    }


def build_knowledge_graph(collection: Any, *, max_concepts: int = 36) -> dict[str, Any]:
    """Derive document/concept and co-mentioned concept edges from Chroma."""
    records = collection.get(include=["documents", "metadatas"])
    docs = records.get("documents") or []
    metas = records.get("metadatas") or []
    document_counts: Counter[str] = Counter()
    concept_counts: Counter[str] = Counter()
    document_concepts: dict[str, set[str]] = defaultdict(set)
    concept_pages: dict[str, dict[str, set[int]]] = defaultdict(lambda: defaultdict(set))

    for passage, meta in zip(docs, metas):
        if not passage or not meta:
            continue
        source = str(meta.get("source", "unknown document"))
        try:
            page = int(meta.get("page", 0))
        except (TypeError, ValueError):
            page = 0
        document_counts[source] += 1
        found = _mentions(str(passage))
        for concept in found:
            concept_counts[concept] += 1
            document_concepts[source].add(concept)
            if page:
                concept_pages[concept][source].add(page)

    selected = {name for name, _ in concept_counts.most_common(max_concepts)}
    nodes: list[dict[str, str]] = []
    edges: list[dict[str, str]] = []
    for source, count in sorted(document_counts.items()):
        nodes.append({"id": f"doc:{source}", "label": source, "kind": "document", "detail": f"{count} indexed passages"})
    for concept in sorted(selected):
        nodes.append({"id": f"concept:{concept}", "label": concept.title(), "kind": "concept", "detail": f"Mentioned in {concept_counts[concept]} passage(s)"})
        for source, concepts in document_concepts.items():
            if concept in concepts:
                pages = sorted(concept_pages[concept][source])
                page_text = ", ".join(map(str, pages[:8])) + ("…" if len(pages) > 8 else "")
                edges.append({"source": f"doc:{source}", "target": f"concept:{concept}", "label": f"pp. {page_text}" if page_text else "mentions"})

    # Concepts are related when they occur in the same indexed passage.
    pair_counts: Counter[tuple[str, str]] = Counter()
    for passage in docs:
        found = sorted(_mentions(str(passage)) & selected) if passage else []
        for i, left in enumerate(found):
            for right in found[i + 1:]:
                pair_counts[(left, right)] += 1
    for (left, right), count in pair_counts.most_common(80):
        if count >= 1:
            edges.append({"source": f"concept:{left}", "target": f"concept:{right}", "label": f"co-mentioned {count}×"})
    return {"nodes": nodes, "edges": edges, "passages": len(docs), "concept_counts": concept_counts}


def graphviz_source(graph: dict[str, Any], focus: str | None = None) -> str:
    """Render a stable Graphviz DOT view; focus can restrict to one concept."""
    nodes = graph["nodes"]
    edges = graph["edges"]
    if focus:
        focus_id = f"concept:{focus}"
        keep = {focus_id}
        for edge in edges:
            if edge["source"] == focus_id or edge["target"] == focus_id:
                keep.update((edge["source"], edge["target"]))
        # Include connected document nodes and their selected concept neighbors.
        for edge in edges:
            if edge["source"] in keep or edge["target"] in keep:
                keep.update((edge["source"], edge["target"]))
        nodes = [node for node in nodes if node["id"] in keep]
        edges = [edge for edge in edges if edge["source"] in keep and edge["target"] in keep]

    def quote(value: str) -> str:
        return value.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")

    lines = ["graph KernelGPT {", '  graph [overlap=false, splines=true, bgcolor="white", ranksep="1.0", nodesep="0.35"];',
             '  node [fontname="Arial", fontsize=10, style=filled, color="#455468", penwidth=1.2];',
             '  edge [fontname="Arial", fontsize=8, color="#9aa7b5", fontcolor="#596579"];']
    for node in nodes:
        if node["kind"] == "document":
            lines.append(f'  "{quote(node["id"])}" [label="{quote(node["label"])}\\n{quote(node["detail"])}", shape=box, fillcolor="#e8f0fe"];')
        else:
            lines.append(f'  "{quote(node["id"])}" [label="{quote(node["label"])}\\n{quote(node["detail"])}", shape=ellipse, fillcolor="#e9f7ef"];')
    for edge in edges:
        lines.append(f'  "{quote(edge["source"])}" -- "{quote(edge["target"])}" [label="{quote(edge["label"])}"];')
    lines.append("}")
    return "\n".join(lines)
