"""Index photos and sampled video frames as searchable visual descriptions."""
from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from pathlib import Path
from typing import BinaryIO

from config import MAX_PDF_SIZE_MB, OLLAMA_HOST, VISION_MODEL
from ingest import TextChunk, clean_text, index_chunks

IMAGE_TYPES = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_TYPES = {".mp4", ".mov", ".mkv", ".avi", ".webm", ".m4v"}
MAX_MEDIA_MB = 200
VIDEO_SAMPLE_SECONDS = 5
MAX_VIDEO_FRAMES = 30


def _describe(image: bytes, *, filename: str, location: str, client=None) -> str:
    if client is None:
        from ollama import Client
        client = Client(host=OLLAMA_HOST)
    response = client.chat(
        model=VISION_MODEL,
        messages=[{"role": "user", "content": (
            "Describe the visible content for a searchable Operating Systems study knowledge base. "
            "Transcribe readable text, labels, diagram components, and their relationships. "
            "Separate visible evidence from uncertainty; do not invent unreadable details. "
            f"This media is {filename}, {location}."
        ), "images": [image]}],
        options={"temperature": 0.1},
    )
    return clean_text(response.message.content or "")


def _make_chunk(text: str, source: str, location: str) -> TextChunk:
    document_id = hashlib.sha256((source + "\n" + text).encode("utf-8")).hexdigest()[:20]
    chunk_id = hashlib.sha256((document_id + "\n" + location).encode("utf-8")).hexdigest()
    return TextChunk(text=text, source=source, page=0, chunk_id=chunk_id,
                     document_id=document_id, location=location)


def _uploaded_bytes(file_obj: BinaryIO, filename: str) -> bytes:
    if not filename or Path(filename).suffix.lower() not in IMAGE_TYPES | VIDEO_TYPES:
        raise ValueError("Supported media: PNG, JPG, JPEG, WEBP, BMP, MP4, MOV, MKV, AVI, WEBM, and M4V.")
    if hasattr(file_obj, "seek"):
        file_obj.seek(0)
    data = file_obj.read()
    if len(data) > MAX_MEDIA_MB * 1024 * 1024:
        raise ValueError(f"Media exceeds the {MAX_MEDIA_MB} MB size limit.")
    return data


def _video_frames(data: bytes) -> list[tuple[str, bytes]]:
    """Sample one JPEG frame every few seconds using imageio-ffmpeg's binary."""
    try:
        import imageio_ffmpeg
    except ImportError as exc:
        raise RuntimeError("Video support needs imageio-ffmpeg. Install project requirements, then restart KernelGPT.") from exc
    with tempfile.TemporaryDirectory(prefix="kernelgpt-video-") as folder:
        video_path = Path(folder) / "input-video"
        video_path.write_bytes(data)
        command = [imageio_ffmpeg.get_ffmpeg_exe(), "-hide_banner", "-loglevel", "error",
                   "-i", str(video_path), "-vf", f"fps=1/{VIDEO_SAMPLE_SECONDS},scale=768:-1",
                   "-frames:v", str(MAX_VIDEO_FRAMES), "-f", "image2pipe", "-vcodec", "mjpeg", "pipe:1"]
        result = subprocess.run(command, capture_output=True, check=False)
        if result.returncode:
            raise ValueError("Could not decode this video. Check that it is a supported, playable video file.")
        stream = result.stdout
        starts = [m.start() for m in re.finditer(b"\xff\xd8", stream)]
        frames = []
        for i, start in enumerate(starts):
            end_marker = stream.find(b"\xff\xd9", start)
            if end_marker < 0:
                continue
            frames.append((f"timestamp {len(frames) * VIDEO_SAMPLE_SECONDS}s", stream[start:end_marker + 2]))
        return frames


def ingest_media(file_obj: BinaryIO, *, filename: str, collection=None, embedding_fn=None, client=None) -> int:
    """Describe and index an image or sampled video frames; return passage count."""
    data = _uploaded_bytes(file_obj, filename)
    suffix = Path(filename).suffix.lower()
    frames = [("image", data)] if suffix in IMAGE_TYPES else _video_frames(data)
    if not frames:
        raise ValueError("No frames could be extracted from this video.")
    chunks = []
    for location, frame in frames:
        description = _describe(frame, filename=filename, location=location, client=client)
        if description:
            chunks.append(_make_chunk(f"Visual description of {filename} ({location}):\n{description}", filename, location))
    if not chunks:
        raise ValueError("The vision model returned no usable descriptions.")
    return index_chunks(chunks, collection=collection, embedding_fn=embedding_fn)
