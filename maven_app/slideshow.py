"""
MAVEN Slideshow Text Extraction: OCRs every slide of a TikTok /photo/ or
Instagram /p/ slideshow post and combines the results with the post caption.
Dispatched from text_extraction.extract_text_url for slideshow URLs; reuses
its OCR, junk-filter, and text-assembly helpers.
Public entry point: extract_slideshow_text(url, platform) → TextExtractionResult.
"""
import shutil
import tempfile
from typing import List

from text_extraction import (NoTextFoundError, TextExtractionResult,
                             _assemble_text, _ocr_frames)
from video_source import Platform, download_slideshow


def _description_from(metadata: dict) -> str:
    """Caption across gallery-dl schemas: TikTok 'desc', Instagram 'description'."""
    return (metadata.get('desc') or metadata.get('description') or '').strip()


def _uploader_from(metadata: dict) -> str:
    """Uploader handle across gallery-dl schemas: TikTok author.uniqueId,
    Instagram username."""
    author = metadata.get('author') or {}
    return (author.get('uniqueId') or metadata.get('username')
            or metadata.get('uploader') or '').strip()


def build_slide_segments(frame_results: List[dict]) -> List[dict]:
    """One segment per slide that produced OCR text; slide numbers are the
    1-based carousel positions. Unlike video frames there is no cross-frame
    merging — each slide is independent content.

    frame_results: [{'ts': idx, 'lines': [(text, confidence), ...]}, ...]
    Returns [{'slide': int, 'text': str}, ...].
    """
    segments = []
    for frame in frame_results:
        lines = frame.get('lines') or []
        if not lines:
            continue
        segments.append({'slide': frame['ts'] + 1,
                         'text': ' '.join(t for t, _ in lines)})
    return segments


def extract_slideshow_text(url: str, platform: Platform) -> TextExtractionResult:
    tmp_dir = tempfile.mkdtemp()
    try:
        images, metadata = download_slideshow(url, tmp_dir)
        description = _description_from(metadata)
        uploader = _uploader_from(metadata)
        # Slides are OCR'd at native resolution — they are text-dense by
        # design, unlike sampled video frames.
        frame_results = _ocr_frames(images, uploader, platform.junk_terms)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    slide_segments = build_slide_segments(frame_results)
    text = _assemble_text(description, slide_segments)
    if not text:
        raise NoTextFoundError('No slide text or caption found in slideshow.')
    return TextExtractionResult(
        description=description,
        overlay_segments=slide_segments,
        text=text,
    )
