"""
MAVEN anonymous Instagram slideshow ingestion: fetches a /p/ post's caption
and slide images through Instagram's public embed endpoint
(/p/<shortcode>/embed/captioned/) using curl_cffi chrome TLS impersonation —
no login cookies needed for public posts. Transport and parse only: all
user-facing message text lives in video_source.py, which falls back to the
gallery-dl + MAVEN_IG_COOKIES path when EmbedUnavailableError is raised.
Data path verified live 2026-07-18 on 12- and 7-slide public carousels.
"""
import json
import re
from pathlib import Path
from typing import List, Tuple

from curl_cffi import requests as cffi_requests

_EMBED_URL = 'https://www.instagram.com/p/{shortcode}/embed/captioned/'
_SHORTCODE_RE = re.compile(r'instagram\.com/p/([A-Za-z0-9_-]+)')
# contextJSON's value is a JSON-encoded string; capture it with its escape
# sequences intact, then json.loads twice (unescape, then decode).
_CONTEXT_JSON_RE = re.compile(r'"contextJSON"\s*:\s*"((?:\\.|[^"\\])*)"')
_TIMEOUT_SECONDS = 30


class EmbedUnavailableError(RuntimeError):
    """Anonymous embed path can't serve this post (blocked, private, removed,
    markup changed, or a network/download failure). Caller may fall back to
    the cookie-authenticated gallery-dl path."""


class NotASlideshowError(RuntimeError):
    """The post is a GraphVideo — definitively not a slideshow; no fallback
    can change that."""


def _fetch_embed_page(shortcode: str) -> str:
    try:
        resp = cffi_requests.get(_EMBED_URL.format(shortcode=shortcode),
                                 impersonate='chrome',
                                 timeout=_TIMEOUT_SECONDS)
    except Exception as e:
        raise EmbedUnavailableError(f'embed page fetch failed: {e}')
    if resp.status_code != 200:
        raise EmbedUnavailableError(f'embed page HTTP {resp.status_code}')
    return resp.text


def _download_image(url: str, dest: Path) -> None:
    try:
        resp = cffi_requests.get(url, impersonate='chrome',
                                 timeout=_TIMEOUT_SECONDS)
    except Exception as e:
        raise EmbedUnavailableError(f'slide download failed: {e}')
    if resp.status_code != 200 or not resp.content:
        raise EmbedUnavailableError(f'slide download HTTP {resp.status_code}')
    dest.write_bytes(resp.content)


def _parse_shortcode_media(html: str) -> dict:
    """gql_data.shortcode_media from the first parseable contextJSON blob."""
    for match in _CONTEXT_JSON_RE.finditer(html):
        try:
            context = json.loads(json.loads(f'"{match.group(1)}"'))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue
        if not isinstance(context, dict):
            continue
        media = (context.get('gql_data') or {}).get('shortcode_media')
        if media:
            return media
    raise EmbedUnavailableError('no shortcode_media in embed page')


def _slide_urls(media: dict) -> List[str]:
    """display_urls in carousel order. GraphVideo children of a mixed carousel
    contribute their poster frame; a plain GraphVideo post raises
    NotASlideshowError."""
    if media.get('__typename') == 'GraphVideo':
        raise NotASlideshowError('post is a video')
    edges = (media.get('edge_sidecar_to_children') or {}).get('edges') or []
    urls = [(e.get('node') or {}).get('display_url') for e in edges]
    urls = [u for u in urls if u]
    if not urls and media.get('display_url'):
        urls = [media['display_url']]
    if not urls:
        raise EmbedUnavailableError('no slide image URLs in embed data')
    return urls


def download_slideshow_anonymous(url: str, tmp_dir: str) -> Tuple[List[Path], dict]:
    """Anonymous counterpart of video_source.download_slideshow for Instagram
    /p/ posts: returns (slide image paths in carousel order, metadata dict
    with the 'description'/'username' keys slideshow.py already normalizes).
    Slides land in tmp_dir/embed/ so a failed attempt's partial files stay
    out of the gallery-dl fallback's root-level glob of tmp_dir."""
    m = _SHORTCODE_RE.search(url)
    if not m:
        raise EmbedUnavailableError(f'no /p/ shortcode in URL: {url}')
    media = _parse_shortcode_media(_fetch_embed_page(m.group(1)))
    urls = _slide_urls(media)

    caption_edges = (media.get('edge_media_to_caption') or {}).get('edges') or []
    caption = ((caption_edges[0].get('node') or {}).get('text') or ''
               if caption_edges else '')
    owner = (media.get('owner') or {}).get('username') or ''

    embed_dir = Path(tmp_dir) / 'embed'
    embed_dir.mkdir(exist_ok=True)
    images = []
    for idx, slide_url in enumerate(urls, start=1):
        dest = embed_dir / f'{idx:02d}.jpg'
        _download_image(slide_url, dest)
        images.append(dest)
    return images, {'description': caption, 'username': owner}
