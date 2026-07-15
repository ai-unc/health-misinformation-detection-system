"""
Instagram Reels platform definition for MAVEN video ingestion.
Shared download/metadata plumbing lives in video_source.py.
Public Reels only — login-walled content is surfaced as a friendly error
by video_source._run; there is no cookie/credential handling.
"""
import re

from video_source import Platform

INSTAGRAM = Platform(
    name='instagram',
    display_name='Instagram',
    # /reel/, /reels/, and /share/ redirect links only; /p/ photo posts and
    # /tv/ never reach yt-dlp — they fail validation with the standard error.
    url_re=re.compile(r'https?://(www\.)?instagram\.com/(reels?|share)/'),
    junk_terms=frozenset({'instagram', 'reels', 'reel'}),
)
