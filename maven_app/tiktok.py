"""
TikTok platform definition for MAVEN video ingestion.
Shared download/metadata plumbing lives in video_source.py.
"""
import re

from video_source import Platform

TIKTOK = Platform(
    name='tiktok',
    display_name='TikTok',
    url_re=re.compile(r'https?://([a-zA-Z0-9-]+\.)?tiktok\.com/'),
    slideshow_url_re=re.compile(r'https?://(www\.)?tiktok\.com/@[^/]+/photo/'),
    junk_terms=frozenset({'tiktok'}),
)
