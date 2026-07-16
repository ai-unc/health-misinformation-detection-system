"""
Instagram platform definition for MAVEN video ingestion.
Shared download/metadata plumbing lives in video_source.py.
Public Reels work anonymously; /p/ slideshow posts additionally require
login cookies via the MAVEN_IG_COOKIES env var (see video_source.py).
"""
import re

from video_source import Platform

INSTAGRAM = Platform(
    name='instagram',
    display_name='Instagram',
    # /reel/, /reels/, and /share/ video links plus /p/ slideshow posts;
    # /tv/ never reaches yt-dlp — it fails validation with the standard error.
    url_re=re.compile(r'https?://(www\.)?instagram\.com/(reels?|share|p)/'),
    slideshow_url_re=re.compile(r'https?://(www\.)?instagram\.com/p/'),
    junk_terms=frozenset({'instagram', 'reels', 'reel'}),
)
