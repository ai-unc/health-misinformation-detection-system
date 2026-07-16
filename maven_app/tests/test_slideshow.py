"""
Tests for slideshow.py and the slideshow plumbing in video_source.py —
TikTok /photo/ and Instagram /p/ slideshow-post text extraction.
Run from maven_app/:  python tests/test_slideshow.py
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_source
from tiktok import TIKTOK
from instagram import INSTAGRAM
from video_source import (
    INSTAGRAM_COOKIE_MESSAGE,
    download_slideshow,
    validate_url,
)
from slideshow import _description_from, _uploader_from, build_slide_segments


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_slideshow_url_detection():
    print('\n=== TEST 1: slideshow URL detection ===')

    assert TIKTOK.is_slideshow('https://www.tiktok.com/@glowingwithgracee/photo/7618717295417756941')
    print('  ✓ TikTok /photo/ URL detected as slideshow')

    assert not TIKTOK.is_slideshow('https://www.tiktok.com/@user/video/123')
    print('  ✓ TikTok /video/ URL not a slideshow')

    assert INSTAGRAM.is_slideshow('https://www.instagram.com/p/DRzdgElEf3N/')
    assert INSTAGRAM.is_slideshow('https://instagram.com/p/DRzdgElEf3N/')
    print('  ✓ Instagram /p/ URLs detected as slideshow (with and without www)')

    assert not INSTAGRAM.is_slideshow('https://www.instagram.com/reel/C8abcDEfGhi/')
    assert not INSTAGRAM.is_slideshow('https://www.instagram.com/share/BAxyz123/')
    print('  ✓ Instagram /reel/ and /share/ URLs not slideshows')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_slideshow_urls_validate():
    print('\n=== TEST 2: slideshow URLs pass validate_url ===')

    url, platform = validate_url('  https://www.instagram.com/p/DRzdgElEf3N/  ')
    assert url == 'https://www.instagram.com/p/DRzdgElEf3N/'
    assert platform is INSTAGRAM
    print('  ✓ Instagram /p/ URL validates and dispatches to INSTAGRAM')

    url, platform = validate_url('https://www.tiktok.com/@drtosinofficial/photo/7648626696140229910')
    assert platform is TIKTOK
    print('  ✓ TikTok /photo/ URL validates and dispatches to TIKTOK')

    try:
        validate_url('https://www.instagram.com/tv/C8abcDEfGhi/')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert str(e) == 'URL is not a supported TikTok or Instagram link.'
        print('  ✓ unsupported URL raises updated error message')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_instagram_slideshow_requires_cookies():
    print('\n=== TEST 3: Instagram slideshow without cookies fails fast ===')

    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tempfile.mkdtemp())
        assert False, 'Expected RuntimeError'
    except RuntimeError as e:
        assert str(e) == INSTAGRAM_COOKIE_MESSAGE
        print('  ✓ missing MAVEN_IG_COOKIES raises the friendly cookie message')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved


# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_login_redirect_maps_to_cookie_message():
    print('\n=== TEST 4: gallery-dl login redirect maps to cookie message ===')

    tmp = tempfile.mkdtemp()
    cookie_file = Path(tmp) / 'cookies.txt'
    cookie_file.write_text('# Netscape HTTP Cookie File\n', encoding='utf-8')

    saved_env = os.environ.get('MAVEN_IG_COOKIES')
    os.environ['MAVEN_IG_COOKIES'] = str(cookie_file)
    fake = MagicMock(returncode=4, stdout='',
                     stderr='[instagram][error] HTTP redirect to login page '
                            '(https://www.instagram.com/accounts/login/)')
    try:
        with patch('video_source.subprocess.run', return_value=fake):
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_COOKIE_MESSAGE
                print('  ✓ stale/rejected cookies map to the friendly cookie message')
    finally:
        if saved_env is None:
            os.environ.pop('MAVEN_IG_COOKIES', None)
        else:
            os.environ['MAVEN_IG_COOKIES'] = saved_env


# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_download_slideshow_filters_and_orders():
    print('\n=== TEST 5: download_slideshow filters non-images, orders slides, reads sidecar ===')

    tmp = tempfile.mkdtemp()
    # Simulate gallery-dl output layout (verified 2026-07-15): numbered jpgs,
    # one mp3 soundtrack, one .json sidecar per file.
    names = ['777_01 caption [aa].jpg', '777_02 caption [bb].jpg',
             '777_10 caption [cc].jpg', '777 caption [dd].mp3']
    for n in names:
        (Path(tmp) / n).write_bytes(b'x')
        (Path(tmp) / (n + '.json')).write_text(
            json.dumps({'desc': 'the caption', 'author': {'uniqueId': 'someuser'}}),
            encoding='utf-8')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stdout='', stderr='')):
        images, meta = download_slideshow('https://www.tiktok.com/@u/photo/777', tmp)

    assert [p.name for p in images] == ['777_01 caption [aa].jpg',
                                        '777_02 caption [bb].jpg',
                                        '777_10 caption [cc].jpg']
    print('  ✓ mp3 and .json sidecars excluded; slides in carousel order')
    assert meta['desc'] == 'the caption'
    assert meta['author']['uniqueId'] == 'someuser'
    print('  ✓ metadata read from first image sidecar')


# ── TEST 6 ─────────────────────────────────────────────────────────────────────

def test_build_slide_segments():
    print('\n=== TEST 6: build_slide_segments ===')

    frame_results = [
        {'ts': 0, 'lines': [('First slide claim', 0.9), ('subtitle', 0.8)]},
        {'ts': 1, 'lines': []},
        {'ts': 2, 'lines': [('Third slide claim', 0.95)]},
    ]
    segments = build_slide_segments(frame_results)
    assert segments == [
        {'slide': 1, 'text': 'First slide claim subtitle'},
        {'slide': 3, 'text': 'Third slide claim'},
    ]
    print('  ✓ one segment per non-empty slide, 1-based numbering, blank slides skipped')

    assert build_slide_segments([]) == []
    assert build_slide_segments([{'ts': 0, 'lines': []}]) == []
    print('  ✓ empty and all-blank inputs produce no segments')


# ── TEST 7 ─────────────────────────────────────────────────────────────────────

def test_metadata_normalization():
    print('\n=== TEST 7: gallery-dl metadata normalization ===')

    tiktok_meta = {'desc': 'TikTok caption', 'author': {'uniqueId': 'ttuser'}}
    assert _description_from(tiktok_meta) == 'TikTok caption'
    assert _uploader_from(tiktok_meta) == 'ttuser'
    print('  ✓ TikTok schema: desc + author.uniqueId')

    ig_meta = {'description': 'IG caption', 'username': 'iguser'}
    assert _description_from(ig_meta) == 'IG caption'
    assert _uploader_from(ig_meta) == 'iguser'
    print('  ✓ Instagram schema: description + username')

    assert _description_from({}) == ''
    assert _uploader_from({}) == ''
    print('  ✓ missing fields degrade to empty strings')


# ── TEST 8 ─────────────────────────────────────────────────────────────────────

def test_text_mode_dispatches_to_slideshow():
    print('\n=== TEST 8: extract_text_url dispatches slideshow URLs ===')

    import slideshow
    import text_extraction

    calls = []
    saved = slideshow.extract_slideshow_text
    slideshow.extract_slideshow_text = lambda url, platform: (
        calls.append((url, platform.name)) or 'SENTINEL')
    try:
        result = text_extraction.extract_text_url(
            'https://www.tiktok.com/@u/photo/777')
    finally:
        slideshow.extract_slideshow_text = saved

    assert result == 'SENTINEL'
    assert calls == [('https://www.tiktok.com/@u/photo/777', 'tiktok')]
    print('  ✓ slideshow URL routed to extract_slideshow_text, video path untouched')


# ── TEST 9 ─────────────────────────────────────────────────────────────────────

def test_audio_mode_rejects_slideshows():
    print('\n=== TEST 9: transcribe_url rejects slideshow URLs ===')

    from transcription import transcribe_url

    for url in ('https://www.tiktok.com/@u/photo/777',
                'https://www.instagram.com/p/DRzdgElEf3N/'):
        try:
            transcribe_url(url)
            assert False, 'Expected ValueError'
        except ValueError as e:
            assert str(e) == 'Slideshow posts are supported in Text mode only.'
    print('  ✓ both platforms rejected in audio mode with the friendly message')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_slideshow_url_detection()
    test_slideshow_urls_validate()
    test_instagram_slideshow_requires_cookies()
    test_login_redirect_maps_to_cookie_message()
    test_download_slideshow_filters_and_orders()
    test_build_slide_segments()
    test_metadata_normalization()
    test_text_mode_dispatches_to_slideshow()
    test_audio_mode_rejects_slideshows()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
