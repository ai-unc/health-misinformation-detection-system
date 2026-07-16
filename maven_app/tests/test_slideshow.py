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
        with patch('video_source.subprocess.run', return_value=fake) as mock_run:
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_COOKIE_MESSAGE
                print('  ✓ stale/rejected cookies map to the friendly cookie message')

        cmd = mock_run.call_args.args[0]
        assert '-D' in cmd and cmd[cmd.index('-D') + 1] == tmp
        assert '--write-metadata' in cmd
        assert '--config-ignore' in cmd
        assert cmd[-1] == 'https://www.instagram.com/p/DRzdgElEf3N/'
        assert '--cookies' in cmd and cmd[cmd.index('--cookies') + 1] == str(cookie_file)
        print('  ✓ command includes -D tmp_dir, --write-metadata, --config-ignore, '
              '--cookies; URL is last argument')
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
    # one mp3 soundtrack, one .json sidecar per file. None of these sidecars
    # carry a 'num' field, so ordering falls back to filename sort.
    names = ['777_01 caption [aa].jpg', '777_02 caption [bb].jpg',
             '777_10 caption [cc].jpg', '777 caption [dd].mp3']
    for n in names:
        (Path(tmp) / n).write_bytes(b'x')
        (Path(tmp) / (n + '.json')).write_text(
            json.dumps({'desc': 'the caption', 'author': {'uniqueId': 'someuser'}}),
            encoding='utf-8')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stdout='', stderr='')) as mock_run:
        images, meta = download_slideshow('https://www.tiktok.com/@u/photo/777', tmp)

    assert [p.name for p in images] == ['777_01 caption [aa].jpg',
                                        '777_02 caption [bb].jpg',
                                        '777_10 caption [cc].jpg']
    print('  ✓ mp3 and .json sidecars excluded; slides in filename fallback order')
    assert meta['desc'] == 'the caption'
    assert meta['author']['uniqueId'] == 'someuser'
    print('  ✓ metadata read from first image sidecar')

    cmd = mock_run.call_args.args[0]
    assert '-D' in cmd and cmd[cmd.index('-D') + 1] == tmp
    assert '--write-metadata' in cmd
    assert '--config-ignore' in cmd
    assert cmd[-1] == 'https://www.tiktok.com/@u/photo/777'
    assert '--cookies' not in cmd  # TikTok: no cookie flag expected
    print('  ✓ command includes -D tmp_dir, --write-metadata, --config-ignore; '
          'URL is last argument')


# ── TEST 5B ────────────────────────────────────────────────────────────────────

def test_download_slideshow_orders_by_sidecar_num():
    print('\n=== TEST 5B: download_slideshow orders by sidecar num, not filename ===')

    tmp = tempfile.mkdtemp()
    # Instagram-style media-id filenames: lexicographic order disagrees with
    # true carousel order, which only the sidecar 'num' field carries.
    files = [
        ('3324422500_c.jpg', 3),   # sorts first lexicographically, but is slide 3
        ('3324421000_a.jpg', 1),   # sorts second lexicographically, is slide 1
        ('3324421999_b.jpg', 2),   # sorts third lexicographically, is slide 2
        ('3324429999_d.jpg', None),  # no 'num' in sidecar: falls back to filename order
    ]
    for name, num in files:
        (Path(tmp) / name).write_bytes(b'x')
        sidecar = {'desc': 'ig caption'}
        if num is not None:
            sidecar['num'] = num
        (Path(tmp) / (name + '.json')).write_text(json.dumps(sidecar), encoding='utf-8')

    cookie_file = Path(tmp) / 'cookies.txt'
    cookie_file.write_text('# Netscape HTTP Cookie File\n', encoding='utf-8')
    saved_env = os.environ.get('MAVEN_IG_COOKIES')
    os.environ['MAVEN_IG_COOKIES'] = str(cookie_file)
    try:
        with patch('video_source.subprocess.run',
                   return_value=MagicMock(returncode=0, stdout='', stderr='')):
            images, _ = download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
    finally:
        if saved_env is None:
            os.environ.pop('MAVEN_IG_COOKIES', None)
        else:
            os.environ['MAVEN_IG_COOKIES'] = saved_env

    assert [p.name for p in images] == [
        '3324421000_a.jpg',  # num=1
        '3324421999_b.jpg',  # num=2
        '3324422500_c.jpg',  # num=3
        '3324429999_d.jpg',  # no num: falls back after all valid-num images
    ]
    print('  ✓ sidecar num order wins over lexicographic filename order; '
          'missing-num image falls back to the end')


# ── TEST 5C ────────────────────────────────────────────────────────────────────

def test_download_slideshow_empty_images_friendly_error():
    print('\n=== TEST 5C: download_slideshow friendly error when only a video is produced ===')

    tmp = tempfile.mkdtemp()
    # A video-only /p/ post: gallery-dl produces an mp4 (filtered out by the
    # image extension allowlist) plus its metadata sidecar, no images.
    (Path(tmp) / '777.mp4').write_bytes(b'x')
    (Path(tmp) / '777.mp4.json').write_text(json.dumps({'desc': 'a video post'}),
                                             encoding='utf-8')

    cookie_file = Path(tmp) / 'cookies.txt'
    cookie_file.write_text('# Netscape HTTP Cookie File\n', encoding='utf-8')
    saved_env = os.environ.get('MAVEN_IG_COOKIES')
    os.environ['MAVEN_IG_COOKIES'] = str(cookie_file)
    try:
        with patch('video_source.subprocess.run',
                   return_value=MagicMock(returncode=0, stdout='', stderr='')):
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == ('No images found in this post — it may be a '
                                  'video post rather than a slideshow.')
                print('  ✓ video-only post raises the friendly no-images message')
    finally:
        if saved_env is None:
            os.environ.pop('MAVEN_IG_COOKIES', None)
        else:
            os.environ['MAVEN_IG_COOKIES'] = saved_env


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


# ── LIVE TESTS (network) ───────────────────────────────────────────────────────

def test_live_tiktok_slideshow():
    print('\n=== LIVE TEST: TikTok slideshow end-to-end ===')

    from text_extraction import extract_text_url

    result = extract_text_url(
        'https://www.tiktok.com/@glowingwithgracee/photo/7618717295417756941')
    assert result.description, 'expected a caption'
    print(f'  ✓ caption extracted ({len(result.description)} chars)')
    assert result.overlay_segments, 'expected OCR text from slides'
    slides = [seg['slide'] for seg in result.overlay_segments]
    assert slides == sorted(slides) and slides[0] >= 1
    print(f'  ✓ {len(result.overlay_segments)} slide segments in order: {slides}')
    assert result.text.startswith(result.description[:20])
    print('  ✓ assembled text begins with caption')
    preview = result.overlay_segments[0]['text'][:80]
    print(f'  slide 1 preview: {preview!r}')


def test_live_instagram_slideshow():
    print('\n=== LIVE TEST: Instagram slideshow ===')

    import video_source
    from text_extraction import extract_text_url
    from video_source import INSTAGRAM_COOKIE_MESSAGE

    url = 'https://www.instagram.com/p/DRzdgElEf3N/'
    if not video_source._instagram_cookies():
        try:
            extract_text_url(url)
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert str(e) == INSTAGRAM_COOKIE_MESSAGE
        print('  ~ MAVEN_IG_COOKIES not set — verified friendly cookie error; '
              'full extraction SKIPPED')
        return

    result = extract_text_url(url)
    assert result.description, 'expected a caption'
    assert result.overlay_segments, 'expected OCR text from slides'
    print(f'  ✓ caption + {len(result.overlay_segments)} slide segments extracted')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_slideshow_url_detection()
    test_slideshow_urls_validate()
    test_instagram_slideshow_requires_cookies()
    test_login_redirect_maps_to_cookie_message()
    test_download_slideshow_filters_and_orders()
    test_download_slideshow_orders_by_sidecar_num()
    test_download_slideshow_empty_images_friendly_error()
    test_build_slide_segments()
    test_metadata_normalization()
    test_text_mode_dispatches_to_slideshow()
    test_audio_mode_rejects_slideshows()
    if '--live' in _sys.argv:
        test_live_tiktok_slideshow()
        test_live_instagram_slideshow()
    else:
        print('\n(live network tests skipped — pass --live to run them)')
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
