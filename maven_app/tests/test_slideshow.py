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
    NO_IMAGES_MESSAGE,
    download_slideshow,
    validate_url,
)
from slideshow import _description_from, _uploader_from, build_slide_segments
import instagram_embed


def _embed_page(media: dict) -> str:
    """Minified embed-page fixture mirroring the real page's JSON-in-string
    shape (verified 2026-07-18), including Instagram's \\/ slash escaping and
    a decoy contextJSON that must be skipped."""
    context = {'context': {'type': media.get('__typename')},
               'gql_data': {'shortcode_media': media}}
    inner = json.dumps(context, separators=(',', ':')).replace('/', '\\/')
    blob = '{"contextJSON":' + json.dumps(inner) + '}'
    return ('<html><head><script>{"contextJSON":"not-json"}</script></head>'
            '<body><script type="application/json">{"require":[[' + blob +
            ']]}</script></body></html>')


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

def test_instagram_anon_failure_without_cookies():
    print('\n=== TEST 3: anonymous failure + no cookies → friendly message, '
          'no gallery-dl call ===')

    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.EmbedUnavailableError('blocked')), \
             patch('video_source.subprocess.run') as mock_run:
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/',
                                   tempfile.mkdtemp())
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_COOKIE_MESSAGE
        mock_run.assert_not_called()
        print('  ✓ EmbedUnavailableError without cookies raises the friendly '
              'message and never invokes gallery-dl')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved


# ── TEST 3B ────────────────────────────────────────────────────────────────────

def test_instagram_anonymous_success_skips_gallery_dl():
    print('\n=== TEST 3B: anonymous success bypasses gallery-dl entirely ===')

    sentinel = ([Path('/x/embed/01.jpg')],
                {'description': 'cap', 'username': 'user'})
    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        with patch('instagram_embed.download_slideshow_anonymous',
                   return_value=sentinel) as mock_anon, \
             patch('video_source.subprocess.run') as mock_run:
            result = download_slideshow(
                'https://www.instagram.com/p/DRzdgElEf3N/', '/x')
        assert result == sentinel
        mock_anon.assert_called_once_with(
            'https://www.instagram.com/p/DRzdgElEf3N/', '/x')
        mock_run.assert_not_called()
        print('  ✓ anonymous result returned as-is; no cookies, no gallery-dl')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved


# ── TEST 3C ────────────────────────────────────────────────────────────────────

def test_instagram_video_post_no_fallback():
    print('\n=== TEST 3C: video post is terminal — no cookie fallback ===')

    tmp = tempfile.mkdtemp()
    cookie_file = Path(tmp) / 'cookies.txt'
    cookie_file.write_text('# Netscape HTTP Cookie File\n', encoding='utf-8')
    saved_env = os.environ.get('MAVEN_IG_COOKIES')
    os.environ['MAVEN_IG_COOKIES'] = str(cookie_file)
    try:
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.NotASlideshowError('video')), \
             patch('video_source.subprocess.run') as mock_run:
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == NO_IMAGES_MESSAGE
        mock_run.assert_not_called()
        print('  ✓ NotASlideshowError maps to the video-post message even '
              'with cookies configured')
    finally:
        if saved_env is None:
            os.environ.pop('MAVEN_IG_COOKIES', None)
        else:
            os.environ['MAVEN_IG_COOKIES'] = saved_env


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
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.EmbedUnavailableError('blocked')), \
             patch('video_source.subprocess.run', return_value=fake) as mock_run:
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
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.EmbedUnavailableError('blocked')), \
             patch('video_source.subprocess.run',
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
        with patch('instagram_embed.download_slideshow_anonymous',
                   side_effect=instagram_embed.EmbedUnavailableError('blocked')), \
             patch('video_source.subprocess.run',
                   return_value=MagicMock(returncode=0, stdout='', stderr='')):
            try:
                download_slideshow('https://www.instagram.com/p/DRzdgElEf3N/', tmp)
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == NO_IMAGES_MESSAGE
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


# ── TEST 10 ────────────────────────────────────────────────────────────────────

def test_embed_parse_carousel():
    print('\n=== TEST 10: embed page parse — carousel ===')

    media = {
        '__typename': 'GraphSidecar',
        'owner': {'username': 'healthaccount'},
        'edge_media_to_caption': {'edges': [{'node': {'text': 'the caption'}}]},
        'edge_sidecar_to_children': {'edges': [
            {'node': {'display_url': 'https://cdn.example/1.jpg'}},
            {'node': {'display_url': 'https://cdn.example/2.jpg'}},
            {'node': {'display_url': 'https://cdn.example/3.jpg'}},
        ]},
    }
    parsed = instagram_embed._parse_shortcode_media(_embed_page(media))
    assert parsed['__typename'] == 'GraphSidecar'
    assert parsed['owner']['username'] == 'healthaccount'
    print('  ✓ shortcode_media recovered through double-encoded contextJSON')

    urls = instagram_embed._slide_urls(parsed)
    assert urls == ['https://cdn.example/1.jpg', 'https://cdn.example/2.jpg',
                    'https://cdn.example/3.jpg']
    print('  ✓ slide URLs in carousel order with \\/ escapes decoded')


# ── TEST 11 ────────────────────────────────────────────────────────────────────

def test_embed_parse_edge_shapes():
    print('\n=== TEST 11: embed parse — single image, video post, missing data ===')

    single = {'__typename': 'GraphImage', 'owner': {'username': 'u'},
              'edge_media_to_caption': {'edges': []},
              'display_url': 'https://cdn.example/only.jpg'}
    urls = instagram_embed._slide_urls(
        instagram_embed._parse_shortcode_media(_embed_page(single)))
    assert urls == ['https://cdn.example/only.jpg']
    print('  ✓ single GraphImage post yields its one display_url')

    video = {'__typename': 'GraphVideo', 'owner': {'username': 'u'},
             'display_url': 'https://cdn.example/poster.jpg'}
    try:
        instagram_embed._slide_urls(video)
        assert False, 'Expected NotASlideshowError'
    except instagram_embed.NotASlideshowError:
        print('  ✓ GraphVideo post raises NotASlideshowError')

    try:
        instagram_embed._parse_shortcode_media('<html><body>no data</body></html>')
        assert False, 'Expected EmbedUnavailableError'
    except instagram_embed.EmbedUnavailableError:
        print('  ✓ page without contextJSON raises EmbedUnavailableError')


# ── TEST 12 ────────────────────────────────────────────────────────────────────

def test_download_slideshow_anonymous():
    print('\n=== TEST 12: download_slideshow_anonymous writes ordered slides ===')

    media = {
        '__typename': 'GraphSidecar',
        'owner': {'username': 'healthaccount'},
        'edge_media_to_caption': {'edges': [{'node': {'text': 'the caption'}}]},
        'edge_sidecar_to_children': {'edges': [
            {'node': {'display_url': 'https://cdn.example/1.jpg'}},
            {'node': {'display_url': 'https://cdn.example/2.jpg'}},
            {'node': {'display_url': 'https://cdn.example/3.jpg'}},
        ]},
    }
    html = _embed_page(media)
    tmp = tempfile.mkdtemp()

    def fake_download(url, dest):
        dest.write_bytes(url.encode())

    with patch('instagram_embed._fetch_embed_page',
               return_value=html) as mock_fetch, \
         patch('instagram_embed._download_image', side_effect=fake_download):
        images, meta = instagram_embed.download_slideshow_anonymous(
            'https://www.instagram.com/p/ABC123xyz_-/', tmp)

    mock_fetch.assert_called_once_with('ABC123xyz_-')
    print('  ✓ shortcode extracted from URL')
    assert [p.name for p in images] == ['01.jpg', '02.jpg', '03.jpg']
    assert all(p.parent.name == 'embed' for p in images)
    print('  ✓ slides written to tmp_dir/embed/ as zero-padded NN.jpg in order')
    assert [p.read_bytes().decode() for p in images] == [
        'https://cdn.example/1.jpg', 'https://cdn.example/2.jpg',
        'https://cdn.example/3.jpg']
    print('  ✓ each slide downloaded from its carousel-ordered display_url')
    assert meta == {'description': 'the caption', 'username': 'healthaccount'}
    print('  ✓ metadata uses the Instagram schema slideshow.py already reads')


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
    print('\n=== LIVE TEST: Instagram slideshow (anonymous) ===')

    from text_extraction import extract_text_url

    # Unset cookies for the call to prove the anonymous path works alone.
    saved = os.environ.pop('MAVEN_IG_COOKIES', None)
    try:
        result = extract_text_url('https://www.instagram.com/p/DRzdgElEf3N/')
    finally:
        if saved is not None:
            os.environ['MAVEN_IG_COOKIES'] = saved

    assert result.description, 'expected a caption'
    print(f'  ✓ caption extracted anonymously ({len(result.description)} chars)')
    assert result.overlay_segments, 'expected OCR text from slides'
    slides = [seg['slide'] for seg in result.overlay_segments]
    assert slides == sorted(slides) and slides[0] >= 1
    print(f'  ✓ {len(result.overlay_segments)} slide segments in order: {slides}')
    assert result.text.startswith(result.description[:20])
    print('  ✓ assembled text begins with caption')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_slideshow_url_detection()
    test_slideshow_urls_validate()
    test_instagram_anon_failure_without_cookies()
    test_instagram_anonymous_success_skips_gallery_dl()
    test_instagram_video_post_no_fallback()
    test_login_redirect_maps_to_cookie_message()
    test_download_slideshow_filters_and_orders()
    test_download_slideshow_orders_by_sidecar_num()
    test_download_slideshow_empty_images_friendly_error()
    test_build_slide_segments()
    test_metadata_normalization()
    test_text_mode_dispatches_to_slideshow()
    test_audio_mode_rejects_slideshows()
    test_embed_parse_carousel()
    test_embed_parse_edge_shapes()
    test_download_slideshow_anonymous()
    if '--live' in _sys.argv:
        test_live_tiktok_slideshow()
        test_live_instagram_slideshow()
    else:
        print('\n(live network tests skipped — pass --live to run them)')
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
