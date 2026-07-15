"""
Tests for instagram.py — the thin Instagram platform definition.
Plumbing tests live in tests/test_video_source.py.
Run from maven_app/:  python tests/test_instagram.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from instagram import INSTAGRAM
from video_source import validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_platform_definition():
    print('\n=== TEST 1: INSTAGRAM platform definition ===')

    assert INSTAGRAM.name == 'instagram'
    assert INSTAGRAM.display_name == 'Instagram'
    assert INSTAGRAM.junk_terms == frozenset({'instagram', 'reels', 'reel'})
    print('  ✓ name, display_name, junk_terms as expected')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_url_matching():
    print('\n=== TEST 2: Instagram URL matching ===')

    assert INSTAGRAM.url_re.match('https://www.instagram.com/reel/C8abcDEfGhi/')
    print('  ✓ /reel/ URL matches')

    assert INSTAGRAM.url_re.match('https://www.instagram.com/reels/C8abcDEfGhi/')
    print('  ✓ /reels/ URL matches')

    assert INSTAGRAM.url_re.match('https://instagram.com/reel/C8abcDEfGhi/')
    print('  ✓ URL without www matches')

    assert INSTAGRAM.url_re.match('https://www.instagram.com/share/BAxyz123/')
    print('  ✓ /share/ redirect URL matches')

    assert not INSTAGRAM.url_re.match('https://www.instagram.com/p/C8abcDEfGhi/')
    print('  ✓ /p/ photo-post URL rejected')

    assert not INSTAGRAM.url_re.match('https://www.instagram.com/tv/C8abcDEfGhi/')
    print('  ✓ /tv/ IGTV URL rejected')

    assert not INSTAGRAM.url_re.match('https://www.instagram.com/some_user/')
    print('  ✓ profile URL rejected')

    assert not INSTAGRAM.url_re.match('https://www.tiktok.com/@user/video/123')
    print('  ✓ TikTok URL does not match Instagram')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_dispatch():
    print('\n=== TEST 3: validate_url dispatch + error message ===')

    url, platform = validate_url('  https://www.instagram.com/reel/C8abcDEfGhi/  ')
    assert url == 'https://www.instagram.com/reel/C8abcDEfGhi/'
    assert platform is INSTAGRAM
    print('  ✓ Reel URL stripped and dispatched to INSTAGRAM platform')

    try:
        validate_url('https://www.instagram.com/p/C8abcDEfGhi/')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert str(e) == 'URL is not a supported TikTok or Instagram Reels link.'
        print('  ✓ /p/ URL raises ValueError naming both platforms')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_platform_definition()
    test_url_matching()
    test_dispatch()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
