"""
Tests for tiktok.py — the thin TikTok platform definition.
Plumbing tests live in tests/test_video_source.py.
Run from maven_app/:  python tests/test_tiktok.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiktok import TIKTOK
from video_source import validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_platform_definition():
    print('\n=== TEST 1: TIKTOK platform definition ===')

    assert TIKTOK.name == 'tiktok'
    assert TIKTOK.display_name == 'TikTok'
    assert 'tiktok' in TIKTOK.junk_terms
    print('  ✓ name, display_name, junk_terms as expected')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_url_matching():
    print('\n=== TEST 2: TikTok URL matching ===')

    assert TIKTOK.url_re.match('https://www.tiktok.com/@user/video/123456')
    print('  ✓ standard TikTok URL matches')

    assert TIKTOK.url_re.match('https://vm.tiktok.com/ZMhAbcDef/')
    print('  ✓ vm.tiktok.com short URL matches')

    assert not TIKTOK.url_re.match('https://www.youtube.com/watch?v=abc123')
    print('  ✓ non-TikTok URL does not match')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_dispatch():
    print('\n=== TEST 3: validate_url dispatches to TIKTOK ===')

    url, platform = validate_url('  https://vm.tiktok.com/ZMhAbcDef/  ')
    assert url == 'https://vm.tiktok.com/ZMhAbcDef/'
    assert platform is TIKTOK
    print('  ✓ TikTok URL stripped and dispatched to TIKTOK platform')


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
