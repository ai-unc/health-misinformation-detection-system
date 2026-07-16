"""
Tests for slideshow.py and the slideshow plumbing in video_source.py —
TikTok /photo/ and Instagram /p/ slideshow-post text extraction.
Run from maven_app/:  python tests/test_slideshow.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tiktok import TIKTOK
from instagram import INSTAGRAM
from video_source import validate_url


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


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_slideshow_url_detection()
    test_slideshow_urls_validate()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
