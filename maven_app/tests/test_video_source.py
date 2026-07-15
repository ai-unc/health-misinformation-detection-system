"""
Tests for video_source.py — URL dispatch, download_audio, download_video,
fetch_metadata. Run from maven_app/:  python tests/test_video_source.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import video_source
from video_source import (
    Platform,
    download_audio,
    download_video,
    fetch_metadata,
    validate_url,
)


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_validate_url_dispatch():
    print('\n=== TEST 1: validate_url dispatch ===')

    url, platform = validate_url('  https://www.tiktok.com/@user/video/123456  ')
    assert url == 'https://www.tiktok.com/@user/video/123456'
    assert isinstance(platform, Platform)
    assert platform.name == 'tiktok'
    print('  ✓ TikTok URL returns stripped url + tiktok Platform')

    try:
        validate_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ unsupported URL raises ValueError')

    try:
        validate_url('')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ empty URL raises ValueError')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_download_audio(tmp_path):
    print('\n=== TEST 2: download_audio ===')

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        output_tpl = cmd[cmd.index('--output') + 1]
        out_dir = Path(output_tpl).parent
        (out_dir / 'fakevideo.mp3').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('video_source.subprocess.run', side_effect=fake_run), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        result = download_audio('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp3'
    assert result.exists()
    print('  ✓ success path returns mp3 Path')

    # Dependency-trap guard: impersonation must always be present
    cmd = captured['cmd']
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    assert '--extract-audio' in cmd
    print('  ✓ command includes --impersonate chrome and --extract-audio')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            download_audio('https://www.tiktok.com/@user/video/bad', str(tmp_path))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            download_audio('https://www.tiktok.com/@user/video/empty', str(empty_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no audio file produced' in str(e)
            print('  ✓ zero-exit but no mp3 raises RuntimeError')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_download_video(tmp_path):
    print('\n=== TEST 3: download_video ===')

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        output_tpl = cmd[cmd.index('--output') + 1]
        out_dir = Path(output_tpl).parent
        (out_dir / 'fakevideo.mp4').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('video_source.subprocess.run', side_effect=fake_run), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        result = download_video('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp4'
    assert result.exists()
    print('  ✓ success path returns mp4 Path')

    cmd = captured['cmd']
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    assert '-f' in cmd and cmd[cmd.index('-f') + 1] == 'mp4'
    assert '--extract-audio' not in cmd
    print('  ✓ command includes --impersonate chrome and -f mp4, no audio extraction')

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            download_video('https://www.tiktok.com/@user/video/empty', str(empty_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no video file produced' in str(e)
            print('  ✓ zero-exit but no mp4 raises RuntimeError')


# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_fetch_metadata():
    print('\n=== TEST 4: fetch_metadata ===')

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        return MagicMock(returncode=0, stderr='',
                         stdout='{"description": "Raspberry leaf tea!", "uploader": "healthmom"}')

    with patch('video_source.subprocess.run', side_effect=fake_run), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        meta = fetch_metadata('https://www.tiktok.com/@user/video/123')
    assert meta['description'] == 'Raspberry leaf tea!'
    assert meta['uploader'] == 'healthmom'
    print('  ✓ returns parsed metadata dict')

    cmd = captured['cmd']
    assert '--dump-json' in cmd and '--skip-download' in cmd
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    print('  ✓ command includes --dump-json --skip-download --impersonate chrome')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='', stdout='not json')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'invalid JSON' in str(e)
            print('  ✓ invalid JSON raises RuntimeError')

    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='blocked')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'blocked' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')

# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_instagram_block_translation():
    print('\n=== TEST 5: Instagram login-wall/rate-limit translation ===')

    from video_source import INSTAGRAM_BLOCK_MESSAGE

    block_stderrs = [
        'ERROR: [Instagram] C8abc: login required (use --cookies to provide account credentials)',
        'ERROR: [Instagram] C8abc: Instagram API is not granting access: rate-limit reached',
        'ERROR: [Instagram] C8abc: Restricted Video: You must be 18 years old or over',
        'ERROR: [Instagram] C8abc: Requested content is not available',
    ]
    for stderr in block_stderrs:
        with patch('video_source.subprocess.run',
                   return_value=MagicMock(returncode=1, stderr=stderr)), \
             patch.object(video_source, 'ensure_ffmpeg', return_value=''):
            try:
                fetch_metadata('https://www.instagram.com/reel/C8abc/')
                assert False, 'Expected RuntimeError'
            except RuntimeError as e:
                assert str(e) == INSTAGRAM_BLOCK_MESSAGE, f'unexpected: {e}'
    print('  ✓ all four block signatures translated to the friendly message')

    # Non-block failures keep stderr passthrough behavior
    with patch('video_source.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')), \
         patch.object(video_source, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.instagram.com/reel/C8abc/')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ other failures still pass stderr through')

# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    import tempfile as _tf
    import pathlib as _pl
    with _tf.TemporaryDirectory() as _td:
        _tmp = _pl.Path(_td)
        test_validate_url_dispatch()
        audio_dir = _tmp / 'audio'
        audio_dir.mkdir()
        test_download_audio(audio_dir)
        video_dir = _tmp / 'video'
        video_dir.mkdir()
        test_download_video(video_dir)
    test_fetch_metadata()
    test_instagram_block_translation()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
