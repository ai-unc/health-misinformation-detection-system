"""
Tests for tiktok.py — URL validation, download_audio, download_video, fetch_metadata.
Run from maven_app/:  python tests/test_tiktok.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import tiktok
from tiktok import download_audio, download_video, fetch_metadata, validate_url


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_validate_url():
    print('\n=== TEST 1: validate_url ===')

    try:
        validate_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'does not appear to be a TikTok link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')

    try:
        validate_url('')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ empty URL raises ValueError')

    assert validate_url('https://www.tiktok.com/@user/video/123456') == \
        'https://www.tiktok.com/@user/video/123456'
    print('  ✓ standard TikTok URL passes')

    assert validate_url('  https://vm.tiktok.com/ZMhAbcDef/  ') == \
        'https://vm.tiktok.com/ZMhAbcDef/'
    print('  ✓ vm.tiktok.com short URL passes and is stripped')


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

    with patch('tiktok.subprocess.run', side_effect=fake_run), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        result = download_audio('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp3'
    assert result.exists()
    print('  ✓ success path returns mp3 Path')

    # Dependency-trap guard: impersonation must always be present
    cmd = captured['cmd']
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    assert '--extract-audio' in cmd
    print('  ✓ command includes --impersonate chrome and --extract-audio')

    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        try:
            download_audio('https://www.tiktok.com/@user/video/bad', str(tmp_path))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')

    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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

    with patch('tiktok.subprocess.run', side_effect=fake_run), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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
    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
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

    with patch('tiktok.subprocess.run', side_effect=fake_run), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        meta = fetch_metadata('https://www.tiktok.com/@user/video/123')
    assert meta['description'] == 'Raspberry leaf tea!'
    assert meta['uploader'] == 'healthmom'
    print('  ✓ returns parsed metadata dict')

    cmd = captured['cmd']
    assert '--dump-json' in cmd and '--skip-download' in cmd
    assert '--impersonate' in cmd and cmd[cmd.index('--impersonate') + 1] == 'chrome'
    print('  ✓ command includes --dump-json --skip-download --impersonate chrome')

    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='', stdout='not json')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'invalid JSON' in str(e)
            print('  ✓ invalid JSON raises RuntimeError')

    with patch('tiktok.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='blocked')), \
         patch.object(tiktok, 'ensure_ffmpeg', return_value=''):
        try:
            fetch_metadata('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'blocked' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    import tempfile as _tf
    import pathlib as _pl
    with _tf.TemporaryDirectory() as _td:
        _tmp = _pl.Path(_td)
        test_validate_url()
        audio_dir = _tmp / 'audio'
        audio_dir.mkdir()
        test_download_audio(audio_dir)
        video_dir = _tmp / 'video'
        video_dir.mkdir()
        test_download_video(video_dir)
    test_fetch_metadata()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
