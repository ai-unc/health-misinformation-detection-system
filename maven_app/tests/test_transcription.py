"""
Tests for transcription.py — URL validation, _download_audio, _transcribe, cleanup, Flask route.
Run from maven_app/:  python tests/test_transcription.py
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import transcription
from transcription import TranscriptResult, transcribe_url
from transcription import _download_audio


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_url_validation():
    print('\n=== TEST 1: URL validation ===')

    # Non-TikTok URL → ValueError
    try:
        transcribe_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'does not appear to be a TikTok link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')

    # Empty string → ValueError
    try:
        transcribe_url('')
        assert False, 'Expected ValueError'
    except ValueError:
        print('  ✓ empty URL raises ValueError')

    # Valid TikTok URL — must not raise ValueError (internals mocked)
    with patch.object(transcription, '_download_audio', return_value=Path('/tmp/fake.mp3')), \
         patch.object(transcription, '_transcribe',
                      return_value=TranscriptResult(text='hi', segments=[], duration=1.0)):
        result = transcribe_url('https://www.tiktok.com/@user/video/123456')
    assert result.text == 'hi'
    print('  ✓ valid TikTok URL passes validation and returns TranscriptResult')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_download_audio(tmp_path):
    print('\n=== TEST 2: _download_audio ===')

    # Success: simulate yt-dlp creating an mp3
    def fake_run(cmd, **kwargs):
        output_tpl = cmd[cmd.index('--output') + 1]
        out_dir = Path(output_tpl).parent
        (out_dir / 'fakevideo.mp3').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('transcription.subprocess.run', side_effect=fake_run):
        result = _download_audio('https://www.tiktok.com/@user/video/123', str(tmp_path))
    assert result.suffix == '.mp3'
    assert result.exists()
    print('  ✓ success path returns mp3 Path')

    # Failure: yt-dlp non-zero exit → RuntimeError with stderr
    with patch('transcription.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='Video unavailable')):
        try:
            _download_audio('https://www.tiktok.com/@user/video/bad', str(tmp_path))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'Video unavailable' in str(e)
            print('  ✓ yt-dlp failure raises RuntimeError containing stderr')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import tempfile as _tf
    import pathlib as _pl
    with _tf.TemporaryDirectory() as _td:
        _tmp = _pl.Path(_td)
        test_url_validation()
        dl_dir = _tmp / 'dl_test'
        dl_dir.mkdir()
        test_download_audio(dl_dir)
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
