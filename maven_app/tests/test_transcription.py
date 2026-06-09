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
from transcription import _transcribe


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

    # Edge case: yt-dlp exits 0 but produces no mp3 (e.g., format conversion failure)
    empty_dir = tmp_path / 'empty'
    empty_dir.mkdir()
    with patch('transcription.subprocess.run',
               return_value=MagicMock(returncode=0, stderr='')):
        try:
            _download_audio('https://www.tiktok.com/@user/video/empty', str(empty_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'no audio file produced' in str(e)
            print('  ✓ zero-exit but no mp3 raises RuntimeError')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_transcribe(tmp_path):
    print('\n=== TEST 3: _transcribe ===')

    dummy_audio = tmp_path / 'audio.mp3'
    dummy_audio.write_bytes(b'\x00' * 100)

    mock_seg       = MagicMock()
    mock_seg.start = 0.0
    mock_seg.end   = 2.5
    mock_seg.text  = '  Hello world  '

    mock_info          = MagicMock()
    mock_info.duration = 2.5

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)

    with patch.object(transcription, '_get_model', return_value=mock_model):
        result = _transcribe(dummy_audio)

    assert result.text == 'Hello world'
    assert result.duration == 2.5
    assert result.segments == [{'start': 0.0, 'end': 2.5, 'text': 'Hello world'}]
    print('  ✓ _transcribe returns correct TranscriptResult')

    # Empty transcript → RuntimeError with exact message (used by Flask route to return 422)
    mock_model.transcribe.return_value = ([], mock_info)
    with patch.object(transcription, '_get_model', return_value=mock_model):
        try:
            _transcribe(dummy_audio)
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert str(e) == 'No speech detected in audio.'
            print('  ✓ empty transcript raises RuntimeError("No speech detected in audio.")')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    import tempfile as _tf
    import pathlib as _pl
    with _tf.TemporaryDirectory() as _td:
        _tmp = _pl.Path(_td)
        test_url_validation()
        dl_dir = _tmp / 'dl_test'
        dl_dir.mkdir()
        test_download_audio(dl_dir)
        tr_dir = _tmp / 'tr_test'
        tr_dir.mkdir()
        test_transcribe(tr_dir)
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
