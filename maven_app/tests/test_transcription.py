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

    # vm.tiktok.com short link (mobile share) → must also pass validation
    with patch.object(transcription, '_download_audio', return_value=Path('/tmp/fake.mp3')), \
         patch.object(transcription, '_transcribe',
                      return_value=TranscriptResult(text='hi', segments=[], duration=1.0)):
        result = transcribe_url('https://vm.tiktok.com/ZMhAbcDef/')
    assert result.text == 'hi'
    print('  ✓ vm.tiktok.com short URL passes validation and returns TranscriptResult')


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
    mock_seg.start = 0.123456
    mock_seg.end   = 2.5678
    mock_seg.text  = '  Hello world  '

    mock_info          = MagicMock()
    mock_info.duration = 2.56789

    mock_model = MagicMock()
    mock_model.transcribe.return_value = ([mock_seg], mock_info)

    with patch.object(transcription, '_get_model', return_value=mock_model):
        result = _transcribe(dummy_audio)

    assert result.text == 'Hello world'
    assert result.duration == 2.57
    assert result.segments == [{'start': 0.12, 'end': 2.57, 'text': 'Hello world'}]
    print('  ✓ _transcribe returns correct TranscriptResult')

    # Empty transcript → NoSpeechError with exact message (used by Flask route to return 422)
    mock_model.transcribe.return_value = ([], mock_info)
    with patch.object(transcription, '_get_model', return_value=mock_model):
        try:
            _transcribe(dummy_audio)
            assert False, 'Expected NoSpeechError'
        except transcription.NoSpeechError as e:
            assert str(e) == 'No speech detected in audio.'
            print('  ✓ empty transcript raises RuntimeError("No speech detected in audio.")')


# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_transcribe_url_cleanup():
    print('\n=== TEST 4: transcribe_url temp file cleanup ===')

    fake_result = TranscriptResult(text='test', segments=[], duration=1.0)

    # Cleanup on success
    with patch('transcription.shutil.rmtree') as mock_rmtree, \
         patch('transcription.tempfile.mkdtemp', return_value='/fake/tmp'), \
         patch.object(transcription, '_download_audio', return_value=Path('/fake/tmp/audio.mp3')), \
         patch.object(transcription, '_transcribe', return_value=fake_result):
        transcribe_url('https://www.tiktok.com/@user/video/123')

    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ shutil.rmtree called after successful transcription')

    # Cleanup on failure
    with patch('transcription.shutil.rmtree') as mock_rmtree, \
         patch('transcription.tempfile.mkdtemp', return_value='/fake/tmp'), \
         patch.object(transcription, '_download_audio', return_value=Path('/fake/tmp/audio.mp3')), \
         patch.object(transcription, '_transcribe', side_effect=RuntimeError('boom')):
        try:
            transcribe_url('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError to propagate'
        except RuntimeError as e:
            assert str(e) == 'boom'

    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ shutil.rmtree called even when _transcribe raises')


# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_flask_transcribe_route():
    print('\n=== TEST 5: /transcribe Flask route ===')
    import json
    from app import app  # loads PubMedBERT — takes ~30-60s on cold cache

    client = app.test_client()

    # Missing URL → 400
    r = client.post('/transcribe', json={})
    assert r.status_code == 400, f'Expected 400, got {r.status_code}'
    assert b'No URL provided' in r.data
    print('  ✓ missing URL → 400')

    # Non-TikTok URL → 400
    r = client.post('/transcribe', json={'url': 'https://youtube.com/watch?v=abc'})
    assert r.status_code == 400, f'Expected 400, got {r.status_code}'
    assert b'does not appear to be a TikTok link' in r.data
    print('  ✓ non-TikTok URL → 400')

    # Valid URL (mocked internals) → 200 with correct shape
    fake = TranscriptResult(
        text='Raspberry leaf tea is safe.',
        segments=[{'start': 0.0, 'end': 3.2, 'text': 'Raspberry leaf tea is safe.'}],
        duration=3.2,
    )
    with patch.object(transcription, '_download_audio', return_value=Path('/fake/audio.mp3')), \
         patch.object(transcription, '_transcribe', return_value=fake):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@user/video/123'})
    assert r.status_code == 200, f'Expected 200, got {r.status_code}: {r.data}'
    body = json.loads(r.data)
    assert body['transcript_text'] == 'Raspberry leaf tea is safe.'
    assert body['segments'] == [{'start': 0.0, 'end': 3.2, 'text': 'Raspberry leaf tea is safe.'}]
    assert body['duration'] == 3.2
    print('  ✓ valid TikTok URL → 200 with transcript_text, segments, duration')

    # No-speech error → 422
    with patch.object(transcription, '_download_audio', return_value=Path('/fake/audio.mp3')), \
         patch.object(transcription, '_transcribe',
                      side_effect=transcription.NoSpeechError('No speech detected in audio.')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@user/video/456'})
    assert r.status_code == 422, f'Expected 422, got {r.status_code}'
    print('  ✓ no-speech → 422')

    # Generic download error → 500
    with patch.object(transcription, '_download_audio', return_value=Path('/fake/audio.mp3')), \
         patch.object(transcription, '_transcribe',
                      side_effect=RuntimeError('network timeout')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@user/video/789'})
    assert r.status_code == 500, f'Expected 500, got {r.status_code}'
    print('  ✓ generic RuntimeError → 500')

    # FileNotFoundError (yt-dlp not installed) → 500
    with patch.object(transcription, '_download_audio',
                      side_effect=FileNotFoundError('yt-dlp not found')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@user/video/789'})
    assert r.status_code == 500, f'Expected 500, got {r.status_code}'
    print('  ✓ FileNotFoundError (yt-dlp not installed) → 500')


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
    test_transcribe_url_cleanup()
    test_flask_transcribe_route()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
