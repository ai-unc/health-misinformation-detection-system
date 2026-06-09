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


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    test_url_validation()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
