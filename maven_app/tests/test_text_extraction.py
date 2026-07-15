"""
Tests for text_extraction.py — junk filtering, overlay grouping, text assembly.
Run from maven_app/:  python tests/test_text_extraction.py
(extract_text_url and Flask route tests are added by later tasks.)
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import text_extraction
from text_extraction import (
    _assemble_text,
    _is_junk,
    _normalize,
    group_overlay_segments,
    NoTextFoundError,
    TextExtractionResult,
    _sample_frames,
    _ocr_frames,
    extract_text_url,
)


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_is_junk():
    print('\n=== TEST 1: _is_junk ===')

    TIKTOK_TERMS    = frozenset({'tiktok'})
    INSTAGRAM_TERMS = frozenset({'instagram', 'reels', 'reel'})

    assert _is_junk('Perfectly good caption', 0.3)
    print('  ✓ low-confidence line dropped')

    assert _is_junk('ab', 0.99)
    print('  ✓ line under 3 characters dropped')

    assert _is_junk('@healthmom', 0.99)
    print('  ✓ @handle line dropped')

    assert _is_junk('TikTok', 0.99, junk_terms=TIKTOK_TERMS)
    print('  ✓ bare "TikTok" watermark dropped with TikTok junk terms')

    assert not _is_junk('TikTok', 0.99, junk_terms=INSTAGRAM_TERMS)
    print('  ✓ junk terms are platform-specific, not global')

    assert _is_junk('Instagram', 0.99, junk_terms=INSTAGRAM_TERMS)
    assert _is_junk('Reels', 0.99, junk_terms=INSTAGRAM_TERMS)
    print('  ✓ bare "Instagram"/"Reels" watermarks dropped with Instagram junk terms')

    assert _is_junk('healthmom', 0.99, uploader='healthmom')
    print('  ✓ uploader handle without @ dropped')

    assert not _is_junk('Raspberry leaf tea induces labor', 0.95, junk_terms=TIKTOK_TERMS)
    print('  ✓ normal caption kept')

    assert not _is_junk('I saw this on TikTok yesterday', 0.95, junk_terms=TIKTOK_TERMS)
    print('  ✓ sentence merely containing "tiktok" kept')

    assert not _is_junk('I saw this on Instagram yesterday', 0.95, junk_terms=INSTAGRAM_TERMS)
    print('  ✓ sentence merely containing "instagram" kept')


# ── TEST 2 ─────────────────────────────────────────────────────────────────────

def test_group_overlay_segments():
    print('\n=== TEST 2: group_overlay_segments ===')

    # Identical consecutive frames merge into one segment
    frames = [
        {'ts': 0, 'lines': [('Raspberry leaf tea', 0.9)]},
        {'ts': 1, 'lines': [('Raspberry leaf tea', 0.9)]},
        {'ts': 2, 'lines': [('Raspberry leaf tea', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert segs == [{'start': 0.0, 'end': 3.0, 'text': 'Raspberry leaf tea'}]
    print('  ✓ identical consecutive frames merge into one segment')

    # Fuzzy OCR jitter merges; highest-confidence variant wins
    frames = [
        {'ts': 0, 'lines': [('Raspberry 1eaf tea', 0.7)]},
        {'ts': 1, 'lines': [('Raspberry leaf tea', 0.95)]},
    ]
    segs = group_overlay_segments(frames)
    assert len(segs) == 1
    assert segs[0]['text'] == 'Raspberry leaf tea'
    print('  ✓ fuzzy jitter absorbed; highest-confidence text wins')

    # Empty frame splits segments (same overlay reappearing = new segment)
    frames = [
        {'ts': 0, 'lines': [('Drink this daily', 0.9)]},
        {'ts': 1, 'lines': []},
        {'ts': 2, 'lines': [('Drink this daily', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert len(segs) == 2
    assert segs[0] == {'start': 0.0, 'end': 1.0, 'text': 'Drink this daily'}
    assert segs[1] == {'start': 2.0, 'end': 3.0, 'text': 'Drink this daily'}
    print('  ✓ gap splits into two segments')

    # Different overlays become separate segments
    frames = [
        {'ts': 0, 'lines': [('Claim one', 0.9)]},
        {'ts': 1, 'lines': [('A totally different overlay', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert len(segs) == 2
    print('  ✓ different overlays produce separate segments')

    # Multi-line frames join their lines in order
    frames = [
        {'ts': 0, 'lines': [('Line one', 0.9), ('line two', 0.9)]},
    ]
    segs = group_overlay_segments(frames)
    assert segs[0]['text'] == 'Line one line two'
    print('  ✓ multi-line frame joins lines with a space')

    assert group_overlay_segments([]) == []
    print('  ✓ empty input produces no segments')


# ── TEST 3 ─────────────────────────────────────────────────────────────────────

def test_assemble_text():
    print('\n=== TEST 3: _assemble_text ===')

    segs = [
        {'start': 0.0, 'end': 3.0, 'text': 'Raspberry leaf tea'},
        {'start': 5.0, 'end': 8.0, 'text': 'raspberry leaf tea'},   # dup (case)
        {'start': 9.0, 'end': 12.0, 'text': 'Avoid your doctor'},
    ]
    out = _assemble_text('My pregnancy hack! #fyp', segs)
    assert out == 'My pregnancy hack! #fyp\nRaspberry leaf tea\nAvoid your doctor'
    print('  ✓ description + unique overlay lines, duplicates removed')

    out = _assemble_text('', segs)
    assert out == 'Raspberry leaf tea\nAvoid your doctor'
    print('  ✓ empty description omitted')

    assert _assemble_text('', []) == ''
    print('  ✓ nothing found produces empty string')


# ── TEST 4 ─────────────────────────────────────────────────────────────────────

def test_sample_frames(tmp_path):
    print('\n=== TEST 4: _sample_frames ===')

    video = tmp_path / 'video.mp4'
    video.write_bytes(b'\x00' * 100)

    captured = {}

    def fake_run(cmd, **kwargs):
        captured['cmd'] = cmd
        # Last arg is the output pattern; fake ffmpeg writing three frames
        out_pattern = Path(cmd[-1])
        for i in (1, 2, 3):
            (out_pattern.parent / f'frame_{i:04d}.png').touch()
        return MagicMock(returncode=0, stderr='')

    with patch('text_extraction.subprocess.run', side_effect=fake_run), \
         patch.object(text_extraction, 'ensure_ffmpeg', return_value=''):
        frames = _sample_frames(video, str(tmp_path))
    assert [f.name for f in frames] == ['frame_0001.png', 'frame_0002.png', 'frame_0003.png']
    print('  ✓ returns sorted frame paths')

    cmd = captured['cmd']
    assert '-t' in cmd and cmd[cmd.index('-t') + 1] == '600'
    print('  ✓ command caps sampling at MAX_VIDEO_SECONDS via -t 600')

    # ffmpeg failure → RuntimeError
    fail_dir = tmp_path / 'fail'
    fail_dir.mkdir()
    with patch('text_extraction.subprocess.run',
               return_value=MagicMock(returncode=1, stderr='corrupt file')), \
         patch.object(text_extraction, 'ensure_ffmpeg', return_value=''):
        try:
            _sample_frames(video, str(fail_dir))
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'corrupt file' in str(e)
            print('  ✓ ffmpeg failure raises RuntimeError containing stderr')


def test_ocr_frames():
    print('\n=== TEST: _ocr_frames ===')

    frames = [Path('/fake/frame_0001.png'), Path('/fake/frame_0002.png'), Path('/fake/frame_0003.png')]
    box = [[0, 0], [10, 0], [10, 10], [0, 10]]
    responses = {
        '/fake/frame_0001.png': ([[box, 'Raspberry leaf tea', 0.95],
                                  [box, '@healthmom', 0.99],
                                  [box, 'blurry noise', 0.3]], 0.1),
        '/fake/frame_0002.png': (None, 0.05),
        '/fake/frame_0003.png': ([[box, 'Raspberry leaf tea', 0.9]], 0.1),
    }

    def fake_engine(path):
        return responses[path.replace('\\', '/')]

    with patch.object(text_extraction, '_get_ocr', return_value=fake_engine):
        results = text_extraction._ocr_frames(frames, 'healthmom',
                                              frozenset({'tiktok'}))

    assert results == [
        {'ts': 0, 'lines': [('Raspberry leaf tea', 0.95)]},
        {'ts': 1, 'lines': []},
        {'ts': 2, 'lines': [('Raspberry leaf tea', 0.9)]},
    ]
    print('  ✓ parses [box, text, score] items, junk-filters, indexes ts from 0')
    print('  ✓ None OCR result yields an empty-lines frame')

    # Instagram junk terms filter the Instagram watermark line
    ig_responses = {
        '/fake/frame_0001.png': ([[box, 'Instagram', 0.99],
                                  [box, 'Castor oil starts labor', 0.95]], 0.1),
    }

    def fake_ig_engine(path):
        return ig_responses[path.replace('\\', '/')]

    with patch.object(text_extraction, '_get_ocr', return_value=fake_ig_engine):
        results = text_extraction._ocr_frames([Path('/fake/frame_0001.png')],
                                              'reelmom',
                                              frozenset({'instagram', 'reels', 'reel'}))
    assert results == [{'ts': 0, 'lines': [('Castor oil starts labor', 0.95)]}]
    print('  ✓ Instagram watermark filtered via platform junk terms')


# ── TEST 5 ─────────────────────────────────────────────────────────────────────

def test_extract_text_url():
    print('\n=== TEST 5: extract_text_url ===')

    # Non-TikTok URL → ValueError
    try:
        extract_text_url('https://www.youtube.com/watch?v=abc123')
        assert False, 'Expected ValueError'
    except ValueError as e:
        assert 'not a supported TikTok or Instagram Reels link' in str(e)
        print('  ✓ non-TikTok URL raises ValueError')

    fake_meta = {'description': 'My pregnancy hack! #fyp', 'uploader': 'healthmom'}
    fake_frames = [Path('/fake/frame_0001.png'), Path('/fake/frame_0002.png')]
    fake_frame_results = [
        {'ts': 0, 'lines': [('Raspberry leaf tea', 0.9)]},
        {'ts': 1, 'lines': [('Raspberry leaf tea', 0.9)]},
    ]

    # Happy path: description + overlays composed into result
    with patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=fake_frame_results):
        result = extract_text_url('https://www.tiktok.com/@user/video/123')
    assert result.description == 'My pregnancy hack! #fyp'
    assert result.overlay_segments == [{'start': 0.0, 'end': 2.0, 'text': 'Raspberry leaf tea'}]
    assert result.text == 'My pregnancy hack! #fyp\nRaspberry leaf tea'
    print('  ✓ happy path composes description + overlay segments + text')

    # Description only (no overlays) still succeeds
    with patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=[]):
        result = extract_text_url('https://www.tiktok.com/@user/video/123')
    assert result.text == 'My pregnancy hack! #fyp'
    assert result.overlay_segments == []
    print('  ✓ description-only video succeeds')

    # Nothing at all → NoTextFoundError
    with patch.object(text_extraction, 'fetch_metadata',
                      return_value={'description': '', 'uploader': 'x'}), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=[]), \
         patch.object(text_extraction, '_ocr_frames', return_value=[]):
        try:
            extract_text_url('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected NoTextFoundError'
        except NoTextFoundError as e:
            assert str(e) == 'No overlay text or description found in video.'
            print('  ✓ empty description + no overlays raises NoTextFoundError')

    # Temp dir cleaned up on success and on failure
    with patch('text_extraction.shutil.rmtree') as mock_rmtree, \
         patch('text_extraction.tempfile.mkdtemp', return_value='/fake/tmp'), \
         patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=fake_frame_results):
        extract_text_url('https://www.tiktok.com/@user/video/123')
    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ temp dir removed after success')

    with patch('text_extraction.shutil.rmtree') as mock_rmtree, \
         patch('text_extraction.tempfile.mkdtemp', return_value='/fake/tmp'), \
         patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video',
                      side_effect=RuntimeError('Download failed: blocked')):
        try:
            extract_text_url('https://www.tiktok.com/@user/video/123')
            assert False, 'Expected RuntimeError'
        except RuntimeError as e:
            assert 'blocked' in str(e)
    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ temp dir removed even when download fails')

    # Instagram Reel URL goes through the same pipeline
    with patch.object(text_extraction, 'fetch_metadata', return_value=fake_meta), \
         patch.object(text_extraction, 'download_video', return_value=Path('/fake/v.mp4')), \
         patch.object(text_extraction, '_sample_frames', return_value=fake_frames), \
         patch.object(text_extraction, '_ocr_frames', return_value=fake_frame_results):
        result = extract_text_url('https://www.instagram.com/reel/C8abcDEfGhi/')
    assert result.text == 'My pregnancy hack! #fyp\nRaspberry leaf tea'
    print('  ✓ Instagram Reel URL passes validation and composes result')


# ── TEST 6 ─────────────────────────────────────────────────────────────────────

def test_flask_transcribe_modes():
    print('\n=== TEST 6: /transcribe mode handling ===')
    import json
    from app import app  # loads PubMedBERT — takes ~30-60s on cold cache

    client = app.test_client()

    # Unknown mode → 400
    r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                         'mode': 'video'})
    assert r.status_code == 400, f'Expected 400, got {r.status_code}'
    assert b'Unknown mode' in r.data
    print('  ✓ unknown mode → 400')

    # Missing mode defaults to audio
    fake_audio = MagicMock()
    fake_audio.text = 'Spoken words.'
    fake_audio.segments = [{'start': 0.0, 'end': 2.0, 'text': 'Spoken words.'}]
    fake_audio.duration = 2.0
    with patch('app.transcribe_url', return_value=fake_audio):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1'})
    assert r.status_code == 200, f'Expected 200, got {r.status_code}: {r.data}'
    body = json.loads(r.data)
    assert body['mode'] == 'audio'
    assert body['text'] == 'Spoken words.'
    assert body['duration'] == 2.0
    print('  ✓ missing mode defaults to audio with unified response shape')

    # Text mode → 200 with text-mode shape
    fake_text = TextExtractionResult(
        description='My pregnancy hack! #fyp',
        overlay_segments=[{'start': 0.0, 'end': 2.0, 'text': 'Raspberry leaf tea'}],
        text='My pregnancy hack! #fyp\nRaspberry leaf tea',
    )
    with patch('app.extract_text_url', return_value=fake_text):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                             'mode': 'text'})
    assert r.status_code == 200, f'Expected 200, got {r.status_code}: {r.data}'
    body = json.loads(r.data)
    assert body['mode'] == 'text'
    assert body['text'] == 'My pregnancy hack! #fyp\nRaspberry leaf tea'
    assert body['segments'] == [{'start': 0.0, 'end': 2.0, 'text': 'Raspberry leaf tea'}]
    assert body['description'] == 'My pregnancy hack! #fyp'
    assert 'duration' not in body
    print('  ✓ text mode → 200 with mode/text/segments/description')

    # NoTextFoundError → 422
    with patch('app.extract_text_url',
               side_effect=NoTextFoundError('No overlay text or description found in video.')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                             'mode': 'text'})
    assert r.status_code == 422, f'Expected 422, got {r.status_code}'
    print('  ✓ NoTextFoundError → 422')

    # Generic failure in text mode → 500
    with patch('app.extract_text_url', side_effect=RuntimeError('Download failed: blocked')):
        r = client.post('/transcribe', json={'url': 'https://www.tiktok.com/@u/video/1',
                                             'mode': 'text'})
    assert r.status_code == 500, f'Expected 500, got {r.status_code}'
    print('  ✓ RuntimeError → 500')

    # Landing page includes the mode toggle (added in the UI task; will pass after it)
    r = client.get('/')
    assert r.status_code == 200
    print('  ✓ landing page renders')


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    import tempfile as _tf
    import pathlib as _pl
    test_is_junk()
    test_group_overlay_segments()
    test_assemble_text()
    with _tf.TemporaryDirectory() as _td:
        test_sample_frames(_pl.Path(_td))
    test_ocr_frames()
    test_extract_text_url()
    test_flask_transcribe_modes()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
