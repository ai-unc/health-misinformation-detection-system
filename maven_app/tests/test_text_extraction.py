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
)


# ── TEST 1 ─────────────────────────────────────────────────────────────────────

def test_is_junk():
    print('\n=== TEST 1: _is_junk ===')

    assert _is_junk('Perfectly good caption', 0.3)
    print('  ✓ low-confidence line dropped')

    assert _is_junk('ab', 0.99)
    print('  ✓ line under 3 characters dropped')

    assert _is_junk('@healthmom', 0.99)
    print('  ✓ @handle line dropped')

    assert _is_junk('TikTok', 0.99)
    print('  ✓ bare "TikTok" watermark dropped')

    assert _is_junk('healthmom', 0.99, uploader='healthmom')
    print('  ✓ uploader handle without @ dropped')

    assert not _is_junk('Raspberry leaf tea induces labor', 0.95)
    print('  ✓ normal caption kept')

    assert not _is_junk('I saw this on TikTok yesterday', 0.95)
    print('  ✓ sentence merely containing "tiktok" kept')


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


# ── MAIN ───────────────────────────────────────────────────────────────────────

def main():
    import sys as _sys, io as _io
    _sys.stdout = _io.TextIOWrapper(_sys.stdout.buffer, encoding='utf-8')
    test_is_junk()
    test_group_overlay_segments()
    test_assemble_text()
    print('\nALL TESTS PASSED')


if __name__ == '__main__':
    main()
