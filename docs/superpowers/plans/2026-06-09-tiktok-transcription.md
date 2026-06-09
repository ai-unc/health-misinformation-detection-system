# TikTok Audio Transcription Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a TikTok URL intake path to MAVEN that downloads audio, transcribes it locally with Faster-Whisper, and feeds the plain transcript text into the existing `score_text()` pipeline.

**Architecture:** A new `transcription.py` module handles download (yt-dlp) and transcription (faster-whisper small, CPU int8) behind a single public function `transcribe_url(url)`. A new `POST /transcribe` Flask route exposes it. The UI gains a URL bar above the textarea that calls `/transcribe` and populates the transcript text; the existing analyze flow is untouched. The notebook gains a standalone section mirroring the module logic for Colab use (no cross-import).

**Tech Stack:** `faster-whisper`, `yt-dlp`, Python `unittest.mock`, Flask test client.

---

## File Map

| File | Change |
|---|---|
| `maven_app/requirements.txt` | Add `faster-whisper`, `yt-dlp` |
| `maven_app/transcription.py` | Create: `TranscriptResult`, `transcribe_url`, `_download_audio`, `_get_model`, `_transcribe` |
| `maven_app/tests/test_transcription.py` | Create: all unit + route tests |
| `maven_app/app.py` | Add `from transcription import transcribe_url` + `POST /transcribe` route |
| `maven_app/templates/index.html` | Add URL bar, transcript reference panel, JS handlers |
| `MAVEN_AI_UNC_SPR2026.ipynb` | Add "TikTok Audio Transcription" section (5 cells) |

---

### Task 1: Add Dependencies

**Files:**
- Modify: `maven_app/requirements.txt`

- [ ] **Step 1: Add the two packages**

Open `maven_app/requirements.txt` and append two lines:

```
faster-whisper
yt-dlp
```

- [ ] **Step 2: Install**

```bash
cd maven_app
pip install faster-whisper yt-dlp
```

Expected: both install cleanly. `faster-whisper` also pulls in `ctranslate2`.

- [ ] **Step 3: Commit**

```bash
git add maven_app/requirements.txt
git commit -m "Add faster-whisper and yt-dlp dependencies for TikTok transcription"
```

---

### Task 2: TranscriptResult + URL Validation (TDD)

**Files:**
- Create: `maven_app/tests/test_transcription.py`
- Create: `maven_app/transcription.py` (skeleton)

- [ ] **Step 1: Write the failing test**

Create `maven_app/tests/test_transcription.py`:

```python
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
```

- [ ] **Step 2: Run test — expect it to fail**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected: `ModuleNotFoundError: No module named 'transcription'`

- [ ] **Step 3: Create transcription.py skeleton**

Create `maven_app/transcription.py`:

```python
"""
MAVEN Transcription: downloads TikTok audio and transcribes it with faster-whisper.
Public entry point: transcribe_url(url) → TranscriptResult.
"""
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

_TIKTOK_RE = re.compile(r'https?://(www\.)?tiktok\.com/')
_model = None  # lazy-loaded on first call to _get_model()


@dataclass
class TranscriptResult:
    text: str
    segments: List[dict]   # [{"start": float, "end": float, "text": str}, ...]
    duration: float


def transcribe_url(url: str) -> TranscriptResult:
    if not _TIKTOK_RE.match(url.strip()):
        raise ValueError("URL does not appear to be a TikTok link.")
    tmp_dir = tempfile.mkdtemp()
    try:
        audio_path = _download_audio(url, tmp_dir)
        return _transcribe(audio_path)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _download_audio(url: str, tmp_dir: str) -> Path:
    raise NotImplementedError


def _get_model():
    raise NotImplementedError


def _transcribe(audio_path: Path) -> TranscriptResult:
    raise NotImplementedError
```

- [ ] **Step 4: Run test — expect it to pass**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected:
```
=== TEST 1: URL validation ===
  ✓ non-TikTok URL raises ValueError
  ✓ empty URL raises ValueError
  ✓ valid TikTok URL passes validation and returns TranscriptResult

ALL TESTS PASSED
```

- [ ] **Step 5: Commit**

```bash
git add maven_app/transcription.py maven_app/tests/test_transcription.py
git commit -m "Add TranscriptResult dataclass and URL validation to transcription module"
```

---

### Task 3: `_download_audio` (TDD)

**Files:**
- Modify: `maven_app/tests/test_transcription.py` — add test
- Modify: `maven_app/transcription.py` — implement `_download_audio`

- [ ] **Step 1: Add the failing test**

In `test_transcription.py`, add this import at the top of the file (after the existing `from transcription import ...` line):

```python
from transcription import _download_audio
```

Then add this function after `test_url_validation`:

```python
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
```

Update `main()` at the bottom of the file:

```python
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
```

- [ ] **Step 2: Run test — expect it to fail**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected: `NotImplementedError` from `_download_audio`.

- [ ] **Step 3: Implement `_download_audio`**

Replace the `_download_audio` stub in `maven_app/transcription.py`:

```python
def _download_audio(url: str, tmp_dir: str) -> Path:
    """Download TikTok audio to tmp_dir as mp3. Raises RuntimeError on failure."""
    output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
    result = subprocess.run(
        [
            'yt-dlp',
            '--extract-audio',
            '--audio-format', 'mp3',
            '--output', output_template,
            '--no-playlist',
            '--quiet',
            url,
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        msg = result.stderr.strip() or f'yt-dlp exited with code {result.returncode}'
        raise RuntimeError(f'Download failed: {msg}')
    mp3_files = list(Path(tmp_dir).glob('*.mp3'))
    if not mp3_files:
        raise RuntimeError('Download failed: no audio file produced.')
    return mp3_files[0]
```

- [ ] **Step 4: Run test — expect it to pass**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected:
```
=== TEST 1: URL validation ===
  ✓ non-TikTok URL raises ValueError
  ✓ empty URL raises ValueError
  ✓ valid TikTok URL passes validation and returns TranscriptResult

=== TEST 2: _download_audio ===
  ✓ success path returns mp3 Path
  ✓ yt-dlp failure raises RuntimeError containing stderr

ALL TESTS PASSED
```

- [ ] **Step 5: Commit**

```bash
git add maven_app/transcription.py maven_app/tests/test_transcription.py
git commit -m "Implement _download_audio using yt-dlp subprocess"
```

---

### Task 4: `_get_model` + `_transcribe` (TDD)

**Files:**
- Modify: `maven_app/tests/test_transcription.py` — add test
- Modify: `maven_app/transcription.py` — implement `_get_model`, `_transcribe`

- [ ] **Step 1: Add the failing test**

Add this import at the top of `test_transcription.py`:

```python
from transcription import _transcribe
```

Add this function after `test_download_audio`:

```python
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
```

Update `main()`:

```python
def main():
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
```

- [ ] **Step 2: Run test — expect it to fail**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected: `NotImplementedError` from `_transcribe`.

- [ ] **Step 3: Implement `_get_model` and `_transcribe`**

Replace both stubs in `maven_app/transcription.py`:

```python
def _get_model():
    """Load WhisperModel once at first call; return cached instance thereafter."""
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        print('[MAVEN] Loading Whisper small model (one-time, ~244 MB)...')
        _model = WhisperModel('small', device='cpu', compute_type='int8')
        print('[MAVEN] Whisper model ready.')
    return _model


def _transcribe(audio_path: Path) -> TranscriptResult:
    """Transcribe audio_path. Raises RuntimeError if no speech is detected."""
    model = _get_model()
    segments_iter, info = model.transcribe(str(audio_path), beam_size=5)
    segments = []
    texts = []
    for seg in segments_iter:
        segments.append({
            'start': round(seg.start, 2),
            'end':   round(seg.end, 2),
            'text':  seg.text.strip(),
        })
        texts.append(seg.text.strip())
    full_text = ' '.join(t for t in texts if t)
    if not full_text.strip():
        raise RuntimeError('No speech detected in audio.')
    return TranscriptResult(
        text=full_text,
        segments=segments,
        duration=round(info.duration, 2),
    )
```

- [ ] **Step 4: Run test — expect it to pass**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected:
```
=== TEST 1: URL validation ===  ✓ ...
=== TEST 2: _download_audio ===  ✓ ...
=== TEST 3: _transcribe ===
  ✓ _transcribe returns correct TranscriptResult
  ✓ empty transcript raises RuntimeError("No speech detected in audio.")

ALL TESTS PASSED
```

- [ ] **Step 5: Commit**

```bash
git add maven_app/transcription.py maven_app/tests/test_transcription.py
git commit -m "Implement _get_model with lazy Whisper loading and _transcribe"
```

---

### Task 5: `transcribe_url` Cleanup Test

**Files:**
- Modify: `maven_app/tests/test_transcription.py` — add cleanup test

- [ ] **Step 1: Add the test**

Add this function after `test_transcribe`:

```python
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
        except RuntimeError:
            pass

    mock_rmtree.assert_called_once_with('/fake/tmp', ignore_errors=True)
    print('  ✓ shutil.rmtree called even when _transcribe raises')
```

Update `main()`:

```python
def main():
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
    print('\nALL TESTS PASSED')
```

- [ ] **Step 2: Run test — expect it to pass** (no implementation change needed; cleanup is already wired in `transcribe_url`)

```bash
cd maven_app
python tests/test_transcription.py
```

Expected: all 4 test groups pass.

- [ ] **Step 3: Commit**

```bash
git add maven_app/tests/test_transcription.py
git commit -m "Add temp file cleanup verification to transcription tests"
```

---

### Task 6: `POST /transcribe` Flask Route (TDD)

**Files:**
- Modify: `maven_app/tests/test_transcription.py` — add route tests
- Modify: `maven_app/app.py` — add route + import

- [ ] **Step 1: Add the failing route tests**

Add this function after `test_transcribe_url_cleanup`:

```python
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
                      side_effect=RuntimeError('No speech detected in audio.')):
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
```

Update `main()`:

```python
def main():
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
```

- [ ] **Step 2: Run test — expect it to fail**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected: the route tests post to `/transcribe` and receive 404 (route doesn't exist yet).

- [ ] **Step 3: Add import to app.py**

In `maven_app/app.py`, add this line after `from pipeline import score_text`:

```python
from transcription import transcribe_url
```

- [ ] **Step 4: Add the route to app.py**

Add this route after the `analyze` function and before `if __name__ == '__main__':`:

```python
@app.route('/transcribe', methods=['POST'])
def transcribe():
    data = request.get_json(silent=True) or {}
    url = (data.get('url') or '').strip()

    if not url:
        return jsonify({'error': 'No URL provided.'}), 400

    try:
        result = transcribe_url(url)
    except ValueError as exc:
        return jsonify({'error': str(exc)}), 400
    except RuntimeError as exc:
        msg = str(exc)
        if msg == 'No speech detected in audio.':
            return jsonify({'error': msg}), 422
        return jsonify({'error': msg}), 500

    return jsonify({
        'transcript_text': result.text,
        'segments':        result.segments,
        'duration':        result.duration,
    })
```

- [ ] **Step 5: Run test — expect it to pass**

```bash
cd maven_app
python tests/test_transcription.py
```

Expected: all 5 test groups pass. Note: `from app import app` loads PubMedBERT (~30-60s the first time if the cache is cold).

- [ ] **Step 6: Commit**

```bash
git add maven_app/app.py maven_app/tests/test_transcription.py
git commit -m "Add /transcribe Flask route with full error handling"
```

---

### Task 7: UI — URL Bar, Transcript Panel, JS Handlers

**Files:**
- Modify: `maven_app/templates/index.html`

- [ ] **Step 1: Add URL bar HTML**

In `index.html`, find this line (the opening of the square textarea wrapper):

```html
<div class="relative aspect-square w-full bg-primary shadow-[0_20px_50px_rgba(241,233,210,0.15)]">
```

Insert the following block **immediately before** that line:

```html
<!-- URL bar -->
<div class="flex gap-2">
  <input
    id="url-input"
    type="url"
    class="flex-1 min-w-0 px-4 py-3 bg-surface-container border border-outline-variant/20 font-label text-xs text-on-surface placeholder:text-outline/30 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-[-2px]"
    placeholder="TikTok URL (optional)"
  />
  <button
    id="transcribe-btn"
    disabled
    class="px-5 py-3 bg-surface-container-highest text-primary font-label font-medium uppercase tracking-widest text-[10px] transition-all duration-300 hover:bg-on-surface hover:text-surface-dim active:scale-[0.98] sharp-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2 disabled:opacity-30 disabled:pointer-events-none"
  >Transcribe</button>
</div>
```

- [ ] **Step 2: Add transcript reference panel HTML**

Find this line (the analyze button):

```html
<button id="analyze-btn" class="w-full py-5 bg-surface-container-highest text-primary font-label font-medium uppercase tracking-widest transition-all duration-300 hover:bg-on-surface hover:text-surface-dim active:scale-[0.98] sharp-0 focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary focus-visible:outline-offset-2">
```

Insert the following block **immediately before** that line:

```html
<!-- Transcript reference panel (hidden until transcription completes) -->
<div id="transcript-panel" hidden class="bg-surface-container-low border border-outline-variant/10">
  <button
    id="transcript-toggle"
    type="button"
    aria-expanded="false"
    aria-controls="transcript-segments"
    class="w-full flex items-center justify-between px-4 py-3 font-label text-[10px] uppercase tracking-widest hover:bg-surface-container transition-colors focus-visible:outline focus-visible:outline-2 focus-visible:outline-primary"
  >
    <span class="text-tertiary-fixed-dim">Transcript Reference</span>
    <span id="transcript-meta" class="text-outline"></span>
  </button>
  <div
    id="transcript-segments"
    hidden
    class="px-4 pb-4 pt-3 flex flex-col gap-3 max-h-48 overflow-y-auto no-scrollbar border-t border-outline-variant/10"
  ></div>
</div>
```

- [ ] **Step 3: Add new DOM references to the JS**

In the `<script>` block, find the existing DOM constant block:

```js
const textarea       = document.getElementById('text-input');
const charCount      = document.getElementById('char-count');
const analyzeBtn     = document.getElementById('analyze-btn');
const cardsContainer = document.getElementById('cards-container');
const watermark      = document.getElementById('watermark');
const statusLabel    = document.getElementById('status-label');
```

Add these six lines immediately after that block:

```js
const urlInput         = document.getElementById('url-input');
const transcribeBtn    = document.getElementById('transcribe-btn');
const transcriptPanel  = document.getElementById('transcript-panel');
const transcriptToggle = document.getElementById('transcript-toggle');
const transcriptMeta   = document.getElementById('transcript-meta');
const transcriptSegs   = document.getElementById('transcript-segments');
```

- [ ] **Step 4: Add URL input listener, transcribe click handler, and helpers**

Add the following block immediately after the existing `textarea.addEventListener('input', ...)` handler:

```js
// ── Enable Transcribe when URL is non-empty ────────────────────────────
urlInput.addEventListener('input', () => {
    transcribeBtn.disabled = !urlInput.value.trim();
});

// ── Transcribe ─────────────────────────────────────────────────────────
transcribeBtn.addEventListener('click', async () => {
    const url = urlInput.value.trim();
    if (!url) return;

    setTranscribeLoading(true);
    transcriptPanel.hidden = true;

    try {
        const res  = await fetch('/transcribe', {
            method:  'POST',
            headers: { 'Content-Type': 'application/json' },
            body:    JSON.stringify({ url }),
        });
        const data = await res.json();

        if (data.error) {
            statusLabel.textContent = 'Transcription failed';
            renderTranscribeError(data.error);
        } else {
            textarea.value = data.transcript_text;
            charCount.textContent = `Character Count: ${data.transcript_text.length}`;
            renderTranscriptPanel(data);
            statusLabel.textContent = 'Transcript ready';
        }
    } catch (_err) {
        statusLabel.textContent = 'Connection failed';
        renderTranscribeError('Transcription failed — ensure the server is running.');
    } finally {
        setTranscribeLoading(false);
    }
});

// ── Transcript panel toggle ────────────────────────────────────────────
transcriptToggle.addEventListener('click', () => {
    const isOpen = transcriptToggle.getAttribute('aria-expanded') === 'true';
    transcriptToggle.setAttribute('aria-expanded', String(!isOpen));
    transcriptSegs.hidden = isOpen;
});

// ── Transcribe loading state ───────────────────────────────────────────
function setTranscribeLoading(on) {
    transcribeBtn.disabled = true;
    analyzeBtn.disabled    = on;
    textarea.readOnly      = on;

    if (on) {
        transcribeBtn.innerHTML = '<div class="shimmer-bar w-full h-1 py-[8px]"></div>';
    } else {
        transcribeBtn.innerHTML = urlInput.value.trim() ? 'Retranscribe' : 'Transcribe';
        transcribeBtn.disabled  = !urlInput.value.trim();
        analyzeBtn.disabled     = false;
        textarea.readOnly       = false;
    }
}

// ── Render transcript reference panel ─────────────────────────────────
function renderTranscriptPanel(data) {
    const mins   = Math.floor(data.duration / 60);
    const secs   = Math.round(data.duration % 60);
    const durStr = `${mins}:${String(secs).padStart(2, '0')}`;
    transcriptMeta.textContent = `${durStr} · ${data.segments.length} segments ▾`;

    transcriptSegs.innerHTML = data.segments.map(seg =>
        `<div class="flex gap-3 items-baseline">
            <span class="font-label text-[9px] text-outline uppercase tracking-tighter whitespace-nowrap">${fmtTime(seg.start)} – ${fmtTime(seg.end)}</span>
            <span class="font-body italic text-[12px] text-on-surface-variant leading-snug">${escapeHtml(seg.text)}</span>
        </div>`
    ).join('');

    transcriptPanel.hidden = false;
    transcriptToggle.setAttribute('aria-expanded', 'false');
    transcriptSegs.hidden = true;
}

// ── Transcribe error display ───────────────────────────────────────────
function renderTranscribeError(message) {
    watermark.style.display = 'none';
    cardsContainer.innerHTML = `
        <article class="bg-[#1E1C19] border border-[#2A2824] border-l-4 border-l-[#7A3A2A] p-6">
            <p class="font-body italic text-[13.5px] leading-relaxed text-on-surface-variant">${escapeHtml(message)}</p>
        </article>`;
}

// ── Seconds → M:SS ────────────────────────────────────────────────────
function fmtTime(seconds) {
    const m = Math.floor(seconds / 60);
    const s = Math.round(seconds % 60);
    return `${m}:${String(s).padStart(2, '0')}`;
}
```

- [ ] **Step 5: Manual verification**

Start the server:
```bash
cd maven_app
python app.py
```

Open `http://localhost:5000` and verify:
1. URL bar appears above the square textarea; Transcribe button is dimmed
2. Typing any text in the URL bar enables the Transcribe button
3. Clearing the URL field re-dims the Transcribe button
4. Clicking Transcribe with a non-TikTok URL shows the error card below the results header
5. The existing text-paste → Analyze Manuscript flow works exactly as before

- [ ] **Step 6: Commit**

```bash
git add maven_app/templates/index.html
git commit -m "Add TikTok URL bar, transcript reference panel, and JS transcription flow to UI"
```

---

### Task 8: Notebook Integration

**Files:**
- Modify: `MAVEN_AI_UNC_SPR2026.ipynb`

- [ ] **Step 1: Locate the insertion point**

Open `MAVEN_AI_UNC_SPR2026.ipynb`. Find the markdown cell whose heading begins `## Text Segmentation` (or similar). The five new cells go **immediately before** this cell.

- [ ] **Step 2: Insert markdown header cell**

Insert a new **markdown** cell:

```markdown
## TikTok Audio Transcription

Downloads audio from a TikTok URL and transcribes it using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (`small` model, CPU int8). The full transcript text is passed directly to `score_text()`. A timestamped segment DataFrame is shown as a reference.
```

- [ ] **Step 3: Insert install cell**

Insert a new **code** cell:

```python
!pip install faster-whisper yt-dlp -q
```

- [ ] **Step 4: Insert imports + TranscriptResult cell**

Insert a new **code** cell:

```python
import re, shutil, subprocess, tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List


@dataclass
class TranscriptResult:
    text: str
    segments: List[dict]   # [{"start": float, "end": float, "text": str}, ...]
    duration: float
```

- [ ] **Step 5: Insert transcribe_url function cell**

Insert a new **code** cell:

```python
def transcribe_url(url: str) -> TranscriptResult:
    """Download TikTok audio and transcribe it. Returns TranscriptResult."""
    if not re.match(r'https?://(www\.)?tiktok\.com/', url.strip()):
        raise ValueError("URL does not appear to be a TikTok link.")

    from faster_whisper import WhisperModel
    print("Loading Whisper small model (one-time, ~244 MB)...")
    model = WhisperModel('small', device='cpu', compute_type='int8')

    tmp_dir = tempfile.mkdtemp()
    try:
        output_template = str(Path(tmp_dir) / '%(id)s.%(ext)s')
        result = subprocess.run(
            ['yt-dlp', '--extract-audio', '--audio-format', 'mp3',
             '--output', output_template, '--no-playlist', '--quiet', url],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr.strip()}")
        mp3_files = list(Path(tmp_dir).glob('*.mp3'))
        if not mp3_files:
            raise RuntimeError("Download failed: no audio file produced.")

        segments_iter, info = model.transcribe(str(mp3_files[0]), beam_size=5)
        segments, texts = [], []
        for seg in segments_iter:
            segments.append({'start': round(seg.start, 2), 'end': round(seg.end, 2), 'text': seg.text.strip()})
            texts.append(seg.text.strip())

        full_text = ' '.join(t for t in texts if t)
        if not full_text.strip():
            raise RuntimeError("No speech detected in audio.")

        return TranscriptResult(text=full_text, segments=segments, duration=round(info.duration, 2))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
```

- [ ] **Step 6: Insert demo cell**

Insert a new **code** cell:

```python
import pandas as pd

# Replace with a real TikTok URL before running
VIDEO_URL = "https://www.tiktok.com/@user/video/..."

result = transcribe_url(VIDEO_URL)

print(f"Duration : {result.duration:.1f}s")
print(f"Segments : {len(result.segments)}")
print(f"\nFull transcript:\n{result.text}\n")

# Timestamped segment reference
display(pd.DataFrame(result.segments))

# Pass transcript into MAVEN pipeline
# (run all pipeline cells above first so score_text is defined)
df = score_text(result.text)
display(df)
```

- [ ] **Step 7: Verify notebook cell order**

Confirm the notebook section order is now:
1. Pipeline Overview
2. **TikTok Audio Transcription** ← new
3. Text Segmentation
4. PubMedBERT Embeddings
5. Misinformation Markers

Run the markdown cell, install cell, and function-definition cells in Colab to confirm they execute without errors. (The demo cell requires a real TikTok URL — skip for now.)

- [ ] **Step 8: Commit**

```bash
git add MAVEN_AI_UNC_SPR2026.ipynb
git commit -m "Add TikTok audio transcription section to MAVEN notebook"
```
