import math

from flask import Flask, render_template, request, jsonify
from pipeline import score_text
from transcription import NoSpeechError, transcribe_url

app = Flask(__name__)

# Optional fields populated by score_text only for flagged chunks.
# Pandas coerces None → NaN when these columns are mixed (some rows have
# values, some don't), so we sanitize back to None before JSON serialization
# to keep the response payload valid JSON (NaN is not legal JSON).
_OPTIONAL_FIELDS = (
    'matched_claim',
    'evidence_correction',
    'misinfo_type',
    'misinfo_type_confidence',
)


def _sanitize_nan(value):
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json(silent=True) or {}
    text = (data.get('text') or '').strip()

    if not text:
        return jsonify({'error': 'No text provided.'}), 400

    try:
        df = score_text(text)
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    if df.empty:
        return jsonify({'error': 'No scorable chunks found in the provided text.'}), 422

    chunks = df.to_dict(orient='records')
    for row in chunks:
        # numpy bool → Python bool for JSON serialization
        row['flagged'] = bool(row['flagged'])
        # NaN → None for optional explainability fields (only set on flagged rows)
        for field in _OPTIONAL_FIELDS:
            row[field] = _sanitize_nan(row.get(field))

    summary = {
        'total':      len(df),
        'flagged':    int(df['flagged'].sum()),
        'mean_score': round(float(df['misinfo_score'].mean()), 4),
    }

    return jsonify({'chunks': chunks, 'summary': summary})


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
    except NoSpeechError as exc:
        return jsonify({'error': str(exc)}), 422
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    return jsonify({
        'transcript_text': result.text,
        'segments':        result.segments,
        'duration':        result.duration,
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
