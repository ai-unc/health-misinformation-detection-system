from flask import Flask, render_template, request, jsonify
from pipeline import score_text

app = Flask(__name__)


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
    # Convert numpy bools to Python bools for JSON serialisation
    for row in chunks:
        row['flagged'] = bool(row['flagged'])

    summary = {
        'total':      len(df),
        'flagged':    int(df['flagged'].sum()),
        'mean_score': round(float(df['misinfo_score'].mean()), 4),
    }

    return jsonify({'chunks': chunks, 'summary': summary})


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
