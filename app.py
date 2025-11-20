from flask import Flask, request, jsonify, render_template, Response
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from transformers import pipeline
import pandas as pd

# Create Flask app with templates and static folder support
app = Flask(__name__, template_folder='templates', static_folder='static')

# Initialize sentiment models
vader = SentimentIntensityAnalyzer()
bert_pipeline = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")

def analyze_sentiment(text, model):
    if model == "textblob":
        polarity = TextBlob(text).sentiment.polarity
        if polarity > 0.1:
            return "positive"
        elif polarity < -0.1:
            return "negative"
        else:
            return "neutral"
    elif model == "vader":
        compound = vader.polarity_scores(text)['compound']
        if compound >= 0.05:
            return "positive"
        elif compound <= -0.05:
            return "negative"
        else:
            return "neutral"
    elif model == "bert":
        label = bert_pipeline(text[:512])[0]['label'].lower()
        return label
    else:
        return "unsupported_model"

# --------- API Endpoints for scripts/programs ---------

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    text = data.get('text', '')
    model = data.get('model', 'vader')  # Default to VADER
    sentiment = analyze_sentiment(text, model)
    return jsonify({'model': model, 'sentiment': sentiment})

@app.route('/batch_predict', methods=['POST'])
def batch_predict():
    try:
        file = request.files['file']
        model = request.form.get('model', 'vader')
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            return jsonify({"error": "CSV must have a 'text' column"}), 400
        df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(str(x), model))
        # Return JSON with correct mimetype for batch
        return Response(df.to_json(orient='records'), mimetype='application/json')
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# --------- UI Endpoints for frontend AJAX ---------

@app.route('/')
def index():
    return render_template('index.html')  # Your frontend HTML page

@app.route('/predict_ui', methods=['POST'])
def predict_ui():
    data = request.get_json()
    text = data.get('text', '')
    model = data.get('model', 'vader')
    sentiment = analyze_sentiment(text, model)
    return jsonify({'model': model, 'sentiment': sentiment})

@app.route('/batch_predict_ui', methods=['POST'])
def batch_predict_ui():
    try:
        file = request.files['file']
        model = request.form.get('model', 'vader')
        df = pd.read_csv(file)
        if 'text' not in df.columns:
            return jsonify({"error": "CSV must have a 'text' column"}), 400
        df['sentiment'] = df['text'].apply(lambda x: analyze_sentiment(str(x), model))
        result = df[['text', 'sentiment']].to_dict(orient='records')
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

# Run the Flask app
if __name__ == '__main__':
    import os
    print("Starting Flask server...")
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)


