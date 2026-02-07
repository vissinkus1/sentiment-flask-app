from flask import Flask, request, jsonify, render_template, Response
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import pandas as pd
import os
import io
from datetime import datetime


app = Flask(
    __name__,
    template_folder="templates",
    static_folder="templates/static",
)

BERT_MODEL_NAME = os.environ.get(
    "BERT_MODEL_NAME",
    "distilbert-base-uncased-finetuned-sst-2-english",
)
DISABLE_BERT = os.environ.get("DISABLE_BERT", "0").strip().lower() in {"1", "true", "yes"}

vader = SentimentIntensityAnalyzer()
bert_pipeline = None
BERT_NUM_LABELS = 0
BERT_LOAD_ATTEMPTED = False


def _clamp(value, low=0.0, high=1.0):
    return max(low, min(high, float(value)))


def _load_bert_pipeline():
    global bert_pipeline, BERT_NUM_LABELS, BERT_LOAD_ATTEMPTED

    if DISABLE_BERT:
        return None

    if BERT_LOAD_ATTEMPTED:
        return bert_pipeline

    BERT_LOAD_ATTEMPTED = True

    try:
        from transformers import pipeline as hf_pipeline

        bert_pipeline = hf_pipeline("sentiment-analysis", model=BERT_MODEL_NAME)
        BERT_NUM_LABELS = int(getattr(bert_pipeline.model.config, "num_labels", 2))
        print(f"Loaded BERT model: {BERT_MODEL_NAME} ({BERT_NUM_LABELS} labels)")
    except Exception as exc:
        print(f"Warning: could not load BERT model '{BERT_MODEL_NAME}'. {exc}")
        bert_pipeline = None
        BERT_NUM_LABELS = 0

    return bert_pipeline


def _normalize_bert_label(raw_label, score=None):
    label = str(raw_label or "").strip().lower()
    normalized = label.replace("-", "_").replace(" ", "_")

    if "positive" in normalized or normalized == "pos":
        return "positive"
    if "negative" in normalized or normalized == "neg":
        return "negative"
    if "neutral" in normalized or normalized == "neu":
        return "neutral"

    if "star" in normalized:
        digits = "".join(ch for ch in normalized if ch.isdigit())
        if digits:
            stars = int(digits[0])
            if stars <= 2:
                return "negative"
            if stars == 3:
                return "neutral"
            return "positive"

    if normalized in {"label_0", "label0"}:
        return "negative"
    if normalized in {"label_1", "label1"}:
        return "positive" if BERT_NUM_LABELS == 2 else "neutral"
    if normalized in {"label_2", "label2"}:
        return "positive"

    if score is not None and BERT_NUM_LABELS == 2:
        return "positive" if float(score) >= 0.5 else "negative"

    return "neutral"


def _analyze_textblob(text):
    polarity = TextBlob(text).sentiment.polarity
    if polarity > 0.1:
        return "positive", _clamp(abs(polarity))
    if polarity < -0.1:
        return "negative", _clamp(abs(polarity))
    return "neutral", _clamp(1.0 - abs(polarity))


def _analyze_vader(text):
    compound = vader.polarity_scores(text)["compound"]
    if compound >= 0.05:
        return "positive", _clamp(abs(compound))
    if compound <= -0.05:
        return "negative", _clamp(abs(compound))
    return "neutral", _clamp(1.0 - abs(compound))


def _analyze_bert(text):
    model = _load_bert_pipeline()
    if model is None:
        return _analyze_vader(text)

    try:
        prediction = model(text, truncation=True, max_length=512)[0]
        sentiment = _normalize_bert_label(prediction.get("label"), prediction.get("score"))
        score = _clamp(prediction.get("score", 0.5))
        if sentiment == "neutral":
            score = _clamp(max(score, 1.0 - score))
        return sentiment, score
    except Exception:
        return _analyze_vader(text)


def analyze_sentiment_with_confidence(text, model):
    text = str(text or "").strip()
    if not text:
        return {
            "sentiment": "neutral",
            "confidence": 0.0,
            "label": "neutral",
            "score": 0.0,
        }

    model = str(model or "vader").lower()

    if model == "textblob":
        sentiment, confidence = _analyze_textblob(text)
    elif model == "vader":
        sentiment, confidence = _analyze_vader(text)
    elif model == "bert":
        sentiment, confidence = _analyze_bert(text)
    else:
        return {
            "sentiment": "unsupported_model",
            "confidence": 0.0,
            "label": "unsupported_model",
            "score": 0.0,
        }

    rounded_confidence = round(_clamp(confidence), 4)
    return {
        "sentiment": sentiment,
        "confidence": rounded_confidence,
        "label": sentiment,
        "score": rounded_confidence,
    }


def analyze_sentiment(text, model):
    return analyze_sentiment_with_confidence(text, model)["sentiment"]


def _load_uploaded_csv(file_storage):
    if file_storage is None or file_storage.filename == "":
        raise ValueError("No CSV file uploaded.")

    df = pd.read_csv(file_storage, keep_default_na=False)
    if "text" not in df.columns:
        raise ValueError("CSV must have a 'text' column")
    return df


def _batch_analyze(file_storage, model):
    df = _load_uploaded_csv(file_storage)
    analysis = df["text"].apply(lambda value: analyze_sentiment_with_confidence(value, model))
    df["sentiment"] = analysis.apply(lambda item: item["sentiment"])
    df["confidence"] = analysis.apply(lambda item: item["confidence"])
    df["label"] = analysis.apply(lambda item: item["label"])
    df["score"] = analysis.apply(lambda item: item["score"])
    return df


def _batch_export_frame(df):
    export_columns = [
        column
        for column in ["text", "label", "score", "sentiment", "confidence"]
        if column in df.columns
    ]
    return df[export_columns] if export_columns else df


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    model = data.get("model", "vader")
    result = analyze_sentiment_with_confidence(text, model)
    return jsonify({"model": model, **result})


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    try:
        file = request.files.get("file")
        model = request.form.get("model", "vader")
        df = _batch_analyze(file, model)
        return jsonify(df.to_dict(orient="records"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/batch_predict_download", methods=["POST"])
def batch_predict_download():
    try:
        file = request.files.get("file")
        model = request.form.get("model", "vader")
        df = _batch_export_frame(_batch_analyze(file, model))

        buffer = io.StringIO()
        df.to_csv(buffer, index=False)
        filename = f"sentiment_results_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
        return Response(
            buffer.getvalue(),
            mimetype="text/csv",
            headers={"Content-Disposition": f"attachment; filename={filename}"},
        )
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


@app.route("/predict_ui", methods=["POST"])
def predict_ui():
    data = request.get_json(silent=True) or {}
    text = data.get("text", "")
    model = data.get("model", "vader")
    result = analyze_sentiment_with_confidence(text, model)
    return jsonify({"model": model, **result})


@app.route("/batch_predict_ui", methods=["POST"])
def batch_predict_ui():
    try:
        file = request.files.get("file")
        model = request.form.get("model", "vader")
        df = _batch_export_frame(_batch_analyze(file, model))
        return jsonify(df.to_dict(orient="records"))
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400


if __name__ == "__main__":
    print("Starting Flask server...")
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
