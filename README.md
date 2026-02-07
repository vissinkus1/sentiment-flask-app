# Sentiment Studio (Flask)

A Flask web application for sentiment analysis with three model options:
- `bert` (Transformers pipeline, loaded lazily)
- `vader` (rule-based)
- `textblob` (polarity-based)

The app supports single-text analysis, batch CSV analysis, and CSV result download.

## Current Project Status

- Backend and UI are integrated and working through Flask routes in `app.py`.
- Batch and single analysis return normalized sentiment fields: `sentiment`, `confidence`, `label`, and `score`.
- BERT loading is optional and fault-tolerant.
- If BERT is disabled or unavailable, the app falls back to VADER for `bert` requests.
- CI is configured in `.github/workflows/ci.yml` to run linting (`ruff`) and tests (`pytest`) with `DISABLE_BERT=1`.

## Features

- Single text prediction from the UI (`/predict_ui`) and API (`/predict`)
- Batch CSV prediction from the UI (`/batch_predict_ui`) and API (`/batch_predict`)
- Downloadable analyzed CSV from API (`/batch_predict_download`)
- Confidence scoring normalized to a `0.0` to `1.0` range
- Responsive frontend in `templates/index.html`

## Tech Stack

- Python 3.11+ recommended
- Flask
- Transformers + Torch
- VADER Sentiment
- TextBlob
- Pandas
- Chart.js (frontend visualization)
- Bootstrap 5

## Installation

```bash
git clone <your-repo-url>
cd sentiment_flask_app
python -m venv .venv
```

Activate virtual environment:

- Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

- macOS/Linux

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
python app.py
```

Default URL:

- `http://127.0.0.1:5000`

## Environment Variables

- `PORT`: server port (default `5000`)
- `BERT_MODEL_NAME`: Hugging Face model id (default `distilbert-base-uncased-finetuned-sst-2-english`)
- `DISABLE_BERT`: disable BERT loading when set to `1`, `true`, or `yes`

Example (PowerShell):

```powershell
$env:DISABLE_BERT='1'
python app.py
```

## API Endpoints

### 1) Single prediction

- `POST /predict`
- JSON input:

```json
{
  "text": "I love this product",
  "model": "vader"
}
```

- JSON output:

```json
{
  "model": "vader",
  "sentiment": "positive",
  "confidence": 0.8,
  "label": "positive",
  "score": 0.8
}
```

### 2) Batch prediction

- `POST /batch_predict`
- `multipart/form-data` with:
- `file`: CSV file containing a `text` column
- `model`: one of `bert`, `vader`, `textblob`

Returns a JSON array of analyzed rows.

### 3) Batch prediction download

- `POST /batch_predict_download`
- Same input format as `/batch_predict`
- Returns `text/csv` attachment with analyzed output columns

### 4) UI-specific routes

- `POST /predict_ui`
- `POST /batch_predict_ui`

These are used by the frontend and return UI-friendly prediction payloads.

## CSV Input Format

Your CSV must include a `text` column:

```csv
text
I love this app
This is average
I hate the delay
```

## Project Structure

```text
sentiment_flask_app/
|-- app.py
|-- README.md
|-- requirements.txt
|-- input.csv
|-- templates/
|   |-- index.html
|   `-- static/
|       `-- style.css
|-- tests/
|   `-- test_app.py
|-- test_api.py
`-- test_batch_api.py
```

## Testing and Linting

Run tests:

```bash
python -m pytest -q
```

Run lint checks:

```bash
ruff check .
```

## Notes

- In constrained environments (CI or low-memory systems), set `DISABLE_BERT=1`.
- `test_api.py` and `test_batch_api.py` are manual API client scripts that require the Flask app to be running.
- Pytest coverage in `tests/test_app.py` includes validation for malformed CSV, missing file, and output schema.
