# Sentiment Studio 🚀

**[Live Demo](https://sentiment-flask-app-i1vg.onrender.com)** 🌐

A powerful sentiment analysis toolkit featuring a Flask web application and a standalone advanced analysis script. It leverages multiple models including **BERT** (Transformers), **VADER**, and **TextBlob** to provide accurate sentiment classification.

## 🌟 Features

### 🖥️ Web Application (`app.py`)
- **Real-time Analysis**: precise sentiment scoring for single text inputs.
- **Batch Processing**: Upload CSV files for bulk sentiment analysis.
- **Multi-Model Support**:
    - `BERT`: Deep learning model for high accuracy (DistilBERT).
    - `VADER`: Rule-based model optimized for social media text.
    - `TextBlob`: Simple polarity-based sentiment.
- **Export Results**: Download analyzed data as CSV.
- **Interactive UI**: Clean, responsive interface built with Bootstrap 5.

### 📊 Advanced Analysis Script (`sentiment_analysis_advanced_v2.py`)
- **Standalone Tool**: Run sentiment analysis without starting the web server.
- **Data Visualization**: Generates sentiment distribution plots and word clouds.
- **Model Comparison**: Compare accuracy across TextBlob, VADER, and BERT.
- **Auto-Report**: Saves detailed results to `sentiment_analysis_results.csv`.

---

## 🛠️ Tech Stack

- **Language**: Python 3.11+
- **Web Framework**: Flask
- **ML/NLP Libraries**: Transformers (Hugging Face), PyTorch, VADER, TextBlob, Scikit-learn
- **Data Handling**: Pandas, NumPy
- **Visualization**: Matplotlib, Seaborn, WordCloud

---

## 🚀 Installation

1. **Clone the repository**
   ```bash
   git clone <your-repo-url>
   cd sentiment_flask_app
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   ```

3. **Activate the environment**
   - **Windows (PowerShell)**:
     ```powershell
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS/Linux**:
     ```bash
     source .venv/bin/activate
     ```

4. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💡 Usage

### Running the Web App

1. Start the Flask server:
   ```bash
   python app.py
   ```
2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000
   ```

### Running the Advanced Analysis Script

Run the standalone script to analyze the sample dataset (`tweet_eval`) and generate reports:

```bash
python sentiment_analysis_advanced_v2.py
```

This will:
- Download necessary NLTK data.
- Load a sample dataset.
- Perform analysis using all three models.
- Display accuracy metrics in the terminal.
- Save the results to `sentiment_analysis_results.csv`.

---

## 📡 API Endpoints

The Flask app provides a robust API for integration:

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/predict` | `POST` | Analyze a single text string. |
| `/batch_predict` | `POST` | Upload a CSV for batch analysis. |
| `/batch_predict_download`| `POST` | Upload CSV and download results immediately. |

### Example API Request (JSON)

**POST** `/predict`
```json
{
  "text": "The service was absolutely amazing!",
  "model": "bert"
}
```

**Response**
```json
{
  "model": "bert",
  "sentiment": "positive",
  "confidence": 0.99,
  "label": "positive",
  "score": 0.99
}
```

---

## 📂 Project Structure

```text
sentiment_flask_app/
├── app.py                          # Main Flask application
├── sentiment_analysis_advanced_v2.py # Standalone analysis script
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
├── templates/                      # HTML templates
│   └── index.html
└── static/                         # Static assets (CSS/JS)
```

## 📝 functionality Notes
- **BERT Model**: The app downloads `distilbert-base-uncased-finetuned-sst-2-english` on the first run. This may take a few moments.
- **CSV Format**: For batch upload, ensure your CSV has a column named `text`.
