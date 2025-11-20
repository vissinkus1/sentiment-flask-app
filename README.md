# 🎭 AI Sentiment Analysis Web Application

A full-stack Flask web application for real-time sentiment analysis using three state-of-the-art NLP approaches: BERT (Deep Learning), VADER (Rule-based), and TextBlob (Statistical).

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Flask](https://img.shields.io/badge/Flask-2.0%2B-green)
![License](https://img.shields.io/badge/License-MIT-yellow)

## 🌟 Features

- **Single Text Analysis**: Analyze sentiment of individual sentences in real-time
- **Batch Processing**: Upload CSV files to analyze multiple texts at once
- **Three AI Models**: Compare results from BERT, VADER, and TextBlob
- **Beautiful UI**: Modern, responsive web interface built with Bootstrap 5
- **REST API**: Programmatic access via JSON endpoints
- **Real-time Results**: Instant sentiment classification (Positive/Negative/Neutral)

## 🚀 Demo

### Web Interface
- Enter text and get instant sentiment analysis
- Upload CSV files for batch processing
- View results in clean, organized tables

### API Endpoints
- `/predict` - Single text sentiment analysis
- `/batch_predict` - Batch CSV file processing

## 🛠️ Technologies Used

**Backend:**
- Flask (Python web framework)
- HuggingFace Transformers (BERT model)
- VADER Sentiment Analyzer
- TextBlob (NLP library)
- Pandas (Data processing)
- PyTorch (Deep learning backend)

**Frontend:**
- HTML5
- Bootstrap 5
- JavaScript (AJAX)
- jQuery

## 📋 Prerequisites

- Python 3.8 or higher
- pip (Python package manager)
- 2GB+ RAM (for BERT model)

## 💻 Installation & Setup

### 1. Clone the repository

git clone https://github.com/vissinkus1/sentiment-flask-app.git
cd sentiment-flask-app

text

### 2. Install dependencies
pip install flask textblob vaderSentiment transformers pandas torch

text

Or use requirements.txt:
pip install -r requirements.txt

text

### 3. Run the application
python app.py

text

### 4. Open in browser
Navigate to: `http://127.0.0.1:5000`

## 📁 Project Structure

sentiment-flask-app/
├── app.py # Main Flask application
├── templates/
│ └── index.html # Frontend UI
├── static/
│ └── style.css # Custom styling
├── input.csv # Sample CSV for testing
├── test_api.py # API testing script
├── test_batch_api.py # Batch testing script
├── requirements.txt # Python dependencies
└── README.md # Project documentation

text

## 🎯 Usage

### Web Interface

1. **Single Text Analysis:**
   - Enter your text in the input box
   - Select a model (BERT/VADER/TextBlob)
   - Click "Analyze" button
   - View sentiment result

2. **Batch CSV Analysis:**
   - Prepare CSV file with a 'text' column
   - Upload the file
   - Select a model
   - Click "Analyze Batch"
   - View results in table format

### API Usage (Python)

**Single Text:**
import requests

url = "http://127.0.0.1:5000/predict"
data = {
"text": "I love this product!",
"model": "bert"
}
response = requests.post(url, json=data)
print(response.json())

text

**Batch CSV:**
import requests

url = "http://127.0.0.1:5000/batch_predict"
files = {'file': open('input.csv', 'rb')}
data = {'model': 'vader'}
response = requests.post(url, files=files, data=data)
print(response.json())

text

### CSV Format for Batch Processing

Your CSV file should have a column named `text`:

text
I love this app!
The service was terrible.
It's okay, nothing special.

text

## 🤖 Models Explained

### 1. BERT (Bidirectional Encoder Representations from Transformers)
- **Type**: Deep Learning (Transformer-based)
- **Strength**: Best accuracy, understands context
- **Use case**: High-accuracy applications

### 2. VADER (Valence Aware Dictionary and sEntiment Reasoner)
- **Type**: Rule-based
- **Strength**: Fast, works well for social media text
- **Use case**: Real-time processing, tweets, reviews

### 3. TextBlob
- **Type**: Statistical NLP
- **Strength**: Simple, lightweight
- **Use case**: Quick prototyping, simple applications

## 📊 Performance

| Model    | Speed    | Accuracy | Best For              |
|----------|----------|----------|-----------------------|
| BERT     | Slow     | Highest  | Production systems    |
| VADER    | Fast     | Good     | Social media analysis |
| TextBlob | Fast     | Good     | General text          |

## 🔧 Configuration

The app runs in debug mode by default. For production:

if name == 'main':
app.run(debug=False, host='0.0.0.0', port=5000)

text

## 🚀 Deployment

### Deploy to Render (Free)

1. Push code to GitHub
2. Sign up at [render.com](https://render.com)
3. Create new Web Service
4. Connect GitHub repository
5. Set build command: `pip install -r requirements.txt`
6. Set start command: `python app.py`
7. Deploy!

### Deploy to Heroku

1. Create `Procfile`:
web: python app.py

text

2. Deploy:
heroku create your-app-name
git push heroku main

text

## 🧪 Testing

Run API tests:
Test single prediction
python test_api.py

Test batch prediction
python test_batch_api.py

text

## 🐛 Troubleshooting

**Issue: "torch not found"**
pip install torch

text

**Issue: "Templates not found"**
- Ensure `templates/` folder exists
- Check `index.html` is inside `templates/`

**Issue: Port already in use**
- Change port in `app.py`: `app.run(port=5001)`

## 🔮 Future Enhancements

- [ ] Add sentiment confidence scores
- [ ] Implement visualization (charts/graphs)
- [ ] Add multi-language support
- [ ] User authentication system
- [ ] Database integration for history
- [ ] Model comparison dashboard
- [ ] Export results to PDF/Excel

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 👨‍💻 Author

**Vishal**  
BTech CSE AIML, IILM University, Greater Noida

- GitHub: [@vissinkus1](https://github.com/vissinkush1)
- LinkedIn: [Vishal singh kushwaha](https://linkedin.com/in/yourprofile)
- Email: singhkushwahavishal344@gmail.com

## 🙏 Acknowledgments

- HuggingFace for Transformers library
- VADER Sentiment Analysis team
- TextBlob developers
- Bootstrap team for UI components

## 📚 References

- [BERT Paper](https://arxiv.org/abs/1810.04805)
- [VADER: A Parsimonious Rule-based Model](https://github.com/cjhutto/vaderSentiment)
- [TextBlob Documentation](https://textblob.readthedocs.io/)

---

⭐ If you found this project helpful, please give it a star!