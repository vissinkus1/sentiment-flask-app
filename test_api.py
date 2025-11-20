import requests

url = "http://127.0.0.1:5000/predict"

data = {
    "text": "I love learning AI and building projects!",
    "model": "textblob"  # Options: "bert", "vader", "textblob"
}

response = requests.post(url, json=data)

print("Response from API:", response.json())
