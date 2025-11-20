import requests

url = "http://127.0.0.1:5000/batch_predict"
# Open your CSV file saved in the same folder
files = {'file': open('input.csv', 'rb')}  # Replace 'input.csv' if your file name is different

# Select model: 'bert', 'vader', or 'textblob'
data = {'model': 'bert'}

response = requests.post(url, files=files, data=data)

print("Batch prediction response:")
print(response.json())
