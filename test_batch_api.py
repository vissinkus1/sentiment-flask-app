import requests

def main():
    url = "http://127.0.0.1:5000/batch_predict"
    data = {"model": "bert"}

    # Open your CSV file saved in the same folder.
    with open("input.csv", "rb") as csv_file:
        files = {"file": csv_file}
        response = requests.post(url, files=files, data=data, timeout=30)

    print("Batch prediction response:")
    print(response.json())


if __name__ == "__main__":
    main()
