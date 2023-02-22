import requests

url = "http://d1gsb2o3ihr2l5.cloudfront.net/models/buffalo_sc.zip"
download_path = "/home/ai/.insightface/models/buffalo_sc"

response = requests.get(url, stream=True)

with open(download_path, "wb") as file:
    for chunk in response.iter_content(chunk_size=1024):
        if chunk:
            file.write(chunk)