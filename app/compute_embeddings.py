import os
import requests
import json
import base64
from pathlib import Path

def compute_embeddings(question=None, image_base64=None):
    url = 'https://api.jina.ai/v1/embeddings'
    headers = {
        'Content-Type': 'application/json',
        'Authorization': 'Bearer ' + os.environ.get('JINA_API_KEY')
    }

    # Construct payload
    input_data = {}
    if question:
        input_data["text"] = question
    if image_base64:
        input_data["image"] = image_base64

    if not input_data:
        raise ValueError("Either `question` or `image_base64` must be provided.")

    payload = {
        "model": "jina-clip-v2",
        "input": [input_data]
    }

    response = requests.post(url, headers=headers, json=payload)

    if response.status_code == 200:
        embeddings = response.json()["data"]
        return embeddings[0]["embedding"] 
    else:
        raise Exception(f"Error: {response.status_code}, {response.text}")


##### Testing the function #####
# file_path = Path(__file__).parent / "download.png"
# with open(file_path, "rb") as f:
#     image_base64 = base64.b64encode(f.read()).decode("utf-8")

# print(compute_embeddings(image_base64=image_base64, question="What is this image about?"))

# working as expected