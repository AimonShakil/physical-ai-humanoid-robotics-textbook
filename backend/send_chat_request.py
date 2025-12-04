import json

import requests

url = "http://127.0.0.1:8000/api/chat"
headers = {"Content-Type": "application/json"}
payload = {"query": "What are the core concepts of ROS 2?"}

try:
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
    print("Status Code:", response.status_code)
    print("Response:", response.json())
except requests.exceptions.ConnectionError as e:
    print(f"Connection error: {e}")
except requests.exceptions.HTTPError as e:
    print(f"HTTP error occurred: {e}")
    print("Response body:", response.json() if response.text else "No response body")
except Exception as e:
    print(f"An unexpected error occurred: {e}")
