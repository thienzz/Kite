import os
import requests
from dotenv import load_dotenv
load_dotenv(override=True)

api_key = os.getenv("GROQ_API_KEY")
url = "https://api.groq.com/openai/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json"
}
data = {
    "model": "llama-3.1-8b-instant",
    "messages": [{"role": "user", "content": "hi"}],
    "max_tokens": 1
}

response = requests.post(url, headers=headers, json=data)
print(f"Status: {response.status_code}")
for k, v in response.headers.items():
    if "x-ratelimit" in k.lower():
        print(f"{k}: {v}")

if response.status_code != 200:
    print(f"Body: {response.text}")
