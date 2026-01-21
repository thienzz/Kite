import os
from groq import Groq
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GROQ_API_KEY")
print(f"Key: {api_key[:5]}...")

client = Groq(api_key=api_key)

try:
    completion = client.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[{"role": "user", "content": "Hello"}]
    )
    print(completion.choices[0].message.content)
except Exception as e:
    print(f"Error: {e}")
