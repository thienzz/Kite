import os
from dotenv import load_dotenv
load_dotenv()

print(f"LLM_PROVIDER: {os.getenv('LLM_PROVIDER')}")
print(f"GROQ_API_KEY: {os.getenv('GROQ_API_KEY')[:10]}...")

try:
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print("Groq library imported and client initialized.")
except ImportError:
    print("Groq library NOT found.")
except Exception as e:
    print(f"Error: {e}")
