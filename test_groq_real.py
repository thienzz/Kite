import os
import sys
from dotenv import load_dotenv
load_dotenv()

print(f"PYTHONPATH: {sys.path}")
print(f"CWD: {os.getcwd()}")

try:
    import groq
    print(f"Groq location: {groq.__file__}")
    from groq import Groq
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    print(f"Groq client: {client}")
    
    # Try a simple chat completion
    print("Testing Groq chat session...")
    response = client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": "Hello, say 'Groq is active'"}]
    )
    print(f"Response: {response.choices[0].message.content}")
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
