import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")
print("API key loaded:", bool(api_key))
print("Key starts with:", api_key[:6])

# NEW client (this is the key change)
client = genai.Client(api_key=api_key)

# List models
for m in client.models.list():
    print("MODEL:", m.name)