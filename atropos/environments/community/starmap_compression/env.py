import os

from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
print("API Key:", api_key[:10] + "..." if api_key else "Not found")
