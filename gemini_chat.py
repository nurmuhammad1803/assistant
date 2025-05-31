import os
from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

def ask_gemini(prompt: str) -> str:
    response = client.models.generate_content(
        model="gemini-1.5-flash",  # or "gemini-1.5-pro"
        contents=prompt
    )
    return response.text
