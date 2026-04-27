import os
from google import genai
from dotenv import load_dotenv

# Betöltjük a kulcsodat a .env-ből 🔑
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# Kapcsolódunk a Google szerveréhez ✨
client = genai.Client(api_key=api_key)

print("💅 Lássuk a felhozatalt, darling! Keresem az embedding modelleket...\n")

# Végigmegyünk a listán és kiírjuk azokat, amiknek a nevében benne van az "embed"
try:
    for model in client.models.list():
        if "embed" in model.name.lower():
            print(f"✨ Elérhető modell: {model.name}")
except Exception as e:
    print(f"Girl, valami nagyon félrement: {e} 💀")