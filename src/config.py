import os

from dotenv import load_dotenv

load_dotenv()

CSV_PATH = os.getenv(
    "CSV_PATH",
    "data/Clean_Medicine_Details.csv"
)

VECTORIZER_PATH = "models/vectorizer.pkl"

SIMILARITY_THRESHOLD = float(
    os.getenv("SIMILARITY_THRESHOLD", 0.25)
)

TOP_N = int(
    os.getenv("TOP_N", 5)
)

OPENROUTER_API_KEY = os.getenv(
    "OPENROUTER_API_KEY"
)