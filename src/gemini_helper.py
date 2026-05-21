import google.generativeai as genai
from src.config import API_KEY, LANGUAGE_INSTRUCTIONS, LANGUAGE

if API_KEY:
    genai.configure(api_key=API_KEY)
    model = genai.GenerativeModel('gemini-2.5-flash')
else:
    model = None

def get_gemini_response(symptoms: str, data_str: str, source: str):
    instruction = LANGUAGE_INSTRUCTIONS.get(LANGUAGE, LANGUAGE_INSTRUCTIONS["English"])

    prompt = f"""
{instruction}

User symptoms: {symptoms}
Source: {source}
{data_str if data_str else "No close match found in database."}

Recommend the best medicine(s) professionally.
For each medicine give:
- Medicine Name
- Why it is suitable
- Possible Side Effects
- Precautions / Note

At the end, clearly write: "This is a general recommendation and not a substitute for professional medical advice. Always consult with a healthcare provider for personalized medical advice."
"""

    if not model:
        return "⚠️ Gemini API key not configured. Please set GEMINI_API_KEY in .env"

    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating AI response: {str(e)}"