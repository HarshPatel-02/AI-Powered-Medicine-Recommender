import requests
import json

from src.config import OPENROUTER_API_KEY


def get_openrouter_response(symptoms: str):

    if not OPENROUTER_API_KEY:
        return {
            "error": "OpenRouter API key not configured"
        }

    prompt = f"""
You are a professional AI medical assistant.

User symptoms:
{symptoms}

Return ONLY valid JSON.

Format:

{{
    "disease": "",
    "medicines": [
        {{
            "name": "",
            "uses": "",
            "side_effects": "",
            "precautions": ""
        }}
    ],
    "advice": ""
}}

Rules:
- Give maximum 3 medicines
- Short and professional response
- JSON only
"""

    try:

        response = requests.post(
            url="https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json"
            },
            json={
                "model": "deepseek/deepseek-v4-flash:free",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            },
            timeout=60
        )

        data = response.json()

        content = data["choices"][0]["message"]["content"]

        return json.loads(content)

    except Exception as e:

        return {
            "error": str(e)
        }