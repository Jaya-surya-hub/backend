import os
from typing import Any, Dict
import httpx


HF_API_TOKEN = os.getenv("HF_API_TOKEN")
HF_MODEL = os.getenv("HF_MODEL", "mistralai/Mistral-7B-Instruct-v0.2")


class WiztricSentinel:
    def __init__(self) -> None:
        self.api_url = f"https://router.huggingface.co/models/{HF_MODEL}"
        self.headers = {}
        if HF_API_TOKEN:
            self.headers["Authorization"] = f"Bearer {HF_API_TOKEN}"

    async def ask(self, user_question: str, plant_context: Dict[str, Any]) -> str:
        system_prompt = """
You are Wiztric Sentinel — an intelligent AI solar plant analyst for Wiztric Technologies.

You analyse:
- Power generation
- Irradiance
- Weather
- Inverter performance
- Anomalies

Provide clear, professional, and concise answers.
Avoid repeating generic phrases.
Be technical but easy to understand.
"""
 
        full_prompt = f"""
{system_prompt}

Plant Data:
{plant_context}

User Question:
{user_question}

Answer:
"""

        if not HF_API_TOKEN:
            raise RuntimeError("HF_API_TOKEN is not configured")

        payload = {
            "inputs": full_prompt,
            "parameters": {
                "max_new_tokens": 300,
                "temperature": 0.7,
            },
        }

        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                self.api_url,
                headers=self.headers,
                json=payload,
            )

        data = resp.json()

        if isinstance(data, list) and data and isinstance(data[0], dict) and "generated_text" in data[0]:
            return str(data[0]["generated_text"])

        if isinstance(data, dict) and "generated_text" in data:
            return str(data["generated_text"])

        return str(data)
