import base64
import os

import httpx

DEFAULT_MODEL = "gemini-2.5-flash-image"
DEFAULT_BASE_URL = "https://generativelanguage.googleapis.com/v1beta"


def build_staging_prompt(user_prompt: str | None, room_type: str | None, style: str | None) -> str:
    parts = [
        "You are a real estate staging assistant.",
        "Furnish the room with realistic, tasteful furniture and decor.",
        "Preserve the room layout, walls, windows, doors, and camera angle.",
        "Do not change the architecture and do not add people.",
        "Lighting should look natural and consistent with the input photo.",
        "Return a single staged image that looks like a professional listing photo.",
    ]

    if room_type:
        parts.append(f"Room type: {room_type}.")
    if style:
        parts.append(f"Style direction: {style}.")
    if user_prompt:
        parts.append(f"User request: {user_prompt}.")

    return " ".join(parts)


def _get_api_key() -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is not set")
    return api_key


def _get_model() -> str:
    return os.getenv("GEMINI_IMAGE_MODEL", DEFAULT_MODEL)


def _get_base_url() -> str:
    return os.getenv("GEMINI_API_BASE", DEFAULT_BASE_URL)


def _build_payload(prompt: str, image_bytes: bytes, mime_type: str) -> dict:
    encoded = base64.b64encode(image_bytes).decode("utf-8")
    return {
        "contents": [
            {
                "parts": [
                    {"text": prompt},
                    {"inline_data": {"mime_type": mime_type, "data": encoded}},
                ]
            }
        ]
    }


def _extract_image_bytes(response_json: dict) -> bytes:
    candidates = response_json.get("candidates", [])
    for candidate in candidates:
        content = candidate.get("content", {})
        for part in content.get("parts", []):
            inline = part.get("inlineData") or part.get("inline_data")
            if inline and inline.get("data"):
                return base64.b64decode(inline["data"])
    raise RuntimeError("No image data returned from Gemini")


async def generate_staged_image(image_bytes: bytes, mime_type: str, prompt: str) -> bytes:
    api_key = _get_api_key()
    model = _get_model()
    base_url = _get_base_url().rstrip("/")

    url = f"{base_url}/models/{model}:generateContent"
    payload = _build_payload(prompt, image_bytes, mime_type)
    headers = {
        "x-goog-api-key": api_key,
        "Content-Type": "application/json",
    }

    timeout = httpx.Timeout(120.0)
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.post(url, headers=headers, json=payload)

    if response.status_code >= 400:
        detail = response.text
        raise RuntimeError(f"Gemini API error ({response.status_code}): {detail}")

    return _extract_image_bytes(response.json())
