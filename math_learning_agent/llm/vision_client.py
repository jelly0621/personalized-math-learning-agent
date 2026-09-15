"""Minimal OpenAI-compatible client for one in-memory image at a time."""

import base64

from openai import OpenAI

from math_learning_agent.config import get_settings


SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/webp"}
)


class VisionLLMClient:
    """Send image bytes as a data URL and return the model's text response."""

    def __init__(self) -> None:
        settings = get_settings()
        self.model = settings.vision_model
        self._client = OpenAI(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
        )

    def invoke_image(
        self,
        image_bytes: bytes,
        mime_type: str,
        prompt: str,
    ) -> str:
        if mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            raise ValueError(f"Unsupported image MIME type: {mime_type}.")
        if not image_bytes:
            raise ValueError("The image is empty.")

        encoded = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime_type};base64,{encoded}"
        completion = self._client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": data_url,
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
        )
        content = completion.choices[0].message.content
        if content is None:
            raise RuntimeError("The Vision response did not contain text content.")
        return content
