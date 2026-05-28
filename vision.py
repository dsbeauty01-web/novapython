"""
Nova vision — Gemini Flash 2.5 for "she sees me" observation.

Day 4 deliverable. Gemini Flash gets a webcam frame and returns
one warm, specific visual detail Nova can comment on.

Why Gemini instead of Claude Haiku:
- More permissive on images of people (Claude refused ~30% of the time)
- ~50% cheaper
- Faster first-token latency
- Free tier: 1500 req/day (plenty for current scale)
"""

import os
import base64
import logging
from typing import Optional

try:
    # NOTE: uses the older google-generativeai library (works fine today).
    # Google's newer library is google-genai — migrate only when this one
    # actually stops working; the API calls below would need rewriting.
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

logger = logging.getLogger("nova-vision")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = "gemini-flash-latest"  # always points at the newest Flash

if GEMINI_AVAILABLE and GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    _model = genai.GenerativeModel(GEMINI_MODEL)
else:
    _model = None


REFUSAL_PATTERNS = [
    "i can't",
    "i cannot",
    "i'm not able",
    "i am not able",
    "i'm unable",
    "as an ai",
    "sorry",
    "safety concerns",
    "can't engage",
    "skip",
]


def is_refusal(text: str) -> bool:
    """Detect if the LLM punted instead of describing."""
    t = text.lower().strip()
    return any(p in t[:50] for p in REFUSAL_PATTERNS) or len(t) > 150


VISION_PROMPT = """Look at this webcam frame. Notice ONE small gentle visual detail to mention warmly — like clothing color, an object behind the person, the lighting, the room, a toy, anything specific you actually see.

The voice is calm and soft, like a gentle big sister. Reply 5-12 words, lowercase, with "..." pauses. No preamble. No disclaimers. No exclamation marks.

If you cannot describe anything specific, reply exactly: SKIP

Good examples:
- "mhm... I love that blue hoodie..."
- "oh... is that a soft teddy behind you?"
- "such warm cozy light in there..."
- "I see your little room... so nice..."

Reply now with just the gentle observation:"""


async def observe_frame(image_bytes: bytes, media_type: str = "image/jpeg") -> Optional[str]:
    """Send a webcam frame to Gemini → return one warm visual observation, or None."""
    if not _model:
        logger.warning("Gemini not configured — vision disabled")
        return None

    try:
        # Use SDK's async generation
        response = await _model.generate_content_async(
            [
                VISION_PROMPT,
                {"mime_type": media_type, "data": image_bytes},
            ],
            generation_config={
                "max_output_tokens": 60,
                "temperature": 0.7,
            },
        )
        text = (response.text or "").strip()
        # strip surrounding quotes
        text = text.strip('"\'')
        
        if is_refusal(text):
            logger.info(f"vision refusal/skip: '{text[:60]}'")
            return None
        
        logger.info(f"vision observation: '{text}'")
        return text
    except Exception as e:
        logger.error(f"vision error: {e}")
        return None


async def observe_from_data_url(data_url: str) -> Optional[str]:
    """Convenience: accept a data:image/...;base64,... URL and observe it."""
    try:
        header, b64 = data_url.split(",", 1)
        # extract media type from data:image/jpeg;base64
        media_type = "image/jpeg"
        if "image/" in header:
            media_type = header.split("data:")[1].split(";")[0]
        image_bytes = base64.b64decode(b64)
        return await observe_frame(image_bytes, media_type)
    except Exception as e:
        logger.error(f"vision parse error: {e}")
        return None
