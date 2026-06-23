"""
Nova vision — Gemini Flash for "she sees me" observation.

Gets a webcam frame and returns one warm, specific visual detail Nova can
comment on.

Why Gemini instead of Claude Haiku:
- More permissive on images of people (Claude refused ~30% of the time)
- ~50% cheaper
- Faster first-token latency
- Free tier: 1500 req/day (plenty for current scale)

Uses the CURRENT google-genai SDK (`from google import genai`); the old
google-generativeai package is deprecated. Import + calls are fully guarded so
a missing package or API mismatch disables vision rather than crashing the
worker / web service that import this module.
"""

import os
import base64
import logging
from typing import Optional

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except Exception:  # ImportError or any load issue
    GEMINI_AVAILABLE = False

logger = logging.getLogger("nova-vision")

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
# google-genai uses concrete model ids; "gemini-flash-latest" alias may not
# resolve here, so default to a known-good Flash. Env-overridable.
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")

_client = None
if GEMINI_AVAILABLE and GEMINI_API_KEY:
    try:
        _client = genai.Client(api_key=GEMINI_API_KEY)
        logger.info(f"vision: google-genai client ready (model={GEMINI_MODEL})")
    except Exception as e:
        logger.error(f"vision: genai.Client init failed — vision disabled: {e}")
        _client = None
else:
    logger.warning(f"vision: disabled (available={GEMINI_AVAILABLE}, key={'set' if GEMINI_API_KEY else 'MISSING'})")


REFUSAL_PATTERNS = [
    "i can't", "i cannot", "i'm not able", "i am not able", "i'm unable",
    "as an ai", "sorry", "safety concerns", "can't engage", "skip",
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
    if not _client:
        logger.warning("Gemini not configured — vision disabled")
        return None

    try:
        response = await _client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[
                types.Part.from_bytes(data=image_bytes, mime_type=media_type),
                VISION_PROMPT,
            ],
            config=types.GenerateContentConfig(
                max_output_tokens=60,
                temperature=0.7,
            ),
        )
        text = (response.text or "").strip().strip('"\'')

        if is_refusal(text):
            logger.info(f"vision refusal/skip: '{text[:60]}'")
            return None

        logger.info(f"vision observation: '{text}'")
        return text
    except Exception as e:
        logger.error(f"vision error: {e}")
        return None


AGE_PROMPT = (
    "This is a webcam frame of a person. Estimate their approximate age range. "
    "Respond with ONLY one letter: a = young child 3-7, b = older child 8-12, "
    "c = teen 13-17, d = adult 18+. If unsure, respond b."
)

async def estimate_age(data_url: str) -> str:
    """Return a single age-range letter a/b/c/d from a webcam frame (defaults 'b')."""
    if not _client:
        return "b"
    try:
        header, b64 = data_url.split(",", 1)
        media_type = "image/jpeg"
        if "image/" in header:
            media_type = header.split("data:")[1].split(";")[0]
        image_bytes = base64.b64decode(b64)
        response = await _client.aio.models.generate_content(
            model=GEMINI_MODEL,
            contents=[types.Part.from_bytes(data=image_bytes, mime_type=media_type), AGE_PROMPT],
            config=types.GenerateContentConfig(max_output_tokens=4, temperature=0.0),
        )
        t = (response.text or "").strip().lower()
        for ch in t:
            if ch in "abcd":
                logger.info(f"age estimate: {ch}")
                return ch
        return "b"
    except Exception as e:
        logger.error(f"age estimate error: {e}")
        return "b"


async def observe_from_data_url(data_url: str) -> Optional[str]:
    """Convenience: accept a data:image/...;base64,... URL and observe it."""
    try:
        header, b64 = data_url.split(",", 1)
        media_type = "image/jpeg"
        if "image/" in header:
            media_type = header.split("data:")[1].split(";")[0]
        image_bytes = base64.b64decode(b64)
        return await observe_frame(image_bytes, media_type)
    except Exception as e:
        logger.error(f"vision parse error: {e}")
        return None
