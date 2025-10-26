from typing import Optional
import logging
import time
import os
import requests
from config import settings

logger = logging.getLogger("guidefarm.llm")

class LLMService:
    """LLM service abstraction. Supports providers: openai|mistral|local fallback.

    Mistral integration uses the environment variable MISTRAL_API_KEY and optional
    MISTRAL_API_URL (if you host a custom endpoint). The implementation is conservative
    and tolerant to different possible response shapes from the provider. All secrets
    must remain in environment variables; do NOT hardcode API keys.
    """
    def __init__(self):
        self.provider = settings.LLM_PROVIDER or "mistral"
        self.mistral_key = settings.MISTRAL_API_KEY
        # Default model and endpoint; can be overridden via env MISTRAL_API_URL
        self.mistral_model = os.environ.get("MISTRAL_MODEL", "mistral-medium-latest")
        self.mistral_url = os.environ.get("MISTRAL_API_URL",
                                         f"https://api.mistral.ai/v1/models/{{self.mistral_model}}/completions")

    def generate_markdown(self, topic: str, lang: str = "ru", provider: Optional[str] = None) -> str:
        provider = provider or self.provider
        logger.info({"event": "llm_call", "provider": provider, "topic": topic})
        if provider == "mistral" and self.mistral_key:
            try:
                return self._call_mistral(topic, lang)
            except Exception:
                logger.exception("Mistral call failed, falling back to local generator")
        # Fallback to local template
        return self._local_markdown(topic)

    def _call_mistral(self, topic: str, lang: str = "ru") -> str:
        """Call Mistral hosted API with simple retries and tolerant parsing.

        The function does not assume a rigid response schema; it tries to find
        textual content in several common places ("output", "outputs", "choices",
        "text", "content"). This makes the code more robust to API changes.
        """
        prompt = self._build_prompt(topic, lang)
        headers = {
            "Authorization": f"Bearer {{self.mistral_key}}",
            "Content-Type": "application/json",
        }
        payload = {
            # conservative payload compatible with common LLM-hosting shapes
            "input": prompt,
            "max_tokens": 2000,
            "temperature": 0.7,
        }
        # simple retry
        last_exc = None
        for attempt in range(3):
            try:
                resp = requests.post(self.mistral_url, headers=headers, json=payload, timeout=20)
                if resp.status_code != 200:
                    logger.warning({"event": "mistral_non_200", "status": resp.status_code, "body": resp.text[:200]})
                    time.sleep(1 + attempt * 2)
                    continue
                data = resp.json()
                text = self._extract_text_from_mistral_response(data)
                if text:
                    return text
                # nothing found; break to fallback
                logger.warning({"event": "mistral_no_text", "data": data})
                break
            except Exception as exc:
                last_exc = exc
                logger.exception("Mistral request error, retrying")
                time.sleep(1 + attempt * 2)
        if last_exc:
            raise last_exc
        raise RuntimeError("Mistral returned no usable text")

    def _extract_text_from_mistral_response(self, data: dict) -> Optional[str]:
        # Common keys to try
        candidates = []
        if not isinstance(data, dict):
            return None
        # Some providers return {"outputs": [{"content": "..."}]}
        if "outputs" in data and isinstance(data["outputs"], list):
            for o in data["outputs"]:
                if isinstance(o, dict):
                    for k in ("content", "text", "output", "message"): 
                        if k in o and isinstance(o[k], str):
                            candidates.append(o[k])
                elif isinstance(o, str):
                    candidates.append(o)
        # Some return {"output": "..."}
        if "output" in data and isinstance(data["output"], str):
            candidates.append(data["output"])
        # OpenAI‑like: {choices: [{text: "..."}]}
        if "choices" in data and isinstance(data["choices"], list):
            for c in data["choices"]:
                if isinstance(c, dict) and isinstance(c.get("text"), str):
                    candidates.append(c.get("text"))
                if isinstance(c, dict) and isinstance(c.get("message"), dict):
                    m = c.get("message")
                    if isinstance(m.get("content"), str):
                        candidates.append(m.get("content"))
        # direct text
        for key in ("text", "content", "result", "message"): 
            if key in data and isinstance(data[key], str):
                candidates.append(data[key])
        # Join candidates by longest
        if not candidates:
            return None
        # Prefer the longest candidate
        candidates = [c.strip() for c in candidates if c and isinstance(c, str)]
        candidates.sort(key=lambda s: len(s), reverse=True)
        return candidates[0]

    def _build_prompt(self, topic: str, lang: str) -> str:
        return f"Сгенерируй подробный guide по теме: {{topic}} в markdown (титул, оглавление, 5 глав, чек-листы)."

    def _local_markdown(self, topic: str) -> str:
        logger.info({"event": "llm_local_fallback", "topic": topic})
        md = f"# {{topic}}\n\n_Premium #DOBRO guide generated locally_\n\n## Оглавление\n\n1. Введение\n2. План\n3. Шаги\n4. Примеры\n5. Чек-лист\n\n## Введение\n\nТема: {{topic}}\n\n## План\n\n- Шаг 1\n- Шаг 2\n\n## Чек-лист\n\n- [ ] Протестировать\n- [ ] Опубликовать\n\n---\n\n*Generated by GuideFarm (local fallback).*"
        return md
