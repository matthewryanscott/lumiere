"""Ollama: a local speech model for text-to-speech.

`phonon` runs on whatever host `OLLAMA_HOST` names, with no API key and no
account: the request carries the text, the model, and optional voice, and the
answer is a WAV. The whole text goes in one request, so it is capped at
`INPUT_MAX` characters: text over the cap is bad input, not a provider
failure, because no request was made.
"""

from __future__ import annotations

import os

import requests

DEFAULT_MODEL = "phonon"
INPUT_MAX = 4096
HOST_DEFAULT = "http://localhost:11434"


def host() -> str:
    """The speech host: `OLLAMA_HOST` from the environment, with a scheme
    added when it has none and a trailing slash cut."""
    raw = os.environ.get("OLLAMA_HOST", "").strip() or HOST_DEFAULT
    if "://" not in raw:
        raw = f"http://{raw}"
    return raw.rstrip("/")


class TooLong(ValueError):
    """Text over `INPUT_MAX` characters: bad input, not a provider failure,
    because no request was made."""


def speech_request(text: str, voice: str | None, model: str) -> dict:
    """The body of one speech request; the voice is left to the model when
    none is given."""
    if len(text) > INPUT_MAX:
        raise TooLong(
            f"{len(text)} characters is over the {INPUT_MAX} a speech request takes"
        )
    body: dict = {"model": model, "input": text, "response_format": "wav"}
    if voice:
        body["voice"] = voice
    return body


def post_speech(request: dict, host: str) -> bytes:
    """One speech request with the given body; the WAV bytes it returns."""
    response = requests.post(
        f"{host}/v1/audio/speech", json=request, timeout=300,
    )
    response.raise_for_status()
    return response.content