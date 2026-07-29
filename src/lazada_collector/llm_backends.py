"""Minimal provider adapters for the ABSA pseudo-labeling runner.

Only provider-neutral, non-streaming JSON generation is supported.  Secrets
are read from an explicitly named environment variable and are never returned
in metadata.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import requests


class LLMBackendError(RuntimeError):
    def __init__(
        self,
        message: str,
        *,
        retryable: bool,
        status_code: int | None = None,
        retry_after_seconds: float | None = None,
    ) -> None:
        super().__init__(message)
        self.retryable = retryable
        self.status_code = status_code
        self.retry_after_seconds = retry_after_seconds


@dataclass(frozen=True)
class GenerationRequest:
    backend: str
    endpoint: str
    model: str
    system_prompt: str
    user_message: str
    temperature: float = 0.0
    max_tokens: int = 4096
    seed: int | None = None
    api_key_env: str | None = None
    timeout_seconds: float = 180.0
    request_json_mode: bool = True
    reasoning_effort: str | None = None
    enable_thinking: bool | None = None


@dataclass(frozen=True)
class GenerationResponse:
    content: str
    provider_model: str
    finish_reason: str | None
    usage: dict[str, int | float | None]
    provider_request_id: str | None


def sanitized_endpoint(endpoint: str) -> str:
    """Return an endpoint safe for manifests and reject embedded credentials."""

    parsed = urlsplit(endpoint)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError(f"Invalid HTTP endpoint: {endpoint!r}")
    if parsed.username or parsed.password:
        raise ValueError("Endpoint must not contain embedded credentials")
    if parsed.query or parsed.fragment:
        raise ValueError("Endpoint must not contain query or fragment data")
    return urlunsplit(
        (parsed.scheme, parsed.netloc, parsed.path.rstrip("/"), "", "")
    )


def _retry_after(response: requests.Response) -> float | None:
    value = response.headers.get("Retry-After")
    if value is None:
        return None
    try:
        seconds = float(value)
    except ValueError:
        return None
    return max(0.0, seconds)


def _raise_http_error(response: requests.Response) -> None:
    status = response.status_code
    retryable = status in {408, 409, 425, 429} or status >= 500
    detail = response.text[:500].replace("\r", " ").replace("\n", " ")
    raise LLMBackendError(
        f"LLM HTTP {status}: {detail}",
        retryable=retryable,
        status_code=status,
        retry_after_seconds=_retry_after(response),
    )


def _usage_dict(value: Any) -> dict[str, int | float | None]:
    if not isinstance(value, dict):
        return {}
    safe: dict[str, int | float | None] = {}
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "prompt_eval_count",
        "eval_count",
        "total_duration",
        "load_duration",
        "prompt_eval_duration",
        "eval_duration",
    ):
        item = value.get(key)
        if isinstance(item, (int, float)) and not isinstance(item, bool):
            safe[key] = item
    return safe


def generate_json(
    request: GenerationRequest,
    *,
    session: requests.Session | None = None,
) -> GenerationResponse:
    if request.backend not in {"ollama", "openai_compatible"}:
        raise ValueError(f"Unsupported LLM backend: {request.backend}")
    if not request.model.strip():
        raise ValueError("A non-empty model name is required")
    endpoint = sanitized_endpoint(request.endpoint)
    if request.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if request.max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    if request.reasoning_effort not in {None, "low", "medium", "high"}:
        raise ValueError("reasoning_effort must be low, medium, high, or None")
    if request.enable_thinking not in {None, True, False}:
        raise ValueError("enable_thinking must be boolean or None")
    if request.timeout_seconds <= 0:
        raise ValueError("timeout_seconds must be positive")

    parsed_endpoint = urlsplit(endpoint)
    if (
        parsed_endpoint.scheme != "https"
        and parsed_endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
    ):
        raise LLMBackendError(
            "Refusing to send review text over non-HTTPS remote transport",
            retryable=False,
        )
    if session is None:
        with requests.Session() as client:
            if request.backend == "ollama":
                return _generate_ollama(client, endpoint, request)
            return _generate_openai_compatible(client, endpoint, request)
    client = session
    if request.backend == "ollama":
        return _generate_ollama(client, endpoint, request)
    return _generate_openai_compatible(client, endpoint, request)


def _generate_ollama(
    client: requests.Session,
    endpoint: str,
    request: GenerationRequest,
) -> GenerationResponse:
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_message},
        ],
        "stream": False,
        "options": {
            "temperature": request.temperature,
            "num_predict": request.max_tokens,
        },
    }
    if request.request_json_mode:
        payload["format"] = "json"
    if request.seed is not None:
        payload["options"]["seed"] = request.seed
    try:
        response = client.post(
            endpoint,
            json=payload,
            timeout=request.timeout_seconds,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise LLMBackendError(
            f"Ollama transport error: {exc}",
            retryable=True,
        ) from exc
    if response.status_code >= 300:
        _raise_http_error(response)
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMBackendError(
            "Ollama returned a non-JSON HTTP response",
            retryable=True,
            status_code=response.status_code,
        ) from exc
    if not isinstance(body, dict):
        raise LLMBackendError(
            "Ollama returned JSON that is not an object",
            retryable=True,
            status_code=response.status_code,
        )
    message = body.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LLMBackendError(
            "Ollama response has no message.content",
            retryable=True,
            status_code=response.status_code,
        )
    usage = _usage_dict(body)
    return GenerationResponse(
        content=content,
        provider_model=str(body.get("model") or request.model),
        finish_reason=str(body.get("done_reason"))
        if body.get("done_reason") is not None
        else None,
        usage=usage,
        provider_request_id=None,
    )


def _generate_openai_compatible(
    client: requests.Session,
    endpoint: str,
    request: GenerationRequest,
) -> GenerationResponse:
    headers = {
        "Accept": "application/json",
        "Content-Type": "application/json",
    }
    if request.api_key_env:
        parsed_endpoint = urlsplit(endpoint)
        if (
            parsed_endpoint.scheme != "https"
            and parsed_endpoint.hostname not in {"localhost", "127.0.0.1", "::1"}
        ):
            raise LLMBackendError(
                "Refusing to send an API key over non-HTTPS remote transport",
                retryable=False,
            )
        api_key = os.environ.get(request.api_key_env)
        if not api_key:
            raise LLMBackendError(
                f"Missing API key environment variable: {request.api_key_env}",
                retryable=False,
            )
        headers["Authorization"] = f"Bearer {api_key}"
    payload: dict[str, Any] = {
        "model": request.model,
        "messages": [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_message},
        ],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "stream": False,
    }
    if request.seed is not None:
        payload["seed"] = request.seed
    if request.request_json_mode:
        payload["response_format"] = {"type": "json_object"}
    if request.reasoning_effort is not None:
        payload["reasoning_effort"] = request.reasoning_effort
    if request.enable_thinking is not None:
        payload["chat_template_kwargs"] = {
            "enable_thinking": request.enable_thinking
        }
    try:
        response = client.post(
            endpoint,
            headers=headers,
            json=payload,
            timeout=request.timeout_seconds,
            allow_redirects=False,
        )
    except requests.RequestException as exc:
        raise LLMBackendError(
            f"OpenAI-compatible transport error: {exc}",
            retryable=True,
        ) from exc
    if response.status_code >= 300:
        _raise_http_error(response)
    try:
        body = response.json()
    except ValueError as exc:
        raise LLMBackendError(
            "OpenAI-compatible endpoint returned non-JSON HTTP",
            retryable=True,
            status_code=response.status_code,
        ) from exc
    if not isinstance(body, dict):
        raise LLMBackendError(
            "OpenAI-compatible endpoint returned JSON that is not an object",
            retryable=True,
            status_code=response.status_code,
        )
    choices = body.get("choices")
    if not isinstance(choices, list) or not choices:
        raise LLMBackendError(
            "OpenAI-compatible response has no choices",
            retryable=True,
            status_code=response.status_code,
        )
    choice = choices[0]
    message = choice.get("message") if isinstance(choice, dict) else None
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        raise LLMBackendError(
            "OpenAI-compatible response has no message.content",
            retryable=True,
            status_code=response.status_code,
        )
    return GenerationResponse(
        content=content,
        provider_model=str(body.get("model") or request.model),
        finish_reason=str(choice.get("finish_reason"))
        if choice.get("finish_reason") is not None
        else None,
        usage=_usage_dict(body.get("usage")),
        provider_request_id=str(body.get("id"))
        if body.get("id") is not None
        else None,
    )
