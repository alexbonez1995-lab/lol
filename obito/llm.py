"""Anbindung lokaler Sprachmodelle (100 % offline, nur Standardbibliothek).

Unterstützte Backends:

* :class:`OllamaBackend`       – Ollama (http://127.0.0.1:11434), native API
* :class:`OpenAICompatBackend` – llama.cpp-Server, LM Studio, vLLM, text-generation-webui
                                 (OpenAI-kompatible API unter ``/v1``)
* :class:`FakeBackend`         – deterministisches Testdouble ohne Netzwerk

Alle Backends teilen die Schnittstelle :class:`LLMBackend`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

from .config import Config

StreamCallback = Callable[[str], None]

# Lokale Server dürfen nie über einen System-Proxy angesprochen werden.
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


class LLMError(Exception):
    """Allgemeiner Fehler der Modell-Anbindung."""


class BackendUnavailable(LLMError):
    """Der Modell-Server ist nicht erreichbar."""


class ModelNotFound(LLMError):
    """Das angeforderte Modell ist nicht installiert."""


@dataclass
class ModelInfo:
    name: str
    size: int | None = None
    family: str | None = None
    parameters: str | None = None
    quantization: str | None = None

    def short(self) -> str:
        extra = ", ".join(x for x in (self.parameters, self.quantization) if x)
        size = f" · {self.size / 1e9:.1f} GB" if self.size else ""
        return f"{self.name}" + (f" ({extra})" if extra else "") + size


@dataclass
class ChatResult:
    text: str
    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration: float = 0.0
    raw: dict = field(default_factory=dict)

    @property
    def total_tokens(self) -> int:
        return self.prompt_tokens + self.completion_tokens


def parse_json(text: str) -> Any:
    """Zieht robust ein JSON-Objekt/-Array aus einer Modellantwort (auch mit Code-Zäunen/Prosa)."""
    if text is None:
        return None
    s = text.strip()
    if not s:
        return None
    fence = re.search(r"```(?:json)?\s*(.*?)```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    # erstes { oder [ bis zum passenden Gegenstück
    for opener, closer in (("{", "}"), ("[", "]")):
        start = s.find(opener)
        if start == -1:
            continue
        depth = 0
        in_str = False
        esc = False
        for i in range(start, len(s)):
            ch = s[i]
            if in_str:
                if esc:
                    esc = False
                elif ch == "\\":
                    esc = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(s[start:i + 1])
                    except json.JSONDecodeError:
                        break
    return None


# ------------------------------------------------------------------ Basis
class LLMBackend:
    name = "basis"
    default_model: str = ""
    default_embed_model: str = ""
    timeout: float = 300.0

    def available(self) -> bool:
        raise NotImplementedError

    def chat(
        self,
        messages: Sequence[dict],
        *,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
        stop: Sequence[str] | None = None,
        stream: StreamCallback | None = None,
        timeout: float | None = None,
    ) -> ChatResult:
        raise NotImplementedError

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]] | None:
        raise NotImplementedError

    def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    def has_model(self, name: str) -> bool:
        wanted = name.strip()
        for m in self.list_models():
            if m.name == wanted or m.name.split(":")[0] == wanted.split(":")[0] and ":" not in wanted:
                return True
        return False

    def pull(self, name: str, progress: Callable[[dict], None] | None = None) -> bool:
        return False

    def info(self) -> dict:
        return {"backend": self.name, "modell": self.default_model, "embedding": self.default_embed_model}


# ---------------------------------------------------------------- HTTP-Hilfe
def _request(method: str, url: str, body: dict | None, timeout: float, stream: bool = False):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/x-ndjson, text/event-stream, application/json")
    try:
        resp = _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")
        msg = payload
        try:
            j = json.loads(payload)
            msg = j.get("error") if isinstance(j, dict) else payload
            if isinstance(msg, dict):
                msg = msg.get("message", payload)
        except json.JSONDecodeError:
            pass
        if e.code == 404 and "not found" in str(msg).lower():
            raise ModelNotFound(str(msg)) from None
        raise LLMError(f"HTTP {e.code}: {msg}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise BackendUnavailable(f"Modell-Server nicht erreichbar ({url}): {e}") from None
    if stream:
        return resp
    with resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload) if payload else {}


# ---------------------------------------------------------------- Ollama
class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, base_url: str = "", model: str = "", embed_model: str = "", timeout: float = 300.0):
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.default_model = model
        self.default_embed_model = embed_model
        self.timeout = timeout

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def available(self) -> bool:
        try:
            _request("GET", self._url("/api/tags"), None, timeout=3)
            return True
        except LLMError:
            return False

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, json_mode=False,
             stop=None, stream=None, timeout=None) -> ChatResult:
        model = model or self.default_model
        if not model:
            raise LLMError("Kein Modell konfiguriert.")
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = max_tokens
        if stop:
            options["stop"] = list(stop)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": stream is not None,
            "options": options,
        }
        if json_mode:
            body["format"] = "json"
        t0 = time.time()
        timeout = timeout or self.timeout
        if stream is None:
            data = _request("POST", self._url("/api/chat"), body, timeout)
            text = (data.get("message") or {}).get("content", "")
            return ChatResult(
                text=text, model=data.get("model", model),
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
                duration=time.time() - t0, raw=data,
            )
        resp = _request("POST", self._url("/api/chat"), body, timeout, stream=True)
        parts: list[str] = []
        last: dict = {}
        with resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMError(str(chunk["error"]))
                piece = (chunk.get("message") or {}).get("content", "")
                if piece:
                    parts.append(piece)
                    stream(piece)
                if chunk.get("done"):
                    last = chunk
        return ChatResult(
            text="".join(parts), model=last.get("model", model),
            prompt_tokens=int(last.get("prompt_eval_count") or 0),
            completion_tokens=int(last.get("eval_count") or 0),
            duration=time.time() - t0, raw=last,
        )

    def embed(self, texts, *, model=None):
        model = model or self.default_embed_model
        if not model or not texts:
            return None
        try:
            data = _request("POST", self._url("/api/embed"), {"model": model, "input": list(texts)}, self.timeout)
            vecs = data.get("embeddings")
            if vecs and len(vecs) == len(texts):
                return [list(map(float, v)) for v in vecs]
        except ModelNotFound:
            return None
        except LLMError:
            pass
        # Ältere Ollama-Versionen: ein Text pro Anfrage
        out: list[list[float]] = []
        for t in texts:
            try:
                data = _request("POST", self._url("/api/embeddings"), {"model": model, "prompt": t}, self.timeout)
            except LLMError:
                return None
            vec = data.get("embedding")
            if not vec:
                return None
            out.append(list(map(float, vec)))
        return out

    def list_models(self) -> list[ModelInfo]:
        data = _request("GET", self._url("/api/tags"), None, timeout=10)
        out = []
        for m in data.get("models", []):
            d = m.get("details") or {}
            out.append(ModelInfo(
                name=m.get("name", ""), size=m.get("size"), family=d.get("family"),
                parameters=d.get("parameter_size"), quantization=d.get("quantization_level"),
            ))
        return out

    def pull(self, name: str, progress=None) -> bool:
        resp = _request("POST", self._url("/api/pull"), {"name": name, "stream": True}, 3600, stream=True)
        ok = False
        with resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line:
                    continue
                chunk = json.loads(line)
                if chunk.get("error"):
                    raise LLMError(str(chunk["error"]))
                if progress:
                    progress(chunk)
                if chunk.get("status") == "success":
                    ok = True
        return ok

    def info(self) -> dict:
        d = super().info()
        d["url"] = self.base_url
        return d


# ------------------------------------------------------- OpenAI-kompatibel
class OpenAICompatBackend(LLMBackend):
    """llama.cpp-Server (``llama-server``), LM Studio, vLLM, Jan, text-generation-webui …"""

    name = "openai"

    def __init__(self, base_url: str = "", model: str = "", embed_model: str = "",
                 timeout: float = 300.0, api_key: str = "lokal"):
        self.base_url = (base_url or "http://127.0.0.1:8080/v1").rstrip("/")
        self.default_model = model
        self.default_embed_model = embed_model
        self.timeout = timeout
        self.api_key = api_key

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def _call(self, method: str, path: str, body: dict | None, timeout: float, stream: bool = False):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self._url(path), data=data, method=method)
        req.add_header("Content-Type", "application/json")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        try:
            resp = _OPENER.open(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            payload = e.read().decode("utf-8", "replace")
            if e.code == 404:
                raise ModelNotFound(payload) from None
            raise LLMError(f"HTTP {e.code}: {payload}") from None
        except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
            raise BackendUnavailable(f"Modell-Server nicht erreichbar ({self.base_url}): {e}") from None
        if stream:
            return resp
        with resp:
            payload = resp.read().decode("utf-8")
        return json.loads(payload) if payload else {}

    def available(self) -> bool:
        try:
            self._call("GET", "/models", None, timeout=3)
            return True
        except LLMError:
            return False

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, json_mode=False,
             stop=None, stream=None, timeout=None) -> ChatResult:
        model = model or self.default_model
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": stream is not None,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = max_tokens
        if stop:
            body["stop"] = list(stop)
        if json_mode:
            body["response_format"] = {"type": "json_object"}
        t0 = time.time()
        timeout = timeout or self.timeout
        if stream is None:
            data = self._call("POST", "/chat/completions", body, timeout)
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            return ChatResult(
                text=(choice.get("message") or {}).get("content") or "",
                model=data.get("model", model),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                duration=time.time() - t0, raw=data,
            )
        resp = self._call("POST", "/chat/completions", body, timeout, stream=True)
        parts: list[str] = []
        usage: dict = {}
        with resp:
            for line in resp:
                line = line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        parts.append(piece)
                        stream(piece)
        return ChatResult(
            text="".join(parts), model=model,
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            duration=time.time() - t0,
        )

    def embed(self, texts, *, model=None):
        model = model or self.default_embed_model
        if not model or not texts:
            return None
        try:
            data = self._call("POST", "/embeddings", {"model": model, "input": list(texts)}, self.timeout)
        except LLMError:
            return None
        items = sorted(data.get("data") or [], key=lambda d: d.get("index", 0))
        if len(items) != len(texts):
            return None
        return [list(map(float, d["embedding"])) for d in items]

    def list_models(self) -> list[ModelInfo]:
        data = self._call("GET", "/models", None, timeout=10)
        return [ModelInfo(name=m.get("id", "")) for m in data.get("data") or []]

    def info(self) -> dict:
        d = super().info()
        d["url"] = self.base_url
        return d


# ------------------------------------------------------------------ Fake
class FakeBackend(LLMBackend):
    """Testdouble ohne Netzwerk.

    ``responder(messages, kwargs) -> str`` liefert die Antwort; alternativ wird eine
    Warteschlange aus ``responses`` abgearbeitet (``push``). Ohne beides wird der letzte
    Nutzertext gespiegelt. Alle Aufrufe werden in ``calls`` protokolliert.
    """

    name = "fake"

    def __init__(self, responder: Callable[[list[dict], dict], str] | None = None,
                 responses: Sequence[str] | None = None, model: str = "fake-modell",
                 embed_dim: int = 16, models: Sequence[str] | None = None, up: bool = True):
        self.responder = responder
        self.responses: list[str] = list(responses or [])
        self.default_model = model
        self.default_embed_model = "fake-embed"
        self.embed_dim = embed_dim
        self.models = list(models or [model, "fake-embed"])
        self.calls: list[dict] = []
        self.up = up

    def push(self, *responses: str) -> None:
        self.responses.extend(responses)

    def available(self) -> bool:
        return self.up

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, json_mode=False,
             stop=None, stream=None, timeout=None) -> ChatResult:
        if not self.up:
            raise BackendUnavailable("Fake-Backend ist abgeschaltet.")
        kwargs = {"model": model or self.default_model, "temperature": temperature,
                  "max_tokens": max_tokens, "json_mode": json_mode, "stop": stop}
        msgs = [dict(m) for m in messages]
        self.calls.append({"messages": msgs, **kwargs})
        if self.responder is not None:
            text = self.responder(msgs, kwargs)
        elif self.responses:
            text = self.responses.pop(0)
        else:
            users = [m["content"] for m in msgs if m["role"] == "user"]
            text = f"Echo: {users[-1] if users else ''}"
        text = text or ""
        if stream is not None:
            for i in range(0, len(text), 12):
                stream(text[i:i + 12])
        ptoks = sum(len(m["content"].split()) for m in msgs)
        return ChatResult(text=text, model=kwargs["model"], prompt_tokens=ptoks,
                          completion_tokens=len(text.split()), duration=0.0)

    def embed(self, texts, *, model=None):
        """Deterministische Wortsack-Vektoren: ähnliche Wörter -> ähnliche Vektoren."""
        if not self.up:
            return None
        out = []
        for t in texts:
            vec = [0.0] * self.embed_dim
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                vec[h % self.embed_dim] += 1.0
            n = math.sqrt(sum(x * x for x in vec)) or 1.0
            out.append([x / n for x in vec])
        return out

    def list_models(self) -> list[ModelInfo]:
        return [ModelInfo(name=m) for m in self.models]

    def pull(self, name, progress=None) -> bool:
        self.models.append(name)
        return True


# --------------------------------------------------------------- Fabrik
def make_backend(cfg: Config) -> LLMBackend:
    """Erzeugt das konfigurierte Backend. ``auto`` wählt Ollama, sonst einen OpenAI-kompatiblen Server."""
    kind = (cfg.backend or "auto").lower()
    if kind == "fake":
        return FakeBackend(model=cfg.model or "fake-modell")
    if kind == "ollama":
        return OllamaBackend(cfg.base_url, cfg.model, cfg.embed_model, cfg.timeout)
    if kind in ("openai", "llamacpp", "lmstudio", "vllm"):
        return OpenAICompatBackend(cfg.base_url, cfg.model, cfg.embed_model, cfg.timeout)
    if kind != "auto":
        raise LLMError(f"Unbekanntes Backend: {cfg.backend!r} (erlaubt: auto, ollama, openai, fake)")
    ollama = OllamaBackend(cfg.base_url if "11434" in cfg.base_url else "", cfg.model, cfg.embed_model, cfg.timeout)
    if ollama.available():
        return ollama
    compat = OpenAICompatBackend(cfg.base_url if "11434" not in cfg.base_url else "", cfg.model,
                                 cfg.embed_model, cfg.timeout)
    if compat.available():
        return compat
    return ollama  # Standard; Aufrufer prüft available() und gibt eine klare Meldung aus
