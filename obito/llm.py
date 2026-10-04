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

    def to_dict(self) -> dict:
        return {"name": self.name, "groesse": self.size, "familie": self.family,
                "parameter": self.parameters, "quantisierung": self.quantization}


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


# ------------------------------------------------------------ JSON-Hilfen
def parse_json(text: str) -> Any:
    """Zieht robust ein JSON-Objekt/-Array aus einer Modellantwort (auch mit Code-Zäunen/Prosa)."""
    if text is None:
        return None
    s = strip_thinking(text).strip()
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


# ------------------------------------------------------- Denk-Modelle
_THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL | re.IGNORECASE)
_THINK_OPEN_RE = re.compile(r"<think>.*\Z", re.DOTALL | re.IGNORECASE)


def strip_thinking(text: str) -> str:
    """Entfernt ``<think>…</think>``-Blöcke (qwen3, deepseek-r1 …), auch unvollständig geöffnete."""
    if not text or "<think" not in text.lower():
        return text or ""
    text = _THINK_RE.sub("", text)
    text = _THINK_OPEN_RE.sub("", text)
    return text.lstrip()


class ThinkFilter:
    """Streaming-Filter, der Text innerhalb von ``<think>…</think>`` zurückhält."""

    OPEN = "<think>"
    CLOSE = "</think>"

    def __init__(self, emit: StreamCallback | None):
        self.emit = emit
        self.buf = ""
        self.inside = False
        self.text_parts: list[str] = []

    def feed(self, piece: str) -> None:
        if not piece:
            return
        self.text_parts.append(piece)
        if self.emit is None:
            return
        self.buf += piece
        while self.buf:
            if self.inside:
                idx = self.buf.lower().find(self.CLOSE)
                if idx == -1:
                    self.buf = self.buf[-(len(self.CLOSE) - 1):] if len(self.buf) >= len(self.CLOSE) else self.buf
                    return
                self.buf = self.buf[idx + len(self.CLOSE):].lstrip()
                self.inside = False
                continue
            idx = self.buf.lower().find(self.OPEN)
            if idx != -1:
                if idx:
                    self.emit(self.buf[:idx])
                self.buf = self.buf[idx + len(self.OPEN):]
                self.inside = True
                continue
            # Ende könnte Präfix von <think> sein -> zurückhalten
            keep = 0
            low = self.buf.lower()
            for n in range(min(len(self.OPEN) - 1, len(low)), 0, -1):
                if self.OPEN.startswith(low[-n:]):
                    keep = n
                    break
            if len(self.buf) - keep > 0:
                self.emit(self.buf[:len(self.buf) - keep])
            self.buf = self.buf[len(self.buf) - keep:]
            return

    def finish(self) -> str:
        if self.emit is not None and self.buf and not self.inside:
            self.emit(self.buf)
        self.buf = ""
        return strip_thinking("".join(self.text_parts))


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
        json_mode: bool | dict = False,
        stop: Sequence[str] | None = None,
        stream: StreamCallback | None = None,
        timeout: float | None = None,
        num_ctx: int | None = None,
        seed: int | None = None,
        keep_alive: str | None = None,
        think: bool | None = None,
    ) -> ChatResult:
        """Chat-Aufruf.

        ``json_mode``: ``True`` = nur gültiges JSON; ``dict`` = JSON-Schema (Ollama ≥ 0.5,
        llama-server, LM Studio, vLLM). ``num_ctx``/``keep_alive``/``think`` gelten für Ollama,
        andere Backends ignorieren sie. ``timeout`` = Sekunden ohne Daten (bei Streaming
        effektiv ein Leerlauf-Timeout).
        """
        raise NotImplementedError

    def embed(self, texts: Sequence[str], *, model: str | None = None) -> list[list[float]] | None:
        raise NotImplementedError

    def list_models(self) -> list[ModelInfo]:
        raise NotImplementedError

    def has_model(self, name: str) -> bool:
        wanted = name.strip()
        for m in self.list_models():
            if m.name == wanted or (":" not in wanted and m.name.split(":")[0] == wanted):
                return True
        return False

    def pull(self, name: str, progress: Callable[[dict], None] | None = None) -> bool:
        return False

    def running(self) -> list[dict]:
        """Geladene Modelle (Ollama: ``/api/ps``) – leer, wenn nicht unterstützt."""
        return []

    def version(self) -> str | None:
        return None

    def info(self) -> dict:
        return {"backend": self.name, "modell": self.default_model, "embedding": self.default_embed_model}


# ---------------------------------------------------------------- HTTP-Hilfe
def _request(method: str, url: str, body: dict | None, timeout: float, stream: bool = False,
             headers: dict | None = None):
    data = json.dumps(body).encode("utf-8") if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Content-Type", "application/json")
    req.add_header("Accept", "application/x-ndjson, text/event-stream, application/json")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        resp = _OPENER.open(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        payload = e.read().decode("utf-8", "replace")
        msg: Any = payload
        try:
            j = json.loads(payload)
            msg = j.get("error") if isinstance(j, dict) else payload
            if isinstance(msg, dict):
                msg = msg.get("message", payload)
        except json.JSONDecodeError:
            pass
        if e.code == 404 and "not found" in str(msg).lower():
            raise ModelNotFound(str(msg)) from None
        if e.code == 404 and method == "GET":
            raise LLMError(f"HTTP 404: {url}") from None
        raise LLMError(f"HTTP {e.code}: {msg}") from None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as e:
        raise BackendUnavailable(f"Modell-Server nicht erreichbar ({url}): {e}") from None
    if stream:
        return resp
    with resp:
        payload = resp.read().decode("utf-8")
    return json.loads(payload) if payload else {}


def _read_stream(resp, timeout_hint: float):
    """Liest Zeilen eines Streams; Socket-Fehler werden als LLMError gemeldet."""
    try:
        with resp:
            for line in resp:
                yield line.decode("utf-8", "replace").strip()
    except (TimeoutError, OSError) as e:
        raise BackendUnavailable(f"Verbindung zum Modell-Server abgebrochen (Timeout {timeout_hint:.0f} s): {e}") from None


# ---------------------------------------------------------------- Ollama
class OllamaBackend(LLMBackend):
    name = "ollama"

    def __init__(self, base_url: str = "", model: str = "", embed_model: str = "", timeout: float = 300.0,
                 keep_alive: str | None = None):
        self.base_url = (base_url or "http://127.0.0.1:11434").rstrip("/")
        self.default_model = model
        self.default_embed_model = embed_model
        self.timeout = timeout
        self.keep_alive = keep_alive

    def _url(self, path: str) -> str:
        return f"{self.base_url}{path}"

    def available(self) -> bool:
        try:
            _request("GET", self._url("/api/tags"), None, timeout=3)
            return True
        except LLMError:
            return False

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, json_mode=False,
             stop=None, stream=None, timeout=None, num_ctx=None, seed=None, keep_alive=None,
             think=None) -> ChatResult:
        model = model or self.default_model
        if not model:
            raise LLMError("Kein Modell konfiguriert.")
        options: dict[str, Any] = {}
        if temperature is not None:
            options["temperature"] = temperature
        if max_tokens:
            options["num_predict"] = int(max_tokens)
        if stop:
            options["stop"] = list(stop)
        if num_ctx:
            options["num_ctx"] = int(num_ctx)
        if seed is not None:
            options["seed"] = int(seed)
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": stream is not None,
            "options": options,
        }
        if json_mode:
            body["format"] = json_mode if isinstance(json_mode, dict) else "json"
        ka = keep_alive or self.keep_alive
        if ka:
            body["keep_alive"] = ka
        if think is not None:
            body["think"] = bool(think)
        t0 = time.time()
        timeout = timeout or self.timeout
        if stream is None:
            data = _request("POST", self._url("/api/chat"), body, timeout)
            text = strip_thinking((data.get("message") or {}).get("content", ""))
            return ChatResult(
                text=text, model=data.get("model", model),
                prompt_tokens=int(data.get("prompt_eval_count") or 0),
                completion_tokens=int(data.get("eval_count") or 0),
                duration=time.time() - t0, raw=data,
            )
        resp = _request("POST", self._url("/api/chat"), body, timeout, stream=True)
        filt = ThinkFilter(stream)
        last: dict = {}
        for line in _read_stream(resp, timeout):
            if not line:
                continue
            chunk = json.loads(line)
            if chunk.get("error"):
                raise LLMError(str(chunk["error"]))
            filt.feed((chunk.get("message") or {}).get("content", ""))
            if chunk.get("done"):
                last = chunk
        text = filt.finish()
        return ChatResult(
            text=text, model=last.get("model", model),
            prompt_tokens=int(last.get("prompt_eval_count") or 0),
            completion_tokens=int(last.get("eval_count") or 0),
            duration=time.time() - t0, raw=last,
        )

    def embed(self, texts, *, model=None):
        model = model or self.default_embed_model
        if not model or not texts:
            return None
        body: dict[str, Any] = {"model": model, "input": list(texts)}
        if self.keep_alive:
            body["keep_alive"] = self.keep_alive
        try:
            data = _request("POST", self._url("/api/embed"), body, self.timeout)
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
        for line in _read_stream(resp, 3600):
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

    def running(self) -> list[dict]:
        try:
            data = _request("GET", self._url("/api/ps"), None, timeout=5)
        except LLMError:
            return []
        return [{"name": m.get("name"), "size": m.get("size"), "size_vram": m.get("size_vram"),
                 "expires_at": m.get("expires_at")} for m in data.get("models", [])]

    def version(self) -> str | None:
        try:
            return _request("GET", self._url("/api/version"), None, timeout=5).get("version")
        except LLMError:
            return None

    def show(self, name: str) -> dict:
        """Modelldetails (``/api/show``), z. B. ``capabilities`` (``thinking``, ``tools``)."""
        try:
            return _request("POST", self._url("/api/show"), {"model": name}, 10)
        except LLMError:
            return {}

    def info(self) -> dict:
        d = super().info()
        d["url"] = self.base_url
        d["version"] = self.version()
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
        try:
            return _request(method, self._url(path), body, timeout, stream,
                            headers={"Authorization": f"Bearer {self.api_key}"})
        except BackendUnavailable:
            raise
        except ModelNotFound:
            raise
        except LLMError as e:
            if "HTTP 404" in str(e) and method == "POST":
                raise ModelNotFound(str(e)) from None
            raise

    def available(self) -> bool:
        try:
            self._call("GET", "/models", None, timeout=3)
            return True
        except LLMError:
            return False

    def chat(self, messages, *, model=None, temperature=None, max_tokens=None, json_mode=False,
             stop=None, stream=None, timeout=None, num_ctx=None, seed=None, keep_alive=None,
             think=None) -> ChatResult:
        model = model or self.default_model
        body: dict[str, Any] = {
            "model": model,
            "messages": [{"role": m["role"], "content": m["content"]} for m in messages],
            "stream": stream is not None,
        }
        if temperature is not None:
            body["temperature"] = temperature
        if max_tokens:
            body["max_tokens"] = int(max_tokens)
        if stop:
            body["stop"] = list(stop)
        if seed is not None:
            body["seed"] = int(seed)
        if isinstance(json_mode, dict):
            body["response_format"] = {"type": "json_schema",
                                       "json_schema": {"name": "antwort", "schema": json_mode}}
        elif json_mode:
            body["response_format"] = {"type": "json_object"}
        if stream is not None:
            body["stream_options"] = {"include_usage": True}
        t0 = time.time()
        timeout = timeout or self.timeout
        if stream is None:
            data = self._call("POST", "/chat/completions", body, timeout)
            choice = (data.get("choices") or [{}])[0]
            usage = data.get("usage") or {}
            return ChatResult(
                text=strip_thinking((choice.get("message") or {}).get("content") or ""),
                model=data.get("model", model),
                prompt_tokens=int(usage.get("prompt_tokens") or 0),
                completion_tokens=int(usage.get("completion_tokens") or 0),
                duration=time.time() - t0, raw=data,
            )
        resp = self._call("POST", "/chat/completions", body, timeout, stream=True)
        filt = ThinkFilter(stream)
        usage: dict = {}
        for line in _read_stream(resp, timeout):
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices") or []:
                filt.feed((choice.get("delta") or {}).get("content") or "")
        return ChatResult(
            text=filt.finish(), model=model,
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
    Nutzertext gespiegelt. Alle Aufrufe werden in ``calls`` protokolliert
    (``messages, model, temperature, max_tokens, json_mode, stop, num_ctx, seed, think``).
    Antworten werden wie bei echten Backends in 12-Zeichen-Stücken gestreamt und von
    ``<think>``-Blöcken befreit.
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
             stop=None, stream=None, timeout=None, num_ctx=None, seed=None, keep_alive=None,
             think=None) -> ChatResult:
        if not self.up:
            raise BackendUnavailable("Fake-Backend ist abgeschaltet.")
        kwargs = {"model": model or self.default_model, "temperature": temperature,
                  "max_tokens": max_tokens, "json_mode": json_mode, "stop": stop,
                  "num_ctx": num_ctx, "seed": seed, "think": think}
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
        filt = ThinkFilter(stream)
        for i in range(0, len(text), 12):
            filt.feed(text[i:i + 12])
        text = filt.finish()
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

    def version(self) -> str | None:
        return "fake"


# --------------------------------------------------------------- Fabrik
def make_backend(cfg: Config) -> LLMBackend:
    """Erzeugt das konfigurierte Backend. ``auto`` wählt Ollama, sonst einen OpenAI-kompatiblen Server."""
    kind = (cfg.backend or "auto").lower()
    keep_alive = getattr(cfg, "keep_alive", None) or None
    if kind == "fake":
        return FakeBackend(model=cfg.model or "fake-modell")
    if kind == "ollama":
        return OllamaBackend(cfg.base_url, cfg.model, cfg.embed_model, cfg.timeout, keep_alive)
    if kind in ("openai", "llamacpp", "lmstudio", "vllm"):
        return OpenAICompatBackend(cfg.base_url, cfg.model, cfg.embed_model, cfg.timeout)
    if kind != "auto":
        raise LLMError(f"Unbekanntes Backend: {cfg.backend!r} (erlaubt: auto, ollama, openai, fake)")
    ollama = OllamaBackend(cfg.base_url if "11434" in cfg.base_url else "", cfg.model, cfg.embed_model,
                           cfg.timeout, keep_alive)
    if ollama.available():
        return ollama
    compat = OpenAICompatBackend(cfg.base_url if "11434" not in cfg.base_url else "", cfg.model,
                                 cfg.embed_model, cfg.timeout)
    if compat.available():
        return compat
    return ollama  # Standard; Aufrufer prüft available() und gibt eine klare Meldung aus
