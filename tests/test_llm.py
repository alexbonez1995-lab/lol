import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from obito.config import Config
from obito.llm import (BackendUnavailable, ChatResult, FakeBackend, LLMError, ModelInfo,
                       ModelNotFound, OllamaBackend, OpenAICompatBackend, ThinkFilter, make_backend,
                       parse_json, strip_thinking)


class ThinkingTest(unittest.TestCase):
    def test_strip(self):
        self.assertEqual(strip_thinking("<think>plan</think>\nAntwort"), "Antwort")
        self.assertEqual(strip_thinking("a <THINK>x</THINK> b"), "a b")
        self.assertEqual(strip_thinking("<think>offen"), "")
        self.assertEqual(strip_thinking("nichts"), "nichts")
        self.assertEqual(strip_thinking(""), "")

    def test_filter_streams_only_outside(self):
        out = []
        f = ThinkFilter(out.append)
        for p in ["Hallo <thi", "nk>geheim</th", "ink> Welt <t", "hink>x</think>!"]:
            f.feed(p)
        self.assertEqual(f.finish(), "Hallo Welt !")
        self.assertEqual("".join(out), "Hallo Welt !")
        self.assertNotIn("geheim", "".join(out))

    def test_filter_plain_and_unclosed(self):
        out = []
        f = ThinkFilter(out.append)
        f.feed("kein denken hier")
        self.assertEqual(f.finish(), "kein denken hier")
        self.assertEqual("".join(out), "kein denken hier")
        out = []
        f = ThinkFilter(out.append)
        f.feed("<think>nie geschlossen")
        self.assertEqual(f.finish(), "")
        self.assertEqual(out, [])

    def test_parse_json_ignores_thinking(self):
        self.assertEqual(parse_json('<think>hm</think>{"a": 1}'), {"a": 1})


class ParseJsonTest(unittest.TestCase):
    def test_plain(self):
        self.assertEqual(parse_json('{"a": 1}'), {"a": 1})
        self.assertEqual(parse_json('[1, 2]'), [1, 2])

    def test_code_fence(self):
        self.assertEqual(parse_json('Hier:\n```json\n{"a": [1, 2]}\n```\nFertig.'), {"a": [1, 2]})

    def test_prose_around_object(self):
        self.assertEqual(parse_json('Antwort: {"k": {"n": "x}y"}} Ende'), {"k": {"n": "x}y"}})

    def test_escaped_quotes(self):
        self.assertEqual(parse_json('{"t": "sag \\"hallo\\""}'), {"t": 'sag "hallo"'})

    def test_invalid(self):
        self.assertIsNone(parse_json("kein json"))
        self.assertIsNone(parse_json(""))
        self.assertIsNone(parse_json(None))
        self.assertIsNone(parse_json("{kaputt: }"))


class FakeBackendTest(unittest.TestCase):
    def test_echo_default(self):
        fb = FakeBackend()
        r = fb.chat([{"role": "user", "content": "hallo"}])
        self.assertIsInstance(r, ChatResult)
        self.assertEqual(r.text, "Echo: hallo")
        self.assertEqual(r.model, "fake-modell")
        self.assertEqual(len(fb.calls), 1)
        self.assertEqual(fb.calls[0]["messages"][0]["content"], "hallo")

    def test_queue_and_stream(self):
        fb = FakeBackend(responses=["eins", "zwei"])
        fb.push("drei")
        out = []
        r = fb.chat([{"role": "user", "content": "x"}], stream=out.append, json_mode=True)
        self.assertEqual(r.text, "eins")
        self.assertEqual("".join(out), "eins")
        self.assertTrue(fb.calls[-1]["json_mode"])
        self.assertEqual(fb.chat([{"role": "user", "content": "x"}]).text, "zwei")
        self.assertEqual(fb.chat([{"role": "user", "content": "x"}]).text, "drei")

    def test_responder(self):
        def resp(messages, kwargs):
            return "system=" + messages[0]["role"] + " modell=" + kwargs["model"]
        fb = FakeBackend(responder=resp)
        r = fb.chat([{"role": "system", "content": "s"}, {"role": "user", "content": "u"}], model="m2")
        self.assertEqual(r.text, "system=system modell=m2")

    def test_records_runtime_options_and_strips_thinking(self):
        fb = FakeBackend(responses=["<think>plan</think>Antwort 42"])
        out = []
        r = fb.chat([{"role": "user", "content": "x"}], stream=out.append, num_ctx=4096, seed=7,
                    json_mode={"type": "object"}, think=False)
        self.assertEqual(r.text, "Antwort 42")
        self.assertEqual("".join(out), "Antwort 42")
        c = fb.calls[0]
        self.assertEqual((c["num_ctx"], c["seed"], c["think"]), (4096, 7, False))
        self.assertEqual(c["json_mode"], {"type": "object"})
        self.assertEqual(fb.version(), "fake")
        self.assertEqual(fb.running(), [])

    def test_embed_deterministic_and_normalized(self):
        fb = FakeBackend(embed_dim=8)
        a, b = fb.embed(["Drohne Carbon", "drohne carbon"])
        self.assertEqual(a, b)
        self.assertAlmostEqual(sum(x * x for x in a), 1.0, places=6)
        self.assertEqual(len(a), 8)

    def test_models(self):
        fb = FakeBackend(models=["llama3.2:3b", "nomic-embed-text"])
        self.assertTrue(fb.has_model("llama3.2:3b"))
        self.assertTrue(fb.has_model("llama3.2"))
        self.assertFalse(fb.has_model("qwen2.5:7b"))
        self.assertTrue(fb.pull("qwen2.5:7b"))
        self.assertTrue(fb.has_model("qwen2.5:7b"))
        self.assertEqual(fb.list_models()[0].short(), "llama3.2:3b")

    def test_down(self):
        fb = FakeBackend(up=False)
        self.assertFalse(fb.available())
        with self.assertRaises(BackendUnavailable):
            fb.chat([{"role": "user", "content": "x"}])
        self.assertIsNone(fb.embed(["x"]))


class MakeBackendTest(unittest.TestCase):
    def test_kinds(self):
        self.assertIsInstance(make_backend(Config(backend="fake")), FakeBackend)
        self.assertIsInstance(make_backend(Config(backend="ollama")), OllamaBackend)
        self.assertIsInstance(make_backend(Config(backend="openai")), OpenAICompatBackend)
        with self.assertRaises(LLMError):
            make_backend(Config(backend="cloud"))

    def test_auto_without_server_returns_ollama(self):
        b = make_backend(Config(backend="auto", base_url="http://127.0.0.1:1", timeout=1))
        self.assertIsInstance(b, OllamaBackend)
        self.assertFalse(b.available())


# ------------------------------------------------- Mini-Server für Ollama/OpenAI
class _Handler(BaseHTTPRequestHandler):
    log: list = []

    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype="application/json"):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/api/version":
            self._send(200, json.dumps({"version": "0.12.0"}).encode())
        elif self.path == "/api/ps":
            self._send(200, json.dumps({"models": [
                {"name": "qwen2.5:7b", "size": 5_000_000_000, "size_vram": 4_000_000_000,
                 "expires_at": "2026-10-04T12:00:00Z"}]}).encode())
        elif self.path == "/api/tags":
            self._send(200, json.dumps({"models": [
                {"name": "qwen2.5:7b", "size": 4_000_000_000,
                 "details": {"family": "qwen2", "parameter_size": "7.6B", "quantization_level": "Q4_K_M"}},
                {"name": "nomic-embed-text:latest", "size": 274_000_000, "details": {}},
            ]}).encode())
        elif self.path == "/v1/models":
            self._send(200, json.dumps({"data": [{"id": "lokal-modell"}]}).encode())
        else:
            self._send(404, b"{}")

    def do_POST(self):
        n = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(n) or b"{}")
        _Handler.log.append((self.path, body))
        if self.path == "/api/chat":
            if body["model"] == "fehlt":
                self._send(404, json.dumps({"error": "model 'fehlt' not found, try pulling it first"}).encode())
                return
            if body.get("stream"):
                lines = [
                    {"model": body["model"], "message": {"role": "assistant", "content": "<think>geheim</think>"}, "done": False},
                    {"model": body["model"], "message": {"role": "assistant", "content": "Hal"}, "done": False},
                    {"model": body["model"], "message": {"role": "assistant", "content": "lo"}, "done": False},
                    {"model": body["model"], "message": {"role": "assistant", "content": ""}, "done": True,
                     "prompt_eval_count": 5, "eval_count": 2},
                ]
                self._send(200, "".join(json.dumps(x) + "\n" for x in lines).encode(), "application/x-ndjson")
            else:
                self._send(200, json.dumps({
                    "model": body["model"], "message": {"role": "assistant", "content": "Hallo"},
                    "prompt_eval_count": 5, "eval_count": 1, "done": True}).encode())
        elif self.path == "/api/show":
            self._send(200, json.dumps({"capabilities": ["completion", "tools", "thinking"]}).encode())
        elif self.path == "/api/embed":
            self._send(200, json.dumps({"embeddings": [[0.1, 0.2] for _ in body["input"]]}).encode())
        elif self.path == "/v1/chat/completions":
            if body.get("stream"):
                chunks = [
                    {"choices": [{"delta": {"content": "Ser"}}]},
                    {"choices": [{"delta": {"content": "vus"}}]},
                    {"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 2}},
                ]
                data = "".join(f"data: {json.dumps(c)}\n\n" for c in chunks) + "data: [DONE]\n\n"
                self._send(200, data.encode(), "text/event-stream")
            else:
                self._send(200, json.dumps({
                    "model": body["model"], "choices": [{"message": {"role": "assistant", "content": "Servus"}}],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 1}}).encode())
        elif self.path == "/v1/embeddings":
            self._send(200, json.dumps({"data": [
                {"index": i, "embedding": [float(i), 1.0]} for i in range(len(body["input"]))]}).encode())
        else:
            self._send(404, b"{}")


class HttpBackendsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _Handler)
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _Handler.log.clear()

    def test_ollama_chat_stream_models_embed(self):
        b = OllamaBackend(f"http://127.0.0.1:{self.port}", "qwen2.5:7b", "nomic-embed-text", timeout=5,
                          keep_alive="30m")
        self.assertTrue(b.available())
        r = b.chat([{"role": "user", "content": "hi"}], temperature=0.1, max_tokens=50, json_mode=True,
                   num_ctx=8192, seed=3, think=False)
        self.assertEqual(r.text, "Hallo")
        self.assertEqual((r.prompt_tokens, r.completion_tokens), (5, 1))
        sent = _Handler.log[-1][1]
        self.assertEqual(sent["format"], "json")
        self.assertEqual(sent["options"], {"temperature": 0.1, "num_predict": 50, "num_ctx": 8192, "seed": 3})
        self.assertEqual(sent["keep_alive"], "30m")
        self.assertIs(sent["think"], False)
        self.assertFalse(sent["stream"])

        schema = {"type": "object", "properties": {"a": {"type": "integer"}}}
        b.chat([{"role": "user", "content": "hi"}], json_mode=schema)
        self.assertEqual(_Handler.log[-1][1]["format"], schema)
        self.assertNotIn("think", _Handler.log[-1][1])

        out = []
        r = b.chat([{"role": "user", "content": "hi"}], stream=out.append)
        self.assertEqual(r.text, "Hallo")
        self.assertEqual("".join(out), "Hallo")
        self.assertEqual(r.completion_tokens, 2)

        self.assertEqual(b.version(), "0.12.0")
        self.assertEqual(b.running()[0]["size_vram"], 4_000_000_000)
        self.assertIn("thinking", b.show("qwen2.5:7b")["capabilities"])
        self.assertEqual(b.info()["version"], "0.12.0")

        models = b.list_models()
        self.assertEqual(models[0].name, "qwen2.5:7b")
        self.assertEqual(models[0].parameters, "7.6B")
        self.assertIn("7.6B", models[0].short())
        self.assertTrue(b.has_model("qwen2.5:7b"))
        self.assertTrue(b.has_model("nomic-embed-text"))
        self.assertFalse(b.has_model("llama3"))
        self.assertEqual(b.embed(["a", "b"]), [[0.1, 0.2], [0.1, 0.2]])
        self.assertIsNone(b.embed([]))
        self.assertEqual(b.info()["backend"], "ollama")

    def test_ollama_model_not_found(self):
        b = OllamaBackend(f"http://127.0.0.1:{self.port}", "fehlt", timeout=5)
        with self.assertRaises(ModelNotFound):
            b.chat([{"role": "user", "content": "hi"}])

    def test_openai_compat(self):
        b = OpenAICompatBackend(f"http://127.0.0.1:{self.port}/v1", "lokal-modell", "embed", timeout=5)
        self.assertTrue(b.available())
        r = b.chat([{"role": "user", "content": "hi"}], json_mode=True, stop=["END"], seed=1, num_ctx=4096)
        self.assertEqual(r.text, "Servus")
        sent = _Handler.log[-1][1]
        self.assertEqual(sent["response_format"], {"type": "json_object"})
        self.assertEqual(sent["stop"], ["END"])
        self.assertEqual(sent["seed"], 1)
        self.assertNotIn("num_ctx", sent)
        schema = {"type": "object"}
        b.chat([{"role": "user", "content": "hi"}], json_mode=schema)
        self.assertEqual(_Handler.log[-1][1]["response_format"]["json_schema"]["schema"], schema)
        self.assertEqual(b.running(), [])
        self.assertIsNone(b.version())
        out = []
        r = b.chat([{"role": "user", "content": "hi"}], stream=out.append)
        self.assertEqual(r.text, "Servus")
        self.assertEqual(r.completion_tokens, 2)
        self.assertEqual([m.name for m in b.list_models()], ["lokal-modell"])
        self.assertEqual(b.embed(["x", "y"]), [[0.0, 1.0], [1.0, 1.0]])

    def test_unreachable(self):
        b = OllamaBackend("http://127.0.0.1:1", "m", timeout=1)
        self.assertFalse(b.available())
        with self.assertRaises(BackendUnavailable):
            b.chat([{"role": "user", "content": "hi"}])
        self.assertIsNone(b.embed(["x"]))


if __name__ == "__main__":
    unittest.main()
