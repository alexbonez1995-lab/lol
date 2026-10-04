import json
import os
import stat
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from obito.llm import BackendUnavailable, FakeBackend, LLMError
from obito.training import evaluate, modelfile, train_lora
from obito.training.evaluate import compare, keyword_score, load_items, run_eval, save_report
from obito.training.modelfile import (BASE_MAP, CHATML_TEMPLATE, build_modelfile, check_adapter_compat,
                                      create_model, default_system_prompt, hf_base_for, ollama_base_for)
from obito.training.train_lora import (LossTracker, build_example, find_eval_path, load_jsonl, params_billion,
                                       preflight, resolve_base, route_hints, vram_need_gb)

REPO_ROOT = Path(__file__).resolve().parent.parent


# =============================================================== modelfile
class HfBaseTest(unittest.TestCase):
    def test_base_map_complete(self):
        expected = {
            "qwen2.5:1.5b": "Qwen/Qwen2.5-1.5B-Instruct", "qwen2.5:3b": "Qwen/Qwen2.5-3B-Instruct",
            "qwen2.5:7b": "Qwen/Qwen2.5-7B-Instruct", "qwen2.5:14b": "Qwen/Qwen2.5-14B-Instruct",
            "qwen2.5-coder:7b": "Qwen/Qwen2.5-Coder-7B-Instruct", "qwen3:8b": "Qwen/Qwen3-8B",
            "qwen3:14b": "Qwen/Qwen3-14B", "llama3.1:8b": "meta-llama/Llama-3.1-8B-Instruct",
            "llama3.2:3b": "meta-llama/Llama-3.2-3B-Instruct", "mistral:7b": "mistralai/Mistral-7B-Instruct-v0.3",
            "gemma3:4b": "google/gemma-3-4b-it", "gemma3:12b": "google/gemma-3-12b-it",
            "phi3.5:3.8b": "microsoft/Phi-3.5-mini-instruct",
        }
        for tag, hf in expected.items():
            self.assertEqual(BASE_MAP.get(tag), hf, tag)

    def test_hf_base_for_variants(self):
        self.assertEqual(hf_base_for("qwen2.5:7b"), "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(hf_base_for("qwen2.5:7b-instruct-q4_K_M"), "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(hf_base_for("QWEN2.5:7B-Instruct-fp16"), "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(hf_base_for("qwen2.5:latest"), "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(hf_base_for("qwen2.5"), "Qwen/Qwen2.5-7B-Instruct")
        self.assertEqual(hf_base_for("qwen3"), "Qwen/Qwen3-8B")
        self.assertEqual(hf_base_for("llama3.2:latest"), "meta-llama/Llama-3.2-3B-Instruct")
        self.assertEqual(hf_base_for("mistral:7b-instruct-v0.3-q4_0"), "mistralai/Mistral-7B-Instruct-v0.3")
        self.assertEqual(hf_base_for("phi3.5:3.8b-mini-instruct-q4_0"), "microsoft/Phi-3.5-mini-instruct")
        self.assertEqual(hf_base_for("registry.ollama.ai/library/gemma3:12b"), "google/gemma-3-12b-it")
        self.assertEqual(hf_base_for(" qwen2.5-coder:7b "), "Qwen/Qwen2.5-Coder-7B-Instruct")
        self.assertIsNone(hf_base_for("qwen2.5:72b"))
        self.assertIsNone(hf_base_for("deepseek-r1:8b"))
        self.assertIsNone(hf_base_for(""))
        self.assertIsNone(hf_base_for(None))

    def test_ollama_base_for(self):
        self.assertEqual(ollama_base_for("Qwen/Qwen2.5-7B-Instruct"), "qwen2.5:7b")
        self.assertEqual(ollama_base_for("qwen/qwen2.5-7b-instruct"), "qwen2.5:7b")
        self.assertEqual(ollama_base_for("/modelle/Llama-3.1-8B-Instruct/"), "llama3.1:8b")
        self.assertIsNone(ollama_base_for("Qwen/Qwen2.5-72B-Instruct"))
        self.assertIsNone(ollama_base_for(""))
        for tag, hf in BASE_MAP.items():
            self.assertEqual(ollama_base_for(hf), tag)

    def test_family_and_stop_tokens(self):
        self.assertEqual(modelfile.family_of("qwen2.5:7b"), "qwen")
        self.assertEqual(modelfile.family_of("/x/Llama-3.1-8B-Instruct"), "llama")
        self.assertEqual(modelfile.family_of("mixtral:8x7b"), "mistral")
        self.assertIsNone(modelfile.family_of("foo:1b"))
        self.assertEqual(modelfile.stop_tokens_for(None, "qwen2.5:7b"), ["<|im_start|>", "<|im_end|>"])
        self.assertEqual(modelfile.stop_tokens_for(None, "gemma3:4b"), ["<start_of_turn>", "<end_of_turn>"])
        self.assertEqual(modelfile.stop_tokens_for(None, "unbekannt:1b"), [])
        self.assertEqual(modelfile.stop_tokens_for(CHATML_TEMPLATE, "x.gguf"), ["<|im_start|>", "<|im_end|>"])
        self.assertEqual(modelfile.stop_tokens_for("{{ .Prompt }}", "x.gguf"), [])


class BuildModelfileTest(unittest.TestCase):
    def test_basic_structure_and_system_quoting(self):
        text = build_modelfile("qwen2.5:7b", "Du bist OBITO.\nAntworte knapp.", temperature=0.2, num_ctx=4096)
        lines = text.splitlines()
        self.assertIn("FROM qwen2.5:7b", lines)
        self.assertIn('SYSTEM """Du bist OBITO.', lines)
        self.assertIn('Antworte knapp."""', lines)
        self.assertIn('SYSTEM """Du bist OBITO.\nAntworte knapp."""', text)
        self.assertIn("PARAMETER temperature 0.2", lines)
        self.assertIn("PARAMETER num_ctx 4096", lines)
        self.assertIn('PARAMETER stop "<|im_start|>"', lines)
        self.assertIn('PARAMETER stop "<|im_end|>"', lines)
        self.assertNotIn("ADAPTER", text)
        self.assertNotIn("TEMPLATE", text)
        self.assertTrue(text.endswith("\n"))
        # Reihenfolge: FROM vor SYSTEM vor PARAMETER
        self.assertLess(text.index("FROM"), text.index("SYSTEM"))
        self.assertLess(text.index("SYSTEM"), text.index("PARAMETER"))

    def test_system_prompt_with_inner_triple_quotes_and_empty(self):
        text = build_modelfile("qwen2.5:7b", 'Regel: """nie""" erfinden')
        self.assertEqual(text.count('"""'), 2)
        self.assertIn("SYSTEM \"\"\"Regel: '''nie''' erfinden\"\"\"", text)
        self.assertNotIn("SYSTEM", build_modelfile("qwen2.5:7b", "   "))
        self.assertNotIn("SYSTEM", build_modelfile("qwen2.5:7b", None))

    def test_gguf_adapter_is_written_absolute(self):
        text = build_modelfile("qwen2.5:7b", "S", adapter="out/adapter.gguf")
        self.assertIn("ADAPTER " + os.path.abspath("out/adapter.gguf"), text)
        self.assertIn("FROM qwen2.5:7b", text)

    def test_safetensors_adapter_only_for_llama_mistral_gemma(self):
        for base in ("llama3.1:8b", "mistral:7b", "gemma3:4b"):
            text = build_modelfile(base, "S", adapter="./lora-dir")
            self.assertIn("ADAPTER " + os.path.abspath("./lora-dir"), text)
        with self.assertRaises(ValueError) as cm:
            build_modelfile("qwen2.5:7b", "S", adapter="./lora-dir")
        msg = str(cm.exception)
        self.assertIn("convert_lora_to_gguf.py", msg)
        self.assertIn("--base Qwen/Qwen2.5-7B-Instruct", msg)
        self.assertIn("--outfile adapter.gguf", msg)
        with self.assertRaises(ValueError):
            build_modelfile("phi3.5:3.8b", "S", adapter="./lora-dir")

    def test_gguf_base_requires_template(self):
        text = build_modelfile("./qwen-merged.q4_k_m.gguf", "S")
        self.assertIn("FROM " + os.path.abspath("./qwen-merged.q4_k_m.gguf"), text)
        self.assertIn('TEMPLATE """' + CHATML_TEMPLATE + '"""', text)
        self.assertIn('PARAMETER stop "<|im_end|>"', text)
        with self.assertRaises(ValueError) as cm:
            build_modelfile("./merged.q4_k_m.gguf", "S")
        self.assertIn("Vorlage", str(cm.exception))
        self.assertIn("--vorlage", str(cm.exception))
        # explizite Vorlage: Stop-Token aus der Vorlage erkannt
        llama_tpl = "{{ .System }}<|start_header_id|>user<|end_header_id|>{{ .Prompt }}<|eot_id|>"
        text = build_modelfile("./merged.q4_k_m.gguf", "S", template=llama_tpl)
        self.assertIn('TEMPLATE """' + llama_tpl + '"""', text)
        self.assertIn('PARAMETER stop "<|eot_id|>"', text)
        self.assertNotIn("<|im_end|>", text)

    def test_extra_params_and_validation(self):
        text = build_modelfile("qwen2.5:7b", "S", extra_params={"num_predict": 512, "repeat_penalty": 1.1,
                                                                 "stop": ["ENDE", "<|im_end|>"], "mirostat": None,
                                                                 "temperature": 0.9})
        self.assertIn("PARAMETER num_predict 512", text)
        self.assertIn("PARAMETER repeat_penalty 1.1", text)
        self.assertIn('PARAMETER stop "ENDE"', text)
        self.assertEqual(text.count('PARAMETER stop "<|im_end|>"'), 1)
        self.assertIn("PARAMETER temperature 0.9", text)
        self.assertNotIn("temperature 0.4", text)
        self.assertNotIn("mirostat", text)
        with self.assertRaises(ValueError):
            build_modelfile("", "S")
        with self.assertRaises(ValueError):
            build_modelfile("qwen2.5:7b", "S", temperature=3.0)
        with self.assertRaises(ValueError):
            build_modelfile("qwen2.5:7b", "S", num_ctx=100)
        with self.assertRaises(ValueError):
            build_modelfile("qwen2.5:7b", "S", extra_params={"num ctx": 1})

    def test_default_system_prompt(self):
        prompt = default_system_prompt()
        self.assertIsInstance(prompt, str)
        self.assertGreater(len(prompt), 100)
        self.assertIn("OBITO", prompt)
        # Fallback, wenn obito.agents nicht importierbar ist
        with mock.patch.dict(sys.modules, {"obito.agents": None}):
            fallback = default_system_prompt()
        self.assertIn("OBITO", fallback)
        self.assertIn("Unsicher", fallback)


def _fake_ollama(directory: str, fail: bool = False) -> str:
    """Legt ein Ersatz-Programm »ollama« an, das seine Argumente und das Modelfile ausgibt."""
    if os.name == "nt":
        path = os.path.join(directory, "ollama.bat")
        body = "@echo off\r\necho args: %*\r\ntype Modelfile\r\n" + ("exit /b 3\r\n" if fail else "exit /b 0\r\n")
        Path(path).write_text(body, encoding="utf-8")
    else:
        path = os.path.join(directory, "ollama")
        body = '#!/bin/sh\necho "args: $@"\ncat Modelfile\n' + ("exit 3\n" if fail else "exit 0\n")
        Path(path).write_text(body, encoding="utf-8")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return path


class CreateModelTest(unittest.TestCase):
    def test_runs_fake_binary_and_writes_modelfile_into_cwd(self):
        with tempfile.TemporaryDirectory() as d:
            binary = _fake_ollama(d)
            adapter_dir = os.path.join(d, "adapter")
            os.makedirs(adapter_dir)
            text = build_modelfile("qwen2.5:7b", "Du bist OBITO.", adapter=os.path.join(adapter_dir, "adapter.gguf"))
            cp = create_model("obito-v1", text, ollama_bin=binary, cwd=adapter_dir)
            self.assertIsInstance(cp, subprocess.CompletedProcess)
            self.assertEqual(cp.returncode, 0)
            self.assertIn("args: create obito-v1 -f Modelfile", cp.stdout)
            self.assertIn("FROM qwen2.5:7b", cp.stdout)
            self.assertIn('SYSTEM """Du bist OBITO."""', cp.stdout)
            written = Path(adapter_dir, "Modelfile").read_text(encoding="utf-8")
            self.assertEqual(written, text)

    def test_temp_cwd_and_failure_code(self):
        with tempfile.TemporaryDirectory() as d:
            binary = _fake_ollama(d, fail=True)
            cp = create_model("obito-v1:test", "FROM qwen2.5:7b\n", ollama_bin=binary)
            self.assertEqual(cp.returncode, 3)
            self.assertIn("FROM qwen2.5:7b", cp.stdout)

    @unittest.skipIf(os.name == "nt", "PATH-Suche nach Shell-Skripten nur unter POSIX")
    def test_binary_found_via_path(self):
        with tempfile.TemporaryDirectory() as d:
            _fake_ollama(d)
            with mock.patch.dict(os.environ, {"PATH": d + os.pathsep + os.environ.get("PATH", "")}):
                cp = create_model("obito-v1", "FROM qwen2.5:7b\n", cwd=d)
            self.assertEqual(cp.returncode, 0)
            self.assertIn("args: create obito-v1 -f Modelfile", cp.stdout)

    def test_missing_binary_is_clear_german_error(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "gibt-es-nicht")
            with self.assertRaises(FileNotFoundError) as cm:
                create_model("obito-v1", "FROM qwen2.5:7b\n", ollama_bin=missing, cwd=d)
            msg = str(cm.exception)
            self.assertIn("nicht gefunden", msg)
            self.assertIn("Ollama", msg)
            self.assertIn(missing, msg)
            self.assertTrue(Path(d, "Modelfile").is_file())

    def test_invalid_input(self):
        with self.assertRaises(ValueError):
            create_model("", "FROM x\n")
        with self.assertRaises(ValueError):
            create_model("böse name", "FROM x\n")
        with self.assertRaises(ValueError):
            create_model("ok", "   ")


class AdapterCompatTest(unittest.TestCase):
    def _write_info(self, d: str, **info) -> None:
        Path(d, "obito_training.json").write_text(json.dumps(info), encoding="utf-8")

    def test_without_info_none(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertIsNone(check_adapter_compat(os.path.join(d, "adapter.gguf"), "qwen2.5:7b"))
            self.assertIsNone(check_adapter_compat(d, "qwen2.5:7b"))
            Path(d, "obito_training.json").write_text("kaputt", encoding="utf-8")
            self.assertIsNone(check_adapter_compat(d, "qwen2.5:7b"))
        self.assertIsNone(check_adapter_compat("", "qwen2.5:7b"))

    def test_matching_base_including_variants(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_info(d, hf_base="Qwen/Qwen2.5-7B-Instruct", ollama_base="qwen2.5:7b")
            self.assertIsNone(check_adapter_compat(os.path.join(d, "adapter.gguf"), "qwen2.5:7b"))
            self.assertIsNone(check_adapter_compat(d, "qwen2.5:7b-instruct-q4_K_M"))
            self.assertIsNone(check_adapter_compat(d, "qwen2.5:latest"))

    def test_mismatch_warns(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_info(d, hf_base="Qwen/Qwen2.5-7B-Instruct", ollama_base="qwen2.5:7b")
            warn = check_adapter_compat(os.path.join(d, "adapter.gguf"), "qwen2.5:3b")
            self.assertIsNotNone(warn)
            self.assertIn("qwen2.5:7b", warn)
            self.assertIn("qwen2.5:3b", warn)
            self.assertIn("exakt gleichen Basismodell", warn)
            # nur hf_base bekannt
            self._write_info(d, hf_base="meta-llama/Llama-3.1-8B-Instruct")
            warn = check_adapter_compat(d, "qwen2.5:7b")
            self.assertIn("meta-llama/llama-3.1-8b-instruct", warn.lower())

    def test_unknown_base_gives_hint(self):
        with tempfile.TemporaryDirectory() as d:
            self._write_info(d, hf_base="Qwen/Qwen2.5-7B-Instruct", ollama_base="qwen2.5:7b")
            warn = check_adapter_compat(d, "meinmodell:latest")
            self.assertIn("nicht prüfbar", warn)
            self.assertIn("qwen2.5:7b", warn)
            # gleicher, aber unbekannter Tag -> kein Hinweis
            self._write_info(d, ollama_base="meinmodell:latest")
            self.assertIsNone(check_adapter_compat(d, "meinmodell:latest"))


# ================================================================ evaluate
class ItemsAndReportTest(unittest.TestCase):
    def test_load_items_jsonl_and_array(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d, "fragen.jsonl")
            p.write_text('{"frage": "Was ist 2+2?", "stichworte": "vier", "muss_zahl": 4, "quelle_id": 7}\n\n'
                         '{"frage": "Farbe?", "erwartet": "blau", "verboten": ["rot"]}\n', encoding="utf-8")
            items = load_items(p)
            self.assertEqual(len(items), 2)
            self.assertEqual(items[0]["stichworte"], ["vier"])
            self.assertEqual(items[0]["muss_zahl"], "4")
            self.assertEqual(items[0]["quelle_id"], 7)
            self.assertEqual(items[1]["verboten"], ["rot"])
            self.assertEqual(items[1]["erwartet"], "blau")
            arr = Path(d, "fragen.json")
            arr.write_text(json.dumps([{"frage": "a"}, {"frage": "b", "stichworte": ["x"]}]), encoding="utf-8")
            self.assertEqual([i["frage"] for i in load_items(arr)], ["a", "b"])
            bad = Path(d, "kaputt.jsonl")
            bad.write_text('{"frage": "ok"}\n{kein json}\n', encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                load_items(bad)
            self.assertIn("Zeile 2", str(cm.exception))
            bad.write_text('{"stichworte": ["ohne frage"]}\n', encoding="utf-8")
            with self.assertRaises(ValueError):
                load_items(bad)
            with self.assertRaises(FileNotFoundError):
                load_items(Path(d, "fehlt.jsonl"))

    def test_example_file_loads(self):
        items = load_items(REPO_ROOT / "beispiele" / "eval_fragen.jsonl")
        self.assertGreaterEqual(len(items), 5)
        self.assertTrue(all(i["frage"] for i in items))

    def test_save_and_load_report(self):
        with tempfile.TemporaryDirectory() as d:
            report = {"modell": "m", "ergebnisse": [{"frage": "ä", "antwort": "ö"}]}
            p = save_report(report, Path(d, "sub", "bericht.json"))
            self.assertTrue(p.is_file())
            self.assertIn("ä", p.read_text(encoding="utf-8"))
            self.assertEqual(evaluate.load_report(p), report)
            Path(d, "x.json").write_text("[]", encoding="utf-8")
            with self.assertRaises(ValueError):
                evaluate.load_report(Path(d, "x.json"))


class KeywordScoreTest(unittest.TestCase):
    def test_prefix_rule_and_fraction(self):
        item = {"frage": "q", "stichworte": ["leichter", "steif", "Vibration", "teurer", "spröde", "leitfähig"]}
        answer = "Carbon ist leichter, bietet hohe Steifigkeit und dämpft Vibrationen; es ist aber teuer."
        self.assertAlmostEqual(keyword_score(answer, item), 0.5)
        self.assertEqual(keyword_score("", item), 0.0)
        self.assertEqual(keyword_score("Leichter, steifer, Vibrationen, teurer, spröder, leitfähiger.", item), 1.0)

    def test_short_words_exact_umlaut_and_case(self):
        item = {"frage": "q", "stichworte": ["Wh", "LÖTSTATION"]}
        self.assertEqual(keyword_score("22 wh an der lötstation", item), 1.0)
        self.assertEqual(keyword_score("22 whatever", item), 0.0)   # "wh" nur exakt

    def test_multiword_and_symbol_keywords(self):
        item = {"frage": "q", "stichworte": ["def", "return False", "return True", "range", "% "]}
        code = "def f(n):\n    for i in range(2, n):\n        if n % i == 0:\n            return False\n    return True"
        self.assertEqual(keyword_score(code, item), 1.0)
        self.assertEqual(keyword_score("return True oder False", item), 0.2)   # "return false" nicht als Phrase

    def test_verboten_zeroes(self):
        item = {"frage": "q", "stichworte": ["Aluminium"], "verboten": ["besser", "Unsicher"]}
        self.assertEqual(keyword_score("Aluminium ist super", item), 1.0)
        self.assertEqual(keyword_score("Aluminium ist besser", item), 0.0)
        self.assertEqual(keyword_score("Aluminium – unsicher, ob …", item), 0.0)

    def test_muss_zahl_tolerance(self):
        item = {"frage": "q", "stichworte": ["Wh"], "muss_zahl": "22.2"}
        self.assertEqual(keyword_score("ca. 22,2 Wh", item), 1.0)
        self.assertEqual(keyword_score("ca. 22,5 Wh", item), 1.0)      # +1,35 % innerhalb ±2 %
        self.assertEqual(keyword_score("ca. 23 Wh", item), 0.0)        # +3,6 % außerhalb
        self.assertEqual(keyword_score("keine Zahl Wh", item), 0.0)
        self.assertEqual(keyword_score("1.500 mAh", {"frage": "q", "muss_zahl": "1500"}), 1.0)
        self.assertEqual(keyword_score("Es sind 4,2 Volt", {"frage": "q", "muss_zahl": "4.2"}), 1.0)
        with self.assertRaises(ValueError):
            keyword_score("x", {"frage": "q", "muss_zahl": "abc"})

    def test_fallback_to_expected_words_and_no_criteria(self):
        item = {"frage": "q", "erwartet": "Carbon ist leichter und steifer als Aluminium."}
        self.assertEqual(evaluate.expected_keywords(item), ["carbon", "leichter", "steifer", "aluminium"])
        self.assertEqual(keyword_score("Carbon ist leichter.", item), 0.5)
        self.assertFalse(evaluate.has_criteria({"frage": "q"}))
        self.assertEqual(keyword_score("irgendwas", {"frage": "q"}), 0.0)


def _eval_items():
    return [
        {"frage": "Vorteile von Carbon?", "erwartet": "leichter und steifer", "stichworte": ["leichter", "steif"],
         "quelle_id": 1},
        {"frage": "Energie 4S 1500 mAh?", "erwartet": "22,2 Wh", "stichworte": ["Wh"], "muss_zahl": "22.2",
         "quelle_id": 2},
        {"frage": "Hauptstadt von Frankreich?", "stichworte": ["Paris"], "verboten": ["Berlin"], "quelle_id": 3},
    ]


def _answer_for(messages: list[dict]) -> str:
    frage = messages[-1]["content"]
    if "Carbon" in frage:
        return "Carbon ist leichter und hat eine hohe Steifigkeit."
    if "Energie" in frage:
        return "14,8 V × 1,5 Ah ≈ 22,2 Wh."
    return "Berlin."   # verboten -> 0


class RunEvalTest(unittest.TestCase):
    def test_without_judge(self):
        fb = FakeBackend(responder=lambda msgs, kw: _answer_for(msgs), model="qwen2.5:7b")
        seen = []
        report = run_eval(fb, "qwen2.5:7b", _eval_items(), system_prompt="Du bist OBITO.", num_ctx=4096,
                          progress=lambda done, total, erg: seen.append((done, total, erg["frage"])))
        self.assertEqual(report["modell"], "qwen2.5:7b")
        self.assertIsNone(report["tiefe"])
        self.assertEqual(report["anzahl"], 3)
        self.assertIsNone(report["richter_score"])
        self.assertEqual([e["stichwort"] for e in report["ergebnisse"]], [1.0, 1.0, 0.0])
        self.assertAlmostEqual(report["stichwort_score"], 2 / 3, places=3)
        self.assertEqual(report["warnungen"], [])
        self.assertTrue(all(e["richter"] is None for e in report["ergebnisse"]))
        self.assertEqual(set(report["ergebnisse"][0]), {"frage", "antwort", "stichwort", "richter", "dauer", "tokens",
                                                        "erwartet", "quelle_id"})
        self.assertIsInstance(report["dauer_mittel"], float)
        self.assertGreater(report["tokens_mittel"], 0)
        self.assertEqual(seen, [(1, 3, "Vorteile von Carbon?"), (2, 3, "Energie 4S 1500 mAh?"),
                                (3, 3, "Hauptstadt von Frankreich?")])
        # Modellaufruf: temperature 0, seed 42, System-Prompt, num_ctx
        self.assertEqual(len(fb.calls), 3)
        call = fb.calls[0]
        self.assertEqual(call["temperature"], 0.0)
        self.assertEqual(call["seed"], 42)
        self.assertEqual(call["num_ctx"], 4096)
        self.assertEqual(call["model"], "qwen2.5:7b")
        self.assertEqual(call["messages"][0], {"role": "system", "content": "Du bist OBITO."})
        self.assertEqual(call["messages"][1]["role"], "user")
        self.assertFalse(call["json_mode"])

    def test_with_judge(self):
        def responder(msgs, kw):
            if kw["model"] == "richter":
                return '{"punkte": 7, "begruendung": "ok"}'
            return _answer_for(msgs)
        fb = FakeBackend(responder=responder, model="qwen2.5:7b")
        report = run_eval(fb, "qwen2.5:7b", _eval_items(), judge_model="richter")
        self.assertEqual(report["richter_score"], 7.0)
        self.assertEqual(report["ergebnisse"][0]["richter"], {"punkte": 7, "begruendung": "ok"})
        self.assertEqual(report["warnungen"], [])
        judge_calls = [c for c in fb.calls if c["model"] == "richter"]
        self.assertEqual(len(judge_calls), 3)
        self.assertIsInstance(judge_calls[0]["json_mode"], dict)
        self.assertEqual(judge_calls[0]["temperature"], 0.0)
        self.assertIn("Carbon", judge_calls[0]["messages"][-1]["content"])
        self.assertIn("leichter und steifer", judge_calls[0]["messages"][-1]["content"])

    def test_judge_retry_once_then_warning(self):
        judge_answers = iter(["kaputt", '{"punkte": "9 Punkte"}', "müll", "noch müll", "x", "y"])

        def responder(msgs, kw):
            if kw["model"] == "richter":
                return next(judge_answers)
            return _answer_for(msgs)
        fb = FakeBackend(responder=responder, model="m")
        report = run_eval(fb, "m", _eval_items()[:2], judge_model="richter")
        self.assertEqual(report["ergebnisse"][0]["richter"]["punkte"], 9)   # Retry hat gegriffen
        self.assertIsNone(report["ergebnisse"][1]["richter"])              # zwei Fehlversuche -> None
        self.assertTrue(any("Richter" in w for w in report["warnungen"]))
        judge_calls = [c for c in fb.calls if c["model"] == "richter"]
        self.assertEqual(len(judge_calls), 4)
        self.assertIs(judge_calls[1]["json_mode"], True)
        self.assertIn("JSON-Objekt", judge_calls[1]["messages"][-1]["content"])
        self.assertEqual(report["richter_score"], 9.0)

    def test_judge_equals_model_is_disabled(self):
        fb = FakeBackend(responder=lambda msgs, kw: _answer_for(msgs), model="qwen2.5:7b")
        report = run_eval(fb, "qwen2.5:7b", _eval_items(), judge_model="qwen2.5:7b")
        self.assertIsNone(report["richter_score"])
        self.assertTrue(all(e["richter"] is None for e in report["ergebnisse"]))
        self.assertTrue(any(w.startswith("Richter = geprüftes Modell") for w in report["warnungen"]))
        self.assertEqual(len(fb.calls), 3)

    def test_exclude_ids(self):
        fb = FakeBackend(responder=lambda msgs, kw: _answer_for(msgs))
        report = run_eval(fb, "m", _eval_items(), exclude_ids=frozenset({1, 3}))
        self.assertEqual(report["anzahl"], 1)
        self.assertEqual(report["ergebnisse"][0]["frage"], "Energie 4S 1500 mAh?")
        self.assertTrue(any("ausgeschlossen" in w for w in report["warnungen"]))

    def test_through_brain(self):
        class Answer:
            def __init__(self, text):
                self.text, self.tokens, self.duration, self.depth = text, 55, 1.5, "mittel"

        class BrainStandIn:
            def __init__(self):
                self.calls = []
                self.cfg = type("Cfg", (), {"model": "qwen2.5:7b"})()

            def ask(self, question, *, session_id="standard", project=None, depth=None, stream=None,
                    progress=None, learn=None, cancel=None):
                self.calls.append((question, session_id, depth, learn))
                return Answer(_answer_for([{"role": "user", "content": question}]))

        brain = BrainStandIn()
        fb = FakeBackend(responder=lambda msgs, kw: '{"punkte": 5, "begruendung": "ok"}', model="richter")
        report = run_eval(fb, "", _eval_items(), through_brain=brain, depth="mittel", judge_model="richter")
        self.assertEqual(report["modell"], "qwen2.5:7b")
        self.assertEqual(report["tiefe"], "mittel")
        self.assertEqual([c[1] for c in brain.calls], ["eval-0", "eval-1", "eval-2"])
        self.assertTrue(all(c[2] == "mittel" and c[3] is False for c in brain.calls))
        self.assertEqual(report["ergebnisse"][0]["tokens"], 55)
        self.assertEqual(report["ergebnisse"][0]["dauer"], 1.5)
        self.assertEqual(report["richter_score"], 5.0)
        # Richter läuft über das Backend, die Antworten über den Denkkern
        self.assertEqual(len(fb.calls), 3)

    def test_model_error_per_item_and_backend_down(self):
        def responder(msgs, kw):
            if "Energie" in msgs[-1]["content"]:
                raise LLMError("Modell streikt")
            return _answer_for(msgs)
        fb = FakeBackend(responder=responder)
        report = run_eval(fb, "m", _eval_items())
        self.assertEqual(report["anzahl"], 3)
        self.assertEqual(report["ergebnisse"][1]["antwort"], "")
        self.assertEqual(report["ergebnisse"][1]["stichwort"], 0.0)
        self.assertTrue(any("Modellfehler" in w and "streikt" in w for w in report["warnungen"]))
        with self.assertRaises(BackendUnavailable):
            run_eval(FakeBackend(up=False), "m", _eval_items())

    def test_items_without_criteria(self):
        fb = FakeBackend(responses=["a", "b"])
        report = run_eval(fb, "m", [{"frage": "ohne Kriterien"}, {"frage": "mit", "stichworte": ["b"]}])
        self.assertIsNone(report["ergebnisse"][0]["stichwort"])
        self.assertEqual(report["ergebnisse"][1]["stichwort"], 1.0)
        self.assertEqual(report["stichwort_score"], 1.0)
        self.assertTrue(any("ohne Stichworte" in w for w in report["warnungen"]))
        empty = run_eval(FakeBackend(), "m", [])
        self.assertEqual(empty["anzahl"], 0)
        self.assertIsNone(empty["stichwort_score"])


def _report(model: str, answers: dict[str, str], stichwort: dict[str, float], dauer=1.0, tokens=100) -> dict:
    return {
        "modell": model, "tiefe": None, "anzahl": len(answers),
        "stichwort_score": sum(stichwort.values()) / len(stichwort), "richter_score": None,
        "dauer_mittel": dauer, "tokens_mittel": tokens, "warnungen": [],
        "ergebnisse": [{"frage": f, "antwort": a, "stichwort": stichwort[f], "richter": None,
                        "dauer": dauer, "tokens": tokens, "erwartet": "Referenz " + f} for f, a in answers.items()],
    }


class CompareTest(unittest.TestCase):
    def setUp(self):
        fragen = [f"Frage {i}" for i in range(4)]
        self.a = _report("basis", {f: f"GUT {f}" if i % 2 == 0 else f"schwach {f}" for i, f in enumerate(fragen)},
                         {f: 0.5 for f in fragen}, dauer=2.0, tokens=200)
        self.b = _report("obito-v1", {f: f"schwach {f}" if i % 2 == 0 else f"GUT {f}" for i, f in enumerate(fragen)},
                         {f: 0.75 for f in fragen}, dauer=1.0, tokens=150)
        self.b["ergebnisse"].append({"frage": "nur in B", "antwort": "x", "stichwort": 1.0, "richter": None,
                                     "dauer": 1.0, "tokens": 1})

    def test_deterministic_without_judge(self):
        res = compare(self.a, self.b)
        self.assertEqual(res["n"], 4)
        self.assertEqual((res["siege_a"], res["siege_b"], res["gleich"]), (0, 0, 0))
        self.assertAlmostEqual(res["stichwort_delta"], 0.25)
        self.assertAlmostEqual(res["dauer_delta"], -1.0)
        self.assertAlmostEqual(res["tokens_delta"], -50)
        self.assertTrue(all(it["richter"] is None for it in res["items"]))
        self.assertEqual(set(res["items"][0]), {"frage", "a", "b", "stichwort_a", "stichwort_b", "richter"})
        self.assertIn("Nicht signifikant", res["hinweis"])
        self.assertIn("Ohne Richter", res["hinweis"])
        self.assertIn("1 Frage(n) kommen nur in einem Bericht", res["hinweis"])
        self.assertEqual((res["modell_a"], res["modell_b"]), ("basis", "obito-v1"))

    @staticmethod
    def _consistent_judge(msgs, kw):
        """Bevorzugt die Antwort mit »GUT« – unabhängig von der Reihenfolge."""
        content = msgs[-1]["content"]
        a1 = content.split("Antwort 1:")[1].split("Antwort 2:")[0]
        a2 = content.split("Antwort 2:")[1]
        if "GUT" in a1 and "GUT" not in a2:
            return '{"besser": 1, "begruendung": "1"}'
        if "GUT" in a2 and "GUT" not in a1:
            return '{"besser": 2, "begruendung": "2"}'
        return '{"besser": 0, "begruendung": "gleich"}'

    def test_pairwise_consistent(self):
        fb = FakeBackend(responder=self._consistent_judge, model="richter")
        res = compare(self.a, self.b, backend=fb, judge_model="richter")
        self.assertEqual((res["siege_a"], res["siege_b"], res["gleich"], res["inkonsistent"]), (2, 2, 0, 0))
        self.assertEqual([it["richter"] for it in res["items"]], ["A", "B", "A", "B"])
        self.assertEqual(len(fb.calls), 8)   # zwei Aufrufe je Frage (vertauscht)
        first, second = fb.calls[0]["messages"][-1]["content"], fb.calls[1]["messages"][-1]["content"]
        self.assertNotEqual(first, second)
        self.assertIn("Referenz Frage 0", first)
        self.assertIn("Nicht signifikant", res["hinweis"])   # n = 4 < 30

    def test_pairwise_inconsistent_counts_as_equal(self):
        fb = FakeBackend(responder=lambda msgs, kw: '{"besser": 1, "begruendung": "immer die erste"}')
        res = compare(self.a, self.b, backend=fb, judge_model="richter")
        self.assertEqual((res["siege_a"], res["siege_b"], res["gleich"], res["inkonsistent"]), (0, 0, 4, 4))
        self.assertTrue(all(it["richter"] == "gleich" for it in res["items"]))
        self.assertIn("kippten", res["hinweis"])
        tie = compare(self.a, self.b, backend=FakeBackend(responder=lambda m, k: '{"besser": 0, "begruendung": ""}'),
                      judge_model="richter")
        self.assertEqual((tie["gleich"], tie["inkonsistent"]), (4, 0))

    def test_significance_hint_with_many_items(self):
        fragen = [f"F{i}" for i in range(40)]
        a = _report("basis", {f: "schwach" for f in fragen}, {f: 0.5 for f in fragen})
        b = _report("neu", {f: "GUT" for f in fragen}, {f: 0.6 for f in fragen})
        fb = FakeBackend(responder=self._consistent_judge)
        res = compare(a, b, backend=fb, judge_model="richter")
        self.assertEqual((res["siege_a"], res["siege_b"]), (0, 40))
        self.assertIn("signifikant", res["hinweis"])
        self.assertNotIn("Nicht signifikant", res["hinweis"])
        # knapper Ausgang: 21:19 bei n=40 -> |2| <= sqrt(40)
        calls = iter(range(80))

        def knapp(msgs, kw):
            i = next(calls)
            pair = i // 2
            if pair < 21:
                return '{"besser": 1}' if i % 2 == 0 else '{"besser": 2}'
            return '{"besser": 2}' if i % 2 == 0 else '{"besser": 1}'
        res = compare(a, b, backend=FakeBackend(responder=knapp), judge_model="richter")
        self.assertEqual((res["siege_a"], res["siege_b"]), (21, 19))
        self.assertIn("Nicht signifikant", res["hinweis"])

    def test_judge_equals_report_model_disabled(self):
        fb = FakeBackend(responder=self._consistent_judge, model="obito-v1")
        res = compare(self.a, self.b, backend=fb)   # judge_model = default_model = Modell von B
        self.assertEqual(fb.calls, [])
        self.assertIn("Richter = geprüftes Modell", res["hinweis"])
        res = compare(self.a, self.b, judge_model="richter")   # ohne Backend
        self.assertIn("kein Backend", res["hinweis"])

    def test_judge_error_is_tolerated(self):
        def responder(msgs, kw):
            raise LLMError("kaputt")
        res = compare(self.a, self.b, backend=FakeBackend(responder=responder), judge_model="richter")
        self.assertEqual(res["gleich"], 4)
        self.assertIn("kaputt", res["hinweis"])


class LocalJudgeHelpersTest(unittest.TestCase):
    def test_parse_judge(self):
        self.assertEqual(evaluate._local_parse_judge('{"punkte": 7, "begruendung": "x"}'), {"punkte": 7, "begruendung": "x"})
        self.assertEqual(evaluate._local_parse_judge('```json\n{"Punkte": "8/10", "Begründung": "ok"}\n```')["punkte"], 8)
        self.assertEqual(evaluate._local_parse_judge('{"punkte": 42}')["punkte"], 10)
        self.assertIsNone(evaluate._local_parse_judge("kein json"))
        self.assertIsNone(evaluate._local_parse_judge('{"begruendung": "ohne punkte"}'))

    def test_parse_pairwise(self):
        self.assertEqual(evaluate._local_parse_pairwise('{"besser": 2, "begruendung": ""}')["besser"], 2)
        self.assertEqual(evaluate._local_parse_pairwise('{"besser": "A"}')["besser"], 1)
        self.assertEqual(evaluate._local_parse_pairwise('{"besser": "gleich"}')["besser"], 0)
        self.assertIsNone(evaluate._local_parse_pairwise('{"besser": 5}'))
        self.assertIsNone(evaluate._local_parse_pairwise("[]"))

    def test_local_messages_have_marker_and_json_hint(self):
        msgs = evaluate._local_judge_messages("F", "E", "A")
        self.assertTrue(msgs[0]["content"].rstrip().endswith("[OBITO:richter:system]"))
        self.assertIn("Antworte nur mit JSON", msgs[1]["content"])
        pair = evaluate._local_pairwise_judge_messages("F", "", "A1", "A2")
        self.assertIn("Antwort 1:\nA1", pair[1]["content"])
        self.assertIn("keine Referenzantwort", pair[1]["content"])


# ============================================================== train_lora
class FakeTokenizer:
    """Tokenizer-Attrappe: je Nachricht ``[Rolle, *Wörter, ENDE]``; Generation-Prompt = Rollen-Token 3."""

    ROLE = {"system": 1, "user": 2, "assistant": 3}
    END = 4
    chat_template = "{% for m in messages %}{{ m.role }}: {{ m.content }}{% endfor %}"
    pad_token_id = 0
    eos_token = "</s>"

    def __init__(self, return_dict: bool = False, generation_token: int = 3):
        self.return_dict = return_dict
        self.generation_token = generation_token
        self.vocab: dict[str, int] = {}

    def _words(self, text: str) -> list[int]:
        out = []
        for w in text.split():
            out.append(self.vocab.setdefault(w, 100 + len(self.vocab)))
        return out

    def apply_chat_template(self, messages, add_generation_prompt=False, tokenize=True):
        ids: list[int] = []
        for m in messages:
            ids += [self.ROLE[m["role"]]] + self._words(m["content"]) + [self.END]
        if add_generation_prompt:
            ids.append(self.generation_token)
        if not tokenize:
            return " ".join(map(str, ids))
        return {"input_ids": ids} if self.return_dict else ids


class BuildExampleTest(unittest.TestCase):
    MESSAGES = [
        {"role": "system", "content": "Du bist OBITO."},
        {"role": "user", "content": "wie viel ist zwei plus zwei"},
        {"role": "assistant", "content": "vier ganz sicher"},
    ]

    def test_labels_only_on_target(self):
        tok = FakeTokenizer()
        ex = build_example(tok, self.MESSAGES, 1024)
        full = tok.apply_chat_template(self.MESSAGES)
        prompt = tok.apply_chat_template(self.MESSAGES[:-1], add_generation_prompt=True)
        self.assertEqual(ex["input_ids"], full)
        self.assertEqual(ex["attention_mask"], [1] * len(full))
        self.assertEqual(ex["labels"][:len(prompt)], [-100] * len(prompt))
        self.assertEqual(ex["labels"][len(prompt):], full[len(prompt):])
        # Ziel = drei Wörter + ENDE
        self.assertEqual(len([l for l in ex["labels"] if l != -100]), 4)

    def test_left_truncation_keeps_target(self):
        tok = FakeTokenizer()
        full = tok.apply_chat_template(self.MESSAGES)
        target_len = 4
        ex = build_example(tok, self.MESSAGES, target_len + 2)
        self.assertEqual(len(ex["input_ids"]), target_len + 2)
        self.assertEqual(ex["input_ids"], full[-(target_len + 2):])
        self.assertEqual(ex["labels"][:2], [-100, -100])
        self.assertEqual(ex["labels"][2:], full[-target_len:])
        # Ziel passt genau
        ex = build_example(tok, self.MESSAGES, target_len)
        self.assertEqual(ex["labels"], full[-target_len:])

    def test_fully_truncated_target_is_none(self):
        tok = FakeTokenizer()
        self.assertIsNone(build_example(tok, self.MESSAGES, 3))   # Ziel hat 4 Tokens
        self.assertIsNone(build_example(tok, self.MESSAGES[:-1], 1024))   # endet nicht mit assistant
        self.assertIsNone(build_example(tok, [], 1024))
        with self.assertRaises(ValueError):
            build_example(tok, self.MESSAGES, 0)

    def test_dict_return_and_common_prefix_fallback(self):
        tok = FakeTokenizer(return_dict=True)
        ex = build_example(tok, self.MESSAGES, 1024)
        self.assertEqual(ex["input_ids"], tok.apply_chat_template(self.MESSAGES)["input_ids"])
        # Generation-Prompt-Token kommt im vollen Text nicht vor -> gemeinsamer Präfix
        tok = FakeTokenizer(generation_token=77)
        ex = build_example(tok, self.MESSAGES, 1024)
        full = tok.apply_chat_template(self.MESSAGES)
        prompt_wo_gen = len(tok.apply_chat_template(self.MESSAGES[:-1]))
        self.assertEqual(ex["labels"][:prompt_wo_gen], [-100] * prompt_wo_gen)
        self.assertEqual(ex["labels"][prompt_wo_gen], 3)   # Rollen-Token des Assistenten wird mitgelernt
        self.assertEqual(ex["labels"][prompt_wo_gen + 1:], full[prompt_wo_gen + 1:])


class LoadJsonlTest(unittest.TestCase):
    def _write(self, d, name, lines):
        p = Path(d, name)
        p.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in lines), encoding="utf-8")
        return p

    def test_chat_alpaca_dpo(self):
        with tempfile.TemporaryDirectory() as d:
            chat = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hallo"}]}
            alpaca = {"instruction": "Frage", "input": "Kontext", "output": "Antwort"}
            dpo = {"prompt": "p", "chosen": "c", "rejected": "r"}
            rows = load_jsonl(self._write(d, "a.jsonl", [chat, alpaca, dpo]))
            self.assertEqual(rows[0], chat)
            self.assertEqual(rows[1]["messages"], [{"role": "user", "content": "Kontext\n\nFrage"},
                                                   {"role": "assistant", "content": "Antwort"}])
            self.assertEqual(rows[2], dpo)
            Path(d, "leer.jsonl").write_text("\n\n", encoding="utf-8")
            self.assertEqual(load_jsonl(Path(d, "leer.jsonl")), [])

    def test_errors(self):
        with tempfile.TemporaryDirectory() as d:
            cases = [
                {"messages": [{"role": "user", "content": "nur nutzer"}]},
                {"messages": [{"role": "user", "content": "x"}, {"role": "assistant", "content": "   "}]},
                {"messages": [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]},
                {"messages": [{"role": "bot", "content": "x"}, {"role": "assistant", "content": "y"}]},
                {"prompt": "p", "chosen": "", "rejected": "r"},
                {"irgendwas": 1},
                [1, 2],
            ]
            for i, bad in enumerate(cases):
                p = self._write(d, f"bad{i}.jsonl", [bad])
                with self.assertRaises(ValueError, msg=str(bad)) as cm:
                    load_jsonl(p)
                self.assertIn("Zeile 1", str(cm.exception))
            Path(d, "json.jsonl").write_text("{kaputt\n", encoding="utf-8")
            with self.assertRaises(ValueError) as cm:
                load_jsonl(Path(d, "json.jsonl"))
            self.assertIn("ungültiges JSON", str(cm.exception))
            with self.assertRaises(FileNotFoundError):
                load_jsonl(Path(d, "fehlt.jsonl"))

    def test_find_eval_path(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d, "daten.jsonl")
            data.write_text("", encoding="utf-8")
            self.assertIsNone(find_eval_path(data))
            Path(d, "daten.eval.jsonl").write_text("", encoding="utf-8")
            self.assertEqual(find_eval_path(data), Path(d, "daten.eval.jsonl"))
            Path(d, "daten.jsonl.eval.jsonl").write_text("", encoding="utf-8")
            self.assertEqual(find_eval_path(data), Path(d, "daten.jsonl.eval.jsonl"))
            explicit = Path(d, "x.jsonl")
            explicit.write_text("", encoding="utf-8")
            self.assertEqual(find_eval_path(data, str(explicit)), explicit)
            with self.assertRaises(FileNotFoundError):
                find_eval_path(data, str(Path(d, "nein.jsonl")))


class PreflightTest(unittest.TestCase):
    def test_cpu_only_aborts(self):
        r = preflight(200, 20, cuda=False, vram_gb=None, params_b=7.0, qlora=False, bitsandbytes=True)
        self.assertFalse(r["ok"])
        self.assertTrue(any("CUDA" in f for f in r["fehler"]))
        self.assertTrue(any("WSL2" in f or "Stufe 1" in f for f in r["fehler"]))

    def test_auto_qlora_under_20gb_with_bitsandbytes(self):
        r = preflight(200, 20, cuda=True, vram_gb=12.0, params_b=7.0, qlora=False, bitsandbytes=True)
        self.assertTrue(r["ok"])
        self.assertTrue(r["qlora"])
        self.assertTrue(any("QLoRA" in w and "automatisch" in w for w in r["warnungen"]))
        self.assertAlmostEqual(r["bedarf_gb"], 10.0)
        # ohne bitsandbytes: kein Auto-QLoRA, bf16-LoRA für 7B passt nicht in 12 GB
        r = preflight(200, 20, cuda=True, vram_gb=12.0, params_b=7.0, qlora=False, bitsandbytes=False)
        self.assertFalse(r["ok"])
        self.assertFalse(r["qlora"])
        self.assertTrue(any("VRAM reicht nicht" in f for f in r["fehler"]))
        self.assertTrue(any("bitsandbytes" in w for w in r["warnungen"]))
        # 24 GB: kein Auto-QLoRA, bf16 für 7B (~19 GB) passt
        r = preflight(200, 20, cuda=True, vram_gb=24.0, params_b=7.0, qlora=False, bitsandbytes=True)
        self.assertTrue(r["ok"])
        self.assertFalse(r["qlora"])

    def test_requested_qlora_without_bitsandbytes(self):
        r = preflight(200, 20, cuda=True, vram_gb=24.0, params_b=7.0, qlora=True, bitsandbytes=False)
        self.assertFalse(r["ok"])
        self.assertTrue(any("pip install bitsandbytes" in f for f in r["fehler"]))

    def test_vram_too_small_even_with_qlora(self):
        r = preflight(200, 20, cuda=True, vram_gb=6.0, params_b=7.0, qlora=True, bitsandbytes=True)
        self.assertFalse(r["ok"])
        self.assertTrue(any("qwen2.5:3b" in f for f in r["fehler"]))
        r = preflight(200, 20, cuda=True, vram_gb=6.0, params_b=1.5, qlora=True, bitsandbytes=True)
        self.assertTrue(r["ok"])

    def test_example_counts(self):
        r = preflight(19, 2, cuda=True, vram_gb=24.0, params_b=3.0, qlora=False, bitsandbytes=True)
        self.assertFalse(r["ok"])
        self.assertTrue(any("19 Trainingsbeispiele" in f for f in r["fehler"]))
        r = preflight(50, 0, cuda=True, vram_gb=24.0, params_b=3.0, qlora=False, bitsandbytes=True)
        self.assertTrue(r["ok"])
        self.assertTrue(any("50 Trainingsbeispiele" in w for w in r["warnungen"]))
        self.assertTrue(any("Keine Eval-Daten" in w for w in r["warnungen"]))
        r = preflight(150, 10, cuda=True, vram_gb=24.0, params_b=3.0, qlora=False, bitsandbytes=True)
        self.assertEqual(r["warnungen"], [])

    def test_force_and_empty(self):
        r = preflight(5, 0, cuda=False, vram_gb=None, params_b=None, qlora=False, bitsandbytes=False, force=True)
        self.assertTrue(r["ok"])
        self.assertEqual(r["fehler"], [])
        self.assertTrue(any(w.startswith("(erzwungen)") for w in r["warnungen"]))
        self.assertTrue(any("nicht erkennbar" in w for w in r["warnungen"]))
        r = preflight(0, 0, cuda=True, vram_gb=24.0, params_b=7.0, qlora=False, bitsandbytes=True, force=True)
        self.assertFalse(r["ok"])
        self.assertIn("leer", r["fehler"][0])

    def test_params_and_need(self):
        self.assertEqual(params_billion("Qwen/Qwen2.5-7B-Instruct"), 7.0)
        self.assertEqual(params_billion("Qwen/Qwen2.5-1.5B-Instruct"), 1.5)
        self.assertEqual(params_billion("microsoft/Phi-3.5-mini-instruct"), None)
        self.assertEqual(params_billion("google/gemma-3-12b-it"), 12.0)
        self.assertEqual(params_billion("/modelle/Llama-3.1-8B-Instruct/"), 8.0)
        self.assertIsNone(params_billion(""))
        self.assertAlmostEqual(vram_need_gb(7.0, True), 10.0)
        self.assertAlmostEqual(vram_need_gb(3.0, True), 6.0)
        self.assertAlmostEqual(vram_need_gb(1.5, True), 4.5)
        self.assertGreater(vram_need_gb(7.0, False), 18.0)
        self.assertIsNone(vram_need_gb(None, True))


class TrainLoraCliTest(unittest.TestCase):
    def test_help_runs_without_optional_dependencies(self):
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
        cp = subprocess.run([sys.executable, "-m", "obito.training.train_lora", "--help"], cwd=str(REPO_ROOT),
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=60, env=env)
        self.assertEqual(cp.returncode, 0, cp.stderr)
        for opt in ("--basis", "--daten", "--eval-daten", "--ausgabe", "--epochen", "--lr", "--rang", "--alpha",
                    "--max-laenge", "--batch", "--grad-akkum", "--qlora", "--zusammenfuehren", "--dpo", "--erzwingen"):
            self.assertIn(opt, cp.stdout)
        self.assertIn("Basismodell", cp.stdout)
        # auch wenn die schweren Pakete explizit blockiert sind
        blocked = {m: None for m in ("torch", "transformers", "peft", "trl", "datasets", "bitsandbytes")}
        with mock.patch.dict(sys.modules, blocked):
            with self.assertRaises(SystemExit) as cm:
                with mock.patch("sys.stdout"):
                    train_lora.main(["--help"])
        self.assertEqual(cm.exception.code, 0)

    def test_missing_dependency_gives_pip_hint(self):
        with tempfile.TemporaryDirectory() as d:
            data = Path(d, "daten.jsonl")
            row = {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hallo"}]}
            data.write_text((json.dumps(row) + "\n") * 25, encoding="utf-8")
            with mock.patch.dict(sys.modules, {"torch": None}):
                with self.assertRaises(SystemExit) as cm:
                    train_lora.main(["--daten", str(data), "--basis", "qwen2.5:7b", "--ausgabe", d])
            msg = str(cm.exception)
            self.assertIn("torch", msg)
            self.assertIn("pip install", msg)
            self.assertIn("Fehlendes Paket", msg)

    def test_data_errors_before_heavy_imports(self):
        with tempfile.TemporaryDirectory() as d:
            with mock.patch.dict(sys.modules, {"torch": None}):
                with self.assertRaises(SystemExit) as cm:
                    train_lora.main(["--daten", str(Path(d, "fehlt.jsonl")), "--basis", "qwen2.5:7b"])
                self.assertIn("nicht gefunden", str(cm.exception))
                data = Path(d, "d.jsonl")
                data.write_text("", encoding="utf-8")
                with self.assertRaises(SystemExit) as cm:
                    train_lora.main(["--daten", str(data), "--basis", "qwen2.5:7b"])
                self.assertIn("leer", str(cm.exception))
                dpo = {"prompt": "p", "chosen": "c", "rejected": "r"}
                data.write_text((json.dumps(dpo) + "\n") * 30, encoding="utf-8")
                with self.assertRaises(SystemExit) as cm:
                    train_lora.main(["--daten", str(data), "--basis", "qwen2.5:7b", "--dpo"])
                self.assertIn("100 Paare", str(cm.exception))
                self.assertIn("30", str(cm.exception))
                with self.assertRaises(SystemExit) as cm:   # DPO-Daten ohne --dpo
                    train_lora.main(["--daten", str(data), "--basis", "qwen2.5:7b"])
                self.assertIn("--dpo", str(cm.exception))
                with self.assertRaises(SystemExit) as cm:   # unbekannter Ollama-Tag
                    train_lora.main(["--daten", str(data), "--basis", "fantasie:9b"])
                self.assertIn("nicht bekannt", str(cm.exception))
                with self.assertRaises(SystemExit) as cm:
                    train_lora.main(["--daten", str(data), "--basis", ""])
                self.assertIn("Keine Basis", str(cm.exception))

    def test_resolve_base(self):
        self.assertEqual(resolve_base("qwen2.5:7b-instruct-q4_K_M"), ("Qwen/Qwen2.5-7B-Instruct", "qwen2.5:7b"))
        self.assertEqual(resolve_base("Qwen/Qwen2.5-3B-Instruct"), ("Qwen/Qwen2.5-3B-Instruct", "qwen2.5:3b"))
        self.assertEqual(resolve_base("Org/Eigenes-Modell-7B"), ("Org/Eigenes-Modell-7B", None))
        with tempfile.TemporaryDirectory() as d:
            local = Path(d, "Qwen2.5-7B-Instruct")
            local.mkdir()
            self.assertEqual(resolve_base(str(local)), (str(local), "qwen2.5:7b"))
        with self.assertRaises(ValueError):
            resolve_base("unbekannt:3b")

    def test_route_hints_exact(self):
        text = route_hints("./lora-v1", "Qwen/Qwen2.5-7B-Instruct", "qwen2.5:7b")
        self.assertIn("convert_lora_to_gguf.py ./lora-v1 --base Qwen/Qwen2.5-7B-Instruct --outfile adapter.gguf", text)
        self.assertIn("python -m obito modelfile --name obito-v1 --basis qwen2.5:7b --adapter adapter.gguf --erstellen", text)
        self.assertIn("convert_hf_to_gguf.py " + os.path.join("./lora-v1", "merged") + " --outfile merged.f16.gguf", text)
        self.assertIn("llama-quantize merged.f16.gguf merged.q4_k_m.gguf Q4_K_M", text)
        self.assertIn("python -m obito modelfile --name obito-v1 --basis ./merged.q4_k_m.gguf --erstellen", text)
        self.assertIn("(A)", text)
        self.assertIn("(B)", text)
        self.assertIn("--basis <ollama-basis>", route_hints("out", "X/Y", None, "out/merged"))

    def test_training_info_and_template_hash(self):
        with tempfile.TemporaryDirectory() as d:
            p = train_lora.write_training_info(Path(d, "sub"), {"hf_base": "Qwen/Qwen2.5-7B-Instruct",
                                                                 "ollama_base": None, "n_train": 1})
            self.assertEqual(p.name, "obito_training.json")
            self.assertEqual(json.loads(p.read_text(encoding="utf-8"))["hf_base"], "Qwen/Qwen2.5-7B-Instruct")
            # passt zu check_adapter_compat
            self.assertIsNone(check_adapter_compat(str(p.parent), "qwen2.5:7b"))
            self.assertIsNotNone(check_adapter_compat(str(p.parent), "qwen2.5:3b"))
        h1 = train_lora.chat_template_hash(FakeTokenizer())
        self.assertEqual(len(h1), 16)
        tok = FakeTokenizer()
        tok.chat_template = "anders"
        self.assertNotEqual(h1, train_lora.chat_template_hash(tok))

    def test_loss_tracker_warns_on_rising_eval_loss(self):
        tracker = LossTracker(3)
        with mock.patch("sys.stdout"), mock.patch("sys.stderr"):
            tracker.on_train_log(1.2, 1.0)
            self.assertIsNone(tracker.on_eval(1.0, 1.0))
            self.assertIsNone(tracker.on_eval(0.9, 2.0))
            msg = tracker.on_eval(0.95, 3.0)
        self.assertIn("Überanpassung", msg)
        self.assertTrue(tracker.warned)
        self.assertEqual(tracker.eval_losses, [1.0, 0.9, 0.95])

    def test_collator_and_dataset_without_torch(self):
        class FakeTorch:
            long = "long"

            @staticmethod
            def tensor(data, dtype=None):
                return data
        collate = train_lora.make_collator(0, FakeTorch)
        batch = collate([{"input_ids": [5, 6], "attention_mask": [1, 1], "labels": [-100, 6]},
                         {"input_ids": [7], "attention_mask": [1], "labels": [7]}])
        self.assertEqual(batch["input_ids"], [[5, 6], [7, 0]])
        self.assertEqual(batch["attention_mask"], [[1, 1], [1, 0]])
        self.assertEqual(batch["labels"], [[-100, 6], [7, -100]])
        ds = train_lora._ListDataset([{"a": 1}])
        self.assertEqual((len(ds), ds[0]), (1, {"a": 1}))


if __name__ == "__main__":
    unittest.main()
