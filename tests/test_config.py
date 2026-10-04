import json
import os
import tempfile
import unittest
from pathlib import Path

from obito.config import Config, find_config_file, load_config, save_config


class ConfigTest(unittest.TestCase):
    def test_defaults_and_paths(self):
        cfg = Config(data_dir="/tmp/obito-test")
        self.assertEqual(cfg.backend, "auto")
        self.assertEqual(cfg.depth, "auto")
        self.assertEqual(cfg.experts_per_question, 5)
        self.assertEqual(cfg.memory_db, Path("/tmp/obito-test/gedaechtnis.db"))
        self.assertEqual(cfg.learning_db, Path("/tmp/obito-test/lernen.db"))
        self.assertEqual(cfg.datasets_dir.name, "datensaetze")
        self.assertEqual(cfg.routing_model, cfg.model)
        cfg.fast_model = "klein"
        self.assertEqual(cfg.routing_model, "klein")

    def test_env_overrides_with_coercion(self):
        env = {"OBITO_MODEL": "llama3.2", "OBITO_ALLOW_TOOLS": "nein", "OBITO_SERVER_PORT": "9000",
               "OBITO_TEMPERATURE": "0.9", "OBITO_AUTO_MEMORY": "ja", "OBITO_DEPTH": "", "OBITO_UNBEKANNT": "x"}
        cfg = load_config(path="/nicht/vorhanden.json", env=env)
        self.assertEqual(cfg.model, "llama3.2")
        self.assertFalse(cfg.allow_tools)
        self.assertTrue(cfg.auto_memory)
        self.assertEqual(cfg.server_port, 9000)
        self.assertAlmostEqual(cfg.temperature, 0.9)
        self.assertEqual(cfg.depth, "auto")   # leer = nicht überschreiben

    def test_file_then_env_and_unknown_keys_ignored(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obito.json"
            p.write_text(json.dumps({"model": "aus-datei", "depth": "tief", "fremd": 1}), encoding="utf-8")
            cfg = load_config(p, env={"OBITO_MODEL": "aus-env"})
            self.assertEqual(cfg.model, "aus-env")
            self.assertEqual(cfg.depth, "tief")
            self.assertFalse(hasattr(cfg, "fremd"))

    def test_invalid_file(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "obito.json"
            p.write_text("[1,2]", encoding="utf-8")
            with self.assertRaises(ValueError):
                load_config(p, env={})

    def test_save_and_roundtrip(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(model="m", server_port=1234, data_dir=d)
            p = save_config(cfg, Path(d) / "sub" / "c.json")
            self.assertTrue(p.exists())
            again = load_config(p, env={})
            self.assertEqual(again.model, "m")
            self.assertEqual(again.server_port, 1234)
            self.assertEqual(again.to_dict()["data_dir"], d)

    def test_find_config_file_env(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.json"
            p.write_text("{}", encoding="utf-8")
            old = os.environ.get("OBITO_CONFIG")
            os.environ["OBITO_CONFIG"] = str(p)
            try:
                self.assertEqual(find_config_file(), p)
            finally:
                if old is None:
                    del os.environ["OBITO_CONFIG"]
                else:
                    os.environ["OBITO_CONFIG"] = old

    def test_ensure_dirs(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = Config(data_dir=os.path.join(d, "daten"))
            cfg.ensure_dirs()
            self.assertTrue(cfg.datasets_dir.is_dir())
            self.assertTrue(cfg.models_dir.is_dir())
            self.assertTrue(cfg.logs_dir.is_dir())


if __name__ == "__main__":
    unittest.main()
