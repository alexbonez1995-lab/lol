"""Tests für obito.sysmon – Parser und Logik ohne Abhängigkeit von der Maschine."""

from __future__ import annotations

import tempfile
import threading
import unittest

from obito import sysmon
from obito.llm import FakeBackend
from obito.tools import ToolRegistry

PROC_STAT_1 = "cpu  1000 50 300 5000 100 0 10 0 0 0\ncpu0 500 25 150 2500 50 0 5 0 0 0\n"
PROC_STAT_2 = "cpu  1500 50 400 5400 100 0 10 0 0 0\n"
MEMINFO = "MemTotal:       16384000 kB\nMemFree:         1000000 kB\nMemAvailable:    8192000 kB\nBuffers: 1 kB\n"
NVIDIA = "NVIDIA GeForce RTX 4070, 12282, 3456, 37, 61\n"
NVIDIA_NA = "Tesla T4, 15360, [N/A], [N/A], 45\n\n"
PS_CPU = '{"Name": "AMD Ryzen 7 5800X", "NumberOfCores": 8, "NumberOfLogicalProcessors": 16, "LoadPercentage": 12}'
PS_CPU_LIST = '[{"Name": "Xeon", "NumberOfCores": 8, "NumberOfLogicalProcessors": 16, "LoadPercentage": 10},' \
              ' {"Name": "Xeon", "NumberOfCores": 8, "NumberOfLogicalProcessors": 16, "LoadPercentage": 30}]'


class ParserTest(unittest.TestCase):
    def test_proc_stat_two_samples(self):
        a = sysmon.parse_proc_stat(PROC_STAT_1)
        b = sysmon.parse_proc_stat(PROC_STAT_2)
        self.assertEqual(a, (5100, 6460))
        load = sysmon.cpu_load_from_samples(a, b)
        # Δidle = 400, Δtotal = 1000 → 60 %
        self.assertAlmostEqual(load, 60.0, places=1)

    def test_proc_stat_invalid(self):
        self.assertIsNone(sysmon.parse_proc_stat(""))
        self.assertIsNone(sysmon.parse_proc_stat("cpu a b c d e"))
        self.assertIsNone(sysmon.cpu_load_from_samples((1, 2), (1, 2)))
        self.assertIsNone(sysmon.cpu_load_from_samples(None, (1, 2)))

    def test_cpuinfo_name_and_physical(self):
        text = ("processor\t: 0\nmodel name\t: Intel(R) Core(TM) i7-12700K\nphysical id\t: 0\ncore id\t: 0\n\n"
                "processor\t: 1\nmodel name\t: Intel(R) Core(TM) i7-12700K\nphysical id\t: 0\ncore id\t: 0\n\n"
                "processor\t: 2\nmodel name\t: Intel(R) Core(TM) i7-12700K\nphysical id\t: 0\ncore id\t: 1\n")
        self.assertEqual(sysmon.parse_cpuinfo_name(text), "Intel(R) Core(TM) i7-12700K")
        self.assertEqual(sysmon.parse_cpuinfo_physical(text), 2)
        self.assertIsNone(sysmon.parse_cpuinfo_name(None))

    def test_meminfo(self):
        mem = sysmon.parse_meminfo(MEMINFO)
        self.assertEqual(mem["gesamt_bytes"], 16384000 * 1024)
        self.assertEqual(mem["frei_bytes"], 8192000 * 1024)
        self.assertIsNone(sysmon.parse_meminfo("Unsinn"))

    def test_nvidia_smi_normal(self):
        gpus = sysmon.parse_nvidia_smi(NVIDIA)
        self.assertEqual(len(gpus), 1)
        g = gpus[0]
        self.assertEqual(g["name"], "NVIDIA GeForce RTX 4070")
        self.assertEqual(g["vram_gesamt_mb"], 12282)
        self.assertEqual(g["vram_belegt_mb"], 3456)
        self.assertEqual(g["auslastung_prozent"], 37.0)
        self.assertEqual(g["temperatur_c"], 61.0)

    def test_nvidia_smi_na_and_empty(self):
        gpus = sysmon.parse_nvidia_smi(NVIDIA_NA)
        self.assertEqual(len(gpus), 1)
        self.assertIsNone(gpus[0]["vram_belegt_mb"])
        self.assertIsNone(gpus[0]["auslastung_prozent"])
        self.assertEqual(sysmon.parse_nvidia_smi(""), [])
        self.assertEqual(sysmon.parse_nvidia_smi(None), [])

    def test_powershell_cpu_object_and_list(self):
        c = sysmon.parse_powershell_cpu(PS_CPU)
        self.assertEqual(c["kerne"], 8)
        self.assertEqual(c["logisch"], 16)
        self.assertEqual(c["last_prozent"], 12.0)
        self.assertIn("Ryzen", c["name"])
        c2 = sysmon.parse_powershell_cpu(PS_CPU_LIST)
        self.assertEqual(c2["kerne"], 16)
        self.assertEqual(c2["last_prozent"], 20.0)
        self.assertIsNone(sysmon.parse_powershell_cpu("kaputt {"))

    def test_vm_stat(self):
        text = ("Mach Virtual Memory Statistics: (page size of 16384 bytes)\nPages free:  100000.\n"
                "Pages inactive: 50000.\nPages speculative: 1000.\n")
        free = sysmon.parse_vm_stat(text)
        self.assertEqual(free, (100000 + 50000 + 1000) * 16384)

    def test_status_rss(self):
        self.assertEqual(sysmon.parse_status_rss("Name:\tpython\nVmRSS:\t  123456 kB\n"), 123456 * 1024)
        self.assertIsNone(sysmon.parse_status_rss("nix"))


class LiveSourcesTest(unittest.TestCase):
    """Die echten Quellen laufen durch – ohne Hardware dürfen sie nur ``None``/leer liefern, nie werfen."""

    def test_cpu_memory_disk_process_never_raise(self):
        c = sysmon.cpu()
        self.assertIn("logisch", c)
        m = sysmon.memory()
        self.assertIn("gesamt_bytes", m)
        d = sysmon.disk(None)
        self.assertGreater(d["gesamt_bytes"], 0)
        p = sysmon.process()
        self.assertGreaterEqual(p["threads"], 1)
        self.assertGreaterEqual(p["laufzeit_s"], 0)

    def test_gpu_without_nvidia_smi(self):
        orig = sysmon._nvidia_smi
        sysmon._nvidia_smi = lambda: None
        try:
            self.assertEqual(sysmon.gpu(), [])
        finally:
            sysmon._nvidia_smi = orig

    def test_gpu_with_patched_output(self):
        orig = sysmon._nvidia_smi
        sysmon._nvidia_smi = lambda: NVIDIA
        try:
            self.assertEqual(sysmon.gpu()[0]["vram_gesamt_mb"], 12282)
        finally:
            sysmon._nvidia_smi = orig

    def test_run_handles_missing_binary_and_timeout(self):
        self.assertIsNone(sysmon._run(["dieses-programm-gibt-es-nicht-xyz"], timeout=1))

    def test_dir_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            with open(f"{tmp}/a.bin", "wb") as f:
                f.write(b"x" * 1000)
            self.assertEqual(sysmon.dir_size(tmp), 1000)
            with open(f"{tmp}/b.bin", "wb") as f:
                f.write(b"y" * 500)
            self.assertEqual(sysmon.dir_size(tmp), 1000)                 # 30 s zwischengespeichert
            self.assertEqual(sysmon.dir_size(tmp, cache=False), 1500)
            self.assertIsNone(sysmon.dir_size(None))
            self.assertIsNone(sysmon.dir_size(f"{tmp}/fehlt"))


class SnapshotTest(unittest.TestCase):
    def test_snapshot_with_fake_backend(self):
        backend = FakeBackend()
        snap = sysmon.snapshot(backend=backend)
        for key in ("zeit", "plattform", "cpu", "ram", "platte", "gpu", "prozess", "modelle_geladen"):
            self.assertIn(key, snap)
        self.assertIsInstance(snap["modelle_geladen"], list)

    def test_loaded_models_errors_give_empty_list(self):
        class _Bad:
            def running(self):
                raise RuntimeError("weg")

        self.assertEqual(sysmon.loaded_models(_Bad()), [])
        self.assertEqual(sysmon.loaded_models(None), [])

    def test_loaded_models_from_backend(self):
        class _B:
            def running(self):
                from obito.llm import ModelInfo
                return [ModelInfo(name="qwen2.5:7b", size=4_700_000_000)]

        models = sysmon.loaded_models(_B())
        self.assertEqual(models[0]["name"], "qwen2.5:7b")
        self.assertEqual(models[0]["groesse"], 4_700_000_000)


class RecommendTest(unittest.TestCase):
    def _snap(self, vram_mb=None, ram_gb=32.0, disk_free_gb=100.0, temp=50.0, logisch=8):
        gpu = [{"name": "GPU", "vram_gesamt_mb": vram_mb, "vram_belegt_mb": 0, "auslastung_prozent": 0,
                "temperatur_c": temp}] if vram_mb else []
        return {"gpu": gpu, "ram": {"gesamt_bytes": ram_gb * 1024 ** 3, "frei_bytes": ram_gb * 1024 ** 3 / 2,
                                    "belegt_prozent": 50.0},
                "platte": {"gesamt_bytes": 500 * 1024 ** 3, "frei_bytes": disk_free_gb * 1024 ** 3, "belegt_prozent": 50.0},
                "cpu": {"name": "x", "kerne": logisch // 2, "logisch": logisch, "last_prozent": 10.0}}

    def test_vram_classes(self):
        for vram, model in ((24576, "32b"), (16384, "14b"), (12288, "14b"), (8192, "7b"), (6144, "7b"), (4096, "3b")):
            hints = " ".join(sysmon.recommend(self._snap(vram_mb=vram)))
            self.assertIn(model, hints, f"{vram} MB → {model}")

    def test_no_gpu_hint(self):
        hints = " ".join(sysmon.recommend(self._snap()))
        self.assertIn("CPU", hints)

    def test_low_ram_disk_and_temperature(self):
        hints = " ".join(sysmon.recommend(self._snap(vram_mb=8192, ram_gb=6, disk_free_gb=2, temp=90)))
        self.assertIn("RAM", hints)
        self.assertIn("frei", hints)
        self.assertIn("°C", hints)


class SamplerTest(unittest.TestCase):
    def test_ring_buffer_order_and_size(self):
        s = sysmon.Sampler(size=3)
        for i in range(5):
            s.add({"zeit": i, "cpu": {"last_prozent": i * 10.0}, "ram": {"belegt_prozent": 50.0},
                   "gpu": [{"auslastung_prozent": 5.0, "vram_gesamt_mb": 8000, "vram_belegt_mb": 4000}]})
        self.assertEqual(len(s), 3)
        ser = s.series()
        self.assertEqual(ser["zeit"], [2, 3, 4])
        self.assertEqual(ser["cpu"], [20.0, 30.0, 40.0])
        self.assertEqual(ser["vram"], [50.0, 50.0, 50.0])
        self.assertEqual(s.last()["zeit"], 4)

    def test_missing_values_are_none(self):
        s = sysmon.Sampler()
        s.add({"zeit": 1})
        ser = s.series()
        self.assertEqual(ser["cpu"], [None])
        self.assertEqual(ser["gpu"], [None])
        self.assertEqual(ser["vram"], [None])

    def test_rejects_non_dict(self):
        with self.assertRaises(TypeError):
            sysmon.Sampler().add("x")  # type: ignore[arg-type]

    def test_thread_safety(self):
        s = sysmon.Sampler(size=1000)

        def work():
            for i in range(200):
                s.add({"zeit": i})

        threads = [threading.Thread(target=work) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(s), 800)


class FormatAndToolTest(unittest.TestCase):
    def test_format_snapshot_contains_sections(self):
        snap = {"zeit": 0, "plattform": {"system": "Linux", "release": "6.1", "python": "3.11"},
                "cpu": {"name": "Test-CPU", "kerne": 4, "logisch": 8, "last_prozent": 12.5},
                "ram": {"gesamt_bytes": 16 * 1024 ** 3, "frei_bytes": 8 * 1024 ** 3, "belegt_prozent": 50.0},
                "platte": {"pfad": "/", "gesamt_bytes": 100 * 1024 ** 3, "frei_bytes": 40 * 1024 ** 3, "belegt_prozent": 60.0},
                "gpu": [{"name": "RTX 4070", "vram_gesamt_mb": 12282, "vram_belegt_mb": 3000, "auslastung_prozent": 10.0,
                         "temperatur_c": 55.0}],
                "prozess": {"rss_bytes": 200 * 1024 ** 2, "threads": 5, "laufzeit_s": 90.0},
                "modelle_geladen": [{"name": "qwen2.5:7b", "groesse_bytes": 4_700_000_000}],
                "datenverzeichnis": "/daten", "datenverzeichnis_bytes": 3 * 1024 ** 3}
        text = sysmon.format_snapshot(snap)
        for word in ("CPU", "RAM", "GPU", "RTX 4070", "qwen2.5:7b", "12,5", "Hinweise"):
            self.assertIn(word, text)

    def test_format_snapshot_without_gpu_or_models(self):
        snap = {"zeit": 0, "plattform": {}, "cpu": {"name": None, "kerne": None, "logisch": None, "last_prozent": None},
                "ram": {"gesamt_bytes": None, "frei_bytes": None, "belegt_prozent": None},
                "platte": {"gesamt_bytes": None, "frei_bytes": None, "belegt_prozent": None},
                "gpu": [], "prozess": {}, "modelle_geladen": []}
        text = sysmon.format_snapshot(snap)
        self.assertIn("unbekannt", text.lower())

    def test_tool_registration_and_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ToolRegistry(workspace=tmp)
            sampler = sysmon.Sampler()
            sysmon.register_tools(reg, backend=FakeBackend(), data_dir=tmp, sampler=sampler)
            tool = next(t for t in reg.list() if t.name == "system_status")
            self.assertFalse(tool.dangerous)
            res = reg.run("system_status", {})
            self.assertTrue(res.ok, res.error)
            self.assertIn("CPU", res.output)
            self.assertEqual(len(sampler), 1)


if __name__ == "__main__":
    unittest.main()
