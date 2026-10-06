"""Phase-3-Unterbefehle (system, geraete, modell3d, simulation, geo) und Chat-Befehle."""

from __future__ import annotations

import io
import json
import os
import tempfile
import unittest

from obito import devices as devmod
from obito import geometry
from obito.brain import Brain
from obito.cli import ChatSession, main
from obito.config import Config
from obito.llm import FakeBackend


class SubcommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.data = os.path.join(self.tmp.name, "daten")
        self.cwd = os.getcwd()
        os.chdir(self.tmp.name)
        self.addCleanup(os.chdir, self.cwd)

    def run_cli(self, *argv):
        out = io.StringIO()
        code = main(["--daten", self.data, "--backend", "fake", *argv], out=out, inp=lambda p: "j",
                    backend=FakeBackend(embed_dim=32))
        return code, out.getvalue()

    def test_system(self):
        code, out = self.run_cli("system")
        self.assertEqual(code, 0)
        self.assertIn("CPU", out)
        self.assertIn("RAM", out)
        code, out = self.run_cli("system", "--json")
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertIn("system", data)
        self.assertIn("hinweise", data)

    def test_geraete(self):
        code, out = self.run_cli("geraete")
        self.assertEqual(code, 2)
        orig = devmod.scan
        devmod.scan = lambda: [devmod._make_device("seriell", "CH340", port="COM3", vid="1a86", pid="7523", raw={})]
        try:
            code, out = self.run_cli("geraete", "scan")
            self.assertEqual(code, 0)
            self.assertIn("COM3", out)
            code, out = self.run_cli("geraete", "scan", "--json")
            self.assertEqual(json.loads(out)[0]["port"], "COM3")
        finally:
            devmod.scan = orig
        code, out = self.run_cli("geraete", "list")
        self.assertEqual(code, 0)
        self.assertIn("COM3", out)
        code, out = self.run_cli("geraete", "notiz", "COM3", "Prüfstand", "links")
        self.assertEqual(code, 0)
        code, out = self.run_cli("geraete", "list")
        self.assertIn("Prüfstand links", out)
        self.assertEqual(self.run_cli("geraete", "notiz", "COM9", "x")[0], 1)
        self.assertEqual(self.run_cli("geraete", "lesen", "COM3", "--baud", "1234")[0], 1)
        self.assertEqual(self.run_cli("geraete", "vergessen", "COM3")[0], 0)
        self.assertEqual(self.run_cli("geraete", "vergessen", "COM3")[0], 1)
        code, out = self.run_cli("geraete", "list")
        self.assertIn("Keine Geräte", out)

    def test_modell3d(self):
        code, out = self.run_cli("modell3d", "arten")
        self.assertEqual(code, 0)
        self.assertIn("lochplatte", out)
        code, out = self.run_cli("modell3d", "neu", "lochplatte", "l=100", "b=50", "t=3", "holes=-30/0/10;30/0/10",
                                 "--name", "Platte", "--material", "alu6061", "--projekt", "Drohne")
        self.assertEqual(code, 0, out)
        self.assertIn("Erzeugt: #1 Platte v1", out)
        self.assertIn("wasserdicht ja", out)
        self.assertIn("Masse", out)
        code, out = self.run_cli("modell3d", "neu", "kugel", "d=-3")
        self.assertEqual(code, 1)
        self.assertIn("Fehler", out)
        code, out = self.run_cli("modell3d", "list", "--projekt", "Drohne")
        self.assertIn("#1 Platte", out)
        code, out = self.run_cli("modell3d", "info", "1")
        self.assertEqual(code, 0)
        self.assertIn("Schwerpunkt", out)
        self.assertEqual(self.run_cli("modell3d", "info", "99")[0], 1)
        code, out = self.run_cli("modell3d", "export", "1", "platte.obj")
        self.assertEqual(code, 0)
        self.assertTrue(os.path.isfile("platte.obj"))
        self.assertGreater(geometry.read_obj("platte.obj").volume(), 0)
        code, out = self.run_cli("modell3d", "export", "1", "platte.stl", "--ascii")
        self.assertTrue(open("platte.stl", "rb").read().startswith(b"solid"))
        geometry.write_stl(geometry.cylinder(10, 5), "extern.stl")
        code, out = self.run_cli("modell3d", "import", "extern.stl", "--name", "Bolzen")
        self.assertEqual(code, 0, out)
        self.assertIn("Importiert: #2 Bolzen", out)
        self.assertEqual(self.run_cli("modell3d", "import", "fehlt.stl")[0], 1)
        self.assertEqual(self.run_cli("modell3d", "loeschen", "1")[0], 0)
        self.assertEqual(self.run_cli("modell3d", "loeschen", "1")[0], 1)

    def test_simulation(self):
        code, out = self.run_cli("simulation")
        self.assertEqual(code, 0)
        self.assertIn("schwebeflug", out)
        code, out = self.run_cli("simulation", "schwebeflug", "mass_g=1200", "cells=4", "mah=1500", "hover_current_a=15",
                                 "--csv", "flug.csv")
        self.assertEqual(code, 0, out)
        self.assertIn("Flugzeit", out)
        with open("flug.csv", encoding="utf-8") as fh:
            lines = fh.read().splitlines()
        self.assertTrue(lines[0].startswith("zeit_s;spannung_v"))
        self.assertGreater(len(lines), 100)
        code, out = self.run_cli("simulation", "balken", "force_n=10", "length_m=0.2", "material=cfk", "width_m=0.02",
                                 "heights_mm=2;4;6", "--json")
        self.assertEqual(code, 0, out)
        self.assertEqual(json.loads(out)["x"], [2.0, 4.0, 6.0])
        self.assertEqual(self.run_cli("simulation", "tornado")[0], 1)
        self.assertEqual(self.run_cli("simulation", "fall", "mass_kg=1")[0], 1)

    def test_geo(self):
        self.assertEqual(self.run_cli("geo")[0], 2)
        code, out = self.run_cli("geo", "ort", "Berlin", "52.52, 13.405", "--notiz", "Start")
        self.assertEqual(code, 0, out)
        self.assertIn("#1 Berlin", out)
        self.assertEqual(self.run_cli("geo", "ort", "X", "99, 1")[0], 1)
        code, out = self.run_cli("geo", "orte")
        self.assertIn("Berlin", out)
        self.assertIn("Start", out)
        code, out = self.run_cli("geo", "distanz", "Berlin", "48.137, 11.575")
        self.assertEqual(code, 0, out)
        self.assertIn("504", out)
        self.assertEqual(self.run_cli("geo", "distanz", "Berlin", "Atlantis")[0], 1)
        code, out = self.run_cli("geo", "plan", "Berlin; 52.6, 13.5", "--geschwindigkeit", "15", "--wind", "20", "--wind-aus", "90")
        self.assertEqual(code, 0, out)
        self.assertIn("Flugplan", out)
        code, out = self.run_cli("geo", "sonne", "Berlin", "--zeit", "2026-06-21")
        self.assertEqual(code, 0, out)
        self.assertIn("Aufgang 02:4", out)
        code, out = self.run_cli("geo", "utm", "Berlin")
        self.assertIn("33U", out)
        self.assertEqual(self.run_cli("geo", "utm", "Atlantis")[0], 1)
        code, out = self.run_cli("geo", "route", "Tour", "Berlin; 48.137,11.575", "--projekt", "T")
        self.assertEqual(code, 0, out)
        self.assertIn("504", out)
        code, out = self.run_cli("geo", "routen", "--projekt", "T")
        self.assertIn("Tour", out)
        self.assertEqual(self.run_cli("geo", "route", "X", "Berlin; Atlantis")[0], 1)
        code, out = self.run_cli("geo", "wetter", "Berlin")
        self.assertEqual(code, 0)
        self.assertIn("ausgeschaltet", out)
        self.assertEqual(self.run_cli("geo", "loeschen", "route", "1")[0], 0)
        self.assertEqual(self.run_cli("geo", "loeschen", "route", "1")[0], 1)
        self.assertEqual(self.run_cli("geo", "loeschen", "ort", "1")[0], 0)

    def test_help_lists_phase3(self):
        import contextlib
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            try:
                main(["--hilfe"], out=buf)
            except SystemExit:
                pass
        text = buf.getvalue()
        for cmd in ("system", "geraete", "modell3d", "simulation", "geo", "app"):
            self.assertIn(cmd, text)


class ChatCommandTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = Config(backend="fake", data_dir=os.path.join(self.tmp.name, "d"), workspace=self.tmp.name)
        self.brain = Brain(self.cfg, backend=FakeBackend(embed_dim=32))
        self.addCleanup(self.brain.close)
        self.out = io.StringIO()
        self.session = ChatSession(self.brain, self.cfg, out=self.out, inp=lambda p: "j")

    def text(self) -> str:
        value = self.out.getvalue()
        self.out.seek(0)
        self.out.truncate(0)
        return value

    def test_system(self):
        self.session.handle_line("/system")
        t = self.text()
        self.assertIn("CPU", t)
        self.assertEqual(len(self.brain.sysmon), 1)

    def test_devices(self):
        self.session.handle_line("/geraete")
        self.assertIn("Keine Geräte", self.text())
        orig = devmod.scan
        devmod.scan = lambda: [devmod._make_device("seriell", "CH340", port="COM3", vid="1a86", pid="7523", raw={})]
        try:
            self.session.handle_line("/geraete scan")
        finally:
            devmod.scan = orig
        self.assertIn("COM3", self.text())
        self.session.handle_line("/geraete")
        self.assertIn("COM3", self.text())
        self.session.handle_line("/geraete lesen")
        self.assertIn("Aufruf", self.text())
        self.session.handle_line("/geraete lesen COM3 1234")
        self.assertIn("Baudrate", self.text())

    def test_model3d(self):
        self.session.handle_line("/modell3d")
        self.assertIn("Keine 3D-Modelle", self.text())
        self.session.handle_line("/modell3d rohr d_outer=20 d_inner=10 h=30 name=Rohr material=cfk")
        t = self.text()
        self.assertIn("Erzeugt: #1 Rohr v1", t)
        self.assertIn("Masse", t)
        self.session.handle_line("/modell3d liste")
        self.assertIn("#1 Rohr", self.text())
        self.session.handle_line("/modell3d info 1")
        self.assertIn("Datei:", self.text())
        self.session.handle_line("/modell3d info 9")
        self.assertIn("unbekannt", self.text())
        self.session.handle_line("/modell3d kugel d=-1")
        self.assertIn("✗", self.text())

    def test_simulation(self):
        self.session.handle_line("/simulation")
        self.assertIn("schwebeflug", self.text())
        self.session.handle_line("/simulation fall mass_kg=0.5 height_m=20 area_m2=0.02")
        self.assertIn("Aufprallgeschwindigkeit", self.text())
        self.session.handle_line("/simulation fall mass_kg=0.5")
        self.assertIn("✗", self.text())

    def test_geo(self):
        self.session.handle_line("/geo")
        self.assertIn("Aufruf", self.text())
        self.session.handle_line("/geo orte")
        self.assertIn("Keine Orte", self.text())
        self.brain.geo.add_place("Berlin", 52.52, 13.405)
        self.session.handle_line("/geo distanz Berlin ; 48.137,11.575")
        self.assertIn("504", self.text())
        self.session.handle_line("/geo distanz Berlin")
        self.assertIn("Aufruf", self.text())
        self.session.handle_line("/geo sonne Berlin 2026-06-21")
        self.assertIn("Aufgang 02:4", self.text())
        self.session.handle_line("/geo wetter Berlin")
        self.assertIn("ausgeschaltet", self.text())
        self.session.handle_line("/geo route Berlin; 52.6,13.5 15")
        self.assertIn("Flugplan", self.text())
        self.session.handle_line("/geo quatsch")
        self.assertIn("Unbekannt", self.text())

    def test_help_mentions_commands(self):
        self.session.handle_line("/hilfe")
        t = self.text()
        for cmd in ("/system", "/geraete", "/modell3d", "/simulation", "/geo"):
            self.assertIn(cmd, t)


if __name__ == "__main__":
    unittest.main()
