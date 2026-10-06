"""Phase-3-Routen des Servers: System, Geräte, 3D-Modelle, Simulation, Welt."""

from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
import urllib.error
import urllib.request

from obito import devices as devmod
from obito import geometry
from obito.brain import Brain
from obito.config import Config
from obito.llm import FakeBackend
from obito.server import ObitoServer


class ServerPhase3Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.cfg = Config(backend="fake", data_dir=cls.tmp.name, workspace=cls.tmp.name)
        cls.brain = Brain(cls.cfg, backend=FakeBackend(embed_dim=32))
        cls.server = ObitoServer(cls.brain, "127.0.0.1", 0)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.port}"
        cls.opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()
        cls.brain.close()
        cls.tmp.cleanup()

    def call(self, path, body=None, method=None, raw=False):
        data = json.dumps(body).encode("utf-8") if body is not None else None
        req = urllib.request.Request(self.base + path, data=data,
                                     method=method or ("POST" if body is not None else "GET"))
        req.add_header("Content-Type", "application/json")
        try:
            with self.opener.open(req, timeout=60) as r:
                payload = r.read()
                return (r.status, payload, dict(r.headers)) if raw else (r.status, json.loads(payload))
        except urllib.error.HTTPError as e:
            payload = e.read()
            return (e.code, payload, dict(e.headers)) if raw else (e.code, json.loads(payload))

    # ------------------------------------------------------------ System
    def test_01_system(self):
        status, body = self.call("/api/system")
        self.assertEqual(status, 200)
        self.assertIn("cpu", body["system"])
        self.assertIn("ram", body["system"])
        self.assertIsInstance(body["hinweise"], list)
        self.assertIn("CPU", body["text"])
        status, body = self.call("/api/system/verlauf")
        self.assertEqual(status, 200)
        self.assertGreaterEqual(len(body["verlauf"]["zeit"]), 1)
        self.assertEqual(len(body["verlauf"]["cpu"]), len(body["verlauf"]["zeit"]))

    # ------------------------------------------------------------ Geräte
    def test_02_devices(self):
        orig = devmod.scan
        devmod.scan = lambda: [devmod._make_device("seriell", "STM32 VCP", port="COM7", vid="0483", pid="5740", raw={})]
        try:
            status, body = self.call("/api/geraete/scan", {})
        finally:
            devmod.scan = orig
        self.assertEqual(status, 200)
        self.assertEqual(len(body["gefunden"]), 1)
        self.assertEqual(body["gefunden"][0]["rolle"], "flugsteuerung")
        self.assertIn("COM7", body["text"])
        status, body = self.call("/api/geraete")
        self.assertEqual(status, 200)
        self.assertEqual(body["statistik"]["gesamt"], 1)
        key = body["geraete"][0]["key"]
        status, body = self.call("/api/geraete/notiz", {"key": key, "notiz": "Prüfstand"})
        self.assertEqual(status, 200)
        self.assertEqual(body["geraet"]["notiz"], "Prüfstand")
        self.assertEqual(self.call("/api/geraete/notiz", {"key": "nope", "notiz": "x"})[0], 404)
        # Nächster Scan ohne Geräte → getrennt
        devmod.scan = lambda: []
        try:
            status, body = self.call("/api/geraete/scan", {})
        finally:
            devmod.scan = orig
        self.assertEqual(body["geraete"][0]["status"], "getrennt")
        status, body = self.call("/api/geraete?verbunden=1")
        self.assertEqual(body["geraete"], [])
        status, body = self.call("/api/geraete/" + urllib.request.quote(key, safe=""), method="DELETE")
        self.assertEqual(status, 200)
        self.assertEqual(self.call("/api/geraete/" + urllib.request.quote(key, safe=""), method="DELETE")[0], 404)

    def test_03_serial_read_validation_and_fake_port(self):
        self.assertEqual(self.call("/api/geraete/lesen", {"port": "COM1", "baud": 1234})[0], 400)
        self.assertEqual(self.call("/api/geraete/lesen", {"baud": 9600})[0], 400)

        class _Reader:
            def __init__(self):
                self.data = [b"akku=15.8\nstrom=12.5\n"]

            def read(self, n, timeout=0.1):
                return self.data.pop(0) if self.data else b""

            def close(self):
                pass

        orig = devmod._open_serial
        devmod._open_serial = lambda port, baud: _Reader()
        try:
            status, body = self.call("/api/geraete/lesen", {"port": "/dev/ttyFAKE", "sekunden": 0.2})
        finally:
            devmod._open_serial = orig
        self.assertEqual(status, 200, body)
        self.assertEqual(body["lesung"]["baud"], self.cfg.serial_baud)
        self.assertEqual(body["telemetrie"]["format"], "kv")
        self.assertEqual(body["telemetrie"]["werte"]["akku"], 15.8)

    # ------------------------------------------------------------ 3D
    def test_04_models3d(self):
        status, body = self.call("/api/modelle3d/arten")
        self.assertEqual(status, 200)
        self.assertIn("quader", [a["art"] for a in body["arten"]])
        status, body = self.call("/api/modelle3d", {"art": "rohr", "parameter": {"d_outer": 20, "d_inner": 10, "h": 30},
                                                    "name": "Rohr", "material": "cfk", "projekt": "Drohne"})
        self.assertEqual(status, 200, body)
        mid = body["modell"]["id"]
        self.assertEqual(body["modell"]["version"], 1)
        self.assertTrue(body["modell"]["statistik"]["wasserdicht"])
        self.assertIsNotNone(body["modell"]["statistik"]["masse_g"])
        status, body = self.call("/api/modelle3d", {"art": "rohr", "parameter": "d_outer=22, d_inner=10, h=30", "name": "Rohr",
                                                    "projekt": "Drohne"})
        self.assertEqual(body["modell"]["version"], 2)
        mid2 = body["modell"]["id"]
        status, body = self.call("/api/modelle3d?projekt=Drohne")
        self.assertEqual(len(body["modelle"]), 1)
        status, body = self.call("/api/modelle3d?name=Rohr&projekt=Drohne")
        self.assertEqual(len(body["modelle"]), 2)
        status, body = self.call(f"/api/modelle3d/{mid}")
        self.assertEqual(body["modell"]["art"], "rohr")
        status, body = self.call(f"/api/modelle3d/{mid2}/mesh")
        self.assertEqual(status, 200)
        self.assertGreater(len(body["mesh"]["dreiecke"]), 100)
        status, payload, headers = self.call(f"/api/modelle3d/{mid2}/stl", raw=True)
        self.assertEqual(status, 200)
        self.assertEqual(headers.get("Content-Type"), "model/stl")
        self.assertIn("Rohr_v2.stl", headers.get("Content-Disposition", ""))
        self.assertEqual(len(payload), 84 + 50 * len(body["mesh"]["dreiecke"]))
        self.assertEqual(self.call("/api/modelle3d/999")[0], 404)
        self.assertEqual(self.call("/api/modelle3d", {"art": "kugel", "parameter": {"d": -1}})[0], 400)
        self.assertEqual(self.call("/api/modelle3d", {"parameter": {}})[0], 400)
        status, body = self.call(f"/api/modelle3d/{mid}", method="DELETE")
        self.assertEqual(status, 200)
        self.assertEqual(self.call(f"/api/modelle3d/{mid}", method="DELETE")[0], 404)

    def test_05_model_import_from_workspace(self):
        path = os.path.join(self.tmp.name, "teil.stl")
        geometry.write_stl(geometry.cylinder(10, 5), path)
        status, body = self.call("/api/modelle3d", {"pfad": "teil.stl", "projekt": "Import"})
        self.assertEqual(status, 200, body)
        self.assertEqual(body["modell"]["name"], "teil")
        self.assertEqual(self.call("/api/modelle3d", {"pfad": "../../../../etc/passwd"})[0], 403)
        self.assertEqual(self.call("/api/modelle3d", {"pfad": "fehlt.stl"})[0], 404)

    # ------------------------------------------------------------ Simulation
    def test_06_simulation(self):
        status, body = self.call("/api/simulation/arten")
        self.assertEqual(status, 200)
        self.assertIn("schwebeflug", [a["art"] for a in body["arten"]])
        status, body = self.call("/api/simulation", {"art": "schwebeflug",
                                                     "parameter": {"mass_g": 1200, "cells": 4, "mah": 1500, "hover_current_a": 15}})
        self.assertEqual(status, 200, body)
        self.assertGreater(body["ergebnis"]["zusammenfassung"]["flugzeit_min"], 2)
        self.assertIn("Flugzeit", body["text"])
        status, body = self.call("/api/simulation", {"art": "fall", "parameter": "mass_kg=0.5, height_m=30, area_m2=0.02"})
        self.assertEqual(status, 200)
        self.assertEqual(self.call("/api/simulation", {"art": "tornado", "parameter": {}})[0], 400)
        self.assertEqual(self.call("/api/simulation", {"art": "fall", "parameter": {"mass_kg": 1}})[0], 400)
        self.assertEqual(self.call("/api/simulation", {"art": "fall", "parameter": 5})[0], 400)

    # ------------------------------------------------------------ Welt
    def test_07_geo_places_routes_plan(self):
        status, body = self.call("/api/geo/orte", {"name": "Berlin", "koordinate": "52.52, 13.405", "notiz": "Start"})
        self.assertEqual(status, 200, body)
        berlin = body["ort"]["id"]
        status, body = self.call("/api/geo/orte", {"name": "München", "lat": 48.137, "lon": 11.575, "projekt": "Tour"})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.call("/api/geo/orte", {"name": "X", "koordinate": "99, 0"})[0], 400)
        self.assertEqual(self.call("/api/geo/orte", {"name": "X"})[0], 400)
        status, body = self.call("/api/geo/orte")
        self.assertEqual(len(body["orte"]), 2)
        status, body = self.call("/api/geo/orte?projekt=Tour")
        self.assertEqual(len(body["orte"]), 1)
        status, body = self.call("/api/geo/routen", {"name": "Tour", "punkte": "Berlin; München", "projekt": "Tour"})
        self.assertEqual(status, 200, body)
        self.assertAlmostEqual(body["route"]["laenge_m"] / 1000, 504, delta=3)
        rid = body["route"]["id"]
        status, body = self.call("/api/geo/routen", {"name": "Kurz", "punkte": [[52.52, 13.405], {"lat": 52.53, "lon": 13.41}]})
        self.assertEqual(status, 200, body)
        self.assertEqual(self.call("/api/geo/routen", {"name": "X", "punkte": "Berlin; Atlantis"})[0], 400)
        self.assertEqual(self.call("/api/geo/routen", {"name": "X", "punkte": "Berlin"})[0], 400)
        status, body = self.call("/api/geo/routen")
        self.assertEqual(len(body["routen"]), 2)
        status, body = self.call("/api/geo/plan", {"punkte": "Berlin; 52.6, 13.5", "geschwindigkeit_m_s": 15,
                                                   "wind_kmh": 20, "wind_aus_deg": 90})
        self.assertEqual(status, 200, body)
        self.assertEqual(len(body["plan"]["abschnitte"]), 1)
        self.assertTrue(body["plan"]["fliegbar"])
        self.assertEqual(self.call("/api/geo/plan", {"punkte": "Berlin; München", "geschwindigkeit_m_s": "schnell"})[0], 400)
        self.assertEqual(self.call(f"/api/geo/routen/{rid}", method="DELETE")[0], 200)
        self.assertEqual(self.call(f"/api/geo/routen/{rid}", method="DELETE")[0], 404)
        self.assertEqual(self.call(f"/api/geo/orte/{berlin}", method="DELETE")[0], 200)
        self.assertEqual(self.call(f"/api/geo/orte/{berlin}", method="DELETE")[0], 404)

    def test_08_geo_sun(self):
        status, body = self.call("/api/geo/sonne?lat=52.52&lon=13.405&zeit=2024-06-21T12:00:00Z")
        self.assertEqual(status, 200, body)
        self.assertTrue(body["sonne"]["aufgang_utc"].startswith("2024-06-21T02:4"))
        self.assertEqual(len(body["sonne"]["terminator"]), 72)
        self.assertEqual(len(body["sonne"]["subsolar"]), 2)
        self.call("/api/geo/orte", {"name": "Hamburg", "koordinate": "53.55, 9.99"})
        status, body = self.call("/api/geo/sonne?ort=Hamburg")
        self.assertEqual(status, 200)
        self.assertEqual(body["sonne"]["breite"], 53.55)
        self.assertEqual(self.call("/api/geo/sonne?ort=Atlantis")[0], 400)
        self.assertEqual(self.call("/api/geo/sonne?lat=1")[0], 400)
        self.assertEqual(self.call("/api/geo/sonne?lat=52&lon=13&zeit=gestern")[0], 400)

    def test_09_geo_weather_offline_and_online(self):
        self.assertEqual(self.call("/api/geo/wetter?lat=52.52&lon=13.405")[0], 403)
        calls = []

        def fake_weather(lat, lon, **kw):
            calls.append((lat, lon))
            return {"breite": lat, "laenge": lon, "zeit": "t", "temperatur_c": 20.0, "luftfeuchte_prozent": 50,
                    "wind_kmh": 10.0, "wind_richtung_deg": 90, "boeen_kmh": 15.0, "niederschlag_mm": 0.0,
                    "bewoelkung_prozent": 10, "luftdruck_hpa": 1010, "wettercode": 0, "wetter": "klar",
                    "quelle": "Test", "flugtauglich": True, "gruende": []}

        self.cfg.online = True
        self.server.weather_fn = fake_weather
        try:
            status, body = self.call("/api/geo/wetter?lat=52.52&lon=13.405")
            self.assertEqual(status, 200, body)
            self.assertTrue(body["wetter"]["flugtauglich"])
            self.assertIn("Flugtauglich: ja", body["text"])
            self.assertEqual(len(calls), 1)

            def broken(lat, lon, **kw):
                raise RuntimeError("Wetterdienst nicht erreichbar")

            self.server.weather_fn = broken
            self.assertEqual(self.call("/api/geo/wetter?lat=52.52&lon=13.405")[0], 502)
        finally:
            self.cfg.online = False
            self.server.weather_fn = None

    def test_10_status_has_phase3_keys(self):
        status, body = self.call("/api/status")
        self.assertEqual(status, 200)
        for key in ("geraete", "modelle3d", "geo", "online"):
            self.assertIn(key, body)
        self.assertFalse(body["online"])


if __name__ == "__main__":
    unittest.main()
