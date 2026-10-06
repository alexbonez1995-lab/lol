"""Tests für obito.geo – Geodäsie, Sonne, UTM, Store, Wetter (lokaler Mock-Server), Werkzeuge."""

from __future__ import annotations

import json
import tempfile
import threading
import unittest
from datetime import date, datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

from obito import geo
from obito.tools import ToolRegistry

BERLIN = (52.52, 13.405)
MUENCHEN = (48.137, 11.575)


class ParseCoordTest(unittest.TestCase):
    def test_variants(self):
        cases = {
            "49.9576, 6.9294": (49.9576, 6.9294),
            "49,9576; 6,9294": (49.9576, 6.9294),
            "49,9576 6,9294": (49.9576, 6.9294),
            "49.9576N 6.9294E": (49.9576, 6.9294),
            "N49.9576 E6.9294": (49.9576, 6.9294),
            "6.9294E 49.9576N": (49.9576, 6.9294),
            "33.9S 151.2E": (-33.9, 151.2),
            "-33.9, 151.2": (-33.9, 151.2),
            "52.52 13.405": (52.52, 13.405),
        }
        for text, expect in cases.items():
            lat, lon = geo.parse_coord(text)
            self.assertAlmostEqual(lat, expect[0], places=4, msg=text)
            self.assertAlmostEqual(lon, expect[1], places=4, msg=text)

    def test_dms(self):
        lat, lon = geo.parse_coord("49°57'27\"N 6°55'46\"E")
        self.assertAlmostEqual(lat, 49.9575, places=4)
        self.assertAlmostEqual(lon, 6.92944, places=4)
        lat, lon = geo.parse_coord("49°57.45'S 6°55.77'W")
        self.assertLess(lat, 0)
        self.assertLess(lon, 0)

    def test_tuple_input_and_errors(self):
        self.assertEqual(geo.parse_coord((1, 2)), (1.0, 2.0))
        for bad in ("", "abc", "95, 10", "10, 200", "1,2,3", "49.9"):
            with self.assertRaises(ValueError, msg=bad):
                geo.parse_coord(bad)

    def test_format(self):
        self.assertEqual(geo.format_coord(52.52, 13.405), "52.52000° N, 13.40500° E")
        self.assertIn("W", geo.format_coord(-10, -20, dms=True))
        self.assertIn("S", geo.format_coord(-10, -20, dms=True))

    def test_parse_points(self):
        pts = geo.parse_points("52.52,13.405; 52.53,13.41\n52.54, 13.40")
        self.assertEqual(len(pts), 3)
        with self.assertRaises(ValueError):
            geo.parse_points("; ".join(["1,1"] * (geo.MAX_ROUTE_POINTS + 1)))


class GeodesyTest(unittest.TestCase):
    def test_distance_berlin_munich(self):
        d = geo.distance_m(*BERLIN, *MUENCHEN)
        self.assertAlmostEqual(d / 1000, 504.3, delta=3)
        self.assertEqual(geo.distance_m(1, 1, 1, 1), 0.0)

    def test_bearing(self):
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 1, 0), 0.0, places=6)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, 0, 1), 90.0, places=6)
        self.assertAlmostEqual(geo.bearing_deg(0, 0, -1, 0), 180.0, places=6)
        self.assertAlmostEqual(geo.bearing_deg(*BERLIN, *MUENCHEN), 195.6, delta=0.5)

    def test_destination_roundtrip(self):
        lat, lon = geo.destination(*BERLIN, 135, 12_345)
        self.assertAlmostEqual(geo.distance_m(*BERLIN, lat, lon), 12_345, delta=0.01)
        self.assertAlmostEqual(geo.bearing_deg(*BERLIN, lat, lon), 135, delta=0.05)
        lat, lon = geo.destination(0, 179.9, 90, 50_000)
        self.assertLess(lon, 0)      # Datumsgrenze

    def test_route_length(self):
        self.assertAlmostEqual(geo.route_length_m([BERLIN, MUENCHEN, BERLIN]) / 1000, 2 * 504.3, delta=6)
        self.assertEqual(geo.route_length_m([BERLIN]), 0.0)

    def test_polygon_area_square(self):
        # ≈ 1 km × 1 km bei 52° N
        dlat = 1000 / 111_195
        dlon = 1000 / (111_195 * __import__("math").cos(__import__("math").radians(52)))
        sq = [(52, 13), (52, 13 + dlon), (52 + dlat, 13 + dlon), (52 + dlat, 13)]
        self.assertAlmostEqual(geo.polygon_area_m2(sq), 1_000_000, delta=10_000)
        with self.assertRaises(ValueError):
            geo.polygon_area_m2([(1, 1), (2, 2)])


class FlightPlanTest(unittest.TestCase):
    def test_wind_components(self):
        north = [(52.52, 13.405), (52.53, 13.405)]
        calm = geo.flight_plan(north, 10)
        tail = geo.flight_plan(north, 10, wind_speed_m_s=5, wind_from_deg=180)
        head = geo.flight_plan(north, 10, wind_speed_m_s=5, wind_from_deg=0)
        self.assertAlmostEqual(tail["abschnitte"][0]["bodengeschwindigkeit_m_s"], 15.0, places=2)
        self.assertAlmostEqual(head["abschnitte"][0]["bodengeschwindigkeit_m_s"], 5.0, places=2)
        self.assertLess(tail["zeit_s"], calm["zeit_s"])
        self.assertGreater(head["zeit_s"], calm["zeit_s"])
        self.assertTrue(calm["fliegbar"])

    def test_not_flyable(self):
        north = [(52.52, 13.405), (52.53, 13.405)]
        plan = geo.flight_plan(north, 4, wind_speed_m_s=5, wind_from_deg=0)
        self.assertFalse(plan["fliegbar"])
        self.assertIsNone(plan["abschnitte"][0]["zeit_s"])
        self.assertTrue(plan["warnungen"])

    def test_hover_time_and_errors(self):
        plan = geo.flight_plan([BERLIN, MUENCHEN], 20, hover_s_per_point=30)
        self.assertAlmostEqual(plan["zeit_s"], plan["abschnitte"][0]["zeit_s"] + 60, delta=0.2)
        with self.assertRaises(ValueError):
            geo.flight_plan([BERLIN], 10)
        with self.assertRaises(ValueError):
            geo.flight_plan([BERLIN, MUENCHEN], 0)
        with self.assertRaises(ValueError):
            geo.flight_plan([BERLIN, MUENCHEN], 10, wind_speed_m_s=-1)


class SunTest(unittest.TestCase):
    def test_berlin_summer_solstice(self):
        s = geo.sun(*BERLIN, datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc))
        rise = datetime.fromisoformat(s["aufgang_utc"])
        sset = datetime.fromisoformat(s["untergang_utc"])
        self.assertAlmostEqual((rise - datetime(2024, 6, 21, 2, 43, tzinfo=timezone.utc)).total_seconds(), 0, delta=300)
        self.assertAlmostEqual((sset - datetime(2024, 6, 21, 19, 33, tzinfo=timezone.utc)).total_seconds(), 0, delta=300)
        self.assertGreater(s["tageslaenge_h"], 16.5)
        self.assertGreater(s["hoehe_deg"], 55)
        self.assertTrue(s["ueber_horizont"])
        self.assertIsNone(s["hinweis"])
        self.assertNotIn("aufgang_lokal", s)

    def test_local_times_with_timezone(self):
        tz = timezone(timedelta(hours=1))
        s = geo.sun(*BERLIN, datetime(2024, 12, 21, 12, 0, tzinfo=tz))
        self.assertIn("aufgang_lokal", s)
        self.assertTrue(s["aufgang_lokal"].endswith("+01:00"))
        self.assertLess(s["tageslaenge_h"], 8.5)
        self.assertLess(s["hoehe_deg"], 20)

    def test_polar_day_and_night(self):
        s = geo.sun(78.22, 15.63, date(2024, 6, 21))
        self.assertIsNone(s["aufgang_utc"])
        self.assertIn("Polartag", s["hinweis"])
        self.assertEqual(s["tageslaenge_h"], 24.0)
        n = geo.sun(78.22, 15.63, date(2024, 12, 21))
        self.assertIn("Polarnacht", n["hinweis"])
        self.assertEqual(n["tageslaenge_h"], 0.0)

    def test_night_elevation_negative(self):
        s = geo.sun(*BERLIN, datetime(2024, 6, 21, 0, 30, tzinfo=timezone.utc))
        self.assertLess(s["hoehe_deg"], 0)
        self.assertFalse(s["ueber_horizont"])

    def test_default_now_and_terminator(self):
        s = geo.sun(*BERLIN)
        self.assertIn("hoehe_deg", s)
        pts = geo.terminator(datetime(2024, 6, 21, 12, tzinfo=timezone.utc))
        self.assertEqual(len(pts), 72)
        for lat, lon in pts:
            self.assertTrue(-90 <= lat <= 90 and -180 <= lon <= 180)
        slat, slon = geo.subsolar_point(datetime(2024, 6, 21, 12, tzinfo=timezone.utc))
        self.assertAlmostEqual(slat, 23.44, delta=0.05)
        self.assertAlmostEqual(slon, 0.0, delta=1.0)

    def test_format_sun(self):
        text = geo.format_sun(geo.sun(*BERLIN, datetime(2024, 6, 21, 12, 0, tzinfo=timezone.utc)))
        self.assertIn("Aufgang 02:43", text)
        self.assertIn("Höhe", text)


class UtmTest(unittest.TestCase):
    def test_berlin(self):
        u = geo.utm(52.516275, 13.377704)
        self.assertEqual(u["zone"], 33)
        self.assertEqual(u["band"], "U")
        self.assertAlmostEqual(u["ostwert_m"], 389_918, delta=100)
        self.assertAlmostEqual(u["nordwert_m"], 5_819_699, delta=100)
        self.assertEqual(u["hemisphaere"], "N")
        self.assertTrue(u["text"].startswith("33U"))

    def test_central_meridian_and_south(self):
        u = geo.utm(0, 3)
        self.assertAlmostEqual(u["ostwert_m"], 500_000, places=2)
        self.assertAlmostEqual(u["nordwert_m"], 0, places=2)
        s = geo.utm(-33.8688, 151.2093)
        self.assertEqual(s["zone"], 56)
        self.assertEqual(s["hemisphaere"], "S")
        self.assertGreater(s["nordwert_m"], 6_000_000)

    def test_norway_exception_and_limits(self):
        self.assertEqual(geo.utm_zone(60.0, 5.0)[0], 32)
        self.assertEqual(geo.utm_zone(75.0, 10.0)[0], 33)
        with self.assertRaises(ValueError):
            geo.utm(85, 10)


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = geo.WaypointStore(Path(self.tmp.name) / "geo.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_places(self):
        p = self.store.add_place("Berlin", *BERLIN, note="Hauptstadt")
        self.assertEqual(p["name"], "Berlin")
        self.assertIn("koordinate", p)
        self.assertEqual(self.store.find_place("berlin")["id"], p["id"])
        self.assertIsNone(self.store.find_place("Paris"))
        self.assertEqual(len(self.store.list_places()), 1)
        self.assertTrue(self.store.delete_place(p["id"]))
        self.assertFalse(self.store.delete_place(p["id"]))
        with self.assertRaises(ValueError):
            self.store.add_place("", 1, 2)
        with self.assertRaises(ValueError):
            self.store.add_place("x", 95, 2)

    def test_project_place_preferred(self):
        self.store.add_place("Start", 1, 1)
        self.store.add_place("Start", 2, 2, project="p1")
        self.assertEqual(self.store.find_place("start", "p1")["lat"], 2)
        self.assertEqual(len(self.store.list_places(project="p1")), 1)

    def test_routes_and_stats(self):
        r = self.store.add_route("Test", "52.52,13.405; 52.53,13.41; 52.54,13.40", project="p1")
        self.assertEqual(len(r["punkte"]), 3)
        self.assertGreater(r["laenge_m"], 2000)
        self.assertEqual(self.store.get_route(r["id"])["name"], "Test")
        self.assertEqual(len(self.store.list_routes(project="p1")), 1)
        st = self.store.stats()
        self.assertEqual(st["routen"], 1)
        self.assertTrue(self.store.delete_route(r["id"]))
        with self.assertRaises(ValueError):
            self.store.add_route("x", "1,1")

    def test_close_idempotent(self):
        self.store.close()
        self.store.close()
        with self.assertRaises(RuntimeError):
            self.store.list_places()


class _WeatherHandler(BaseHTTPRequestHandler):
    payload: dict | None = None
    status = 200
    raw: bytes | None = None
    last_path = ""

    def do_GET(self):
        _WeatherHandler.last_path = self.path
        body = self.raw if self.raw is not None else json.dumps(self.payload or {}).encode()
        self.send_response(self.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *a):  # noqa: D401
        pass


class WeatherTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = HTTPServer(("127.0.0.1", 0), _WeatherHandler)
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()
        cls.base = f"http://127.0.0.1:{cls.server.server_address[1]}"

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        _WeatherHandler.status = 200
        _WeatherHandler.raw = None
        _WeatherHandler.payload = {"current": {"time": "2026-06-21T12:00", "temperature_2m": 21.5,
                                               "relative_humidity_2m": 55, "wind_speed_10m": 12.0,
                                               "wind_direction_10m": 230, "wind_gusts_10m": 20.0,
                                               "precipitation": 0.0, "cloud_cover": 40, "weather_code": 2,
                                               "pressure_msl": 1013.2}}

    def _opener(self):
        import urllib.request
        return urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def test_ok(self):
        w = geo.weather(*BERLIN, base_url=self.base, opener=self._opener())
        self.assertEqual(w["temperatur_c"], 21.5)
        self.assertEqual(w["wind_kmh"], 12.0)
        self.assertEqual(w["wetter"], "teils bewölkt")
        self.assertTrue(w["flugtauglich"])
        self.assertIn("Open-Meteo", w["quelle"])
        self.assertIn("latitude=52.5200", _WeatherHandler.last_path)
        text = geo.format_weather(w)
        self.assertIn("Flugtauglich: ja", text)

    def test_not_flyable(self):
        _WeatherHandler.payload["current"].update({"wind_speed_10m": 40.0, "wind_gusts_10m": 60.0,
                                                   "precipitation": 1.2, "weather_code": 95})
        w = geo.weather(*BERLIN, base_url=self.base, opener=self._opener())
        self.assertFalse(w["flugtauglich"])
        self.assertEqual(len(w["gruende"]), 4)
        self.assertEqual(w["wetter"], "Gewitter")

    def test_http_error(self):
        _WeatherHandler.status = 500
        with self.assertRaises(RuntimeError):
            geo.weather(*BERLIN, base_url=self.base, opener=self._opener())

    def test_invalid_json_and_missing_current(self):
        _WeatherHandler.raw = b"kein json"
        with self.assertRaises(RuntimeError):
            geo.weather(*BERLIN, base_url=self.base, opener=self._opener())
        _WeatherHandler.raw = None
        _WeatherHandler.payload = {"foo": 1}
        with self.assertRaises(RuntimeError):
            geo.weather(*BERLIN, base_url=self.base, opener=self._opener())

    def test_unreachable(self):
        with self.assertRaises(RuntimeError):
            geo.weather(*BERLIN, base_url="http://127.0.0.1:1", timeout=1, opener=self._opener())

    def test_flight_conditions_helper(self):
        self.assertTrue(geo.flight_conditions(10, 20, 0, 1)["flugtauglich"])
        self.assertFalse(geo.flight_conditions(None, None, 0.8, None)["flugtauglich"])
        self.assertEqual(geo.weather_text(None), "unbekannt")
        self.assertIn("Wettercode", geo.weather_text(123))


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = ToolRegistry(workspace=self.tmp.name)
        self.store = geo.WaypointStore(Path(self.tmp.name) / "geo.db")
        self.store.add_place("Berlin", *BERLIN)
        self.store.add_place("München", *MUENCHEN)
        self.online = False
        self.weather_calls = []

        def fake_weather(lat, lon, **kw):
            self.weather_calls.append((lat, lon))
            return {"breite": lat, "laenge": lon, "zeit": "t", "temperatur_c": 20.0, "luftfeuchte_prozent": 50,
                    "wind_kmh": 10.0, "wind_richtung_deg": 90, "boeen_kmh": 15.0, "niederschlag_mm": 0.0,
                    "bewoelkung_prozent": 10, "luftdruck_hpa": 1010, "wettercode": 0, "wetter": "klar",
                    "quelle": "Test", "flugtauglich": True, "gruende": []}

        geo.register_tools(self.reg, self.store, lambda: self.online, weather_fn=fake_weather)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_registered_not_dangerous(self):
        names = {t.name for t in self.reg.list()}
        for n in geo.TOOL_NAMES:
            self.assertIn(n, names)
        self.assertTrue(all(not t.dangerous for t in self.reg.list()))

    def test_distance_with_names_and_coords(self):
        res = self.reg.run("geo_distanz", {"von": "Berlin", "nach": "münchen"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("504", res.output)
        self.assertIn("UTM", res.output)
        res = self.reg.run("geo_distanz", {"von": "52.52, 13.405", "nach": "Nirgendwo"})
        self.assertFalse(res.ok)
        self.assertIn("Nirgendwo", res.error)

    def test_route_tool(self):
        res = self.reg.run("geo_route", {"punkte": "Berlin; 52.6,13.5", "geschwindigkeit_m_s": 15,
                                         "wind_kmh": 20, "wind_aus_deg": 90})
        self.assertTrue(res.ok, res.error)
        self.assertIn("Flugplan", res.output)
        self.assertIn("Kurs", res.output)
        res = self.reg.run("geo_route", {"punkte": "Berlin", "geschwindigkeit_m_s": 15})
        self.assertFalse(res.ok)

    def test_sun_tool(self):
        res = self.reg.run("sonnenstand", {"ort": "Berlin", "datum": "21.06.2026 14:30"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("Berlin", res.output)
        self.assertIn("Aufgang 02:4", res.output)
        res = self.reg.run("sonnenstand", {"ort": "Berlin", "datum": "gestern"})
        self.assertFalse(res.ok)

    def test_weather_tool_offline_and_online(self):
        res = self.reg.run("wetter", {"ort": "Berlin"})
        self.assertTrue(res.ok)
        self.assertIn("ausgeschaltet", res.output)
        self.assertEqual(self.weather_calls, [])
        self.online = True
        res = self.reg.run("wetter", {"ort": "Berlin"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("Flugtauglich: ja", res.output)
        self.assertIn("Berlin", res.output)
        self.assertEqual(len(self.weather_calls), 1)


if __name__ == "__main__":
    unittest.main()
