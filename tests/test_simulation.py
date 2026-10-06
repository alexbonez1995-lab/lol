"""Tests für obito.simulation – Plausibilität der Modelle, Toleranz der Eingaben, Werkzeug."""

from __future__ import annotations

import math
import tempfile
import time
import unittest

from obito import simulation as sim
from obito.tools import ToolRegistry


def _monotone_decreasing(values):
    return all(b <= a + 1e-9 for a, b in zip(values, values[1:]))


class HoverTest(unittest.TestCase):
    def test_flight_time_plausible(self):
        r = sim.hover_flight(1200, 4, 1500, 15)
        s = r["zusammenfassung"]
        self.assertGreater(s["flugzeit_min"], 3.0)
        self.assertLess(s["flugzeit_min"], 6.0)
        self.assertTrue(s["grund"])
        self.assertTrue(_monotone_decreasing(r["reihen"]["spannung_v"]))
        self.assertTrue(_monotone_decreasing(r["reihen"]["soc_prozent"]))
        self.assertEqual(len(r["zeit_s"]), len(r["reihen"]["spannung_v"]))
        self.assertIn("spannung_v", r["einheiten"])

    def test_payload_shortens_and_usable_matters(self):
        base = sim.hover_flight(1200, 4, 1500, 15)["zusammenfassung"]["flugzeit_min"]
        heavy = sim.hover_flight(1200, 4, 1500, 15, payload_g=400)["zusammenfassung"]["flugzeit_min"]
        self.assertLess(heavy, base)
        little = sim.hover_flight(1200, 4, 1500, 5, usable=0.3)["zusammenfassung"]
        self.assertIn("30", little["grund"])
        self.assertGreater(little["restkapazitaet_prozent"], 60)

    def test_bigger_battery_flies_longer(self):
        a = sim.hover_flight(1200, 4, 1500, 15)["zusammenfassung"]["flugzeit_min"]
        b = sim.hover_flight(1200, 4, 3000, 15)["zusammenfassung"]["flugzeit_min"]
        self.assertGreater(b, a * 1.6)

    def test_errors(self):
        for kwargs in (dict(mass_g=0), dict(cells=0), dict(mah=-1), dict(hover_current_a=0), dict(usable=1.5),
                       dict(peukert=2.0), dict(payload_g=-5), dict(dt=0.001)):
            args = dict(mass_g=1200, cells=4, mah=1500, hover_current_a=15)
            args.update(kwargs)
            with self.assertRaises(ValueError, msg=str(kwargs)):
                sim.hover_flight(**args)

    def test_cell_ocv_curve(self):
        self.assertAlmostEqual(sim.cell_ocv(1.0), 4.20)
        self.assertAlmostEqual(sim.cell_ocv(0.0), 3.30)
        self.assertAlmostEqual(sim.cell_ocv(0.5), 3.80)
        self.assertAlmostEqual(sim.cell_ocv(0.95), 4.13, places=2)


class BatteryTest(unittest.TestCase):
    def test_time_until_empty(self):
        r = sim.battery_discharge(4, 1500, 1.5, usable=0.9, peukert=1.0)
        # 1,5 Ah bei 1,5 A → 60 min, 90 % nutzbar → ≈ 54 min (Abbruch über Spannung etwas früher möglich)
        self.assertGreater(r["zusammenfassung"]["zeit_bis_leer_min"], 45)
        self.assertLessEqual(r["zusammenfassung"]["zeit_bis_leer_min"], 54.5)
        self.assertAlmostEqual(r["zusammenfassung"]["c_rate"], 1.0)

    def test_high_c_rate_warning(self):
        r = sim.battery_discharge(4, 1000, 40)
        self.assertTrue(any("C" in w for w in r["warnungen"]))


class ClimbTest(unittest.TestCase):
    def test_reaches_target_without_large_overshoot(self):
        r = sim.climb_profile(1200, 30, 50)
        s = r["zusammenfassung"]
        self.assertIsNotNone(s["zeit_bis_hoehe_s"])
        self.assertLess(s["ueberschwingen_m"], 5.0)
        self.assertAlmostEqual(s["endhoehe_m"], 50, delta=1.0)
        self.assertGreater(s["max_steigrate_m_s"], 1.0)
        self.assertEqual(r["warnungen"], [])

    def test_too_little_thrust(self):
        with self.assertRaises(ValueError):
            sim.climb_profile(1200, 10, 50)

    def test_more_thrust_is_faster(self):
        slow = sim.climb_profile(1200, 15, 50)["zusammenfassung"]["zeit_bis_hoehe_s"]
        fast = sim.climb_profile(1200, 60, 50)["zusammenfassung"]["zeit_bis_hoehe_s"]
        self.assertLess(fast, slow)


class DropTest(unittest.TestCase):
    def test_vacuum_matches_formula(self):
        r = sim.drop_with_drag(0.5, 100, cd=0)
        s = r["zusammenfassung"]
        self.assertAlmostEqual(s["aufprallgeschwindigkeit_m_s"], math.sqrt(2 * 9.81 * 100), places=2)
        self.assertAlmostEqual(s["fallzeit_s"], math.sqrt(2 * 100 / 9.81), places=2)
        self.assertIsNone(s["grenzgeschwindigkeit_m_s"])

    def test_drag_slows_down(self):
        r = sim.drop_with_drag(0.5, 100, cd=1.0, area_m2=0.05)
        s = r["zusammenfassung"]
        self.assertLess(s["aufprallgeschwindigkeit_m_s"], s["ohne_luft_geschwindigkeit_m_s"])
        self.assertGreater(s["fallzeit_s"], s["ohne_luft_fallzeit_s"])
        self.assertLessEqual(s["aufprallgeschwindigkeit_m_s"], s["grenzgeschwindigkeit_m_s"] + 1e-6)
        self.assertEqual(r["reihen"]["hoehe_m"][-1], 0.0)

    def test_terminal_velocity_warning(self):
        r = sim.drop_with_drag(0.1, 2000, cd=1.0, area_m2=0.5)
        self.assertTrue(any("Grenz" in w for w in r["warnungen"]))

    def test_errors(self):
        with self.assertRaises(ValueError):
            sim.drop_with_drag(0, 10)
        with self.assertRaises(ValueError):
            sim.drop_with_drag(1, 10, cd=-1)


class ThermalTest(unittest.TestCase):
    def test_analytic_values(self):
        r = sim.thermal_rc(5, 50, 0.9, 10, 0.005, duration_s=600)
        s = r["zusammenfassung"]
        self.assertAlmostEqual(s["endtemperatur_c"], 25 + 5 / (10 * 0.005), places=3)
        self.assertAlmostEqual(s["zeitkonstante_s"], 50 * 0.9 / (10 * 0.005), delta=0.02 * 900)
        self.assertAlmostEqual(s["zeit_bis_80_prozent_s"], 900 * math.log(5), delta=1.0)
        # Nach 600 s: T = 25 + 100·(1 − e^(−600/900))
        expect = 25 + 100 * (1 - math.exp(-600 / 900))
        self.assertAlmostEqual(s["temperatur_nach_dauer_c"], expect, delta=0.5)
        self.assertTrue(any("100" in w for w in r["warnungen"]))

    def test_cool_part_no_warning(self):
        r = sim.thermal_rc(1, 100, 0.9, 20, 0.02)
        self.assertEqual(r["warnungen"], [])
        self.assertLess(r["zusammenfassung"]["endtemperatur_c"], 30)

    def test_errors(self):
        with self.assertRaises(ValueError):
            sim.thermal_rc(-1, 50, 0.9, 10, 0.005)
        with self.assertRaises(ValueError):
            sim.thermal_rc(5, 50, 0.9, 10, 0.005, duration_s=1e9)


class PidTest(unittest.TestCase):
    def test_p_only_has_steady_state_error(self):
        s = sim.pid_step(2, 0, 0)["zusammenfassung"]
        self.assertGreater(s["stationaerer_fehler"], 0.2)

    def test_pi_eliminates_error(self):
        r = sim.pid_step(2, 5, 0.05)
        s = r["zusammenfassung"]
        self.assertLess(abs(s["stationaerer_fehler"]), 0.01)
        self.assertIsNotNone(s["einschwingzeit_s"])
        self.assertIsNotNone(s["anstiegszeit_s"])
        self.assertEqual(r["warnungen"], [])

    def test_overshoot_warning(self):
        r = sim.pid_step(1, 20, 0)
        self.assertGreater(r["zusammenfassung"]["ueberschwingen_prozent"], 20)
        self.assertTrue(any("Überschwingen" in w for w in r["warnungen"]))

    def test_unstable_warning(self):
        r = sim.pid_step(0, 0, 80, plant_tau_s=0.05, dt=0.005)
        self.assertTrue(r["warnungen"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            sim.pid_step(-1, 0, 0)
        with self.assertRaises(ValueError):
            sim.pid_step(1, 0, 0, dt=0.5)
        with self.assertRaises(ValueError):
            sim.pid_step(1, 0, 0, setpoint=0)


class BeamSweepTest(unittest.TestCase):
    def test_deflection_scales_with_cubed_height(self):
        r = sim.beam_sweep(10, 0.2, "Alu 6061", 0.02, [2, 4])
        d = r["reihen"]["durchbiegung_mm"]
        self.assertAlmostEqual(d[0] / d[1], 8.0, places=2)
        self.assertEqual(r["x"], [2.0, 4.0])
        self.assertEqual(r["x_einheit"], "mm")
        self.assertIn("material", r["zusammenfassung"])

    def test_material_as_number_and_safe_height(self):
        r = sim.beam_sweep(10, 0.2, 70, 0.02, [2, 3, 5, 8])
        self.assertIsNone(r["zusammenfassung"]["min_hoehe_sicher_mm"])
        r2 = sim.beam_sweep(10, 0.2, "cfk", 0.02, "2; 3; 5; 8")
        self.assertIsNotNone(r2["zusammenfassung"]["min_hoehe_sicher_mm"])

    def test_errors(self):
        with self.assertRaises(ValueError):
            sim.beam_sweep(10, 0.2, "Unobtainium", 0.02, [2])
        with self.assertRaises(ValueError):
            sim.beam_sweep(10, 0.2, "cfk", 0.02, [])


class RunAndTextTest(unittest.TestCase):
    def test_thinning_keeps_ends(self):
        r = sim.battery_discharge(4, 20000, 2, dt=1.0)
        self.assertLessEqual(len(r["zeit_s"]), sim.MAX_POINTS)
        self.assertEqual(r["zeit_s"][0], 0.0)
        self.assertEqual(r["reihen"]["soc_prozent"][0], 100.0)

    def test_run_tolerant(self):
        r = sim.run("Schwebeflug", "mass_g=1200, cells=4, mah=1500, hover_current_a=15, usable=0,8")
        self.assertEqual(r["typ"], "schwebeflug")
        r = sim.run("pid", {"kp": "2", "ki": "5", "kd": "0,05"})
        self.assertEqual(r["typ"], "regler")
        r = sim.run("balken", "force_n=10, length_m=0.2, material=cfk, width_m=0.02, heights_mm=2; 4; 6")
        self.assertEqual(r["x"], [2.0, 4.0, 6.0])

    def test_run_errors(self):
        with self.assertRaises(ValueError):
            sim.run("tornado", {})
        with self.assertRaises(ValueError):
            sim.run("fall", {"mass_kg": 1})
        with self.assertRaises(ValueError):
            sim.run("fall", {"mass_kg": 1, "height_m": 10, "quatsch": 1})
        with self.assertRaises(ValueError):
            sim.run("fall", {"mass_kg": "abc", "height_m": 10})

    def test_step_limit(self):
        with self.assertRaises(ValueError):
            sim.thermal_rc(5, 50, 0.9, 10, 0.005, duration_s=86400, dt=0.01)

    def test_summary_text(self):
        text = sim.summary_text(sim.run("thermik", "power_w=5, mass_g=50, cp_j_per_gk=0,9, h_w_per_m2k=10, area_m2=0.005"))
        for word in ("Simulation", "thermik", "Endtemperatur", "125", "Warnung", "Annahmen", "°C"):
            self.assertIn(word, text)
        text2 = sim.summary_text(sim.run("balken", "force_n=10, length_m=0.2, material=cfk, width_m=0.02, heights_mm=2; 4"))
        self.assertIn("hoehe_mm", text2)

    def test_describe_lists_all(self):
        text = sim.describe()
        for key in sim.SIMULATIONS:
            self.assertIn(key, text)

    def test_runtime_is_short(self):
        t0 = time.time()
        for kind, params in (("schwebeflug", "mass_g=1200, cells=4, mah=1500, hover_current_a=15"),
                             ("steigflug", "mass_g=1200, thrust_max_n=30, target_alt_m=100"),
                             ("fall", "mass_kg=0.5, height_m=100, area_m2=0.02"),
                             ("regler", "kp=2, ki=5, kd=0.05")):
            sim.run(kind, params)
        self.assertLess(time.time() - t0, 5.0)


class ToolTest(unittest.TestCase):
    def test_tool(self):
        with tempfile.TemporaryDirectory() as tmp:
            reg = ToolRegistry(workspace=tmp)
            sim.register_tools(reg)
            tool = next(t for t in reg.list() if t.name == "simulation_starten")
            self.assertFalse(tool.dangerous)
            res = reg.run("simulation_starten", {"art": "fall", "parameter": "mass_kg=0.5, height_m=30, area_m2=0.02"})
            self.assertTrue(res.ok, res.error)
            self.assertIn("Aufprallgeschwindigkeit", res.output)
            bad = reg.run("simulation_starten", {"art": "fall", "parameter": "mass_kg=0.5"})
            self.assertFalse(bad.ok)
            self.assertIn("height_m", bad.error)


if __name__ == "__main__":
    unittest.main()
