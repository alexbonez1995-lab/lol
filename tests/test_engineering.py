import math
import unittest

from obito import engineering as eng
from obito.engineering import (CATEGORIES, COST_CLASSES, MATERIALS, TOOL_NAMES, Material, awg_area_mm2,
                               awg_diameter_mm, awg_for_area, awg_label, battery_c_check, beam_cantilever,
                               compare_materials, find_material, flight_time, fmt_number, format_result,
                               lipo_energy, list_materials, mass_from_volume, material_info, material_table,
                               motor_rpm, ohm, prop_static_thrust, register_tools, required_thrust,
                               thrust_to_weight, torque, unit_convert, voltage_divider, wire_size)
from obito.tools import ToolRegistry, default_registry

REQUIRED_KEYS = ("cfk", "gfk", "alu_6061", "alu_7075", "stahl_s235", "edelstahl_1_4301", "titan_grade5",
                 "pla", "petg", "abs", "asa", "nylon_pa12", "pa_cf", "tpu", "resin_standard", "pom", "pc",
                 "balsa", "birkensperrholz", "kupfer", "messing", "magnesium_az31")


def _has_result_shape(test, res):
    test.assertIsInstance(res, dict)
    test.assertIsInstance(res["formel"], str)
    test.assertTrue(res["formel"])
    test.assertIsInstance(res["annahmen"], list)
    test.assertTrue(res["annahmen"])
    for a in res["annahmen"]:
        test.assertIsInstance(a, str)


# ------------------------------------------------------------------ Materialdaten
class MaterialTableDataTest(unittest.TestCase):
    def test_required_materials_present(self):
        for key in REQUIRED_KEYS:
            self.assertIn(key, MATERIALS, key)
        self.assertGreaterEqual(len(MATERIALS), 22)

    def test_values_plausible(self):
        for key, m in MATERIALS.items():
            self.assertIsInstance(m, Material)
            self.assertEqual(m.key, key)
            self.assertIn(m.category, CATEGORIES)
            self.assertIn(m.cost, COST_CLASSES)
            self.assertIn(m.printable, ("FDM", "SLA", None))
            self.assertTrue(0.05 < m.density < 25, key)
            lo, hi = m.tensile_mpa
            self.assertTrue(0 < lo <= hi < 3000, key)
            self.assertTrue(0 < m.youngs_gpa < 500, key)
            if m.max_temp_c is not None:
                self.assertTrue(-50 < m.max_temp_c < 1500, key)
            self.assertTrue(m.notes and m.typical_use and m.name)

    def test_some_known_values(self):
        self.assertAlmostEqual(MATERIALS["alu_6061"].density, 2.70)
        self.assertAlmostEqual(MATERIALS["stahl_s235"].youngs_gpa, 210.0)
        self.assertGreater(MATERIALS["alu_7075"].tensile_mpa[0], MATERIALS["alu_6061"].tensile_mpa[1])
        self.assertLess(MATERIALS["balsa"].density, 0.3)
        self.assertEqual(MATERIALS["pla"].printable, "FDM")
        self.assertEqual(MATERIALS["resin_standard"].printable, "SLA")
        self.assertIsNone(MATERIALS["cfk"].printable)
        self.assertIsNone(MATERIALS["balsa"].max_temp_c)
        self.assertEqual(MATERIALS["titan_grade5"].cost, "teuer")
        self.assertEqual(MATERIALS["cfk"].category, "verbund")
        self.assertEqual(MATERIALS["birkensperrholz"].category, "holz")

    def test_frozen_and_to_dict(self):
        m = MATERIALS["cfk"]
        with self.assertRaises(Exception):
            m.density = 1.0  # type: ignore[misc]
        d = m.to_dict()
        for k in ("schluessel", "name", "kategorie", "dichte_g_cm3", "zugfestigkeit_mpa", "e_modul_gpa",
                  "max_temp_c", "kosten", "druckbar", "hinweise", "typischer_einsatz", "richtwert"):
            self.assertIn(k, d)
        self.assertEqual(d["schluessel"], "cfk")
        self.assertEqual(d["zugfestigkeit_mpa"], [600.0, 1500.0])
        self.assertTrue(d["richtwert"])
        self.assertIsNone(MATERIALS["balsa"].to_dict()["max_temp_c"])
        self.assertAlmostEqual(m.specific_strength, 600 / 1.55, places=3)
        self.assertAlmostEqual(m.specific_stiffness, 70 / 1.55, places=3)

    def test_list_materials(self):
        all_mats = list_materials()
        self.assertEqual(len(all_mats), len(MATERIALS))
        metals = list_materials("metall")
        self.assertTrue(metals)
        self.assertTrue(all(m.category == "metall" for m in metals))
        self.assertEqual(list_materials("sonstiges"), [])


class FindMaterialTest(unittest.TestCase):
    def test_exact_keys_and_names(self):
        for key, m in MATERIALS.items():
            self.assertIs(find_material(key), m)
            self.assertIs(find_material(m.name), m)
            self.assertIs(find_material(key.upper()), m)

    def test_aliases(self):
        cases = {
            "Carbon": "cfk", "CFK": "cfk", "Kohlefaser": "cfk", "carbon fibre": "cfk", "Karbon": "cfk",
            "GFK": "gfk", "Glasfaser": "gfk", "Fiberglass": "gfk",
            "Alu": "alu_6061", "Aluminium": "alu_6061", "aluminum": "alu_6061", "6061": "alu_6061",
            "Alu 7075": "alu_7075", "Aluminium 7075 T6": "alu_7075", "7075": "alu_7075", "Flugzeugalu": "alu_7075",
            "Stahl": "stahl_s235", "S235": "stahl_s235", "Baustahl": "stahl_s235",
            "Edelstahl": "edelstahl_1_4301", "Edelstahl 1.4301": "edelstahl_1_4301", "1.4301": "edelstahl_1_4301",
            "V2A": "edelstahl_1_4301", "AISI 304": "edelstahl_1_4301", "Niro": "edelstahl_1_4301",
            "Titan": "titan_grade5", "Ti-6Al-4V": "titan_grade5", "Titanium": "titan_grade5",
            "PLA": "pla", "PLA+": "pla", "PETG": "petg", "PET-G": "petg", "ABS": "abs", "ASA": "asa",
            "Nylon": "nylon_pa12", "PA12": "nylon_pa12", "PA 12": "nylon_pa12", "Polyamid": "nylon_pa12",
            "PA-CF": "pa_cf", "Nylon CF": "pa_cf", "PA12-CF": "pa_cf",
            "TPU": "tpu", "TPU 95A": "tpu", "Flex": "tpu",
            "Resin": "resin_standard", "Harz": "resin_standard", "Tough Resin": "resin_standard", "SLA": "resin_standard",
            "POM": "pom", "Delrin": "pom", "PC": "pc", "Polycarbonat": "pc",
            "Holz": "balsa", "Balsa": "balsa", "Balsaholz": "balsa",
            "Birkensperrholz": "birkensperrholz", "Sperrholz": "birkensperrholz", "Plywood": "birkensperrholz",
            "Birkensperrholz 3mm": "birkensperrholz",
            "Kupfer": "kupfer", "copper": "kupfer", "Cu": "kupfer",
            "Messing": "messing", "Brass": "messing", "Messingrohr": "messing",
            "Magnesium": "magnesium_az31", "AZ31": "magnesium_az31",
        }
        for query, key in cases.items():
            m = find_material(query)
            self.assertIsNotNone(m, query)
            self.assertEqual(m.key, key, query)

    def test_substring_prefix_and_typos(self):
        self.assertEqual(find_material("Edelstahlrohr").key, "edelstahl_1_4301")   # länger als "stahl"
        self.assertEqual(find_material("Stahlblech").key, "stahl_s235")
        self.assertEqual(find_material("Aluplatte").key, "alu_6061")             # Präfix vor "pla"
        self.assertEqual(find_material("Kohlefaser-Platte").key, "cfk")
        self.assertEqual(find_material("Sperholz").key, "birkensperrholz")        # Tippfehler
        self.assertEqual(find_material("Aluminum").key, "alu_6061")

    def test_unknown(self):
        for q in ("", "   ", "Beton", "Gold", "xyz", "Quantenschaum"):
            self.assertIsNone(find_material(q), q)
        self.assertIsNone(find_material(None))  # type: ignore[arg-type]


class CompareAndTableTest(unittest.TestCase):
    def test_compare_list_and_string(self):
        mats = compare_materials(["CFK", "Alu 7075", "PETG"])
        self.assertEqual([m.key for m in mats], ["cfk", "alu_7075", "petg"])
        mats = compare_materials("CFK, Alu 7075; PETG und Balsa")
        self.assertEqual([m.key for m in mats], ["cfk", "alu_7075", "petg", "balsa"])
        # Duplikate entfernt
        mats = compare_materials("Carbon, CFK, Kohlefaser")
        self.assertEqual([m.key for m in mats], ["cfk"])

    def test_compare_errors(self):
        with self.assertRaises(ValueError) as cm:
            compare_materials(["CFK", "Beton", "Gold"])
        self.assertIn("Beton", str(cm.exception))
        self.assertIn("Gold", str(cm.exception))
        self.assertIn("Unbekannte Materialien", str(cm.exception))
        with self.assertRaises(ValueError):
            compare_materials([])
        with self.assertRaises(ValueError):
            compare_materials("  ,  ")

    def test_table_aligned(self):
        mats = compare_materials(["CFK", "Alu 6061", "PETG", "Balsa"])
        table = material_table(mats)
        lines = table.splitlines()
        self.assertIn("Material", lines[0])
        self.assertIn("Dichte g/cm³", lines[0])
        self.assertIn("Zugfestigkeit MPa", lines[0])
        self.assertIn("E-Modul GPa", lines[0])
        self.assertTrue(set(lines[1]) <= {"-", "+"})
        body = lines[:2 + len(mats)]
        self.assertEqual(len({len(ln) for ln in body}), 1, body)       # alle Zeilen gleich lang
        # Spaltentrenner an identischen Positionen
        positions = [[i for i, ch in enumerate(ln) if ch == "|"] for ln in body if "|" in ln]
        self.assertTrue(all(p == positions[0] for p in positions))
        self.assertIn("600–1500", table)
        self.assertIn("1,55", table)
        self.assertIn("Balsaholz", table)
        self.assertIn("–", lines[2 + 3])                   # Balsa ohne max. Temperatur
        self.assertIn("Richtwerte", table)
        self.assertEqual(material_table([]), "Keine Materialien.")

    def test_material_info_text(self):
        text = material_info("Carbon")
        for part in ("CFK", "Kategorie: Verbund", "Dichte: 1,55 g/cm³", "Zugfestigkeit: 600–1500 MPa",
                     "E-Modul: 70 GPa", "120 °C", "Kosten: teuer", "3D-Druck: nein", "Hinweise:",
                     "Typischer Einsatz:", "Richtwerte"):
            self.assertIn(part, text)
        self.assertIn("3D-Druck: FDM", material_info("PETG"))
        self.assertIn("nicht definierbar", material_info("Balsa"))
        with self.assertRaises(ValueError) as cm:
            material_info("Beton")
        self.assertIn("Unbekanntes Material", str(cm.exception))


# ------------------------------------------------------------------ Akku
class BatteryTest(unittest.TestCase):
    def test_lipo_energy(self):
        res = lipo_energy(4, 1500)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["spannung_v"], 14.8)
        self.assertAlmostEqual(res["energie_wh"], 22.2)
        self.assertAlmostEqual(res["spannung_voll_v"], 16.8)
        self.assertAlmostEqual(res["kapazitaet_ah"], 1.5)
        self.assertEqual(res["zellen"], 4)
        self.assertIn("3,7 V", res["formel"])
        self.assertAlmostEqual(lipo_energy(6, 5000)["energie_wh"], 111.0)
        self.assertAlmostEqual(lipo_energy("3", "2200")["energie_wh"], 24.42)   # tolerant bei Strings

    def test_lipo_energy_errors(self):
        for cells, mah in ((0, 1500), (-1, 1500), (4, 0), (4, -5), (2.5, 1000), (99, 1000), ("x", 1000), (4, "abc")):
            with self.assertRaises(ValueError, msg=(cells, mah)):
                lipo_energy(cells, mah)

    def test_flight_time(self):
        res = flight_time(1500, 4, 20)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["flugzeit_min"], 3.6)
        self.assertAlmostEqual(res["kapazitaet_nutzbar_mah"], 1200)
        self.assertAlmostEqual(res["leistung_mittel_w"], 296.0)
        self.assertAlmostEqual(flight_time(5000, 6, 10, usable=1.0)["flugzeit_min"], 30.0)
        self.assertAlmostEqual(flight_time(1500, 4, 20, usable=0.5)["flugzeit_min"], 2.25)

    def test_flight_time_errors(self):
        with self.assertRaises(ValueError):
            flight_time(1500, 4, 0)
        with self.assertRaises(ValueError):
            flight_time(1500, 4, 20, usable=0)
        with self.assertRaises(ValueError):
            flight_time(1500, 4, 20, usable=1.5)
        with self.assertRaises(ValueError):
            flight_time(-1, 4, 20)

    def test_battery_c_check(self):
        res = battery_c_check(1500, 45, 20)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["max_strom_a"], 67.5)
        self.assertAlmostEqual(res["benoetigte_c_rate"], 13.33, places=2)
        self.assertTrue(res["ausreichend"])
        self.assertTrue(res["bewertung"].startswith("ok"))
        res = battery_c_check(1000, 20, 25)
        self.assertFalse(res["ausreichend"])
        self.assertAlmostEqual(res["auslastung_prozent"], 125.0)
        self.assertIn("zu hoch", res["bewertung"])
        self.assertIn("grenzwertig", battery_c_check(1000, 20, 18)["bewertung"])
        with self.assertRaises(ValueError):
            battery_c_check(1000, 0, 5)
        with self.assertRaises(ValueError):
            battery_c_check(1000, 20, -1)


# ------------------------------------------------------------------ Schub
class ThrustTest(unittest.TestCase):
    def test_required_thrust(self):
        res = required_thrust(1200, twr=2, motors=4)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["schub_gesamt_g"], 2400.0)
        self.assertAlmostEqual(res["schub_je_motor_g"], 600.0)
        self.assertAlmostEqual(res["schub_gesamt_n"], 2.4 * 9.81, places=3)
        self.assertAlmostEqual(res["schub_je_motor_n"], 0.6 * 9.81, places=3)
        self.assertAlmostEqual(res["schwebe_gas_prozent"], 50.0)
        self.assertAlmostEqual(res["schwebeschub_je_motor_g"], 300.0)
        res = required_thrust(800, twr=3, motors=6)
        self.assertAlmostEqual(res["schub_je_motor_g"], 400.0)
        self.assertAlmostEqual(required_thrust(1000)["schub_gesamt_g"], 2000.0)   # Standard TWR 2, 4 Motoren

    def test_required_thrust_errors(self):
        for args in ((0,), (-5,), (1000, 0), (1000, 2, 0), (1000, 2, 3.5), (1000, 2, 99)):
            with self.assertRaises(ValueError, msg=args):
                required_thrust(*args)

    def test_thrust_to_weight(self):
        res = thrust_to_weight(1200, 800, 4)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["twr"], 2.67, places=2)
        self.assertAlmostEqual(res["schub_gesamt_g"], 3200.0)
        self.assertIn("sportlich", res["bewertung"])
        self.assertIn("hebt nicht ab", thrust_to_weight(2000, 400, 4)["bewertung"])
        self.assertIn("knapp", thrust_to_weight(1000, 300, 4)["bewertung"])
        self.assertIn("Racing", thrust_to_weight(500, 1000, 4)["bewertung"])
        with self.assertRaises(ValueError):
            thrust_to_weight(1000, 0)
        with self.assertRaises(ValueError):
            thrust_to_weight(1000, 500, 0)


# ------------------------------------------------------------------ Elektrik
class ElectricTest(unittest.TestCase):
    def test_ohm_all_pairs(self):
        res = ohm(u=12, r=47)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["strom_a"], 0.2553, places=4)
        self.assertAlmostEqual(res["leistung_w"], 144 / 47, places=3)
        self.assertEqual(res["gegeben"], ["Spannung", "Widerstand"])
        res = ohm(u=12, i=2)
        self.assertAlmostEqual(res["widerstand_ohm"], 6.0)
        self.assertAlmostEqual(res["leistung_w"], 24.0)
        res = ohm(u=12, p=24)
        self.assertAlmostEqual(res["strom_a"], 2.0)
        self.assertAlmostEqual(res["widerstand_ohm"], 6.0)
        res = ohm(i=2, r=6)
        self.assertAlmostEqual(res["spannung_v"], 12.0)
        self.assertAlmostEqual(res["leistung_w"], 24.0)
        res = ohm(i=2, p=24)
        self.assertAlmostEqual(res["spannung_v"], 12.0)
        self.assertAlmostEqual(res["widerstand_ohm"], 6.0)
        res = ohm(r=6, p=24)
        self.assertAlmostEqual(res["spannung_v"], 12.0)
        self.assertAlmostEqual(res["strom_a"], 2.0)

    def test_ohm_errors(self):
        with self.assertRaises(ValueError):
            ohm()
        with self.assertRaises(ValueError):
            ohm(u=12)
        with self.assertRaises(ValueError):
            ohm(u=12, i=1, r=12)
        with self.assertRaises(ValueError):
            ohm(u=12, r=0)
        with self.assertRaises(ValueError):
            ohm(u=12, i=0)
        with self.assertRaises(ValueError):
            ohm(u=12, p=0)
        with self.assertRaises(ValueError):
            ohm(r=-5, p=10)
        with self.assertRaises(ValueError):
            ohm(u="zwölf", i=1)

    def test_voltage_divider(self):
        res = voltage_divider(12, 10000, 4700)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["spannung_aus_v"], 12 * 4700 / 14700, places=4)
        self.assertAlmostEqual(res["strom_ma"], 12 / 14700 * 1000, places=3)
        self.assertAlmostEqual(res["verhaeltnis"], 4700 / 14700, places=5)
        res = voltage_divider(10, 1000, 1000)
        self.assertAlmostEqual(res["spannung_aus_v"], 5.0)
        self.assertAlmostEqual(res["leistung_r1_w"], 0.025)
        self.assertAlmostEqual(res["leistung_gesamt_w"], 0.05)
        self.assertAlmostEqual(voltage_divider(5, 0, 1000)["spannung_aus_v"], 5.0)
        with self.assertRaises(ValueError):
            voltage_divider(5, 0, 0)
        with self.assertRaises(ValueError):
            voltage_divider(5, -10, 100)

    def test_awg_helpers(self):
        self.assertAlmostEqual(awg_diameter_mm(14), 1.628, places=3)
        self.assertAlmostEqual(awg_area_mm2(14), 2.08, places=2)
        self.assertAlmostEqual(awg_diameter_mm(36), 0.127)
        self.assertAlmostEqual(awg_diameter_mm(0), 8.25, places=2)
        self.assertAlmostEqual(awg_area_mm2(-3), 107.2, delta=0.3)
        self.assertEqual(awg_for_area(1.9444), 14)
        self.assertEqual(awg_for_area(2.08), 14)        # exakt passend
        self.assertEqual(awg_for_area(2.2), 13)
        self.assertEqual(awg_for_area(0.5), 20)
        self.assertIsNone(awg_for_area(200))
        self.assertEqual(awg_label(14), "AWG 14")
        self.assertEqual(awg_label(0), "AWG 1/0")
        self.assertEqual(awg_label(-3), "AWG 4/0")
        with self.assertRaises(ValueError):
            awg_diameter_mm(50)

    def test_wire_size(self):
        res = wire_size(20, 1, 12, 3)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["max_spannungsabfall_v"], 0.36)
        self.assertAlmostEqual(res["querschnitt_mm2"], 1.9444, places=4)
        self.assertEqual(res["awg"], 14)
        self.assertEqual(res["awg_bezeichnung"], "AWG 14")
        self.assertAlmostEqual(res["spannungsabfall_v"], 0.0175 * 2 * 20 / awg_area_mm2(14), places=4)
        self.assertLess(res["spannungsabfall_v"], 0.36)
        self.assertAlmostEqual(res["spannungsabfall_prozent"], res["spannungsabfall_v"] / 12 * 100, places=2)
        self.assertAlmostEqual(res["verlustleistung_w"], res["spannungsabfall_v"] * 20, places=3)
        self.assertIn("0,0175", res["formel"])
        # kleinerer Strom -> dünnerer Draht (höhere AWG-Zahl)
        self.assertGreater(wire_size(2, 1, 12)["awg"], 14)
        # höhere Spannung -> dünnerer Draht
        self.assertGreater(wire_size(20, 1, 48)["awg"], 14)

    def test_wire_size_errors(self):
        for args in ((0, 1, 12), (20, 0, 12), (20, 1, 0), (20, 1, 12, 0), (20, 1, 12, 80)):
            with self.assertRaises(ValueError, msg=args):
                wire_size(*args)
        with self.assertRaises(ValueError) as cm:
            wire_size(500, 50, 12, 1)        # weit jenseits 4/0
        self.assertIn("4/0", str(cm.exception))


# ------------------------------------------------------------------ Mechanik
class MechanicsTest(unittest.TestCase):
    def test_torque_from_mass(self):
        res = torque(0.15, mass_kg=2)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["drehmoment_nm"], 2.943, places=3)
        self.assertAlmostEqual(res["drehmoment_mit_reserve_nm"], 2.943 * 1.5, places=3)
        self.assertAlmostEqual(res["kraft_n"], 19.62, places=2)
        self.assertAlmostEqual(res["drehmoment_kgcm"], 30.0, places=2)
        self.assertAlmostEqual(res["drehmoment_ncm"], 294.3, places=1)

    def test_torque_from_force(self):
        res = torque(0.5, force_n=10, safety=2)
        self.assertAlmostEqual(res["drehmoment_nm"], 5.0)
        self.assertAlmostEqual(res["drehmoment_mit_reserve_nm"], 10.0)
        self.assertAlmostEqual(res["masse_kg"], 10 / 9.81, places=4)

    def test_torque_errors(self):
        with self.assertRaises(ValueError):
            torque(0.15)
        with self.assertRaises(ValueError):
            torque(0.15, force_n=1, mass_kg=1)
        with self.assertRaises(ValueError):
            torque(0, mass_kg=1)
        with self.assertRaises(ValueError):
            torque(0.1, mass_kg=-1)
        with self.assertRaises(ValueError):
            torque(0.1, mass_kg=1, safety=0.5)

    def test_motor_rpm(self):
        res = motor_rpm(2300, 14.8)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["drehzahl_leerlauf_rpm"], 34040.0)
        self.assertAlmostEqual(res["drehzahl_rpm"], round(34040 * 0.85))
        self.assertAlmostEqual(motor_rpm(1000, 10, load_factor=1.0)["drehzahl_rpm"], 10000.0)
        for args in ((0, 10), (1000, 0), (1000, 10, 0), (1000, 10, 1.2)):
            with self.assertRaises(ValueError, msg=args):
                motor_rpm(*args)

    def test_prop_static_thrust(self):
        res = prop_static_thrust(5, 4.5, 25000)
        _has_result_shape(self, res)
        expected = 4.392e-8 * 25000 * 5 ** 3.5 / math.sqrt(4.5) * (4.233e-4 * 25000 * 4.5)
        self.assertAlmostEqual(expected, 6.89, places=2)
        self.assertAlmostEqual(res["schub_n"], expected, places=2)
        self.assertAlmostEqual(res["schub_g"], expected / 9.81 * 1000, places=0)
        self.assertAlmostEqual(res["blattspitzengeschwindigkeit_m_s"], math.pi * 5 * 0.0254 * 25000 / 60, places=0)
        self.assertTrue(any("±30" in a for a in res["annahmen"]))
        # Warnung bei Überschall-Blattspitzen
        fast = prop_static_thrust(10, 5, 40000)
        self.assertTrue(any("Warnung" in a for a in fast["annahmen"]))
        self.assertFalse(any("Warnung" in a for a in res["annahmen"]))
        for args in ((0, 4.5, 25000), (5, 0, 25000), (5, 4.5, 0), (100, 4.5, 1000), (5, 20, 1000)):
            with self.assertRaises(ValueError, msg=args):
                prop_static_thrust(*args)

    def test_beam_cantilever(self):
        res = beam_cantilever(10, 0.2, 70, 0.02, 0.005)
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["flaechentraegheitsmoment_mm4"], 208.333, places=2)
        self.assertAlmostEqual(res["biegemoment_nm"], 2.0)
        self.assertAlmostEqual(res["durchbiegung_mm"], 1.8286, places=3)
        self.assertAlmostEqual(res["biegespannung_mpa"], 24.0, places=3)
        self.assertAlmostEqual(res["durchbiegung_relativ_prozent"], 0.914, places=2)
        # doppelte Höhe -> 1/8 der Durchbiegung, 1/4 der Spannung
        res2 = beam_cantilever(10, 0.2, 70, 0.02, 0.010)
        self.assertAlmostEqual(res2["durchbiegung_mm"], res["durchbiegung_mm"] / 8, places=3)
        self.assertAlmostEqual(res2["biegespannung_mpa"], res["biegespannung_mpa"] / 4, places=3)
        # doppelte Länge -> 8-fache Durchbiegung
        res3 = beam_cantilever(10, 0.4, 70, 0.02, 0.005)
        self.assertAlmostEqual(res3["durchbiegung_mm"], res["durchbiegung_mm"] * 8, places=2)
        for args in ((0, 0.2, 70, 0.02, 0.005), (10, 0, 70, 0.02, 0.005), (10, 0.2, 0, 0.02, 0.005),
                     (10, 0.2, 70, -1, 0.005), (10, 0.2, 70, 0.02, 0)):
            with self.assertRaises(ValueError, msg=args):
                beam_cantilever(*args)

    def test_mass_from_volume(self):
        res = mass_from_volume(10, "Alu 6061")
        _has_result_shape(self, res)
        self.assertAlmostEqual(res["masse_g"], 27.0)
        self.assertAlmostEqual(res["masse_kg"], 0.027)
        self.assertEqual(res["material_schluessel"], "alu_6061")
        self.assertAlmostEqual(mass_from_volume(100, "PETG")["masse_g"], 127.0)
        self.assertAlmostEqual(mass_from_volume(1, "Stahl")["masse_g"], 7.85)
        with self.assertRaises(ValueError):
            mass_from_volume(0, "PETG")
        with self.assertRaises(ValueError) as cm:
            mass_from_volume(10, "Beton")
        self.assertIn("Unbekanntes Material", str(cm.exception))


# ------------------------------------------------------------------ Einheiten
class UnitConvertTest(unittest.TestCase):
    def conv(self, value, a, b):
        res = unit_convert(value, a, b)
        _has_result_shape(self, res)
        return res

    def test_length(self):
        self.assertAlmostEqual(self.conv(25, "mm", "in")["wert"], 0.984252, places=6)
        self.assertAlmostEqual(self.conv(1, "in", "mm")["wert"], 25.4)
        self.assertAlmostEqual(self.conv(1, "Zoll", "Millimeter")["wert"], 25.4)
        self.assertAlmostEqual(self.conv(1, "ft", "m")["wert"], 0.3048)
        self.assertAlmostEqual(self.conv(1.5, "km", "m")["wert"], 1500.0)
        self.assertAlmostEqual(self.conv(100, "cm", "m")["wert"], 1.0)
        res = self.conv(1, "m", "mm")
        self.assertEqual((res["wert"], res["einheit"], res["von_einheit"], res["groesse"]), (1000.0, "mm", "m", "Länge"))
        self.assertAlmostEqual(res["faktor"], 1000.0)

    def test_mass_force_pressure(self):
        self.assertAlmostEqual(self.conv(1, "lb", "g")["wert"], 453.59237, places=4)
        self.assertAlmostEqual(self.conv(16, "oz", "lb")["wert"], 1.0, places=6)
        self.assertAlmostEqual(self.conv(2.5, "kg", "g")["wert"], 2500.0)
        self.assertAlmostEqual(self.conv(1, "kgf", "N")["wert"], 9.81)
        self.assertAlmostEqual(self.conv(1, "lbf", "N")["wert"], 4.44822, places=4)
        self.assertAlmostEqual(self.conv(1000, "N", "kN")["wert"], 1.0)
        self.assertAlmostEqual(self.conv(1, "bar", "psi")["wert"], 14.5038, places=3)
        self.assertAlmostEqual(self.conv(100, "psi", "bar")["wert"], 6.89476, places=4)
        self.assertAlmostEqual(self.conv(1, "MPa", "N/mm²")["wert"], 1.0)
        self.assertAlmostEqual(self.conv(1013.25, "hPa", "atm")["wert"], 1.0, places=6)
        self.assertAlmostEqual(self.conv(1, "bar", "kPa")["wert"], 100.0)

    def test_temperature(self):
        self.assertAlmostEqual(self.conv(20, "°C", "F")["wert"], 68.0)
        self.assertAlmostEqual(self.conv(20, "C", "°F")["wert"], 68.0)
        self.assertAlmostEqual(self.conv(212, "Fahrenheit", "Celsius")["wert"], 100.0)
        self.assertAlmostEqual(self.conv(0, "C", "K")["wert"], 273.15)
        self.assertAlmostEqual(self.conv(300, "K", "C")["wert"], 26.85)
        self.assertAlmostEqual(self.conv(-40, "C", "F")["wert"], -40.0)
        self.assertAlmostEqual(self.conv(32, "F", "K")["wert"], 273.15)
        res = self.conv(100, "Grad Celsius", "Kelvin")
        self.assertEqual((res["einheit"], res["von_einheit"], res["groesse"]), ("K", "°C", "Temperatur"))
        self.assertIsNone(res["faktor"])
        self.assertIn("273,15", res["formel"])
        with self.assertRaises(ValueError):
            unit_convert(-300, "C", "K")
        with self.assertRaises(ValueError):
            unit_convert(-1, "K", "C")

    def test_speed_energy_power(self):
        self.assertAlmostEqual(self.conv(100, "km/h", "m/s")["wert"], 27.7778, places=3)
        self.assertAlmostEqual(self.conv(100, "kmh", "ms")["wert"], 27.7778, places=3)
        self.assertAlmostEqual(self.conv(100, "KM/H", "m/s")["wert"], 27.7778, places=3)
        self.assertAlmostEqual(self.conv(1, "mph", "km/h")["wert"], 1.609344, places=5)
        self.assertAlmostEqual(self.conv(10, "Knoten", "km/h")["wert"], 18.52, places=2)
        self.assertAlmostEqual(self.conv(1, "kWh", "J")["wert"], 3.6e6)
        self.assertAlmostEqual(self.conv(1, "Wh", "J")["wert"], 3600.0)
        self.assertAlmostEqual(self.conv(1, "kcal", "kJ")["wert"], 4.184)
        self.assertAlmostEqual(self.conv(1, "PS", "kW")["wert"], 0.73549875)
        self.assertAlmostEqual(self.conv(1, "hp", "W")["wert"], 745.7, places=1)
        self.assertAlmostEqual(self.conv(100, "ps", "hp")["wert"], 98.63, places=2)
        self.assertAlmostEqual(self.conv(2, "MW", "W")["wert"], 2e6)
        self.assertAlmostEqual(self.conv(2, "mW", "W")["wert"], 0.002)
        self.assertAlmostEqual(self.conv(2, "mw", "W")["wert"], 0.002)       # klein = Milliwatt

    def test_ambiguous_kn(self):
        self.assertAlmostEqual(self.conv(1, "kn", "km/h")["wert"], 1.852, places=3)   # Knoten
        self.assertAlmostEqual(self.conv(1, "kn", "N")["wert"], 1000.0)                # Kilonewton
        self.assertAlmostEqual(self.conv(1, "kN", "N")["wert"], 1000.0)

    def test_incompatible_and_unknown(self):
        for args in ((1, "m", "kg"), (1, "mm", "°C"), (1, "N", "psi"), (1, "Wh", "W"), (1, "km/h", "m")):
            with self.assertRaises(ValueError, msg=args) as cm:
                unit_convert(*args)
            self.assertIn("nicht umrechenbar", str(cm.exception))
        with self.assertRaises(ValueError) as cm:
            unit_convert(1, "furlong", "m")
        self.assertIn("Unbekannte Einheit", str(cm.exception))
        with self.assertRaises(ValueError):
            unit_convert(1, "", "m")
        with self.assertRaises(ValueError):
            unit_convert("viel", "m", "mm")
        with self.assertRaises(ValueError):
            unit_convert(float("nan"), "m", "mm")

    def test_mah_hint(self):
        with self.assertRaises(ValueError) as cm:
            unit_convert(1500, "mAh", "Wh")
        self.assertIn("Ladung", str(cm.exception))
        self.assertIn("akku_rechner", str(cm.exception))

    def test_roundtrip(self):
        for a, b in (("mm", "in"), ("kg", "lb"), ("bar", "psi"), ("km/h", "mph"), ("kWh", "J"), ("PS", "hp")):
            v = unit_convert(unit_convert(123.456, a, b)["wert"], b, a)["wert"]
            self.assertAlmostEqual(v, 123.456, places=5, msg=(a, b))


# ------------------------------------------------------------------ Formatierung
class FormatTest(unittest.TestCase):
    def test_fmt_number(self):
        self.assertEqual(fmt_number(14.8), "14,8")
        self.assertEqual(fmt_number(22.2), "22,2")
        self.assertEqual(fmt_number(0.2553), "0,2553")
        self.assertEqual(fmt_number(2400.0), "2400")
        self.assertEqual(fmt_number(34040.0), "34040")
        self.assertEqual(fmt_number(600.0), "600")
        self.assertEqual(fmt_number(1234.56), "1234,6")
        self.assertEqual(fmt_number(0), "0")
        self.assertEqual(fmt_number(0.0), "0")
        self.assertEqual(fmt_number(7), "7")
        self.assertEqual(fmt_number(True), "ja")
        self.assertEqual(fmt_number(1 / 3, 10), "0,3333333333")
        self.assertEqual(fmt_number("text"), "text")

    def test_format_result(self):
        text = format_result("Titel", lipo_energy(4, 1500))
        lines = text.splitlines()
        self.assertEqual(lines[0], "Titel")
        self.assertIn("  Spannung (nominal): 14,8 V", lines)
        self.assertIn("  Energie: 22,2 Wh", lines)
        self.assertIn("Formel: E [Wh] = Zellen × 3,7 V × Kapazität [mAh] / 1000", lines)
        self.assertIn("Annahmen:", lines)
        self.assertTrue(any(ln.startswith("- LiPo-Nennspannung") for ln in lines))
        # Auswahl der Schlüssel, unbekannter Schlüssel generisch, None als Strich
        text = format_result("T", {"neuer_wert": None, "x_y": 1.5, "formel": "f", "annahmen": ["a"]}, ("x_y", "neuer_wert"))
        self.assertEqual(text.splitlines()[1], "  x y: 1,5")
        self.assertEqual(text.splitlines()[2], "  neuer wert: –")
        self.assertEqual(format_result("Nur Titel", {}), "Nur Titel")


# ------------------------------------------------------------------ Werkzeuge
class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.reg = ToolRegistry(".")
        register_tools(self.reg)

    def run_ok(self, name, args):
        res = self.reg.run(name, args)
        self.assertTrue(res.ok, (name, res.error))
        self.assertIn("Formel:", res.output)
        self.assertIn("Annahmen:", res.output)
        return res.output

    def test_all_registered_not_dangerous(self):
        names = [t.name for t in self.reg.list()]
        self.assertEqual(names, list(TOOL_NAMES))
        self.assertEqual(len(names), 12)
        for t in self.reg.list():
            self.assertFalse(t.dangerous, t.name)
            self.assertTrue(t.description, t.name)
            self.assertEqual(t.parameters["type"], "object")
            self.assertIn("required", t.parameters)
            self.assertIn("properties", t.parameters)
            for req in t.parameters["required"]:
                self.assertIn(req, t.parameters["properties"], (t.name, req))
            for pname, spec in t.parameters["properties"].items():
                self.assertIn(spec["type"], ("number", "integer", "string"), (t.name, pname))
                self.assertTrue(spec.get("description"), (t.name, pname))
        self.assertEqual(self.reg.get("akku_rechner").parameters["required"], ["zellen", "mah"])
        self.assertEqual(self.reg.get("elektro_rechner").parameters["required"], [])
        self.assertEqual(self.reg.get("balken_rechner").parameters["required"],
                         ["kraft_n", "laenge_m", "material", "breite_m", "hoehe_m"])
        self.assertEqual(self.reg.get("einheiten_umrechnen").parameters["required"], ["wert", "von", "nach"])

    def test_coexists_with_default_registry(self):
        reg = default_registry(".")
        before = len(reg.list())
        self.assertTrue(all(reg.get(n) is not None for n in TOOL_NAMES))   # schon über default_registry dabei
        register_tools(reg)                                                  # erneut registrieren ist idempotent
        self.assertEqual(len(reg.list()), before)
        self.assertTrue(reg.run("rechnen", {"ausdruck": "1+1"}).ok)
        desc = reg.describe()
        self.assertIn("- akku_rechner(zellen, mah, strom_a?, c_rate?):", desc)
        self.assertIn("- schub_rechner(masse_g, motoren=4, schub_je_motor_g?, twr=2.0):", desc)
        self.assertIn("- kabel_rechner(strom_a, laenge_m, spannung, max_abfall_prozent=3):", desc)
        one = reg.describe_one("einheiten_umrechnen")
        self.assertIn("wert (number, Pflicht)", one)
        self.assertIn('"von": "km/h"', one)

    def test_material_info_tool(self):
        out = self.reg.run("material_info", {"name": "Carbon"})
        self.assertTrue(out.ok)
        self.assertIn("CFK", out.output)
        self.assertIn("Richtwerte", out.output)
        self.assertIn("Dichte: 1,55 g/cm³", out.output)
        res = self.reg.run("material_info", {"name": "Beton"})
        self.assertFalse(res.ok)
        self.assertIn("Unbekanntes Material »Beton«", res.error)
        self.assertEqual(self.reg.run("material_info", {}).error, "Fehlende Parameter: name")

    def test_compare_tool(self):
        out = self.reg.run("materialien_vergleichen", {"namen": "CFK, Alu 7075, PETG"})
        self.assertTrue(out.ok, out.error)
        self.assertIn("Material", out.output)
        self.assertIn("Aluminium 7075-T6", out.output)
        self.assertIn("Leichtestes:", out.output)
        self.assertIn("Richtwerte", out.output)
        # ein Material -> Info-Text
        out = self.reg.run("materialien_vergleichen", {"namen": "PETG"})
        self.assertTrue(out.ok)
        self.assertIn("3D-Druck: FDM", out.output)
        res = self.reg.run("materialien_vergleichen", {"namen": "CFK, Käse"})
        self.assertFalse(res.ok)
        self.assertIn("Käse", res.error)

    def test_akku_tool_with_string_numbers(self):
        out = self.run_ok("akku_rechner", {"zellen": "4", "mah": "1500"})
        self.assertIn("Spannung (nominal): 14,8 V", out)
        self.assertIn("Energie: 22,2 Wh", out)
        self.assertNotIn("Flugzeit", out)
        out = self.run_ok("akku_rechner", {"zellen": 4, "mah": "1500", "strom_a": "20", "c_rate": "45"})
        self.assertIn("Flugzeit: 3,6 min", out)
        self.assertIn("Max. Dauerstrom: 67,5 A", out)
        self.assertIn("Bewertung: ok", out)
        res = self.reg.run("akku_rechner", {"zellen": "4,5", "mah": 1500})
        self.assertFalse(res.ok)
        self.assertIn("zellen", res.error)
        res = self.reg.run("akku_rechner", {"zellen": 0, "mah": 1500})
        self.assertFalse(res.ok)
        self.assertIn("Zellenzahl", res.error)
        self.assertEqual(self.reg.run("akku_rechner", {"zellen": 4}).error, "Fehlende Parameter: mah")

    def test_schub_tool(self):
        out = self.run_ok("schub_rechner", {"masse_g": "1200"})
        self.assertIn("Gesamtschub: 2400 g", out)
        self.assertIn("Schub je Motor: 600 g", out)
        self.assertIn("Gesamtschub: 23,54 N", out)
        self.assertNotIn("Vorhandener Schub", out)
        out = self.run_ok("schub_rechner", {"masse_g": 1200, "motoren": "6", "schub_je_motor_g": "800", "twr": "3"})
        self.assertIn("Schub je Motor: 600 g", out)          # 3600 g / 6
        self.assertIn("Vorhandener Schub", out)
        self.assertIn("Schub-Gewichts-Verhältnis: 4", out)   # 4800 / 1200
        res = self.reg.run("schub_rechner", {"masse_g": -1})
        self.assertFalse(res.ok)
        self.assertIn("Masse", res.error)

    def test_elektro_tool(self):
        out = self.run_ok("elektro_rechner", {"u": "12", "r": "47"})
        self.assertIn("Strom: 0,2553 A", out)
        self.assertIn("Widerstand: 47 Ω", out)
        self.assertIn("Gegeben: Spannung, Widerstand", out)
        res = self.reg.run("elektro_rechner", {})
        self.assertFalse(res.ok)
        self.assertIn("Genau zwei", res.error)
        res = self.reg.run("elektro_rechner", {"u": 12, "i": 1, "r": 12})
        self.assertFalse(res.ok)
        res = self.reg.run("elektro_rechner", {"u": 12, "r": 0})
        self.assertFalse(res.ok)
        self.assertIn("Kurzschluss", res.error)

    def test_spannungsteiler_tool(self):
        out = self.run_ok("spannungsteiler", {"u_in": "12", "r1": "10000", "r2": "4700"})
        self.assertIn("Ausgangsspannung: 3,837 V", out)
        self.assertEqual(self.reg.run("spannungsteiler", {"u_in": 12}).error, "Fehlende Parameter: r1, r2")
        self.assertFalse(self.reg.run("spannungsteiler", {"u_in": 12, "r1": 0, "r2": 0}).ok)

    def test_drehmoment_tool(self):
        out = self.run_ok("drehmoment_rechner", {"hebel_m": "0,15", "masse_kg": "2"})
        self.assertIn("Drehmoment: 2,943 Nm", out)
        self.assertIn("Drehmoment mit Reserve: 4,415 Nm", out)
        self.assertIn("Drehmoment: 30 kg·cm", out)
        out = self.run_ok("drehmoment_rechner", {"hebel_m": 0.5, "kraft_n": 10, "sicherheit": 2})
        self.assertIn("Drehmoment mit Reserve: 10 Nm", out)
        res = self.reg.run("drehmoment_rechner", {"hebel_m": 0.15})
        self.assertFalse(res.ok)
        self.assertIn("Kraft", res.error)
        res = self.reg.run("drehmoment_rechner", {"hebel_m": 0.15, "kraft_n": 1, "masse_kg": 1})
        self.assertFalse(res.ok)

    def test_motor_tool(self):
        out = self.run_ok("motor_rechner", {"kv": "2300", "spannung": "14.8"})
        self.assertIn("Drehzahl Leerlauf: 34040 U/min", out)
        self.assertIn("Drehzahl unter Last: 28934 U/min", out)
        self.assertNotIn("Standschub", out)
        out = self.run_ok("motor_rechner", {"kv": 2300, "spannung": 14.8, "prop_zoll": "5", "steigung_zoll": "4.5"})
        self.assertIn("Propeller-Standschub", out)
        self.assertIn("Standschub:", out)
        self.assertIn("±30", out)
        res = self.reg.run("motor_rechner", {"kv": 2300, "spannung": 14.8, "prop_zoll": 5})
        self.assertFalse(res.ok)
        self.assertIn("steigung_zoll", res.error)
        self.assertFalse(self.reg.run("motor_rechner", {"kv": 0, "spannung": 14.8}).ok)

    def test_balken_tool(self):
        out = self.run_ok("balken_rechner", {"kraft_n": "10", "laenge_m": "0.2", "material": "Alu 6061",
                                             "breite_m": "0.02", "hoehe_m": "0.005"})
        self.assertIn("Aluminium 6061-T6", out)
        self.assertIn("E-Modul: 69 GPa", out)
        self.assertIn("Biegespannung: 24 MPa", out)
        self.assertIn("Sicherheit gegen Bruch:", out)
        self.assertIn("Durchbiegung:", out)
        # E-Modul direkt als Zahl
        out = self.run_ok("balken_rechner", {"kraft_n": 10, "laenge_m": 0.2, "material": "70",
                                             "breite_m": 0.02, "hoehe_m": 0.005})
        self.assertIn("Durchbiegung: 1,829 mm", out)
        self.assertNotIn("Sicherheit gegen Bruch", out)
        # Warnung bei geringer Sicherheit
        out = self.run_ok("balken_rechner", {"kraft_n": 200, "laenge_m": 0.3, "material": "PLA",
                                             "breite_m": 0.01, "hoehe_m": 0.005})
        self.assertIn("Warnung", out)
        res = self.reg.run("balken_rechner", {"kraft_n": 10, "laenge_m": 0.2, "material": "Beton",
                                              "breite_m": 0.02, "hoehe_m": 0.005})
        self.assertFalse(res.ok)
        self.assertIn("Unbekanntes Material", res.error)
        self.assertTrue(self.reg.run("balken_rechner", {"kraft_n": 10}).error.startswith("Fehlende Parameter:"))

    def test_kabel_tool(self):
        out = self.run_ok("kabel_rechner", {"strom_a": "20", "laenge_m": "1", "spannung": "12"})
        self.assertIn("Rechnerischer Querschnitt: 1,944 mm²", out)
        self.assertIn("AWG: 14", out)
        self.assertIn("Spannungsabfall: 0,3364 V", out)
        out = self.run_ok("kabel_rechner", {"strom_a": 20, "laenge_m": 1, "spannung": 12, "max_abfall_prozent": "1"})
        self.assertIn("AWG: 9", out)       # 5,83 mm² -> AWG 9 (6,63 mm²)
        res = self.reg.run("kabel_rechner", {"strom_a": 20, "laenge_m": 1, "spannung": 12, "max_abfall_prozent": 0})
        self.assertFalse(res.ok)

    def test_einheiten_tool(self):
        out = self.run_ok("einheiten_umrechnen", {"wert": "100", "von": "km/h", "nach": "m/s"})
        self.assertTrue(out.startswith("100 km/h = 27,77777778 m/s"), out)
        self.assertIn("Größe: Geschwindigkeit", out)
        out = self.run_ok("einheiten_umrechnen", {"wert": 20, "von": "°C", "nach": "F"})
        self.assertTrue(out.startswith("20 °C = 68 °F"), out)
        out = self.run_ok("einheiten_umrechnen", {"wert": "2,5", "von": "PS", "nach": "kW"})
        self.assertIn("kW", out)
        res = self.reg.run("einheiten_umrechnen", {"wert": 1, "von": "m", "nach": "kg"})
        self.assertFalse(res.ok)
        self.assertIn("nicht umrechenbar", res.error)
        res = self.reg.run("einheiten_umrechnen", {"wert": 1, "von": "mAh", "nach": "Wh"})
        self.assertFalse(res.ok)
        self.assertIn("akku_rechner", res.error)
        self.assertEqual(self.reg.run("einheiten_umrechnen", {"wert": 1}).error, "Fehlende Parameter: von, nach")

    def test_masse_tool(self):
        out = self.run_ok("masse_aus_volumen", {"volumen_cm3": "12,5", "material": "PETG"})
        self.assertIn("Masse: 15,88 g", out)
        self.assertIn("PETG", out)
        res = self.reg.run("masse_aus_volumen", {"volumen_cm3": 10, "material": "Beton"})
        self.assertFalse(res.ok)
        res = self.reg.run("masse_aus_volumen", {"volumen_cm3": "abc", "material": "PETG"})
        self.assertFalse(res.ok)
        self.assertIn("volumen_cm3", res.error)

    def test_unknown_keys_dropped_and_output_bounded(self):
        out = self.reg.run("akku_rechner", {"zellen": 4, "mah": 1500, "quatsch": 1})
        self.assertTrue(out.ok)
        for name in TOOL_NAMES:
            tool = self.reg.get(name)
            self.assertIsNotNone(tool)
        # Ausgaben bleiben unter dem Werkzeug-Limit (keine Kürzung)
        out = self.reg.run("materialien_vergleichen", {"namen": ", ".join(MATERIALS)})
        self.assertTrue(out.ok, out.error)
        self.assertNotIn("[gekürzt", out.output)
        for key in MATERIALS:
            self.assertIn(MATERIALS[key].name, out.output)

    def test_engineering_tools_factory(self):
        tools = eng.engineering_tools()
        self.assertEqual([t.name for t in tools], list(TOOL_NAMES))
        # register_tools ist idempotent (gleiche Namen überschreiben)
        register_tools(self.reg)
        self.assertEqual(len(self.reg.list()), 12)


if __name__ == "__main__":
    unittest.main()
