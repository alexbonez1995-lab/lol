"""Tests für obito.geometry – Primitive, Netzoperationen, Dateien, ModelStore, Werkzeuge."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from obito import geometry as g
from obito.tools import ToolRegistry


class PrimitiveTest(unittest.TestCase):
    def test_box_volume_area_watertight(self):
        m = g.box(100, 50, 10)
        self.assertAlmostEqual(m.volume(), 50_000, places=6)
        self.assertAlmostEqual(m.surface_area(), 2 * (100 * 50 + 100 * 10 + 50 * 10), places=6)
        self.assertTrue(m.is_watertight())
        self.assertEqual(len(m.faces), 12)

    def test_cylinder(self):
        m = g.cylinder(20, 10, segments=96)
        self.assertAlmostEqual(m.volume(), math.pi * 100 * 10, delta=0.3 * math.pi * 100 * 10 / 100)
        self.assertTrue(m.is_watertight())

    def test_tube_volume_is_outer_minus_inner(self):
        m = g.tube(20, 10, 10, segments=96)
        expect = math.pi * (100 - 25) * 10
        self.assertAlmostEqual(m.volume(), expect, delta=expect * 0.005)
        self.assertTrue(m.is_watertight())
        with self.assertRaises(ValueError):
            g.tube(10, 10, 5)

    def test_cone_and_frustum(self):
        full = g.cone(20, 0, 30, segments=96)
        self.assertAlmostEqual(full.volume(), math.pi * 100 * 30 / 3, delta=20)
        self.assertTrue(full.is_watertight())
        frustum = g.cone(20, 10, 30, segments=96)
        self.assertTrue(frustum.is_watertight())
        self.assertGreater(frustum.volume(), full.volume())

    def test_sphere(self):
        m = g.sphere(20, segments=32)
        expect = math.pi * 20 ** 3 / 6
        self.assertAlmostEqual(m.volume(), expect, delta=expect * 0.02)
        self.assertTrue(m.is_watertight())

    def test_plate_with_holes_volume(self):
        m = g.plate_with_holes(100, 50, 3, [(-30, 0, 10), (30, 0, 10)], segments=48)
        expect = 100 * 50 * 3 - 2 * math.pi * 25 * 3
        self.assertAlmostEqual(m.volume(), expect, delta=expect * 0.005)
        self.assertTrue(m.is_watertight())

    def test_plate_without_holes(self):
        m = g.plate_with_holes(100, 50, 3, [])
        self.assertAlmostEqual(m.volume(), 15_000, places=3)
        self.assertTrue(m.is_watertight())

    def test_plate_hole_errors(self):
        with self.assertRaises(ValueError):
            g.plate_with_holes(100, 50, 3, [(48, 0, 10)])          # schneidet Rand
        with self.assertRaises(ValueError):
            g.plate_with_holes(100, 50, 3, [(0, 0, 10), (5, 0, 10)])  # überlappen
        with self.assertRaises(ValueError):
            g.plate_with_holes(100, 50, 3, [(0, 0)])               # falsches Format

    def test_drone_frame_parts_watertight(self):
        parts = g.drone_frame_parts(250, 12, 4, 60)
        self.assertGreaterEqual(len(parts), 1 + 4 * 2)
        for part in parts:
            self.assertTrue(part.is_watertight(), part.name)
            self.assertGreater(part.volume(), 0)
        frame = g.drone_frame(250, 12, 4, 60)
        self.assertTrue(frame.is_watertight())
        self.assertAlmostEqual(frame.volume(), sum(p.volume() for p in parts), places=3)
        size = frame.bounds()["groesse"]
        self.assertGreater(size[0], 180)
        with self.assertRaises(ValueError):
            g.drone_frame(250, 12, 4, 60, arms=2)

    def test_invalid_values(self):
        for bad in ((0, 1, 1), (-1, 1, 1), (float("nan"), 1, 1)):
            with self.assertRaises(ValueError):
                g.box(*bad)
        with self.assertRaises(ValueError):
            g.build("quader", {"l": 20_000, "b": 1, "h": 1})
        with self.assertRaises(ValueError):
            g.cylinder(10, 10, segments=2)
        with self.assertRaises(ValueError):
            g.cylinder(10, 10, segments=1000)


class MeshOpsTest(unittest.TestCase):
    def test_translate_and_center_of_mass(self):
        m = g.box(10, 10, 10).translate(100, 0, 0)
        cx, cy, cz = m.center_of_mass()
        self.assertAlmostEqual(cx, 100, places=6)
        self.assertAlmostEqual(cz, 5, places=6)
        b = m.bounds()
        self.assertAlmostEqual(b["min"][0], 95, places=6)

    def test_rotate_90_exact(self):
        m = g.box(100, 10, 10).rotate("z", 90)
        size = m.bounds()["groesse"]
        self.assertAlmostEqual(size[0], 10, places=6)
        self.assertAlmostEqual(size[1], 100, places=6)
        self.assertTrue(m.is_watertight())
        with self.assertRaises(ValueError):
            g.box(1, 1, 1).rotate("w", 10)

    def test_scale_and_mirror(self):
        m = g.box(10, 10, 10).scale(2)
        self.assertAlmostEqual(m.volume(), 8000, places=6)
        mm = g.cone(20, 0, 10).mirror("z")
        self.assertGreater(mm.volume(), 0)
        self.assertTrue(mm.is_watertight())
        self.assertLess(mm.bounds()["min"][2], 0)

    def test_merge_assembly(self):
        a = g.box(10, 10, 10)
        b = g.box(10, 10, 10).translate(50, 0, 0)
        m = a.merge(b)
        self.assertEqual(len(m.faces), 24)
        self.assertAlmostEqual(m.volume(), 2000, places=6)
        self.assertTrue(m.is_watertight())

    def test_inertia_box(self):
        m = g.box(100, 50, 10)
        inertia = m.inertia(2.7)
        self.assertGreater(inertia["ixx"], 0)
        # Quader: I_zz = m(l²+b²)/12 mit m = 135 g → 135·(10000+2500)/12
        self.assertAlmostEqual(inertia["izz"], 135 * 12500 / 12, delta=135 * 12500 / 12 * 0.05)

    def test_stats_with_material(self):
        st = g.box(100, 50, 10).stats("Alu 6061")
        self.assertAlmostEqual(st["volumen_cm3"], 50, places=3)
        self.assertAlmostEqual(st["masse_g"], 50 * 2.7, delta=1)
        self.assertTrue(st["wasserdicht"])
        self.assertIsNone(g.box(1, 1, 1).stats("Unobtainium")["masse_g"])

    def test_to_json_and_validate(self):
        m = g.box(1, 2, 3)
        js = m.to_json()
        self.assertEqual(len(js["punkte"]), 8)
        self.assertEqual(len(js["dreiecke"]), 12)
        v = m.validate()
        self.assertEqual(v["ungueltige_indizes"], 0)
        bad = g.Mesh(list(m.vertices), list(m.faces) + [(0, 0, 1), (0, 1, 99)])
        v = bad.validate()
        self.assertEqual(v["ungueltige_indizes"], 1)
        self.assertEqual(v["degenerierte_dreiecke"], 1)
        self.assertFalse(bad.is_watertight())


class FileTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def test_stl_binary_roundtrip(self):
        m = g.tube(20, 10, 10)
        path = self.dir / "a.stl"
        g.write_stl(m, path)
        r = g.read_stl(path)
        self.assertEqual(len(r.faces), len(m.faces))
        self.assertAlmostEqual(r.volume(), m.volume(), places=3)
        self.assertTrue(r.is_watertight())

    def test_stl_ascii_roundtrip(self):
        m = g.box(10, 20, 30)
        path = self.dir / "b.stl"
        g.write_stl(m, path, binary=False)
        text = path.read_text(encoding="ascii")
        self.assertTrue(text.startswith("solid"))
        r = g.read_stl(path)
        self.assertAlmostEqual(r.volume(), 6000, places=3)
        self.assertEqual(len(r.vertices), 8)

    def test_obj_roundtrip_with_quads(self):
        path = self.dir / "c.obj"
        path.write_text("o w\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nv 0 0 1\nv 1 0 1\nv 1 1 1\nv 0 1 1\n"
                        "f 1/1 4/4 3/3 2/2\nf 5 6 7 8\nf 1 2 6 5\nf 2 3 7 6\nf 3 4 8 7\nf 4 1 5 8\n", encoding="utf-8")
        m = g.read_obj(path)
        self.assertEqual(len(m.faces), 12)
        self.assertAlmostEqual(m.volume(), 1.0, places=6)
        out = self.dir / "d.obj"
        g.write_obj(m, out)
        r = g.read_obj(out)
        self.assertAlmostEqual(r.volume(), 1.0, places=6)

    def test_load_by_extension_and_limits(self):
        g.write_stl(g.box(1, 1, 1), self.dir / "x.stl")
        self.assertEqual(len(g.load(self.dir / "x.stl").faces), 12)
        with self.assertRaises(ValueError):
            g.load(self.dir / "x.txt")
        big = self.dir / "big.stl"
        with open(big, "wb") as f:
            f.seek(g.MAX_FILE_BYTES + 1)
            f.write(b"\0")
        with self.assertRaises(ValueError):
            g.load(big)
        with self.assertRaises((FileNotFoundError, ValueError)):
            g.load(self.dir / "fehlt.stl")

    def test_read_stl_garbage(self):
        p = self.dir / "g.stl"
        p.write_bytes(b"solid x\n  facet normal 0 0 0\n kaputt\n")
        with self.assertRaises(ValueError):
            g.read_stl(p)


class BuildTest(unittest.TestCase):
    def test_build_tolerant(self):
        m = g.build("Quader", {"l": "100", "b": "50,5", "h": 10})
        self.assertAlmostEqual(m.volume(), 100 * 50.5 * 10, places=3)
        m = g.build("lochplatte", "l=100, b=50, t=3, holes=-30/0/10; 30/0/10")
        self.assertTrue(m.is_watertight())

    def test_build_errors(self):
        with self.assertRaises(ValueError):
            g.build("torus", {})
        with self.assertRaises(ValueError):
            g.build("quader", {"l": 1, "b": 1})
        with self.assertRaises(ValueError):
            g.build("kugel", {"d": 10, "segments": 300})

    def test_primitives_table(self):
        for key in ("quader", "zylinder", "rohr", "kegel", "kugel", "lochplatte", "drohnenrahmen"):
            self.assertIn(key, g.PRIMITIVES)
            self.assertTrue(g.PRIMITIVES[key]["beschreibung"])
            self.assertTrue(g.PRIMITIVES[key]["parameter"])

    def test_parse_params(self):
        self.assertEqual(g.parse_params("l=1, b=2,5, h=3"), {"l": "1", "b": "2,5", "h": "3"})
        self.assertEqual(g.parse_params({"a": 1}), {"a": 1})
        self.assertEqual(g.parse_params('{"a": 2}'), {"a": 2})
        self.assertEqual(g.parse_params(""), {})
        with self.assertRaises(ValueError):
            g.parse_params("ohnegleich")


class ModelStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.store = g.ModelStore(self.dir / "m.db", self.dir / "files")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_save_versions_list_get_mesh(self):
        rec = self.store.save(g.box(10, 10, 10), "Würfel", kind="quader", params={"l": 10}, material="pla")
        self.assertEqual(rec["version"], 1)
        self.assertTrue(Path(rec["datei"]).is_file())
        self.assertAlmostEqual(rec["statistik"]["volumen_cm3"], 1.0, places=3)
        rec2 = self.store.save(g.box(20, 10, 10), "Würfel", kind="quader", params={"l": 20})
        self.assertEqual(rec2["version"], 2)
        self.assertEqual(len(self.store.list()), 1)            # neueste Version je Name
        self.assertEqual(len(self.store.versions("Würfel")), 2)
        self.assertEqual(self.store.get(rec2["id"])["version"], 2)
        self.assertAlmostEqual(self.store.mesh(rec2["id"]).volume(), 2000, places=3)
        self.assertIsNone(self.store.get(999))

    def test_projects_separate(self):
        self.store.save(g.box(1, 1, 1), "Teil", project="A")
        self.store.save(g.box(1, 1, 1), "Teil", project="B")
        self.assertEqual(len(self.store.list(project="A")), 1)
        self.assertEqual(len(self.store.list()), 2)
        self.assertEqual(self.store.stats()["projekte"], 2)

    def test_delete_removes_file(self):
        rec = self.store.save(g.box(1, 1, 1), "x")
        path = Path(rec["datei"])
        self.assertTrue(self.store.delete(rec["id"]))
        self.assertFalse(path.exists())
        self.assertFalse(self.store.delete(rec["id"]))

    def test_import_file(self):
        src = self.dir / "ext.stl"
        g.write_stl(g.cylinder(10, 10), src)
        rec = self.store.import_file(src, project="P")
        self.assertEqual(rec["name"], "ext")
        self.assertEqual(rec["projekt"], "P")
        self.assertGreater(rec["statistik"]["dreiecke"], 0)

    def test_close_idempotent(self):
        self.store.close()
        self.store.close()
        again = g.ModelStore(self.dir / "m.db", self.dir / "files")
        again.close()


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = ToolRegistry(workspace=self.tmp.name)
        self.store = g.ModelStore(Path(self.tmp.name) / "m.db", Path(self.tmp.name) / "f")
        g.register_tools(self.reg, self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_tools_registered(self):
        names = {t.name for t in self.reg.list()}
        for n in g.TOOL_NAMES:
            self.assertIn(n, names)
        self.assertTrue(all(not t.dangerous for t in self.reg.list()))

    def test_create_info_list(self):
        res = self.reg.run("modell_erzeugen", {"art": "rohr", "parameter": "d_outer=20, d_inner=10, h=30",
                                               "name": "Rohr", "material": "cfk"})
        self.assertTrue(res.ok, res.error)
        self.assertIn("Rohr", res.output)
        self.assertIn("Volumen", res.output)
        self.assertIn("Masse", res.output)
        info = self.reg.run("modell_info", {"name_oder_id": "Rohr"})
        self.assertTrue(info.ok, info.error)
        self.assertIn("Wasserdicht: ja", info.output)
        lst = self.reg.run("modell_liste", {})
        self.assertTrue(lst.ok)
        self.assertIn("Rohr", lst.output)

    def test_create_error_is_reported(self):
        res = self.reg.run("modell_erzeugen", {"art": "kugel", "parameter": "d=-1", "name": "x"})
        self.assertFalse(res.ok)
        self.assertTrue(res.error)
        res = self.reg.run("modell_info", {"name_oder_id": "gibtsnicht"})
        self.assertIn("Kein Modell", res.output)


if __name__ == "__main__":
    unittest.main()
