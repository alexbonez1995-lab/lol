"""3D-Modellierung von OBITO: parametrische Bauteile, Baugruppen, STL/OBJ, Masse und Schwerpunkt.

Dreiecksnetze in reinem Python (Listen von Punkten und Dreiecken), ausreichend für Bauteile mit
bis zu :data:`MAX_FACES` Dreiecken. Alle Maße in Millimetern.

* :class:`Mesh` – Punkte, Dreiecke, Transformationen (verschieben, drehen, skalieren, spiegeln),
  Kennwerte (Volumen, Oberfläche, Schwerpunkt, Trägheitsmomente, Wasserdichtigkeit) und
  JSON-Export für die HUD.
* Primitive :func:`box`, :func:`cylinder`, :func:`tube`, :func:`cone`, :func:`sphere`,
  :func:`plate_with_holes` (Ohrenschneiden mit Lochbrücken) und die Baugruppe :func:`drone_frame`.
  Alle Primitive sind wasserdicht mit nach außen zeigenden Normalen; Ursprung ist die Mitte der
  Grundfläche (z = 0 … h), bei der Kugel der Mittelpunkt.
* :data:`PRIMITIVES` + :func:`build` – Tabelle mit deutschen Namen und toleranter Parameter-Übernahme
  (Dezimalkomma, Strings, Alias-Namen).
* STL (ASCII und binär) und OBJ lesen/schreiben, :func:`load` nach Endung mit Größenlimit.
* :class:`ModelStore` – SQLite-Ablage mit Versionen und STL-Dateien je Modell.
* :func:`register_tools` – ``modell_erzeugen``, ``modell_info``, ``modell_liste``.

Volumen und Schwerpunkt werden über die signierte Tetraeder-Zerlegung (Ursprung + Dreieck)
berechnet; sie sind für das Dreiecksnetz exakt, gekrümmte Flächen sind durch Facetten angenähert.
"""

from __future__ import annotations

import json
import math
import os
import re
import sqlite3
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Iterable, Sequence

from .engineering import find_material, fmt_number

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfer
    from .tools import ToolRegistry

Vec = tuple[float, float, float]
Face = tuple[int, int, int]

MAX_FACES = 200_000            # Dreieckslimit je Modell (build, load)
MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_DIM_MM = 10_000.0          # größtes zulässiges Maß in build()
MIN_SEGMENTS, MAX_SEGMENTS = 3, 256
MIN_ARMS, MAX_ARMS = 3, 8
MAX_POLY_POINTS = 2_000        # Punkte des Deckel-Polygons einer Lochplatte (Ohrenschneiden ist O(n²))
_DEDUP_DIGITS = 6              # Rundung beim Zusammenführen gleicher Punkte (1e-6 mm)
_EPS = 1e-9

_KEY_SEP = re.compile(r"[,\n]\s*(?=[A-Za-zÄÖÜäöü_][\wÄÖÜäöü]*\s*=)")
_HOLE_SEP = re.compile(r"[;\n]")
_HOLE_PART = re.compile(r"[/|:]|\s+")


# ------------------------------------------------------------------ Vektor-Hilfen
def _sub(a: Vec, b: Vec) -> Vec:
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _cross(a: Vec, b: Vec) -> Vec:
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _dot(a: Vec, b: Vec) -> float:
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _norm(a: Vec) -> float:
    return math.sqrt(_dot(a, a))


def _cos_sin(degrees: float) -> tuple[float, float]:
    """Kosinus/Sinus – für Vielfache von 90° exakt (keine 6e-17-Reste)."""
    r = degrees % 360.0
    exact = {0.0: (1.0, 0.0), 90.0: (0.0, 1.0), 180.0: (-1.0, 0.0), 270.0: (0.0, -1.0)}
    if r in exact:
        return exact[r]
    rad = math.radians(degrees)
    return math.cos(rad), math.sin(rad)


def _vec(p: Sequence[float]) -> Vec:
    return (float(p[0]), float(p[1]), float(p[2]))


def _face(f: Sequence[int]) -> Face:
    return (int(f[0]), int(f[1]), int(f[2]))


def _round3(v: Sequence[float]) -> list[float]:
    return [round(float(x), 3) for x in v]


# ------------------------------------------------------------------ Mesh
@dataclass
class Mesh:
    """Dreiecksnetz: ``vertices`` (x, y, z in mm) und ``faces`` (Indizes, gegen den Uhrzeigersinn von
    außen gesehen → Normale nach außen). Transformationen liefern neue Netze."""

    vertices: list[Vec] = field(default_factory=list)
    faces: list[Face] = field(default_factory=list)
    name: str = ""

    # ------------------------------------------------------------ Transformationen
    def _map(self, fn: Callable[[Vec], Vec], flip: bool = False) -> "Mesh":
        verts = [fn(v) for v in self.vertices]
        faces = [(a, c, b) for a, b, c in self.faces] if flip else list(self.faces)
        return Mesh(verts, faces, self.name)

    def translate(self, dx: float, dy: float = 0.0, dz: float = 0.0) -> "Mesh":
        dx, dy, dz = float(dx), float(dy), float(dz)
        return self._map(lambda v: (v[0] + dx, v[1] + dy, v[2] + dz))

    def rotate(self, axis: str, degrees: float) -> "Mesh":
        """Dreht um die Achse ``"x"``, ``"y"`` oder ``"z"`` durch den Ursprung (Rechte-Hand-Regel)."""
        ax = str(axis).strip().lower()
        c, s = _cos_sin(float(degrees))
        if ax == "x":
            fn = lambda v: (v[0], v[1] * c - v[2] * s, v[1] * s + v[2] * c)  # noqa: E731
        elif ax == "y":
            fn = lambda v: (v[0] * c + v[2] * s, v[1], -v[0] * s + v[2] * c)  # noqa: E731
        elif ax == "z":
            fn = lambda v: (v[0] * c - v[1] * s, v[0] * s + v[1] * c, v[2])  # noqa: E731
        else:
            raise ValueError(f"Unbekannte Achse {axis!r} (erlaubt: x, y, z).")
        return self._map(fn)

    def scale(self, sx: float, sy: float | None = None, sz: float | None = None) -> "Mesh":
        """Skaliert um den Ursprung; ``sy``/``sz`` fehlen → gleichmäßig. Negative Faktoren spiegeln
        (die Dreiecksorientierung wird dann passend umgedreht)."""
        sx = float(sx)
        sy = sx if sy is None else float(sy)
        sz = sx if sz is None else float(sz)
        if sx == 0 or sy == 0 or sz == 0:
            raise ValueError("Skalierungsfaktor 0 ist nicht erlaubt.")
        flip = (sx * sy * sz) < 0
        return self._map(lambda v: (v[0] * sx, v[1] * sy, v[2] * sz), flip)

    def mirror(self, axis: str) -> "Mesh":
        """Spiegelt an der Ebene senkrecht zur Achse durch den Ursprung und dreht die
        Dreiecksorientierung um, damit die Normalen weiter nach außen zeigen."""
        ax = str(axis).strip().lower()
        if ax == "x":
            fn = lambda v: (-v[0], v[1], v[2])  # noqa: E731
        elif ax == "y":
            fn = lambda v: (v[0], -v[1], v[2])  # noqa: E731
        elif ax == "z":
            fn = lambda v: (v[0], v[1], -v[2])  # noqa: E731
        else:
            raise ValueError(f"Unbekannte Achse {axis!r} (erlaubt: x, y, z).")
        return self._map(fn, flip=True)

    def merge(self, other: "Mesh") -> "Mesh":
        """Baugruppe: hängt ``other`` an (Konkatenation, keine Boolesche Operation)."""
        off = len(self.vertices)
        faces = list(self.faces) + [(a + off, b + off, c + off) for a, b, c in other.faces]
        name = self.name or other.name
        return Mesh(list(self.vertices) + list(other.vertices), faces, name)

    def copy(self) -> "Mesh":
        return Mesh(list(self.vertices), list(self.faces), self.name)

    # ------------------------------------------------------------ Kennwerte
    def bounds(self) -> dict:
        """``{"min": (x,y,z), "max": (x,y,z), "groesse": (dx,dy,dz)}`` in mm (leeres Netz → Nullen)."""
        if not self.vertices:
            z = (0.0, 0.0, 0.0)
            return {"min": z, "max": z, "groesse": z}
        xs = [v[0] for v in self.vertices]
        ys = [v[1] for v in self.vertices]
        zs = [v[2] for v in self.vertices]
        lo = (min(xs), min(ys), min(zs))
        hi = (max(xs), max(ys), max(zs))
        return {"min": lo, "max": hi, "groesse": _sub(hi, lo)}

    def signed_volume(self) -> float:
        """Signierte Tetraeder-Summe (Ursprung + Dreieck) in mm³; positiv bei Normalen nach außen."""
        vs = self.vertices
        total = 0.0
        for a, b, c in self.faces:
            total += _dot(vs[a], _cross(vs[b], vs[c]))
        return total / 6.0

    def volume(self) -> float:
        """Volumen in mm³ (Betrag der signierten Tetraeder-Summe)."""
        return abs(self.signed_volume())

    def surface_area(self) -> float:
        """Oberfläche in mm²."""
        vs = self.vertices
        total = 0.0
        for a, b, c in self.faces:
            total += _norm(_cross(_sub(vs[b], vs[a]), _sub(vs[c], vs[a])))
        return total / 2.0

    def center_of_mass(self) -> Vec:
        """Volumenschwerpunkt über Tetraeder-Schwerpunkte (Ursprung + Dreieck, Gewicht = signiertes
        Volumen). Bei Volumen ≈ 0 (offene Flächen) flächengewichteter Schwerpunkt der Dreiecke."""
        vs = self.vertices
        if not self.faces:
            return (0.0, 0.0, 0.0)
        sx = sy = sz = 0.0
        vol = 0.0
        for a, b, c in self.faces:
            pa, pb, pc = vs[a], vs[b], vs[c]
            v6 = _dot(pa, _cross(pb, pc))          # 6 × signiertes Tetraedervolumen
            vol += v6
            sx += v6 * (pa[0] + pb[0] + pc[0])
            sy += v6 * (pa[1] + pb[1] + pc[1])
            sz += v6 * (pa[2] + pb[2] + pc[2])
        scale = max(1.0, _norm(self.bounds()["groesse"])) ** 3
        if abs(vol) > 1e-9 * scale:
            # Tetraeder-Schwerpunkt = (0 + a + b + c) / 4
            return (sx / (4.0 * vol), sy / (4.0 * vol), sz / (4.0 * vol))
        ax = ay = az = 0.0
        area = 0.0
        for a, b, c in self.faces:
            pa, pb, pc = vs[a], vs[b], vs[c]
            w = _norm(_cross(_sub(pb, pa), _sub(pc, pa)))
            area += w
            ax += w * (pa[0] + pb[0] + pc[0]) / 3.0
            ay += w * (pa[1] + pb[1] + pc[1]) / 3.0
            az += w * (pa[2] + pb[2] + pc[2]) / 3.0
        if area <= 0:
            n = len(vs)
            return (sum(v[0] for v in vs) / n, sum(v[1] for v in vs) / n, sum(v[2] for v in vs) / n)
        return (ax / area, ay / area, az / area)

    def inertia(self, density_g_cm3: float) -> dict:
        """Massenträgheitsmomente ``{"ixx", "iyy", "izz"}`` in g·mm² um den Schwerpunkt (Achsen
        parallel zu x/y/z).

        Näherung: der Körper wird in signierte Tetraeder (Ursprung + Dreieck) zerlegt; für jeden
        Tetraeder gilt die geschlossene Formel von Tonon (2004), summiert um den Ursprung, danach
        Satz von Steiner zum Schwerpunkt. Für das Dreiecksnetz ist das exakt – gekrümmte Flächen
        (Zylinder, Kugel) sind nur durch Facetten angenähert, Baugruppen zählen Überlappungen doppelt.
        Deviationsmomente werden nicht ausgegeben."""
        rho = float(density_g_cm3) / 1000.0        # g/cm³ → g/mm³
        vs = self.vertices
        sxx = syy = szz = 0.0                       # Σ x², Σ y², Σ z² gewichtet (ohne ρ/60)
        vol6 = 0.0
        for a, b, c in self.faces:
            (x1, y1, z1), (x2, y2, z2), (x3, y3, z3) = vs[a], vs[b], vs[c]
            det = _dot(vs[a], _cross(vs[b], vs[c]))
            vol6 += det
            sxx += det * (x1 * x1 + x2 * x2 + x3 * x3 + x1 * x2 + x1 * x3 + x2 * x3)
            syy += det * (y1 * y1 + y2 * y2 + y3 * y3 + y1 * y2 + y1 * y3 + y2 * y3)
            szz += det * (z1 * z1 + z2 * z2 + z3 * z3 + z1 * z2 + z1 * z3 + z2 * z3)
        sign = -1.0 if vol6 < 0 else 1.0           # falsch orientierte Netze trotzdem positiv
        ixx_o = sign * rho * (syy + szz) / 60.0
        iyy_o = sign * rho * (sxx + szz) / 60.0
        izz_o = sign * rho * (sxx + syy) / 60.0
        mass = rho * abs(vol6) / 6.0
        cx, cy, cz = self.center_of_mass()
        return {
            "ixx": ixx_o - mass * (cy * cy + cz * cz),
            "iyy": iyy_o - mass * (cx * cx + cz * cz),
            "izz": izz_o - mass * (cx * cx + cy * cy),
        }

    def is_watertight(self) -> bool:
        """Wahr, wenn jede ungerichtete Kante genau zweimal vorkommt – einmal je Richtung."""
        if not self.faces:
            return False
        directed: dict[tuple[int, int], int] = {}
        for a, b, c in self.faces:
            if a == b or b == c or a == c:
                return False
            for e in ((a, b), (b, c), (c, a)):
                directed[e] = directed.get(e, 0) + 1
        for (a, b), n in directed.items():
            if n != 1 or directed.get((b, a), 0) != 1:
                return False
        return True

    def validate(self) -> dict:
        """Prüft Indizes und zählt degenerierte Dreiecke (doppelte Indizes oder Fläche ≈ 0)."""
        n = len(self.vertices)
        bad_index = 0
        degenerate = 0
        for f in self.faces:
            if len(f) != 3 or any((not isinstance(i, int)) or i < 0 or i >= n for i in f):
                bad_index += 1
                continue
            a, b, c = f
            if a == b or b == c or a == c:
                degenerate += 1
                continue
            area2 = _norm(_cross(_sub(self.vertices[b], self.vertices[a]),
                                 _sub(self.vertices[c], self.vertices[a])))
            if area2 <= 1e-12:
                degenerate += 1
        return {
            "gueltig": bad_index == 0,
            "punkte": n,
            "dreiecke": len(self.faces),
            "ungueltige_indizes": bad_index,
            "degenerierte_dreiecke": degenerate,
        }

    def stats(self, material: str | None = None) -> dict:
        """Kennzahlen für Ausgabe/HUD: Dreiecke, Punkte, Volumen cm³, Fläche cm², Bounding-Box,
        Schwerpunkt, Wasserdichtigkeit, Masse g (über ``engineering.find_material``)."""
        b = self.bounds()
        vol_cm3 = self.volume() / 1000.0
        mat = find_material(material) if material else None
        return {
            "dreiecke": len(self.faces),
            "punkte": len(self.vertices),
            "volumen_cm3": round(vol_cm3, 4),
            "flaeche_cm2": round(self.surface_area() / 100.0, 4),
            "bounding_box": {"min": _round3(b["min"]), "max": _round3(b["max"]), "groesse": _round3(b["groesse"])},
            "schwerpunkt": _round3(self.center_of_mass()),
            "wasserdicht": self.is_watertight(),
            "masse_g": round(vol_cm3 * mat.density, 3) if mat else None,
            "material": mat.name if mat else None,
        }

    def to_json(self) -> dict:
        """``{"punkte": [[x,y,z],…], "dreiecke": [[a,b,c],…]}`` – Koordinaten auf 3 Dezimalstellen."""
        return {"punkte": [_round3(v) for v in self.vertices],
                "dreiecke": [[int(a), int(b), int(c)] for a, b, c in self.faces]}


# ------------------------------------------------------------------ Bau-Hilfen
def _quad(faces: list[Face], a: int, b: int, c: int, d: int) -> None:
    """Viereck a→b→c→d (gegen den Uhrzeigersinn von außen) als zwei Dreiecke."""
    faces.append((a, b, c))
    faces.append((a, c, d))


def _wall(faces: list[Face], lo: Sequence[int], hi: Sequence[int]) -> None:
    """Mantel zwischen zwei gleich langen Ringen (Ringe gegen den Uhrzeigersinn von +z gesehen
    → Normale nach außen; im Uhrzeigersinn → Normale zur Achse, z. B. Lochwand)."""
    n = len(lo)
    for i in range(n):
        j = (i + 1) % n
        _quad(faces, lo[i], lo[j], hi[j], hi[i])


def _fan(faces: list[Face], center: int, ring: Sequence[int], up: bool) -> None:
    """Deckel (``up`` → Normale +z) oder Boden um einen Mittelpunkt."""
    n = len(ring)
    for i in range(n):
        j = (i + 1) % n
        if up:
            faces.append((center, ring[i], ring[j]))
        else:
            faces.append((center, ring[j], ring[i]))


def _annulus(faces: list[Face], inner: Sequence[int], outer: Sequence[int], up: bool) -> None:
    """Ringfläche zwischen Innen- und Außenring (beide gegen den Uhrzeigersinn)."""
    n = len(inner)
    for i in range(n):
        j = (i + 1) % n
        if up:
            _quad(faces, outer[i], outer[j], inner[j], inner[i])
        else:
            _quad(faces, inner[i], inner[j], outer[j], outer[i])


def _ring(verts: list[Vec], r: float, z: float, n: int, cx: float = 0.0, cy: float = 0.0) -> list[int]:
    """Fügt ``n`` Punkte eines Kreises (gegen den Uhrzeigersinn) hinzu und liefert ihre Indizes."""
    start = len(verts)
    for i in range(n):
        a = 2.0 * math.pi * i / n
        verts.append((cx + r * math.cos(a), cy + r * math.sin(a), z))
    return list(range(start, start + n))


def _pos(name: str, value: Any) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0:
        raise ValueError(f"{name} muss größer als 0 sein (erhalten: {fmt_number(float(value))}).")
    return v


def _segments(value: Any) -> int:
    try:
        f = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"segments muss eine ganze Zahl sein, nicht {value!r}.") from None
    if not f.is_integer():
        raise ValueError(f"segments muss eine ganze Zahl sein (erhalten: {fmt_number(f)}).")
    n = int(f)
    if n < MIN_SEGMENTS or n > MAX_SEGMENTS:
        raise ValueError(f"segments muss zwischen {MIN_SEGMENTS} und {MAX_SEGMENTS} liegen (erhalten: {n}).")
    return n


# ------------------------------------------------------------------ Primitive
def box(l: float, b: float, h: float) -> Mesh:
    """Quader l × b × h, mittig in x/y, z von 0 bis h."""
    l, b, h = _pos("l", l), _pos("b", b), _pos("h", h)
    x, y = l / 2.0, b / 2.0
    verts: list[Vec] = [(-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0),
                        (-x, -y, h), (x, -y, h), (x, y, h), (-x, y, h)]
    faces: list[Face] = []
    _wall(faces, [0, 1, 2, 3], [4, 5, 6, 7])
    _quad(faces, 0, 3, 2, 1)       # Boden (Normale -z)
    _quad(faces, 4, 5, 6, 7)       # Deckel (Normale +z)
    return Mesh(verts, faces, "quader")


def cylinder(d: float, h: float, segments: int = 48) -> Mesh:
    """Zylinder mit Durchmesser d und Höhe h, Achse z, Grundfläche bei z = 0."""
    d, h, n = _pos("d", d), _pos("h", h), _segments(segments)
    verts: list[Vec] = [(0.0, 0.0, 0.0), (0.0, 0.0, h)]
    lo = _ring(verts, d / 2.0, 0.0, n)
    hi = _ring(verts, d / 2.0, h, n)
    faces: list[Face] = []
    _wall(faces, lo, hi)
    _fan(faces, 0, lo, up=False)
    _fan(faces, 1, hi, up=True)
    return Mesh(verts, faces, "zylinder")


def tube(d_outer: float, d_inner: float, h: float, segments: int = 48) -> Mesh:
    """Rohr (Hohlzylinder) mit Außen-/Innendurchmesser und Höhe h, Grundfläche bei z = 0."""
    do, di, h, n = _pos("d_outer", d_outer), _pos("d_inner", d_inner), _pos("h", h), _segments(segments)
    if di >= do:
        raise ValueError(f"d_inner ({fmt_number(di)}) muss kleiner als d_outer ({fmt_number(do)}) sein.")
    verts: list[Vec] = []
    out_lo = _ring(verts, do / 2.0, 0.0, n)
    out_hi = _ring(verts, do / 2.0, h, n)
    in_lo = _ring(verts, di / 2.0, 0.0, n)
    in_hi = _ring(verts, di / 2.0, h, n)
    faces: list[Face] = []
    _wall(faces, out_lo, out_hi)
    _wall(faces, in_hi, in_lo)             # Innenwand: Normale zur Achse
    _annulus(faces, in_lo, out_lo, up=False)
    _annulus(faces, in_hi, out_hi, up=True)
    return Mesh(verts, faces, "rohr")


def cone(d_bottom: float, d_top: float, h: float, segments: int = 48) -> Mesh:
    """Kegel(stumpf): Durchmesser unten/oben (``d_top = 0`` → Spitze), Höhe h, Grundfläche bei z = 0."""
    db, h, n = _pos("d_bottom", d_bottom), _pos("h", h), _segments(segments)
    dt = float(d_top)
    if not math.isfinite(dt) or dt < 0:
        raise ValueError("d_top darf nicht negativ sein.")
    verts: list[Vec] = [(0.0, 0.0, 0.0)]
    lo = _ring(verts, db / 2.0, 0.0, n)
    faces: list[Face] = []
    _fan(faces, 0, lo, up=False)
    if dt <= 0:
        apex = len(verts)
        verts.append((0.0, 0.0, h))
        for i in range(n):
            faces.append((lo[i], lo[(i + 1) % n], apex))
    else:
        top_c = len(verts)
        verts.append((0.0, 0.0, h))
        hi = _ring(verts, dt / 2.0, h, n)
        _wall(faces, lo, hi)
        _fan(faces, top_c, hi, up=True)
    return Mesh(verts, faces, "kegel")


def sphere(d: float, segments: int = 32) -> Mesh:
    """UV-Kugel mit Durchmesser d um den Ursprung; ``segments`` Längen- und ebenso viele Breitenkreise."""
    d, n = _pos("d", d), _segments(segments)
    r = d / 2.0
    stacks = max(2, n)
    verts: list[Vec] = [(0.0, 0.0, r), (0.0, 0.0, -r)]
    rings: list[list[int]] = []
    for j in range(1, stacks):
        phi = math.pi * j / stacks               # 0 = Nordpol
        rings.append(_ring(verts, r * math.sin(phi), r * math.cos(phi), n))
    faces: list[Face] = []
    for i in range(n):
        faces.append((0, rings[0][i], rings[0][(i + 1) % n]))
    for upper, lower in zip(rings, rings[1:]):
        _wall(faces, lower, upper)
    last = rings[-1]
    for i in range(n):
        faces.append((1, last[(i + 1) % n], last[i]))
    return Mesh(verts, faces, "kugel")


# ------------------------------------------------------------------ Polygon-Triangulation (Ohrenschneiden)
def _cross2(o: Sequence[float], a: Sequence[float], b: Sequence[float]) -> float:
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def _same2(a: Sequence[float], b: Sequence[float]) -> bool:
    return abs(a[0] - b[0]) <= _EPS and abs(a[1] - b[1]) <= _EPS


def _in_triangle(p: Sequence[float], a: Sequence[float], b: Sequence[float], c: Sequence[float]) -> bool:
    """Punkt im (gegen den Uhrzeigersinn orientierten) Dreieck – Rand zählt mit."""
    return (_cross2(a, b, p) >= -_EPS and _cross2(b, c, p) >= -_EPS and _cross2(c, a, p) >= -_EPS)


def _bridge_hole(pts: Sequence[Sequence[float]], poly: list[int], hole: list[int]) -> list[int]:
    """Verbindet ein Loch (im Uhrzeigersinn) über eine Brücke mit dem Polygon (gegen den Uhrzeigersinn)
    nach Eberly: Lochpunkt M mit größtem x, Strahl in +x-Richtung auf die nächste Polygonkante,
    sichtbaren Polygonpunkt P wählen, Polygon an P aufschneiden und das Loch einfügen."""
    mi = max(range(len(hole)), key=lambda k: (pts[hole[k]][0], pts[hole[k]][1]))
    m = pts[hole[mi]]
    n = len(poly)
    best_x = math.inf
    best_edge = -1
    for i in range(n):
        a, b = pts[poly[i]], pts[poly[(i + 1) % n]]
        if (a[1] - m[1]) * (b[1] - m[1]) > 0:        # beide Enden auf derselben Seite der Strahlgeraden
            continue
        if a[1] == b[1]:                              # waagerechte Kante auf dem Strahl
            x = min(x for x in (a[0], b[0]) if x >= m[0]) if max(a[0], b[0]) >= m[0] else None
        else:
            x = a[0] + (m[1] - a[1]) * (b[0] - a[0]) / (b[1] - a[1])
        if x is None or x < m[0] - _EPS or x >= best_x:
            continue
        best_x, best_edge = x, i
    if best_edge < 0:
        raise ValueError("Loch liegt nicht innerhalb der Platte.")
    a_i, b_i = best_edge, (best_edge + 1) % n
    a, b = pts[poly[a_i]], pts[poly[b_i]]
    inter = (best_x, m[1])
    if _same2(inter, a):
        p_i = a_i
    elif _same2(inter, b):
        p_i = b_i
    else:
        p_i = a_i if a[0] > b[0] else b_i
        p = pts[poly[p_i]]
        # verdeckende Reflex-Punkte im Dreieck (M, I, P) → den mit kleinstem Winkel zum Strahl nehmen
        best: tuple[float, float, int] | None = None
        for k in range(n):
            q = pts[poly[k]]
            if k == p_i or _same2(q, m):
                continue
            if _cross2(pts[poly[k - 1]], q, pts[poly[(k + 1) % n]]) >= -_EPS:
                continue                              # konvex → kann nicht verdecken
            tri = (m, inter, p) if _cross2(m, inter, p) >= 0 else (m, p, inter)
            if not _in_triangle(q, *tri):
                continue
            dx, dy = q[0] - m[0], q[1] - m[1]
            key = (math.atan2(abs(dy), dx) if dx > 0 else math.pi, dx * dx + dy * dy, k)
            if best is None or key < best:
                best = key
        if best is not None:
            p_i = best[2]
    rotated = hole[mi:] + hole[:mi]
    return poly[:p_i + 1] + rotated + [hole[mi]] + poly[p_i:]


def _ear_clip(pts: Sequence[Sequence[float]], poly: Sequence[int]) -> list[Face]:
    """Trianguliert ein einfaches Polygon (gegen den Uhrzeigersinn, Brücken-Doppelpunkte erlaubt)."""
    idx = list(poly)
    tris: list[Face] = []
    if len(idx) < 3:
        return tris

    def convex(k: int) -> float:
        return _cross2(pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % len(idx)]])

    guard = 0
    while len(idx) > 3:
        n = len(idx)
        crosses = [convex(k) for k in range(n)]
        reflex = [k for k in range(n) if crosses[k] < -_EPS]
        found = -1
        for k in range(n):
            if crosses[k] <= _EPS:
                continue
            a, b, c = pts[idx[k - 1]], pts[idx[k]], pts[idx[(k + 1) % n]]
            blocked = False
            for r in reflex:
                q = pts[idx[r]]
                if _same2(q, a) or _same2(q, b) or _same2(q, c):
                    continue
                if _in_triangle(q, a, b, c):
                    blocked = True
                    break
            if not blocked:
                found = k
                break
        if found < 0:
            # Notausgang (kollineare oder numerisch heikle Stelle): Punkt mit kleinstem |Kreuzprodukt| entfernen
            k = min(range(n), key=lambda i: abs(crosses[i]))
            if abs(crosses[k]) > _EPS:
                tris.append((idx[k - 1], idx[k], idx[(k + 1) % n]))
            del idx[k]
        else:
            tris.append((idx[found - 1], idx[found], idx[(found + 1) % n]))
            del idx[found]
        guard += 1
        if guard > 4 * len(poly) + 16:            # pragma: no cover - Sicherheitsnetz
            raise ValueError("Triangulation konvergiert nicht.")
    if len(idx) == 3 and abs(convex(1)) > _EPS:
        tris.append((idx[0], idx[1], idx[2]))
    return tris


def triangulate_polygon(points: Sequence[Sequence[float]], outer: Sequence[int],
                        holes: Sequence[Sequence[int]] = ()) -> list[Face]:
    """Trianguliert ein Polygon mit Löchern (Indizes in ``points``, 2D). Außenring gegen den
    Uhrzeigersinn, Löcher im Uhrzeigersinn; Rückgabe: Dreiecke gegen den Uhrzeigersinn."""
    poly = list(outer)
    ordered = sorted((list(h) for h in holes), key=lambda h: -max(points[i][0] for i in h))
    for hole in ordered:
        poly = _bridge_hole(points, poly, hole)
    return _ear_clip(points, poly)


def plate_with_holes(l: float, b: float, t: float, holes: Sequence[Sequence[float]] = (),
                     segments: int = 24) -> Mesh:
    """Platte l × b × t (mittig in x/y, z von 0 bis t) mit Durchgangsbohrungen ``[(x, y, d), …]``.
    Deckel und Boden werden exakt trianguliert (Ohrenschneiden mit Lochbrücken, kein Boolean).
    Löcher müssen vollständig innerhalb liegen und dürfen sich nicht überlappen (sonst ``ValueError``)."""
    l, b, t, n = _pos("l", l), _pos("b", b), _pos("t", t), _segments(segments)
    hl: list[tuple[float, float, float]] = []
    for k, h in enumerate(holes or ()):
        if len(h) != 3:
            raise ValueError(f"Loch {k + 1}: erwartet (x, y, d).")
        x, y, d = float(h[0]), float(h[1]), _pos(f"Loch {k + 1}: d", h[2])
        if abs(x) + d / 2.0 >= l / 2.0 or abs(y) + d / 2.0 >= b / 2.0:
            raise ValueError(f"Loch {k + 1} ({fmt_number(x)}/{fmt_number(y)}, d={fmt_number(d)}) schneidet den Plattenrand.")
        for j, (x2, y2, d2) in enumerate(hl):
            if math.hypot(x - x2, y - y2) <= (d + d2) / 2.0:
                raise ValueError(f"Loch {k + 1} überlappt Loch {j + 1}.")
        hl.append((x, y, d))
    if 4 + len(hl) * (n + 2) > MAX_POLY_POINTS:
        raise ValueError(f"Zu viele Lochpunkte (max. {MAX_POLY_POINTS}) – weniger Löcher oder Segmente wählen.")

    verts: list[Vec] = []
    x, y = l / 2.0, b / 2.0
    # Boden (z = 0): Außenring gegen den Uhrzeigersinn, Lochringe im Uhrzeigersinn
    outer_lo = list(range(4))
    verts += [(-x, -y, 0.0), (x, -y, 0.0), (x, y, 0.0), (-x, y, 0.0)]
    holes_lo: list[list[int]] = []
    for hx, hy, hd in hl:
        ring = _ring(verts, hd / 2.0, 0.0, n, hx, hy)
        holes_lo.append([ring[0]] + ring[1:][::-1])
    # Deckel (z = t): gleiche Reihenfolge, versetzt
    off = len(verts)
    verts += [(vx, vy, t) for vx, vy, _ in verts[:off]]
    outer_hi = [i + off for i in outer_lo]
    holes_hi = [[i + off for i in h] for h in holes_lo]

    pts2d = [(vx, vy) for vx, vy, _ in verts[:off]]
    cap = triangulate_polygon(pts2d, outer_lo, holes_lo)
    faces: list[Face] = []
    faces += [(a + off, b_, c + off) for a, b_, c in ((a, b_ + off, c) for a, b_, c in cap)]   # Deckel +z
    faces += [(a, c, b_) for a, b_, c in cap]                                                 # Boden -z
    _wall(faces, outer_lo, outer_hi)
    for lo, hi in zip(holes_lo, holes_hi):
        _wall(faces, lo, hi)                      # Lochringe im Uhrzeigersinn → Normale zur Lochachse
    return Mesh(verts, faces, "lochplatte")


def drone_frame_parts(wheelbase_mm: float, arm_width_mm: float, arm_thickness_mm: float, plate_mm: float,
                      motor_hole_mm: float = 12.0, arms: int = 4) -> list[Mesh]:
    """Einzelteile des Drohnenrahmens: Mittelplatte, ``arms`` Arme (gedrehte Quader) und Motorböden
    (Rohre mit Motorloch) – jedes Teil für sich wasserdicht."""
    wb, aw, at, pl = (_pos("wheelbase_mm", wheelbase_mm), _pos("arm_width_mm", arm_width_mm),
                      _pos("arm_thickness_mm", arm_thickness_mm), _pos("plate_mm", plate_mm))
    mh = _pos("motor_hole_mm", motor_hole_mm)
    try:
        na = int(float(arms))
    except (TypeError, ValueError):
        raise ValueError(f"arms muss eine ganze Zahl sein, nicht {arms!r}.") from None
    if na < MIN_ARMS or na > MAX_ARMS or float(arms) != na:
        raise ValueError(f"arms muss zwischen {MIN_ARMS} und {MAX_ARMS} liegen (erhalten: {arms!r}).")
    wall = max(3.0, aw * 0.25)
    mount_d = max(mh + 2.0 * wall, aw)
    radius = wb / 2.0
    arm_start = pl / 2.0
    arm_end = radius - mount_d / 2.0
    if arm_end <= arm_start:
        raise ValueError("Radstand zu klein für Mittelplatte und Motorböden.")
    parts = [box(pl, pl, at)]
    parts[0].name = "mittelplatte"
    length = arm_end - arm_start
    for i in range(na):
        angle = 360.0 * i / na + 180.0 / na
        arm = box(length, aw, at).translate((arm_start + arm_end) / 2.0, 0.0, 0.0).rotate("z", angle)
        arm.name = f"arm_{i + 1}"
        c, s = _cos_sin(angle)
        mount = tube(mount_d, mh, at).translate(radius * c, radius * s, 0.0)
        mount.name = f"motorboden_{i + 1}"
        parts.append(arm)
        parts.append(mount)
    return parts


def drone_frame(wheelbase_mm: float, arm_width_mm: float, arm_thickness_mm: float, plate_mm: float,
                motor_hole_mm: float = 12.0, arms: int = 4) -> Mesh:
    """X-Rahmen als Baugruppe: Mittelplatte + Arme + Motorböden (Konkatenation; Überlappungen an den
    Verbindungen zählen im Volumen doppelt)."""
    parts = drone_frame_parts(wheelbase_mm, arm_width_mm, arm_thickness_mm, plate_mm, motor_hole_mm, arms)
    mesh = parts[0]
    for p in parts[1:]:
        mesh = mesh.merge(p)
    mesh.name = "drohnenrahmen"
    return mesh


# ------------------------------------------------------------------ Tabelle + build
PRIMITIVES: dict[str, dict] = {
    "quader": {
        "fn": box,
        "parameter": [("l", "Länge in mm (x)", None), ("b", "Breite in mm (y)", None), ("h", "Höhe in mm (z)", None)],
        "beschreibung": "Quader l × b × h, mittig in x/y, steht auf z = 0.",
    },
    "zylinder": {
        "fn": cylinder,
        "parameter": [("d", "Durchmesser in mm", None), ("h", "Höhe in mm", None),
                      ("segments", "Segmente am Umfang (3–256)", 48)],
        "beschreibung": "Zylinder um die z-Achse, steht auf z = 0.",
    },
    "rohr": {
        "fn": tube,
        "parameter": [("d_outer", "Außendurchmesser in mm (Alias: da, d_aussen)", None),
                      ("d_inner", "Innendurchmesser in mm (Alias: di, d_innen)", None),
                      ("h", "Höhe in mm", None), ("segments", "Segmente am Umfang (3–256)", 48)],
        "beschreibung": "Rohr/Hohlzylinder um die z-Achse.",
    },
    "kegel": {
        "fn": cone,
        "parameter": [("d_bottom", "Durchmesser unten in mm (Alias: d_unten)", None),
                      ("d_top", "Durchmesser oben in mm, 0 = Spitze (Alias: d_oben)", 0.0),
                      ("h", "Höhe in mm", None), ("segments", "Segmente am Umfang (3–256)", 48)],
        "beschreibung": "Kegel oder Kegelstumpf, steht auf z = 0.",
    },
    "kugel": {
        "fn": sphere,
        "parameter": [("d", "Durchmesser in mm", None), ("segments", "Segmente (3–256)", 32)],
        "beschreibung": "UV-Kugel um den Ursprung.",
    },
    "lochplatte": {
        "fn": plate_with_holes,
        "parameter": [("l", "Länge in mm", None), ("b", "Breite in mm", None), ("t", "Dicke in mm", None),
                      ("holes", "Bohrungen »x/y/d; x/y/d« in mm (Alias: loecher)", []),
                      ("segments", "Segmente je Bohrung (3–256)", 24)],
        "beschreibung": "Platte mit Durchgangsbohrungen (exakte Triangulation, kein Boolean).",
    },
    "drohnenrahmen": {
        "fn": drone_frame,
        "parameter": [("wheelbase_mm", "Radstand Motor–Motor in mm (Alias: radstand)", None),
                      ("arm_width_mm", "Armbreite in mm (Alias: armbreite)", None),
                      ("arm_thickness_mm", "Armdicke in mm (Alias: armdicke, dicke)", None),
                      ("plate_mm", "Kantenlänge der Mittelplatte in mm (Alias: platte)", None),
                      ("motor_hole_mm", "Motorloch-Durchmesser in mm (Alias: motorloch)", 12.0),
                      ("arms", "Anzahl Arme (3–8) (Alias: arme)", 4)],
        "beschreibung": "X-Rahmen als Baugruppe: Mittelplatte, Arme, Motorböden.",
    },
}

_KIND_ALIASES = {
    "box": "quader", "wuerfel": "quader", "würfel": "quader", "klotz": "quader", "block": "quader", "platte": "quader",
    "cylinder": "zylinder", "zyl": "zylinder", "rundstab": "zylinder", "scheibe": "zylinder",
    "tube": "rohr", "hohlzylinder": "rohr", "ring": "rohr", "huelse": "rohr", "hülse": "rohr",
    "cone": "kegel", "kegelstumpf": "kegel", "trichter": "kegel",
    "sphere": "kugel", "ball": "kugel",
    "plate_with_holes": "lochplatte", "bohrplatte": "lochplatte", "flansch": "lochplatte",
    "drone_frame": "drohnenrahmen", "rahmen": "drohnenrahmen", "frame": "drohnenrahmen", "drohne": "drohnenrahmen",
}
_PARAM_ALIASES = {
    "laenge": "l", "länge": "l", "length": "l", "x": "l",
    "breite": "b", "width": "b", "y": "b",
    "hoehe": "h", "höhe": "h", "height": "h", "z": "h",
    "dicke": "t", "staerke": "t", "stärke": "t", "thickness": "t",
    "durchmesser": "d", "dm": "d", "diameter": "d",
    "da": "d_outer", "d_aussen": "d_outer", "d_außen": "d_outer", "aussen": "d_outer", "außen": "d_outer",
    "di": "d_inner", "d_innen": "d_inner", "innen": "d_inner",
    "d_unten": "d_bottom", "unten": "d_bottom", "d1": "d_bottom",
    "d_oben": "d_top", "oben": "d_top", "d2": "d_top",
    "segmente": "segments", "seg": "segments", "n": "segments", "aufloesung": "segments", "auflösung": "segments",
    "loecher": "holes", "löcher": "holes", "bohrungen": "holes", "holes": "holes",
    "radstand": "wheelbase_mm", "wheelbase": "wheelbase_mm",
    "armbreite": "arm_width_mm", "arm_width": "arm_width_mm",
    "armdicke": "arm_thickness_mm", "arm_thickness": "arm_thickness_mm",
    "platte": "plate_mm", "plate": "plate_mm", "mittelplatte": "plate_mm",
    "motorloch": "motor_hole_mm", "motor_hole": "motor_hole_mm",
    "arme": "arms", "anzahl_arme": "arms",
}
_INT_PARAMS = {"segments", "arms"}
_ZERO_OK = {"d_top"}


def _norm_key(text: str) -> str:
    s = str(text).strip().lower()
    for a, b in (("ä", "ae"), ("ö", "oe"), ("ü", "ue"), ("ß", "ss")):
        s = s.replace(a, b)
    return re.sub(r"[\s\-]+", "_", s)


def _to_float(name: str, value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} muss eine Zahl sein, nicht ja/nein.")
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        s = value.strip().lower()
        for unit in ("millimeter", "mm"):
            if s.endswith(unit):
                s = s[: -len(unit)].strip()
        if s.count(",") == 1 and "." not in s:
            s = s.replace(",", ".")
        s = s.replace(" ", "")
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"{name} muss eine Zahl sein, nicht {value!r}.") from None
    else:
        raise ValueError(f"{name} muss eine Zahl sein, nicht {value!r}.")
    if not math.isfinite(f):
        raise ValueError(f"{name} muss eine endliche Zahl sein.")
    return f


def parse_holes(value: Any) -> list[tuple[float, float, float]]:
    """Bohrungen aus ``"x/y/d; x/y/d"`` (auch ``x,y,d`` mit Dezimalkomma nur bei eindeutiger Trennung),
    aus Listen von Tripeln oder Dicts ``{"x","y","d"}``."""
    if value is None:
        return []
    out: list[tuple[float, float, float]] = []
    if isinstance(value, str):
        for k, part in enumerate(p.strip() for p in _HOLE_SEP.split(value) if p.strip()):
            fields = [f for f in _HOLE_PART.split(part) if f]
            if len(fields) != 3:
                # Fallback: Komma als Trenner ("10,20,5")
                fields = [f.strip() for f in part.split(",") if f.strip()]
            if len(fields) != 3:
                raise ValueError(f"Loch {k + 1}: erwartet »x/y/d«, erhalten {part!r}.")
            out.append(tuple(_to_float(f"Loch {k + 1}", f) for f in fields))  # type: ignore[arg-type]
        return out
    if isinstance(value, dict):
        value = [value]
    for k, h in enumerate(value):
        if isinstance(h, dict):
            h = (h.get("x", 0), h.get("y", 0), h.get("d"))
        if not isinstance(h, (list, tuple)) or len(h) != 3:
            raise ValueError(f"Loch {k + 1}: erwartet (x, y, d).")
        out.append(tuple(_to_float(f"Loch {k + 1}", v) for v in h))  # type: ignore[arg-type]
    return out


def parse_params(text: Any) -> dict:
    """``"l=100, b=50, h=10"`` → ``{"l": "100", …}``; Dicts werden durchgereicht, JSON-Objekte geparst.
    Kommas gelten nur vor ``name=`` als Trenner, damit Dezimalkommas erhalten bleiben."""
    if text is None:
        return {}
    if isinstance(text, dict):
        return dict(text)
    s = str(text).strip()
    if not s:
        return {}
    if s.startswith("{"):
        try:
            data = json.loads(s)
        except json.JSONDecodeError:
            data = None
        if isinstance(data, dict):
            return data
    out: dict[str, Any] = {}
    for part in _KEY_SEP.split(s):
        part = part.strip().strip(",")
        if not part:
            continue
        if "=" in part:
            key, _, val = part.partition("=")
        elif ":" in part:
            key, _, val = part.partition(":")
        else:
            raise ValueError(f"Parameter ohne »name=wert«: {part!r}")
        out[key.strip()] = val.strip()
    return out


def resolve_kind(kind: Any) -> str:
    """Deutscher Primitiv-Name aus toleranter Eingabe (Groß/Klein, Umlaute, englische Aliasse)."""
    key = _norm_key(kind or "")
    if key in PRIMITIVES:
        return key
    if key in _KIND_ALIASES:
        return _KIND_ALIASES[key]
    for k in PRIMITIVES:
        if key and (k.startswith(key) or key.startswith(k)) and len(key) >= 4:
            return k
    raise ValueError(f"Unbekannte Bauteil-Art {kind!r}. Verfügbar: {', '.join(PRIMITIVES)}.")


def build(kind: str, params: dict | str | None) -> Mesh:
    """Baut ein Primitiv tolerant: Dezimalkomma, Strings, Alias-Namen, Löcher als »x/y/d; …«.
    Prüft Werte (> 0 und ≤ 10 000 mm, Segmente 3–256) und das Dreieckslimit :data:`MAX_FACES`."""
    name = resolve_kind(kind)
    spec = PRIMITIVES[name]
    raw = parse_params(params)
    known = {p[0] for p in spec["parameter"]}
    kwargs: dict[str, Any] = {}
    for key, value in raw.items():
        nk = _norm_key(key)
        nk = _PARAM_ALIASES.get(nk, nk)
        if nk not in known:
            raise ValueError(f"Unbekannter Parameter »{key}« für {name}. Erlaubt: {', '.join(sorted(known))}.")
        if value is None or (isinstance(value, str) and not value.strip()):
            continue
        if nk == "holes":
            kwargs[nk] = parse_holes(value)
            continue
        f = _to_float(nk, value)
        if nk in _INT_PARAMS:
            if not f.is_integer():
                raise ValueError(f"{nk} muss eine ganze Zahl sein (erhalten: {fmt_number(f)}).")
            kwargs[nk] = int(f)
            continue
        if f < 0 or (f == 0 and nk not in _ZERO_OK):
            raise ValueError(f"{nk} muss größer als 0 sein (erhalten: {fmt_number(f)}).")
        if f > MAX_DIM_MM:
            raise ValueError(f"{nk} darf höchstens {fmt_number(MAX_DIM_MM)} mm sein (erhalten: {fmt_number(f)}).")
        kwargs[nk] = f
    missing = [p[0] for p in spec["parameter"] if p[2] is None and p[0] not in kwargs]
    if missing:
        raise ValueError(f"Fehlende Parameter für {name}: {', '.join(missing)}.")
    for hx, hy, hd in kwargs.get("holes", []):
        if hd <= 0 or hd > MAX_DIM_MM or abs(hx) > MAX_DIM_MM or abs(hy) > MAX_DIM_MM:
            raise ValueError("Lochmaße müssen > 0 und ≤ 10 000 mm sein.")
    mesh = spec["fn"](**kwargs)
    if len(mesh.faces) > MAX_FACES:
        raise ValueError(f"Modell hat {len(mesh.faces)} Dreiecke – mehr als erlaubt ({MAX_FACES}). "
                         "Weniger Segmente wählen.")
    mesh.name = name
    return mesh


# ------------------------------------------------------------------ Dateien
def _dedupe(triangles: Iterable[tuple[Vec, Vec, Vec]], name: str = "") -> Mesh:
    """Baut aus losen Dreiecken ein indiziertes Netz; gleiche Punkte (Rundung 1e-6) werden zusammengeführt."""
    index: dict[tuple[float, float, float], int] = {}
    verts: list[Vec] = []
    faces: list[Face] = []
    for tri in triangles:
        ids = []
        for p in tri:
            key = (round(p[0], _DEDUP_DIGITS), round(p[1], _DEDUP_DIGITS), round(p[2], _DEDUP_DIGITS))
            i = index.get(key)
            if i is None:
                i = len(verts)
                index[key] = i
                verts.append(_vec(p))
            ids.append(i)
        faces.append((ids[0], ids[1], ids[2]))
    return Mesh(verts, faces, name)


def _normal(a: Vec, b: Vec, c: Vec) -> Vec:
    n = _cross(_sub(b, a), _sub(c, a))
    length = _norm(n)
    return (n[0] / length, n[1] / length, n[2] / length) if length > 0 else (0.0, 0.0, 0.0)


def _check_size(path: str | Path) -> None:
    size = os.path.getsize(path)
    if size > MAX_FILE_BYTES:
        raise ValueError(f"Datei zu groß ({size // (1024 * 1024)} MB, max. {MAX_FILE_BYTES // (1024 * 1024)} MB).")


def _check_faces(mesh: Mesh) -> Mesh:
    if len(mesh.faces) > MAX_FACES:
        raise ValueError(f"Datei enthält {len(mesh.faces)} Dreiecke – mehr als erlaubt ({MAX_FACES}).")
    return mesh


def write_stl(mesh: Mesh, path: str | Path, binary: bool = True) -> None:
    """Schreibt STL (binär, sonst ASCII). Normalen werden aus der Dreiecksorientierung berechnet."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    vs = mesh.vertices
    if binary:
        header = (mesh.name or "OBITO").encode("utf-8", "replace")[:70].ljust(80, b"\0")
        if header.lower().startswith(b"solid"):
            header = b"OBITO " + header[6:]
        out = bytearray(header)
        out += struct.pack("<I", len(mesh.faces))
        pack = struct.Struct("<12fH").pack
        for a, b, c in mesh.faces:
            n = _normal(vs[a], vs[b], vs[c])
            out += pack(*n, *vs[a], *vs[b], *vs[c], 0)
        path.write_bytes(bytes(out))
        return
    name = re.sub(r"\s+", "_", mesh.name or "obito")
    lines = [f"solid {name}"]
    for a, b, c in mesh.faces:
        n = _normal(vs[a], vs[b], vs[c])
        lines.append(f"  facet normal {n[0]:.6e} {n[1]:.6e} {n[2]:.6e}")
        lines.append("    outer loop")
        for p in (vs[a], vs[b], vs[c]):
            lines.append(f"      vertex {p[0]:.6e} {p[1]:.6e} {p[2]:.6e}")
        lines.append("    endloop")
        lines.append("  endfacet")
    lines.append(f"endsolid {name}")
    path.write_text("\n".join(lines) + "\n", encoding="ascii")


def _is_binary_stl(data: bytes) -> bool:
    if len(data) < 84:
        return False
    n = struct.unpack_from("<I", data, 80)[0]
    if 84 + 50 * n == len(data):
        return True
    head = data[:512].lstrip()
    if not head.lower().startswith(b"solid"):
        return True
    try:
        text = data[:4096].decode("ascii")
    except UnicodeDecodeError:
        return True
    return "facet" not in text and "endsolid" not in text and len(data) > 84


def read_stl(path: str | Path) -> Mesh:
    """Liest STL (ASCII oder binär, Erkennung über Header und Dateigröße). STL kennt keine Indizes –
    gleiche Punkte werden mit Rundung 1e-6 zusammengeführt."""
    path = Path(path)
    _check_size(path)
    data = path.read_bytes()
    name = path.stem
    if _is_binary_stl(data):
        n = struct.unpack_from("<I", data, 80)[0]
        if n > MAX_FACES:
            raise ValueError(f"Datei enthält {n} Dreiecke – mehr als erlaubt ({MAX_FACES}).")
        if 84 + 50 * n > len(data):
            raise ValueError("Binäre STL-Datei ist unvollständig.")
        body = data[84:84 + 50 * n]
        tris = (((r[3], r[4], r[5]), (r[6], r[7], r[8]), (r[9], r[10], r[11]))
                for r in struct.iter_unpack("<12fH", body))
        return _dedupe(tris, name)
    text = data.decode("utf-8", "replace")
    tris: list[tuple[Vec, Vec, Vec]] = []
    cur: list[Vec] = []
    count = 0
    for line in text.splitlines():
        parts = line.split()
        if not parts:
            continue
        word = parts[0].lower()
        if word == "vertex" and len(parts) >= 4:
            try:
                cur.append((float(parts[1]), float(parts[2]), float(parts[3])))
            except ValueError:
                raise ValueError(f"Ungültige Koordinate in STL: {line.strip()!r}") from None
        elif word == "endfacet" or word == "endloop":
            if len(cur) >= 3:
                tris.append((cur[0], cur[1], cur[2]))
                count += 1
                if count > MAX_FACES:
                    raise ValueError(f"Datei enthält mehr als {MAX_FACES} Dreiecke.")
            cur = []
        elif word == "solid" and len(parts) > 1 and not tris:
            name = " ".join(parts[1:])
    if cur and len(cur) >= 3:
        tris.append((cur[0], cur[1], cur[2]))
    if not tris:
        raise ValueError("Keine Dreiecke in der STL-Datei gefunden.")
    return _dedupe(tris, name)


def write_obj(mesh: Mesh, path: str | Path) -> None:
    """Schreibt Wavefront OBJ (``v``/``f``, Indizes ab 1)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    obj_name = re.sub(r"\s+", "_", mesh.name or "modell")
    lines = [f"# OBITO 3D – {mesh.name or 'modell'}", f"o {obj_name}"]
    lines += [f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}" for v in mesh.vertices]
    lines += [f"f {a + 1} {b + 1} {c + 1}" for a, b, c in mesh.faces]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_obj(path: str | Path) -> Mesh:
    """Liest OBJ: ``v`` und ``f`` (auch ``f 1/1/1 2/2/2 …`` und Polygone > 3 → Fächer-Triangulation,
    negative Indizes relativ zum Ende)."""
    path = Path(path)
    _check_size(path)
    verts: list[Vec] = []
    faces: list[Face] = []
    name = path.stem
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for raw in fh:
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            tag = parts[0]
            if tag == "v" and len(parts) >= 4:
                try:
                    verts.append((float(parts[1]), float(parts[2]), float(parts[3])))
                except ValueError:
                    raise ValueError(f"Ungültiger Punkt in OBJ: {line!r}") from None
            elif tag == "f" and len(parts) >= 4:
                idx: list[int] = []
                for tok in parts[1:]:
                    first = tok.split("/", 1)[0]
                    try:
                        i = int(first)
                    except ValueError:
                        raise ValueError(f"Ungültiger Index in OBJ: {line!r}") from None
                    if i < 0:
                        i = len(verts) + i
                    else:
                        i -= 1
                    if i < 0 or i >= len(verts):
                        raise ValueError(f"Index außerhalb des Bereichs in OBJ: {line!r}")
                    idx.append(i)
                for k in range(1, len(idx) - 1):
                    faces.append((idx[0], idx[k], idx[k + 1]))
                if len(faces) > MAX_FACES:
                    raise ValueError(f"Datei enthält mehr als {MAX_FACES} Dreiecke.")
            elif tag in ("o", "g") and len(parts) > 1 and not faces:
                name = " ".join(parts[1:])
    if not faces:
        raise ValueError("Keine Dreiecke in der OBJ-Datei gefunden.")
    return Mesh(verts, faces, name)


def load(path: str | Path) -> Mesh:
    """Lädt STL oder OBJ nach Dateiendung; ``ValueError`` bei unbekannter Endung, zu großer Datei
    oder mehr als :data:`MAX_FACES` Dreiecken."""
    p = Path(path)
    if not p.is_file():
        raise ValueError(f"Datei nicht gefunden: {path}")
    _check_size(p)
    ext = p.suffix.lower()
    if ext == ".stl":
        return _check_faces(read_stl(p))
    if ext == ".obj":
        return _check_faces(read_obj(p))
    raise ValueError(f"Unbekannte Endung {ext or '(keine)'} – unterstützt: .stl, .obj")


# ------------------------------------------------------------------ ModelStore
_SCHEMA = """
CREATE TABLE IF NOT EXISTS modelle (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    name        TEXT    NOT NULL,
    version     INTEGER NOT NULL DEFAULT 1,
    project     TEXT,
    kind        TEXT,
    params      TEXT,
    material    TEXT,
    stats       TEXT,
    file        TEXT,
    note        TEXT    NOT NULL DEFAULT '',
    created_at  REAL    NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_modelle_name ON modelle(name, project);
CREATE INDEX IF NOT EXISTS idx_modelle_project ON modelle(project);
"""


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _loads(raw: str | None, default: Any) -> Any:
    if not raw:
        return default
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return default


class ModelStore:
    """Thread-sicherer Modellspeicher: Metadaten in SQLite, Geometrie als binäre STL-Datei je Version
    unter ``<files_dir>/<id>_v<version>.stl``."""

    def __init__(self, path: str | Path, files_dir: str | Path):
        self.path = str(path)
        self.files_dir = Path(files_dir)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.files_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._closed = False

    # ------------------------------------------------------------ intern
    @staticmethod
    def _row(row: sqlite3.Row | None) -> dict | None:
        if row is None:
            return None
        return {
            "id": row["id"],
            "name": row["name"],
            "version": row["version"],
            "projekt": row["project"],
            "art": row["kind"],
            "parameter": _loads(row["params"], {}),
            "material": row["material"],
            "statistik": _loads(row["stats"], {}),
            "datei": row["file"],
            "notiz": row["note"] or "",
            "erstellt": _iso(row["created_at"]),
        }

    @staticmethod
    def _clean_project(project: str | None) -> str | None:
        p = (project or "").strip() if isinstance(project, str) else project
        return p or None

    def _file_for(self, model_id: int, version: int) -> Path:
        return self.files_dir / f"{model_id}_v{version}.stl"

    # ------------------------------------------------------------ API
    def save(self, mesh: Mesh, name: str, *, kind: str | None = None, params: dict | None = None,
             project: str | None = None, material: str | None = None, note: str = "") -> dict:
        """Speichert ``mesh`` als neue Version (wenn Name + Projekt existieren) oder als Version 1."""
        name = (name or "").strip()
        if not name:
            raise ValueError("Name darf nicht leer sein.")
        if not mesh.faces:
            raise ValueError("Leeres Netz kann nicht gespeichert werden.")
        project = self._clean_project(project)
        stats = mesh.stats(material)
        with self._lock:
            row = self._db.execute("SELECT MAX(version) AS v FROM modelle WHERE name = ? AND project IS ?",
                                   (name, project)).fetchone()
            version = int(row["v"] or 0) + 1
            cur = self._db.execute(
                "INSERT INTO modelle(name, version, project, kind, params, material, stats, file, note, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (name, version, project, kind, json.dumps(params or {}, ensure_ascii=False),
                 material, json.dumps(stats, ensure_ascii=False), "", note or "", time.time()))
            model_id = int(cur.lastrowid)
            file = self._file_for(model_id, version)
            write_stl(mesh, file, binary=True)
            self._db.execute("UPDATE modelle SET file = ? WHERE id = ?", (str(file), model_id))
            self._db.commit()
            return self.get(model_id)  # type: ignore[return-value]

    def get(self, model_id: int) -> dict | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM modelle WHERE id = ?", (int(model_id),)).fetchone()
            return self._row(row)

    def find(self, name_or_id: Any, project: str | None = None) -> dict | None:
        """Modell per ID oder per Name (neueste Version, optional im Projekt)."""
        s = str(name_or_id).strip()
        if s.isdigit():
            found = self.get(int(s))
            if found:
                return found
        with self._lock:
            if project is not None:
                row = self._db.execute(
                    "SELECT * FROM modelle WHERE lower(name) = lower(?) AND project IS ? "
                    "ORDER BY version DESC LIMIT 1", (s, self._clean_project(project))).fetchone()
            else:
                row = self._db.execute(
                    "SELECT * FROM modelle WHERE lower(name) = lower(?) ORDER BY created_at DESC, version DESC LIMIT 1",
                    (s,)).fetchone()
            return self._row(row)

    def mesh(self, model_id: int) -> Mesh:
        """Lädt die Geometrie eines Modells (``ValueError`` wenn unbekannt oder Datei fehlt)."""
        info = self.get(model_id)
        if info is None:
            raise ValueError(f"Modell {model_id} ist unbekannt.")
        file = Path(info["datei"] or "")
        if not file.is_file():
            raise ValueError(f"Datei zu Modell {model_id} fehlt: {file}")
        mesh = read_stl(file)
        mesh.name = info["name"]
        return mesh

    def list(self, project: str | None = None, name: str | None = None) -> list[dict]:
        """Neueste Version je Name+Projekt, neueste zuerst; optional nach Projekt/Name gefiltert."""
        where = ["m.version = (SELECT MAX(version) FROM modelle x WHERE x.name = m.name AND x.project IS m.project)"]
        args: list[Any] = []
        if project is not None:
            where.append("m.project IS ?")
            args.append(self._clean_project(project))
        if name:
            where.append("lower(m.name) = lower(?)")
            args.append(name.strip())
        sql = "SELECT * FROM modelle m WHERE " + " AND ".join(where) + " ORDER BY m.created_at DESC, m.id DESC"
        with self._lock:
            return [self._row(r) for r in self._db.execute(sql, args).fetchall()]  # type: ignore[misc]

    def versions(self, name: str, project: str | None = None) -> list[dict]:
        """Alle Versionen eines Modells, aufsteigend."""
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM modelle WHERE lower(name) = lower(?) AND project IS ? ORDER BY version ASC",
                ((name or "").strip(), self._clean_project(project))).fetchall()
            return [self._row(r) for r in rows]  # type: ignore[misc]

    def delete(self, model_id: int) -> bool:
        """Löscht eine Modellversion samt Datei."""
        with self._lock:
            info = self.get(model_id)
            if info is None:
                return False
            self._db.execute("DELETE FROM modelle WHERE id = ?", (int(model_id),))
            self._db.commit()
        try:
            if info["datei"]:
                Path(info["datei"]).unlink(missing_ok=True)
        except OSError:
            pass
        return True

    def import_file(self, path: str | Path, name: str | None = None, project: str | None = None) -> dict:
        """Importiert eine STL/OBJ-Datei als Modell (Art ``import``)."""
        p = Path(path)
        mesh = load(p)
        label = (name or "").strip() or p.stem
        mesh.name = label
        return self.save(mesh, label, kind="import", params={"quelle": str(p), "dreiecke": len(mesh.faces)},
                         project=project, note=f"Import aus {p.name}")

    def stats(self) -> dict:
        with self._lock:
            total = self._db.execute("SELECT COUNT(*) AS n FROM modelle").fetchone()["n"]
            names = self._db.execute("SELECT COUNT(*) AS n FROM (SELECT DISTINCT name, project FROM modelle)").fetchone()["n"]
            projects = self._db.execute(
                "SELECT COUNT(DISTINCT project) AS n FROM modelle WHERE project IS NOT NULL").fetchone()["n"]
            rows = self._db.execute("SELECT stats, file FROM modelle").fetchall()
        tris = 0
        size = 0
        for r in rows:
            tris += int((_loads(r["stats"], {}) or {}).get("dreiecke", 0) or 0)
            try:
                size += os.path.getsize(r["file"]) if r["file"] else 0
            except OSError:
                pass
        return {"versionen": int(total), "modelle": int(names), "projekte": int(projects),
                "dreiecke": tris, "dateien_bytes": size}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True


# ------------------------------------------------------------------ Werkzeuge
def _p_str(desc: str, example: str | None = None) -> dict:
    spec: dict[str, Any] = {"type": "string", "description": desc}
    if example is not None:
        spec["example"] = example
    return spec


def _schema(props: dict, required: Sequence[str]) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


def _fmt3(v: Sequence[float]) -> str:
    return "(" + "; ".join(fmt_number(float(x)) for x in v) + ")"


def stats_text(stats: dict) -> str:
    """Deutsche Kennzahl-Zeilen aus :meth:`Mesh.stats` (Dezimalkomma)."""
    bb = stats.get("bounding_box") or {}
    size = bb.get("groesse") or [0, 0, 0]
    lines = [
        f"  Dreiecke: {stats.get('dreiecke', 0)} | Punkte: {stats.get('punkte', 0)}",
        f"  Volumen: {fmt_number(float(stats.get('volumen_cm3', 0)))} cm³ | "
        f"Oberfläche: {fmt_number(float(stats.get('flaeche_cm2', 0)))} cm²",
        f"  Maße (x × y × z): {fmt_number(float(size[0]))} × {fmt_number(float(size[1]))} × "
        f"{fmt_number(float(size[2]))} mm",
        f"  Schwerpunkt: {_fmt3(stats.get('schwerpunkt') or [0, 0, 0])} mm",
        f"  Wasserdicht: {'ja' if stats.get('wasserdicht') else 'nein'}",
    ]
    if stats.get("masse_g") is not None:
        lines.append(f"  Masse: {fmt_number(float(stats['masse_g']))} g ({stats.get('material')})")
    elif stats.get("material_angabe"):
        lines.append(f"  Masse: unbekanntes Material »{stats['material_angabe']}«")
    return "\n".join(lines)


def _model_text(info: dict) -> str:
    head = (f"Modell »{info['name']}« (ID {info['id']}, Version {info['version']}"
            f"{', Projekt ' + info['projekt'] if info.get('projekt') else ''}"
            f"{', Art ' + info['art'] if info.get('art') else ''})")
    lines = [head]
    params = info.get("parameter") or {}
    if params:
        lines.append("  Parameter: " + ", ".join(f"{k}={_fmt_param(v)}" for k, v in params.items()))
    lines.append(stats_text(info.get("statistik") or {}))
    if info.get("material") and (info.get("statistik") or {}).get("masse_g") is None:
        lines.append(f"  Material »{info['material']}« ist nicht in der Datenbank – keine Masse berechnet.")
    if info.get("notiz"):
        lines.append(f"  Notiz: {info['notiz']}")
    lines.append(f"  Datei: {info.get('datei')} ({info.get('erstellt')})")
    return "\n".join(lines)


def _fmt_param(v: Any) -> str:
    if isinstance(v, bool):
        return "ja" if v else "nein"
    if isinstance(v, (int, float)):
        return fmt_number(float(v)) if isinstance(v, float) else str(v)
    if isinstance(v, (list, tuple)):
        if v and all(isinstance(h, (list, tuple)) and len(h) == 3 for h in v):
            return "; ".join("/".join(fmt_number(float(x)) for x in h) for h in v)
        return ", ".join(_fmt_param(x) for x in v)
    return str(v)


def register_tools(registry: "ToolRegistry", store: ModelStore) -> None:
    """Registriert ``modell_erzeugen``, ``modell_info`` und ``modell_liste`` (nicht gefährlich)."""
    from .tools import Tool

    def modell_erzeugen(art: str, parameter: str, name: str, projekt: str | None = None,
                        material: str | None = None) -> str:
        mesh = build(art, parse_params(parameter))
        kind = resolve_kind(art)
        clean_params = {k: v for k, v in parse_params(parameter).items()}
        mat = (material or "").strip() or None
        info = store.save(mesh, name, kind=kind, params=clean_params, project=(projekt or "").strip() or None,
                          material=mat)
        text = _model_text(info)
        if mat and find_material(mat) is None:
            text += f"\nHinweis: Material »{mat}« unbekannt – Masse nicht berechnet."
        return text

    def modell_info(name_oder_id: str) -> str:
        info = store.find(name_oder_id)
        if info is None:
            return f"Kein Modell »{name_oder_id}« gefunden."
        versions = store.versions(info["name"], info["projekt"])
        text = _model_text(info)
        if len(versions) > 1:
            text += f"\n  Versionen: {len(versions)} (" + ", ".join(f"v{v['version']} ID {v['id']}" for v in versions) + ")"
        return text

    def modell_liste(projekt: str | None = None) -> str:
        proj = (projekt or "").strip() or None
        items = store.list(project=proj) if proj else store.list()
        if not items:
            return "Keine Modelle gespeichert." if not proj else f"Keine Modelle im Projekt »{proj}«."
        lines = [f"{len(items)} Modell(e)" + (f" im Projekt »{proj}«" if proj else "") + ":"]
        for it in items:
            st = it.get("statistik") or {}
            size = (st.get("bounding_box") or {}).get("groesse") or [0, 0, 0]
            lines.append(
                f"- [{it['id']}] {it['name']} v{it['version']}"
                f"{' (' + it['projekt'] + ')' if it.get('projekt') else ''}: {it.get('art') or '–'}, "
                f"{fmt_number(float(size[0]))} × {fmt_number(float(size[1]))} × {fmt_number(float(size[2]))} mm, "
                f"{fmt_number(float(st.get('volumen_cm3', 0)))} cm³, {st.get('dreiecke', 0)} Dreiecke"
                + (f", {fmt_number(float(st['masse_g']))} g" if st.get("masse_g") is not None else ""))
        return "\n".join(lines)

    arten = ", ".join(f"{k} ({', '.join(p[0] for p in v['parameter'])})" for k, v in PRIMITIVES.items())
    registry.register(Tool(
        name="modell_erzeugen",
        description="Erzeugt ein parametrisches 3D-Bauteil (mm), speichert es als STL mit Version und liefert "
                    "Volumen, Oberfläche, Maße, Schwerpunkt und Masse (bei Material). Arten: " + arten + ". "
                    "Löcher als »x/y/d; x/y/d«.",
        parameters=_schema({
            "art": _p_str("Bauteil-Art: " + ", ".join(PRIMITIVES), "quader"),
            "parameter": _p_str("Parameter als »name=wert, …« (Dezimalkomma erlaubt)", "l=100, b=50, h=10"),
            "name": _p_str("Modellname (gleicher Name + Projekt → neue Version)", "Halterplatte"),
            "projekt": _p_str("Projektname (optional)"),
            "material": _p_str("Material für die Massenberechnung, z. B. PETG, Alu 6061, CFK (optional)"),
        }, ["art", "parameter", "name"]),
        fn=modell_erzeugen,
    ))
    registry.register(Tool(
        name="modell_info",
        description="Zeigt Kennzahlen und Parameter eines gespeicherten 3D-Modells (per Name oder ID).",
        parameters=_schema({"name_oder_id": _p_str("Modellname oder ID", "Halterplatte")}, ["name_oder_id"]),
        fn=modell_info,
    ))
    registry.register(Tool(
        name="modell_liste",
        description="Listet gespeicherte 3D-Modelle (neueste Version je Name), optional je Projekt.",
        parameters=_schema({"projekt": _p_str("Projektname (optional)")}, []),
        fn=modell_liste,
    ))


TOOL_NAMES = ("modell_erzeugen", "modell_info", "modell_liste")

__all__ = [
    "MAX_FACES", "MAX_FILE_BYTES", "MAX_DIM_MM", "MAX_SEGMENTS", "MIN_SEGMENTS", "PRIMITIVES", "TOOL_NAMES",
    "Mesh", "ModelStore", "box", "build", "cone", "cylinder", "drone_frame", "drone_frame_parts", "load",
    "parse_holes", "parse_params", "plate_with_holes", "read_obj", "read_stl", "register_tools", "resolve_kind",
    "sphere", "stats_text", "triangulate_polygon", "tube", "write_obj", "write_stl",
]
