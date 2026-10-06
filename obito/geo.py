"""Welt & Karten von OBITO: Geodäsie, Flugplanung, Sonnenstand, UTM, Orte/Routen, optionales Wetter.

Alles außer :func:`weather` arbeitet offline mit der Standardbibliothek. :func:`weather` fragt
Open-Meteo (ohne Schlüssel) ab und wird vom Brain nur aufgerufen, wenn ``cfg.online`` gesetzt ist.
Koordinaten sind WGS84 (Breite, Länge in Grad), Entfernungen in Metern, Kurse in Grad (0 = Nord,
im Uhrzeigersinn).
"""

from __future__ import annotations

import json
import logging
import math
import re
import sqlite3
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .engineering import _num, _pos, _round, fmt_number, _p_number, _p_str, _schema

if TYPE_CHECKING:  # pragma: no cover
    from .tools import ToolRegistry

log = logging.getLogger("obito.geo")

EARTH_RADIUS_M = 6371008.8
MAX_ROUTE_POINTS = 500
MAX_NAME_CHARS = 120
WEATHER_SOURCE = "Open-Meteo (open-meteo.com)"
WIND_LIMIT_KMH = 36.0
GUST_LIMIT_KMH = 50.0
RAIN_LIMIT_MM = 0.5
MIN_GROUND_SPEED = 0.5

Point = tuple[float, float]


# ------------------------------------------------------------------ Koordinaten
_DMS = re.compile(r"""(?P<deg>\d+(?:[.,]\d+)?)\s*[°º]\s*(?:(?P<min>\d+(?:[.,]\d+)?)\s*['′]\s*)?
                      (?:(?P<sec>\d+(?:[.,]\d+)?)\s*(?:"|″|'')\s*)?\s*(?P<hemi>[NSEWO])?""", re.X | re.I)
_DEC = re.compile(r"(?P<pre>[NSEWO])?\s*(?P<num>[-+]?\d+(?:[.,]\d+)?)\s*(?P<post>[NSEWO])?", re.I)


def _f(text: str) -> float:
    return float(text.replace(",", "."))


def _check(lat: float, lon: float) -> Point:
    if not (-90.0 <= lat <= 90.0):
        raise ValueError(f"Breite {fmt_number(lat)} liegt außerhalb von −90…90°.")
    if not (-180.0 <= lon <= 180.0):
        raise ValueError(f"Länge {fmt_number(lon)} liegt außerhalb von −180…180°.")
    return float(lat), float(lon)


def parse_coord(text: Any) -> Point:
    """Liest eine Koordinate aus Text: ``"49.9576, 6.9294"``, ``"49,9576; 6,9294"``,
    ``"49°57'27\\"N 6°55'46\\"E"``, ``"49.9576N 6.9294E"`` oder ``"N49.9576 E6.9294"``."""
    if isinstance(text, (tuple, list)) and len(text) == 2:
        return _check(_num("Breite", text[0]), _num("Länge", text[1]))
    s = str(text or "").strip()
    if not s:
        raise ValueError("Leere Koordinate.")
    # Grad/Minuten/Sekunden
    if "°" in s or "º" in s:
        parts = [m for m in _DMS.finditer(s) if m.group("deg")]
        if len(parts) != 2:
            raise ValueError(f"Koordinate nicht lesbar: {s!r}")
        vals = []
        for m in parts:
            v = _f(m.group("deg")) + (_f(m.group("min")) / 60.0 if m.group("min") else 0.0) \
                + (_f(m.group("sec")) / 3600.0 if m.group("sec") else 0.0)
            hemi = (m.group("hemi") or "").upper()
            if hemi in ("S", "W"):
                v = -v
            vals.append((v, hemi))
        (a, ha), (b, hb) = vals
        if ha in ("E", "W", "O") or hb in ("N", "S"):
            a, b = b, a
        return _check(a, b)
    # Dezimal: Semikolon/Leerzeichen trennen (Dezimalkomma erlaubt); sonst trennt das Komma
    if ";" in s:
        chunks = [c for c in re.split(r"[;\s]+", s) if c]
    elif "," in s:
        if re.search(r"\s", s.strip()) and s.count(",") == 2:
            chunks = [c for c in re.split(r"\s+", s.strip()) if c]
        else:
            chunks = [c.strip() for c in s.split(",") if c.strip()]
    else:
        chunks = s.split()
    if len(chunks) != 2:
        raise ValueError(f"Koordinate nicht lesbar: {s!r} (erwartet »Breite, Länge«).")
    vals = []
    for c in chunks:
        m = _DEC.fullmatch(c.strip())
        if not m:
            raise ValueError(f"Koordinate nicht lesbar: {c!r}")
        v = _f(m.group("num"))
        hemi = (m.group("pre") or m.group("post") or "").upper()
        if hemi in ("S", "W"):
            v = -abs(v)
        vals.append((v, hemi))
    (a, ha), (b, hb) = vals
    if ha in ("E", "W", "O") or hb in ("N", "S"):
        a, b = b, a
    return _check(a, b)


def format_coord(lat: float, lon: float, dms: bool = False) -> str:
    """``"49.9576° N, 6.9294° E"`` oder als Grad/Minuten/Sekunden."""
    lat, lon = _check(lat, lon)
    if not dms:
        return f"{abs(lat):.5f}° {'N' if lat >= 0 else 'S'}, {abs(lon):.5f}° {'E' if lon >= 0 else 'W'}"

    def one(v: float, pos: str, neg: str) -> str:
        h = pos if v >= 0 else neg
        v = abs(v)
        d = int(v)
        m = int((v - d) * 60)
        s = (v - d - m / 60.0) * 3600.0
        return f"{d}°{m:02d}'{s:04.1f}\"{h}"

    return f"{one(lat, 'N', 'S')} {one(lon, 'E', 'W')}"


def parse_points(text: Any) -> list[Point]:
    """``"lat,lon; lat,lon; …"`` (oder Liste von Paaren) → Liste von Koordinaten."""
    if isinstance(text, (list, tuple)):
        pts = [parse_coord(p) for p in text]
    else:
        s = str(text or "").strip()
        chunks = [c for c in re.split(r"\s*;\s*|\s*\|\s*|\n", s) if c.strip()]
        pts = [parse_coord(c) for c in chunks]
    if len(pts) > MAX_ROUTE_POINTS:
        raise ValueError(f"Höchstens {MAX_ROUTE_POINTS} Punkte.")
    return pts


# ------------------------------------------------------------------ Geodäsie
def distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Großkreis-Entfernung (Haversine) in Metern."""
    _check(lat1, lon1)
    _check(lat2, lon2)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = p2 - p1
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Anfangskurs von Punkt 1 nach Punkt 2 in Grad (0…360)."""
    _check(lat1, lon1)
    _check(lat2, lon2)
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    x = math.sin(dl) * math.cos(p2)
    y = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def destination(lat: float, lon: float, bearing: float, dist_m: float) -> Point:
    """Zielpunkt nach ``dist_m`` Metern in Richtung ``bearing`` (Grad)."""
    _check(lat, lon)
    d = float(dist_m) / EARTH_RADIUS_M
    b = math.radians(bearing)
    p1 = math.radians(lat)
    l1 = math.radians(lon)
    p2 = math.asin(math.sin(p1) * math.cos(d) + math.cos(p1) * math.sin(d) * math.cos(b))
    l2 = l1 + math.atan2(math.sin(b) * math.sin(d) * math.cos(p1), math.cos(d) - math.sin(p1) * math.sin(p2))
    lon2 = (math.degrees(l2) + 540.0) % 360.0 - 180.0
    return math.degrees(p2), lon2


def route_length_m(points: Sequence[Point]) -> float:
    pts = [parse_coord(p) for p in points]
    return sum(distance_m(*a, *b) for a, b in zip(pts, pts[1:]))


def polygon_area_m2(points: Sequence[Point]) -> float:
    """Fläche eines Polygons (Schnürformel in lokaler äquirektangularer Projektion um den Schwerpunkt)."""
    pts = [parse_coord(p) for p in points]
    if len(pts) < 3:
        raise ValueError("Ein Polygon braucht mindestens 3 Punkte.")
    lat0 = sum(p[0] for p in pts) / len(pts)
    lon0 = sum(p[1] for p in pts) / len(pts)
    k = math.cos(math.radians(lat0))
    xy = [(math.radians(lon - lon0) * k * EARTH_RADIUS_M, math.radians(lat - lat0) * EARTH_RADIUS_M) for lat, lon in pts]
    area = 0.0
    for (x1, y1), (x2, y2) in zip(xy, xy[1:] + xy[:1]):
        area += x1 * y2 - x2 * y1
    return abs(area) / 2.0


def flight_plan(points: Sequence[Point], speed_m_s: float, *, wind_speed_m_s: float = 0.0,
                wind_from_deg: float = 0.0, hover_s_per_point: float = 0.0) -> dict:
    """Flugplan entlang von Wegpunkten: je Abschnitt Distanz, Kurs, Bodengeschwindigkeit und Zeit.
    Der Wind weht AUS ``wind_from_deg``; seine Komponente entlang des Kurses verändert die
    Bodengeschwindigkeit (Rückenwind bei Kurs = wind_from + 180°)."""
    pts = [parse_coord(p) for p in points]
    if len(pts) < 2:
        raise ValueError("Ein Flugplan braucht mindestens 2 Punkte.")
    v = _pos("Fluggeschwindigkeit (m/s)", speed_m_s)
    w = _num("Windgeschwindigkeit (m/s)", wind_speed_m_s)
    if w < 0:
        raise ValueError("Windgeschwindigkeit darf nicht negativ sein.")
    w_from = float(wind_from_deg) % 360.0
    hover = _num("Schwebezeit je Punkt (s)", hover_s_per_point)
    if hover < 0:
        raise ValueError("Schwebezeit darf nicht negativ sein.")
    wind_to = (w_from + 180.0) % 360.0
    legs = []
    total_d = 0.0
    total_t = 0.0
    warnings: list[str] = []
    for i, (a, b) in enumerate(zip(pts, pts[1:])):
        d = distance_m(*a, *b)
        course = bearing_deg(*a, *b)
        tail = w * math.cos(math.radians(course - wind_to))     # Rückenwind positiv
        ground = v + tail
        flyable = ground > MIN_GROUND_SPEED
        t = d / ground if flyable else None
        if not flyable:
            warnings.append(f"Abschnitt {i + 1}: Gegenwind zu stark (Bodengeschwindigkeit "
                            f"{fmt_number(ground, 3)} m/s) – nicht fliegbar.")
        legs.append({"von": [_round(a[0], 6), _round(a[1], 6)], "nach": [_round(b[0], 6), _round(b[1], 6)],
                     "distanz_m": _round(d, 1), "kurs_deg": _round(course, 1),
                     "bodengeschwindigkeit_m_s": _round(ground, 2), "zeit_s": _round(t, 1) if t is not None else None,
                     "fliegbar": flyable})
        total_d += d
        if t is not None:
            total_t += t
    total_t += hover * len(pts)
    if w >= v:
        warnings.append("Wind erreicht oder übersteigt die Fluggeschwindigkeit – Rückweg gegen den Wind unmöglich.")
    return {"abschnitte": legs, "strecke_m": _round(total_d, 1), "zeit_s": _round(total_t, 1),
            "zeit_min": _round(total_t / 60.0, 2), "punkte": len(pts), "warnungen": warnings,
            "fliegbar": all(l["fliegbar"] for l in legs)}


# ------------------------------------------------------------------ Sonne (NOAA)
def _julian_day(dt: datetime) -> float:
    y, m = dt.year, dt.month
    d = dt.day + (dt.hour + dt.minute / 60.0 + dt.second / 3600.0) / 24.0
    if m <= 2:
        y -= 1
        m += 12
    a = y // 100
    b = 2 - a + a // 4
    return int(365.25 * (y + 4716)) + int(30.6001 * (m + 1)) + d + b - 1524.5


def _solar_params(t: float) -> tuple[float, float]:
    """Deklination (Grad) und Zeitgleichung (Minuten) für das Jahrhundert ``t`` seit J2000."""
    l0 = (280.46646 + t * (36000.76983 + 0.0003032 * t)) % 360.0
    m = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    e = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    mr = math.radians(m)
    c = (math.sin(mr) * (1.914602 - t * (0.004817 + 0.000014 * t)) + math.sin(2 * mr) * (0.019993 - 0.000101 * t)
         + math.sin(3 * mr) * 0.000289)
    true_long = l0 + c
    omega = 125.04 - 1934.136 * t
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(omega))
    eps0 = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    eps = eps0 + 0.00256 * math.cos(math.radians(omega))
    decl = math.degrees(math.asin(math.sin(math.radians(eps)) * math.sin(math.radians(app_long))))
    y = math.tan(math.radians(eps / 2.0)) ** 2
    l0r = math.radians(l0)
    eot = 4.0 * math.degrees(y * math.sin(2 * l0r) - 2 * e * math.sin(mr) + 4 * e * y * math.sin(mr) * math.cos(2 * l0r)
                             - 0.5 * y * y * math.sin(4 * l0r) - 1.25 * e * e * math.sin(2 * mr))
    return decl, eot


def _as_utc(when: datetime | date | None) -> datetime:
    if when is None:
        return datetime.now(timezone.utc)
    if isinstance(when, datetime):
        return when.astimezone(timezone.utc) if when.tzinfo else when.replace(tzinfo=timezone.utc)
    return datetime(when.year, when.month, when.day, 12, tzinfo=timezone.utc)


def sun(lat: float, lon: float, when: datetime | date | None = None) -> dict:
    """Sonnenaufgang/-untergang (UTC), Tageslänge, Sonnenhöhe und Azimut zum Zeitpunkt ``when``
    nach dem NOAA-Algorithmus (Genauigkeit wenige Minuten). Ist ``when`` zeitzonenbehaftet, werden
    die Zeiten zusätzlich lokal ausgegeben."""
    lat, lon = _check(lat, lon)
    tz = when.tzinfo if isinstance(when, datetime) and when.tzinfo else None
    if tz is not None and when.utcoffset() == timedelta(0):
        tz = None                                   # UTC braucht keine zweite Ausgabe
    dt = _as_utc(when)
    day0 = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    # Deklination/Zeitgleichung für den Mittag des Tages (Auf-/Untergang) und für den Zeitpunkt (Position)
    t_noon = (_julian_day(day0 + timedelta(hours=12)) - 2451545.0) / 36525.0
    decl, eot = _solar_params(t_noon)
    noon_min = 720.0 - 4.0 * lon - eot
    phi = math.radians(lat)
    dr = math.radians(decl)
    cos_ha = (math.cos(math.radians(90.833)) / (math.cos(phi) * math.cos(dr))) - math.tan(phi) * math.tan(dr)
    hint = None
    rise = sunset = None
    if cos_ha > 1.0:
        hint = "Polarnacht: Die Sonne geht an diesem Tag nicht auf."
        day_len = 0.0
    elif cos_ha < -1.0:
        hint = "Polartag: Die Sonne geht an diesem Tag nicht unter."
        day_len = 24.0
    else:
        ha = math.degrees(math.acos(cos_ha))
        rise = day0 + timedelta(minutes=noon_min - 4.0 * ha)
        sunset = day0 + timedelta(minutes=noon_min + 4.0 * ha)
        day_len = 8.0 * ha / 60.0
    noon = day0 + timedelta(minutes=noon_min)
    # Position zum Zeitpunkt
    t_now = (_julian_day(dt) - 2451545.0) / 36525.0
    decl_now, eot_now = _solar_params(t_now)
    minutes = dt.hour * 60 + dt.minute + dt.second / 60.0
    tst = (minutes + eot_now + 4.0 * lon) % 1440.0
    ha_now = tst / 4.0 - 180.0
    drn = math.radians(decl_now)
    cos_zen = math.sin(phi) * math.sin(drn) + math.cos(phi) * math.cos(drn) * math.cos(math.radians(ha_now))
    zen = math.degrees(math.acos(max(-1.0, min(1.0, cos_zen))))
    elev = 90.0 - zen
    sz = math.sin(math.radians(zen))
    if sz > 1e-9 and abs(math.cos(phi)) > 1e-9:
        az_cos = max(-1.0, min(1.0, ((math.sin(phi) * math.cos(math.radians(zen))) - math.sin(drn)) / (math.cos(phi) * sz)))
        az = math.degrees(math.acos(az_cos))
        azimuth = (az + 180.0) % 360.0 if ha_now > 0 else (540.0 - az) % 360.0
    else:
        azimuth = 0.0

    def iso(d: datetime | None) -> str | None:
        return d.replace(microsecond=0).isoformat() if d else None

    out: dict[str, Any] = {
        "breite": lat, "laenge": lon, "datum": day0.date().isoformat(), "zeit_utc": iso(dt),
        "aufgang_utc": iso(rise), "untergang_utc": iso(sunset), "mittag_utc": iso(noon),
        "tageslaenge_h": _round(day_len, 2), "hoehe_deg": _round(elev, 2), "azimut_deg": _round(azimuth, 2),
        "deklination_deg": _round(decl_now, 3), "zeitgleichung_min": _round(eot_now, 2),
        "ueber_horizont": elev > -0.833, "hinweis": hint,
    }
    if tz is not None:
        out["zeitzone"] = str(tz)
        out["aufgang_lokal"] = iso(rise.astimezone(tz)) if rise else None
        out["untergang_lokal"] = iso(sunset.astimezone(tz)) if sunset else None
        out["mittag_lokal"] = iso(noon.astimezone(tz))
    return out


def subsolar_point(when: datetime | None = None) -> Point:
    """Punkt, über dem die Sonne im Zenit steht (Breite = Deklination)."""
    dt = _as_utc(when)
    t = (_julian_day(dt) - 2451545.0) / 36525.0
    decl, eot = _solar_params(t)
    minutes = dt.hour * 60 + dt.minute + dt.second / 60.0
    lon = -((minutes + eot) / 4.0 - 180.0)
    lon = (lon + 540.0) % 360.0 - 180.0
    return decl, lon


def terminator(when: datetime | None = None, points: int = 72) -> list[Point]:
    """Tag-Nacht-Grenze als Punktliste (Breite, Länge), 90° vom Subsolarpunkt entfernt."""
    n = max(8, min(720, int(points)))
    slat, slon = subsolar_point(when)
    quarter = EARTH_RADIUS_M * math.pi / 2.0
    return [tuple(_round(v, 4) for v in destination(slat, slon, 360.0 * i / n, quarter)) for i in range(n)]  # type: ignore[misc]


# ------------------------------------------------------------------ UTM
_A = 6378137.0
_F = 1 / 298.257223563
_E2 = _F * (2 - _F)
_EP2 = _E2 / (1 - _E2)
_K0 = 0.9996
_BANDS = "CDEFGHJKLMNPQRSTUVWX"


def utm_zone(lat: float, lon: float) -> tuple[int, str]:
    lat, lon = _check(lat, lon)
    zone = int((lon + 180.0) // 6.0) + 1
    if lon >= 180.0:
        zone = 60
    # Ausnahmen Norwegen/Svalbard
    if 56.0 <= lat < 64.0 and 3.0 <= lon < 12.0:
        zone = 32
    if 72.0 <= lat < 84.0:
        if 0.0 <= lon < 9.0:
            zone = 31
        elif 9.0 <= lon < 21.0:
            zone = 33
        elif 21.0 <= lon < 33.0:
            zone = 35
        elif 33.0 <= lon < 42.0:
            zone = 37
    if lat < -80.0 or lat >= 84.0:
        raise ValueError("UTM gilt nur zwischen 80° S und 84° N (sonst UPS).")
    band = _BANDS[min(19, int((lat + 80.0) // 8.0))]
    return zone, band


def utm(lat: float, lon: float) -> dict:
    """UTM-Koordinaten (WGS84, Transverse-Mercator-Reihe nach Snyder, Genauigkeit < 1 m)."""
    lat, lon = _check(lat, lon)
    zone, band = utm_zone(lat, lon)
    lon0 = (zone - 1) * 6 - 180 + 3
    phi = math.radians(lat)
    n = _A / math.sqrt(1 - _E2 * math.sin(phi) ** 2)
    t = math.tan(phi) ** 2
    c = _EP2 * math.cos(phi) ** 2
    a = math.radians(lon - lon0) * math.cos(phi)
    e4, e6 = _E2 ** 2, _E2 ** 3
    m = _A * ((1 - _E2 / 4 - 3 * e4 / 64 - 5 * e6 / 256) * phi
              - (3 * _E2 / 8 + 3 * e4 / 32 + 45 * e6 / 1024) * math.sin(2 * phi)
              + (15 * e4 / 256 + 45 * e6 / 1024) * math.sin(4 * phi)
              - (35 * e6 / 3072) * math.sin(6 * phi))
    east = _K0 * n * (a + (1 - t + c) * a ** 3 / 6 + (5 - 18 * t + t * t + 72 * c - 58 * _EP2) * a ** 5 / 120) + 500000.0
    north = _K0 * (m + n * math.tan(phi) * (a * a / 2 + (5 - t + 9 * c + 4 * c * c) * a ** 4 / 24
                                             + (61 - 58 * t + t * t + 600 * c - 330 * _EP2) * a ** 6 / 720))
    south = lat < 0
    if south:
        north += 10_000_000.0
    return {"zone": zone, "band": band, "ostwert_m": _round(east, 2), "nordwert_m": _round(north, 2),
            "hemisphaere": "S" if south else "N", "text": f"{zone}{band} {east:.0f} {north:.0f}"}


# ------------------------------------------------------------------ Store
_SCHEMA = """
CREATE TABLE IF NOT EXISTS orte (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, lat REAL NOT NULL, lon REAL NOT NULL,
    project TEXT, note TEXT NOT NULL DEFAULT '', created_at REAL NOT NULL);
CREATE INDEX IF NOT EXISTS orte_name ON orte(name);
CREATE TABLE IF NOT EXISTS routen (
    id INTEGER PRIMARY KEY AUTOINCREMENT, name TEXT NOT NULL, points TEXT NOT NULL, project TEXT,
    length_m REAL NOT NULL, created_at REAL NOT NULL);
"""


def _clean_name(name: Any) -> str:
    s = " ".join(str(name or "").split())[:MAX_NAME_CHARS]
    if not s:
        raise ValueError("Name darf nicht leer sein.")
    return s


class WaypointStore:
    """Orte und Routen je Projekt in ``geo.db``."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn: sqlite3.Connection | None = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
            except sqlite3.DatabaseError:
                pass
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            raise RuntimeError("WaypointStore ist geschlossen.")
        return self._conn

    @staticmethod
    def _place(row: sqlite3.Row) -> dict:
        return {"id": row["id"], "name": row["name"], "lat": row["lat"], "lon": row["lon"], "projekt": row["project"],
                "notiz": row["note"], "erstellt": row["created_at"], "koordinate": format_coord(row["lat"], row["lon"])}

    @staticmethod
    def _route(row: sqlite3.Row) -> dict:
        return {"id": row["id"], "name": row["name"], "punkte": json.loads(row["points"]), "projekt": row["project"],
                "laenge_m": row["length_m"], "erstellt": row["created_at"]}

    def add_place(self, name: str, lat: float, lon: float, *, project: str | None = None, note: str = "") -> dict:
        name = _clean_name(name)
        lat, lon = _check(_num("Breite", lat), _num("Länge", lon))
        with self._lock:
            cur = self.conn.execute("INSERT INTO orte(name, lat, lon, project, note, created_at) VALUES (?,?,?,?,?,?)",
                                    (name, lat, lon, project or None, str(note or "")[:2000], time.time()))
            self.conn.commit()
            return self.get_place(int(cur.lastrowid))  # type: ignore[return-value]

    def get_place(self, place_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM orte WHERE id=?", (int(place_id),)).fetchone()
        return self._place(row) if row else None

    def find_place(self, name: str, project: str | None = None) -> dict | None:
        """Ort nach Name (ohne Groß-/Kleinschreibung); Projektorte vor allgemeinen."""
        key = " ".join(str(name or "").split()).lower()
        if not key:
            return None
        with self._lock:
            rows = self.conn.execute("SELECT * FROM orte WHERE lower(name)=? ORDER BY (project IS ?) DESC, id DESC",
                                     (key, project or None)).fetchall()
        return self._place(rows[0]) if rows else None

    def list_places(self, project: str | None = None, limit: int = 500) -> list[dict]:
        with self._lock:
            if project:
                rows = self.conn.execute("SELECT * FROM orte WHERE project=? ORDER BY name LIMIT ?", (project, limit)).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM orte ORDER BY name LIMIT ?", (limit,)).fetchall()
        return [self._place(r) for r in rows]

    def delete_place(self, place_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM orte WHERE id=?", (int(place_id),))
            self.conn.commit()
            return cur.rowcount > 0

    def add_route(self, name: str, points: Sequence[Point] | str, *, project: str | None = None) -> dict:
        name = _clean_name(name)
        pts = parse_points(points)
        if len(pts) < 2:
            raise ValueError("Eine Route braucht mindestens 2 Punkte.")
        length = route_length_m(pts)
        with self._lock:
            cur = self.conn.execute("INSERT INTO routen(name, points, project, length_m, created_at) VALUES (?,?,?,?,?)",
                                    (name, json.dumps([[_round(a, 6), _round(b, 6)] for a, b in pts]),
                                     project or None, _round(length, 1), time.time()))
            self.conn.commit()
            return self.get_route(int(cur.lastrowid))  # type: ignore[return-value]

    def get_route(self, route_id: int) -> dict | None:
        with self._lock:
            row = self.conn.execute("SELECT * FROM routen WHERE id=?", (int(route_id),)).fetchone()
        return self._route(row) if row else None

    def list_routes(self, project: str | None = None, limit: int = 500) -> list[dict]:
        with self._lock:
            if project:
                rows = self.conn.execute("SELECT * FROM routen WHERE project=? ORDER BY id DESC LIMIT ?", (project, limit)).fetchall()
            else:
                rows = self.conn.execute("SELECT * FROM routen ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [self._route(r) for r in rows]

    def delete_route(self, route_id: int) -> bool:
        with self._lock:
            cur = self.conn.execute("DELETE FROM routen WHERE id=?", (int(route_id),))
            self.conn.commit()
            return cur.rowcount > 0

    def stats(self) -> dict:
        with self._lock:
            places = self.conn.execute("SELECT COUNT(*) FROM orte").fetchone()[0]
            routes = self.conn.execute("SELECT COUNT(*) FROM routen").fetchone()[0]
            length = self.conn.execute("SELECT COALESCE(SUM(length_m), 0) FROM routen").fetchone()[0]
        return {"orte": int(places), "routen": int(routes), "routen_laenge_m": _round(float(length), 1)}

    def close(self) -> None:
        with self._lock:
            if self._conn is not None:
                try:
                    self._conn.close()
                finally:
                    self._conn = None


# ------------------------------------------------------------------ Wetter (online, optional)
_WMO = {
    0: "klar", 1: "überwiegend klar", 2: "teils bewölkt", 3: "bedeckt", 45: "Nebel", 48: "Reifnebel",
    51: "leichter Nieselregen", 53: "Nieselregen", 55: "starker Nieselregen", 56: "gefrierender Nieselregen",
    57: "starker gefrierender Nieselregen", 61: "leichter Regen", 63: "Regen", 65: "starker Regen",
    66: "gefrierender Regen", 67: "starker gefrierender Regen", 71: "leichter Schneefall", 73: "Schneefall",
    75: "starker Schneefall", 77: "Schneegriesel", 80: "leichte Regenschauer", 81: "Regenschauer",
    82: "heftige Regenschauer", 85: "Schneeschauer", 86: "starke Schneeschauer", 95: "Gewitter",
    96: "Gewitter mit Hagel", 99: "Gewitter mit starkem Hagel",
}


def weather_text(code: int | None) -> str:
    if code is None:
        return "unbekannt"
    return _WMO.get(int(code), f"Wettercode {code}")


def flight_conditions(wind_kmh: float | None, gust_kmh: float | None, rain_mm: float | None,
                      code: int | None) -> dict:
    """Flugtauglichkeit nach einfachen Grenzwerten (Wind, Böen, Niederschlag, Gewitter)."""
    reasons = []
    if wind_kmh is not None and wind_kmh > WIND_LIMIT_KMH:
        reasons.append(f"Wind {fmt_number(wind_kmh, 3)} km/h über {fmt_number(WIND_LIMIT_KMH)} km/h")
    if gust_kmh is not None and gust_kmh > GUST_LIMIT_KMH:
        reasons.append(f"Böen {fmt_number(gust_kmh, 3)} km/h über {fmt_number(GUST_LIMIT_KMH)} km/h")
    if rain_mm is not None and rain_mm > RAIN_LIMIT_MM:
        reasons.append(f"Niederschlag {fmt_number(rain_mm, 3)} mm")
    if code is not None and 95 <= int(code) <= 99:
        reasons.append("Gewitter")
    return {"flugtauglich": not reasons, "gruende": reasons}


def weather(lat: float, lon: float, *, base_url: str = "https://api.open-meteo.com", timeout: float = 10.0,
            opener: Any = None) -> dict:
    """Aktuelles Wetter von Open-Meteo (Internet nötig). Löst ``RuntimeError`` bei Netz-/Datenfehlern aus."""
    lat, lon = _check(lat, lon)
    query = urllib.parse.urlencode({
        "latitude": f"{lat:.4f}", "longitude": f"{lon:.4f}",
        "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,wind_direction_10m,wind_gusts_10m,"
                   "precipitation,cloud_cover,weather_code,pressure_msl",
        "wind_speed_unit": "kmh", "timezone": "UTC",
    })
    url = f"{base_url.rstrip('/')}/v1/forecast?{query}"
    op = opener or urllib.request.build_opener()
    req = urllib.request.Request(url, headers={"User-Agent": "OBITO/4.0 (lokale Engineering-KI)"})
    try:
        with op.open(req, timeout=timeout) as resp:
            raw = resp.read(1_000_000)
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"Wetterdienst antwortet mit HTTP {e.code}.") from None
    except (urllib.error.URLError, OSError, TimeoutError) as e:
        raise RuntimeError(f"Wetterdienst nicht erreichbar: {getattr(e, 'reason', e)}") from None
    try:
        data = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        raise RuntimeError("Wetterdienst lieferte keine gültigen Daten.") from None
    cur = data.get("current") if isinstance(data, dict) else None
    if not isinstance(cur, dict):
        raise RuntimeError("Wetterdienst lieferte keine aktuellen Werte.")

    def num(key: str) -> float | None:
        v = cur.get(key)
        return float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else None

    code = cur.get("weather_code")
    code_i = int(code) if isinstance(code, (int, float)) else None
    wind, gust, rain = num("wind_speed_10m"), num("wind_gusts_10m"), num("precipitation")
    out = {
        "breite": lat, "laenge": lon, "zeit": cur.get("time"), "temperatur_c": num("temperature_2m"),
        "luftfeuchte_prozent": num("relative_humidity_2m"), "wind_kmh": wind,
        "wind_richtung_deg": num("wind_direction_10m"), "boeen_kmh": gust, "niederschlag_mm": rain,
        "bewoelkung_prozent": num("cloud_cover"), "luftdruck_hpa": num("pressure_msl"),
        "wettercode": code_i, "wetter": weather_text(code_i), "quelle": WEATHER_SOURCE,
    }
    out.update(flight_conditions(wind, gust, rain, code_i))
    return out


def format_weather(w: dict) -> str:
    def f(v: Any, unit: str = "") -> str:
        return "–" if v is None else f"{fmt_number(float(v), 3)}{unit}"
    lines = [f"Wetter bei {format_coord(w['breite'], w['laenge'])} ({w.get('zeit') or '?'} UTC): {w.get('wetter')}",
             f"  Temperatur {f(w.get('temperatur_c'), ' °C')}, Luftfeuchte {f(w.get('luftfeuchte_prozent'), ' %')}, "
             f"Luftdruck {f(w.get('luftdruck_hpa'), ' hPa')}",
             f"  Wind {f(w.get('wind_kmh'), ' km/h')} aus {f(w.get('wind_richtung_deg'), '°')}, Böen "
             f"{f(w.get('boeen_kmh'), ' km/h')}, Niederschlag {f(w.get('niederschlag_mm'), ' mm')}, Bewölkung "
             f"{f(w.get('bewoelkung_prozent'), ' %')}",
             "  Flugtauglich: " + ("ja" if w.get("flugtauglich") else "nein – " + "; ".join(w.get("gruende") or [])),
             f"  Quelle: {w.get('quelle')}"]
    return "\n".join(lines)


# ------------------------------------------------------------------ Werkzeuge
TOOL_NAMES = ("geo_distanz", "geo_route", "sonnenstand", "wetter")


def _resolve(store: WaypointStore | None, text: Any, project: str | None = None) -> tuple[Point, str]:
    """Koordinate aus Text oder gespeichertem Ortsnamen; liefert (Punkt, Anzeigename)."""
    s = str(text or "").strip()
    try:
        return parse_coord(s), s
    except ValueError:
        pass
    if store is not None:
        place = store.find_place(s, project)
        if place:
            return (place["lat"], place["lon"]), f"{place['name']} ({place['koordinate']})"
    raise ValueError(f"»{s}« ist weder eine Koordinate (Breite, Länge) noch ein gespeicherter Ort.")


def _parse_when(text: Any) -> datetime | None:
    s = str(text or "").strip()
    if not s:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S%z", "%Y-%m-%dT%H:%M%z", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M",
                "%Y-%m-%d", "%d.%m.%Y %H:%M", "%d.%m.%Y"):
        try:
            dt = datetime.strptime(s.replace("Z", "+0000"), fmt)
        except ValueError:
            continue
        if fmt in ("%Y-%m-%d", "%d.%m.%Y"):
            dt = dt.replace(hour=12)
        return dt
    raise ValueError(f"Datum nicht lesbar: {s!r} (z. B. 2026-06-21 oder 21.06.2026 14:30).")


def format_sun(info: dict) -> str:
    def hm(iso: str | None) -> str:
        return iso[11:16] if iso else "–"
    lines = [f"Sonne bei {format_coord(info['breite'], info['laenge'])} am {info['datum']}:"]
    if info.get("hinweis"):
        lines.append(f"  {info['hinweis']}")
    lines.append(f"  Aufgang {hm(info.get('aufgang_utc'))} UTC, Mittag {hm(info.get('mittag_utc'))} UTC, "
                 f"Untergang {hm(info.get('untergang_utc'))} UTC, Tageslänge {fmt_number(info['tageslaenge_h'], 3)} h")
    if info.get("aufgang_lokal") or info.get("untergang_lokal"):
        lines.append(f"  Lokal ({info.get('zeitzone')}): Aufgang {hm(info.get('aufgang_lokal'))}, "
                     f"Untergang {hm(info.get('untergang_lokal'))}")
    lines.append(f"  Um {hm(info.get('zeit_utc'))} UTC: Höhe {fmt_number(info['hoehe_deg'], 3)}°, "
                 f"Azimut {fmt_number(info['azimut_deg'], 4)}° ({'über' if info['ueber_horizont'] else 'unter'} dem Horizont)")
    return "\n".join(lines)


def register_tools(registry: "ToolRegistry", store: WaypointStore | None = None,
                   online: Callable[[], bool] | None = None, *, weather_fn: Callable[..., dict] = weather) -> None:
    """Registriert ``geo_distanz``, ``geo_route``, ``sonnenstand`` und ``wetter`` (nicht gefährlich).
    ``online()`` entscheidet zur Laufzeit, ob das Wetter abgefragt werden darf."""
    from .tools import Tool

    is_online = online or (lambda: False)

    def geo_distanz(von: str, nach: str) -> str:
        (a, name_a), (b, name_b) = _resolve(store, von), _resolve(store, nach)
        d = distance_m(*a, *b)
        brg = bearing_deg(*a, *b)
        u = utm(*a)
        return (f"Entfernung {name_a} → {name_b}: {fmt_number(d / 1000.0, 4)} km ({fmt_number(d, 6)} m), "
                f"Anfangskurs {fmt_number(brg, 4)}°, Rückkurs {fmt_number(bearing_deg(*b, *a), 4)}°.\n"
                f"Start in UTM: {u['text']}")

    def geo_route(punkte: str, geschwindigkeit_m_s: float, wind_kmh: float = 0.0, wind_aus_deg: float = 0.0) -> str:
        chunks = [c for c in re.split(r"\s*;\s*|\n", str(punkte or "")) if c.strip()]
        pts = [_resolve(store, c)[0] for c in chunks]
        plan = flight_plan(pts, geschwindigkeit_m_s, wind_speed_m_s=float(wind_kmh) / 3.6, wind_from_deg=wind_aus_deg)
        lines = [f"Flugplan über {plan['punkte']} Punkte: {fmt_number(plan['strecke_m'] / 1000.0, 4)} km, "
                 f"{fmt_number(plan['zeit_min'], 3)} min bei {fmt_number(float(geschwindigkeit_m_s), 3)} m/s"
                 + (f", Wind {fmt_number(float(wind_kmh), 3)} km/h aus {fmt_number(float(wind_aus_deg), 3)}°" if float(wind_kmh) else "")]
        for i, leg in enumerate(plan["abschnitte"], 1):
            lines.append(f"  {i}. {fmt_number(leg['distanz_m'], 5)} m, Kurs {fmt_number(leg['kurs_deg'], 4)}°, "
                         f"über Grund {fmt_number(leg['bodengeschwindigkeit_m_s'], 3)} m/s, "
                         + (f"{fmt_number(leg['zeit_s'], 4)} s" if leg["zeit_s"] is not None else "nicht fliegbar"))
        for w in plan["warnungen"]:
            lines.append(f"Warnung: {w}")
        return "\n".join(lines)

    def sonnenstand(ort: str, datum: str | None = None) -> str:
        (p, name) = _resolve(store, ort)
        info = sun(p[0], p[1], _parse_when(datum))
        return format_sun(info).replace(format_coord(p[0], p[1]), name, 1)

    def wetter(ort: str) -> str:
        (p, name) = _resolve(store, ort)
        if not is_online():
            return ("Online-Funktionen sind ausgeschaltet (Einstellung »online« auf true setzen, z. B. "
                    "python -m obito config --setzen online=true). Ohne Internet gibt es keine Wetterdaten.")
        try:
            w = weather_fn(p[0], p[1])
        except RuntimeError as e:
            return f"Wetter für {name} nicht abrufbar: {e}"
        return format_weather(w).replace(format_coord(p[0], p[1]), name, 1)

    registry.register(Tool(
        name="geo_distanz",
        description="Entfernung (Großkreis), Anfangskurs und UTM-Zone zwischen zwei Punkten – Koordinaten "
                    "(»49.95, 6.93« oder 49°57'N 6°55'E) oder gespeicherte Ortsnamen.",
        parameters=_schema({"von": _p_str("Startpunkt", "52.52, 13.405"), "nach": _p_str("Zielpunkt", "48.137, 11.575")},
                           ["von", "nach"]),
        fn=geo_distanz,
    ))
    registry.register(Tool(
        name="geo_route",
        description="Flugplan über Wegpunkte (»lat,lon; lat,lon; …« oder Ortsnamen, mit Semikolon getrennt): "
                    "Distanz, Kurs, Bodengeschwindigkeit und Zeit je Abschnitt, Wind optional.",
        parameters=_schema({"punkte": _p_str("Wegpunkte, mit Semikolon getrennt", "52.52,13.405; 52.53,13.41"),
                            "geschwindigkeit_m_s": _p_number("Fluggeschwindigkeit in m/s", 12),
                            "wind_kmh": _p_number("Windgeschwindigkeit in km/h", default=0),
                            "wind_aus_deg": _p_number("Windrichtung (woher) in Grad", default=0)},
                           ["punkte", "geschwindigkeit_m_s"]),
        fn=geo_route,
    ))
    registry.register(Tool(
        name="sonnenstand",
        description="Sonnenaufgang, -untergang, Tageslänge sowie Höhe/Azimut der Sonne für einen Ort und "
                    "Zeitpunkt (Standard: jetzt).",
        parameters=_schema({"ort": _p_str("Koordinate oder gespeicherter Ort", "52.52, 13.405"),
                            "datum": _p_str("Datum/Zeit (ISO oder 21.06.2026 14:30, optional)")}, ["ort"]),
        fn=sonnenstand,
    ))
    registry.register(Tool(
        name="wetter",
        description="Aktuelles Wetter und Flugtauglichkeit (Wind, Böen, Regen, Gewitter) für einen Ort – nur wenn "
                    "Online-Funktionen eingeschaltet sind (Open-Meteo).",
        parameters=_schema({"ort": _p_str("Koordinate oder gespeicherter Ort", "52.52, 13.405")}, ["ort"]),
        fn=wetter,
    ))


__all__ = [
    "EARTH_RADIUS_M", "TOOL_NAMES", "WaypointStore", "bearing_deg", "destination", "distance_m", "flight_conditions",
    "flight_plan", "format_coord", "format_sun", "format_weather", "parse_coord", "parse_points", "polygon_area_m2",
    "register_tools", "route_length_m", "subsolar_point", "sun", "terminator", "utm", "utm_zone", "weather",
    "weather_text",
]
