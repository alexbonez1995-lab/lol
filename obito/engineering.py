"""Ingenieur-Rechner und Materialdaten von OBITO.

Reine Funktionen mit SI-Einheiten (keine Datenbank, kein Modell, nur Standardbibliothek).
Jede Rechnung liefert ein ``dict`` mit Ergebniswerten (Einheit im Schlüssel, z. B.
``"spannung_v"``), einer lesbaren ``"formel"`` und einer Liste ``"annahmen"``.
Unsinnige Eingaben lösen ``ValueError`` mit deutschem Text aus, damit
:meth:`obito.tools.ToolRegistry.run` sie als Fehler meldet.

Alle Materialwerte sind **Richtwerte** (typische Werte je Werkstoffklasse) und als
solche gekennzeichnet – für die Auslegung gilt immer das Datenblatt des Herstellers.
"""

from __future__ import annotations

import difflib
import math
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfer
    from .tools import ToolRegistry

# ------------------------------------------------------------------ Konstanten
G = 9.81                      # Erdbeschleunigung in m/s²
CELL_NOMINAL_V = 3.7          # LiPo-Nennspannung je Zelle
CELL_FULL_V = 4.2             # LiPo voll geladen
CELL_STORAGE_V = 3.8          # LiPo Lagerspannung
CELL_EMPTY_V = 3.5            # empfohlener Entladeschluss unter Last
COPPER_RESISTIVITY = 0.0175   # Ω·mm²/m bei 20 °C
MAX_CELLS = 24
MAX_MOTORS = 16
MIN_AWG = -3                  # 4/0
MAX_AWG = 40
MAX_TOOL_OUTPUT_CHARS = 4000  # entspricht tools.MAX_OUTPUT_CHARS (Ausgabe wird sonst gekürzt)
RICHTWERT_HINWEIS = "Richtwerte: typische Werte je Werkstoffklasse, keine Garantie – Datenblatt prüfen."

CATEGORIES = ("metall", "kunststoff", "verbund", "holz", "sonstiges")
COST_CLASSES = ("günstig", "mittel", "teuer")
PRINTABLE = ("FDM", "SLA", None)

_CATEGORY_LABEL = {"metall": "Metall", "kunststoff": "Kunststoff", "verbund": "Verbund",
                   "holz": "Holz", "sonstiges": "Sonstiges"}


# ------------------------------------------------------------------ Hilfen
def _num(name: str, value: Any) -> float:
    """Wandelt ``value`` in ``float`` (auch aus Strings mit Dezimalkomma); sonst ``ValueError``."""
    if isinstance(value, bool):
        raise ValueError(f"{name} muss eine Zahl sein, nicht ja/nein.")
    if isinstance(value, (int, float)):
        f = float(value)
    elif isinstance(value, str):
        s = value.strip().replace(",", ".")
        try:
            f = float(s)
        except ValueError:
            raise ValueError(f"{name} muss eine Zahl sein, nicht {value!r}.") from None
    else:
        raise ValueError(f"{name} muss eine Zahl sein, nicht {value!r}.")
    if math.isnan(f) or math.isinf(f):
        raise ValueError(f"{name} muss eine endliche Zahl sein.")
    return f


def _pos(name: str, value: Any) -> float:
    """Zahl > 0, sonst ``ValueError``."""
    f = _num(name, value)
    if f <= 0:
        raise ValueError(f"{name} muss größer als 0 sein (erhalten: {fmt_number(f)}).")
    return f


def _nonneg(name: str, value: Any) -> float:
    """Zahl ≥ 0, sonst ``ValueError``."""
    f = _num(name, value)
    if f < 0:
        raise ValueError(f"{name} darf nicht negativ sein (erhalten: {fmt_number(f)}).")
    return f


def _int_range(name: str, value: Any, low: int, high: int) -> int:
    f = _num(name, value)
    if not f.is_integer():
        raise ValueError(f"{name} muss eine ganze Zahl sein (erhalten: {fmt_number(f)}).")
    n = int(f)
    if n < low or n > high:
        raise ValueError(f"{name} muss zwischen {low} und {high} liegen (erhalten: {n}).")
    return n


def fmt_number(value: Any, digits: int = 4) -> str:
    """Formatiert eine Zahl lesbar mit deutschem Dezimalkomma und ``digits`` signifikanten Stellen."""
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, int):
        return str(value)
    if not isinstance(value, float):
        return str(value)
    if math.isnan(value) or math.isinf(value):
        return str(value)
    if value == 0:
        return "0"
    a = abs(value)
    if a >= 10000:
        s = f"{value:.0f}"
    elif a >= 1000:
        s = f"{value:.1f}"
    else:
        s = f"{value:.{digits}g}"
    if "e" not in s and "." in s:
        s = s.rstrip("0").rstrip(".")
    return s.replace(".", ",")


def _round(value: float, ndigits: int = 4) -> float:
    """Rundet auf ``ndigits`` Nachkommastellen, lässt ganze Werte als float."""
    return float(round(value, ndigits))


# ------------------------------------------------------------------ Material
@dataclass(frozen=True)
class Material:
    """Werkstoff mit typischen Kennwerten (Richtwerte)."""

    key: str
    name: str
    category: str                      # "metall" | "kunststoff" | "verbund" | "holz" | "sonstiges"
    density: float                     # g/cm³ (typisch)
    tensile_mpa: tuple[float, float]   # Zugfestigkeit min/max in MPa
    youngs_gpa: float                  # E-Modul (typisch) in GPa
    max_temp_c: float | None           # Dauergebrauchstemperatur in °C
    cost: str                          # "günstig" | "mittel" | "teuer"
    printable: str | None              # "FDM" | "SLA" | None
    notes: str
    typical_use: str

    @property
    def specific_strength(self) -> float:
        """Zugfestigkeit (min) je Dichte in MPa/(g/cm³) – Leichtbau-Kennzahl."""
        return self.tensile_mpa[0] / self.density

    @property
    def specific_stiffness(self) -> float:
        """E-Modul je Dichte in GPa/(g/cm³)."""
        return self.youngs_gpa / self.density

    def to_dict(self) -> dict:
        return {
            "schluessel": self.key,
            "name": self.name,
            "kategorie": self.category,
            "dichte_g_cm3": self.density,
            "zugfestigkeit_mpa": [self.tensile_mpa[0], self.tensile_mpa[1]],
            "e_modul_gpa": self.youngs_gpa,
            "max_temp_c": self.max_temp_c,
            "kosten": self.cost,
            "druckbar": self.printable,
            "hinweise": self.notes,
            "typischer_einsatz": self.typical_use,
            "spezifische_festigkeit": _round(self.specific_strength, 1),
            "spezifische_steifigkeit": _round(self.specific_stiffness, 2),
            "richtwert": True,
        }


def _m(key: str, name: str, category: str, density: float, tensile: tuple[float, float],
       youngs: float, max_temp: float | None, cost: str, printable: str | None,
       notes: str, use: str) -> Material:
    return Material(key, name, category, density, tensile, youngs, max_temp, cost, printable, notes, use)


MATERIALS: dict[str, Material] = {m.key: m for m in (
    # ------------------------------------------------------------ Verbund
    _m("cfk", "CFK (Kohlefaser-Epoxid, quasi-isotrop)", "verbund", 1.55, (600.0, 1500.0), 70.0, 120.0,
       "teuer", None,
       "Anisotrop: unidirektional bis ~1500 MPa / 135 GPa, quasi-isotropes Laminat ~600 MPa / 70 GPa. "
       "Elektrisch leitfähig (Funk abschirmend), spröde beim Bruch, Staub beim Fräsen gesundheitsschädlich. "
       "Keine Kerben/scharfe Bohrkanten; Verbindungen kleben oder mit Hülsen.",
       "Drohnen-Arme und -Platten, Flugzeugholme, Fahrwerke, Leichtbau-Streben."),
    _m("gfk", "GFK (Glasfaser-Epoxid)", "verbund", 1.90, (200.0, 600.0), 25.0, 120.0,
       "mittel", None,
       "Günstiger und zäher als CFK, aber schwerer und deutlich weicher. Funktransparent (gut für Antennen). "
       "Gewebe lässt sich gut laminieren und schleifen.",
       "Rümpfe, Hauben, Antennenträger, Leiterplatten-Basis (FR4), Schutzschalen."),
    _m("pa_cf", "PA-CF (Nylon kohlefaserverstärkt, FDM)", "verbund", 1.15, (80.0, 130.0), 6.0, 130.0,
       "teuer", "FDM",
       "Steifestes gängiges FDM-Filament, geringe Verzugsneigung, matte Oberfläche. Braucht gehärtete Düse "
       "(≥ 0,4 mm) und trockenes Filament; Festigkeit vor allem in Druckrichtung (Schichthaftung schwächer).",
       "Funktionsteile, Motorhalter, Halterungen mit Steifigkeitsanspruch, Drohnen-Kleinteile."),
    # ------------------------------------------------------------ Metalle
    _m("alu_6061", "Aluminium 6061-T6", "metall", 2.70, (260.0, 310.0), 69.0, 150.0,
       "günstig", None,
       "Gut zerspanbar, schweißbar, eloxierbar, korrosionsbeständig. Standard-Konstruktionslegierung; "
       "Festigkeit sinkt ab ~150 °C deutlich.",
       "Frästeile, Halterungen, Profile, Motorträger, Gehäuse."),
    _m("alu_7075", "Aluminium 7075-T6", "metall", 2.81, (500.0, 570.0), 72.0, 120.0,
       "mittel", None,
       "Hochfeste Luftfahrtlegierung (Festigkeit wie Baustahl bei einem Drittel der Dichte). Schlecht "
       "schweißbar, kerb- und spannungsrisskorrosionsempfindlich; eloxieren möglich.",
       "Hochbelastete Frästeile, Fahrwerksteile, Achsen, Rahmenelemente."),
    _m("stahl_s235", "Baustahl S235JR", "metall", 7.85, (360.0, 510.0), 210.0, 400.0,
       "günstig", None,
       "Streckgrenze ~235 MPa. Sehr gut schweißbar, zäh, günstig; rostet ohne Beschichtung. "
       "Hohe Dichte – für Leichtbau ungeeignet.",
       "Gestelle, Vorrichtungen, Schweißkonstruktionen, Prüfstände."),
    _m("edelstahl_1_4301", "Edelstahl 1.4301 (V2A, AISI 304)", "metall", 7.90, (500.0, 700.0), 193.0, 600.0,
       "mittel", None,
       "Austenitisch, nicht magnetisch, sehr korrosionsbeständig, lebensmittelecht. Zäh, neigt beim "
       "Zerspanen zum Verfestigen (scharfe Werkzeuge, wenig Drehzahl). Schlechter Wärmeleiter.",
       "Schrauben, Wellen, Küchen-/Außenanwendungen, Behälter, Federn (nur 1.4310)."),
    _m("titan_grade5", "Titan Grade 5 (Ti-6Al-4V)", "metall", 4.43, (895.0, 1000.0), 114.0, 350.0,
       "teuer", None,
       "Beste spezifische Festigkeit der Metalle, exzellent korrosionsbeständig, biokompatibel. "
       "Schwierig zu zerspanen (geringe Wärmeleitung), teuer; E-Modul nur halb so hoch wie Stahl.",
       "Schrauben im Leichtbau, Implantate, Luftfahrt-Fittings, Hochleistungs-Fahrwerke."),
    _m("kupfer", "Kupfer (Cu-ETP, halbhart)", "metall", 8.93, (200.0, 350.0), 120.0, 250.0,
       "mittel", None,
       "Hervorragende elektrische (58 MS/m) und thermische Leitfähigkeit (~390 W/mK). Weich, duktil, "
       "lötbar; schwer und teuer für Strukturteile.",
       "Leiter, Kabel, Kühlkörper, Wärmeleitplatten, Sammelschienen."),
    _m("messing", "Messing (CuZn37, Ms63)", "metall", 8.45, (340.0, 460.0), 100.0, 200.0,
       "mittel", None,
       "Sehr gut zerspanbar und lötbar, korrosionsbeständig, dekorativ. Gute Gleiteigenschaften, "
       "hohe Dichte.",
       "Gewindeeinsätze, Buchsen, Düsen, Drehteile, Zierteile."),
    _m("magnesium_az31", "Magnesium AZ31B", "metall", 1.78, (240.0, 290.0), 45.0, 120.0,
       "teuer", None,
       "Leichtestes Konstruktionsmetall (35 % leichter als Aluminium), gute Dämpfung. Korrosionsanfällig "
       "(Beschichtung nötig), Späne brennbar, begrenzte Festigkeit und Kriechneigung.",
       "Gehäuse von Kameras/Laptops, Leichtbau-Halter, Flugmodelle (Blech)."),
    # ------------------------------------------------------------ Kunststoffe
    _m("pla", "PLA (Polylactid)", "kunststoff", 1.24, (45.0, 65.0), 3.5, 55.0,
       "günstig", "FDM",
       "Einfachster Druck (Verzug gering), steif, aber spröde und ab ~55 °C weich (Auto im Sommer!). "
       "Nicht UV-/witterungsbeständig, kriecht unter Dauerlast.",
       "Prototypen, Passformtests, Innenraum-Teile, Vorrichtungen ohne Wärme."),
    _m("petg", "PETG", "kunststoff", 1.27, (45.0, 55.0), 2.1, 75.0,
       "günstig", "FDM",
       "Zäher und temperaturfester als PLA, geringe Feuchteaufnahme, chemisch beständig, leicht flexibel. "
       "Neigt zum Fädenziehen; Schichthaftung sehr gut.",
       "Funktionsteile, Halterungen, Behälter, Außenanwendungen (bedingt UV-stabil)."),
    _m("abs", "ABS", "kunststoff", 1.04, (35.0, 45.0), 2.2, 90.0,
       "günstig", "FDM",
       "Schlagzäh, temperaturbeständig, mit Aceton glättbar/klebbar. Starker Verzug und Dämpfe beim "
       "Druck (Gehäuse + Lüftung nötig), nicht UV-stabil.",
       "Gehäuse, Clips, Teile mit Wärmekontakt (Elektronik), Zahnräder (gering belastet)."),
    _m("asa", "ASA", "kunststoff", 1.07, (40.0, 50.0), 2.3, 95.0,
       "günstig", "FDM",
       "Wie ABS, aber UV- und witterungsbeständig (kein Vergilben). Ebenfalls Gehäuse-Druck empfohlen.",
       "Außenteile, Fahrzeug-/Drohnen-Hauben, Gartengeräte, Halter im Freien."),
    _m("nylon_pa12", "Nylon PA12", "kunststoff", 1.01, (45.0, 55.0), 1.6, 110.0,
       "mittel", "FDM",
       "Sehr zäh und abriebfest, geringe Reibung, chemisch beständig. Nimmt Feuchte auf (vor Druck "
       "trocknen), Verzug moderat (PA12 geringer als PA6). Auch SLS-Standardwerkstoff.",
       "Zahnräder, Scharniere, Schnappverbindungen, Gleitlager, Riemenscheiben."),
    _m("tpu", "TPU 95A (flexibel)", "kunststoff", 1.21, (25.0, 45.0), 0.05, 80.0,
       "mittel", "FDM",
       "Gummiartig (Shore 95A), extrem zäh, abriebfest, dämpfend. E-Modul stark dehnungsabhängig "
       "(Richtwert bei kleiner Dehnung). Langsam drucken, Direct-Drive-Extruder.",
       "Dämpfer, Füße, Schutzhüllen, Dichtungen, Landekufen, flexible Scharniere."),
    _m("resin_standard", "Standard-Resin (SLA, UV-härtend)", "kunststoff", 1.15, (40.0, 65.0), 2.0, 50.0,
       "mittel", "SLA",
       "Sehr feine Details und glatte Oberfläche, aber spröde, altert unter UV und wird ab ~50 °C weich. "
       "Nachhärten nötig; Harz hautreizend (Handschuhe, Lüftung). Tough/ABS-like-Resins zäher.",
       "Sichtmodelle, Miniaturen, Formen, Passteile ohne Stoßbelastung."),
    _m("pom", "POM (Polyoxymethylen, Delrin)", "kunststoff", 1.41, (60.0, 70.0), 2.8, 100.0,
       "mittel", None,
       "Sehr maßhaltig, steif, geringe Reibung, kaum Feuchteaufnahme – der Zerspanungs-Kunststoff. "
       "Schlecht klebbar, FDM-Druck kaum möglich (Haftung).",
       "Zahnräder, Gleitlager, Führungen, Schnapphaken, Präzisionsteile (gefräst/gedreht)."),
    _m("pc", "Polycarbonat (PC)", "kunststoff", 1.20, (60.0, 70.0), 2.3, 120.0,
       "mittel", "FDM",
       "Sehr schlagzäh, transparent, hohe Wärmeformbeständigkeit. Druck braucht ≥ 270 °C und Gehäuse; "
       "kerbempfindlich, nicht beständig gegen Alkohole/Lösungsmittel.",
       "Schutzscheiben, Hauben, Gehäuse mit Wärme, FPV-Kamera-Schutz."),
    # ------------------------------------------------------------ Holz
    _m("balsa", "Balsaholz", "holz", 0.16, (10.0, 20.0), 3.5, None,
       "mittel", None,
       "Leichtestes Nutzholz (0,10–0,25 g/cm³), Werte längs zur Faser; quer zur Faser sehr weich. "
       "Dauergebrauchstemperatur nicht sinnvoll definierbar (Feuchte/Leim begrenzen). Mit Sekundenkleber "
       "oder Weißleim verbinden, bespannen oder lackieren.",
       "Flugmodell-Rippen, Leitwerke, Holme (mit CFK verstärkt), Kerne für Sandwich-Bauteile."),
    _m("birkensperrholz", "Birkensperrholz", "holz", 0.68, (40.0, 70.0), 10.0, None,
       "günstig", None,
       "Mehrlagig verleimt, dadurch in beiden Richtungen tragfähig; Werte in Plattenebene. Gut zu lasern "
       "und fräsen, quillt bei Feuchte (versiegeln). Dauergebrauchstemperatur leimabhängig, nicht definierbar.",
       "Spanten, Laser-Bausätze, Motorspanten, Gehäuse, Vorrichtungen, Lehren."),
)}


# ------------------------------------------------------------------ Suche
_ALIASES: dict[str, tuple[str, ...]] = {
    "cfk": ("cfk", "carbon", "karbon", "kohlefaser", "kohlenstofffaser", "carbonfaser", "cfrp",
            "carbonfiber", "carbonfibre", "kohlefaserverstaerkt", "carbonplatte", "carbonrohr"),
    "gfk": ("gfk", "glasfaser", "gfrp", "fiberglass", "fiberglas", "glasfiber", "glasfaserverstaerkt", "fr4"),
    "pa_cf": ("pacf", "pa12cf", "pa6cf", "nyloncf", "nyloncarbon", "carbonnylon", "polyamidcf", "pahtcf",
              "nylonkohlefaser", "pakohlefaser"),
    "alu_6061": ("alu6061", "aluminium6061", "aluminum6061", "al6061", "6061", "6061t6", "alu", "aluminium",
                 "aluminum", "al", "almgsi"),
    "alu_7075": ("alu7075", "aluminium7075", "aluminum7075", "al7075", "7075", "7075t6", "alznmgcu",
                 "flugzeugaluminium", "flugzeugalu"),
    "stahl_s235": ("stahls235", "s235", "s235jr", "stahl", "baustahl", "steel", "st37", "schwarzstahl"),
    "edelstahl_1_4301": ("edelstahl14301", "14301", "v2a", "aisi304", "304", "edelstahl", "niro", "nirosta",
                         "rostfrei", "inox", "stainless", "stainlesssteel", "x5crni1810", "rostfreierstahl"),
    "titan_grade5": ("titangrade5", "titan", "titanium", "ti6al4v", "ti64", "grade5", "titangrad5", "tial6v4",
                     "ti"),
    "pla": ("pla", "polylactid", "polymilchsaeure", "plaplus"),
    "petg": ("petg", "pet", "polyethylenterephthalat"),
    "abs": ("abs", "acrylnitrilbutadienstyrol"),
    "asa": ("asa", "acrylesterstyrolacrylnitril"),
    "nylon_pa12": ("nylonpa12", "nylon", "pa12", "pa", "polyamid", "pa6", "pa66", "polyamid12"),
    "tpu": ("tpu", "flex", "flexfilament", "polyurethan", "tpu95a", "tpe"),
    "resin_standard": ("resinstandard", "resin", "harz", "kunstharz", "sla", "slaresin", "standardresin",
                       "uvharz", "uvresin", "photopolymer", "giessharz"),
    "pom": ("pom", "delrin", "polyoxymethylen", "acetal", "polyacetal"),
    "pc": ("pc", "polycarbonat", "polycarbonate", "makrolon", "lexan"),
    "balsa": ("balsa", "balsaholz", "holz", "wood"),
    "birkensperrholz": ("birkensperrholz", "sperrholz", "birke", "birkenholz", "plywood", "multiplex",
                        "birchply", "birch", "pappelsperrholz", "flugzeugsperrholz"),
    "kupfer": ("kupfer", "copper", "cu", "cuetp", "ecu", "elektrolytkupfer"),
    "messing": ("messing", "brass", "ms58", "ms63", "cuzn", "cuzn37", "cuzn39pb3"),
    "magnesium_az31": ("magnesiumaz31", "az31", "az31b", "magnesium", "mg", "mgal3zn1"),
}

_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "é": "e", "è": "e"})
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _norm_name(text: str) -> str:
    """Normalform eines Materialnamens: klein, Umlaute umgeschrieben, nur Buchstaben/Ziffern."""
    return _NON_ALNUM.sub("", str(text).lower().translate(_UMLAUTS))


def _alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for key, mat in MATERIALS.items():
        index.setdefault(_norm_name(key), key)
        index.setdefault(_norm_name(mat.name), key)
    for key, aliases in _ALIASES.items():
        for a in aliases:
            index.setdefault(_norm_name(a), key)
    return index


_ALIAS_INDEX = _alias_index()


def find_material(name: str) -> Material | None:
    """Findet ein Material tolerant: Schlüssel, Name, Alias („Carbon", „CFK", „Alu 7075",
    „Edelstahl 1.4301", „V2A", „PETG", „Harz", „Holz" …), Teilwort (längster Treffer, Präfix
    bevorzugt) und zuletzt Tippfehler-Toleranz. ``None`` wenn nichts passt."""
    if name is None:
        return None
    norm = _norm_name(name)
    if not norm:
        return None
    key = _ALIAS_INDEX.get(norm)
    if key:
        return MATERIALS[key]
    # sehr naher Tippfehler ("Sperholz" -> sperrholz) vor der Teilwortsuche
    close = difflib.get_close_matches(norm, list(_ALIAS_INDEX), n=1, cutoff=0.85)
    if close:
        return MATERIALS[_ALIAS_INDEX[close[0]]]
    # Teilwort-Treffer: Alias im Text oder Text im Alias (mind. 3 Zeichen), längster gewinnt,
    # Präfix-Treffer bevorzugt ("Aluplatte" -> alu, nicht pla)
    best: tuple[int, int, str] | None = None
    for alias, k in _ALIAS_INDEX.items():
        if len(alias) < 3 or len(norm) < 3:
            continue
        if alias in norm:
            score = (1 if norm.startswith(alias) else 0, len(alias))
        elif norm in alias and len(norm) >= 5:
            score = (0, len(norm))
        else:
            continue
        if best is None or score > best[:2]:
            best = (score[0], score[1], k)
    if best:
        return MATERIALS[best[2]]
    close = difflib.get_close_matches(norm, list(_ALIAS_INDEX), n=1, cutoff=0.8)
    if close:
        return MATERIALS[_ALIAS_INDEX[close[0]]]
    return None


def list_materials(category: str | None = None) -> list[Material]:
    """Alle Materialien (optional nach Kategorie), sortiert nach Kategorie und Name."""
    mats = [m for m in MATERIALS.values() if category is None or m.category == category]
    return sorted(mats, key=lambda m: (CATEGORIES.index(m.category), m.name.lower()))


def _split_names(names: Iterable[str] | str) -> list[str]:
    if isinstance(names, str):
        parts = re.split(r"[,;/|]| und | vs\.? | oder |\n", names)
    else:
        parts = list(names)
    return [str(p).strip() for p in parts if p is not None and str(p).strip()]


def compare_materials(names: list[str] | str) -> list[Material]:
    """Löst eine Liste von Namen (oder einen kommagetrennten String) in Materialien auf.
    Doppelte werden entfernt; unbekannte Namen lösen ``ValueError`` aus."""
    parts = _split_names(names)
    if not parts:
        raise ValueError("Keine Materialnamen angegeben.")
    result: list[Material] = []
    unknown: list[str] = []
    for p in parts:
        m = find_material(p)
        if m is None:
            unknown.append(p)
        elif m not in result:
            result.append(m)
    if unknown:
        raise ValueError(
            "Unbekannte Materialien: " + ", ".join(f"»{u}«" for u in unknown)
            + ". Bekannt: " + ", ".join(sorted(MATERIALS)) + ".")
    return result


def _fmt_temp(t: float | None) -> str:
    return "–" if t is None else fmt_number(float(t), 3)


def material_table(materials: Sequence[Material]) -> str:
    """Ausgerichtete ASCII-Tabelle (deutsch) mit den wichtigsten Kennwerten."""
    if not materials:
        return "Keine Materialien."
    header = ("Material", "Kategorie", "Dichte g/cm³", "Zugfestigkeit MPa", "E-Modul GPa",
              "max. °C", "Kosten", "3D-Druck")
    rows: list[tuple[str, ...]] = []
    for m in materials:
        rows.append((
            m.name,
            _CATEGORY_LABEL.get(m.category, m.category),
            fmt_number(m.density, 3),
            f"{fmt_number(m.tensile_mpa[0], 4)}–{fmt_number(m.tensile_mpa[1], 4)}",
            fmt_number(m.youngs_gpa, 3),
            _fmt_temp(m.max_temp_c),
            m.cost,
            m.printable or "–",
        ))
    widths = [max(len(header[i]), *(len(r[i]) for r in rows)) for i in range(len(header))]
    right = {2, 3, 4, 5}        # Zahlenspalten rechtsbündig

    def line(cells: Sequence[str]) -> str:
        parts = []
        for i, c in enumerate(cells):
            parts.append(c.rjust(widths[i]) if i in right else c.ljust(widths[i]))
        return " | ".join(parts)

    out = [line(header), "-+-".join("-" * w for w in widths)]
    out.extend(line(r) for r in rows)
    out.append("")
    out.append(RICHTWERT_HINWEIS)
    return "\n".join(out)


def material_info(name: str) -> str:
    """Ausführliche deutsche Beschreibung eines Materials (für das Werkzeug ``material_info``)."""
    m = find_material(name)
    if m is None:
        raise ValueError(f"Unbekanntes Material »{name}«. Bekannt: " + ", ".join(sorted(MATERIALS)) + ".")
    lines = [
        f"{m.name} [{m.key}] – Richtwerte",
        f"Kategorie: {_CATEGORY_LABEL.get(m.category, m.category)}",
        f"Dichte: {fmt_number(m.density, 3)} g/cm³",
        f"Zugfestigkeit: {fmt_number(m.tensile_mpa[0])}–{fmt_number(m.tensile_mpa[1])} MPa",
        f"E-Modul: {fmt_number(m.youngs_gpa, 3)} GPa",
        "Dauergebrauchstemperatur: " + ("nicht definierbar" if m.max_temp_c is None
                                        else f"ca. {fmt_number(float(m.max_temp_c), 3)} °C"),
        f"Kosten: {m.cost}",
        "3D-Druck: " + (m.printable if m.printable else "nein"),
        f"Spezifische Festigkeit: {fmt_number(m.specific_strength, 3)} MPa/(g/cm³), "
        f"spezifische Steifigkeit: {fmt_number(m.specific_stiffness, 3)} GPa/(g/cm³)",
        f"Hinweise: {m.notes}",
        f"Typischer Einsatz: {m.typical_use}",
        RICHTWERT_HINWEIS,
    ]
    return "\n".join(lines)


# ------------------------------------------------------------------ Akku
def lipo_energy(cells: int, mah: float) -> dict:
    """Energieinhalt eines LiPo-Akkus: Nennspannung (3,7 V/Zelle) und Wh."""
    n = _int_range("Zellenzahl", cells, 1, MAX_CELLS)
    cap = _pos("Kapazität (mAh)", mah)
    v_nom = n * CELL_NOMINAL_V
    wh = v_nom * cap / 1000.0
    return {
        "zellen": n,
        "kapazitaet_mah": _round(cap, 1),
        "kapazitaet_ah": _round(cap / 1000.0, 4),
        "spannung_v": _round(v_nom, 2),
        "spannung_voll_v": _round(n * CELL_FULL_V, 2),
        "spannung_lagerung_v": _round(n * CELL_STORAGE_V, 2),
        "spannung_leer_v": _round(n * CELL_EMPTY_V, 2),
        "energie_wh": _round(wh, 3),
        "energie_kj": _round(wh * 3.6, 3),
        "formel": "E [Wh] = Zellen × 3,7 V × Kapazität [mAh] / 1000",
        "annahmen": [
            f"LiPo-Nennspannung {fmt_number(CELL_NOMINAL_V)} V je Zelle (voll {fmt_number(CELL_FULL_V)} V, "
            f"Lagerung {fmt_number(CELL_STORAGE_V)} V, Entladeschluss ≈ {fmt_number(CELL_EMPTY_V)} V)",
            "Richtwert: nutzbare Kapazität sinkt mit Alter, Kälte und hohem Entladestrom",
        ],
    }


def flight_time(mah: float, cells: int, avg_current_a: float, usable: float = 0.8) -> dict:
    """Flug-/Laufzeit in Minuten aus Kapazität, Zellenzahl und mittlerem Strom."""
    cap = _pos("Kapazität (mAh)", mah)
    n = _int_range("Zellenzahl", cells, 1, MAX_CELLS)
    i_avg = _pos("mittlerer Strom (A)", avg_current_a)
    u = _num("nutzbarer Anteil", usable)
    if not 0 < u <= 1:
        raise ValueError(f"Nutzbarer Anteil muss zwischen 0 und 1 liegen (erhalten: {fmt_number(u)}).")
    usable_mah = cap * u
    minutes = usable_mah / 1000.0 / i_avg * 60.0
    v_nom = n * CELL_NOMINAL_V
    return {
        "flugzeit_min": _round(minutes, 2),
        "flugzeit_s": _round(minutes * 60.0, 0),
        "kapazitaet_nutzbar_mah": _round(usable_mah, 1),
        "nutzbarer_anteil_prozent": _round(u * 100.0, 1),
        "strom_mittel_a": _round(i_avg, 3),
        "spannung_v": _round(v_nom, 2),
        "leistung_mittel_w": _round(v_nom * i_avg, 2),
        "energie_wh": _round(v_nom * cap / 1000.0, 3),
        "formel": "t [min] = Kapazität [mAh] × nutzbarer Anteil / 1000 / Strom [A] × 60",
        "annahmen": [
            f"Nur {fmt_number(u * 100.0, 3)} % der Kapazität nutzbar (LiPo nicht unter ≈ 3,5 V/Zelle entladen)",
            "Konstanter mittlerer Strom; Schwebeflug, Wind und Flugstil ändern ihn stark",
            f"Nennspannung {fmt_number(CELL_NOMINAL_V)} V je Zelle für Leistung/Energie",
            "Richtwert ±20 %",
        ],
    }


def battery_c_check(mah: float, c_rating: float, current_a: float) -> dict:
    """Prüft, ob ein Akku mit C-Rate den geforderten Dauerstrom liefern kann."""
    cap = _pos("Kapazität (mAh)", mah)
    c = _pos("C-Rate", c_rating)
    i = _nonneg("Strom (A)", current_a)
    max_i = cap / 1000.0 * c
    load = i / max_i * 100.0
    needed_c = i / (cap / 1000.0)
    if load <= 70:
        rating = "ok – ausreichende Reserve"
    elif load <= 100:
        rating = "grenzwertig – Akku wird heiß, Spannungseinbruch unter Last"
    else:
        rating = "zu hoch – Akku überlastet (Gefahr), größere Kapazität oder höhere C-Rate wählen"
    return {
        "max_strom_a": _round(max_i, 2),
        "strom_a": _round(i, 3),
        "benoetigte_c_rate": _round(needed_c, 2),
        "auslastung_prozent": _round(load, 1),
        "reserve_a": _round(max_i - i, 2),
        "ausreichend": i <= max_i,
        "bewertung": rating,
        "formel": "I_max [A] = Kapazität [mAh] / 1000 × C-Rate",
        "annahmen": [
            "C-Rate laut Hersteller gilt für neue Akkus bei Raumtemperatur; real oft optimistisch",
            "Empfehlung: Dauerstrom ≤ 70 % von I_max",
        ],
    }


# ------------------------------------------------------------------ Schub
def _twr_rating(twr: float) -> str:
    if twr < 1.0:
        return "hebt nicht ab (Schub < Gewicht)"
    if twr < 1.5:
        return "sehr knapp – nur ruhiges Schweben, keine Reserve für Wind"
    if twr < 2.0:
        return "ausreichend für ruhige Foto-/Videoflüge"
    if twr < 3.0:
        return "gut – sportliches Fliegen möglich"
    if twr < 5.0:
        return "hoch – Freestyle/Racing"
    return "sehr hoch – Racing, starke Motoren nötig"


def required_thrust(mass_g: float, twr: float = 2.0, motors: int = 4) -> dict:
    """Nötiger Gesamt- und Motorschub für ein Abfluggewicht bei gegebenem Schub-Gewichts-Verhältnis."""
    m = _pos("Masse (g)", mass_g)
    t = _pos("Schub-Gewichts-Verhältnis", twr)
    n = _int_range("Motorenzahl", motors, 1, MAX_MOTORS)
    total_g = m * t
    total_n = total_g / 1000.0 * G
    return {
        "masse_g": _round(m, 1),
        "gewicht_n": _round(m / 1000.0 * G, 3),
        "twr": _round(t, 2),
        "motoren": n,
        "schub_gesamt_g": _round(total_g, 1),
        "schub_gesamt_n": _round(total_n, 3),
        "schub_je_motor_g": _round(total_g / n, 1),
        "schub_je_motor_n": _round(total_n / n, 3),
        "schwebeschub_je_motor_g": _round(m / n, 1),
        "schwebe_gas_prozent": _round(100.0 / t, 1),
        "bewertung": _twr_rating(t),
        "formel": "Schub_gesamt = Masse × TWR; Schub_je_Motor = Schub_gesamt / Motoren; F [N] = m [kg] × 9,81",
        "annahmen": [
            f"g = {fmt_number(G)} m/s²",
            "Abfluggewicht inkl. Akku und Nutzlast",
            "TWR 2 = Schweben bei ~50 % Gas (Standard für Foto/Video), Racing 4–10",
        ],
    }


def thrust_to_weight(mass_g: float, thrust_per_motor_g: float, motors: int = 4) -> dict:
    """Schub-Gewichts-Verhältnis aus Abfluggewicht und maximalem Schub je Motor."""
    m = _pos("Masse (g)", mass_g)
    t_motor = _pos("Schub je Motor (g)", thrust_per_motor_g)
    n = _int_range("Motorenzahl", motors, 1, MAX_MOTORS)
    total_g = t_motor * n
    twr = total_g / m
    return {
        "twr": _round(twr, 2),
        "masse_g": _round(m, 1),
        "motoren": n,
        "schub_je_motor_g": _round(t_motor, 1),
        "schub_gesamt_g": _round(total_g, 1),
        "schub_gesamt_n": _round(total_g / 1000.0 * G, 3),
        "schwebe_gas_prozent": _round(100.0 / twr, 1) if twr > 0 else None,
        "max_abfluggewicht_twr2_g": _round(total_g / 2.0, 1),
        "bewertung": _twr_rating(twr),
        "formel": "TWR = Schub_je_Motor × Motoren / Masse",
        "annahmen": [
            "Schub je Motor laut Hersteller-Messung (Prüfstand, Meereshöhe); real 10–20 % weniger",
            f"g = {fmt_number(G)} m/s²",
        ],
    }


# ------------------------------------------------------------------ Elektrik
def ohm(u: float | None = None, i: float | None = None, r: float | None = None,
        p: float | None = None) -> dict:
    """Ohmsches Gesetz und Leistung: genau zwei Größen angeben, die anderen beiden werden berechnet."""
    given = {k: v for k, v in (("u", u), ("i", i), ("r", r), ("p", p)) if v is not None}
    if len(given) != 2:
        raise ValueError("Genau zwei der Größen Spannung (u), Strom (i), Widerstand (r), Leistung (p) angeben.")
    vals: dict[str, float] = {}
    for k, v in given.items():
        name = {"u": "Spannung (V)", "i": "Strom (A)", "r": "Widerstand (Ω)", "p": "Leistung (W)"}[k]
        vals[k] = _nonneg(name, v) if k in ("r", "p") else _num(name, v)
    keys = frozenset(vals)
    if keys == {"u", "i"}:
        if vals["i"] == 0:
            raise ValueError("Strom 0 A: Widerstand nicht bestimmbar.")
        res = {"u": vals["u"], "i": vals["i"], "r": vals["u"] / vals["i"], "p": vals["u"] * vals["i"]}
        formel = "R = U / I; P = U × I"
    elif keys == {"u", "r"}:
        if vals["r"] == 0:
            raise ValueError("Widerstand 0 Ω: Strom nicht bestimmbar (Kurzschluss).")
        res = {"u": vals["u"], "r": vals["r"], "i": vals["u"] / vals["r"], "p": vals["u"] ** 2 / vals["r"]}
        formel = "I = U / R; P = U² / R"
    elif keys == {"u", "p"}:
        if vals["u"] == 0:
            raise ValueError("Spannung 0 V: Strom nicht bestimmbar.")
        res = {"u": vals["u"], "p": vals["p"], "i": vals["p"] / vals["u"],
               "r": vals["u"] ** 2 / vals["p"] if vals["p"] else math.inf}
        formel = "I = P / U; R = U² / P"
    elif keys == {"i", "r"}:
        res = {"i": vals["i"], "r": vals["r"], "u": vals["i"] * vals["r"], "p": vals["i"] ** 2 * vals["r"]}
        formel = "U = I × R; P = I² × R"
    elif keys == {"i", "p"}:
        if vals["i"] == 0:
            raise ValueError("Strom 0 A: Spannung nicht bestimmbar.")
        res = {"i": vals["i"], "p": vals["p"], "u": vals["p"] / vals["i"], "r": vals["p"] / vals["i"] ** 2}
        formel = "U = P / I; R = P / I²"
    else:  # r, p
        if vals["r"] == 0:
            raise ValueError("Widerstand 0 Ω: Strom nicht bestimmbar.")
        res = {"r": vals["r"], "p": vals["p"], "i": math.sqrt(vals["p"] / vals["r"]),
               "u": math.sqrt(vals["p"] * vals["r"])}
        formel = "I = √(P / R); U = √(P × R)"
    if math.isinf(res["r"]):
        raise ValueError("Leistung 0 W bei Spannung ≠ 0: Widerstand unendlich (offener Kreis).")
    labels = {"u": "Spannung", "i": "Strom", "r": "Widerstand", "p": "Leistung"}
    return {
        "spannung_v": _round(res["u"], 4),
        "strom_a": _round(res["i"], 4),
        "widerstand_ohm": _round(res["r"], 4),
        "leistung_w": _round(res["p"], 4),
        "gegeben": [labels[k] for k in ("u", "i", "r", "p") if k in given],
        "formel": formel,
        "annahmen": ["Gleichstrom bzw. Effektivwerte, rein ohmscher Verbraucher (Widerstand temperaturunabhängig)"],
    }


def voltage_divider(u_in: float, r1: float, r2: float) -> dict:
    """Unbelasteter Spannungsteiler: U_aus = U_ein × R2 / (R1 + R2)."""
    u = _num("Eingangsspannung (V)", u_in)
    a = _nonneg("R1 (Ω)", r1)
    b = _nonneg("R2 (Ω)", r2)
    if a + b == 0:
        raise ValueError("R1 + R2 darf nicht 0 Ω sein.")
    i = u / (a + b)
    u_out = u * b / (a + b)
    return {
        "spannung_ein_v": _round(u, 4),
        "spannung_aus_v": _round(u_out, 4),
        "verhaeltnis": _round(b / (a + b), 5),
        "strom_a": _round(abs(i), 6),
        "strom_ma": _round(abs(i) * 1000.0, 3),
        "leistung_r1_w": _round(i ** 2 * a, 6),
        "leistung_r2_w": _round(i ** 2 * b, 6),
        "leistung_gesamt_w": _round(i ** 2 * (a + b), 6),
        "formel": "U_aus = U_ein × R2 / (R1 + R2); I = U_ein / (R1 + R2); P = I² × R",
        "annahmen": [
            "Unbelasteter Teiler – eine Last parallel zu R2 senkt U_aus (Last ≥ 10 × R2 wählen oder ADC-Eingang hochohmig)",
            "Widerstandstoleranzen (1 %/5 %) wirken direkt auf U_aus",
        ],
    }


def wire_size(current_a: float, length_m: float, voltage: float, max_drop_pct: float = 3.0) -> dict:
    """Kupfer-Leiterquerschnitt und AWG für einen zulässigen Spannungsabfall (Hin- und Rückleiter)."""
    i = _pos("Strom (A)", current_a)
    length = _pos("Länge (m)", length_m)
    u = _pos("Spannung (V)", voltage)
    pct = _num("max. Spannungsabfall (%)", max_drop_pct)
    if not 0 < pct <= 50:
        raise ValueError(f"Max. Spannungsabfall muss zwischen 0 und 50 % liegen (erhalten: {fmt_number(pct)}).")
    drop_allowed = u * pct / 100.0
    area_needed = COPPER_RESISTIVITY * 2.0 * length * i / drop_allowed
    awg = awg_for_area(area_needed)
    if awg is None:
        raise ValueError(
            f"Benötigter Querschnitt {fmt_number(area_needed)} mm² übersteigt AWG 4/0 (107 mm²): "
            "mehrere Leiter parallel, kürzere Leitung oder höhere Spannung.")
    area_awg = awg_area_mm2(awg)
    resistance = COPPER_RESISTIVITY * 2.0 * length / area_awg
    drop = resistance * i
    return {
        "strom_a": _round(i, 3),
        "laenge_m": _round(length, 3),
        "leiterlaenge_gesamt_m": _round(2.0 * length, 3),
        "spannung_v": _round(u, 3),
        "max_spannungsabfall_v": _round(drop_allowed, 4),
        "querschnitt_mm2": _round(area_needed, 4),
        "awg": awg,
        "awg_bezeichnung": awg_label(awg),
        "awg_querschnitt_mm2": _round(area_awg, 4),
        "awg_durchmesser_mm": _round(awg_diameter_mm(awg), 4),
        "widerstand_ohm": _round(resistance, 6),
        "spannungsabfall_v": _round(drop, 4),
        "spannungsabfall_prozent": _round(drop / u * 100.0, 3),
        "verlustleistung_w": _round(drop * i, 4),
        "formel": "A [mm²] = ρ × 2 × L × I / ΔU_max; ΔU = ρ × 2 × L × I / A_AWG; ρ_Cu = 0,0175 Ω·mm²/m",
        "annahmen": [
            "Kupferleiter bei 20 °C (bei 60 °C ≈ +16 % Widerstand), Hin- und Rückleiter (2 × Länge)",
            "Nur Spannungsabfall betrachtet – Erwärmung/Strombelastbarkeit nach Isolationsklasse zusätzlich prüfen "
            "(Silikonlitze grob 5–8 A/mm² frei verlegt, kurzzeitig mehr)",
            "Nächstgrößerer AWG-Querschnitt gewählt; Steckverbinder und Lötstellen addieren Widerstand",
        ],
    }


def awg_diameter_mm(awg: int) -> float:
    """Leiterdurchmesser eines AWG-Werts (−3 = 4/0 … 40) in mm."""
    n = _int_range("AWG", awg, MIN_AWG, MAX_AWG)
    return 0.127 * 92 ** ((36 - n) / 39.0)


def awg_area_mm2(awg: int) -> float:
    """Leiterquerschnitt eines AWG-Werts in mm²."""
    d = awg_diameter_mm(awg)
    return math.pi * d * d / 4.0


def awg_for_area(area_mm2: float) -> int | None:
    """Kleinster (dickster passender) AWG, dessen Querschnitt ≥ ``area_mm2`` ist; ``None`` wenn > 4/0."""
    a = _pos("Querschnitt (mm²)", area_mm2)
    for n in range(MAX_AWG, MIN_AWG - 1, -1):      # dünn -> dick
        if awg_area_mm2(n) >= a - 1e-9:
            return n
    return None


def awg_label(awg: int) -> str:
    """„AWG 14" bzw. „AWG 2/0" für Werte ≤ 0."""
    n = int(awg)
    if n <= 0:
        return f"AWG {1 - n}/0"
    return f"AWG {n}"


# ------------------------------------------------------------------ Mechanik
def torque(lever_m: float, force_n: float | None = None, mass_kg: float | None = None,
           safety: float = 1.5) -> dict:
    """Drehmoment aus Hebelarm und Kraft (oder Masse), mit Sicherheitsreserve."""
    lever = _pos("Hebelarm (m)", lever_m)
    if (force_n is None) == (mass_kg is None):
        raise ValueError("Entweder Kraft (N) oder Masse (kg) angeben – genau eine der beiden Größen.")
    s = _num("Sicherheitsfaktor", safety)
    if s < 1:
        raise ValueError(f"Sicherheitsfaktor muss ≥ 1 sein (erhalten: {fmt_number(s)}).")
    if force_n is not None:
        f = _pos("Kraft (N)", force_n)
        mass = f / G
        source = "Kraft direkt gegeben"
    else:
        mass = _pos("Masse (kg)", mass_kg)
        f = mass * G
        source = f"F = m × g = {fmt_number(mass)} kg × {fmt_number(G)} m/s²"
    t = f * lever
    return {
        "drehmoment_nm": _round(t, 4),
        "drehmoment_mit_reserve_nm": _round(t * s, 4),
        "drehmoment_ncm": _round(t * 100.0, 2),
        "drehmoment_kgcm": _round(t / G * 100.0, 3),
        "kraft_n": _round(f, 4),
        "masse_kg": _round(mass, 4),
        "hebel_m": _round(lever, 4),
        "sicherheitsfaktor": _round(s, 2),
        "formel": "M [Nm] = F [N] × Hebelarm [m]; M_Reserve = M × Sicherheitsfaktor; kg·cm = Nm / 9,81 × 100",
        "annahmen": [
            source,
            "Kraft wirkt senkrecht zum Hebelarm (sonst × sin(Winkel))",
            f"Sicherheitsfaktor {fmt_number(s)} für Reibung, Beschleunigung und Spannungsabfall (Servos/Motoren)",
        ],
    }


def motor_rpm(kv: float, voltage: float, load_factor: float = 0.85) -> dict:
    """Drehzahl eines Brushless-Motors aus KV und Spannung (Leerlauf und unter Last)."""
    k = _pos("KV (U/min je Volt)", kv)
    u = _pos("Spannung (V)", voltage)
    lf = _num("Lastfaktor", load_factor)
    if not 0 < lf <= 1:
        raise ValueError(f"Lastfaktor muss zwischen 0 und 1 liegen (erhalten: {fmt_number(lf)}).")
    idle = k * u
    loaded = idle * lf
    return {
        "kv": _round(k, 1),
        "spannung_v": _round(u, 2),
        "lastfaktor": _round(lf, 3),
        "drehzahl_leerlauf_rpm": _round(idle, 0),
        "drehzahl_rpm": _round(loaded, 0),
        "drehzahl_hz": _round(loaded / 60.0, 2),
        "formel": "n_Leerlauf [U/min] = KV × U; n_Last = n_Leerlauf × Lastfaktor",
        "annahmen": [
            f"Lastfaktor {fmt_number(lf)}: Drehzahleinbruch unter Propellerlast (typisch 0,75–0,9)",
            "Spannung = Akkuspannung unter Last (LiPo 3,7 V/Zelle nominal, nicht 4,2 V)",
            "KV-Angabe des Herstellers ist ein Richtwert (±5 %)",
        ],
    }


def prop_static_thrust(diameter_in: float, pitch_in: float, rpm: float) -> dict:
    """Statischer Propellerschub nach der empirischen Staples-Näherung (±30 %)."""
    d = _pos("Propellerdurchmesser (Zoll)", diameter_in)
    p = _pos("Steigung (Zoll)", pitch_in)
    n = _pos("Drehzahl (U/min)", rpm)
    if d > 60:
        raise ValueError(f"Propellerdurchmesser {fmt_number(d)} Zoll ist unplausibel (max. 60).")
    if p > 2 * d:
        raise ValueError("Steigung größer als der doppelte Durchmesser ist unplausibel.")
    thrust_n = 4.392e-8 * n * d ** 3.5 / math.sqrt(p) * (4.233e-4 * n * p)
    tip_speed = math.pi * d * 0.0254 * n / 60.0
    pitch_speed = n * p * 0.0254 / 60.0
    annahmen = [
        "Empirische Näherung nach Staples für den Standschub (Fluggeschwindigkeit 0)",
        "Grobe Schätzung ±30 % – Propellerprofil, Blattzahl, Luftdichte (Meereshöhe, 15 °C) und Motorleistung "
        "nicht berücksichtigt",
        "Der Motor muss die zugehörige Leistung auch liefern können",
    ]
    if tip_speed > 270:
        annahmen.append(f"Warnung: Blattspitzen {fmt_number(tip_speed, 3)} m/s > 0,8 Mach – Wirkungsgrad "
                        "bricht ein, Lärm, Materialgrenze")
    return {
        "durchmesser_zoll": _round(d, 2),
        "steigung_zoll": _round(p, 2),
        "drehzahl_rpm": _round(n, 0),
        "schub_n": _round(thrust_n, 3),
        "schub_g": _round(thrust_n / G * 1000.0, 1),
        "blattspitzengeschwindigkeit_m_s": _round(tip_speed, 1),
        "steigungsgeschwindigkeit_m_s": _round(pitch_speed, 1),
        "steigungsgeschwindigkeit_km_h": _round(pitch_speed * 3.6, 1),
        "formel": "T [N] = 4,392e-8 × n × d^3,5 / √p × (4,233e-4 × n × p)   (n in U/min, d, p in Zoll)",
        "annahmen": annahmen,
    }


def beam_cantilever(force_n: float, length_m: float, youngs_gpa: float, width_m: float,
                    height_m: float) -> dict:
    """Einseitig eingespannter Rechteckbalken mit Einzellast am freien Ende:
    Durchbiegung f = F L³ / (3 E I), I = b h³ / 12, Biegespannung σ = M h/2 / I."""
    f = _pos("Kraft (N)", force_n)
    length = _pos("Länge (m)", length_m)
    e_gpa = _pos("E-Modul (GPa)", youngs_gpa)
    b = _pos("Breite (m)", width_m)
    h = _pos("Höhe (m)", height_m)
    e_pa = e_gpa * 1e9
    inertia = b * h ** 3 / 12.0                   # m⁴
    deflection = f * length ** 3 / (3.0 * e_pa * inertia)
    moment = f * length
    stress = moment * (h / 2.0) / inertia         # Pa
    return {
        "kraft_n": _round(f, 3),
        "laenge_m": _round(length, 4),
        "e_modul_gpa": _round(e_gpa, 3),
        "breite_mm": _round(b * 1000.0, 3),
        "hoehe_mm": _round(h * 1000.0, 3),
        "flaechentraegheitsmoment_mm4": _round(inertia * 1e12, 3),
        "biegemoment_nm": _round(moment, 4),
        "durchbiegung_mm": _round(deflection * 1000.0, 4),
        "durchbiegung_relativ_prozent": _round(deflection / length * 100.0, 3),
        "biegespannung_mpa": _round(stress / 1e6, 3),
        "formel": "I = b × h³ / 12; f = F × L³ / (3 × E × I); σ = (F × L) × (h/2) / I",
        "annahmen": [
            "Euler-Bernoulli-Balken: kleine Verformungen (f ≪ L), linear-elastisch, homogen",
            "Einzellast senkrecht am freien Ende, feste Einspannung, Rechteckquerschnitt, Eigengewicht und "
            "Schubverformung vernachlässigt",
            "Vergleich mit Streck-/Zugfestigkeit des Materials nötig (Sicherheitsfaktor ≥ 2 bei dynamischer Last)",
        ],
    }


def mass_from_volume(volume_cm3: float, material: str) -> dict:
    """Masse eines Bauteils aus Volumen (cm³) und Materialdichte."""
    v = _pos("Volumen (cm³)", volume_cm3)
    m = find_material(material)
    if m is None:
        raise ValueError(f"Unbekanntes Material »{material}«. Bekannt: " + ", ".join(sorted(MATERIALS)) + ".")
    mass_g = v * m.density
    return {
        "volumen_cm3": _round(v, 3),
        "material": m.name,
        "material_schluessel": m.key,
        "dichte_g_cm3": m.density,
        "masse_g": _round(mass_g, 3),
        "masse_kg": _round(mass_g / 1000.0, 6),
        "gewicht_n": _round(mass_g / 1000.0 * G, 4),
        "formel": "m [g] = V [cm³] × ρ [g/cm³]",
        "annahmen": [
            f"Dichte {fmt_number(m.density, 3)} g/cm³ ist ein Richtwert ({m.name})",
            "Vollmaterial – bei 3D-Druck Füllgrad (Infill) und Wandstärke berücksichtigen",
        ],
    }


# ------------------------------------------------------------------ Einheiten
# Familie -> kanonische Einheit -> (Faktor zur Basiseinheit, Schreibweisen)
_UNIT_FAMILIES: dict[str, dict[str, tuple[float, tuple[str, ...]]]] = {
    "Länge": {
        "µm": (1e-6, ("um", "µm", "mikrometer", "micrometer", "mikron")),
        "mm": (1e-3, ("mm", "millimeter", "millimetre")),
        "cm": (1e-2, ("cm", "zentimeter", "centimeter", "centimetre")),
        "dm": (1e-1, ("dm", "dezimeter")),
        "m": (1.0, ("m", "meter", "metre", "meters", "metres")),
        "km": (1e3, ("km", "kilometer", "kilometre")),
        "in": (0.0254, ("in", "inch", "inches", "zoll", '"', "″", "''")),
        "ft": (0.3048, ("ft", "feet", "foot", "fuss", "fuß", "'", "′")),
        "yd": (0.9144, ("yd", "yard", "yards")),
        "mi": (1609.344, ("mi", "mile", "miles", "meile", "meilen")),
        "nmi": (1852.0, ("nmi", "seemeile", "seemeilen", "nauticalmile")),
    },
    "Masse": {
        "mg": (1e-6, ("mg", "milligramm", "milligram")),
        "g": (1e-3, ("g", "gramm", "gram", "grams")),
        "kg": (1.0, ("kg", "kilogramm", "kilogram", "kilo")),
        "t": (1000.0, ("t", "tonne", "tonnen", "ton")),
        "lb": (0.45359237, ("lb", "lbs", "pound", "pounds")),
        "oz": (0.028349523125, ("oz", "ounce", "ounces", "unze", "unzen")),
    },
    "Kraft": {
        "mN": (1e-3, ("mn", "millinewton")),
        "N": (1.0, ("n", "newton")),
        "kN": (1e3, ("knkraft", "kilonewton")),
        "gf": (G / 1000.0, ("gf", "grammkraft", "gramforce", "p", "pond")),
        "kgf": (G, ("kgf", "kp", "kilopond", "kilogrammkraft", "kilogramforce")),
        "lbf": (4.4482216152605, ("lbf", "poundforce", "pfundkraft")),
    },
    "Druck": {
        "Pa": (1.0, ("pa", "pascal", "n/m2", "nm2")),
        "hPa": (100.0, ("hpa", "hektopascal")),
        "kPa": (1e3, ("kpa", "kilopascal")),
        "MPa": (1e6, ("mpa", "megapascal", "n/mm2", "nmm2")),
        "GPa": (1e9, ("gpa", "gigapascal")),
        "mbar": (100.0, ("mbar", "millibar")),
        "bar": (1e5, ("bar",)),
        "psi": (6894.757293168, ("psi", "lbf/in2", "poundspersquareinch")),
        "atm": (101325.0, ("atm", "atmosphaere", "atmosphere")),
        "mmHg": (133.322387415, ("mmhg", "torr")),
    },
    "Geschwindigkeit": {
        "m/s": (1.0, ("m/s", "ms", "mps", "meterprosekunde", "meter/s", "m/sek")),
        "km/h": (1 / 3.6, ("km/h", "kmh", "kph", "kilometerprostunde", "km/std", "kilometer/h")),
        "mph": (0.44704, ("mph", "milesperhour", "meilenprostunde", "mi/h")),
        "kn": (0.514444, ("knknoten", "kt", "kts", "knoten", "knots", "knot")),
        "ft/s": (0.3048, ("ft/s", "fts", "fps", "feetpersecond")),
    },
    "Energie": {
        "J": (1.0, ("j", "joule", "ws", "wattsekunde")),
        "kJ": (1e3, ("kj", "kilojoule")),
        "MJ": (1e6, ("mj", "megajoule")),
        "Wh": (3600.0, ("wh", "wattstunde", "wattstunden", "watthour")),
        "kWh": (3.6e6, ("kwh", "kilowattstunde", "kilowattstunden", "kilowatthour")),
        "cal": (4.184, ("cal", "kalorie", "calorie")),
        "kcal": (4184.0, ("kcal", "kilokalorie", "kilocalorie")),
        "eV": (1.602176634e-19, ("ev", "elektronenvolt", "electronvolt")),
    },
    "Leistung": {
        "mW": (1e-3, ("milliwatt", "mwmilli")),
        "W": (1.0, ("w", "watt")),
        "kW": (1e3, ("kw", "kilowatt")),
        "MW": (1e6, ("megawatt", "mwmega")),
        "PS": (735.49875, ("ps", "pferdestaerke", "pferdestaerken", "metrichorsepower")),
        "hp": (745.6998715822702, ("hp", "horsepower", "bhp")),
    },
    "Temperatur": {
        "°C": (1.0, ("c", "°c", "celsius", "gradcelsius", "grad", "degc", "deg")),
        "°F": (1.0, ("f", "°f", "fahrenheit", "gradfahrenheit", "degf")),
        "K": (1.0, ("k", "kelvin")),
    },
}

# Mehrdeutige Kurzformen: werden über die Familie der Gegenseite aufgelöst
_AMBIGUOUS: dict[str, tuple[str, ...]] = {
    "kn": ("knkraft", "knknoten"),          # Kilonewton oder Knoten – Gegenseite entscheidet
    "mw": ("mwmilli",),                     # klein geschrieben: Milliwatt (MW groß = Megawatt)
}
_CASE_SENSITIVE: dict[str, str] = {"MW": "mwmega", "mW": "mwmilli", "kN": "knkraft"}
_CHARGE_UNITS = {"mah", "ah", "milliamperestunde", "milliamperestunden", "amperestunde", "amperestunden"}


def _unit_lookup() -> dict[str, list[tuple[str, str, float]]]:
    """Schreibweise -> Liste (Familie, kanonische Einheit, Faktor)."""
    table: dict[str, list[tuple[str, str, float]]] = {}
    for family, units in _UNIT_FAMILIES.items():
        for canon, (factor, spellings) in units.items():
            for s in spellings:
                table.setdefault(_norm_unit(s), []).append((family, canon, factor))
    return table


def _norm_unit(text: str) -> str:
    s = str(text).strip()
    if s in _CASE_SENSITIVE:
        return _CASE_SENSITIVE[s]
    s = s.lower().translate(_UMLAUTS)
    s = s.replace("²", "2").replace("³", "3").replace("⁄", "/").replace("µ", "u").replace("°", "")
    s = s.replace("grad ", "").replace("degrees", "").replace("degree", "")
    return re.sub(r"[\s\.\-_,]+", "", s)


_UNIT_TABLE = _unit_lookup()


def _resolve_unit(text: str) -> list[tuple[str, str, float]]:
    norm = _norm_unit(text)
    if not norm:
        raise ValueError("Leere Einheit.")
    if norm in _CHARGE_UNITS:
        raise ValueError(
            f"»{text}« ist eine Ladung (Kapazität), keine Energie: Energie [Wh] = mAh × Spannung [V] / 1000 "
            "– nutze akku_rechner mit Zellenzahl.")
    candidates: list[tuple[str, str, float]] = []
    if norm in _AMBIGUOUS:
        for alt in _AMBIGUOUS[norm]:
            candidates.extend(_UNIT_TABLE.get(alt, []))
    candidates.extend(_UNIT_TABLE.get(norm, []))
    if not candidates:
        known = sorted({c for fam in _UNIT_FAMILIES.values() for c in fam})
        raise ValueError(f"Unbekannte Einheit »{text}«. Bekannt u. a.: {', '.join(known)}.")
    return candidates


def _convert_temperature(value: float, src: str, dst: str) -> tuple[float, str]:
    if src == "°C":
        kelvin = value + 273.15
    elif src == "°F":
        kelvin = (value - 32.0) * 5.0 / 9.0 + 273.15
    else:
        kelvin = value
    if kelvin < 0:
        raise ValueError(f"{fmt_number(value)} {src} liegt unter dem absoluten Nullpunkt.")
    if dst == "°C":
        out = kelvin - 273.15
    elif dst == "°F":
        out = (kelvin - 273.15) * 9.0 / 5.0 + 32.0
    else:
        out = kelvin
    formulas = {
        ("°C", "°F"): "T[°F] = T[°C] × 9/5 + 32", ("°F", "°C"): "T[°C] = (T[°F] − 32) × 5/9",
        ("°C", "K"): "T[K] = T[°C] + 273,15", ("K", "°C"): "T[°C] = T[K] − 273,15",
        ("°F", "K"): "T[K] = (T[°F] − 32) × 5/9 + 273,15", ("K", "°F"): "T[°F] = (T[K] − 273,15) × 9/5 + 32",
    }
    return out, formulas.get((src, dst), f"T[{dst}] = T[{src}]")


def unit_convert(value: float, from_unit: str, to_unit: str) -> dict:
    """Rechnet zwischen Einheiten derselben Größe um (Länge, Masse, Kraft, Druck, Temperatur,
    Geschwindigkeit, Energie, Leistung). Schreibweisen sind tolerant (mm/Millimeter, °C/C,
    km/h/kmh, PS/hp …). Unverträgliche Einheiten lösen ``ValueError`` aus."""
    v = _num("Wert", value)
    src_c = _resolve_unit(from_unit)
    dst_c = _resolve_unit(to_unit)
    pairs = [(s, d) for s in src_c for d in dst_c if s[0] == d[0]]
    if not pairs:
        fam_s = ", ".join(sorted({s[0] for s in src_c}))
        fam_d = ", ".join(sorted({d[0] for d in dst_c}))
        raise ValueError(
            f"Einheiten nicht umrechenbar: »{from_unit}« ({fam_s}) nach »{to_unit}« ({fam_d}).")
    (family, src, f_src), (_, dst, f_dst) = pairs[0]
    annahmen = ["Exakte Umrechnungsfaktoren (SI); Ergebnis auf 10 signifikante Stellen gerundet"]
    if family == "Temperatur":
        out, formel = _convert_temperature(v, src, dst)
        factor = None
    else:
        factor = f_src / f_dst
        out = v * factor
        formel = f"{dst} = {src} × {fmt_number(factor, 10)}"
        if family == "Kraft":
            annahmen.append(f"Kraft aus Masse mit g = {fmt_number(G)} m/s² (kgf, gf)")
        if family == "Leistung" and {src, dst} & {"PS", "hp"}:
            annahmen.append("PS = metrische Pferdestärke (735,5 W), hp = mechanische Horsepower (745,7 W)")
    out = float(f"{out:.10g}")
    return {
        "wert": out,
        "einheit": dst,
        "von_wert": v,
        "von_einheit": src,
        "groesse": family,
        "faktor": factor,
        "formel": formel,
        "annahmen": annahmen,
    }


# ------------------------------------------------------------------ Textausgabe
_LABELS: dict[str, tuple[str, str]] = {
    # Akku
    "zellen": ("Zellen", "S"), "kapazitaet_mah": ("Kapazität", "mAh"), "kapazitaet_ah": ("Kapazität", "Ah"),
    "spannung_v": ("Spannung (nominal)", "V"), "spannung_voll_v": ("Spannung voll", "V"),
    "spannung_lagerung_v": ("Spannung Lagerung", "V"), "spannung_leer_v": ("Spannung Entladeschluss", "V"),
    "energie_wh": ("Energie", "Wh"), "energie_kj": ("Energie", "kJ"),
    "flugzeit_min": ("Flugzeit", "min"), "flugzeit_s": ("Flugzeit", "s"),
    "kapazitaet_nutzbar_mah": ("Nutzbare Kapazität", "mAh"), "nutzbarer_anteil_prozent": ("Nutzbarer Anteil", "%"),
    "strom_mittel_a": ("Mittlerer Strom", "A"), "leistung_mittel_w": ("Mittlere Leistung", "W"),
    "max_strom_a": ("Max. Dauerstrom", "A"), "benoetigte_c_rate": ("Benötigte C-Rate", "C"),
    "auslastung_prozent": ("Auslastung", "%"), "reserve_a": ("Reserve", "A"),
    "ausreichend": ("Ausreichend", ""), "bewertung": ("Bewertung", ""),
    # Schub
    "masse_g": ("Masse", "g"), "gewicht_n": ("Gewicht", "N"), "twr": ("Schub-Gewichts-Verhältnis", ""),
    "motoren": ("Motoren", ""), "schub_gesamt_g": ("Gesamtschub", "g"), "schub_gesamt_n": ("Gesamtschub", "N"),
    "schub_je_motor_g": ("Schub je Motor", "g"), "schub_je_motor_n": ("Schub je Motor", "N"),
    "schwebeschub_je_motor_g": ("Schwebeschub je Motor", "g"), "schwebe_gas_prozent": ("Gas im Schwebeflug", "%"),
    "max_abfluggewicht_twr2_g": ("Max. Abfluggewicht bei TWR 2", "g"),
    # Elektrik
    "strom_a": ("Strom", "A"), "strom_ma": ("Strom", "mA"), "widerstand_ohm": ("Widerstand", "Ω"),
    "leistung_w": ("Leistung", "W"), "gegeben": ("Gegeben", ""),
    "spannung_ein_v": ("Eingangsspannung", "V"), "spannung_aus_v": ("Ausgangsspannung", "V"),
    "verhaeltnis": ("Teilerverhältnis", ""), "leistung_r1_w": ("Leistung R1", "W"),
    "leistung_r2_w": ("Leistung R2", "W"), "leistung_gesamt_w": ("Gesamtleistung", "W"),
    "laenge_m": ("Länge", "m"), "leiterlaenge_gesamt_m": ("Leiterlänge (hin + zurück)", "m"),
    "max_spannungsabfall_v": ("Zulässiger Spannungsabfall", "V"), "querschnitt_mm2": ("Rechnerischer Querschnitt", "mm²"),
    "awg": ("AWG", ""), "awg_bezeichnung": ("Gewählt", ""), "awg_querschnitt_mm2": ("Querschnitt gewählt", "mm²"),
    "awg_durchmesser_mm": ("Leiterdurchmesser", "mm"), "spannungsabfall_v": ("Spannungsabfall", "V"),
    "spannungsabfall_prozent": ("Spannungsabfall", "%"), "verlustleistung_w": ("Verlustleistung", "W"),
    # Mechanik
    "drehmoment_nm": ("Drehmoment", "Nm"), "drehmoment_mit_reserve_nm": ("Drehmoment mit Reserve", "Nm"),
    "drehmoment_ncm": ("Drehmoment", "Ncm"), "drehmoment_kgcm": ("Drehmoment", "kg·cm"),
    "kraft_n": ("Kraft", "N"), "masse_kg": ("Masse", "kg"), "hebel_m": ("Hebelarm", "m"),
    "sicherheitsfaktor": ("Sicherheitsfaktor", ""), "kv": ("KV", "U/min/V"), "lastfaktor": ("Lastfaktor", ""),
    "drehzahl_leerlauf_rpm": ("Drehzahl Leerlauf", "U/min"), "drehzahl_rpm": ("Drehzahl unter Last", "U/min"),
    "drehzahl_hz": ("Drehzahl", "Hz"), "durchmesser_zoll": ("Durchmesser", "Zoll"),
    "steigung_zoll": ("Steigung", "Zoll"), "schub_n": ("Standschub", "N"), "schub_g": ("Standschub", "g"),
    "blattspitzengeschwindigkeit_m_s": ("Blattspitzengeschwindigkeit", "m/s"),
    "steigungsgeschwindigkeit_m_s": ("Steigungsgeschwindigkeit", "m/s"),
    "steigungsgeschwindigkeit_km_h": ("Steigungsgeschwindigkeit", "km/h"),
    "e_modul_gpa": ("E-Modul", "GPa"), "breite_mm": ("Breite", "mm"), "hoehe_mm": ("Höhe", "mm"),
    "flaechentraegheitsmoment_mm4": ("Flächenträgheitsmoment", "mm⁴"), "biegemoment_nm": ("Biegemoment", "Nm"),
    "durchbiegung_mm": ("Durchbiegung", "mm"), "durchbiegung_relativ_prozent": ("Durchbiegung relativ", "% der Länge"),
    "biegespannung_mpa": ("Biegespannung", "MPa"), "zugfestigkeit_min_mpa": ("Zugfestigkeit (min)", "MPa"),
    "sicherheit_gegen_bruch": ("Sicherheit gegen Bruch", ""),
    "volumen_cm3": ("Volumen", "cm³"), "material": ("Material", ""), "material_schluessel": ("Schlüssel", ""),
    "dichte_g_cm3": ("Dichte", "g/cm³"),
    # Einheiten
    "wert": ("Ergebnis", ""), "einheit": ("Einheit", ""), "von_wert": ("Ausgangswert", ""),
    "von_einheit": ("Ausgangseinheit", ""), "groesse": ("Größe", ""), "faktor": ("Faktor", ""),
}
_SKIP_KEYS = {"formel", "annahmen"}


def _fmt_value(value: Any) -> str:
    if value is None:
        return "–"
    if isinstance(value, bool):
        return "ja" if value else "nein"
    if isinstance(value, (int, float)):
        return fmt_number(value)
    if isinstance(value, (list, tuple)):
        return ", ".join(_fmt_value(v) for v in value)
    return str(value)


def format_result(title: str, result: dict, keys: Sequence[str] | None = None) -> str:
    """Deutscher Text aus einem Rechner-Ergebnis: Titel, Ergebniszeilen, Formel, Annahmen."""
    lines = [title]
    for key in (keys if keys is not None else result):
        if key in _SKIP_KEYS or key not in result:
            continue
        label, unit = _LABELS.get(key, (key.replace("_", " "), ""))
        value = _fmt_value(result[key])
        lines.append(f"  {label}: {value}{(' ' + unit) if unit and result[key] is not None else ''}")
    if result.get("formel"):
        lines.append(f"Formel: {result['formel']}")
    if result.get("annahmen"):
        lines.append("Annahmen:")
        lines.extend(f"- {a}" for a in result["annahmen"])
    return "\n".join(lines)


# ------------------------------------------------------------------ Werkzeuge
def _p_number(desc: str, example: float | None = None, default: float | None = None) -> dict:
    spec: dict[str, Any] = {"type": "number", "description": desc}
    if example is not None:
        spec["example"] = example
    if default is not None:
        spec["default"] = default
    return spec


def _p_int(desc: str, example: int | None = None, default: int | None = None) -> dict:
    spec: dict[str, Any] = {"type": "integer", "description": desc}
    if example is not None:
        spec["example"] = example
    if default is not None:
        spec["default"] = default
    return spec


def _p_str(desc: str, example: str | None = None) -> dict:
    spec: dict[str, Any] = {"type": "string", "description": desc}
    if example is not None:
        spec["example"] = example
    return spec


def _schema(props: dict, required: Sequence[str]) -> dict:
    return {"type": "object", "properties": props, "required": list(required)}


def _tool_material_info(name: str) -> str:
    return material_info(name)


def _tool_compare(namen: str) -> str:
    mats = compare_materials(namen)
    if len(mats) < 2:
        return material_info(mats[0].key)
    lines = [material_table(mats), ""]
    lightest = min(mats, key=lambda m: m.density)
    strongest = max(mats, key=lambda m: m.tensile_mpa[0])
    stiffest = max(mats, key=lambda m: m.youngs_gpa)
    spec_best = max(mats, key=lambda m: m.specific_strength)
    lines.append(f"Leichtestes: {lightest.name} ({fmt_number(lightest.density, 3)} g/cm³); "
                 f"festestes: {strongest.name} (≥ {fmt_number(strongest.tensile_mpa[0])} MPa); "
                 f"steifestes: {stiffest.name} ({fmt_number(stiffest.youngs_gpa, 3)} GPa); "
                 f"beste spezifische Festigkeit: {spec_best.name} "
                 f"({fmt_number(spec_best.specific_strength, 3)} MPa/(g/cm³)).")
    # Einsatzgebiete nur, solange die Ausgabe unter dem Werkzeug-Limit bleibt (sonst würde gekürzt)
    budget = MAX_TOOL_OUTPUT_CHARS - sum(len(ln) + 1 for ln in lines) - 80
    for m in mats:
        entry = f"- {m.name}: {m.typical_use}"
        if len(entry) + 1 > budget:
            lines.append("(weitere Einsatzgebiete über material_info)")
            break
        lines.append(entry)
        budget -= len(entry) + 1
    return "\n".join(lines)


def _tool_akku(zellen: int, mah: float, strom_a: float | None = None, c_rate: float | None = None) -> str:
    parts = [format_result("Akku-Rechner (LiPo)", lipo_energy(zellen, mah))]
    if strom_a is not None:
        parts.append(format_result("Flugzeit", flight_time(mah, zellen, strom_a),
                                   ("flugzeit_min", "kapazitaet_nutzbar_mah", "strom_mittel_a",
                                    "leistung_mittel_w", "formel", "annahmen")))
        if c_rate is not None:
            parts.append(format_result("C-Rate-Prüfung", battery_c_check(mah, c_rate, strom_a)))
    return "\n\n".join(parts)


def _tool_schub(masse_g: float, motoren: int = 4, schub_je_motor_g: float | None = None,
                twr: float = 2.0) -> str:
    parts = [format_result(f"Schub-Rechner (Ziel-TWR {fmt_number(float(_num('TWR', twr)))})",
                           required_thrust(masse_g, twr, motoren))]
    if schub_je_motor_g is not None:
        parts.append(format_result("Vorhandener Schub", thrust_to_weight(masse_g, schub_je_motor_g, motoren)))
    return "\n\n".join(parts)


def _tool_elektro(u: float | None = None, i: float | None = None, r: float | None = None,
                  p: float | None = None) -> str:
    return format_result("Elektro-Rechner (Ohmsches Gesetz)", ohm(u, i, r, p))


def _tool_teiler(u_in: float, r1: float, r2: float) -> str:
    return format_result("Spannungsteiler", voltage_divider(u_in, r1, r2))


def _tool_drehmoment(hebel_m: float, kraft_n: float | None = None, masse_kg: float | None = None,
                     sicherheit: float = 1.5) -> str:
    return format_result("Drehmoment-Rechner", torque(hebel_m, kraft_n, masse_kg, sicherheit))


def _tool_motor(kv: float, spannung: float, prop_zoll: float | None = None,
                steigung_zoll: float | None = None, lastfaktor: float = 0.85) -> str:
    rpm = motor_rpm(kv, spannung, lastfaktor)
    parts = [format_result("Motor-Rechner (Brushless)", rpm)]
    if (prop_zoll is None) != (steigung_zoll is None):
        raise ValueError("Für den Propellerschub Durchmesser (prop_zoll) und Steigung (steigung_zoll) angeben.")
    if prop_zoll is not None and steigung_zoll is not None:
        parts.append(format_result("Propeller-Standschub (bei Drehzahl unter Last)",
                                   prop_static_thrust(prop_zoll, steigung_zoll, rpm["drehzahl_rpm"])))
    return "\n\n".join(parts)


def _youngs_from_material(material: str) -> tuple[float, Material | None]:
    """E-Modul in GPa aus Materialname – oder direkt als Zahl („70")."""
    try:
        return _pos("E-Modul (GPa)", material), None
    except ValueError:
        pass
    m = find_material(material)
    if m is None:
        raise ValueError(f"Unbekanntes Material »{material}« (oder E-Modul in GPa als Zahl angeben). Bekannt: "
                         + ", ".join(sorted(MATERIALS)) + ".")
    return m.youngs_gpa, m


def _tool_balken(kraft_n: float, laenge_m: float, material: str, breite_m: float, hoehe_m: float) -> str:
    e_gpa, mat = _youngs_from_material(material)
    res = beam_cantilever(kraft_n, laenge_m, e_gpa, breite_m, hoehe_m)
    title = "Balken-Rechner (Kragbalken, Einzellast am Ende)"
    if mat is not None:
        title += f" – {mat.name}"
        res["zugfestigkeit_min_mpa"] = mat.tensile_mpa[0]
        res["sicherheit_gegen_bruch"] = _round(mat.tensile_mpa[0] / res["biegespannung_mpa"], 2) \
            if res["biegespannung_mpa"] > 0 else None
        res["annahmen"] = list(res["annahmen"]) + [
            f"E-Modul {fmt_number(mat.youngs_gpa, 3)} GPa und Zugfestigkeit ≥ {fmt_number(mat.tensile_mpa[0])} MPa "
            f"sind Richtwerte für {mat.name}"]
        if res["sicherheit_gegen_bruch"] is not None and res["sicherheit_gegen_bruch"] < 2:
            res["annahmen"].append("Warnung: Sicherheit gegen Bruch < 2 – Querschnitt vergrößern oder "
                                   "steiferes Material wählen")
    return format_result(title, res)


def _tool_kabel(strom_a: float, laenge_m: float, spannung: float, max_abfall_prozent: float = 3.0) -> str:
    return format_result("Kabel-Rechner (Kupfer, Hin- und Rückleiter)",
                         wire_size(strom_a, laenge_m, spannung, max_abfall_prozent))


def _tool_einheiten(wert: float, von: str, nach: str) -> str:
    res = unit_convert(wert, von, nach)
    head = f"{fmt_number(res['von_wert'], 10)} {res['von_einheit']} = {fmt_number(res['wert'], 10)} {res['einheit']}"
    return format_result(head, res, ("groesse", "formel", "annahmen"))


def _tool_masse(volumen_cm3: float, material: str) -> str:
    return format_result("Masse aus Volumen", mass_from_volume(volumen_cm3, material))


def engineering_tools() -> list:
    """Alle Ingenieur-Werkzeuge als :class:`obito.tools.Tool`-Objekte (nicht gefährlich)."""
    from .tools import Tool

    return [
        Tool(
            name="material_info",
            description="Kennwerte eines Werkstoffs (Dichte, Zugfestigkeit, E-Modul, Temperatur, Kosten, 3D-Druck, "
                        "Einsatz) – Richtwerte; tolerant bei Namen wie Carbon/CFK, Alu 7075, V2A, PETG, Harz.",
            parameters=_schema({"name": _p_str("Materialname, z. B. CFK, Alu 7075, Edelstahl 1.4301, PETG",
                                               "CFK")}, ["name"]),
            fn=_tool_material_info,
        ),
        Tool(
            name="materialien_vergleichen",
            description="Vergleicht mehrere Werkstoffe in einer Tabelle (Dichte, Festigkeit, Steifigkeit, Temperatur, "
                        "Kosten) und nennt das leichteste/festeste/steifeste.",
            parameters=_schema({"namen": _p_str("Materialnamen, kommagetrennt, z. B. »CFK, Alu 7075, PETG«",
                                                "CFK, Alu 6061, PETG")}, ["namen"]),
            fn=_tool_compare,
        ),
        Tool(
            name="akku_rechner",
            description="LiPo-Akku: Nennspannung und Energie (Wh) aus Zellenzahl und mAh; mit Strom auch Flugzeit, "
                        "mit C-Rate auch Belastungsprüfung.",
            parameters=_schema({"zellen": _p_int("Zellenzahl (S), 1–24", 4),
                                "mah": _p_number("Kapazität in mAh", 1500),
                                "strom_a": _p_number("Mittlerer Strom in A (optional, für Flugzeit)"),
                                "c_rate": _p_number("C-Rate des Akkus (optional, für Belastungsprüfung)")},
                               ["zellen", "mah"]),
            fn=_tool_akku,
        ),
        Tool(
            name="schub_rechner",
            description="Nötiger Gesamt- und Motorschub (g und N) für ein Abfluggewicht und Ziel-TWR; mit "
                        "vorhandenem Schub je Motor auch das erreichte Schub-Gewichts-Verhältnis.",
            parameters=_schema({"masse_g": _p_number("Abfluggewicht in g", 1200),
                                "motoren": _p_int("Anzahl Motoren", default=4),
                                "schub_je_motor_g": _p_number("Vorhandener max. Schub je Motor in g (optional)"),
                                "twr": _p_number("Ziel-Schub-Gewichts-Verhältnis", default=2.0)},
                               ["masse_g"]),
            fn=_tool_schub,
        ),
        Tool(
            name="elektro_rechner",
            description="Ohmsches Gesetz und Leistung: genau zwei von Spannung u (V), Strom i (A), Widerstand r (Ω), "
                        "Leistung p (W) angeben, die anderen werden berechnet.",
            parameters=_schema({"u": _p_number("Spannung in V"), "i": _p_number("Strom in A"),
                                "r": _p_number("Widerstand in Ω"), "p": _p_number("Leistung in W")}, []),
            fn=_tool_elektro,
        ),
        Tool(
            name="spannungsteiler",
            description="Unbelasteter Spannungsteiler: Ausgangsspannung, Strom und Verlustleistung aus U_ein, R1, R2.",
            parameters=_schema({"u_in": _p_number("Eingangsspannung in V", 12), "r1": _p_number("R1 in Ω (oben)", 10000),
                                "r2": _p_number("R2 in Ω (unten, an Masse)", 4700)}, ["u_in", "r1", "r2"]),
            fn=_tool_teiler,
        ),
        Tool(
            name="drehmoment_rechner",
            description="Drehmoment (Nm, kg·cm) aus Hebelarm und Kraft oder Masse, inkl. Sicherheitsreserve "
                        "(z. B. Servo-/Motorauswahl).",
            parameters=_schema({"hebel_m": _p_number("Hebelarm in m", 0.15), "kraft_n": _p_number("Kraft in N (oder Masse angeben)"),
                                "masse_kg": _p_number("Masse in kg (oder Kraft angeben)"),
                                "sicherheit": _p_number("Sicherheitsfaktor ≥ 1", default=1.5)}, ["hebel_m"]),
            fn=_tool_drehmoment,
        ),
        Tool(
            name="motor_rechner",
            description="Brushless-Motor: Drehzahl aus KV und Spannung (Leerlauf/Last); mit Propeller (Zoll) auch "
                        "geschätzter Standschub (±30 %).",
            parameters=_schema({"kv": _p_number("KV des Motors (U/min je V)", 2300), "spannung": _p_number("Spannung in V", 14.8),
                                "prop_zoll": _p_number("Propellerdurchmesser in Zoll (optional)"),
                                "steigung_zoll": _p_number("Propellersteigung in Zoll (optional)"),
                                "lastfaktor": _p_number("Drehzahl unter Last relativ zum Leerlauf (0–1)", default=0.85)},
                               ["kv", "spannung"]),
            fn=_tool_motor,
        ),
        Tool(
            name="balken_rechner",
            description="Kragbalken (Rechteckquerschnitt, Last am freien Ende): Durchbiegung und Biegespannung; "
                        "Material per Name (E-Modul aus Datenbank) oder E-Modul in GPa als Zahl.",
            parameters=_schema({"kraft_n": _p_number("Kraft am freien Ende in N", 10), "laenge_m": _p_number("Länge in m", 0.2),
                                "material": _p_str("Materialname (z. B. Alu 6061, CFK) oder E-Modul in GPa", "Alu 6061"),
                                "breite_m": _p_number("Breite in m", 0.02), "hoehe_m": _p_number("Höhe (in Lastrichtung) in m", 0.005)},
                               ["kraft_n", "laenge_m", "material", "breite_m", "hoehe_m"]),
            fn=_tool_balken,
        ),
        Tool(
            name="kabel_rechner",
            description="Kupferkabel: nötiger Querschnitt (mm²) und AWG für Strom, Länge und zulässigen Spannungsabfall "
                        "(Hin- und Rückleiter).",
            parameters=_schema({"strom_a": _p_number("Strom in A", 20), "laenge_m": _p_number("Einfache Länge in m", 1),
                                "spannung": _p_number("Systemspannung in V", 12),
                                "max_abfall_prozent": _p_number("Zulässiger Spannungsabfall in %", default=3)},
                               ["strom_a", "laenge_m", "spannung"]),
            fn=_tool_kabel,
        ),
        Tool(
            name="einheiten_umrechnen",
            description="Einheiten umrechnen: Länge, Masse, Kraft, Druck, Temperatur, Geschwindigkeit, Energie, Leistung "
                        "(z. B. 25 mm → in, 100 km/h → m/s, 20 °C → F, 2 PS → kW).",
            parameters=_schema({"wert": _p_number("Zahlenwert", 100), "von": _p_str("Ausgangseinheit", "km/h"),
                                "nach": _p_str("Zieleinheit", "m/s")}, ["wert", "von", "nach"]),
            fn=_tool_einheiten,
        ),
        Tool(
            name="masse_aus_volumen",
            description="Masse eines Bauteils aus Volumen (cm³, z. B. aus CAD) und Material (Dichte aus Datenbank).",
            parameters=_schema({"volumen_cm3": _p_number("Volumen in cm³", 12.5),
                                "material": _p_str("Materialname", "PETG")}, ["volumen_cm3", "material"]),
            fn=_tool_masse,
        ),
    ]


TOOL_NAMES = ("material_info", "materialien_vergleichen", "akku_rechner", "schub_rechner", "elektro_rechner",
              "spannungsteiler", "drehmoment_rechner", "motor_rechner", "balken_rechner", "kabel_rechner",
              "einheiten_umrechnen", "masse_aus_volumen")


def register_tools(registry: "ToolRegistry") -> None:
    """Registriert alle zwölf Ingenieur-Werkzeuge (nicht gefährlich) in der Registry."""
    for tool in engineering_tools():
        registry.register(tool)


__all__ = [
    "CATEGORIES", "CELL_NOMINAL_V", "COPPER_RESISTIVITY", "COST_CLASSES", "G", "MATERIALS", "RICHTWERT_HINWEIS",
    "TOOL_NAMES", "Material", "awg_area_mm2", "awg_diameter_mm", "awg_for_area", "awg_label",
    "battery_c_check", "beam_cantilever", "compare_materials", "engineering_tools", "find_material",
    "flight_time", "fmt_number", "format_result", "lipo_energy", "list_materials", "mass_from_volume",
    "material_info", "material_table", "motor_rpm", "ohm", "prop_static_thrust", "register_tools",
    "required_thrust", "thrust_to_weight", "torque", "unit_convert", "voltage_divider", "wire_size",
]
