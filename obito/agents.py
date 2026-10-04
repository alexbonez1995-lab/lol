"""Experten, Prompts und Normalisierer von OBITO.

Dieses Modul kennt kein Modell und keine Datenbank. Es liefert ausschließlich:

* die **Experten** (Personas) des Gremiums, den KRITIKER und OMEGA,
* die **Prompt-Bauer** je Stufe (``*_messages``), deren erste System-Nachricht in der
  letzten Zeile den Stufen-Marker ``[OBITO:<stage>:<who>]`` trägt,
* das **Zeichen-/Token-Budget** (``clip``, ``estimate_tokens``, ``fit_messages``),
* die **JSON-Schemata** der strukturierten Stufen und ihre toleranten **Normalisierer**
  (``parse_*``), die auch das unzuverlässige JSON kleiner Modelle verwerten.

Alle Texte an das Modell und an den Nutzer sind deutsch.
"""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from .llm import parse_json
from .memory import KINDS, Memory

if TYPE_CHECKING:  # pragma: no cover – nur für Typprüfer, zur Laufzeit Duck-Typing
    from .learning import Lesson
    from .tools import ToolResult

# ------------------------------------------------------------------ Budgets
MAX_CONTEXT_CHARS = 3000          # gesamter Kontextblock (Projekt + Lektionen + Erinnerungen + extra)
MAX_MEMORY_CONTEXT_CHARS = 1200   # Anteil Erinnerungen im Kontextblock
MAX_LESSON_CONTEXT_CHARS = 800    # Anteil Lektionen im Kontextblock
MAX_LESSON_CHARS = 300            # je Lektion
MAX_HISTORY_CHARS = 3000          # Gesprächsverlauf im Prompt
MAX_HISTORY_ASSISTANT_CHARS = 600  # Assistant-Nachrichten im Verlauf, wenn das Budget knapp wird
MAX_EXAMPLE_CHARS = 400           # je Beispielantwort (Few-Shot)
MAX_ANSWER_IN_PROMPT = 2500       # je Expertenantwort in Kritiker/Revision/Synthese
MAX_CRITIQUE_CHARS = 1500         # Kritik-Text in Revision/Synthese
MAX_QUESTION_CHARS = 6000         # Frage in JSON-Stufen (Extraktion, Lektion, Richter)
MIN_LAST_USER_CHARS = 200         # Kopf der letzten Nutzer-Nachricht, der immer bleibt

CLIP_SUFFIX = " … [gekürzt]"

STAGE_TAG = "[OBITO:{stage}:{who}]"
_STAGE_RE = re.compile(r"\[OBITO:(\w+):([\w-]+)\]\s*$")
_WORD = re.compile(r"\w+", re.UNICODE)

STAGES = ("routing", "schnell", "experte", "kritiker", "revision", "synthese",
          "extraktion", "lektion", "korrektur", "richter")


# ------------------------------------------------------------------ Experten
@dataclass(frozen=True)
class Expert:
    """Ein Fachexperte des Gremiums.

    ``system_prompt`` ist die Persona (steht bei Experten in der Nutzer-Nachricht, damit der
    System-Präfix für alle Experten identisch bleibt); ``keywords`` sind kleingeschrieben und
    umlautfrei (``ae/oe/ue/ss``) und werden als Wortanfang gegen die Frage geprüft."""

    id: str
    name: str
    role: str
    system_prompt: str
    keywords: tuple[str, ...]
    temperature: float = 0.4


EXPERTS: dict[str, Expert] = {
    "ALPHA": Expert(
        id="ALPHA",
        name="3D-Konstruktion",
        role="CAD, Mechanik, Fertigung (3D-Druck, Fräsen, Blech)",
        system_prompt=(
            "Du bist ALPHA, Konstrukteurin mit 15 Jahren Erfahrung in Maschinenbau, CAD (Fusion 360, "
            "FreeCAD, SolidWorks) und additiver Fertigung. Du denkst in Baugruppen, Lastpfaden, "
            "Toleranzen und Fertigbarkeit: Wandstärken, Druckrichtung, Passungen, Schraub- und "
            "Klebeverbindungen. Du nennst Maße in mm, Gewichte in g, Drehmomente in Nm und sagst "
            "klar, was gedruckt, gefräst oder gekauft werden sollte – und warum."
        ),
        keywords=(
            "cad", "konstruktion", "konstruier", "3d-druck", "3d druck", "3d-drucker", "drucker",
            "fusion 360", "freecad", "solidworks", "stl", "step-datei", "toleranz", "passung",
            "gehaeuse", "halterung", "halter", "rahmen", "chassis", "schraube", "gewinde", "mutter",
            "lager", "kugellager", "getriebe", "zahnrad", "welle", "achse", "mechanik", "mechanisch",
            "bauteil", "baugruppe", "wandstaerke", "fdm", "sla", "filament", "spritzguss", "fraes",
            "drehteil", "blech", "zeichnung", "modellier", "scharnier", "gelenk", "feder", "adapter",
            "montage", "verschraub",
        ),
        temperature=0.4,
    ),
    "BETA": Expert(
        id="BETA",
        name="Elektronik & Steuerung",
        role="Schaltungen, Mikrocontroller, Motoren, Akkus, Regelung",
        system_prompt=(
            "Du bist BETA, Elektronik-Ingenieur mit Schwerpunkt Embedded-Systeme, Leistungselektronik "
            "und Regelungstechnik. Du rechnest Spannungen, Ströme, Leistungen und Verluste nach, "
            "wählst Bauteile nach Datenblatt (Spannungsfestigkeit, Dauerstrom, Wirkungsgrad) und "
            "kennst die Fallen der Praxis: Masseführung, Entstörung, Spannungseinbrüche, LiPo-Sicherheit. "
            "Du gibst konkrete Bauteilklassen, Werte mit Einheiten und Pinbelegungen an."
        ),
        keywords=(
            "elektronik", "elektrisch", "schaltung", "schaltplan", "platine", "pcb", "arduino", "esp32",
            "esp8266", "raspberry", "mikrocontroller", "stm32", "attiny", "motor", "servo", "esc",
            "brushless", "schrittmotor", "akku", "lipo", "batterie", "zelle", "spannung", "strom",
            "ampere", "volt", "widerstand", "kondensator", "mosfet", "transistor", "diode", "pwm",
            "i2c", "spi", "uart", "regler", "pid", "steuerung", "steuern", "relais", "netzteil",
            "loet", "verkabel", "kabel", "stecker", "led", "firmware", "flugcontroller", "betaflight",
            "sensorwert", "adc", "oszilloskop", "multimeter", "kurzschluss", "sicherung",
        ),
        temperature=0.35,
    ),
    "GAMMA": Expert(
        id="GAMMA",
        name="Physik & Simulation",
        role="Mechanik, Thermik, Strömung, Festigkeit, Überschlagsrechnungen",
        system_prompt=(
            "Du bist GAMMA, Physikerin und Simulationsingenieurin. Du führst jede Behauptung auf eine "
            "Größengleichung zurück, rechnest Überschläge mit Einheiten vor (SI, Zwischenschritte "
            "sichtbar) und nennst Annahmen, Gültigkeitsgrenzen und Sicherheitsfaktoren. Du kennst "
            "Festigkeitslehre, Dynamik, Wärmeübertragung, Aerodynamik und weißt, wann eine FEM/CFD "
            "sinnvoll ist und wann die Handrechnung reicht."
        ),
        keywords=(
            "physik", "physikalisch", "kraft", "kraefte", "drehmoment", "newton", "energie", "leistung",
            "watt", "joule", "geschwindigkeit", "beschleunigung", "reibung", "schwerpunkt",
            "traegheit", "schwingung", "resonanz", "frequenz", "waerme", "temperatur", "kuehlung",
            "thermisch", "thermodynamik", "stroemung", "aerodynamik", "auftrieb", "luftwiderstand",
            "druck", "fem", "simulation", "simulier", "cfd", "festigkeit", "biegung", "biegemoment",
            "verformung", "durchbiegung", "knick", "schub", "impuls", "magnet", "optik", "formel",
            "berechnung", "hebel", "masse", "gewichtskraft", "fallhoehe", "flugzeit", "reichweite",
        ),
        temperature=0.3,
    ),
    "DELTA": Expert(
        id="DELTA",
        name="Code & Software",
        role="Programmierung, Debugging, Architektur, Werkzeuge",
        system_prompt=(
            "Du bist DELTA, Senior-Softwareentwickler (Python, C/C++, JavaScript/TypeScript, Rust, "
            "Shell, SQL). Du schreibst lauffähigen, getesteten Code mit klaren Namen und erklärst "
            "Fehlermeldungen Zeile für Zeile. Du nennst Versionen, Abhängigkeiten und Randfälle, "
            "bevorzugst einfache Lösungen vor cleveren und weist auf Sicherheits- und "
            "Performance-Fallen hin. Code immer in Markdown-Zäunen mit Sprachangabe."
        ),
        keywords=(
            "code", "programm", "programmier", "python", "javascript", "typescript", "java", "c++",
            "c#", "rust", "golang", "bash", "powershell", "sql", "skript", "script", "funktion",
            "klasse", "methode", "api", "bug", "fehlermeldung", "exception", "traceback", "debug",
            "kompilier", "compiler", "bibliothek", "library", "framework", "docker", "git", "github",
            "linux", "windows", "software", "algorithmus", "datenbank", "sqlite", "server", "html",
            "css", "regex", "json", "unittest", "pytest", "variable", "schleife", "array", "liste",
            "import", "modul", "paket", "pip", "npm", "terminal", "kommandozeile", "ide", "vscode",
            "refactor", "repository",
        ),
        temperature=0.3,
    ),
    "EPSILON": Expert(
        id="EPSILON",
        name="Datenanalyse",
        role="Statistik, Messreihen, Auswertung, Machine Learning",
        system_prompt=(
            "Du bist EPSILON, Data Scientist mit Statistik-Hintergrund. Du fragst zuerst nach "
            "Datenqualität, Stichprobengröße und Messunsicherheit, wählst dann die passende Methode "
            "(deskriptiv, Regression, Test, Modell) und erklärst, was das Ergebnis bedeutet – und was "
            "nicht. Du nennst Kennzahlen mit Einheit und Unsicherheit, warnst vor Scheinkorrelation "
            "und Überanpassung und schlägst eine konkrete Auswertung (z. B. pandas/numpy) vor."
        ),
        keywords=(
            "daten", "datensatz", "csv", "excel", "tabelle", "statistik", "statistisch", "mittelwert",
            "median", "standardabweichung", "varianz", "korrelation", "regression", "verteilung",
            "wahrscheinlichkeit", "diagramm", "plot", "visualisier", "pandas", "numpy", "auswert",
            "auswertung", "messreihe", "messwert", "messdaten", "ausreisser", "stichprobe",
            "signifikan", "hypothese", "machine learning", "maschinelles lernen", "trainier",
            "klassifik", "cluster", "prognose", "vorhersage", "trend", "kennzahl", "histogramm",
            "zeitreihe", "logdaten", "sensordaten", "filter", "glaett", "mittel",
        ),
        temperature=0.3,
    ),
    "ZETA": Expert(
        id="ZETA",
        name="Vision & Sensorik",
        role="Kameras, Bildverarbeitung, Sensoren, Sensorfusion",
        system_prompt=(
            "Du bist ZETA, Expertin für Bildverarbeitung und Sensorik (OpenCV, Kamera-Kalibrierung, "
            "IMU/GPS/Lidar/Ultraschall, Kalman-Filter). Du bewertest Sensoren nach Auflösung, "
            "Abtastrate, Rauschen, Latenz, Reichweite und Umgebungsbedingungen (Licht, Vibration, "
            "Temperatur) und entwirfst robuste Verarbeitungsketten vom Rohsignal bis zur Entscheidung. "
            "Du nennst konkrete Sensor-Klassen, Schnittstellen und typische Fehlerquellen."
        ),
        keywords=(
            "kamera", "bild", "video", "opencv", "vision", "bilderkennung", "bildverarbeitung",
            "objekterkennung", "gesichtserkennung", "yolo", "lidar", "ultraschall", "infrarot",
            "radar", "imu", "gyroskop", "gyro", "beschleunigungssensor", "gps", "gnss", "sensorfusion",
            "kalman", "kalibrier", "aufloesung", "belichtung", "objektiv", "brennweite", "tiefenkamera",
            "stereo", "tracking", "slam", "abstandssensor", "lichtschranke", "encoder", "sensorik",
            "sensor", "barometer", "magnetometer", "kompass", "qr-code", "barcode", "ocr", "lidarscan",
            "framerate", "fps", "pixel",
        ),
        temperature=0.35,
    ),
    "THETA": Expert(
        id="THETA",
        name="Optimierung",
        role="Zielkonflikte, Auslegung, Effizienz, Kosten/Gewicht/Leistung",
        system_prompt=(
            "Du bist THETA, Optimierungsexperte. Du machst Zielfunktion und Nebenbedingungen explizit "
            "(Gewicht, Kosten, Zeit, Leistung, Sicherheit), zeigst Zielkonflikte als Zahlen "
            "(Trade-off-Tabelle), suchst den größten Hebel und schlägst eine iterative Vorgehensweise "
            "mit Messgrößen vor. Du warnst vor Scheinoptimierung an der falschen Stelle und sagst, "
            "wann ein Kompromiss gut genug ist."
        ),
        keywords=(
            "optimier", "optimal", "minimier", "maximier", "effizien", "gewicht reduzier", "leichter",
            "schneller", "guenstiger", "billiger", "kosten senk", "sparen", "trade-off", "tradeoff",
            "kompromiss", "zielkonflikt", "parameter", "auslegung", "ausleg", "dimensionier",
            "verbessern", "verbesser", "tuning", "benchmark", "performance", "durchsatz", "latenz",
            "zielfunktion", "nebenbedingung", "pareto", "topologie", "iterativ", "feinabstimmung",
            "wirkungsgrad", "bestmoeglich", "ideal", "maximal", "minimal", "reduzier", "steiger",
            "laufzeit", "skalier",
        ),
        temperature=0.35,
    ),
    "IOTA": Expert(
        id="IOTA",
        name="Materialien",
        role="Werkstoffwahl, Kennwerte, Verbinden, Beständigkeit",
        system_prompt=(
            "Du bist IOTA, Werkstoffingenieurin. Du vergleichst Materialien anhand von Kennwerten "
            "(Dichte in g/cm³, E-Modul in GPa, Zugfestigkeit in MPa, Wärmeformbeständigkeit in °C, "
            "Preis), berücksichtigst Verarbeitung (Druck, Fräsen, Kleben, Schweißen) und Umgebung "
            "(UV, Feuchte, Temperatur, Chemie). Du nennst typische Wertebereiche, empfiehlst Alternativen "
            "und warnst vor Mischpaarungen (Korrosion, Ausdehnung) und Gesundheitsrisiken (Stäube, Dämpfe)."
        ),
        keywords=(
            "material", "werkstoff", "carbon", "kohlefaser", "cfk", "gfk", "glasfaser", "aluminium",
            "alu", "stahl", "edelstahl", "titan", "messing", "kupfer", "kunststoff", "pla", "petg",
            "abs", "asa", "tpu", "nylon", "pa12", "pa6", "polycarbonat", "pc", "pom", "harz", "epoxid",
            "holz", "sperrholz", "legierung", "steifigkeit", "haerte", "zugfestigkeit", "dichte",
            "e-modul", "korrosion", "rost", "uv-bestaendig", "hitzebestaendig", "temperaturbestaendig",
            "klebstoff", "kleben", "schweiss", "oberflaeche", "beschichtung", "lackier", "eloxier",
            "gummi", "silikon", "schaum", "glas", "keramik", "faser", "verbundwerkstoff",
        ),
        temperature=0.3,
    ),
    "KAPPA": Expert(
        id="KAPPA",
        name="Recherche & Fakten",
        role="Faktenprüfung, Normen, Datenblätter, Stand der Technik",
        system_prompt=(
            "Du bist KAPPA, Fachrechercheur. Du trennst gesichertes Wissen scharf von Vermutung, "
            "nennst Normen, Datenblattwerte und Größenordnungen nur, wenn du sie kennst, und "
            "markierst alles andere mit „Unsicher:“. Du erklärst Begriffe präzise, ordnest Aussagen "
            "in den Stand der Technik ein und sagst konkret, wo und wie der Nutzer die Angabe "
            "selbst verifizieren kann (Datenblatt, Norm, Herstellerseite, Messung)."
        ),
        keywords=(
            "recherch", "quelle", "beleg", "fakt", "faktencheck", "definition", "geschichte",
            "hintergrund", "norm", "din", "iso", "vde", "vorschrift", "gesetz", "richtlinie",
            "zulassung", "ce-kennzeichnung", "datenblatt", "spezifikation", "hersteller",
            "ueberblick", "literatur", "studie", "stand der technik", "begriff", "wikipedia",
            "stimmt es", "ist es wahr", "pruefe", "ueberpruef", "nachschlag", "fachbegriff",
            "historie", "entwicklung", "patent", "lizenz", "standard", "versicherung", "haftung",
        ),
        temperature=0.25,
    ),
    "LAMBDA": Expert(
        id="LAMBDA",
        name="Planung & Projekte",
        role="Vorgehen, Zeitplan, Stückliste, Risiken, Beschaffung",
        system_prompt=(
            "Du bist LAMBDA, Projektleiterin für technische Entwicklungsprojekte. Du zerlegst Vorhaben "
            "in klare Arbeitspakete mit Reihenfolge, Abhängigkeiten, Aufwand (Stunden/Tage) und "
            "Kosten (€), definierst Meilensteine mit prüfbaren Ergebnissen, führst eine Stückliste und "
            "benennst die drei größten Risiken samt Gegenmaßnahme. Du planst Puffer ein und sagst, "
            "was man zuerst ausprobieren sollte, um teure Irrwege früh zu erkennen."
        ),
        keywords=(
            "plan", "planung", "projekt", "zeitplan", "meilenstein", "schritte", "schritt-fuer-schritt",
            "vorgehen", "vorgehensweise", "roadmap", "aufgaben", "priorit", "budget", "kosten",
            "aufwand", "ressourcen", "risiko", "termin", "deadline", "sprint", "kanban", "scrum",
            "organisier", "struktur", "checkliste", "stueckliste", "bom", "beschaffung", "bestell",
            "lieferzeit", "reihenfolge", "strategie", "konzept", "anforderung", "lastenheft",
            "pflichtenheft", "ablauf", "phasen", "zeitaufwand", "wie fange ich an", "wo anfangen",
            "vorbereit", "einkaufsliste",
        ),
        temperature=0.4,
    ),
    "GENERALIST": Expert(
        id="GENERALIST",
        name="Allgemeinwissen",
        role="Alltag, Sprache, Erklärungen, Einordnung",
        system_prompt=(
            "Du bist GENERALIST, belesener Allrounder mit gesundem Menschenverstand. Du erklärst "
            "Dinge verständlich, ohne zu vereinfachen, formulierst Texte in gutem Deutsch, bringst "
            "Alltags- und Praxiswissen ein und achtest darauf, dass die Antwort wirklich auf die "
            "Frage des Nutzers passt – inklusive dem, was er wahrscheinlich eigentlich wissen will. "
            "Bei Fachfragen benennst du die Grenzen deines Wissens offen."
        ),
        keywords=(
            "alltag", "sprache", "text", "formulier", "uebersetz", "grammatik", "rechtschreib", "brief",
            "e-mail", "email", "bewerbung", "lebenslauf", "rezept", "kochen", "backen", "reise", "urlaub",
            "gesundheit", "ernaehrung", "tipp", "empfehlung", "meinung", "geschichte erzaehl", "gedicht",
            "witz", "hallo", "hi ", "hey", "guten morgen", "guten tag", "guten abend", "wie geht",
            "danke", "zusammenfass", "erklaer", "einfach erklaert", "kinder", "familie", "freizeit",
            "sport", "musik", "film", "buch", "geschenk", "haushalt", "garten", "tier", "hund", "katze",
            "wetter", "feiertag", "wochenende", "spiel", "name", "idee",
        ),
        temperature=0.5,
    ),
}

CRITIC = Expert(
    id="KRITIKER",
    name="Kritiker",
    role="Prüft Expertenantworten auf Fehler, Widersprüche und Lücken",
    system_prompt=(
        "Du bist KRITIKER, unabhängiger Gutachter des OBITO-Gremiums. Du suchst gezielt nach "
        "Rechenfehlern, falschen Einheiten, physikalisch unmöglichen Werten, erfundenen Fakten, "
        "unbegründeten Annahmen, Sicherheitslücken und Widersprüchen zwischen den Experten. Du "
        "lobst nicht, du schmeichelst nicht, du bewertest nur: Jeder Befund nennt den Experten, das "
        "konkrete Problem, eine Korrektur und die Schwere. Was fehlt, benennst du ebenso klar wie "
        "das, was falsch ist. Bist du dir bei einem Punkt nicht sicher, sagst du das."
    ),
    keywords=(),
    temperature=0.2,
)

OMEGA = Expert(
    id="OMEGA",
    name="Koordination & Synthese",
    role="Führt die Expertenmeinungen zu einer geprüften Antwort zusammen",
    system_prompt=(
        "Du bist OBITO (Instanz OMEGA), ein lokaler, lernfähiger KI-Assistent, der direkt mit dem "
        "Nutzer spricht. Du antwortest wie ein erfahrener Kollege: klar, ehrlich, konkret, mit Zahlen "
        "und Einheiten, ohne Füllsätze. Du führst die Perspektiven des Expertengremiums und die "
        "Kritik zu einer einzigen, in sich stimmigen Antwort zusammen, entscheidest bei Widersprüchen "
        "begründet und nennst verbleibende Unsicherheiten offen. Was der Nutzer dir früher gesagt "
        "hat (Erinnerungen, Lektionen), beachtest du zuverlässig."
    ),
    keywords=(),
    temperature=0.4,
)

BASE_RULES = (
    "Du bist Teil von OBITO, einem lokalen KI-Assistenten. Verbindliche Regeln:\n"
    "1. Antworte auf Deutsch, präzise und konkret; keine Floskeln, keine Wiederholung der Frage.\n"
    "2. Zahlen immer mit Einheiten (mm, g, V, A, W, Nm, °C, €, s); Annahmen ausdrücklich nennen "
    "und Rechenwege kurz zeigen.\n"
    "3. Erfinde keine Fakten, Quellen, Teilenummern oder Messwerte. Was du nicht sicher weißt, "
    "markierst du mit „Unsicher:“ und sagst, wie man es prüfen kann.\n"
    "4. Erinnerungen und Lektionen im Kontext sind verbindlich: befolge sie und weiche nicht "
    "stillschweigend davon ab.\n"
    "5. Fehlen entscheidende Informationen, stelle höchstens drei gezielte Rückfragen – liefere "
    "trotzdem eine belastbare Antwort mit begründeten Annahmen.\n"
    "6. Sicherheit zuerst: Warne klar vor Gefahren (Netzspannung, LiPo-Akkus, Laser, rotierende "
    "Teile, Chemikalien, Stäube) und nenne Schutzmaßnahmen; keine Anleitung zu Gefährlichem oder "
    "Illegalem.\n"
    "7. Struktur: wichtigste Aussage zuerst, kurze Absätze oder Listen, Code in Markdown-Zäunen."
)

# Reihenfolge zum Auffüllen von select_experts (Rest in EXPERTS-Reihenfolge)
FILL_ORDER = ("GENERALIST", "KAPPA", "DELTA", "GAMMA", "ALPHA", "BETA", "EPSILON", "ZETA", "THETA",
              "IOTA", "LAMBDA")

TOOL_INSTRUCTION = (
    "Werkzeuge: Du kannst die unten aufgeführten Werkzeuge nutzen. Rufe ein Werkzeug exakt in dieser "
    "Form auf und beende danach deine Ausgabe – das Ergebnis bekommst du als nächste Nachricht:\n"
    '<werkzeug>{"name": "rechnen", "args": {"ausdruck": "2*21"}}</werkzeug>\n'
    "Nutze Werkzeuge nur, wenn sie die Antwort wirklich verbessern (Rechnen, Datum/Uhrzeit, Dateien, "
    "System). Übernimm Werkzeug-Ergebnisse wörtlich, rate nie ein Ergebnis."
)


# ------------------------------------------------------------- Text-Hilfen
_UMLAUTS = str.maketrans({"ä": "ae", "ö": "oe", "ü": "ue", "ß": "ss", "Ä": "ae", "Ö": "oe", "Ü": "ue"})


def _fold(text: Any) -> str:
    """Kleinschreibung, Umlaute → ae/oe/ue/ss, sonstige Diakritika entfernt."""
    if text is None:
        return ""
    s = str(text).translate(_UMLAUTS).lower()
    s = unicodedata.normalize("NFKD", s)
    return "".join(ch for ch in s if not unicodedata.combining(ch))


def _key(text: Any) -> str:
    """Normalisierter Schlüsselname: ``Gilt für`` → ``gilt_fuer``."""
    return re.sub(r"[^a-z0-9]+", "_", _fold(text)).strip("_")


def _text(value: Any) -> str:
    """Beliebigen Wert als bereinigten String (None → '')."""
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def clip(text: str, n: int) -> str:
    """Kürzt ``text`` auf ``n`` Zeichen und hängt ``" … [gekürzt]"`` an."""
    text = "" if text is None else str(text)
    n = max(0, int(n))
    if len(text) <= n:
        return text
    return text[:n].rstrip() + CLIP_SUFFIX


def estimate_tokens(text: str) -> int:
    """Konservative Token-Schätzung für deutsche Texte: ``len(text) // 3 + 1``."""
    return len(text or "") // 3 + 1


def _message_tokens(messages: Iterable[dict]) -> int:
    return sum(estimate_tokens(m.get("content") or "") for m in messages)


def fit_messages(messages: Sequence[dict], budget_tokens: int, keep_last_user: bool = True) -> list[dict]:
    """Passt eine Nachrichtenliste in ein Token-Budget.

    Reihenfolge der Maßnahmen:
    1. älteste Verlaufs-/Beispiel-Nachrichten zwischen System-Nachricht und letzter
       Nutzer-Nachricht entfernen (der jüngste Austausch – zwei Nachrichten – bleibt zunächst),
    2. verbleibende Assistant-Nachrichten im Verlauf auf 600 Zeichen kürzen,
    3. reicht das nicht, auch den Rest des Verlaufs entfernen,
    4. die letzte Nutzer-Nachricht kürzen (Kopf behalten, mindestens 200 Zeichen).

    Die System-Nachricht wird nie entfernt oder gekürzt; mit ``keep_last_user=True`` bleibt die
    letzte Nutzer-Nachricht immer erhalten. Es wird eine neue Liste (Kopien) zurückgegeben."""
    msgs: list[dict] = []
    for m in messages:
        d = dict(m)
        if not isinstance(d.get("content"), str):
            d["content"] = _text(d.get("content"))
        msgs.append(d)
    budget = max(0, int(budget_tokens))
    if _message_tokens(msgs) <= budget:
        return msgs

    head: list[dict] = msgs[:1] if msgs and msgs[0].get("role") == "system" else []
    rest = msgs[len(head):]
    last_user = None
    if keep_last_user:
        for i in range(len(rest) - 1, -1, -1):
            if rest[i].get("role") == "user":
                last_user = i
                break
    if last_user is None:
        middle, tail = rest, []
    else:
        middle, tail = rest[:last_user], rest[last_user:]

    def over() -> bool:
        return _message_tokens(head) + _message_tokens(middle) + _message_tokens(tail) > budget

    # 1. älteste Nachrichten im Mittelteil entfernen, der jüngste Austausch bleibt zunächst
    while over() and len(middle) > 2:
        middle.pop(0)

    # 2. Assistant-Nachrichten im Verlauf kürzen (älteste zuerst)
    if over():
        for m in middle:
            if m.get("role") == "assistant" and len(m["content"]) > MAX_HISTORY_ASSISTANT_CHARS:
                m["content"] = clip(m["content"], MAX_HISTORY_ASSISTANT_CHARS)
                if not over():
                    break

    # 3. reicht das nicht: restlichen Verlauf entfernen
    while over() and middle:
        middle.pop(0)

    # 4. letzte Nutzer-Nachricht kürzen (Kopf behalten)
    target = tail[0] if tail else (middle[-1] if middle else None)
    if over() and target is not None:
        others = _message_tokens(head) + _message_tokens(middle) + _message_tokens(tail) \
            - estimate_tokens(target["content"])
        remaining = budget - others
        allowed = (remaining - 1) * 3 - len(CLIP_SUFFIX)
        allowed = max(MIN_LAST_USER_CHARS, allowed)
        if len(target["content"]) > allowed:
            target["content"] = clip(target["content"], allowed)
    return head + middle + tail


# ------------------------------------------------------------- Marker
def _system(stage: str, who: str, body: str) -> dict:
    """System-Nachricht, deren letzte Zeile exakt der Stufen-Marker ist."""
    body = (body or "").rstrip()
    tag = STAGE_TAG.format(stage=stage, who=who)
    return {"role": "system", "content": (body + "\n" + tag) if body else tag}


def stage_of(messages: Sequence[dict]) -> tuple[str, str]:
    """Liest ``(stage, who)`` aus dem Marker der ersten System-Nachricht, sonst ``("?", "?")``."""
    try:
        for m in messages:
            if isinstance(m, dict) and m.get("role") == "system":
                content = m.get("content")
                if isinstance(content, str):
                    hit = _STAGE_RE.search(content)
                    if hit:
                        return hit.group(1), hit.group(2)
                return "?", "?"
    except TypeError:
        pass
    return "?", "?"


# ------------------------------------------------------------- Experten-Wahl
def _keyword_scores(question: str) -> dict[str, float]:
    """Keyword-Treffer je Experte (Wortanfang in der umlautgefalteten Frage)."""
    folded = " " + re.sub(r"\s+", " ", _fold(question)) + " "
    scores: dict[str, float] = {}
    for eid, ex in EXPERTS.items():
        score = 0.0
        for kw in ex.keywords:
            k = _fold(kw).strip()
            if not k:
                continue
            # kurze Schlüsselwörter (pla, abs, alu …) nur als ganzes Wort, längere als Wortanfang
            pattern = r"(?<![a-z0-9])" + re.escape(k) + (r"(?![a-z0-9])" if len(k) <= 4 else "")
            if re.search(pattern, folded):
                score += 1.0 + min(len(k), 20) / 20.0
        if score > 0:
            scores[eid] = score
    return scores


def resolve_expert_id(value: Any, known: Iterable[str] | None = None) -> str | None:
    """Löst eine Experten-Angabe (ID, Name, „Experte ALPHA“) auf eine bekannte ID auf."""
    if value is None or isinstance(value, (bool, int, float, list, dict)):
        return None
    raw = str(value).strip()
    if not raw:
        return None
    known_ids = [k.upper() for k in (known if known is not None else EXPERTS.keys())]
    upper = raw.upper()
    if upper in known_ids:
        return upper
    tokens = re.findall(r"[A-Z][A-Z0-9-]*", upper)
    for t in tokens:
        if t in known_ids:
            return t
    folded = _fold(raw)
    for kid in known_ids:
        ex = EXPERTS.get(kid)
        if ex is None:
            continue
        name = _fold(ex.name)
        if folded == name or (len(folded) >= 4 and folded in name) or (len(name) >= 4 and name in folded):
            return kid
    return None


def _resolve_expert_ids(value: Any, known: Iterable[str] | None = None) -> list[str]:
    if isinstance(value, str):
        parts = [p for p in re.split(r"[,;/\n]+", value) if p.strip()]
        items: list[Any] = parts if len(parts) > 1 else [value]
    elif isinstance(value, dict):
        items = list(value.keys())
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = []
    known_list = list(known) if known is not None else list(EXPERTS.keys())
    out: list[str] = []
    for item in items:
        if isinstance(item, dict):
            item = item.get("id") or item.get("experte") or item.get("name")
        eid = resolve_expert_id(item, known_list)
        if eid and eid not in out:
            out.append(eid)
    return out


def select_experts(question: str, k: int = 5, hint: list[str] | None = None) -> list[Expert]:
    """Wählt ``k`` Experten: erst ``hint`` (unbekannte ignoriert), dann Keyword-Treffer nach Score,
    dann Auffüllen in ``FILL_ORDER``. Nie Duplikate, nie KRITIKER/OMEGA."""
    k = max(0, int(k))
    chosen: list[str] = []
    for h in hint or ():
        eid = resolve_expert_id(h, EXPERTS.keys())
        if eid and eid not in chosen and len(chosen) < k:
            chosen.append(eid)
    scores = _keyword_scores(question or "")
    order = {eid: i for i, eid in enumerate(FILL_ORDER)}
    ranked = sorted(scores, key=lambda e: (-scores[e], order.get(e, 99)))
    for eid in ranked:
        if len(chosen) >= k:
            break
        if eid not in chosen:
            chosen.append(eid)
    for eid in list(FILL_ORDER) + [e for e in EXPERTS if e not in FILL_ORDER]:
        if len(chosen) >= k:
            break
        if eid not in chosen:
            chosen.append(eid)
    return [EXPERTS[e] for e in chosen[:k] if e in EXPERTS]


_STRONG_VERBS = re.compile(
    r"(?<![a-z])(entwirf\w*|entwerf\w*|vergleich\w*|berechn\w*|optimier\w*|plan(?:e|en|t|ung|st)?|"
    r"analysier\w*|konzipier\w*|dimensionier\w*|ausleg\w*)(?![a-z])"
)
_WHY = re.compile(r"(?<![a-z])(warum|weshalb|wieso)(?![a-z])")
_SMALLTALK = re.compile(
    r"(?<![a-z])(hallo|hi|hey|moin|servus|guten (morgen|tag|abend)|wie geht|danke|tschuess|"
    r"bis spaeter|gute nacht|wer bist du|was kannst du)(?![a-z])"
)


def heuristic_complexity(question: str) -> str:
    """Schätzt die Komplexität einer Frage: ``"einfach" | "mittel" | "komplex"``.

    komplex: ≥ 2 Fragezeichen, Aufforderungen wie entwirf/vergleiche/berechne/optimiere/plane/
    analysiere (bei mindestens 6 Wörtern oder Fachbezug), „warum“ mit Fachbezug oder ≥ 12 Wörtern,
    lange Fragen (≥ 40 Wörter oder ≥ 350 Zeichen), ≥ 3 Fachgebiete.
    einfach: < 12 Wörter ohne Fachwörter, oder Smalltalk ohne Fachwörter. Sonst mittel."""
    q = (question or "").strip()
    if not q:
        return "einfach"
    words = _WORD.findall(q)
    n = len(words)
    folded = _fold(q)
    domains = [eid for eid in _keyword_scores(q) if eid != "GENERALIST"]
    n_questions = q.count("?")
    strong = _STRONG_VERBS.search(folded) is not None
    if n_questions >= 2 or n >= 40 or len(q) >= 350 or len(domains) >= 3:
        return "komplex"
    if strong and (n >= 6 or domains):
        return "komplex"
    if _WHY.search(folded) and (domains or n >= 12):
        return "komplex"
    if not domains and (n < 12 or (_SMALLTALK.search(folded) and n < 20)):
        return "einfach"
    return "mittel"


# ------------------------------------------------------------- Kontext
def _render_lesson(lesson: Any) -> str:
    render = getattr(lesson, "render", None)
    text = ""
    if callable(render):
        try:
            text = _text(render())
        except Exception:
            text = ""
    if not text:
        text = _text(getattr(lesson, "rule", None)) or _text(lesson)
    return clip(" ".join(text.split()), MAX_LESSON_CHARS)


def _bulleted(lines: Iterable[str], budget: int) -> list[str]:
    """Aufzählungszeilen bis zum Zeichenbudget; die letzte wird ggf. gekürzt."""
    out: list[str] = []
    used = 0
    for line in lines:
        line = "- " + line
        if used + len(line) + 1 > budget:
            rest = budget - used - 1
            if rest >= 60:
                out.append(clip(line, rest - len(CLIP_SUFFIX)))
            break
        out.append(line)
        used += len(line) + 1
    return out


def context_block(memories: list[Memory], lessons: list["Lesson"], project: str | None, extra: str = "") -> str:
    """Kontextblock für System-Nachrichten (Budget ``MAX_CONTEXT_CHARS``).

    Abschnitte in dieser Reihenfolge: verbindliche Lektionen (direkt nach den Regeln),
    Projekt, Erinnerungen („- [art] inhalt“), zusätzlicher Text."""
    parts: list[str] = []
    if lessons:
        lines = _bulleted((_render_lesson(l) for l in lessons), MAX_LESSON_CONTEXT_CHARS)
        if lines:
            parts.append("Verbindliche Lektionen aus früherem Feedback (befolgen):\n" + "\n".join(lines))
    if project:
        parts.append(f"Projekt: {' '.join(str(project).split())}")
    if memories:
        def mem_lines():
            for m in memories:
                kind = _text(getattr(m, "kind", None)) or "notiz"
                content = " ".join(_text(getattr(m, "content", m)).split())
                if content:
                    yield clip(f"[{kind}] {content}", 400)
        lines = _bulleted(mem_lines(), MAX_MEMORY_CONTEXT_CHARS)
        if lines:
            parts.append("Erinnerungen über den Nutzer/Projekt:\n" + "\n".join(lines))
    text = "\n\n".join(parts)
    extra = (extra or "").strip()
    if extra:
        remaining = MAX_CONTEXT_CHARS - len(text) - (2 if text else 0)
        if remaining > 60:
            text = (text + "\n\n" if text else "") + clip(extra, remaining - len(CLIP_SUFFIX))
    if len(text) > MAX_CONTEXT_CHARS:
        text = clip(text, MAX_CONTEXT_CHARS - len(CLIP_SUFFIX))
    return text


def example_messages(examples: list[tuple[str, str]]) -> list[dict]:
    """Few-Shot-Paare (user/assistant); Antwort je ≤ ``MAX_EXAMPLE_CHARS``."""
    out: list[dict] = []
    for ex in examples or ():
        try:
            q, a = ex[0], ex[1]
        except (TypeError, IndexError, KeyError):
            continue
        q, a = _text(q), _text(a)
        if not q or not a:
            continue
        out.append({"role": "user", "content": clip(q, MAX_ANSWER_IN_PROMPT)})
        out.append({"role": "assistant", "content": clip(a, MAX_EXAMPLE_CHARS)})
    return out


def _history_messages(history: Sequence[dict] | None, max_chars: int = MAX_HISTORY_CHARS) -> list[dict]:
    """Jüngste Verlaufsnachrichten (user/assistant) bis ``max_chars`` Zeichen, chronologisch."""
    out: list[dict] = []
    used = 0
    for m in reversed(list(history or ())):
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        content = m.get("content")
        if role not in ("user", "assistant") or not isinstance(content, str) or not content.strip():
            continue
        content = clip(content.strip(), max_chars // 2)
        if used + len(content) > max_chars:
            break
        out.append({"role": role, "content": content})
        used += len(content)
    out.reverse()
    return out


def _expert_label(eid: str) -> str:
    ex = EXPERTS.get(str(eid).upper())
    if ex is None:
        if str(eid).upper() == CRITIC.id:
            return f"{CRITIC.id} ({CRITIC.name})"
        if str(eid).upper() == OMEGA.id:
            return f"{OMEGA.id} ({OMEGA.name})"
        return str(eid)
    return f"{ex.id} ({ex.name})"


def _answers_block(answers: dict[str, str]) -> str:
    chunks = []
    for eid, text in (answers or {}).items():
        body = clip((_text(text) or "(keine Antwort)"), MAX_ANSWER_IN_PROMPT)
        chunks.append(f"### Antwort von {_expert_label(eid)}\n{body}")
    return "\n\n".join(chunks) if chunks else "(keine Expertenantworten vorhanden)"


def _json_tail(example: dict) -> str:
    return "Antworte nur mit JSON.\nBeispiel:\n" + json.dumps(example, ensure_ascii=False)


def _tools_section(tools_block: str) -> str:
    tools_block = (tools_block or "").strip()
    if not tools_block:
        return ""
    return TOOL_INSTRUCTION + "\n" + tools_block


def _join(*parts: str) -> str:
    return "\n\n".join(p.strip() for p in parts if p and p.strip())


# ------------------------------------------------------------- Prompt-Bauer
def routing_messages(question: str, expert_ids: Sequence[str] | None) -> list[dict]:
    """Stufe ``routing``: Komplexität, Experten und Werkzeugbedarf als JSON."""
    ids = [e for e in (expert_ids or ()) if str(e).upper() in EXPERTS] or list(EXPERTS.keys())
    listing = "\n".join(f"- {EXPERTS[str(e).upper()].id}: {EXPERTS[str(e).upper()].name} – "
                        f"{EXPERTS[str(e).upper()].role}" for e in ids)
    body = (
        "Du bist der Router von OBITO, einem lokalen KI-Assistenten mit einem Gremium aus "
        "Fachexperten. Du beantwortest die Frage NICHT, sondern ordnest sie ein.\n\n"
        "Komplexität:\n"
        "- einfach: Smalltalk, einzelner Fakt, kurze Umformulierung – ein Aufruf reicht.\n"
        "- mittel: eine Fachfrage mit klarer Antwort, 1–2 Fachgebiete.\n"
        "- komplex: Entwurf, Vergleich, Berechnung, Optimierung, Planung, mehrere Fachgebiete "
        "oder mehrere Teilfragen.\n\n"
        f"Verfügbare Experten:\n{listing}\n\n"
        "Wähle 0–5 Experten (nur die aufgeführten IDs, passendste zuerst; bei einfach: leer). "
        "„werkzeuge“ ist true, wenn Rechnen, Datum/Uhrzeit, Dateien oder Systeminformationen nötig "
        "sind. Die Begründung umfasst höchstens einen Satz.\n\n"
        "Ausgabeformat: {\"komplexitaet\": \"einfach|mittel|komplex\", \"experten\": [IDs], "
        "\"werkzeuge\": true|false, \"begruendung\": \"…\"}"
    )
    user = _join(
        f"Frage des Nutzers:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        _json_tail({"komplexitaet": "mittel", "experten": ["ALPHA", "IOTA"], "werkzeuge": False,
                    "begruendung": "Konstruktionsfrage mit Materialwahl."}),
    )
    return [_system("routing", "system", body), {"role": "user", "content": user}]


def _expert_task(expert: Expert) -> str:
    return (
        f"Du antwortest als Experte {expert.id} – {expert.name}: {expert.role}.\n"
        f"{expert.system_prompt}"
    )


def expert_messages(expert: Expert, question: str, context: str, history: Sequence[dict] | None) -> list[dict]:
    """Stufe ``experte``: System = BASE_RULES + Kontext + Marker (gleicher Präfix für alle Experten),
    Persona und Aufgabe stehen in der Nutzer-Nachricht."""
    system = _system("experte", expert.id, _join(BASE_RULES, context or ""))
    user = _join(
        _expert_task(expert),
        f"Frage des Nutzers:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        "Aufgabe: Antworte aus deiner Fachperspektive mit (1) kurzer Analyse, (2) konkreten "
        "Vorschlägen mit Zahlen und Einheiten, (3) Risiken/Fallen, (4) offenen Fragen, falls "
        "entscheidende Angaben fehlen. Kompakt, höchstens etwa 400 Wörter, keine Einleitung. "
        "Was außerhalb deines Fachgebiets liegt, lässt du weg oder markierst es als „Unsicher:“.",
    )
    return [system, *_history_messages(history), {"role": "user", "content": user}]


def critic_messages(question: str, answers: dict[str, str], context: str) -> list[dict]:
    """Stufe ``kritiker``: prüft alle Expertenantworten, Ausgabe als JSON."""
    system = _system("kritiker", CRITIC.id, _join(CRITIC.system_prompt, BASE_RULES, context or ""))
    ids = ", ".join(str(e).upper() for e in (answers or {}).keys()) or "–"
    user = _join(
        f"Frage des Nutzers:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        _answers_block(answers),
        "Prüfe jede Antwort auf Rechen- und Einheitenfehler, physikalisch unmögliche oder erfundene "
        "Angaben, unbegründete Annahmen, Sicherheitslücken und Widersprüche zwischen den Experten. "
        "Nenne, was für eine vollständige Antwort fehlt.\n"
        f"„experte“ ist eine dieser IDs: {ids}. „schwere“: hoch = Antwort dadurch falsch oder gefährlich, "
        "mittel = wesentliche Ungenauigkeit, niedrig = Schönheitsfehler. „bewertung“ 1–10 für die "
        "Gesamtqualität aller Antworten, „sicher“ true, wenn du dir bei deinem Urteil sicher bist.",
        _json_tail({"bewertung": 7,
                    "fehler": [{"experte": "ALPHA", "problem": "Wandstärke 0,4 mm ist für PETG zu dünn.",
                                "korrektur": "Mindestens 1,2 mm (3 Linien) vorsehen.", "schwere": "hoch"}],
                    "widersprueche": ["ALPHA empfiehlt PLA, IOTA rät wegen Hitze davon ab."],
                    "fehlt": ["Angabe zur Umgebungstemperatur"],
                    "sicher": True}),
    )
    return [system, {"role": "user", "content": user}]


def _critique_items_block(items: Sequence[dict] | None) -> str:
    lines = []
    for it in items or ():
        if isinstance(it, dict):
            problem = _text(it.get("problem")) or "(ohne Beschreibung)"
            fix = _text(it.get("korrektur"))
            severity = _text(it.get("schwere")) or "mittel"
            line = f"- [{severity}] {problem}"
            if fix:
                line += f" → Korrektur: {fix}"
        else:
            line = f"- {_text(it)}"
        lines.append(line)
    return clip("\n".join(lines), MAX_CRITIQUE_CHARS) if lines else "- (keine konkreten Punkte übermittelt)"


def revision_messages(expert: Expert, question: str, previous: str, critique_items: list[dict]) -> list[dict]:
    """Stufe ``revision``: ein bemängelter Experte überarbeitet seine Antwort."""
    system = _system("revision", expert.id, BASE_RULES)
    user = _join(
        _expert_task(expert),
        f"Frage des Nutzers:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        f"Deine bisherige Antwort:\n{clip(_text(previous), MAX_ANSWER_IN_PROMPT)}",
        f"Befunde des Kritikers zu deiner Antwort:\n{_critique_items_block(critique_items)}",
        "Aufgabe: Überarbeite deine Antwort. Behebe jeden Befund, dem du zustimmst, und sage bei "
        "Befunden, denen du begründet widersprichst, kurz warum. Gib die vollständige überarbeitete "
        "Antwort aus (kein Änderungsprotokoll, kein Verweis auf die alte Fassung), gleicher Umfang "
        "wie zuvor, höchstens etwa 400 Wörter.",
    )
    return [system, {"role": "user", "content": user}]


def _render_critique(critique: dict | None) -> str:
    if not critique:
        return ""
    lines: list[str] = []
    rating = critique.get("bewertung")
    if rating is not None:
        lines.append(f"Bewertung des Kritikers: {rating}/10")
    errors = critique.get("fehler") or []
    if errors:
        lines.append("Fehler:")
        for it in errors:
            if isinstance(it, dict):
                who = _text(it.get("experte")) or "allgemein"
                sev = _text(it.get("schwere")) or "mittel"
                problem = _text(it.get("problem"))
                fix = _text(it.get("korrektur"))
                line = f"- [{who}, {sev}] {problem}"
                if fix:
                    line += f" → {fix}"
                lines.append(line)
            else:
                lines.append(f"- {_text(it)}")
    contradictions = critique.get("widersprueche") or []
    if contradictions:
        lines.append("Widersprüche:")
        lines.extend(f"- {_text(c)}" for c in contradictions)
    missing = critique.get("fehlt") or []
    if missing:
        lines.append("Fehlt:")
        lines.extend(f"- {_text(c)}" for c in missing)
    revised = critique.get("revidiert") or []
    if revised:
        lines.append("Bereits überarbeitet: " + ", ".join(_text(r) for r in revised))
    raw = critique.get("roh")
    if raw and not errors and not contradictions and not missing:
        lines.append(f"Kritik (unstrukturiert):\n{_text(raw)}")
    return clip("\n".join(lines), MAX_CRITIQUE_CHARS)


def synthesis_messages(question: str, answers: dict[str, str], critique: dict | None, context: str,
                       history: Sequence[dict] | None, examples: list[tuple[str, str]] | None,
                       tools_block: str = "", medium: bool = False) -> list[dict]:
    """Stufe ``synthese``: OMEGA formt aus Expertenantworten (und Kritik) die finale Antwort."""
    system = _system("synthese", OMEGA.id,
                     _join(OMEGA.system_prompt, BASE_RULES, context or "", _tools_section(tools_block)))
    if medium:
        critique_part = ("Es gab keinen Kritiker. Benenne Widersprüche zwischen den Experten selbst. "
                         "Prüfe Zahlen und Einheiten nach und entscheide begründet.")
    else:
        rendered = _render_critique(critique)
        critique_part = (f"Kritik des Gutachters (berücksichtigen, Fehler nicht übernehmen):\n{rendered}"
                         if rendered else
                         "Der Kritiker lieferte keine verwertbare Kritik – prüfe die Antworten selbst "
                         "auf Fehler und Widersprüche.")
    user = _join(
        f"Frage des Nutzers:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        "Antworten des Expertengremiums:\n\n" + _answers_block(answers),
        critique_part,
        "Aufgabe: Schreibe die endgültige Antwort direkt an den Nutzer – eine einzige, in sich "
        "stimmige Antwort in gutem Deutsch. Übernimm das Richtige, verwirf das Falsche, löse "
        "Widersprüche begründet, nenne Zahlen mit Einheiten und markiere Offenes mit „Unsicher:“. "
        "Beschreibe nicht den internen Ablauf und zähle die Experten nicht einzeln auf; eine kurze "
        "Nennung unterschiedlicher Sichtweisen ist erlaubt, wenn sie dem Nutzer hilft. Wichtigste "
        "Aussage zuerst, dann Details, zum Schluss nächste Schritte oder gezielte Rückfragen.",
    )
    return [system, *example_messages(examples or []), *_history_messages(history),
            {"role": "user", "content": user}]


def fast_messages(question: str, context: str, history: Sequence[dict] | None,
                  examples: list[tuple[str, str]] | None, tools_block: str = "") -> list[dict]:
    """Stufe ``schnell``: ein Aufruf mit Kontext in der OMEGA-Persona."""
    system = _system("schnell", OMEGA.id,
                     _join(OMEGA.system_prompt, BASE_RULES, context or "", _tools_section(tools_block)))
    return [system, *example_messages(examples or []), *_history_messages(history),
            {"role": "user", "content": (question or "").strip() or "(leere Frage)"}]


def tool_result_message(name: str, result: "ToolResult") -> dict:
    """Werkzeug-Ergebnis als Nutzer-Nachricht für das Modell (exakte Texte laut Vertrag)."""
    ok = bool(getattr(result, "ok", False))
    if ok:
        output = getattr(result, "output", "")
        content = f"Ergebnis von Werkzeug »{name}«:\n{'' if output is None else output}"
    else:
        error = getattr(result, "error", None)
        if not error:
            error = getattr(result, "output", None) or "unbekannter Fehler"
        content = (f"Fehler bei Werkzeug »{name}«: {error}\n"
                   "Antworte ohne dieses Ergebnis oder korrigiere den Aufruf.")
    return {"role": "user", "content": content}


def memory_extraction_messages(question: str, answer: str, project: str | None) -> list[dict]:
    """Stufe ``extraktion``: dauerhaft merkwürdige Fakten aus einer Interaktion als JSON."""
    kinds = ", ".join(KINDS)
    body = (
        "Du bist das Gedächtnis-Modul von OBITO. Du liest eine Frage und die gegebene Antwort und "
        "extrahierst nur, was sich für künftige Gespräche dauerhaft zu merken lohnt: Fakten über den "
        "Nutzer oder sein Projekt (Geräte, Maße, Ziele, Rahmenbedingungen), Präferenzen, getroffene "
        "Entscheidungen, funktionierende Lösungen, aufgetretene Fehler.\n\n"
        "Nicht merken: Allgemeinwissen, Floskeln, die Frage selbst („Nutzer hat gefragt …“), "
        "Vermutungen der KI, Unsicheres, Wiederholungen. Jeder Eintrag: ein eigenständiger, konkreter "
        "Satz (12–300 Zeichen) in der dritten Person, mit Zahlen und Einheiten.\n"
        f"„art“ ist eine von: {kinds}. „wichtigkeit“ 0–1 (0,9 = zentrale Projektvorgabe, 0,4 = "
        "Randnotiz). Höchstens 3 Einträge; wenn nichts Merkwürdiges enthalten ist: "
        "{\"erinnerungen\": []}."
    )
    user = _join(
        f"Projekt: {project}" if project else "",
        f"Frage:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        f"Antwort:\n{clip(answer or '', MAX_ANSWER_IN_PROMPT)}",
        _json_tail({"erinnerungen": [
            {"inhalt": "Der Nutzer baut einen 5-Zoll-Quadcopter mit 4S-LiPo und Zielgewicht 600 g.",
             "art": "fakt", "wichtigkeit": 0.8, "tags": ["drohne", "akku"]}]}),
    )
    return [_system("extraktion", "system", body), {"role": "user", "content": user}]


def lesson_extraction_messages(question: str, answer: str, comment: str | None,
                               correction: str | None) -> list[dict]:
    """Stufe ``lektion``: aus negativem Feedback eine wiederverwendbare Regel ableiten (JSON)."""
    body = (
        "Du bist das Lern-Modul von OBITO. Der Nutzer hat eine Antwort kritisiert oder korrigiert. "
        "Leite daraus genau eine Regel ab, die künftige Antworten besser macht: konkret, prüfbar, "
        "als Anweisung formuliert (z. B. „Bei Wandstärken für FDM-Druck immer Vielfache der "
        "Linienbreite angeben.“), höchstens 300 Zeichen, ohne Bezug auf diese eine Frage.\n\n"
        "„gilt_fuer“: 1–5 Stichwörter der Themen, für die die Regel gilt (klein geschrieben). "
        "„allgemein“: true nur, wenn die Regel für praktisch jede Antwort gilt (Stil, Sprache, "
        "Sicherheit, Ehrlichkeit) – dann darf gilt_fuer leer sein."
    )
    user = _join(
        f"Frage:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        f"Kritisierte Antwort:\n{clip(answer or '', MAX_ANSWER_IN_PROMPT)}",
        f"Kommentar des Nutzers:\n{clip(_text(comment), MAX_CRITIQUE_CHARS)}" if comment else "",
        f"Korrektur des Nutzers (bessere Antwort):\n{clip(_text(correction), MAX_ANSWER_IN_PROMPT)}"
        if correction else "",
        _json_tail({"regel": "Bei Akku-Empfehlungen immer die Zellenzahl (S) und den Dauerstrom in A nennen.",
                    "gilt_fuer": ["akku", "lipo", "drohne"], "allgemein": False}),
    )
    return [_system("lektion", "system", body), {"role": "user", "content": user}]


def correction_rewrite_messages(question: str, answer: str, correction: str) -> list[dict]:
    """Stufe ``korrektur``: ursprüngliche Antwort mit eingearbeiteter Korrektur neu schreiben."""
    system = _system("korrektur", OMEGA.id, _join(OMEGA.system_prompt, BASE_RULES))
    user = _join(
        f"Frage des Nutzers:\n{clip(question or '', MAX_QUESTION_CHARS)}",
        f"Ursprüngliche Antwort:\n{clip(answer or '', MAX_ANSWER_IN_PROMPT)}",
        f"Korrektur des Nutzers (hat Vorrang vor allem in der ursprünglichen Antwort):\n"
        f"{clip(correction or '', MAX_ANSWER_IN_PROMPT)}",
        "Aufgabe: Schreibe die ursprüngliche Antwort so um, dass die Korrektur vollständig eingearbeitet "
        "ist. Alles, was der Korrektur widerspricht, wird ersetzt; alles Richtige bleibt erhalten. "
        "Gib die vollständige, eigenständig lesbare Antwort aus – ohne Hinweise wie „korrigierte "
        "Fassung“, ohne Änderungsliste, ohne Kommentare zur Korrektur.",
    )
    return [system, {"role": "user", "content": user}]


def judge_messages(frage: str, erwartet: str, antwort: str) -> list[dict]:
    """Stufe ``richter``: bewertet eine Antwort gegen eine erwartete Antwort (0–10)."""
    body = (
        "Du bist ein strenger, fairer Prüfer. Du bewertest, ob eine Antwort inhaltlich mit der "
        "erwarteten Antwort übereinstimmt: Kernaussagen, Zahlen (±2 % Toleranz), Einheiten, "
        "Sicherheitshinweise. Stil und Länge zählen nicht; zusätzliche richtige Informationen "
        "schaden nicht, falsche Aussagen schon.\n\n"
        "Skala: 10 = alle Kernaussagen richtig und vollständig; 7–9 = richtig mit kleinen Lücken; "
        "4–6 = teilweise richtig oder wesentliche Lücke; 1–3 = überwiegend falsch oder am Thema vorbei; "
        "0 = leer, unbrauchbar oder gefährlich falsch. Die Begründung umfasst höchstens zwei Sätze."
    )
    user = _join(
        f"Frage:\n{clip(frage or '', MAX_QUESTION_CHARS)}",
        f"Erwartete Antwort:\n{clip(erwartet or '', MAX_ANSWER_IN_PROMPT)}",
        f"Zu bewertende Antwort:\n{clip(antwort or '', MAX_ANSWER_IN_PROMPT)}",
        _json_tail({"punkte": 8, "begruendung": "Kernaussagen und Zahlen stimmen; der Sicherheitshinweis fehlt."}),
    )
    return [_system("richter", "system", body), {"role": "user", "content": user}]


def pairwise_judge_messages(frage: str, erwartet: str, antwort_1: str, antwort_2: str) -> list[dict]:
    """Stufe ``richter`` (paarweise): welche von zwei Antworten ist näher an der erwarteten?"""
    body = (
        "Du bist ein strenger, fairer Prüfer. Du vergleichst zwei Antworten auf dieselbe Frage mit der "
        "erwarteten Antwort und entscheidest, welche inhaltlich besser ist: richtige Kernaussagen, "
        "Zahlen und Einheiten, keine falschen Behauptungen, Sicherheitshinweise. Länge, Stil und "
        "Reihenfolge der Antworten sind unerheblich.\n\n"
        "„besser“: 1 = Antwort 1 ist besser, 2 = Antwort 2 ist besser, 0 = gleichwertig oder beide "
        "unbrauchbar. Die Begründung umfasst höchstens zwei Sätze."
    )
    user = _join(
        f"Frage:\n{clip(frage or '', MAX_QUESTION_CHARS)}",
        f"Erwartete Antwort:\n{clip(erwartet or '', MAX_ANSWER_IN_PROMPT)}",
        f"Antwort 1:\n{clip(antwort_1 or '', MAX_ANSWER_IN_PROMPT)}",
        f"Antwort 2:\n{clip(antwort_2 or '', MAX_ANSWER_IN_PROMPT)}",
        _json_tail({"besser": 2, "begruendung": "Antwort 2 nennt die richtige Zellenzahl, Antwort 1 verwechselt V und A."}),
    )
    return [_system("richter", "system", body), {"role": "user", "content": user}]


# ------------------------------------------------------------- JSON-Schemata
ROUTING_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "komplexitaet": {"type": "string", "enum": ["einfach", "mittel", "komplex"]},
        "experten": {"type": "array", "items": {"type": "string"}},
        "werkzeuge": {"type": "boolean"},
        "begruendung": {"type": "string"},
    },
    "required": ["komplexitaet", "experten", "werkzeuge", "begruendung"],
}

CRITIC_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "bewertung": {"type": "integer", "minimum": 1, "maximum": 10},
        "fehler": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "experte": {"type": "string"},
                    "problem": {"type": "string"},
                    "korrektur": {"type": "string"},
                    "schwere": {"type": "string", "enum": ["hoch", "mittel", "niedrig"]},
                },
                "required": ["experte", "problem", "korrektur", "schwere"],
            },
        },
        "widersprueche": {"type": "array", "items": {"type": "string"}},
        "fehlt": {"type": "array", "items": {"type": "string"}},
        "sicher": {"type": "boolean"},
    },
    "required": ["bewertung", "fehler", "widersprueche", "fehlt", "sicher"],
}

MEMORY_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "erinnerungen": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "inhalt": {"type": "string"},
                    "art": {"type": "string", "enum": list(KINDS)},
                    "wichtigkeit": {"type": "number", "minimum": 0, "maximum": 1},
                    "tags": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["inhalt", "art", "wichtigkeit", "tags"],
            },
        },
    },
    "required": ["erinnerungen"],
}

LESSON_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "regel": {"type": "string"},
        "gilt_fuer": {"type": "array", "items": {"type": "string"}},
        "allgemein": {"type": "boolean"},
    },
    "required": ["regel", "gilt_fuer", "allgemein"],
}

JUDGE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "punkte": {"type": "integer", "minimum": 0, "maximum": 10},
        "begruendung": {"type": "string"},
    },
    "required": ["punkte", "begruendung"],
}

PAIRWISE_SCHEMA: dict = {
    "type": "object",
    "properties": {
        "besser": {"type": "integer", "enum": [0, 1, 2]},
        "begruendung": {"type": "string"},
    },
    "required": ["besser", "begruendung"],
}


# ------------------------------------------------------------- Normalisierer
_TRUE_WORDS = {"ja", "true", "wahr", "1", "yes", "y", "j", "richtig", "sicher", "stimmt", "an", "on", "x"}
_FALSE_WORDS = {"nein", "false", "falsch", "0", "no", "n", "unsicher", "aus", "off", "none", "null", "keine", "kein"}
_NUMBER_RE = re.compile(r"[-+]?\d+(?:[.,]\d+)?")


def _load(text: Any) -> Any:
    """Modellantwort (oder bereits geparstes Objekt) als Python-Wert; ``None`` bei Müll."""
    if text is None:
        return None
    if isinstance(text, (dict, list, int, float, bool)):
        return text
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    if not isinstance(text, str):
        return None
    s = text.strip()
    if not s:
        return None
    try:
        return parse_json(s)
    except Exception:
        return None


def _dict(obj: Any) -> dict | None:
    """Objekt mit normalisierten Schlüsseln; einelementige Listen mit Objekt werden entpackt."""
    if isinstance(obj, list):
        dicts = [o for o in obj if isinstance(o, dict)]
        if len(dicts) == 1:
            obj = dicts[0]
    if not isinstance(obj, dict):
        return None
    out: dict = {}
    for k, v in obj.items():
        nk = _key(k)
        if nk and nk not in out:
            out[nk] = v
    return out


def _get(d: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def _to_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        hit = _NUMBER_RE.search(value)
        if hit:
            try:
                return float(hit.group(0).replace(",", "."))
            except ValueError:
                return None
    if isinstance(value, (list, tuple)) and len(value) == 1:
        return _to_float(value[0])
    return None


def _to_int(value: Any, lo: int | None = None, hi: int | None = None) -> int | None:
    f = _to_float(value)
    if f is None or f != f:  # NaN
        return None
    i = int(round(f))
    if lo is not None:
        i = max(lo, i)
    if hi is not None:
        i = min(hi, i)
    return i


def _to_bool(value: Any, default: bool | None = False) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return value != 0
    if isinstance(value, str):
        s = _fold(value).strip().strip(".!")
        if s in _TRUE_WORDS:
            return True
        if s in _FALSE_WORDS:
            return False
        if s.startswith(("ja", "true", "wahr", "yes")):
            return True
        if s.startswith(("nein", "false", "falsch", "no")):
            return False
    if isinstance(value, list):
        return bool(value) if value else default
    return default


def _to_str_list(value: Any, max_len: int = 500) -> list[str]:
    """Einzelstring → Liste; Listen bereinigt (Strings, Zahlen, Objekte als Text); Duplikate weg."""
    items: list[Any]
    if value is None:
        return []
    if isinstance(value, str):
        items = [value]
    elif isinstance(value, dict):
        items = [f"{k}: {_text(v)}" if _text(v) else str(k) for k, v in value.items()]
    elif isinstance(value, (list, tuple, set)):
        items = list(value)
    else:
        items = [value]
    out: list[str] = []
    for it in items:
        if isinstance(it, dict):
            s = _text(_get(_dict(it) or {}, "text", "inhalt", "problem", "beschreibung", "name")) or _text(it)
        else:
            s = _text(it)
        s = " ".join(s.split())
        if s and s not in out:
            out.append(clip(s, max_len))
    return out


def parse_routing(text: Any, known_ids: Iterable[str] | None = None) -> dict | None:
    """Normalisiert die Routing-Antwort: ``{"komplexitaet","experten","werkzeuge","begruendung"}``."""
    d = _dict(_load(text))
    if d is None:
        return None
    known = [str(k).upper() for k in (known_ids if known_ids is not None else EXPERTS.keys())]
    raw = _get(d, "komplexitaet", "komplexitat", "komplexitaet_", "complexity", "schwierigkeit", "stufe", "tiefe")
    level: str | None = None
    if isinstance(raw, (int, float)) and not isinstance(raw, bool):
        level = {1: "einfach", 2: "mittel", 3: "komplex"}.get(int(round(raw)))
    elif raw is not None:
        s = _fold(raw)
        if re.search(r"einfach|simpel|leicht|trivial|niedrig|gering|schnell|easy|simple|low", s):
            level = "einfach"
        elif re.search(r"komplex|schwer|schwierig|hoch|tief|complex|hard|high|anspruchsvoll", s):
            level = "komplex"
        elif re.search(r"mittel|medium|moderat|normal|middle", s):
            level = "mittel"
        else:
            level = {"1": "einfach", "2": "mittel", "3": "komplex"}.get(s.strip())
    if level is None:
        return None
    experts = _resolve_expert_ids(_get(d, "experten", "experts", "experte", "ids", default=[]), known)
    tools = _to_bool(_get(d, "werkzeuge", "werkzeug", "tools", "tool"), default=False)
    reason = _text(_get(d, "begruendung", "begrundung", "grund", "reason", "erklaerung", default=""))
    return {"komplexitaet": level, "experten": experts, "werkzeuge": bool(tools),
            "begruendung": clip(reason, 500)}


_SEVERITY_MAP = (
    (re.compile(r"hoch|high|kritisch|critical|schwer|gravierend|fatal|major|3"), "hoch"),
    (re.compile(r"niedrig|gering|low|minor|klein|leicht|unwichtig|1"), "niedrig"),
    (re.compile(r"mittel|medium|moderat|normal|2"), "mittel"),
)


def _severity(value: Any) -> str:
    s = _fold(value)
    for rx, name in _SEVERITY_MAP:
        if rx.search(s):
            return name
    return "mittel"


def parse_critique(text: Any, expert_ids: Iterable[str] | None = None) -> dict | None:
    """Normalisiert die Kritiker-Antwort auf genau die Schlüssel von ``CRITIC_SCHEMA``
    (fehlende werden ergänzt). ``None``, wenn nichts Verwertbares enthalten ist."""
    d = _dict(_load(text))
    if d is None:
        return None
    known = [str(k).upper() for k in (expert_ids if expert_ids is not None else EXPERTS.keys())]
    keys_present = [k for k in ("bewertung", "fehler", "widersprueche", "widerspruche", "fehlt", "sicher",
                                "score", "note", "punkte", "errors", "probleme", "maengel", "mangel",
                                "widerspruch", "luecken", "lucken", "missing", "fehlend", "confident")
                    if k in d]
    if not keys_present:
        return None

    rating = _to_int(_get(d, "bewertung", "score", "note", "punkte", "rating", "gesamt"), 1, 10)

    errors_raw = _get(d, "fehler", "errors", "probleme", "maengel", "mangel", "befunde", "issues", default=[])
    if isinstance(errors_raw, (str, dict)):
        errors_raw = [errors_raw]
    errors: list[dict] = []
    if isinstance(errors_raw, (list, tuple)):
        for it in errors_raw:
            if isinstance(it, dict):
                item = _dict(it) or {}
                problem = _text(_get(item, "problem", "fehler", "beschreibung", "text", "issue", "aussage", default=""))
                fix = _text(_get(item, "korrektur", "korrigiert", "loesung", "losung", "fix", "vorschlag",
                                 "richtig", default=""))
                who_raw = _get(item, "experte", "expert", "wer", "id", "name", "quelle")
                who = resolve_expert_id(who_raw, known)
                if not problem and not fix:
                    continue
                errors.append({"experte": who, "problem": clip(problem, 600), "korrektur": clip(fix, 600),
                               "schwere": _severity(_get(item, "schwere", "severity", "gewicht", "prioritaet",
                                                         "prioritat", default="mittel"))})
            else:
                problem = " ".join(_text(it).split())
                if problem:
                    who = None
                    hit = re.match(r"^\s*([A-Za-z]+)\s*[:\-–]\s*(.+)$", problem)
                    if hit:
                        who = resolve_expert_id(hit.group(1), known)
                        if who:
                            problem = hit.group(2).strip()
                    errors.append({"experte": who, "problem": clip(problem, 600), "korrektur": "",
                                   "schwere": "mittel"})

    contradictions = _to_str_list(_get(d, "widersprueche", "widerspruche", "widerspruch", "konflikte",
                                       "contradictions", default=[]))
    missing = _to_str_list(_get(d, "fehlt", "fehlend", "luecken", "lucken", "missing", "offen", default=[]))
    sure = _to_bool(_get(d, "sicher", "confident", "sicherheit", "zuversicht"), default=False)
    return {"bewertung": rating, "fehler": errors, "widersprueche": contradictions, "fehlt": missing,
            "sicher": bool(sure)}


def parse_memories(text: Any) -> list[dict]:
    """Nur gültige Erinnerungen: ``{"inhalt","art","wichtigkeit","tags"}``; Müll → ``[]``."""
    obj = _load(text)
    if isinstance(obj, dict):
        d = _dict(obj) or {}
        items = _get(d, "erinnerungen", "erinnerung", "memories", "memory", "fakten", "eintraege", "eintrage", "items")
        if items is None:
            # Vielleicht ist das Objekt selbst ein einzelner Eintrag
            items = [obj] if _get(d, "inhalt", "content", "text") is not None else []
    elif isinstance(obj, list):
        items = obj
    else:
        return []
    if isinstance(items, dict):
        items = [items]
    if not isinstance(items, list):
        return []
    out: list[dict] = []
    seen: set[str] = set()
    for it in items:
        if not isinstance(it, dict):
            continue
        item = _dict(it) or {}
        content = " ".join(_text(_get(item, "inhalt", "content", "text", "fakt", "erinnerung", default="")).split())
        if not content:
            continue
        kind = _fold(_get(item, "art", "kind", "typ", "type", "kategorie", default="notiz")).strip()
        kind = kind if kind in KINDS else next((k for k in KINDS if kind.startswith(k[:5])), "notiz")
        importance = _to_float(_get(item, "wichtigkeit", "importance", "gewicht", "relevanz", "prioritaet"))
        importance = 0.5 if importance is None else max(0.0, min(1.0, importance))
        tags = [t.lower() for t in _to_str_list(_get(item, "tags", "tag", "schlagworte", "stichworte", default=[]), 40)]
        norm = content.lower()
        if norm in seen:
            continue
        seen.add(norm)
        out.append({"inhalt": clip(content, 600), "art": kind, "wichtigkeit": importance, "tags": tags})
    return out


def parse_lesson(text: Any) -> dict | None:
    """Normalisiert eine Lektion: ``{"regel","gilt_fuer","allgemein"}``; ohne Regel ``None``."""
    d = _dict(_load(text))
    if d is None:
        return None
    rule_raw = _get(d, "regel", "rule", "lektion", "lesson", "anweisung", "lehre", "regeln")
    if isinstance(rule_raw, (list, tuple)):
        rules = _to_str_list(rule_raw)
        rule_raw = " ".join(rules) if rules else None
    rule = " ".join(_text(rule_raw).split())
    if not rule:
        return None
    topics = [t.lower() for t in _to_str_list(_get(d, "gilt_fuer", "gilt_fur", "themen", "topics", "bereiche",
                                                   "stichworte", "scope", default=[]), 60)]
    general = _to_bool(_get(d, "allgemein", "general", "generell", "global", "immer"), default=False)
    return {"regel": clip(rule, MAX_LESSON_CHARS), "gilt_fuer": topics, "allgemein": bool(general)}


def parse_judge(text: Any) -> dict | None:
    """Normalisiert ein Richter-Urteil: ``{"punkte": 0–10, "begruendung"}``."""
    obj = _load(text)
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        return {"punkte": _to_int(obj, 0, 10), "begruendung": ""}
    d = _dict(obj)
    if d is None:
        return None
    points = _to_int(_get(d, "punkte", "score", "bewertung", "note", "points", "wert", "punktzahl"), 0, 10)
    if points is None:
        return None
    reason = _text(_get(d, "begruendung", "begrundung", "grund", "reason", "erklaerung", "kommentar", default=""))
    return {"punkte": points, "begruendung": clip(reason, 1000)}


def parse_pairwise(text: Any) -> dict | None:
    """Normalisiert ein paarweises Urteil: ``{"besser": 1|2|0, "begruendung"}``."""
    obj = _load(text)
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        d: dict | None = {"besser": obj}
    else:
        d = _dict(obj)
    if d is None:
        return None
    raw = _get(d, "besser", "better", "gewinner", "winner", "sieger", "antwort", "wahl", "ergebnis", "urteil")
    better: int | None = None
    if isinstance(raw, bool):
        better = None
    elif isinstance(raw, (int, float)):
        i = int(round(raw))
        better = i if i in (0, 1, 2) else None
    elif isinstance(raw, str):
        s = _fold(raw).strip()
        if re.search(r"gleich|unentschieden|beide|keine|tie|equal|draw|weder|^0$", s):
            better = 0
        else:
            hit = re.search(r"\b([12])\b", s)
            if hit:
                better = int(hit.group(1))
            elif re.search(r"(?<![a-z])(a|erste\w*|first|links)(?![a-z])", s):
                better = 1
            elif re.search(r"(?<![a-z])(b|zweite\w*|second|rechts)(?![a-z])", s):
                better = 2
    if better is None:
        return None
    reason = _text(_get(d, "begruendung", "begrundung", "grund", "reason", "erklaerung", "kommentar", default=""))
    return {"besser": better, "begruendung": clip(reason, 1000)}


__all__ = [
    "Expert", "EXPERTS", "CRITIC", "OMEGA", "BASE_RULES", "FILL_ORDER", "STAGE_TAG", "STAGES",
    "MAX_CONTEXT_CHARS", "MAX_MEMORY_CONTEXT_CHARS", "MAX_LESSON_CONTEXT_CHARS", "MAX_LESSON_CHARS",
    "MAX_HISTORY_CHARS", "MAX_HISTORY_ASSISTANT_CHARS", "MAX_EXAMPLE_CHARS", "MAX_ANSWER_IN_PROMPT",
    "MAX_CRITIQUE_CHARS", "CLIP_SUFFIX",
    "stage_of", "clip", "estimate_tokens", "fit_messages", "select_experts", "resolve_expert_id",
    "heuristic_complexity", "context_block", "example_messages", "routing_messages", "expert_messages",
    "critic_messages", "revision_messages", "synthesis_messages", "fast_messages", "tool_result_message",
    "memory_extraction_messages", "lesson_extraction_messages", "correction_rewrite_messages",
    "judge_messages", "pairwise_judge_messages",
    "ROUTING_SCHEMA", "CRITIC_SCHEMA", "MEMORY_SCHEMA", "LESSON_SCHEMA", "JUDGE_SCHEMA", "PAIRWISE_SCHEMA",
    "parse_routing", "parse_critique", "parse_memories", "parse_lesson", "parse_judge", "parse_pairwise",
]
