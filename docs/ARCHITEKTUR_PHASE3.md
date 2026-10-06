# OBITO – Architektur Phase 3 (Geräte, System, 3D, Simulation, Welt, Desktop)

Ergänzt `docs/ARCHITEKTUR.md` (Phase 1: KI-Kern) und `docs/ARCHITEKTUR_PHASE2.md` (Phase 2:
Engineering-Plattform). Beide Phasen sind implementiert und getestet (791 Tests); Phase 3 ist umgesetzt (≈ 1000 Tests gesamt). Phase 3 setzt
die restlichen Bereiche der OBITO-Konzeptbilder um:

| Modul | Konzeptbereich |
|---|---|
| `obito/devices.py` | Geräte & Sensoren: automatische Erkennung angeschlossener Geräte (USB, seriell), Telemetrie lesen |
| `obito/sysmon.py` | System & Performance: CPU, RAM, Platte, GPU/VRAM, laufende Modelle – echte Messwerte |
| `obito/geometry.py` | 3D-Modellierung: parametrische Bauteile, Baugruppen, STL/OBJ, Masse/Volumen/Schwerpunkt |
| `obito/simulation.py` | Simulation: Zeitschritt-Modelle (Flug/Akku, Thermik, Fall mit Luftwiderstand, Regler) |
| `obito/geo.py` | Welt & Karten: Geodäsie, Routen, Sonnenstand, optionale Online-Wetterdaten |
| `obito/desktop.py` | Native App: Server + App-Fenster in einem Prozess, EXE-Bau |
| Integration | Brain, Server-Routen, CLI, HUD-Tabs, Tests |

Alle Grundsätze aus Phase 1 und 2 gelten: **100 % lokal, nur Standardbibliothek im Kern, deutsch,
keine Fake-Daten, testbar ohne Hardware/Netz.** Was nicht gemessen werden kann, wird als
`None`/„nicht verfügbar“ gemeldet – niemals erfunden. Netzzugriff gibt es ausschließlich in
`geo.weather()` und nur bei `cfg.online = True` (Standard `False`).

Konventionen wie in Phase 2: `to_dict()` mit deutschen Schlüsseln; Datenbanken wie `memory.py`
(`sqlite3`, `check_same_thread=False`, `RLock`, WAL, idempotentes `close()`); Werkzeuge über
`register_tools(registry, …)`; Fehler als `ValueError` mit deutschem Text.

---

## `obito/devices.py` – Geräte & Sensoren

Erkennt **echte** angeschlossene Geräte; ohne Hardware liefert alles leere Listen.

```python
@dataclass
class Device:
    kind: str            # "usb" | "seriell" | "netz"
    name: str            # Klartext (Produkt / FriendlyName / Port)
    port: str | None     # COM3, /dev/ttyUSB0 …
    vendor_id: str | None; product_id: str | None   # hex, klein, 4-stellig
    vendor: str | None; product: str | None
    role: str            # "flugsteuerung" | "mikrocontroller" | "usb_seriell" | "drohne" | "kamera" |
                         # "speicher" | "eingabe" | "unbekannt"
    status: str          # "verbunden" | "fehler" | "unbekannt"
    raw: dict            # Rohdaten der Quelle
    def to_dict(self) -> dict
    @property
    def key(self) -> str  # stabiler Schlüssel: port oder vid:pid+name

KNOWN_VENDORS: dict[str, str]              # "2ca3": "DJI", "0483": "STMicroelectronics", "2341": "Arduino",
                                           # "303a": "Espressif", "0403": "FTDI", "10c4": "Silicon Labs",
                                           # "1a86": "QinHeng (CH340)", "2e8a": "Raspberry Pi", "26ac": "3D Robotics",
                                           # "1209": "pid.codes (Open Source)", "0bda": "Realtek", "046d": "Logitech", …
KNOWN_PRODUCTS: dict[tuple[str, str], tuple[str, str]]   # (vid,pid) -> (Name, role) z. B. ("0483","5740") -> ("STM32 Virtual COM Port (Betaflight/INAV)", "flugsteuerung")

def classify(vendor_id, product_id, name) -> str          # role-Heuristik (Name + Tabellen)
def list_serial_ports() -> list[Device]
    # Windows: winreg HKLM\HARDWARE\DEVICEMAP\SERIALCOMM; Linux: /sys/class/tty + /dev/serial/by-id;
    # macOS: /dev/cu.*; optional pyserial (serial.tools.list_ports) wenn installiert
def list_usb_devices() -> list[Device]
    # Windows: PowerShell Get-PnpDevice -PresentOnly (JSON, Timeout 15 s); Linux: /sys/bus/usb/devices/*
    # (idVendor, idProduct, manufacturer, product); macOS: system_profiler SPUSBDataType -json
def scan() -> list[Device]                                 # vereinigt beide Quellen, dedupliziert über key
def read_serial(port: str, baud: int = 115200, seconds: float = 2.0, max_bytes: int = 4096) -> dict
    # liest Rohdaten (pyserial falls vorhanden, sonst POSIX termios; Windows ohne pyserial -> ValueError mit Hinweis)
    # -> {"port", "baud", "bytes", "text" (utf-8, ersetzt), "zeilen": [...], "dauer_s"}
def parse_telemetry(text: str) -> dict                     # erkennt NMEA ($GPGGA/$GPRMC -> lat/lon/alt/speed),
                                                           # MAVLink-Textzeilen nicht; Key=Value-Zeilen; JSON-Zeilen

class DeviceStore:      # <data>/geraete.db – Verlauf: zuerst/zuletzt gesehen, Anzahl Sichtungen, Notiz
    def __init__(self, path: Path)
    def update(self, devices: list[Device]) -> list[dict]   # upsert, markiert nicht mehr gesehene als "getrennt"
    def list(self, connected_only: bool = False) -> list[dict]
    def note(self, key: str, text: str) -> None
    def forget(self, key: str) -> bool
    def stats(self) -> dict
    def close(self) -> None

def register_tools(registry, store: DeviceStore | None = None) -> None
    # geraete_scannen() -> Text; seriell_lesen(port, baud=115200, sekunden=2) -> Text (nicht gefährlich: nur lesen)
```

Tests simulieren die Quellen über Monkeypatching (`_read_sysfs`, `_run_powershell`), nie echte Hardware.

## `obito/sysmon.py` – System & Performance

```python
def cpu() -> dict        # {"kerne", "logisch", "last_prozent" (None wenn nicht messbar), "name"}
def memory() -> dict     # {"gesamt_bytes", "frei_bytes", "belegt_prozent"} – Linux /proc/meminfo, Windows ctypes GlobalMemoryStatusEx, macOS vm_stat/sysctl
def disk(path) -> dict   # shutil.disk_usage
def gpu() -> list[dict]  # nvidia-smi --query-gpu=name,memory.total,memory.used,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
                         # (Timeout 5 s); AMD/Intel: [] ; jede Karte {"name","vram_gesamt_mb","vram_belegt_mb","auslastung_prozent","temperatur_c"}
def process() -> dict    # eigener Prozess: RSS (resource/ctypes/psutil-frei), Threads, Laufzeit
def snapshot(backend=None, data_dir=None) -> dict   # alles zusammen + "modelle_geladen" (backend.running()) + Datenverzeichnis-Größe
def recommend(snapshot: dict) -> list[str]           # deutsche Hinweise (z. B. "8 GB VRAM: 7B-Q4 mit num_ctx 8192")
class Sampler:           # Ringpuffer der letzten N Snapshots (für HUD-Verlauf), Thread-sicher
    def __init__(self, size: int = 120)
    def add(self, snap: dict) -> None
    def series(self) -> dict   # {"zeit": [...], "cpu": [...], "ram": [...], "gpu": [...]}
def register_tools(registry) -> None   # system_status() -> Text
```

## `obito/geometry.py` – 3D-Modellierung

Dreiecksnetze in reinem Python (Listen von Punkten/Dreiecken), ausreichend für Bauteile mit
≤ 200 000 Dreiecken.

```python
@dataclass
class Mesh:
    vertices: list[tuple[float,float,float]]
    faces: list[tuple[int,int,int]]
    name: str = ""
    def translate(dx,dy,dz) -> Mesh; rotate(axis: "x"|"y"|"z", degrees) -> Mesh; scale(sx,sy=None,sz=None) -> Mesh; mirror(axis) -> Mesh
    def merge(other) -> Mesh                # Baugruppe (Konkatenation, keine Boolesche Operation)
    def bounds() -> dict                    # min/max/größe je Achse (mm)
    def volume() -> float                   # mm³, signierte Tetraeder-Summe
    def surface_area() -> float             # mm²
    def center_of_mass() -> tuple
    def inertia(density_g_cm3) -> dict      # Trägheitsmomente um Schwerpunkt (g·mm²), Näherung über Tetraeder
    def is_watertight() -> bool             # jede Kante genau zweimal, entgegengesetzt
    def stats(material: str | None = None) -> dict   # Dreiecke, Volumen cm³, Fläche cm², Bounding-Box, Masse g (über engineering.find_material)
    def to_json() -> dict                   # {"punkte": [[x,y,z],…], "dreiecke": [[a,b,c],…]} für die HUD

# Primitive (alle in mm, Ursprung = Mitte der Grundfläche bzw. Mitte, Segmente begrenzt 3..256)
def box(l, b, h) -> Mesh
def cylinder(d, h, segments=48) -> Mesh
def tube(d_outer, d_inner, h, segments=48) -> Mesh
def cone(d_bottom, d_top, h, segments=48) -> Mesh
def sphere(d, segments=32) -> Mesh
def plate_with_holes(l, b, t, holes: list[tuple[x,y,d]], segments=24) -> Mesh
    # Platte mit Durchgangsbohrungen (exakte Triangulation des Randes mit Ohrenschneiden – kein Boolean)
def drone_frame(wheelbase_mm, arm_width_mm, arm_thickness_mm, plate_mm, motor_hole_mm=12.0, arms=4) -> Mesh
    # X-Rahmen: Mittelplatte + Arme + Motorböden; Baugruppe aus Primitiven

PRIMITIVES: dict[str, dict]   # name -> {"fn", "parameter": [(name, beschreibung, default)], "beschreibung"}
def build(kind: str, params: dict) -> Mesh      # tolerant (Dezimalkomma), ValueError bei Unsinn

# Dateien
def write_stl(mesh, path, binary=True) -> None;  def read_stl(path) -> Mesh      # ASCII + binär
def write_obj(mesh, path) -> None;               def read_obj(path) -> Mesh
def load(path) -> Mesh   # nach Endung; Dateigröße begrenzt (MAX_FILE_BYTES = 64 MB)

class ModelStore:        # <data>/modelle3d.db + Dateien unter <data>/modelle3d/<id>_v<version>.stl
    def __init__(self, path: Path, files_dir: Path)
    def save(self, mesh, name, *, kind=None, params=None, project=None, material=None, note="") -> dict
        # neue Version wenn Name+Projekt existiert; dict: id, name, version, projekt, art, parameter, material, statistik, datei, erstellt
    def get(self, model_id) -> dict | None; def mesh(self, model_id) -> Mesh
    def list(self, project=None, name=None) -> list[dict]; def versions(self, name, project=None) -> list[dict]
    def delete(self, model_id) -> bool; def import_file(self, path, name=None, project=None) -> dict
    def stats(self) -> dict; def close(self) -> None

def register_tools(registry, store: ModelStore) -> None
    # modell_erzeugen(art, parameter: "l=100, b=50, h=10", name, projekt=None, material=None) -> Text mit Statistik + Datei
    # modell_info(name_oder_id) -> Text; modell_liste(projekt=None) -> Text
```

## `obito/simulation.py` – Simulation

Zeitschritt-Modelle mit festem Schritt (RK4 wo sinnvoll), Ergebnis immer:
`{"typ", "parameter", "zeit_s": [...], "reihen": {"name": [...]}, "einheiten": {...}, "zusammenfassung": {...},
"annahmen": [...], "warnungen": [...]}`. Reihen werden auf ≤ 600 Punkte ausgedünnt (`MAX_POINTS`).

```python
def hover_flight(mass_g, cells, mah, hover_current_a, *, usable=0.8, peukert=1.05, dt=1.0, payload_g=0.0) -> dict
    # Akku-Entladung mit einfachem Innenwiderstand (0,004 Ω/Zelle·(5000/mAh)), Spannung/Zelle über SOC-Kennlinie
    # (Stützpunkte LiPo), Abbruch bei CELL_EMPTY_V; Zusammenfassung: flugzeit_min, energie_wh, restkapazitaet
def climb_profile(mass_g, thrust_max_n, target_alt_m, *, drag_cd=1.0, area_m2=0.05, dt=0.05) -> dict
    # vertikale Dynamik m·a = T - m·g - ½ρ cd A v²; Schub geregelt (P-Regler) auf Zielhöhe; Zeit bis Höhe, max. Steigrate
def drop_with_drag(mass_kg, height_m, *, cd=0.47, area_m2, dt=0.01) -> dict   # Fall mit Luftwiderstand, Aufprallgeschwindigkeit
def thermal_rc(power_w, mass_g, cp_j_per_gk, h_w_per_m2k, area_m2, *, ambient_c=25.0, duration_s=600, dt=1.0) -> dict
    # Ein-Knoten-Modell: m·cp·dT/dt = P - h·A·(T-T_amb); Endtemperatur, Zeitkonstante, Zeit bis 80 % (z. B. ESC/Motor)
def pid_step(kp, ki, kd, *, plant_tau_s=0.3, plant_gain=1.0, setpoint=1.0, duration_s=5.0, dt=0.005) -> dict
    # PT1-Strecke mit PID; Überschwingen %, Einschwingzeit (2 %), stationärer Fehler, Stabilitätswarnung
def battery_discharge(cells, mah, current_a, *, dt=5.0) -> dict   # Spannungs- und SOC-Verlauf, Zeit bis leer
def beam_sweep(force_n, length_m, material, width_m, heights_mm: list[float]) -> dict   # Parameterstudie über engineering.beam_cantilever

SIMULATIONS: dict[str, dict]    # name -> {"fn", "parameter": [...], "beschreibung"}
def run(kind: str, params: dict) -> dict          # tolerant, ValueError bei Unsinn
def summary_text(result: dict) -> str             # deutsche Kurzfassung für Werkzeug/CLI
def register_tools(registry) -> None              # simulation_starten(art, parameter: "k=v, …") -> Text
```

## `obito/geo.py` – Welt & Karten (offline)

```python
EARTH_RADIUS_M = 6371008.8
def parse_coord(text) -> tuple[float, float]       # "49.9576, 6.9294" | "49°57'27\"N 6°55'46\"E" | "49.9576N 6.9294E"
def format_coord(lat, lon, dms=False) -> str
def distance_m(lat1, lon1, lat2, lon2) -> float    # Haversine
def bearing_deg(lat1, lon1, lat2, lon2) -> float
def destination(lat, lon, bearing_deg, distance_m) -> tuple
def route_length_m(points: list[tuple]) -> float
def polygon_area_m2(points) -> float               # sphärischer Exzess (klein) / Projektions-Shoelace
def flight_plan(points, speed_m_s, *, wind_speed_m_s=0.0, wind_from_deg=0.0, hover_s_per_point=0.0) -> dict
    # je Abschnitt Distanz, Kurs, Bodengeschwindigkeit (Windkomponente), Zeit; Gesamtzeit, Gesamtstrecke
def sun(lat, lon, when: datetime | None = None) -> dict     # NOAA-Algorithmus: aufgang, untergang, tageslaenge_h, hoehe_deg, azimut_deg (UTC-Zeiten + lokale wenn tz)
def terminator(when) -> list[tuple[lat, lon]]      # Tag-Nacht-Grenze (72 Punkte) für die Karte
def utm(lat, lon) -> dict                          # Zone, Ostwert, Nordwert (WGS84, Transverse Mercator)
def mgrs? – nein.

class WaypointStore:     # <data>/geo.db – Orte/Routen je Projekt
    def add_place(name, lat, lon, *, project=None, note="") -> dict
    def add_route(name, points: list[tuple], *, project=None) -> dict
    def list_places(project=None) -> list[dict]; def list_routes(project=None) -> list[dict]
    def delete_place(id) -> bool; def delete_route(id) -> bool; def stats() -> dict; def close()

def weather(lat, lon, *, base_url="https://api.open-meteo.com", timeout=10, opener=None) -> dict
    # NUR wenn vom Aufrufer erlaubt (Brain prüft cfg.online); Open-Meteo ohne Schlüssel; Rückgabe
    # {"temperatur_c","wind_kmh","wind_richtung_deg","boeen_kmh","niederschlag_mm","bewoelkung_prozent","quelle","zeit"}
    # und Flugtauglichkeit: {"flugtauglich": bool, "gruende": [...]} (Wind > 36 km/h, Böen > 50, Regen > 0,5 mm/h)
def register_tools(registry, store: WaypointStore, online: Callable[[], bool]) -> None
    # geo_distanz(von, nach), geo_route(punkte "lat,lon; lat,lon", geschwindigkeit_m_s, wind_kmh=0, wind_aus_deg=0),
    # sonnenstand(ort, datum=None), wetter(ort) (nur online; sonst Hinweis)
```

## `obito/desktop.py` – Native App

```python
def find_browser() -> tuple[str, list[str]] | None   # Edge/Chrome/Chromium: (exe, basis-argumente); Windows-Pfade + PATH
def run_app(cfg, *, backend=None, width=1600, height=950, open_browser=True, wait=True) -> int
    # startet ObitoServer im Hintergrund, öffnet Browser-App-Fenster (--app=URL --window-size --user-data-dir=<data>/fenster),
    # wartet auf Fensterende, stoppt Server sauber; ohne Browser: webbrowser.open + Strg+C; pywebview wird genutzt, wenn installiert
```
Dazu `packaging/obito.spec` (PyInstaller, onedir, static/ und beispiele/ eingebunden) und
`build_exe.bat`. `OBITO.bat` ruft `python -m obito app` statt separatem Server + Edge.

## Integration

- `Config`: `online: bool = False`, `serial_baud: int = 115200`; Pfade `devices_db`, `models3d_db`,
  `models3d_dir`, `geo_db`.
- `Brain`: `self.devices`, `self.models3d`, `self.geo`, `self.sysmon` (Sampler); registriert alle
  Werkzeuge; `status()` ergänzt `system` (Snapshot), `geraete`, `modelle3d`, `geo`.
- `agents.needs_tools`: Stichwörter (Gerät, USB, COM, seriell, STL, Modell erzeugen, Simulation,
  Entfernung, Koordinaten, Sonnen, Wetter, CPU, RAM, VRAM).
- `server.py` Routen:
  `GET /api/system`, `GET /api/system/verlauf`,
  `GET /api/geraete`, `POST /api/geraete/scan`, `POST /api/geraete/lesen`, `DELETE /api/geraete/<key>`,
  `GET /api/modelle3d`, `POST /api/modelle3d` (art+parameter | datei), `GET /api/modelle3d/<id>`,
  `GET /api/modelle3d/<id>/mesh`, `GET /api/modelle3d/<id>/stl` (Download), `DELETE /api/modelle3d/<id>`,
  `GET /api/modelle3d/arten`,
  `GET /api/simulation/arten`, `POST /api/simulation`,
  `GET /api/geo/orte`, `POST /api/geo/orte`, `DELETE /api/geo/orte/<id>`, `GET /api/geo/routen`,
  `POST /api/geo/routen`, `DELETE /api/geo/routen/<id>`, `POST /api/geo/plan`, `GET /api/geo/sonne`,
  `GET /api/geo/wetter` (403 wenn offline).
- CLI: `python -m obito system|geraete|modell3d|simulation|geo|app`; Chat-Befehle `/system`,
  `/geraete`, `/modell3d`, `/simulation`, `/geo`.
- Zusätzlich umgesetzt: `POST /api/geraete/notiz`; `GET /api/status` liefert `kacheln_url` und die CSP erlaubt
  `img-src tile.openstreetmap.org` nur bei `online = true`; `packaging/obito.spec` + `build_exe.bat` (PyInstaller).
- HUD-Tabs: **System** (Live-Verlauf CPU/RAM/GPU, Modelle geladen, Empfehlungen), **Geräte**
  (Scan, Verlauf, seriell lesen), **3D-Modellierung** (Primitiv wählen → Parameter → Vorschau in
  einem eigenen WebGL/Canvas-Renderer mit Orbit, Statistik, STL-Download, Versionen), **Simulation**
  (Art wählen → Parameter → Diagramm + Zusammenfassung), **Weltkarte** (Canvas-Karte mit Gradnetz,
  Tag-Nacht-Grenze, Orte, Routen, Flugplan; Online-Kacheln von OpenStreetMap nur per Schalter, Hinweis
  auf Lizenz). Keine externen Bibliotheken, kein CDN.
