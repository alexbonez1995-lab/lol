"""Geräte & Sensoren: angeschlossene USB- und serielle Geräte erkennen, Telemetrie lesen.

Alles 100 % lokal und nur mit der Standardbibliothek:

* :func:`list_usb_devices` liest echte Geräte aus dem Betriebssystem (Linux ``/sys/bus/usb``,
  Windows ``Get-PnpDevice`` über PowerShell, macOS ``system_profiler``).
* :func:`list_serial_ports` findet serielle Schnittstellen (Linux ``/sys/class/tty`` und
  ``/dev/serial/by-id``, Windows Registry ``SERIALCOMM``, macOS ``/dev/cu.*``); ist ``pyserial``
  installiert, wird es zusätzlich genutzt.
* :func:`scan` vereinigt beide Quellen; :func:`classify` ordnet jedem Gerät eine Rolle zu
  (Flugsteuerung, Mikrocontroller, USB-Seriell-Wandler, Drohne, Sensor …).
* :func:`read_serial` liest Rohdaten von einem Port (``pyserial`` falls vorhanden, sonst POSIX
  ``termios``), :func:`parse_telemetry` erkennt NMEA-Sätze (mit Prüfsummenprüfung), Key=Value-
  und JSON-Zeilen.
* :class:`DeviceStore` merkt sich den Verlauf (zuerst/zuletzt gesehen, Sichtungen, Notiz).
* :func:`register_tools` stellt dem Modell ``geraete_scannen`` und ``seriell_lesen`` bereit.

Ohne Hardware liefern alle Funktionen leere Listen – es werden nie Geräte erfunden. Alle
Systemquellen sind in kleine Funktionen gekapselt (``_read_sysfs_usb``, ``_read_sysfs_tty``,
``_run_powershell``, ``_winreg_serial``, ``_system_profiler``, ``_open_serial``), damit Tests sie
ohne Hardware ersetzen können.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import re
import sqlite3
import subprocess
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Sequence

if TYPE_CHECKING:  # pragma: no cover - nur für Typprüfer
    from .tools import ToolRegistry

log = logging.getLogger("obito.devices")

MAX_FIELD_CHARS = 200            # Strings aus Geräten werden hierauf gekürzt
SUBPROCESS_TIMEOUT = 15.0        # Sekunden für PowerShell / system_profiler
MIN_READ_SECONDS = 0.1
MAX_READ_SECONDS = 30.0
MAX_READ_BYTES = 1024 * 1024
DEFAULT_BAUD = 115200

#: Zulässige Baudraten für :func:`read_serial`.
COMMON_BAUDS = (300, 1200, 2400, 4800, 9600, 14400, 19200, 38400, 57600, 115200, 230400, 250000,
                460800, 500000, 921600, 1000000, 1500000, 2000000)

#: Alle Rollen, die :func:`classify` vergibt.
ROLES = ("flugsteuerung", "mikrocontroller", "usb_seriell", "drohne", "kamera", "speicher", "eingabe",
         "sensor", "netz", "hub", "unbekannt")

#: Gerätearten.
KINDS = ("usb", "seriell", "netz")

#: Statuswerte eines Geräts.
STATUSES = ("verbunden", "fehler", "unbekannt", "getrennt")

#: Virtuelle Root-Hubs des Linux-Kernels sind keine angeschlossenen Geräte.
SKIP_VENDORS = {"1d6b"}

#: Bekannte USB-Hersteller (Vendor-ID hex, klein, 4-stellig).
KNOWN_VENDORS: dict[str, str] = {
    "2ca3": "DJI",
    "0483": "STMicroelectronics",
    "2341": "Arduino",
    "303a": "Espressif",
    "0403": "FTDI",
    "10c4": "Silicon Labs",
    "1a86": "QinHeng (CH340)",
    "2e8a": "Raspberry Pi",
    "26ac": "3D Robotics",
    "1209": "pid.codes (Open Source)",
    "0bda": "Realtek",
    "046d": "Logitech",
    "8087": "Intel",
    "045e": "Microsoft",
    "0781": "SanDisk",
    "0951": "Kingston",
    "067b": "Prolific",
    "16c0": "PJRC (Teensy)",
    "2672": "GoPro",
    "054c": "Sony",
    "04e8": "Samsung",
    "05ac": "Apple",
    "0955": "Nvidia",
    "091e": "Garmin",
    "1546": "u-blox",
    "1d50": "OpenMoko (Open Source)",
    "0d28": "Arm mbed",
    "1366": "SEGGER",
    "1fc9": "NXP",
    "04d8": "Microchip",
}

#: Standardrolle je Hersteller, wenn weder Produkt noch Name eindeutig sind.
VENDOR_ROLES: dict[str, str] = {
    "2ca3": "drohne",
    "0483": "mikrocontroller",
    "2341": "mikrocontroller",
    "303a": "mikrocontroller",
    "0403": "usb_seriell",
    "10c4": "usb_seriell",
    "1a86": "usb_seriell",
    "067b": "usb_seriell",
    "2e8a": "mikrocontroller",
    "26ac": "flugsteuerung",
    "1209": "mikrocontroller",
    "16c0": "mikrocontroller",
    "1546": "sensor",
    "091e": "sensor",
    "0781": "speicher",
    "0951": "speicher",
    "2672": "kamera",
    "046d": "eingabe",
    "045e": "eingabe",
    "0d28": "mikrocontroller",
    "1366": "mikrocontroller",
    "04d8": "mikrocontroller",
}

#: Bekannte Produkte: (vid, pid) -> (Name, Rolle).
KNOWN_PRODUCTS: dict[tuple[str, str], tuple[str, str]] = {
    ("0483", "5740"): ("STM32 Virtual COM Port (Betaflight/INAV)", "flugsteuerung"),
    ("0483", "df11"): ("STM32 DFU-Bootloader", "flugsteuerung"),
    ("0483", "3748"): ("ST-LINK/V2", "mikrocontroller"),
    ("0483", "374b"): ("ST-LINK/V2-1", "mikrocontroller"),
    ("26ac", "0011"): ("PX4 FMU (Pixhawk)", "flugsteuerung"),
    ("26ac", "0010"): ("PX4 FMU v2", "flugsteuerung"),
    ("26ac", "0032"): ("PX4 FMU v5 (Pixhawk 4)", "flugsteuerung"),
    ("1209", "5740"): ("Holybro/PX4 Flugsteuerung (pid.codes)", "flugsteuerung"),
    ("1209", "5741"): ("PX4 FMU v6X (pid.codes)", "flugsteuerung"),
    ("3162", "004b"): ("Holybro Pixhawk 6X", "flugsteuerung"),
    ("3162", "004c"): ("Holybro Pixhawk 6C", "flugsteuerung"),
    ("1546", "01a8"): ("u-blox GNSS-Empfänger (NEO-M8/M9)", "sensor"),
    ("1546", "01a7"): ("u-blox GNSS-Empfänger (6/7)", "sensor"),
    ("1546", "01a9"): ("u-blox GNSS-Empfänger (F9/ZED-F9P)", "sensor"),
    ("2341", "0043"): ("Arduino Uno", "mikrocontroller"),
    ("2341", "0001"): ("Arduino Uno (alt)", "mikrocontroller"),
    ("2341", "0042"): ("Arduino Mega 2560", "mikrocontroller"),
    ("2341", "8036"): ("Arduino Leonardo", "mikrocontroller"),
    ("2341", "0036"): ("Arduino Leonardo (Bootloader)", "mikrocontroller"),
    ("2341", "804d"): ("Arduino Zero", "mikrocontroller"),
    ("2341", "8057"): ("Arduino Nano 33 IoT", "mikrocontroller"),
    ("303a", "1001"): ("ESP32-S3 (USB-JTAG/Seriell)", "mikrocontroller"),
    ("303a", "0002"): ("ESP32-S2 (USB-Bootloader)", "mikrocontroller"),
    ("303a", "4001"): ("ESP32-S3 (USB CDC)", "mikrocontroller"),
    ("1a86", "7523"): ("CH340 USB-Seriell-Wandler", "usb_seriell"),
    ("1a86", "55d4"): ("CH9102 USB-Seriell-Wandler", "usb_seriell"),
    ("1a86", "7522"): ("CH340 USB-Seriell-Wandler (Variante)", "usb_seriell"),
    ("0403", "6001"): ("FT232R USB-Seriell-Wandler", "usb_seriell"),
    ("0403", "6010"): ("FT2232 USB-Seriell-Wandler (2 Kanäle)", "usb_seriell"),
    ("0403", "6015"): ("FT231X USB-Seriell-Wandler", "usb_seriell"),
    ("10c4", "ea60"): ("CP210x USB-Seriell-Wandler", "usb_seriell"),
    ("10c4", "ea70"): ("CP2105 USB-Seriell-Wandler (2 Kanäle)", "usb_seriell"),
    ("067b", "2303"): ("PL2303 USB-Seriell-Wandler", "usb_seriell"),
    ("067b", "23a3"): ("PL2303GC USB-Seriell-Wandler", "usb_seriell"),
    ("2e8a", "0003"): ("RP2040 Boot (UF2-Bootloader)", "mikrocontroller"),
    ("2e8a", "000a"): ("Raspberry Pi Pico", "mikrocontroller"),
    ("2e8a", "000f"): ("Raspberry Pi Pico W (Bootloader)", "mikrocontroller"),
    ("2e8a", "000c"): ("Raspberry Pi Debug Probe", "mikrocontroller"),
    ("16c0", "0483"): ("Teensy USB Serial", "mikrocontroller"),
    ("16c0", "0478"): ("Teensy HalfKay-Bootloader", "mikrocontroller"),
    ("0d28", "0204"): ("Arm mbed CMSIS-DAP (DAPLink)", "mikrocontroller"),
    ("2ca3", "001f"): ("DJI Fluggerät/Fernsteuerung", "drohne"),
    ("2ca3", "0040"): ("DJI Assistant-Gerät", "drohne"),
    ("2ca3", "0022"): ("DJI Goggles", "drohne"),
    ("2672", "0049"): ("GoPro HERO (USB)", "kamera"),
    ("2672", "004b"): ("GoPro HERO (MTP)", "kamera"),
    ("046d", "0825"): ("Logitech Webcam C270", "kamera"),
    ("046d", "085e"): ("Logitech BRIO Webcam", "kamera"),
    ("046d", "c52b"): ("Logitech Unifying-Empfänger", "eingabe"),
    ("045e", "028e"): ("Microsoft Xbox 360 Controller", "eingabe"),
    ("054c", "05c4"): ("Sony DualShock 4", "eingabe"),
    ("054c", "0ce6"): ("Sony DualSense", "eingabe"),
    ("0781", "5567"): ("SanDisk Cruzer Blade", "speicher"),
    ("0781", "5583"): ("SanDisk Ultra Fit", "speicher"),
    ("0951", "1666"): ("Kingston DataTraveler 100 G3", "speicher"),
    ("0bda", "8153"): ("Realtek RTL8153 Gigabit-Ethernet", "netz"),
    ("0bda", "b812"): ("Realtek RTL88x2BU WLAN", "netz"),
    ("8087", "0a2b"): ("Intel Bluetooth", "netz"),
    ("8087", "0026"): ("Intel AX201 Bluetooth", "netz"),
    ("0955", "7020"): ("Nvidia Jetson (USB-Gerätemodus)", "mikrocontroller"),
    ("091e", "0003"): ("Garmin GPS (USB)", "sensor"),
    ("05ac", "12a8"): ("Apple iPhone", "unbekannt"),
    ("04e8", "6860"): ("Samsung Galaxy (MTP)", "unbekannt"),
}

# Reihenfolge wichtig: spezifische Begriffe zuerst (z. B. „Flight Controller“ vor „Controller“).
_NAME_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("flugsteuerung", ("betaflight", "inav", "pixhawk", "ardupilot", "px4", "flight controller",
                       "flugsteuerung", "fmu", "cleanflight", "kiss fc", "speedybee", "matek", "cube orange",
                       "holybro")),
    ("drohne", ("dji", "drohne", "drone", "quadcopter", "mavic", "phantom", "skydio", "parrot", "autel")),
    ("sensor", ("gps", "gnss", "u-blox", "ublox", "neo-m", "zed-f9", "imu", "lidar", "barometer", "magnetometer",
                "sensor", "garmin", "receiver gps")),
    ("kamera", ("camera", "webcam", "kamera", "gopro", "capture", "video device", "uvc")),
    ("speicher", ("mass storage", "massenspeicher", "usb-stick", "flash drive", "disk drive", "datenträger",
                  "card reader", "kartenleser", "sd card", "ssd", "hdd", "diskdrive", "storage")),
    ("eingabe", ("keyboard", "tastatur", "mouse", "maus", "gamepad", "joystick", "controller", "touchpad",
                 "trackball", "hid", "hidclass", "unifying", "dualshock", "dualsense", "xbox", "eingabe")),
    ("hub", ("hub",)),
    ("netz", ("bluetooth", "wlan", "wifi", "wi-fi", "wireless", "ethernet", "lan adapter", "network", "netzwerk",
              "802.11", "modem", "lte", "rndis", "ncm", "ecm")),
    ("mikrocontroller", ("arduino", "esp32", "esp8266", "pico", "rp2040", "teensy", "stm32", "st-link", "stlink",
                         "dfu", "bootloader", "mikrocontroller", "microcontroller", "cmsis-dap", "daplink", "j-link",
                         "jlink", "nucleo", "feather", "atmega", "samd", "nrf52", "debug probe")),
    ("usb_seriell", ("ch340", "ch341", "ch9102", "cp210", "cp2102", "cp2104", "ft232", "ft231", "ftdi", "pl2303",
                     "prolific", "usb-serial", "usb serial", "usb-seriell", "usb seriell", "uart", "serial port",
                     "serielle schnittstelle", "usbser", "cdc acm", "ttyusb", "ttyacm", "communications port",
                     "seriell", "serial")),
)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS geraete (
    key         TEXT    PRIMARY KEY,
    kind        TEXT    NOT NULL,
    name        TEXT    NOT NULL,
    port        TEXT,
    vendor_id   TEXT,
    product_id  TEXT,
    vendor      TEXT,
    product     TEXT,
    role        TEXT    NOT NULL DEFAULT 'unbekannt',
    status      TEXT    NOT NULL DEFAULT 'unbekannt',
    first_seen  REAL    NOT NULL,
    last_seen   REAL    NOT NULL,
    seen_count  INTEGER NOT NULL DEFAULT 1,
    note        TEXT    NOT NULL DEFAULT '',
    raw         TEXT    NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_geraete_status ON geraete(status);
CREATE INDEX IF NOT EXISTS idx_geraete_role ON geraete(role);
"""

_HEX4 = re.compile(r"([0-9a-f]{1,4})(?![0-9a-z])")
_PNP_VID_PID = re.compile(r"VID_([0-9A-Fa-f]{4})&PID_([0-9A-Fa-f]{4})", re.IGNORECASE)
_COM_IN_NAME = re.compile(r"\((COM\d+)\)", re.IGNORECASE)
_NMEA = re.compile(r"^\$([A-Z]{2})([A-Z]{3}),(.*)\*([0-9A-Fa-f]{2})\s*$")
_KV = re.compile(r"^\s*([A-Za-z_][\w.\-]*)\s*[=:]\s*(.+?)\s*$")
_NUMBER = re.compile(r"^[+-]?(\d+\.?\d*|\.\d+)([eE][+-]?\d+)?$")


# ------------------------------------------------------------------ Hilfen
def _short(value: Any, n: int = MAX_FIELD_CHARS) -> str:
    """Wandelt ``value`` in einen bereinigten String mit höchstens ``n`` Zeichen."""
    if value is None:
        return ""
    text = str(value).replace("\x00", "").strip()
    return text[:n]


def _norm_id(value: Any) -> str | None:
    """Normalisiert eine USB-ID (``0x0483``, ``VID_0483``, ``1155``, ``"0483"``) auf 4 hex-Zeichen klein."""
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return f"{value & 0xFFFF:04x}"
    s = str(value).strip().lower()
    if not s:
        return None
    s = re.sub(r"^(0x|vid_|pid_)", "", s)
    m = _HEX4.match(s)
    if not m:
        return None
    return m.group(1).zfill(4)


def _iso(ts: float | None) -> str | None:
    if ts is None:
        return None
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(ts))


def _platform() -> str:
    """``platform.system()`` – getrennt, damit Tests andere Betriebssysteme simulieren können."""
    return platform.system()


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None


def _vendor_name(vid: str | None, fallback: Any = None) -> str | None:
    if vid and vid in KNOWN_VENDORS:
        return KNOWN_VENDORS[vid]
    fb = _short(fallback)
    return fb or None


# ------------------------------------------------------------------ Datenklasse
@dataclass
class Device:
    """Ein erkanntes Gerät (USB, seriell oder Netz)."""

    kind: str
    name: str
    port: str | None = None
    vendor_id: str | None = None
    product_id: str | None = None
    vendor: str | None = None
    product: str | None = None
    role: str = "unbekannt"
    status: str = "unbekannt"
    raw: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.kind = _short(self.kind, 20) or "usb"
        self.name = _short(self.name) or (self.port or "Unbekanntes Gerät")
        self.port = _short(self.port) or None
        self.vendor_id = _norm_id(self.vendor_id)
        self.product_id = _norm_id(self.product_id)
        self.vendor = _short(self.vendor) or None
        self.product = _short(self.product) or None
        self.role = self.role if self.role in ROLES else "unbekannt"
        self.status = self.status if self.status in STATUSES else "unbekannt"
        if not isinstance(self.raw, dict):
            self.raw = {"wert": _short(self.raw)}

    @property
    def key(self) -> str:
        """Stabiler Schlüssel: der Port, sonst ``vid:pid:name``."""
        if self.port:
            return self.port
        vid = self.vendor_id or "????"
        pid = self.product_id or "????"
        return f"{vid}:{pid}:{self.name}"

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "art": self.kind,
            "name": self.name,
            "port": self.port,
            "vendor_id": self.vendor_id,
            "product_id": self.product_id,
            "hersteller": self.vendor,
            "produkt": self.product,
            "rolle": self.role,
            "status": self.status,
            "rohdaten": dict(self.raw),
        }


# ------------------------------------------------------------------ Klassifikation
def classify(vendor_id: str | None, product_id: str | None, name: str | None = None) -> str:
    """Ordnet einem Gerät eine Rolle aus :data:`ROLES` zu.

    Reihenfolge: bekannte Produkttabelle → Begriffe im Namen → Standardrolle des Herstellers →
    ``"unbekannt"``."""
    vid = _norm_id(vendor_id)
    pid = _norm_id(product_id)
    if vid and pid and (vid, pid) in KNOWN_PRODUCTS:
        return KNOWN_PRODUCTS[(vid, pid)][1]
    text = _short(name, 1000).lower()
    if text:
        for role, words in _NAME_RULES:
            for w in words:
                if w in text:
                    return role
    if vid and vid in VENDOR_ROLES:
        return VENDOR_ROLES[vid]
    return "unbekannt"


def _product_name(vid: str | None, pid: str | None) -> str | None:
    if vid and pid and (vid, pid) in KNOWN_PRODUCTS:
        return KNOWN_PRODUCTS[(vid, pid)][0]
    return None


def _make_device(kind: str, name: str, *, port: str | None = None, vid: Any = None, pid: Any = None,
                 vendor: Any = None, product: Any = None, status: str = "verbunden", raw: dict | None = None,
                 class_hint: str = "") -> Device:
    """Baut ein :class:`Device` und ergänzt Herstellername, Produktname und Rolle aus den Tabellen."""
    nvid, npid = _norm_id(vid), _norm_id(pid)
    known = _product_name(nvid, npid)
    product_str = _short(product) or known
    display = _short(name) or product_str or known or (port or "")
    if not display and nvid:
        display = f"USB-Gerät {nvid}:{npid or '????'}"
    role = classify(nvid, npid, " ".join(x for x in (display, product_str or "", class_hint) if x))
    return Device(kind=kind, name=display, port=port, vendor_id=nvid, product_id=npid,
                  vendor=_vendor_name(nvid, vendor), product=product_str, role=role, status=status,
                  raw=raw or {})


# ------------------------------------------------------------------ Quellen: Linux sysfs
def _read_sysfs_usb(root: str | Path = "/sys/bus/usb/devices") -> list[dict]:
    """Liest alle USB-Geräte mit ``idVendor`` unterhalb von ``root`` (Linux). Ohne sysfs: ``[]``."""
    base = Path(root)
    out: list[dict] = []
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return out
    for entry in entries:
        try:
            if not (entry / "idVendor").is_file():
                continue
        except OSError:
            continue
        rec = {"pfad": _short(entry.name)}
        for key in ("idVendor", "idProduct", "manufacturer", "product", "serial", "bDeviceClass", "busnum", "devnum"):
            val = _read_text(entry / key)
            if val is not None:
                rec[key] = _short(val)
        out.append(rec)
    return out


def _read_sysfs_tty(root: str | Path = "/sys/class/tty") -> list[dict]:
    """Liest serielle Schnittstellen aus ``/sys/class/tty`` (Linux): nur Einträge mit ``device``.

    Für USB-Geräte werden ``idVendor``/``idProduct``/``manufacturer``/``product`` aus den Elternverzeichnissen
    übernommen. Rein virtuelle Konsolen (ohne ``device``) und Platform-Ports ohne Hardwarebezug werden ausgelassen."""
    base = Path(root)
    out: list[dict] = []
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return out
    for entry in entries:
        dev = entry / "device"
        try:
            if not dev.exists():
                continue
            start = dev.resolve()
        except OSError:
            continue
        rec: dict[str, Any] = {"name": _short(entry.name)}
        subsystem = start / "subsystem"
        try:
            if subsystem.exists():
                rec["subsystem"] = _short(subsystem.resolve().name)
        except OSError:
            pass
        driver = start / "driver"
        try:
            if driver.exists():
                rec["driver"] = _short(driver.resolve().name)
        except OSError:
            pass
        node: Path | None = start
        for _ in range(5):
            if node is None:
                break
            try:
                has_vid = (node / "idVendor").is_file()
            except OSError:
                has_vid = False
            if has_vid:
                for key in ("idVendor", "idProduct", "manufacturer", "product", "serial"):
                    val = _read_text(node / key)
                    if val is not None:
                        rec[key] = _short(val)
                break
            node = node.parent if node.parent != node else None
        if rec.get("subsystem") == "platform" and "idVendor" not in rec and entry.name.startswith("ttyS"):
            # Onboard-UARTs ohne angeschlossenes Gerät (ttyS0–ttyS3) würden sonst immer erscheinen
            continue
        out.append(rec)
    return out


def _read_serial_by_id(root: str | Path = "/dev/serial/by-id") -> dict[str, str]:
    """``/dev/serial/by-id``: Zielgerät (``/dev/ttyUSB0``) -> sprechender Linkname."""
    base = Path(root)
    out: dict[str, str] = {}
    try:
        entries = sorted(base.iterdir())
    except OSError:
        return out
    for link in entries:
        try:
            target = os.path.realpath(str(link))
        except OSError:
            continue
        out[target] = _short(link.name)
    return out


def _list_dev_cu() -> list[str]:
    """macOS: alle ``/dev/cu.*`` außer dem eingebauten Bluetooth-Port."""
    try:
        names = sorted(os.listdir("/dev"))
    except OSError:
        return []
    return ["/dev/" + n for n in names if n.startswith("cu.") and "bluetooth" not in n.lower()]


# ------------------------------------------------------------------ Quellen: Windows
def _winreg_serial() -> list[tuple[str, str]]:
    """Windows-Registry ``HKLM\\HARDWARE\\DEVICEMAP\\SERIALCOMM``: ``[(Gerätename, "COM3"), …]``."""
    try:
        import winreg  # type: ignore[import-not-found]
    except ImportError:
        return []
    out: list[tuple[str, str]] = []
    try:
        key = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"HARDWARE\DEVICEMAP\SERIALCOMM")
    except OSError:
        return out
    try:
        i = 0
        while True:
            try:
                name, value, _ = winreg.EnumValue(key, i)
            except OSError:
                break
            i += 1
            out.append((_short(name), _short(value)))
    finally:
        try:
            winreg.CloseKey(key)
        except OSError:
            pass
    return out


_PNP_SCRIPT = (
    "Get-PnpDevice -PresentOnly | Where-Object { $_.InstanceId -like 'USB\\*' } | "
    "Select-Object FriendlyName, InstanceId, Class, Status, Manufacturer | ConvertTo-Json -Compress -Depth 2"
)


def _run_powershell(script: str, timeout: float = SUBPROCESS_TIMEOUT) -> str:
    """Führt ein PowerShell-Skript aus und liefert stdout (``""`` bei jedem Fehler)."""
    for exe in ("powershell", "pwsh"):
        try:
            proc = subprocess.run([exe, "-NoProfile", "-NonInteractive", "-Command", script],
                                  capture_output=True, text=True, timeout=timeout, encoding="utf-8",
                                  errors="replace")
        except FileNotFoundError:
            continue
        except subprocess.TimeoutExpired:
            log.warning("PowerShell hat nach %.0f s nicht geantwortet.", timeout)
            return ""
        except OSError as e:
            log.warning("PowerShell konnte nicht gestartet werden: %s", e)
            return ""
        if proc.returncode != 0:
            log.warning("PowerShell-Fehler (%s): %s", proc.returncode, _short(proc.stderr))
            return ""
        return proc.stdout or ""
    log.debug("Keine PowerShell gefunden.")
    return ""


def _parse_pnp_json(text: str) -> list[dict]:
    """``ConvertTo-Json``-Ausgabe (Liste oder Einzelobjekt) -> Liste von Dicts; defekt -> ``[]``."""
    text = (text or "").strip().lstrip("﻿")
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        log.warning("PowerShell lieferte kein gültiges JSON.")
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [d for d in data if isinstance(d, dict)]


def _devices_from_pnp(records: list[dict]) -> list[Device]:
    out: list[Device] = []
    for rec in records:
        inst = _short(rec.get("InstanceId") or rec.get("DeviceID") or "")
        friendly = _short(rec.get("FriendlyName") or rec.get("Name") or "")
        cls = _short(rec.get("Class") or "")
        status_raw = _short(rec.get("Status") or "").lower()
        status = "verbunden" if status_raw == "ok" else "fehler" if status_raw in ("error", "degraded") else "unbekannt"
        m = _PNP_VID_PID.search(inst)
        vid = m.group(1) if m else None
        pid = m.group(2) if m else None
        if not friendly and not vid:
            continue
        raw = {k: _short(v) for k, v in rec.items() if v is not None}
        port = None
        cm = _COM_IN_NAME.search(friendly)
        if cm:
            port = cm.group(1).upper()
        out.append(_make_device("usb", friendly, port=port, vid=vid, pid=pid, vendor=rec.get("Manufacturer"),
                                status=status, raw=raw, class_hint=cls))
    return out


# ------------------------------------------------------------------ Quellen: macOS
def _system_profiler(timeout: float = SUBPROCESS_TIMEOUT) -> str:
    """macOS ``system_profiler SPUSBDataType -json`` -> stdout (``""`` bei Fehler)."""
    try:
        proc = subprocess.run(["system_profiler", "SPUSBDataType", "-json"], capture_output=True, text=True,
                              timeout=timeout, encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""
    except subprocess.TimeoutExpired:
        log.warning("system_profiler hat nach %.0f s nicht geantwortet.", timeout)
        return ""
    except OSError as e:
        log.warning("system_profiler konnte nicht gestartet werden: %s", e)
        return ""
    if proc.returncode != 0:
        log.warning("system_profiler-Fehler (%s)", proc.returncode)
        return ""
    return proc.stdout or ""


def _parse_system_profiler(text: str) -> list[dict]:
    """Flacht die verschachtelten ``_items`` von ``system_profiler`` zu einer Liste von Geräten ab."""
    text = (text or "").strip()
    if not text:
        return []
    try:
        data = json.loads(text)
    except ValueError:
        log.warning("system_profiler lieferte kein gültiges JSON.")
        return []
    out: list[dict] = []

    def walk(items: Any, depth: int = 0) -> None:
        if depth > 12 or not isinstance(items, list):
            return
        for it in items:
            if not isinstance(it, dict):
                continue
            if "vendor_id" in it or "product_id" in it:
                out.append({k: v for k, v in it.items() if k != "_items"})
            walk(it.get("_items"), depth + 1)

    walk((data or {}).get("SPUSBDataType") if isinstance(data, dict) else None)
    return out


def _devices_from_profiler(records: list[dict]) -> list[Device]:
    out: list[Device] = []
    for rec in records:
        vid = _norm_id(rec.get("vendor_id"))
        pid = _norm_id(rec.get("product_id"))
        name = _short(rec.get("_name") or "")
        if not name and not vid:
            continue
        raw = {k: _short(v) for k, v in rec.items() if not isinstance(v, (dict, list))}
        out.append(_make_device("usb", name, vid=vid, pid=pid, vendor=rec.get("manufacturer"), raw=raw))
    return out


# ------------------------------------------------------------------ Quellen: pyserial (optional)
def _pyserial_ports() -> list[dict]:
    """``serial.tools.list_ports.comports()`` als Dicts – ``[]`` wenn pyserial fehlt."""
    try:
        from serial.tools import list_ports  # type: ignore[import-not-found]
    except Exception:  # noqa: BLE001 - jede Importstörung heißt: nicht verfügbar
        return []
    out: list[dict] = []
    try:
        ports = list(list_ports.comports())
    except Exception as e:  # noqa: BLE001
        log.warning("pyserial konnte Ports nicht auflisten: %s", e)
        return []
    for p in ports:
        out.append({
            "device": _short(getattr(p, "device", "")),
            "name": _short(getattr(p, "name", "")),
            "description": _short(getattr(p, "description", "")),
            "hwid": _short(getattr(p, "hwid", "")),
            "vid": getattr(p, "vid", None),
            "pid": getattr(p, "pid", None),
            "manufacturer": _short(getattr(p, "manufacturer", "")),
            "product": _short(getattr(p, "product", "")),
            "serial_number": _short(getattr(p, "serial_number", "")),
        })
    return out


# ------------------------------------------------------------------ Öffentliche Listen
def list_serial_ports() -> list[Device]:
    """Serielle Schnittstellen des Systems (ohne Hardware: leere Liste)."""
    found: dict[str, Device] = {}
    system = _platform()
    if system == "Linux":
        by_id = _read_serial_by_id()
        for rec in _read_sysfs_tty():
            port = "/dev/" + rec["name"]
            link = by_id.get(port)
            name = _short(rec.get("product") or link or rec["name"])
            raw = dict(rec)
            if link:
                raw["by_id"] = link
            found[port] = _make_device("seriell", name, port=port, vid=rec.get("idVendor"), pid=rec.get("idProduct"),
                                       vendor=rec.get("manufacturer"), product=rec.get("product"), raw=raw)
    elif system == "Windows":
        for devname, port in _winreg_serial():
            if not port:
                continue
            found[port] = _make_device("seriell", f"{port} ({devname})" if devname else port, port=port,
                                       raw={"geraet": devname})
    elif system == "Darwin":
        for port in _list_dev_cu():
            found[port] = _make_device("seriell", os.path.basename(port), port=port, raw={"pfad": port})
    for rec in _pyserial_ports():
        port = rec.get("device") or ""
        if not port:
            continue
        desc = rec.get("description") or ""
        name = rec.get("product") or (desc if desc.lower() != "n/a" else "") or port
        dev = _make_device("seriell", name, port=port, vid=rec.get("vid"), pid=rec.get("pid"),
                           vendor=rec.get("manufacturer"), product=rec.get("product"), raw=rec)
        old = found.get(port)
        if old is None:
            found[port] = dev
        else:
            # vorhandene Felder behalten, Lücken aus pyserial füllen
            merged_raw = dict(old.raw)
            merged_raw.update({k: v for k, v in rec.items() if v not in (None, "")})
            found[port] = Device(kind="seriell", name=old.name if old.name != port else dev.name, port=port,
                                 vendor_id=old.vendor_id or dev.vendor_id, product_id=old.product_id or dev.product_id,
                                 vendor=old.vendor or dev.vendor, product=old.product or dev.product,
                                 role=old.role if old.role != "unbekannt" else dev.role, status=old.status,
                                 raw=merged_raw)
    return [found[k] for k in sorted(found)]


def list_usb_devices() -> list[Device]:
    """USB-Geräte des Systems (ohne Hardware: leere Liste)."""
    system = _platform()
    devices: list[Device] = []
    if system == "Linux":
        for rec in _read_sysfs_usb():
            vid = _norm_id(rec.get("idVendor"))
            if vid in SKIP_VENDORS:
                continue
            cls_hint = "hub" if _short(rec.get("bDeviceClass")) == "09" else ""
            name = _short(rec.get("product") or "")
            devices.append(_make_device("usb", name, vid=vid, pid=rec.get("idProduct"),
                                        vendor=rec.get("manufacturer"), product=rec.get("product"), raw=dict(rec),
                                        class_hint=cls_hint))
    elif system == "Windows":
        devices.extend(_devices_from_pnp(_parse_pnp_json(_run_powershell(_PNP_SCRIPT))))
    elif system == "Darwin":
        devices.extend(_devices_from_profiler(_parse_system_profiler(_system_profiler())))
    return devices


def _merge_serial_usb(serial: Sequence[Device], usb: Sequence[Device]) -> list[Device]:
    """Vereinigt beide Listen, dedupliziert über ``key`` und ergänzt serielle Ports um USB-Informationen."""
    out: dict[str, Device] = {}
    for d in serial:
        out[d.key] = d
    for u in usb:
        # Windows: PnP-Name „USB-Serial CH340 (COM3)“ gehört zum Port COM3 der Registry
        if u.port and u.port in out:
            s = out[u.port]
            out[u.port] = Device(kind="seriell", name=u.name or s.name, port=u.port,
                                 vendor_id=s.vendor_id or u.vendor_id, product_id=s.product_id or u.product_id,
                                 vendor=s.vendor or u.vendor, product=s.product or u.product,
                                 role=u.role if s.role == "unbekannt" else s.role, status=u.status,
                                 raw={**s.raw, **u.raw})
            continue
        if u.key not in out:
            out[u.key] = u
    return list(out.values())


def scan() -> list[Device]:
    """Alle erkannten Geräte (USB + seriell), über ``key`` dedupliziert."""
    usb = list_usb_devices()
    serial = list_serial_ports()
    return _merge_serial_usb(serial, usb)


# ------------------------------------------------------------------ Seriell lesen
class _PosixSerial:
    """Minimaler serieller Leser über ``termios`` (nur POSIX, ohne pyserial)."""

    def __init__(self, port: str, baud: int):
        import select  # noqa: F401 - Verfügbarkeit prüfen
        import termios

        const = getattr(termios, f"B{baud}", None)
        if const is None:
            raise ValueError(f"Baudrate {baud} wird von termios nicht unterstützt.")
        self.fd = os.open(port, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
        try:
            attrs = termios.tcgetattr(self.fd)
            iflag, oflag, cflag, lflag, _ispeed, _ospeed, cc = attrs
            cflag |= termios.CLOCAL | termios.CREAD
            cflag &= ~termios.CSIZE
            cflag |= termios.CS8
            cflag &= ~(termios.PARENB | termios.CSTOPB)
            if hasattr(termios, "CRTSCTS"):
                cflag &= ~termios.CRTSCTS
            lflag &= ~(termios.ICANON | termios.ECHO | termios.ECHOE | termios.ISIG | termios.IEXTEN)
            oflag &= ~termios.OPOST
            iflag &= ~(termios.IXON | termios.IXOFF | termios.IXANY | termios.ICRNL | termios.INLCR
                       | termios.IGNCR | termios.BRKINT | termios.INPCK | termios.ISTRIP)
            cc[termios.VMIN] = 0
            cc[termios.VTIME] = 0
            termios.tcsetattr(self.fd, termios.TCSANOW, [iflag, oflag, cflag, lflag, const, const, cc])
            termios.tcflush(self.fd, termios.TCIFLUSH)
        except Exception:
            os.close(self.fd)
            raise

    def read(self, n: int, timeout: float = 0.1) -> bytes:
        import select

        r, _, _ = select.select([self.fd], [], [], max(0.0, timeout))
        if not r:
            return b""
        try:
            return os.read(self.fd, max(1, n))
        except BlockingIOError:
            return b""

    def close(self) -> None:
        try:
            os.close(self.fd)
        except OSError:
            pass


class _PySerialReader:
    def __init__(self, port: str, baud: int):
        import serial  # type: ignore[import-not-found]

        self.ser = serial.Serial(port, baudrate=baud, timeout=0.1)

    def read(self, n: int, timeout: float = 0.1) -> bytes:
        self.ser.timeout = max(0.01, timeout)
        return self.ser.read(max(1, n)) or b""

    def close(self) -> None:
        try:
            self.ser.close()
        except Exception:  # noqa: BLE001
            pass


def _open_serial(port: str, baud: int):
    """Öffnet ``port``: pyserial falls installiert, sonst POSIX ``termios``. Windows ohne pyserial -> ValueError."""
    try:
        import serial  # type: ignore[import-not-found]  # noqa: F401
        have_pyserial = True
    except Exception:  # noqa: BLE001
        have_pyserial = False
    if have_pyserial:
        try:
            return _PySerialReader(port, baud)
        except Exception as e:  # noqa: BLE001
            raise ValueError(f"Port »{_short(port, 80)}« konnte nicht geöffnet werden: {_short(e, 200)}") from e
    if os.name != "posix":
        raise ValueError("Serielles Lesen unter Windows braucht pyserial: »pip install pyserial«.")
    try:
        return _PosixSerial(port, baud)
    except FileNotFoundError:
        raise ValueError(f"Port »{_short(port, 80)}« existiert nicht.") from None
    except PermissionError:
        raise ValueError(f"Keine Berechtigung für Port »{_short(port, 80)}« (Gruppe dialout?).") from None
    except OSError as e:
        raise ValueError(f"Port »{_short(port, 80)}« konnte nicht geöffnet werden: {_short(e, 200)}") from e


def _check_baud(baud: Any) -> int:
    try:
        b = int(baud)
    except (TypeError, ValueError):
        raise ValueError("Baudrate muss eine ganze Zahl sein.") from None
    if b not in COMMON_BAUDS:
        raise ValueError("Ungültige Baudrate. Gängig sind: " + ", ".join(str(x) for x in COMMON_BAUDS) + ".")
    return b


def _check_seconds(seconds: Any) -> float:
    try:
        s = float(seconds)
    except (TypeError, ValueError):
        raise ValueError("Sekunden müssen eine Zahl sein.") from None
    if not (MIN_READ_SECONDS <= s <= MAX_READ_SECONDS):
        raise ValueError(f"Sekunden müssen zwischen {MIN_READ_SECONDS:g} und {MAX_READ_SECONDS:g} liegen.")
    return s


def read_serial(port: str, baud: int = DEFAULT_BAUD, seconds: float = 2.0, max_bytes: int = 4096) -> dict:
    """Liest ``seconds`` lang Rohdaten von ``port``.

    Liefert ``{"port", "baud", "bytes", "text", "zeilen", "dauer_s"}``; der Text ist UTF-8 mit
    Ersetzungen. Ungültige Baudrate, Sekunden außerhalb 0,1–30 oder ein nicht öffnbarer Port
    führen zu ``ValueError`` mit deutschem Text."""
    port = _short(port)
    if not port:
        raise ValueError("Kein Port angegeben.")
    b = _check_baud(baud)
    s = _check_seconds(seconds)
    try:
        limit = int(max_bytes)
    except (TypeError, ValueError):
        raise ValueError("max_bytes muss eine ganze Zahl sein.") from None
    limit = max(1, min(limit, MAX_READ_BYTES))
    reader = _open_serial(port, b)
    chunks: list[bytes] = []
    got = 0
    start = time.monotonic()
    deadline = start + s
    try:
        while got < limit:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            data = reader.read(limit - got, min(0.1, remaining))
            if data:
                chunks.append(data)
                got += len(data)
            else:
                time.sleep(min(0.01, max(0.0, remaining)))
    finally:
        reader.close()
    dauer = time.monotonic() - start
    blob = b"".join(chunks)[:limit]
    text = blob.decode("utf-8", errors="replace")
    lines = [ln.strip("\r") for ln in text.split("\n")]
    lines = [ln for ln in lines if ln.strip()]
    return {"port": port, "baud": b, "bytes": len(blob), "text": text, "zeilen": lines, "dauer_s": round(dauer, 3)}


# ------------------------------------------------------------------ Telemetrie
def nmea_checksum(body: str) -> str:
    """XOR-Prüfsumme über ``body`` (der Teil zwischen ``$`` und ``*``) als zwei Hex-Zeichen groß."""
    cs = 0
    for ch in body:
        cs ^= ord(ch)
    return f"{cs:02X}"


def _nmea_coord(value: str, hemi: str) -> float | None:
    """``4807.038`` + ``N`` -> 48.1173 (Grad dezimal); ungültig -> ``None``."""
    value = value.strip()
    if not value:
        return None
    try:
        raw = float(value)
    except ValueError:
        return None
    deg = int(raw // 100)
    minutes = raw - deg * 100
    dec = deg + minutes / 60.0
    if hemi.strip().upper() in ("S", "W"):
        dec = -dec
    return round(dec, 7)


def _to_number(text: str) -> float | int | None:
    text = text.strip()
    if re.fullmatch(r"[+-]?\d+,\d+", text):          # deutsches Dezimalkomma
        text = text.replace(",", ".")
    if not _NUMBER.match(text):
        return None
    try:
        if re.fullmatch(r"[+-]?\d+", text):
            return int(text)
        return float(text)
    except ValueError:
        return None


def _parse_nmea_line(line: str) -> dict | None:
    """Ein NMEA-Satz -> Werte-Dict, ``None`` wenn kein gültiger Satz oder Prüfsumme falsch."""
    m = _NMEA.match(line.strip())
    if not m:
        return None
    talker, kind, payload, given = m.groups()
    body = f"{talker}{kind},{payload}"
    if nmea_checksum(body) != given.upper():
        return None
    f = payload.split(",")
    vals: dict[str, Any] = {"nmea_talker": talker}
    if kind == "GGA" and len(f) >= 9:
        lat = _nmea_coord(f[1], f[2])
        lon = _nmea_coord(f[3], f[4])
        if lat is not None:
            vals["lat"] = lat
        if lon is not None:
            vals["lon"] = lon
        if f[0]:
            vals["zeit_utc"] = f[0]
        q = _to_number(f[5])
        if q is not None:
            vals["fix"] = int(q)
        sats = _to_number(f[6])
        if sats is not None:
            vals["satelliten"] = int(sats)
        hdop = _to_number(f[7])
        if hdop is not None:
            vals["hdop"] = float(hdop)
        alt = _to_number(f[8])
        if alt is not None:
            vals["hoehe_m"] = float(alt)
        return vals
    if kind == "RMC" and len(f) >= 8:
        if f[0]:
            vals["zeit_utc"] = f[0]
        status = f[1].strip().upper()
        if status in ("A", "V"):
            vals.setdefault("fix", 1 if status == "A" else 0)
            vals["gueltig"] = status == "A"
        lat = _nmea_coord(f[2], f[3])
        lon = _nmea_coord(f[4], f[5])
        if lat is not None:
            vals["lat"] = lat
        if lon is not None:
            vals["lon"] = lon
        knots = _to_number(f[6])
        if knots is not None:
            vals["geschwindigkeit_kmh"] = round(float(knots) * 1.852, 3)
        course = _to_number(f[7])
        if course is not None:
            vals["kurs_deg"] = float(course)
        if len(f) >= 9 and f[8]:
            vals["datum"] = f[8]
        return vals
    if kind == "VTG" and len(f) >= 7:
        course = _to_number(f[0])
        if course is not None:
            vals["kurs_deg"] = float(course)
        kmh = _to_number(f[6])
        if kmh is not None:
            vals["geschwindigkeit_kmh"] = float(kmh)
        return vals
    # anderer gültiger Satz (GSA, GSV …): zählt als NMEA, trägt aber keine Werte bei
    return vals


def parse_telemetry(text: str) -> dict:
    """Erkennt NMEA-, Key=Value- und JSON-Zeilen in ``text``.

    Liefert ``{"format": "nmea"|"kv"|"json"|"unbekannt", "werte": {...}, "zeilen": n,
    "erkannt": n_gueltig, "ungueltig": n_fehlerhaft}``. NMEA-Sätze mit falscher Prüfsumme werden
    ignoriert (und als ungültig gezählt)."""
    text = "" if text is None else str(text)
    lines = [ln.strip() for ln in text.replace("\r", "\n").split("\n")]
    lines = [ln for ln in lines if ln]
    werte: dict[str, Any] = {}
    counts = {"nmea": 0, "json": 0, "kv": 0}
    invalid = 0
    for ln in lines:
        if ln.startswith("$"):
            vals = _parse_nmea_line(ln)
            if vals is None:
                invalid += 1
                continue
            counts["nmea"] += 1
            vals.pop("gueltig", None)
            # GGA-Fixqualität hat Vorrang vor dem RMC-Status
            if "fix" in vals and "fix" in werte and vals.get("nmea_talker") and ln[3:6] == "RMC":
                vals.pop("fix")
            werte.update(vals)
            continue
        if ln.startswith("{") and ln.endswith("}"):
            try:
                obj = json.loads(ln)
            except ValueError:
                obj = None
            if isinstance(obj, dict):
                counts["json"] += 1
                for k, v in obj.items():
                    werte[_short(k, 80)] = v if isinstance(v, (int, float, str, bool, type(None))) else json.loads(
                        json.dumps(v))
                continue
            invalid += 1
            continue
        m = _KV.match(ln)
        if m:
            key, val = m.group(1), m.group(2)
            num = _to_number(val)
            werte[_short(key, 80)] = num if num is not None else _short(val)
            counts["kv"] += 1
            continue
        invalid += 1
    if counts["nmea"]:
        fmt = "nmea"
    elif counts["json"]:
        fmt = "json"
    elif counts["kv"]:
        fmt = "kv"
    else:
        fmt = "unbekannt"
    werte.pop("nmea_talker", None) if fmt != "nmea" else None
    return {"format": fmt, "werte": werte, "zeilen": len(lines), "erkannt": sum(counts.values()),
            "ungueltig": invalid}


# ------------------------------------------------------------------ Speicher
class DeviceStore:
    """SQLite-Verlauf erkannter Geräte: zuerst/zuletzt gesehen, Sichtungen, Notiz. Thread-sicher."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(self.path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.executescript(_SCHEMA)
        self._db.commit()
        self._closed = False

    # ------------------------------------------------------------ intern
    @staticmethod
    def _row(r: sqlite3.Row) -> dict:
        try:
            raw = json.loads(r["raw"] or "{}")
        except ValueError:
            raw = {}
        return {
            "key": r["key"], "art": r["kind"], "name": r["name"], "port": r["port"],
            "vendor_id": r["vendor_id"], "product_id": r["product_id"], "hersteller": r["vendor"],
            "produkt": r["product"], "rolle": r["role"], "status": r["status"],
            "zuerst_gesehen": _iso(r["first_seen"]), "zuletzt_gesehen": _iso(r["last_seen"]),
            "sichtungen": int(r["seen_count"]), "notiz": r["note"] or "", "rohdaten": raw,
        }

    def _check_open(self) -> None:
        if self._closed:
            raise ValueError("DeviceStore ist geschlossen.")

    # ------------------------------------------------------------ API
    def update(self, devices: Sequence[Device]) -> list[dict]:
        """Upsert aller ``devices``; alle anderen Einträge werden als ``getrennt`` markiert. Liefert :meth:`list`."""
        now = time.time()
        seen: list[str] = []
        with self._lock:
            self._check_open()
            for d in devices:
                if not isinstance(d, Device):
                    continue
                key = d.key
                seen.append(key)
                raw = json.dumps(d.raw, ensure_ascii=False, default=str)[:4000]
                self._db.execute(
                    """INSERT INTO geraete (key, kind, name, port, vendor_id, product_id, vendor, product, role,
                                            status, first_seen, last_seen, seen_count, note, raw)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, '', ?)
                       ON CONFLICT(key) DO UPDATE SET
                           kind = excluded.kind, name = excluded.name, port = excluded.port,
                           vendor_id = COALESCE(excluded.vendor_id, geraete.vendor_id),
                           product_id = COALESCE(excluded.product_id, geraete.product_id),
                           vendor = COALESCE(excluded.vendor, geraete.vendor),
                           product = COALESCE(excluded.product, geraete.product),
                           role = CASE WHEN excluded.role = 'unbekannt' THEN geraete.role ELSE excluded.role END,
                           status = excluded.status, last_seen = excluded.last_seen,
                           seen_count = geraete.seen_count + 1, raw = excluded.raw""",
                    (key, d.kind, d.name, d.port, d.vendor_id, d.product_id, d.vendor, d.product, d.role,
                     d.status, now, now, raw),
                )
            if seen:
                marks = ",".join("?" for _ in seen)
                self._db.execute(f"UPDATE geraete SET status = 'getrennt' WHERE status != 'getrennt' "
                                 f"AND key NOT IN ({marks})", seen)
            else:
                self._db.execute("UPDATE geraete SET status = 'getrennt' WHERE status != 'getrennt'")
            self._db.commit()
            return self.list()

    def list(self, connected_only: bool = False) -> list[dict]:
        with self._lock:
            self._check_open()
            sql = "SELECT * FROM geraete"
            if connected_only:
                sql += " WHERE status != 'getrennt'"
            sql += " ORDER BY last_seen DESC, key"
            return [self._row(r) for r in self._db.execute(sql)]

    def get(self, key: str) -> dict | None:
        with self._lock:
            self._check_open()
            r = self._db.execute("SELECT * FROM geraete WHERE key = ?", (str(key),)).fetchone()
            return self._row(r) if r else None

    def note(self, key: str, text: str) -> None:
        """Setzt die Notiz eines Geräts; unbekannter Schlüssel -> ``ValueError``."""
        text = _short(text, 2000)
        with self._lock:
            self._check_open()
            cur = self._db.execute("UPDATE geraete SET note = ? WHERE key = ?", (text, str(key)))
            self._db.commit()
            if cur.rowcount == 0:
                raise ValueError(f"Unbekanntes Gerät »{_short(key, 80)}«.")

    def forget(self, key: str) -> bool:
        with self._lock:
            self._check_open()
            cur = self._db.execute("DELETE FROM geraete WHERE key = ?", (str(key),))
            self._db.commit()
            return cur.rowcount > 0

    def stats(self) -> dict:
        with self._lock:
            self._check_open()
            total = self._db.execute("SELECT COUNT(*) FROM geraete").fetchone()[0]
            connected = self._db.execute("SELECT COUNT(*) FROM geraete WHERE status != 'getrennt'").fetchone()[0]
            roles = {r[0]: r[1] for r in self._db.execute(
                "SELECT role, COUNT(*) FROM geraete GROUP BY role ORDER BY role")}
            last = self._db.execute("SELECT MAX(last_seen) FROM geraete").fetchone()[0]
            return {"gesamt": int(total), "verbunden": int(connected), "getrennt": int(total - connected),
                    "rollen": roles, "zuletzt": _iso(last)}

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._db.close()
                self._closed = True


# ------------------------------------------------------------------ Textausgabe
def format_device(d: Device | dict) -> str:
    """Eine Zeile für ein Gerät (deutsch)."""
    data = d.to_dict() if isinstance(d, Device) else d
    parts = [f"- {data.get('name') or data.get('key')}"]
    if data.get("port"):
        parts.append(f"Port {data['port']}")
    ids = []
    if data.get("vendor_id"):
        ids.append(f"{data['vendor_id']}:{data.get('product_id') or '????'}")
    if data.get("hersteller"):
        ids.append(data["hersteller"])
    if ids:
        parts.append("[" + ", ".join(ids) + "]")
    parts.append(f"Rolle: {data.get('rolle', 'unbekannt')}")
    parts.append(f"Status: {data.get('status', 'unbekannt')}")
    if data.get("sichtungen"):
        parts.append(f"{data['sichtungen']}× gesehen")
    if data.get("notiz"):
        parts.append(f"Notiz: {data['notiz']}")
    return " | ".join(parts)


def format_scan(devices: Sequence[Device], store_rows: Sequence[dict] | None = None) -> str:
    """Scanergebnis als deutscher Text; mit ``store_rows`` auch früher gesehene, jetzt getrennte Geräte."""
    lines: list[str] = []
    if not devices:
        lines.append("Keine angeschlossenen Geräte erkannt.")
    else:
        usb = [d for d in devices if d.kind == "usb"]
        ser = [d for d in devices if d.kind == "seriell"]
        lines.append(f"{len(devices)} Gerät(e) erkannt ({len(usb)} USB, {len(ser)} seriell):")
        for d in devices:
            lines.append(format_device(d))
    if store_rows:
        gone = [r for r in store_rows if r.get("status") == "getrennt"]
        if gone:
            lines.append("")
            lines.append(f"Früher gesehen, jetzt getrennt ({len(gone)}):")
            for r in gone[:20]:
                lines.append(f"- {r['name']} (zuletzt {r.get('zuletzt_gesehen')})")
    return "\n".join(lines)


def format_telemetry(result: dict, parsed: dict) -> str:
    lines = [f"Port {result['port']} @ {result['baud']} Baud: {result['bytes']} Byte in {result['dauer_s']} s, "
             f"{len(result['zeilen'])} Zeile(n)."]
    if result["bytes"] == 0:
        lines.append("Keine Daten empfangen (Baudrate prüfen, sendet das Gerät?).")
        return "\n".join(lines)
    lines.append(f"Format: {parsed['format']} ({parsed['erkannt']} erkannt, {parsed['ungueltig']} ungültig)")
    if parsed["werte"]:
        lines.append("Werte:")
        for k, v in list(parsed["werte"].items())[:40]:
            lines.append(f"  {k}: {v}")
    lines.append("Rohdaten (Anfang):")
    for ln in result["zeilen"][:15]:
        lines.append("  " + _short(ln, 160))
    return "\n".join(lines)


# ------------------------------------------------------------------ Werkzeuge
def _p_number(desc: str, default: float | None = None) -> dict:
    spec: dict[str, Any] = {"type": "number", "description": desc}
    if default is not None:
        spec["default"] = default
    return spec


def _p_int(desc: str, default: int | None = None) -> dict:
    spec: dict[str, Any] = {"type": "integer", "description": desc}
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


TOOL_NAMES = ("geraete_scannen", "seriell_lesen")


def register_tools(registry: "ToolRegistry", store: DeviceStore | None = None) -> None:
    """Registriert ``geraete_scannen`` und ``seriell_lesen`` (beide nur lesend, nicht gefährlich)."""
    from .tools import Tool

    def tool_scan() -> str:
        devices = scan()
        rows = store.update(devices) if store is not None else None
        return format_scan(devices, rows)

    def tool_read(port: str, baud: int = DEFAULT_BAUD, sekunden: float = 2.0) -> str:
        try:
            sek = float(sekunden)
        except (TypeError, ValueError):
            sek = 2.0
        sek = max(MIN_READ_SECONDS, min(MAX_READ_SECONDS, sek))
        result = read_serial(port, baud=baud, seconds=sek)
        parsed = parse_telemetry(result["text"])
        return format_telemetry(result, parsed)

    registry.register(Tool(
        name="geraete_scannen",
        description="Erkennt angeschlossene USB-Geräte und serielle Schnittstellen (Flugsteuerungen, "
                    "Mikrocontroller, GNSS-Empfänger, USB-Seriell-Wandler …) und nennt Rolle, IDs und Port.",
        parameters=_schema({}, []),
        fn=tool_scan,
        dangerous=False,
    ))
    registry.register(Tool(
        name="seriell_lesen",
        description="Liest einige Sekunden Rohdaten von einer seriellen Schnittstelle (nur lesen) und erkennt "
                    "NMEA/GPS, Key=Value- und JSON-Telemetrie.",
        parameters=_schema({"port": _p_str("Port, z. B. COM3 oder /dev/ttyUSB0", "/dev/ttyUSB0"),
                            "baud": _p_int("Baudrate (gängig: 9600, 57600, 115200)", DEFAULT_BAUD),
                            "sekunden": _p_number("Lesedauer in Sekunden (0,1–30)", 2)}, ["port"]),
        fn=tool_read,
        dangerous=False,
    ))


__all__ = [
    "COMMON_BAUDS", "DEFAULT_BAUD", "KINDS", "KNOWN_PRODUCTS", "KNOWN_VENDORS", "MAX_READ_SECONDS",
    "MIN_READ_SECONDS", "ROLES", "STATUSES", "TOOL_NAMES", "VENDOR_ROLES", "Device", "DeviceStore", "classify",
    "format_device", "format_scan", "format_telemetry", "list_serial_ports", "list_usb_devices", "nmea_checksum",
    "parse_telemetry", "read_serial", "register_tools", "scan",
]
