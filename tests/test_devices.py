"""Tests für obito.devices – ohne echte Hardware (Quellen werden gepatcht)."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path

from obito import devices
from obito.devices import Device, DeviceStore
from obito.tools import ToolRegistry


def _nmea(body: str) -> str:
    return f"${body}*{devices.nmea_checksum(body)}"


class ClassifyTest(unittest.TestCase):
    def test_known_product_wins(self):
        self.assertEqual(devices.classify("0483", "5740", "irgendwas"), "flugsteuerung")
        self.assertEqual(devices.classify("1a86", "7523", None), "usb_seriell")
        self.assertEqual(devices.classify("2341", "0043", None), "mikrocontroller")

    def test_vendor_tables_are_real_ids(self):
        self.assertGreaterEqual(len(devices.KNOWN_VENDORS), 20)
        for vid in devices.KNOWN_VENDORS:
            self.assertRegex(vid, r"^[0-9a-f]{4}$")
        self.assertEqual(devices.KNOWN_VENDORS["2ca3"], "DJI")

    def test_name_heuristics(self):
        self.assertEqual(devices.classify(None, None, "Betaflight STM32"), "flugsteuerung")
        self.assertEqual(devices.classify(None, None, "USB Mass Storage Device"), "speicher")
        self.assertEqual(devices.classify(None, None, "HD Webcam C920"), "kamera")
        self.assertEqual(devices.classify(None, None, "USB Keyboard"), "eingabe")
        self.assertEqual(devices.classify(None, None, "u-blox GPS receiver"), "sensor")
        self.assertEqual(devices.classify(None, None, "Generic USB Hub"), "hub")
        self.assertEqual(devices.classify(None, None, "Dingsbums"), "unbekannt")

    def test_vendor_only_hint(self):
        self.assertEqual(devices.classify("2ca3", "ffff", "Unbekanntes DJI Gerät"), "drohne")


class DeviceTest(unittest.TestCase):
    def test_key_prefers_port(self):
        d = Device(kind="seriell", name="x", port="COM3", vendor_id=None, product_id=None, vendor=None,
                   product=None, role="unbekannt", status="verbunden", raw={})
        self.assertEqual(d.key, "COM3")
        u = Device(kind="usb", name="Stick", port=None, vendor_id="0781", product_id="5567", vendor=None,
                   product=None, role="speicher", status="verbunden", raw={})
        self.assertIn("0781:5567", u.key)

    def test_to_dict_german_keys(self):
        d = Device(kind="usb", name="x", port=None, vendor_id="0483", product_id="5740", vendor="ST",
                   product="VCP", role="flugsteuerung", status="verbunden", raw={"a": 1})
        data = d.to_dict()
        for key in ("art", "name", "rolle", "status", "vendor_id", "product_id", "hersteller", "produkt"):
            self.assertIn(key, data)
        self.assertEqual(data["rolle"], "flugsteuerung")
        json.dumps(data)


class TelemetryTest(unittest.TestCase):
    def test_gga_parsing(self):
        line = _nmea("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,")
        res = devices.parse_telemetry(line)
        self.assertEqual(res["format"], "nmea")
        w = res["werte"]
        self.assertAlmostEqual(w["lat"], 48.1173, places=3)
        self.assertAlmostEqual(w["lon"], 11.5167, places=3)
        self.assertEqual(w["satelliten"], 8)
        self.assertAlmostEqual(w["hoehe_m"], 545.4)

    def test_rmc_parsing_speed_and_course(self):
        line = _nmea("GPRMC,123519,A,4807.038,N,01131.000,W,022.4,084.4,230394,003.1,W")
        w = devices.parse_telemetry(line)["werte"]
        self.assertLess(w["lon"], 0)
        self.assertAlmostEqual(w["geschwindigkeit_kmh"], 22.4 * 1.852, places=1)
        self.assertAlmostEqual(w["kurs_deg"], 84.4)

    def test_invalid_checksum_ignored(self):
        bad = "$GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,*00"
        res = devices.parse_telemetry(bad)
        self.assertNotIn("lat", res["werte"])
        self.assertGreaterEqual(res.get("ungueltig", 0), 1)

    def test_key_value_and_json(self):
        res = devices.parse_telemetry("akku=15.8\nstrom = 12,5\ntemp: 41")
        self.assertEqual(res["format"], "kv")
        self.assertEqual(res["werte"]["akku"], 15.8)
        self.assertEqual(res["werte"]["strom"], 12.5)
        res2 = devices.parse_telemetry('{"alt": 12.5, "sats": 9}')
        self.assertEqual(res2["format"], "json")
        self.assertEqual(res2["werte"]["sats"], 9)

    def test_unknown_text(self):
        res = devices.parse_telemetry("Hallo Welt\nnichts zu sehen")
        self.assertEqual(res["format"], "unbekannt")

    def test_checksum_helper(self):
        self.assertEqual(devices.nmea_checksum("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,"), "47")


class SysfsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)

    def tearDown(self):
        self.tmp.cleanup()

    def _usb(self, name: str, vid: str, pid: str, product: str = "", manufacturer: str = "", cls: str = "00"):
        d = self.root / "usb" / name
        d.mkdir(parents=True)
        (d / "idVendor").write_text(vid + "\n")
        (d / "idProduct").write_text(pid + "\n")
        (d / "bDeviceClass").write_text(cls + "\n")
        if product:
            (d / "product").write_text(product + "\n")
        if manufacturer:
            (d / "manufacturer").write_text(manufacturer + "\n")
        return d

    def test_read_sysfs_usb(self):
        self._usb("1-1", "0483", "5740", "STM32 Virtual ComPort", "STMicroelectronics")
        self._usb("1-2", "1d6b", "0002", "xHCI Host Controller", "Linux", cls="09")
        (self.root / "usb" / "usb1").mkdir()   # ohne idVendor → ignoriert
        recs = devices._read_sysfs_usb(self.root / "usb")
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["idVendor"], "0483")
        self.assertEqual(recs[0]["product"], "STM32 Virtual ComPort")

    def test_read_sysfs_usb_missing_root(self):
        self.assertEqual(devices._read_sysfs_usb(self.root / "fehlt"), [])

    def test_read_sysfs_tty_with_usb_parent(self):
        usb = self._usb("1-3", "1a86", "7523", "USB Serial", "QinHeng")
        iface = usb / "1-3:1.0"
        iface.mkdir()
        ttydir = self.root / "tty" / "ttyUSB0"
        ttydir.mkdir(parents=True)
        os.symlink(iface, ttydir / "device")
        plat = self.root / "tty" / "ttyS0"
        plat.mkdir()
        pdev = self.root / "platform" / "serial8250"
        pdev.mkdir(parents=True)
        subsys = self.root / "bus" / "platform"
        subsys.mkdir(parents=True)
        os.symlink(subsys, pdev / "subsystem")
        os.symlink(pdev, plat / "device")
        (self.root / "tty" / "tty0").mkdir()   # ohne device → ignoriert
        recs = devices._read_sysfs_tty(self.root / "tty")
        names = [r["name"] for r in recs]
        self.assertIn("ttyUSB0", names)
        self.assertNotIn("ttyS0", names)        # Onboard-UART ohne Gerät ausgeblendet
        rec = next(r for r in recs if r["name"] == "ttyUSB0")
        self.assertEqual(rec["idVendor"], "1a86")
        self.assertEqual(rec["product"], "USB Serial")

    def test_list_serial_ports_linux(self):
        orig = (devices._platform, devices._read_sysfs_tty, devices._read_serial_by_id, devices._pyserial_ports)
        devices._platform = lambda: "Linux"
        devices._read_sysfs_tty = lambda: [{"name": "ttyUSB0", "idVendor": "0403", "idProduct": "6001",
                                            "manufacturer": "FTDI", "product": "FT232R USB UART"}]
        devices._read_serial_by_id = lambda: {"/dev/ttyUSB0": "usb-FTDI_FT232R_USB_UART_A1-if00-port0"}
        devices._pyserial_ports = lambda: []
        try:
            ports = devices.list_serial_ports()
        finally:
            devices._platform, devices._read_sysfs_tty, devices._read_serial_by_id, devices._pyserial_ports = orig
        self.assertEqual(len(ports), 1)
        self.assertEqual(ports[0].port, "/dev/ttyUSB0")
        self.assertEqual(ports[0].role, "usb_seriell")
        self.assertEqual(ports[0].vendor, "FTDI")
        self.assertEqual(ports[0].kind, "seriell")

    def test_list_usb_devices_linux_skips_root_hubs(self):
        orig = (devices._platform, devices._read_sysfs_usb)
        devices._platform = lambda: "Linux"
        devices._read_sysfs_usb = lambda: [
            {"idVendor": "1d6b", "idProduct": "0002", "product": "xHCI Host Controller"},
            {"idVendor": "2ca3", "idProduct": "001f", "product": "DJI Device", "manufacturer": "DJI"},
        ]
        try:
            usb = devices.list_usb_devices()
        finally:
            devices._platform, devices._read_sysfs_usb = orig
        self.assertEqual(len(usb), 1)
        self.assertEqual(usb[0].role, "drohne")
        self.assertEqual(usb[0].vendor, "DJI")


class WindowsSourcesTest(unittest.TestCase):
    def test_pnp_json_list_and_single(self):
        recs = devices._parse_pnp_json(json.dumps([{"FriendlyName": "A"}, {"FriendlyName": "B"}]))
        self.assertEqual(len(recs), 2)
        recs = devices._parse_pnp_json(json.dumps({"FriendlyName": "A"}))
        self.assertEqual(len(recs), 1)
        self.assertEqual(devices._parse_pnp_json("kein json {"), [])
        self.assertEqual(devices._parse_pnp_json(""), [])
        self.assertEqual(devices._parse_pnp_json("﻿[]"), [])

    def test_devices_from_pnp(self):
        recs = [{"Class": "Ports", "FriendlyName": "USB-SERIAL CH340 (COM3)", "InstanceId": "USB\\VID_1A86&PID_7523\\5&1",
                 "Status": "OK", "Manufacturer": "wch.cn"},
                {"Class": "USB", "FriendlyName": "Unbekannt", "InstanceId": "USB\\VID_0483&PID_DF11\\1", "Status": "Error"}]
        devs = devices._devices_from_pnp(recs)
        self.assertEqual(len(devs), 2)
        self.assertEqual(devs[0].port, "COM3")
        self.assertEqual(devs[0].vendor_id, "1a86")
        self.assertEqual(devs[0].role, "usb_seriell")
        self.assertEqual(devs[1].status, "fehler")
        self.assertEqual(devs[1].role, "flugsteuerung")

    def test_list_usb_windows_uses_powershell(self):
        orig = (devices._platform, devices._run_powershell)
        devices._platform = lambda: "Windows"
        devices._run_powershell = lambda script, timeout=15: json.dumps(
            [{"Class": "USB", "FriendlyName": "Pixhawk", "InstanceId": "USB\\VID_1209&PID_5741\\1", "Status": "OK"}])
        try:
            devs = devices.list_usb_devices()
        finally:
            devices._platform, devices._run_powershell = orig
        self.assertEqual(len(devs), 1)
        self.assertEqual(devs[0].role, "flugsteuerung")

    def test_list_serial_windows_registry(self):
        orig = (devices._platform, devices._winreg_serial, devices._pyserial_ports)
        devices._platform = lambda: "Windows"
        devices._winreg_serial = lambda: [("\\Device\\USBSER000", "COM5"), ("\\Device\\Serial0", "COM1")]
        devices._pyserial_ports = lambda: []
        try:
            ports = devices.list_serial_ports()
        finally:
            devices._platform, devices._winreg_serial, devices._pyserial_ports = orig
        self.assertEqual([p.port for p in ports], ["COM1", "COM5"])

    def test_scan_merges_usb_info_into_serial(self):
        orig = (devices.list_usb_devices, devices.list_serial_ports)
        serial = [devices._make_device("seriell", "COM3", port="COM3", raw={})]
        usb = [devices._make_device("usb", "USB-SERIAL CH340 (COM3)", port="COM3", vid="1a86", pid="7523", raw={}),
               devices._make_device("usb", "Stick", vid="0781", pid="5567", raw={})]
        devices.list_usb_devices = lambda: usb
        devices.list_serial_ports = lambda: serial
        try:
            found = devices.scan()
        finally:
            devices.list_usb_devices, devices.list_serial_ports = orig
        self.assertEqual(len(found), 2)
        com = next(d for d in found if d.port == "COM3")
        self.assertEqual(com.vendor_id, "1a86")
        self.assertEqual(com.role, "usb_seriell")
        self.assertEqual(com.kind, "seriell")

    def test_no_hardware_gives_empty_lists(self):
        orig = (devices._platform, devices._read_sysfs_usb, devices._read_sysfs_tty, devices._pyserial_ports,
                devices._read_serial_by_id)
        devices._platform = lambda: "Linux"
        devices._read_sysfs_usb = lambda: []
        devices._read_sysfs_tty = lambda: []
        devices._pyserial_ports = lambda: []
        devices._read_serial_by_id = lambda: {}
        try:
            self.assertEqual(devices.scan(), [])
        finally:
            (devices._platform, devices._read_sysfs_usb, devices._read_sysfs_tty, devices._pyserial_ports,
             devices._read_serial_by_id) = orig


class ReadSerialTest(unittest.TestCase):
    def test_invalid_baud_and_seconds(self):
        with self.assertRaises(ValueError):
            devices.read_serial("/dev/ttyUSB0", baud=12345)
        with self.assertRaises(ValueError):
            devices.read_serial("/dev/ttyUSB0", seconds=0)
        with self.assertRaises(ValueError):
            devices.read_serial("/dev/ttyUSB0", seconds=999)
        with self.assertRaises(ValueError):
            devices.read_serial("", seconds=1)

    def test_reads_with_fake_reader(self):
        class _Reader:
            def __init__(self):
                self.chunks = [b"$GPGGA,1\r\n", b"akku=15.8\n", b""]

            def read(self, n, timeout=0.1):
                return self.chunks.pop(0) if self.chunks else b""

            def close(self):
                pass

        orig = devices._open_serial
        devices._open_serial = lambda port, baud: _Reader()
        try:
            res = devices.read_serial("/dev/ttyFAKE", baud=9600, seconds=0.2)
        finally:
            devices._open_serial = orig
        self.assertEqual(res["baud"], 9600)
        self.assertEqual(res["zeilen"], ["$GPGGA,1", "akku=15.8"])
        self.assertGreater(res["bytes"], 0)
        self.assertLessEqual(res["dauer_s"], 1.0)

    def test_missing_port_error(self):
        orig = devices._open_serial

        def boom(port, baud):
            raise ValueError("Port »x« existiert nicht.")

        devices._open_serial = boom
        try:
            with self.assertRaises(ValueError):
                devices.read_serial("/dev/ttyNOPE", seconds=0.1)
        finally:
            devices._open_serial = orig


class DeviceStoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.store = DeviceStore(Path(self.tmp.name) / "geraete.db")

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def _dev(self, port="COM3", vid="1a86", pid="7523", name="CH340"):
        return devices._make_device("seriell", name, port=port, vid=vid, pid=pid, raw={"x": 1})

    def test_upsert_and_disconnect(self):
        rows = self.store.update([self._dev(), self._dev(port="COM4", vid="0483", pid="5740")])
        self.assertEqual(len(rows), 2)
        rows = self.store.update([self._dev()])
        by_port = {r["port"]: r for r in rows}
        self.assertEqual(by_port["COM3"]["status"], "verbunden")
        self.assertEqual(by_port["COM3"]["sichtungen"], 2)
        self.assertEqual(by_port["COM4"]["status"], "getrennt")
        self.assertEqual(len(self.store.list(connected_only=True)), 1)

    def test_note_forget_stats(self):
        self.store.update([self._dev()])
        key = self.store.list()[0]["key"]
        self.store.note(key, "Flugsteuerung am Prüfstand")
        self.assertEqual(self.store.get(key)["notiz"], "Flugsteuerung am Prüfstand")
        stats = self.store.stats()
        self.assertEqual(stats["gesamt"], 1)
        self.assertTrue(self.store.forget(key))
        self.assertFalse(self.store.forget(key))
        self.assertEqual(self.store.stats()["gesamt"], 0)

    def test_empty_update_marks_all_disconnected(self):
        self.store.update([self._dev()])
        rows = self.store.update([])
        self.assertEqual(rows[0]["status"], "getrennt")

    def test_close_idempotent_and_reopen(self):
        self.store.update([self._dev()])
        self.store.close()
        self.store.close()
        again = DeviceStore(Path(self.tmp.name) / "geraete.db")
        try:
            self.assertEqual(len(again.list()), 1)
        finally:
            again.close()


class ToolsTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.reg = ToolRegistry(workspace=self.tmp.name)
        self.store = DeviceStore(Path(self.tmp.name) / "g.db")
        devices.register_tools(self.reg, self.store)

    def tearDown(self):
        self.store.close()
        self.tmp.cleanup()

    def test_tools_registered_not_dangerous(self):
        names = {t.name for t in self.reg.list()}
        self.assertIn("geraete_scannen", names)
        self.assertIn("seriell_lesen", names)
        for t in self.reg.list():
            self.assertFalse(t.dangerous)

    def test_scan_tool_with_patched_scan(self):
        orig = devices.scan
        devices.scan = lambda: [devices._make_device("seriell", "STM32 VCP", port="COM7", vid="0483", pid="5740", raw={})]
        try:
            res = self.reg.run("geraete_scannen", {})
        finally:
            devices.scan = orig
        self.assertTrue(res.ok, res.error)
        self.assertIn("COM7", res.output)
        self.assertIn("flugsteuerung", res.output.lower())
        self.assertEqual(len(self.store.list()), 1)

    def test_scan_tool_without_hardware(self):
        orig = devices.scan
        devices.scan = lambda: []
        try:
            res = self.reg.run("geraete_scannen", {})
        finally:
            devices.scan = orig
        self.assertTrue(res.ok)
        self.assertIn("Keine", res.output)

    def test_read_tool_validates(self):
        res = self.reg.run("seriell_lesen", {"port": "COM1", "baud": 1234})
        self.assertFalse(res.ok)
        self.assertIn("Baudrate", res.error)

    def test_read_tool_parses_telemetry(self):
        class _Reader:
            def __init__(self):
                self.data = [_nmea("GPGGA,123519,4807.038,N,01131.000,E,1,08,0.9,545.4,M,46.9,M,,").encode() + b"\r\n"]

            def read(self, n, timeout=0.1):
                return self.data.pop(0) if self.data else b""

            def close(self):
                pass

        orig = devices._open_serial
        devices._open_serial = lambda port, baud: _Reader()
        try:
            res = self.reg.run("seriell_lesen", {"port": "/dev/ttyUSB0", "sekunden": 0.2})
        finally:
            devices._open_serial = orig
        self.assertTrue(res.ok, res.error)
        self.assertIn("48,117", res.output.replace(".", ","))


if __name__ == "__main__":
    unittest.main()
