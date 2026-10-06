"""Simulationen von OBITO: Zeitschritt-Modelle für Flug, Akku, Thermik, Fall und Regler.

Reine Standardbibliothek. Jede Simulation liefert ein ``dict`` mit festem Aufbau::

    {"typ": "...", "parameter": {...}, "zeit_s": [...], "reihen": {"name": [...]},
     "einheiten": {"name": "V"}, "zusammenfassung": {...}, "annahmen": [...], "warnungen": [...]}

Reihen werden auf höchstens :data:`MAX_POINTS` Stützstellen ausgedünnt (erster und letzter Punkt
bleiben). Die Parameterstudie ``beam_sweep`` nutzt statt ``zeit_s`` die Achse ``x`` mit
``x_einheit``. Unsinnige Eingaben lösen ``ValueError`` mit deutschem Text aus. Alle Modelle sind
bewusst einfache Ingenieur-Näherungen – die Annahmen stehen im Ergebnis.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Sequence

from .engineering import (CELL_EMPTY_V, G, _num, _pos, _round, beam_cantilever, find_material, fmt_number,
                          _p_int, _p_number, _p_str, _schema)
from .geometry import parse_params

if TYPE_CHECKING:  # pragma: no cover
    from .tools import ToolRegistry

AIR_DENSITY = 1.225           # kg/m³ (Meereshöhe, 15 °C)
MAX_POINTS = 600              # Stützstellen je Reihe im Ergebnis
MAX_STEPS = 200_000           # Schutz vor endlosen Läufen
MAX_LIST_ITEMS = 200

# Leerlaufspannung je LiPo-Zelle über dem Ladezustand (SOC 1,0 … 0,0), linear interpoliert
LIPO_OCV: tuple[tuple[float, float], ...] = (
    (1.0, 4.20), (0.9, 4.06), (0.8, 3.98), (0.7, 3.91), (0.6, 3.85), (0.5, 3.80),
    (0.4, 3.76), (0.3, 3.72), (0.2, 3.66), (0.1, 3.55), (0.0, 3.30),
)
CELL_INTERNAL_OHM_5AH = 0.004   # Innenwiderstand je Zelle bei 5000 mAh (skaliert mit 5000/mAh)
MAX_CELLS = 24
MAX_DURATION_S = 86_400.0


# ------------------------------------------------------------------ Hilfen
def _result(kind: str, params: dict, t: list[float], series: dict[str, list[float]], units: dict[str, str],
            summary: dict, assumptions: list[str], warnings: list[str], *, x_name: str | None = None,
            x_unit: str | None = None) -> dict:
    t2, series2 = _thin(t, series)
    out: dict[str, Any] = {"typ": kind, "parameter": params}
    if x_name is None:
        out["zeit_s"] = t2
    else:
        out["x"] = t2
        out["x_name"] = x_name
        out["x_einheit"] = x_unit or ""
    out.update({"reihen": series2, "einheiten": units, "zusammenfassung": summary,
                "annahmen": assumptions, "warnungen": warnings})
    return out


def _thin(t: Sequence[float], series: dict[str, Sequence[float]]) -> tuple[list[float], dict[str, list[float]]]:
    """Dünnt alle Reihen gleichmäßig auf ≤ MAX_POINTS aus; erster und letzter Punkt bleiben erhalten."""
    n = len(t)
    if n <= MAX_POINTS:
        return [_r(v) for v in t], {k: [_r(v) for v in vals] for k, vals in series.items()}
    idx = sorted({round(i * (n - 1) / (MAX_POINTS - 1)) for i in range(MAX_POINTS)})
    return [_r(t[i]) for i in idx], {k: [_r(vals[i]) for i in idx] for k, vals in series.items()}


def _r(v: float) -> float:
    if v is None or isinstance(v, bool):
        return v  # type: ignore[return-value]
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return v
    return float(f"{float(v):.6g}")


def _dt(name: str, value: Any, lo: float = 1e-4, hi: float = 3600.0) -> float:
    v = _pos(name, value)
    if v < lo or v > hi:
        raise ValueError(f"{name} muss zwischen {fmt_number(lo)} und {fmt_number(hi)} s liegen.")
    return v


def _steps(duration: float, dt: float) -> int:
    n = int(math.ceil(duration / dt))
    if n > MAX_STEPS:
        raise ValueError(f"Zu viele Zeitschritte ({n:,} > {MAX_STEPS:,}) – größeres dt oder kürzere Dauer wählen."
                         .replace(",", "."))
    return max(n, 1)


def cell_ocv(soc: float) -> float:
    """Leerlaufspannung einer LiPo-Zelle (V) beim Ladezustand ``soc`` (0…1), linear interpoliert."""
    soc = max(0.0, min(1.0, float(soc)))
    for (s_hi, v_hi), (s_lo, v_lo) in zip(LIPO_OCV, LIPO_OCV[1:]):
        if soc >= s_lo:
            f = (soc - s_lo) / (s_hi - s_lo)
            return v_lo + f * (v_hi - v_lo)
    return LIPO_OCV[-1][1]


def _cells(value: Any) -> int:
    n = int(_num("Zellenzahl", value))
    if n < 1 or n > MAX_CELLS:
        raise ValueError(f"Zellenzahl muss zwischen 1 und {MAX_CELLS} liegen.")
    return n


def _fraction(name: str, value: Any) -> float:
    v = _num(name, value)
    if not 0.05 <= v <= 1.0:
        raise ValueError(f"{name} muss zwischen 0,05 und 1 liegen.")
    return v


# ------------------------------------------------------------------ Akku & Flug
def _discharge(cells: int, mah: float, current_fn: Callable[[float, float], float], *, usable: float,
               peukert: float, dt: float, max_s: float) -> tuple[list[float], dict[str, list[float]], str]:
    """Gemeinsamer Entladekern: ``current_fn(t, soc)`` liefert den Strom in A.
    Rückgabe: Zeiten, Reihen (spannung_v, strom_a, soc_prozent, leistung_w, zellenspannung_v), Abbruchgrund."""
    cap_ah = mah / 1000.0
    r_cell = CELL_INTERNAL_OHM_5AH * (5000.0 / mah)
    soc = 1.0
    t = 0.0
    times: list[float] = []
    s_v: list[float] = []
    s_i: list[float] = []
    s_soc: list[float] = []
    s_p: list[float] = []
    s_cell: list[float] = []
    reason = "Zeitlimit erreicht"
    n = _steps(max_s, dt)
    for _ in range(n + 1):
        i = max(0.0, current_fn(t, soc))
        v_cell = cell_ocv(soc) - i * r_cell
        v = v_cell * cells
        times.append(t)
        s_v.append(v)
        s_i.append(i)
        s_soc.append(soc * 100.0)
        s_p.append(v * i)
        s_cell.append(v_cell)
        if v_cell < CELL_EMPTY_V:
            reason = f"Zellenspannung unter Last unter {fmt_number(CELL_EMPTY_V)} V"
            break
        if soc <= 1.0 - usable + 1e-9:
            reason = f"nutzbare Kapazität ({fmt_number(usable * 100)} %) verbraucht"
            break
        # Peukert: effektiver Verbrauch steigt mit dem Strom relativ zur 1-C-Rate
        c_rate = i / cap_ah if cap_ah > 0 else 0.0
        factor = c_rate ** (peukert - 1.0) if c_rate > 0 else 0.0
        soc -= i * factor * dt / 3600.0 / cap_ah if factor else 0.0
        soc = max(soc, 0.0)
        t += dt
    return times, {"spannung_v": s_v, "strom_a": s_i, "soc_prozent": s_soc, "leistung_w": s_p,
                   "zellenspannung_v": s_cell}, reason


def hover_flight(mass_g: float, cells: int, mah: float, hover_current_a: float, *, usable: float = 0.8,
                 peukert: float = 1.05, dt: float = 1.0, payload_g: float = 0.0) -> dict:
    """Schwebeflug: Flugzeit aus Akku-Entladung mit Innenwiderstand, Peukert-Effekt und Nutzlast.
    Der Schwebestrom skaliert mit (Gesamtmasse/Masse)^1,5 (Schub ∝ Leistung^(2/3))."""
    mass = _pos("Abfluggewicht (g)", mass_g)
    cells_n = _cells(cells)
    cap = _pos("Kapazität (mAh)", mah)
    i_hover = _pos("Schwebestrom (A)", hover_current_a)
    usable_f = _fraction("Nutzbarer Anteil", usable)
    peukert_f = _num("Peukert-Exponent", peukert)
    if not 1.0 <= peukert_f <= 1.3:
        raise ValueError("Peukert-Exponent muss zwischen 1,0 und 1,3 liegen.")
    payload = _num("Nutzlast (g)", payload_g)
    if payload < 0:
        raise ValueError("Nutzlast darf nicht negativ sein.")
    step = _dt("dt", dt, 0.1, 60.0)
    total = mass + payload
    i_total = i_hover * (total / mass) ** 1.5
    nominal_v = 3.7 * cells_n
    if i_total * nominal_v > 20_000:
        raise ValueError("Leistung über 20 kW – Eingaben prüfen.")

    times, series, reason = _discharge(cells_n, cap, lambda t, soc: i_total, usable=usable_f, peukert=peukert_f,
                                       dt=step, max_s=MAX_DURATION_S)
    flight_s = times[-1]
    energy_wh = sum(p * step for p in series["leistung_w"][:-1]) / 3600.0
    warnings: list[str] = []
    if flight_s < 120:
        warnings.append("Flugzeit unter 2 Minuten – Akku oder Strombedarf prüfen.")
    if series["zellenspannung_v"] and min(series["zellenspannung_v"]) < 3.6:
        warnings.append("Zellenspannung unter Last fällt unter 3,6 V – hohe Belastung (C-Rate prüfen).")
    summary = {
        "flugzeit_min": _round(flight_s / 60.0, 2),
        "flugzeit_s": _round(flight_s, 1),
        "strom_gesamt_a": _round(i_total, 2),
        "gesamtmasse_g": _round(total, 1),
        "energie_wh": _round(energy_wh, 2),
        "restkapazitaet_prozent": _round(series["soc_prozent"][-1], 1),
        "endspannung_v": _round(series["spannung_v"][-1], 2),
        "grund": reason,
    }
    assumptions = [
        "Konstanter Schwebestrom, Strom skaliert mit (Gesamtmasse/Masse)^1,5 für Nutzlast",
        f"LiPo-Kennlinie (Leerlaufspannung je Zelle 4,20…3,30 V), Innenwiderstand "
        f"{fmt_number(CELL_INTERNAL_OHM_5AH * 5000.0 / cap, 3)} Ω je Zelle",
        f"Peukert-Exponent {fmt_number(peukert_f)}, Abbruch bei Zellenspannung < {fmt_number(CELL_EMPTY_V)} V "
        f"oder {fmt_number(usable_f * 100)} % Entnahme",
        "Kein Wind, keine Temperaturabhängigkeit, Elektronik-Verbrauch im Schwebestrom enthalten",
    ]
    params = {"mass_g": mass, "cells": cells_n, "mah": cap, "hover_current_a": i_hover, "usable": usable_f,
              "peukert": peukert_f, "dt": step, "payload_g": payload}
    units = {"spannung_v": "V", "strom_a": "A", "soc_prozent": "%", "leistung_w": "W", "zellenspannung_v": "V"}
    return _result("schwebeflug", params, times, series, units, summary, assumptions, warnings)


def battery_discharge(cells: int, mah: float, current_a: float, *, dt: float = 5.0, usable: float = 1.0,
                      peukert: float = 1.05) -> dict:
    """Akku-Entladung mit konstantem Strom: Spannungs- und Ladezustandsverlauf, Zeit bis leer."""
    cells_n = _cells(cells)
    cap = _pos("Kapazität (mAh)", mah)
    i = _pos("Strom (A)", current_a)
    step = _dt("dt", dt, 0.1, 600.0)
    usable_f = _fraction("Nutzbarer Anteil", usable)
    peukert_f = _num("Peukert-Exponent", peukert)
    if not 1.0 <= peukert_f <= 1.3:
        raise ValueError("Peukert-Exponent muss zwischen 1,0 und 1,3 liegen.")
    times, series, reason = _discharge(cells_n, cap, lambda t, soc: i, usable=usable_f, peukert=peukert_f,
                                       dt=step, max_s=MAX_DURATION_S)
    c_rate = i / (cap / 1000.0)
    warnings = []
    if c_rate > 30:
        warnings.append(f"Entladerate {fmt_number(c_rate, 3)} C ist sehr hoch – Akku-C-Rating prüfen.")
    summary = {
        "zeit_bis_leer_min": _round(times[-1] / 60.0, 2),
        "c_rate": _round(c_rate, 2),
        "energie_wh": _round(sum(p * step for p in series["leistung_w"][:-1]) / 3600.0, 2),
        "endspannung_v": _round(series["spannung_v"][-1], 2),
        "restkapazitaet_prozent": _round(series["soc_prozent"][-1], 1),
        "grund": reason,
    }
    assumptions = ["Konstanter Strom; LiPo-Leerlaufkennlinie mit Innenwiderstand; Peukert-Effekt",
                   f"Abbruch bei Zellenspannung < {fmt_number(CELL_EMPTY_V)} V oder {fmt_number(usable_f * 100)} % Entnahme"]
    params = {"cells": cells_n, "mah": cap, "current_a": i, "dt": step, "usable": usable_f, "peukert": peukert_f}
    units = {"spannung_v": "V", "strom_a": "A", "soc_prozent": "%", "leistung_w": "W", "zellenspannung_v": "V"}
    return _result("akku", params, times, series, units, summary, assumptions, warnings)


def climb_profile(mass_g: float, thrust_max_n: float, target_alt_m: float, *, drag_cd: float = 1.0,
                  area_m2: float = 0.05, dt: float = 0.05, max_s: float = 120.0) -> dict:
    """Senkrechter Steigflug auf eine Zielhöhe mit begrenztem Schub, Luftwiderstand und PD-Höhenregler."""
    m = _pos("Masse (g)", mass_g) / 1000.0
    t_max = _pos("Maximalschub (N)", thrust_max_n)
    z_target = _pos("Zielhöhe (m)", target_alt_m)
    cd = _num("cw-Wert", drag_cd)
    area = _num("Stirnfläche (m²)", area_m2)
    if cd < 0 or area < 0:
        raise ValueError("cw-Wert und Fläche dürfen nicht negativ sein.")
    step = _dt("dt", dt, 1e-3, 1.0)
    duration = _pos("Maximaldauer (s)", max_s)
    weight = m * G
    if t_max <= weight:
        raise ValueError(f"Maximalschub {fmt_number(t_max)} N reicht nicht für das Gewicht {fmt_number(weight)} N.")
    # PD-Regler: Verstärkungen aus Masse und verfügbarem Überschuss (kritisch gedämpft, ω ≈ 1 rad/s)
    omega = 1.0
    kp = m * omega * omega
    kd = 2.0 * m * omega
    z = 0.0
    v = 0.0
    times: list[float] = []
    s_z: list[float] = []
    s_v: list[float] = []
    s_t: list[float] = []
    reach_t: float | None = None
    max_rate = 0.0
    n = _steps(duration, step)
    for k in range(n + 1):
        t = k * step
        thrust = weight + kp * (z_target - z) - kd * v
        thrust = max(0.0, min(t_max, thrust))
        drag = 0.5 * AIR_DENSITY * cd * area * v * abs(v)
        a = (thrust - weight - drag) / m
        times.append(t)
        s_z.append(z)
        s_v.append(v)
        s_t.append(thrust)
        if reach_t is None and z >= 0.98 * z_target:
            reach_t = t
        max_rate = max(max_rate, v)
        # semi-implizites Euler-Verfahren
        v += a * step
        z += v * step
        if z < 0:
            z, v = 0.0, max(v, 0.0)
    overshoot = max(0.0, max(s_z) - z_target)
    warnings = []
    if reach_t is None:
        warnings.append("Zielhöhe innerhalb der Maximaldauer nicht erreicht.")
    if overshoot > 0.1 * z_target:
        warnings.append("Überschwingen über 10 % der Zielhöhe.")
    summary = {
        "zeit_bis_hoehe_s": _round(reach_t, 2) if reach_t is not None else None,
        "max_steigrate_m_s": _round(max_rate, 2),
        "ueberschwingen_m": _round(overshoot, 3),
        "endhoehe_m": _round(s_z[-1], 2),
        "schub_gewicht_verhaeltnis": _round(t_max / weight, 2),
        "schwebeschub_n": _round(weight, 3),
    }
    assumptions = ["Nur vertikale Bewegung: m·a = T − m·g − ½·ρ·cw·A·v·|v| (ρ = 1,225 kg/m³)",
                   "PD-Höhenregler (kritisch gedämpft, ω = 1 rad/s), Schub zwischen 0 und Maximalschub",
                   "Keine Motordynamik, kein Wind, konstante Masse"]
    params = {"mass_g": mass_g, "thrust_max_n": t_max, "target_alt_m": z_target, "drag_cd": cd, "area_m2": area,
              "dt": step, "max_s": duration}
    units = {"hoehe_m": "m", "geschwindigkeit_m_s": "m/s", "schub_n": "N"}
    return _result("steigflug", params, times, {"hoehe_m": s_z, "geschwindigkeit_m_s": s_v, "schub_n": s_t},
                   units, summary, assumptions, warnings)


def drop_with_drag(mass_kg: float, height_m: float, *, cd: float = 0.47, area_m2: float = 0.01,
                   dt: float = 0.01) -> dict:
    """Freier Fall mit Luftwiderstand (RK4): Aufprallgeschwindigkeit, Fallzeit, Grenzgeschwindigkeit."""
    m = _pos("Masse (kg)", mass_kg)
    h0 = _pos("Höhe (m)", height_m)
    cd_v = _num("cw-Wert", cd)
    area = _num("Fläche (m²)", area_m2)
    if cd_v < 0 or area < 0:
        raise ValueError("cw-Wert und Fläche dürfen nicht negativ sein.")
    step = _dt("dt", dt, 1e-4, 1.0)
    k = 0.5 * AIR_DENSITY * cd_v * area

    def accel(v: float) -> float:
        return G - k * v * abs(v) / m

    h = h0
    v = 0.0
    t = 0.0
    times = [0.0]
    s_h = [h0]
    s_v = [0.0]
    t_vac = math.sqrt(2 * h0 / G)
    t_est = 10.0 * t_vac + 60.0
    if k > 0:
        t_est = max(t_est, 2.0 * h0 / math.sqrt(m * G / k) + 60.0)   # langsamer Fall nahe der Grenzgeschwindigkeit
    t_est = min(MAX_DURATION_S, t_est)
    step = max(step, t_est / MAX_STEPS)          # RK4 bleibt auch mit gröberem Schritt genau
    max_steps = _steps(t_est, step)
    for _ in range(max_steps):
        k1v = accel(v)
        k1h = -v
        k2v = accel(v + 0.5 * step * k1v)
        k2h = -(v + 0.5 * step * k1v)
        k3v = accel(v + 0.5 * step * k2v)
        k3h = -(v + 0.5 * step * k2v)
        k4v = accel(v + step * k3v)
        k4h = -(v + step * k3v)
        v_new = v + step / 6.0 * (k1v + 2 * k2v + 2 * k3v + k4v)
        h_new = h + step / 6.0 * (k1h + 2 * k2h + 2 * k3h + k4h)
        t += step
        if h_new <= 0.0:
            # lineare Interpolation auf den Aufprall
            frac = h / (h - h_new) if h != h_new else 1.0
            t = t - step + frac * step
            v = v + frac * (v_new - v)
            h = 0.0
            times.append(t)
            s_h.append(0.0)
            s_v.append(v)
            break
        h, v = h_new, v_new
        times.append(t)
        s_h.append(h)
        s_v.append(v)
    else:
        raise ValueError("Fall endet nicht innerhalb des Zeitlimits – Eingaben prüfen.")
    v_terminal = math.sqrt(m * G / k) if k > 0 else None
    v_vacuum = math.sqrt(2 * G * h0)
    t_vacuum = math.sqrt(2 * h0 / G)
    energy = 0.5 * m * v * v
    summary = {
        "aufprallgeschwindigkeit_m_s": _round(v, 3),
        "aufprallgeschwindigkeit_kmh": _round(v * 3.6, 2),
        "fallzeit_s": _round(t, 3),
        "grenzgeschwindigkeit_m_s": _round(v_terminal, 3) if v_terminal else None,
        "ohne_luft_geschwindigkeit_m_s": _round(v_vacuum, 3),
        "ohne_luft_fallzeit_s": _round(t_vacuum, 3),
        "aufprallenergie_j": _round(energy, 2),
    }
    warnings = []
    if v_terminal and v >= 0.99 * v_terminal:
        warnings.append("Grenzgeschwindigkeit erreicht – größere Höhe ändert den Aufprall kaum.")
    assumptions = ["Punktmasse, Luftwiderstand F = ½·ρ·cw·A·v² mit ρ = 1,225 kg/m³, g = 9,81 m/s²",
                   "Keine Rotation, kein Wind, Runge-Kutta 4. Ordnung"]
    params = {"mass_kg": m, "height_m": h0, "cd": cd_v, "area_m2": area, "dt": step}
    return _result("fall", params, times, {"hoehe_m": s_h, "geschwindigkeit_m_s": s_v},
                   {"hoehe_m": "m", "geschwindigkeit_m_s": "m/s"}, summary, assumptions, warnings)


def thermal_rc(power_w: float, mass_g: float, cp_j_per_gk: float, h_w_per_m2k: float, area_m2: float, *,
               ambient_c: float = 25.0, duration_s: float = 600.0, dt: float = 1.0) -> dict:
    """Ein-Knoten-Wärmemodell: m·cp·dT/dt = P − h·A·(T − T_umgebung) (z. B. Motor, ESC, Akku)."""
    p = _num("Leistung (W)", power_w)
    if p < 0:
        raise ValueError("Verlustleistung darf nicht negativ sein.")
    m = _pos("Masse (g)", mass_g)
    cp = _pos("spezifische Wärmekapazität (J/(g·K))", cp_j_per_gk)
    h = _pos("Wärmeübergang (W/(m²·K))", h_w_per_m2k)
    area = _pos("Oberfläche (m²)", area_m2)
    t_amb = _num("Umgebungstemperatur (°C)", ambient_c)
    duration = _pos("Dauer (s)", duration_s)
    if duration > MAX_DURATION_S:
        raise ValueError("Dauer höchstens 86 400 s.")
    step = _dt("dt", dt, 1e-3, 600.0)
    c_th = m * cp                 # J/K
    g_th = h * area               # W/K
    tau = c_th / g_th
    t_final = t_amb + p / g_th
    temp = t_amb
    times: list[float] = []
    s_t: list[float] = []
    t80 = tau * math.log(5.0) if p > 0 else None     # analytisch: 1 − e^(−t/τ) = 0,8
    n = _steps(duration, step)
    for k in range(n + 1):
        t = k * step
        times.append(t)
        s_t.append(temp)
        # exakte Lösung je Schritt (stabil für beliebiges dt)
        temp = t_final + (temp - t_final) * math.exp(-step / tau)
    warnings = []
    if t_final > 100:
        warnings.append(f"Endtemperatur {fmt_number(t_final, 3)} °C über 100 °C – Kühlung oder weniger Leistung nötig.")
    if t_final > 80:
        warnings.append("Über 80 °C: typische Grenze für LiPo-Zellen und viele Elektronikbauteile.")
    summary = {
        "endtemperatur_c": _round(t_final, 2),
        "temperatur_nach_dauer_c": _round(s_t[-1], 2),
        "zeitkonstante_s": _round(tau, 2),
        "zeit_bis_80_prozent_s": _round(t80, 1) if t80 is not None else None,
        "waermekapazitaet_j_k": _round(c_th, 2),
        "waermeleitwert_w_k": _round(g_th, 4),
    }
    assumptions = ["Ein Knoten mit gleichmäßiger Temperatur, konstante Leistung und Umgebung",
                   "Konvektion + Strahlung als linearer Wärmeübergang h·A zusammengefasst",
                   "τ = m·cp/(h·A); Endtemperatur = T_umgebung + P/(h·A)"]
    params = {"power_w": p, "mass_g": m, "cp_j_per_gk": cp, "h_w_per_m2k": h, "area_m2": area,
              "ambient_c": t_amb, "duration_s": duration, "dt": step}
    return _result("thermik", params, times, {"temperatur_c": s_t}, {"temperatur_c": "°C"}, summary,
                   assumptions, warnings)


def pid_step(kp: float, ki: float, kd: float, *, plant_tau_s: float = 0.3, plant_gain: float = 1.0,
             setpoint: float = 1.0, duration_s: float = 5.0, dt: float = 0.005) -> dict:
    """Sprungantwort eines PID-Reglers an einer PT1-Strecke (Zeitkonstante, Verstärkung) mit Anti-Windup."""
    kp_v = _num("Kp", kp)
    ki_v = _num("Ki", ki)
    kd_v = _num("Kd", kd)
    if kp_v < 0 or ki_v < 0 or kd_v < 0:
        raise ValueError("Kp, Ki und Kd dürfen nicht negativ sein.")
    tau = _pos("Streckenzeitkonstante (s)", plant_tau_s)
    gain = _pos("Streckenverstärkung", plant_gain)
    sp = _num("Sollwert", setpoint)
    if sp == 0:
        raise ValueError("Sollwert darf nicht 0 sein.")
    duration = _pos("Dauer (s)", duration_s)
    step = _dt("dt", dt, 1e-5, 1.0)
    if step > tau / 5:
        raise ValueError("dt muss höchstens ein Fünftel der Streckenzeitkonstante sein.")
    u_limit = 10.0 * abs(sp) / gain
    y = 0.0
    integral = 0.0
    prev_y = 0.0
    times: list[float] = []
    s_y: list[float] = []
    s_u: list[float] = []
    n = _steps(duration, step)
    for k in range(n + 1):
        t = k * step
        err = sp - y
        d_meas = (y - prev_y) / step if k > 0 else 0.0
        u_raw = kp_v * err + ki_v * integral - kd_v * d_meas
        u = max(-u_limit, min(u_limit, u_raw))
        if u == u_raw or (u_raw > u and err < 0) or (u_raw < u and err > 0):
            integral += err * step               # Anti-Windup: Integrator nur ohne Sättigung laden
        times.append(t)
        s_y.append(y)
        s_u.append(u)
        prev_y = y
        # PT1: tau·dy/dt = K·u − y (exakter Schritt)
        y = gain * u + (y - gain * u) * math.exp(-step / tau)
        if not math.isfinite(y) or abs(y) > 1e6 * abs(sp):
            y = 1e6 * abs(sp) * (1 if y > 0 else -1)
    peak = max(s_y) if sp > 0 else min(s_y)
    overshoot = max(0.0, (peak - sp) / sp * 100.0) if sp > 0 else max(0.0, (sp - peak) / abs(sp) * 100.0)
    band = 0.02 * abs(sp)
    settle: float | None = None
    for i in range(len(s_y)):
        if all(abs(v - sp) <= band for v in s_y[i:]):
            settle = times[i]
            break
    t10 = next((times[i] for i, v in enumerate(s_y) if abs(v) >= 0.1 * abs(sp)), None)
    t90 = next((times[i] for i, v in enumerate(s_y) if abs(v) >= 0.9 * abs(sp)), None)
    rise = (t90 - t10) if (t10 is not None and t90 is not None) else None
    steady_err = sp - s_y[-1]
    warnings = []
    # Instabilität: Amplitude der Abweichung wächst in der zweiten Hälfte
    half = len(s_y) // 2
    if half > 10:
        a1 = max(abs(v - sp) for v in s_y[half // 2:half])
        a2 = max(abs(v - sp) for v in s_y[half:])
        if a2 > 1.5 * a1 and a2 > 0.1 * abs(sp):
            warnings.append("Regelkreis instabil: Schwingung wächst – Kp/Kd verringern.")
    if overshoot > 25:
        warnings.append(f"Überschwingen {fmt_number(overshoot, 3)} % – Kd erhöhen oder Kp senken.")
    if settle is None and not warnings:
        warnings.append("2-%-Band innerhalb der Dauer nicht dauerhaft erreicht.")
    summary = {
        "ueberschwingen_prozent": _round(overshoot, 2),
        "einschwingzeit_s": _round(settle, 3) if settle is not None else None,
        "anstiegszeit_s": _round(rise, 3) if rise is not None else None,
        "stationaerer_fehler": _round(steady_err, 4),
        "stationaerer_fehler_prozent": _round(steady_err / abs(sp) * 100.0, 2),
        "endwert": _round(s_y[-1], 4),
        "stellgroesse_max": _round(max(abs(u) for u in s_u), 3),
    }
    assumptions = ["Strecke PT1: τ·dy/dt = K·u − y; Regler PID mit D-Anteil auf den Messwert",
                   f"Stellgrößenbegrenzung ±{fmt_number(u_limit, 3)} mit Anti-Windup",
                   "Sprung des Sollwerts bei t = 0 aus der Ruhelage"]
    params = {"kp": kp_v, "ki": ki_v, "kd": kd_v, "plant_tau_s": tau, "plant_gain": gain, "setpoint": sp,
              "duration_s": duration, "dt": step}
    return _result("regler", params, times, {"istwert": s_y, "stellgroesse": s_u},
                   {"istwert": "", "stellgroesse": ""}, summary, assumptions, warnings)


def beam_sweep(force_n: float, length_m: float, material: str | float, width_m: float,
               heights_mm: Sequence[float]) -> dict:
    """Parameterstudie Kragbalken: Durchbiegung und Biegespannung über der Querschnittshöhe."""
    if isinstance(heights_mm, str):
        heights_mm = _parse_list("Höhe (mm)", heights_mm)
    heights = [_pos("Höhe (mm)", h) for h in (heights_mm or [])]
    if not heights:
        raise ValueError("Mindestens eine Höhe in mm angeben.")
    if len(heights) > MAX_LIST_ITEMS:
        raise ValueError(f"Höchstens {MAX_LIST_ITEMS} Höhen.")
    heights = sorted(set(heights))
    mat = None
    e_gpa: float
    if isinstance(material, (int, float)) and not isinstance(material, bool):
        e_gpa = _pos("E-Modul (GPa)", material)
        mat_name = f"E = {fmt_number(e_gpa)} GPa"
    else:
        text = str(material).strip()
        mat = find_material(text)
        if mat is not None:
            e_gpa = mat.youngs_gpa
            mat_name = mat.name
        else:
            try:
                e_gpa = _pos("E-Modul (GPa)", text)
                mat_name = f"E = {fmt_number(e_gpa)} GPa"
            except ValueError:
                raise ValueError(f"Material »{text}« unbekannt und keine Zahl (E-Modul in GPa).") from None
    defl: list[float] = []
    stress: list[float] = []
    for h in heights:
        r = beam_cantilever(force_n, length_m, e_gpa, width_m, h / 1000.0)
        defl.append(r["durchbiegung_mm"])
        stress.append(r["biegespannung_mpa"])
    summary: dict[str, Any] = {"material": mat_name, "e_modul_gpa": _round(e_gpa, 2)}
    warnings: list[str] = []
    if mat is not None:
        limit = mat.tensile_mpa[0] / 2.0
        summary["zulaessige_spannung_mpa"] = _round(limit, 2)
        ok = [h for h, s in zip(heights, stress) if s <= limit]
        summary["min_hoehe_sicher_mm"] = ok[0] if ok else None
        if not ok:
            warnings.append("Keine der Höhen hält den Sicherheitsfaktor 2 gegen die Zugfestigkeit ein.")
    else:
        summary["min_hoehe_sicher_mm"] = None
    summary["durchbiegung_min_mm"] = _round(min(defl), 4)
    summary["durchbiegung_max_mm"] = _round(max(defl), 4)
    assumptions = beam_cantilever(force_n, length_m, e_gpa, width_m, heights[0] / 1000.0)["annahmen"]
    if mat is not None:
        assumptions.append("Zulässige Spannung = Zugfestigkeit (unterer Richtwert) / 2")
    params = {"force_n": float(force_n), "length_m": float(length_m), "material": str(material),
              "width_m": float(width_m), "heights_mm": heights}
    return _result("balken", params, heights, {"durchbiegung_mm": defl, "spannung_mpa": stress},
                   {"durchbiegung_mm": "mm", "spannung_mpa": "MPa"}, summary, assumptions, warnings,
                   x_name="hoehe_mm", x_unit="mm")


# ------------------------------------------------------------------ Registry
SIMULATIONS: dict[str, dict] = {
    "schwebeflug": {
        "fn": hover_flight,
        "beschreibung": "Flugzeit im Schwebeflug aus Akku, Schwebestrom und Nutzlast (LiPo-Entladung).",
        "parameter": [("mass_g", "Abfluggewicht in g", None), ("cells", "Zellenzahl (S)", None),
                      ("mah", "Kapazität in mAh", None), ("hover_current_a", "Schwebestrom in A", None),
                      ("usable", "nutzbarer Anteil 0,05–1", 0.8), ("peukert", "Peukert-Exponent 1,0–1,3", 1.05),
                      ("dt", "Zeitschritt in s", 1.0), ("payload_g", "Nutzlast in g", 0.0)],
    },
    "steigflug": {
        "fn": climb_profile,
        "beschreibung": "Senkrechter Steigflug auf Zielhöhe mit Maximalschub, Luftwiderstand und Höhenregler.",
        "parameter": [("mass_g", "Masse in g", None), ("thrust_max_n", "Maximalschub in N", None),
                      ("target_alt_m", "Zielhöhe in m", None), ("drag_cd", "cw-Wert", 1.0),
                      ("area_m2", "Stirnfläche in m²", 0.05), ("dt", "Zeitschritt in s", 0.05),
                      ("max_s", "Maximaldauer in s", 120.0)],
    },
    "fall": {
        "fn": drop_with_drag,
        "beschreibung": "Fall mit Luftwiderstand: Aufprallgeschwindigkeit, Fallzeit, Grenzgeschwindigkeit.",
        "parameter": [("mass_kg", "Masse in kg", None), ("height_m", "Fallhöhe in m", None),
                      ("cd", "cw-Wert", 0.47), ("area_m2", "Stirnfläche in m²", 0.01), ("dt", "Zeitschritt in s", 0.01)],
    },
    "thermik": {
        "fn": thermal_rc,
        "beschreibung": "Erwärmung eines Bauteils (Motor, ESC, Akku) bei Verlustleistung: Endtemperatur, Zeitkonstante.",
        "parameter": [("power_w", "Verlustleistung in W", None), ("mass_g", "Masse in g", None),
                      ("cp_j_per_gk", "spez. Wärmekapazität in J/(g·K), z. B. Alu 0,9, Kupfer 0,385, LiPo ≈ 1,0", None),
                      ("h_w_per_m2k", "Wärmeübergang in W/(m²·K): ruhend 5–10, Propellerwind 25–60", None),
                      ("area_m2", "Oberfläche in m²", None), ("ambient_c", "Umgebung in °C", 25.0),
                      ("duration_s", "Dauer in s", 600.0), ("dt", "Zeitschritt in s", 1.0)],
    },
    "regler": {
        "fn": pid_step,
        "beschreibung": "Sprungantwort eines PID-Reglers an einer PT1-Strecke: Überschwingen, Einschwingzeit, Fehler.",
        "parameter": [("kp", "Proportionalverstärkung", None), ("ki", "Integralverstärkung", None),
                      ("kd", "Differenzialverstärkung", None), ("plant_tau_s", "Streckenzeitkonstante in s", 0.3),
                      ("plant_gain", "Streckenverstärkung", 1.0), ("setpoint", "Sollwert", 1.0),
                      ("duration_s", "Dauer in s", 5.0), ("dt", "Zeitschritt in s", 0.005)],
    },
    "akku": {
        "fn": battery_discharge,
        "beschreibung": "Akku-Entladung bei konstantem Strom: Spannung, Ladezustand, Zeit bis leer.",
        "parameter": [("cells", "Zellenzahl (S)", None), ("mah", "Kapazität in mAh", None),
                      ("current_a", "Strom in A", None), ("dt", "Zeitschritt in s", 5.0),
                      ("usable", "nutzbarer Anteil 0,05–1", 1.0), ("peukert", "Peukert-Exponent", 1.05)],
    },
    "balken": {
        "fn": beam_sweep,
        "beschreibung": "Parameterstudie Kragbalken: Durchbiegung und Spannung über der Querschnittshöhe.",
        "parameter": [("force_n", "Kraft am freien Ende in N", None), ("length_m", "Länge in m", None),
                      ("material", "Materialname oder E-Modul in GPa", None), ("width_m", "Breite in m", None),
                      ("heights_mm", "Höhen in mm, getrennt mit Semikolon oder Leerzeichen", None)],
    },
}

_ALIASES = {"hover": "schwebeflug", "flug": "schwebeflug", "flugzeit": "schwebeflug", "climb": "steigflug",
            "steigen": "steigflug", "drop": "fall", "sturz": "fall", "thermal": "thermik", "waerme": "thermik",
            "wärme": "thermik", "temperatur": "thermik", "pid": "regler", "regelung": "regler", "battery": "akku",
            "batterie": "akku", "entladung": "akku", "beam": "balken", "biegung": "balken"}


def resolve_kind(kind: Any) -> str:
    key = str(kind or "").strip().lower().replace("-", "").replace("_", "")
    if key in SIMULATIONS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    raise ValueError(f"Unbekannte Simulation »{kind}«. Verfügbar: {', '.join(SIMULATIONS)}.")


def _parse_list(name: str, value: Any) -> list[float]:
    if isinstance(value, (list, tuple)):
        return [_num(name, v) for v in value]
    text = str(value).strip()
    if not text:
        raise ValueError(f"{name}: leere Liste.")
    parts = [p for p in text.replace(";", " ").split() if p]
    if len(parts) == 1 and "," in text and text.count(",") > 1:
        parts = [p for p in text.split(",") if p.strip()]
    return [_num(name, p) for p in parts]


def run(kind: str, params: dict | str | None) -> dict:
    """Führt die Simulation ``kind`` mit Parametern aus einem Dict oder ``"k=v, k=v"``-Text aus."""
    key = resolve_kind(kind)
    spec = SIMULATIONS[key]
    raw = parse_params(params)
    names = [p[0] for p in spec["parameter"]]
    kwargs: dict[str, Any] = {}
    for k, v in raw.items():
        name = str(k).strip().lower()
        if name not in names:
            raise ValueError(f"Unbekannter Parameter »{k}« für {key}. Erlaubt: {', '.join(names)}.")
        kwargs[name] = v
    missing = [n for n, _d, default in spec["parameter"] if default is None and n not in kwargs]
    if missing:
        raise ValueError(f"Fehlende Parameter für {key}: {', '.join(missing)}.")
    if key == "balken":
        kwargs["heights_mm"] = _parse_list("heights_mm", kwargs["heights_mm"])
    else:
        for n in list(kwargs):
            if n == "material":
                continue
            kwargs[n] = _num(n, kwargs[n])
        if "cells" in kwargs:
            kwargs["cells"] = int(kwargs["cells"])
    fn = spec["fn"]
    positional = [n for n, _d, default in spec["parameter"] if default is None]
    args = [kwargs.pop(n) for n in positional]
    return fn(*args, **kwargs)


# ------------------------------------------------------------------ Text
_LABELS = {
    "flugzeit_min": "Flugzeit", "flugzeit_s": "Flugzeit (s)", "strom_gesamt_a": "Gesamtstrom", "gesamtmasse_g": "Gesamtmasse",
    "energie_wh": "Energie", "restkapazitaet_prozent": "Restkapazität", "endspannung_v": "Endspannung", "grund": "Ende",
    "zeit_bis_hoehe_s": "Zeit bis Zielhöhe", "max_steigrate_m_s": "max. Steigrate", "ueberschwingen_m": "Überschwingen",
    "endhoehe_m": "Endhöhe", "schub_gewicht_verhaeltnis": "Schub/Gewicht", "schwebeschub_n": "Schwebeschub",
    "aufprallgeschwindigkeit_m_s": "Aufprallgeschwindigkeit", "aufprallgeschwindigkeit_kmh": "Aufprall (km/h)",
    "fallzeit_s": "Fallzeit", "grenzgeschwindigkeit_m_s": "Grenzgeschwindigkeit", "ohne_luft_geschwindigkeit_m_s": "ohne Luft",
    "ohne_luft_fallzeit_s": "Fallzeit ohne Luft", "aufprallenergie_j": "Aufprallenergie", "endtemperatur_c": "Endtemperatur",
    "temperatur_nach_dauer_c": "Temperatur am Ende", "zeitkonstante_s": "Zeitkonstante", "zeit_bis_80_prozent_s": "Zeit bis 80 %",
    "waermekapazitaet_j_k": "Wärmekapazität", "waermeleitwert_w_k": "Wärmeleitwert", "ueberschwingen_prozent": "Überschwingen",
    "einschwingzeit_s": "Einschwingzeit (2 %)", "anstiegszeit_s": "Anstiegszeit", "stationaerer_fehler": "stationärer Fehler",
    "stationaerer_fehler_prozent": "stationärer Fehler (%)", "endwert": "Endwert", "stellgroesse_max": "max. Stellgröße",
    "zeit_bis_leer_min": "Zeit bis leer", "c_rate": "C-Rate", "material": "Material", "e_modul_gpa": "E-Modul",
    "zulaessige_spannung_mpa": "zulässige Spannung", "min_hoehe_sicher_mm": "kleinste sichere Höhe",
    "durchbiegung_min_mm": "Durchbiegung min", "durchbiegung_max_mm": "Durchbiegung max",
}
_UNIT_OF_KEY = {"_min": "min", "_s": "s", "_a": "A", "_g": "g", "_wh": "Wh", "_prozent": "%", "_v": "V", "_m": "m",
                "_m_s": "m/s", "_kmh": "km/h", "_n": "N", "_j": "J", "_c": "°C", "_j_k": "J/K", "_w_k": "W/K",
                "_mm": "mm", "_gpa": "GPa", "_mpa": "MPa"}


def _unit_for(key: str) -> str:
    for suffix in sorted(_UNIT_OF_KEY, key=len, reverse=True):
        if key.endswith(suffix):
            return _UNIT_OF_KEY[suffix]
    return ""


def _fmt(v: Any) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "ja" if v else "nein"
    if isinstance(v, (int, float)):
        return fmt_number(float(v) if isinstance(v, float) else v, 4)
    return str(v)


def summary_text(result: dict) -> str:
    """Deutsche Kurzfassung eines Simulationsergebnisses (für Werkzeug, CLI, Mission)."""
    kind = result.get("typ", "?")
    spec = SIMULATIONS.get(kind, {})
    lines = [f"Simulation »{kind}« – {spec.get('beschreibung', '')}".rstrip(" –")]
    params = result.get("parameter") or {}
    if params:
        lines.append("Parameter: " + ", ".join(f"{k}={_fmt(v)}" for k, v in params.items()))
    axis = result.get("zeit_s")
    if axis is not None:
        n = len(axis)
        lines.append(f"Verlauf: {n} Punkte über {_fmt(axis[-1] if axis else 0)} s, Reihen: "
                     + ", ".join(result.get("reihen", {}).keys()))
    else:
        lines.append(f"Verlauf über {result.get('x_name', 'x')} ({result.get('x_einheit', '')}): "
                     f"{len(result.get('x', []))} Punkte")
    lines.append("Ergebnis:")
    for k, v in (result.get("zusammenfassung") or {}).items():
        label = _LABELS.get(k, k)
        unit = _unit_for(k) if isinstance(v, (int, float)) and not isinstance(v, bool) else ""
        lines.append(f"  {label}: {_fmt(v)}{(' ' + unit) if unit else ''}")
    for w in result.get("warnungen") or []:
        lines.append(f"Warnung: {w}")
    ann = result.get("annahmen") or []
    if ann:
        lines.append("Annahmen: " + "; ".join(ann))
    return "\n".join(lines)


def describe() -> str:
    """Übersicht aller Simulationsarten mit Parametern (für Hilfe/CLI)."""
    lines = []
    for key, spec in SIMULATIONS.items():
        lines.append(f"{key}: {spec['beschreibung']}")
        for name, desc, default in spec["parameter"]:
            d = "" if default is None else f" (Standard {_fmt(default)})"
            lines.append(f"    {name} – {desc}{d}")
    return "\n".join(lines)


# ------------------------------------------------------------------ Werkzeuge
TOOL_NAMES = ("simulation_starten",)


def _tool_simulation(art: str, parameter: str = "") -> str:
    result = run(art, parameter)
    return summary_text(result)


def register_tools(registry: "ToolRegistry") -> None:
    """Registriert ``simulation_starten`` (nicht gefährlich)."""
    from .tools import Tool

    arten = ", ".join(SIMULATIONS)
    registry.register(Tool(
        name="simulation_starten",
        description="Führt eine Zeitschritt-Simulation aus und liefert Kennzahlen, Warnungen und Annahmen. "
                    f"Arten: {arten}. Parameter als »name=wert, name=wert« (Dezimalkomma erlaubt; Listen mit "
                    "Semikolon). Pflichtparameter je Art: schwebeflug mass_g,cells,mah,hover_current_a; steigflug "
                    "mass_g,thrust_max_n,target_alt_m; fall mass_kg,height_m; thermik power_w,mass_g,cp_j_per_gk,"
                    "h_w_per_m2k,area_m2; regler kp,ki,kd; akku cells,mah,current_a; balken force_n,length_m,"
                    "material,width_m,heights_mm.",
        parameters=_schema({"art": _p_str("Simulationsart", "schwebeflug"),
                            "parameter": _p_str("Parameter »name=wert, …«", "mass_g=1200, cells=4, mah=1500, hover_current_a=15")},
                           ["art"]),
        fn=_tool_simulation,
    ))


__all__ = [
    "AIR_DENSITY", "LIPO_OCV", "MAX_POINTS", "MAX_STEPS", "SIMULATIONS", "TOOL_NAMES", "battery_discharge",
    "beam_sweep", "cell_ocv", "climb_profile", "describe", "drop_with_drag", "hover_flight", "pid_step",
    "register_tools", "resolve_kind", "run", "summary_text", "thermal_rc",
]
