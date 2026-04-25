"""
FOL - Finding Oil Losses
Pipeline Leak Detection System - CLI Edition
PT Pertamina EP Jambi Field

Developed by Research & Development
Version 2.0 CLI - No Streamlit

Behavior identik dengan fol_leak_detect_V2.py (Streamlit fixed):
  1. Load .sav  -> analyzer (base_config DARI .sav, bukan dari kode ini)
  2. Patch HANYA PSI_PER_METER = 0.1209
  3. Load fresh .xlsx -> override analyzer.elevation_df
     - Gagal + .sav punya elev  -> pakai elev bawaan .sav (warning)
     - Gagal + .sav tidak punya -> pure fallback, has_elevation_data=False
  4. predict() -> semua config murni dari .sav
"""

import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
from datetime import datetime
import sys
import traceback

try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.prompt import Prompt
    from rich import box
    from rich.rule import Rule
    RICH = True
    console = Console()
except ImportError:
    RICH = False
    console = None

def ok(msg):
    if RICH: console.print(f"[bold green]OK {msg}[/bold green]")
    else: print(f"[OK]   {msg}")
def warn(msg):
    if RICH: console.print(f"[bold yellow]WARN {msg}[/bold yellow]")
    else: print(f"[WARN] {msg}")
def err(msg):
    if RICH: console.print(f"[bold red]ERR {msg}[/bold red]")
    else: print(f"[ERR]  {msg}")
def info(msg):
    if RICH: console.print(f"[cyan]{msg}[/cyan]")
    else: print(f"[i]   {msg}")
def sep():
    if RICH: console.print(Rule(style="bright_blue"))
    else: print("-" * 60)

PIPELINE_CONFIGS = {
    "BJG-TPN": {
        "model_file": "bjg_model.sav", "elevation_file": "bjg_elevasi.xlsx",
        "total_length": 26.6,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 7.14 SPOT 2 FOL", "location": 7.14},
            {"name": "III. KP 15.4 SPOT 3 FOL", "location": 15.4},
            {"name": "IV. KP 19.7 SPOT 4 FOL", "location": 19.7},
        ],
        "example_normal": [136.0, 112.14, 95.4, 37.1],
        "example_drop": [133.0, 110.1, 82.5, 34.5],
        "description": "Pipeline dari Betara Jambi (BJG) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil",
    },
    "BTJ-TPN": {
        "model_file": "btj_model.sav", "elevation_file": "btj_elevasi.xlsx",
        "total_length": 13.8,
        "sensors": [
            {"name": "I. KP 0 SP BTJ", "location": 0.0},
            {"name": "II. KP SIMP PS Gajah 5.5", "location": 5.5},
            {"name": "III. KP Sandtrap II 9.7", "location": 9.7},
        ],
        "example_normal": [190.2, 80.0, 34.5],
        "example_drop": [185.5, 75.0, 32.1],
        "description": "Pipeline dari Betara Jambi (BTJ) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil",
    },
    "KAS-TPN": {
        "model_file": "kas_model.sav", "elevation_file": "kas_elevasi.xlsx",
        "total_length": 23.2,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 7.8 SPOT 2 FOL", "location": 7.8},
            {"name": "III. KP 15.4 SPOT 3 FOL", "location": 15.4},
            {"name": "IV. KP 23.2 SPOT 4 FOL", "location": 23.2},
        ],
        "example_normal": [150.727, 125.92, 84.0367, 43.2134],
        "example_drop": [143.778, 104.54, 64.3846, 40.0],
        "description": "Pipeline dari Kenali Asam (KAS) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil",
    },
    "KTT-KAS": {
        "model_file": "ktt_model.sav", "elevation_file": "ktt_elevasi.xlsx",
        "total_length": 44.6,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 44.6 SPOT 2 FOL", "location": 44.6},
        ],
        "example_normal": [190.2, 22.0],
        "example_drop": [165.4, 20.5],
        "description": "Pipeline dari Kenali Asam Timur (KTT) menuju Kenali Asam (KAS)",
        "fluid_type": "Crude Oil",
    },
    "SG-KAS": {
        "model_file": "sg_model.sav", "elevation_file": "sg_elevasi.xlsx",
        "total_length": 11.2,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 11.2 SPOT 2 FOL", "location": 11.2},
        ],
        "example_normal": [276.6, 20.0],
        "example_drop": [268.01, 18.0],
        "description": "Pipeline dari Sei Gelam (SG) menuju Kenali Asam (KAS)",
        "fluid_type": "Crude Oil",
    },
}

class EnhancedLeakAnalyzer:
    def __init__(self, base_config, elevation_df=None):
        self.base_config = base_config
        self.elevation_df = elevation_df
        self.has_elevation_data = elevation_df is not None

    def predict(self, sensor_locations, normal_pressure, drop_pressure, sensor_names=None):
        locations = np.array(sensor_locations)
        normal_p = np.array(normal_pressure)
        drop_p = np.array(drop_pressure)
        n_sensors = len(locations)
        if len(normal_p) != n_sensors:
            raise ValueError(f"normal_pressure harus {n_sensors} elemen (sesuai jumlah sensor)")
        if len(drop_p) != n_sensors:
            raise ValueError(f"drop_pressure harus {n_sensors} elemen (sesuai jumlah sensor)")
        if sensor_names is None:
            sensor_names = [f'Sensor {i+1} (KP {loc:.1f})' for i, loc in enumerate(locations)]
        if self.has_elevation_data:
            elev_interp = interpolate.interp1d(
                self.elevation_df['distance_km'], self.elevation_df['elevation'],
                kind='cubic', fill_value='extrapolate')
            elevations = elev_interp(locations)
        else:
            elevations = np.zeros(n_sensors)
        delta_p = normal_p - drop_p
        abs_delta_p = np.abs(delta_p)
        with np.errstate(divide='ignore', invalid='ignore'):
            pressure_ratio = abs_delta_p / np.abs(normal_p) * 100
        pressure_ratio = np.nan_to_num(pressure_ratio, 0.0)
        suspicion_index = self._calculate_suspicion_index(abs_delta_p, pressure_ratio, n_sensors)
        susp_loc = self._suspicion_method(locations, suspicion_index)
        grad_loc = self._gradient_method(locations, normal_p, drop_p)
        interp_loc = self._interpolation_method(locations, abs_delta_p)
        weighted_loc = self._weighted_method(locations, suspicion_index)
        elev_loc = self._elevation_method(locations, normal_p, drop_p, elevations, n_sensors)
        cfg = self.base_config
        final_estimate = (
            susp_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['suspicion'] +
            interp_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['interpolation'] +
            grad_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['gradient'] +
            elev_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['elevation'] +
            weighted_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['weighted']
        )
        estimate_std = np.std([susp_loc, interp_loc, grad_loc, elev_loc, weighted_loc])
        if estimate_std < 2: confidence = "VERY HIGH (95%+)"
        elif estimate_std < 4: confidence = "HIGH (90-95%)"
        elif estimate_std < 6: confidence = "HIGH (85-90%)"
        else: confidence = "MEDIUM (75-85%)"
        return {
            'final_estimate': final_estimate, 'estimate_std': estimate_std,
            'confidence': confidence,
            'zones': {
                'focus': (final_estimate - 3, final_estimate + 3),
                'critical': (final_estimate - 5, final_estimate + 5),
                'primary': (final_estimate - 10, final_estimate + 10),
            },
            'top_sensor_idx': np.argmax(suspicion_index),
            'methods': {'suspicion': susp_loc, 'interpolation': interp_loc,
                        'gradient': grad_loc, 'elevation': elev_loc, 'weighted': weighted_loc},
            'sensor_data': {
                'locations': locations, 'names': sensor_names, 'elevations': elevations,
                'normal_pressure': normal_p, 'drop_pressure': drop_p,
                'delta_pressure': delta_p, 'abs_delta_pressure': abs_delta_p,
                'pressure_ratio': pressure_ratio, 'suspicion_index': suspicion_index,
            },
        }

    def _calculate_suspicion_index(self, abs_delta_p, pressure_ratio, n_sensors):
        cfg = self.base_config
        suspicion_index = np.zeros(n_sensors)
        for i in range(n_sensors):
            delta_factor = abs_delta_p[i]
            ratio_factor = pressure_ratio[i]
            if i > 0 and i < n_sensors - 1:
                neighbor_avg = (abs_delta_p[i-1] + abs_delta_p[i+1]) / 2
                neighbor_diff = abs_delta_p[i] - neighbor_avg
            elif i == 0 and n_sensors > 1:
                neighbor_diff = abs_delta_p[i] - abs_delta_p[i+1]
            elif i == n_sensors - 1 and n_sensors > 1:
                neighbor_diff = abs_delta_p[i] - abs_delta_p[i-1]
            else:
                neighbor_diff = 0
            neighbor_factor = max(0, neighbor_diff)
            suspicion_index[i] = (
                delta_factor * cfg['SUSPICION_WEIGHTS'][0] +
                ratio_factor * cfg['SUSPICION_WEIGHTS'][1] +
                neighbor_factor * cfg['SUSPICION_WEIGHTS'][2]
            )
        return suspicion_index

    def _suspicion_method(self, locations, suspicion_index):
        cfg = self.base_config
        top_idx = np.argmax(suspicion_index)
        location = locations[top_idx] + cfg['UPSTREAM_BIAS_PRIMARY']
        return max(location, locations[0])

    def _gradient_method(self, locations, normal_p, drop_p):
        cfg = self.base_config
        if len(locations) < 2: return locations[0]
        changes, locs = [], []
        for i in range(len(locations) - 1):
            dist = locations[i+1] - locations[i]
            if dist > 0:
                norm_grad = (normal_p[i+1] - normal_p[i]) / dist
                drop_grad = (drop_p[i+1] - drop_p[i]) / dist
                changes.append(np.abs(norm_grad - drop_grad))
                locs.append((locations[i] + locations[i+1]) / 2)
        if not changes: return locations[0]
        return locs[np.argmax(changes)] + cfg['UPSTREAM_BIAS_GRADIENT']

    def _interpolation_method(self, locations, abs_delta_p):
        cfg = self.base_config
        if len(locations) < 4:
            return locations[np.argmax(abs_delta_p)] + cfg['UPSTREAM_BIAS_INTERP']
        try:
            f = interpolate.interp1d(locations, abs_delta_p, kind='cubic', fill_value='extrapolate')
            x_fine = np.linspace(locations.min(), locations.max(), 2000)
            y_fine = f(x_fine)
            return x_fine[np.argmax(y_fine)] + cfg['UPSTREAM_BIAS_INTERP']
        except Exception:
            return locations[np.argmax(abs_delta_p)] + cfg['UPSTREAM_BIAS_INTERP']

    def _weighted_method(self, locations, suspicion_index):
        cfg = self.base_config
        total = np.sum(suspicion_index)
        if total == 0: return np.mean(locations)
        return np.sum(suspicion_index * locations) / total + cfg['UPSTREAM_BIAS_WEIGHTED']

    def _elevation_method(self, locations, normal_p, drop_p, elevations, n_sensors):
        cfg = self.base_config
        if not self.has_elevation_data:
            return locations[np.argmax(np.abs(normal_p - drop_p))] + cfg['UPSTREAM_BIAS_PRIMARY']
        psi_per_meter = cfg['PSI_PER_METER']
        ref_elev = elevations[0]
        elev_corr = (elevations - ref_elev) * psi_per_meter
        normal_corr = normal_p - elev_corr
        drop_corr = drop_p - elev_corr
        anomaly_scores = np.zeros(n_sensors)
        for i in range(1, n_sensors):
            dist = locations[i] - locations[i-1]
            if dist > 0:
                exp_grad = (normal_corr[i-1] - normal_corr[i]) / dist
                act_grad = (drop_corr[i-1] - drop_corr[i]) / dist
                anom = abs(act_grad - exp_grad)
                anomaly_scores[i-1] += anom * 0.5
                anomaly_scores[i] += anom * 0.5
        return locations[np.argmax(anomaly_scores)] + cfg['UPSTREAM_BIAS_PRIMARY']


class GPSLocationMapper:
    def __init__(self, elevation_df):
        self.elevation_df = elevation_df
        self.lat_interpolator = interpolate.interp1d(
            elevation_df['distance_km'], elevation_df['latitude'],
            kind='cubic', fill_value='extrapolate')
        self.lon_interpolator = interpolate.interp1d(
            elevation_df['distance_km'], elevation_df['longitude'],
            kind='cubic', fill_value='extrapolate')
    def get_coordinates(self, kp_km):
        return float(self.lat_interpolator(kp_km)), float(self.lon_interpolator(kp_km))
    def get_google_maps_link(self, kp_km, zoom=18):
        lat, lon = self.get_coordinates(kp_km)
        return f"https://maps.google.com/?q={lat},{lon}&ll={lat},{lon}&z={zoom}"


def load_elevation_data(file_path):
    try:
        df = pd.read_excel(file_path)
        df.columns = ['latitude', 'longitude', 'elevation']
        distances = [0.0]
        for i in range(1, len(df)):
            lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
            lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            R = 6371
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            distances.append(distances[-1] + R * c)
        df['distance_km'] = distances
        return df, None
    except FileNotFoundError:
        return None, f"File '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading elevation data: {str(e)}"


def load_model_local(file_path):
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"


def filter_active_sensors(config, normal_pressure, drop_pressure):
    active_sensors, active_locations, active_names, active_normal, active_drop = [], [], [], [], []
    for i, (n, d) in enumerate(zip(normal_pressure, drop_pressure)):
        if n is not None and d is not None:
            active_sensors.append(i)
            active_locations.append(config["sensors"][i]["location"])
            active_names.append(config["sensors"][i]["name"])
            active_normal.append(n)
            active_drop.append(d)
    return {"indices": active_sensors, "locations": np.array(active_locations),
            "names": active_names, "normal": np.array(active_normal), "drop": np.array(active_drop)}


def load_and_prepare_analyzer(config):
    model_obj, error = load_model_local(config["model_file"])
    if model_obj is None:
        err(f"Failed to load model: {error}")
        sys.exit(1)
    analyzer = model_obj
    ok(f"Model loaded: {config['model_file']}")
    cfg = analyzer.base_config
    fw = cfg.get('FINAL_ESTIMATE_WEIGHTS', {})
    info(f"base_config dari .sav:")
    info(f"  UPSTREAM_BIAS    = [{cfg.get('UPSTREAM_BIAS_PRIMARY')}, {cfg.get('UPSTREAM_BIAS_GRADIENT')}, {cfg.get('UPSTREAM_BIAS_INTERP')}, {cfg.get('UPSTREAM_BIAS_WEIGHTED')}]")
    info(f"  SUSPICION_WEIGHTS = {cfg.get('SUSPICION_WEIGHTS')}")
    info(f"  FINAL_WEIGHTS    = susp:{fw.get('suspicion')} interp:{fw.get('interpolation')} grad:{fw.get('gradient')} elev:{fw.get('elevation')} w:{fw.get('weighted')}")
    info(f"  PSI_PER_METER (before patch) = {cfg.get('PSI_PER_METER')}")
    analyzer.base_config['PSI_PER_METER'] = 0.1209
    info("PSI_PER_METER di-patch -> 0.1209 psi/m")
    info(f"Loading elevation: {config['elevation_file']}")
    elev_df, elev_error = load_elevation_data(config["elevation_file"])
    gps_mapper = None
    elev_source = "NONE"
    if elev_df is None:
        warn(f"Elevation data tidak tersedia: {elev_error}")
        if analyzer.has_elevation_data and analyzer.elevation_df is not None:
            info("Prediksi dilanjutkan menggunakan elevation data bawaan model .sav")
            gps_mapper = GPSLocationMapper(analyzer.elevation_df)
            elev_source = f"sav_builtin ({len(analyzer.elevation_df)} titik)"
        else:
            info("Prediksi dilanjutkan tanpa elevation data (pure fallback)")
            gps_mapper = None
            elev_source = "NONE - pure fallback"
    else:
        analyzer.elevation_df = elev_df
        analyzer.has_elevation_data = True
        gps_mapper = GPSLocationMapper(elev_df)
        elev_source = f"xlsx_fresh ({len(elev_df)} titik)"
        ok(f"Elevation data dan GPS mapping loaded: {config['elevation_file']} ({len(elev_df)} titik)")
    return analyzer, gps_mapper, elev_df, elev_source


def input_pressure_data(config, use_example=False):
    normal_pressure, drop_pressure = [], []
    sep()
    if RICH:
        console.print("[bold cyan]SENSOR DATA INPUT[/bold cyan]")
        if not use_example:
            console.print("[dim]Ketik 0 atau Enter untuk sensor tidak aktif[/dim]")
        console.print()
    else:
        print("SENSOR DATA INPUT")
        if not use_example: print("Ketik 0 atau Enter untuk sensor tidak aktif\n")
    for i, sensor in enumerate(config["sensors"]):
        if RICH:
            console.print(f"  [bold yellow]Sensor {i+1}:[/bold yellow] {sensor['name']}  [dim](KP {sensor['location']} km)[/dim]")
        else:
            print(f"\nSensor {i+1}: {sensor['name']} (KP {sensor['location']} km)")
        if use_example:
            n_val = config["example_normal"][i]
            d_val = config["example_drop"][i]
            if RICH:
                console.print(f"    Normal : [green]{n_val:.2f}[/green] psi  Drop : [red]{d_val:.2f}[/red] psi  [dim](example)[/dim]")
            else:
                print(f"    Normal: {n_val:.2f} psi  |  Drop: {d_val:.2f} psi  (example)")
        else:
            n_val = _float_input("    Normal Pressure (psi) [Enter=skip]: ")
            d_val = _float_input("    Drop Pressure   (psi) [Enter=skip]: ")
        normal_pressure.append(n_val)
        drop_pressure.append(d_val)
        print()
    return normal_pressure, drop_pressure


def _float_input(prompt):
    while True:
        try:
            raw = input(prompt).strip()
            if raw == "": return None
            val = float(raw)
            if val < 0:
                print("      -> Tidak boleh negatif.")
                continue
            return None if val == 0 else val
        except ValueError:
            print("      -> Input tidak valid.")


def display_results(results, analyzer, config, pipeline_name, gps_mapper, elev_source):
    sep()
    final_kp = results['final_estimate']
    std = results['estimate_std']
    conf = results['confidence']
    focus = results['zones']['focus']
    critical = results['zones']['critical']
    primary = results['zones']['primary']
    if gps_mapper is not None:
        tl = config['total_length']
        focus = (max(0, focus[0]), min(tl, focus[1]))
        critical = (max(0, critical[0]), min(tl, critical[1]))
        primary = (max(0, primary[0]), min(tl, primary[1]))
    if RICH:
        txt = (f"[bold red]ESTIMATED LEAK LOCATION[/bold red]\n\n"
               f"[bold white]KP {final_kp:.2f} km[/bold white]  [dim](+/- {std:.2f} km)[/dim]\n"
               f"Confidence  : [bold green]{conf}[/bold green]\n"
               f"Elev source : [dim]{elev_source}[/dim]")
        console.print(Panel(txt, title="[bold red]LEAK DETECTION RESULTS[/bold red]",
                            border_style="red", padding=(1, 4)))
    else:
        print("\n" + "="*60)
        print("  LEAK DETECTION RESULTS")
        print("="*60)
        print(f"  Estimated Location : KP {final_kp:.2f} km  (+/-{std:.2f} km)")
        print(f"  Confidence         : {conf}")
        print(f"  Elevation source   : {elev_source}")
        print("="*60)
    if gps_mapper is not None:
        try:
            leak_lat, leak_lon = gps_mapper.get_coordinates(final_kp)
            maps_link = gps_mapper.get_google_maps_link(final_kp, zoom=18)
            if RICH:
                console.print(f"\n[bold cyan]GPS Coordinates[/bold cyan]")
                console.print(f"   Lat  : [green]{leak_lat:.6f}[/green]")
                console.print(f"   Lon  : [green]{leak_lon:.6f}[/green]")
                console.print(f"   Maps : [link={maps_link}]{maps_link}[/link]")
            else:
                print(f"\nGPS  : {leak_lat:.6f}, {leak_lon:.6f}")
                print(f"Maps : {maps_link}")
        except Exception as e:
            warn(f"GPS error: {e}")
    sep()
    if RICH:
        console.print("[bold cyan]PRIORITY INSPECTION ZONES[/bold cyan]\n")
        zt = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        zt.add_column("Zone", style="bold", width=20)
        zt.add_column("From (KP)", justify="right", width=12)
        zt.add_column("To (KP)", justify="right", width=12)
        zt.add_column("Priority", width=18)
        zt.add_row("[red]FOCUS[/red]", f"{focus[0]:.1f}", f"{focus[1]:.1f}", "[bold red]HIGHEST[/bold red]")
        zt.add_row("[yellow]CRITICAL[/yellow]", f"{critical[0]:.1f}", f"{critical[1]:.1f}", "Secondary")
        zt.add_row("[green]PRIMARY[/green]", f"{primary[0]:.1f}", f"{primary[1]:.1f}", "Tertiary")
        console.print(zt)
    else:
        print("\nPRIORITY INSPECTION ZONES")
        print(f"  FOCUS    : KP {focus[0]:.1f} - {focus[1]:.1f}   <- HIGHEST PRIORITY")
        print(f"  CRITICAL : KP {critical[0]:.1f} - {critical[1]:.1f}")
        print(f"  PRIMARY  : KP {primary[0]:.1f} - {primary[1]:.1f}")
    if gps_mapper is not None:
        try:
            for zone_name, zone in [("FOCUS", focus), ("CRITICAL", critical), ("PRIMARY", primary)]:
                lat_s, lon_s = gps_mapper.get_coordinates(zone[0])
                lat_e, lon_e = gps_mapper.get_coordinates(zone[1])
                link_s = gps_mapper.get_google_maps_link(zone[0])
                link_e = gps_mapper.get_google_maps_link(zone[1])
                if RICH:
                    console.print(f"\n  [dim]{zone_name}: Start[/dim] {lat_s:.6f},{lon_s:.6f}  [link={link_s}]Maps[/link]")
                    console.print(f"  [dim]{zone_name}: End  [/dim] {lat_e:.6f},{lon_e:.6f}  [link={link_e}]Maps[/link]")
                else:
                    print(f"\n  {zone_name}:")
                    print(f"    Start : {lat_s:.6f}, {lon_s:.6f}  -> {link_s}")
                    print(f"    End   : {lat_e:.6f}, {lon_e:.6f}  -> {link_e}")
        except Exception as e:
            warn(f"Zone GPS error: {e}")
    sep()
    fw = analyzer.base_config.get('FINAL_ESTIMATE_WEIGHTS', {})
    m = results['methods']
    if RICH:
        console.print("[bold cyan]DETECTION METHOD BREAKDOWN[/bold cyan]\n")
        mt = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        mt.add_column("Method", width=26)
        mt.add_column("Location (KP)", justify="right", width=16)
        mt.add_column("Weight", justify="right", width=8)
        mt.add_row("Suspicion Index",     f"{m['suspicion']:.2f}",     f"{fw.get('suspicion',0)*100:.0f}%")
        mt.add_row("Interpolation",       f"{m['interpolation']:.2f}", f"{fw.get('interpolation',0)*100:.0f}%")
        mt.add_row("Gradient Analysis",   f"{m['gradient']:.2f}",      f"{fw.get('gradient',0)*100:.0f}%")
        mt.add_row("Elevation/Hydraulic", f"{m['elevation']:.2f}",     f"{fw.get('elevation',0)*100:.0f}%")
        mt.add_row("Weighted Average",    f"{m['weighted']:.2f}",      f"{fw.get('weighted',0)*100:.0f}%")
        console.print(mt)
    else:
        print("\nDETECTION METHOD BREAKDOWN")
        print(f"  {'Method':<25} {'KP':>8}  {'Weight':>6}")
        print("  " + "-"*44)
        print(f"  {'Suspicion Index':<25} {m['suspicion']:>8.2f}  {fw.get('suspicion',0)*100:>5.0f}%")
        print(f"  {'Interpolation':<25} {m['interpolation']:>8.2f}  {fw.get('interpolation',0)*100:>5.0f}%")
        print(f"  {'Gradient Analysis':<25} {m['gradient']:>8.2f}  {fw.get('gradient',0)*100:>5.0f}%")
        print(f"  {'Elevation/Hydraulic':<25} {m['elevation']:>8.2f}  {fw.get('elevation',0)*100:>5.0f}%")
        print(f"  {'Weighted Average':<25} {m['weighted']:>8.2f}  {fw.get('weighted',0)*100:>5.0f}%")
    sep()
    sd = results['sensor_data']
    has_elev = analyzer.has_elevation_data
    if RICH:
        console.print("[bold cyan]ACTIVE SENSOR DETAILS[/bold cyan]\n")
        tbl = Table(show_header=True, header_style="bold magenta", box=box.ROUNDED)
        tbl.add_column("Sensor", width=32)
        tbl.add_column("KP (km)", justify="right", width=8)
        tbl.add_column("Elev (m)", justify="right", width=10)
        tbl.add_column("Normal (psi)", justify="right", width=14)
        tbl.add_column("Drop (psi)", justify="right", width=12)
        tbl.add_column("dP (psi)", justify="right", width=10)
        tbl.add_column("Suspicion", justify="right", width=10)
        for i in range(len(sd['locations'])):
            marker = " (!)" if i == results['top_sensor_idx'] else ""
            elev_str = f"{sd['elevations'][i]:.1f}" if has_elev else "N/A"
            tbl.add_row(
                f"{sd['names'][i]}{marker}", f"{sd['locations'][i]:.1f}", elev_str,
                f"{sd['normal_pressure'][i]:.2f}", f"{sd['drop_pressure'][i]:.2f}",
                f"{sd['delta_pressure'][i]:.2f}", f"{sd['suspicion_index'][i]:.2f}")
        console.print(tbl)
    else:
        print("\nACTIVE SENSOR DETAILS")
        print(f"  {'Sensor':<34} {'KP':>5} {'Normal':>10} {'Drop':>10} {'dP':>8} {'Susp':>8}")
        print("  " + "-"*80)
        for i in range(len(sd['locations'])):
            mk = " *" if i == results['top_sensor_idx'] else ""
            print(f"  {sd['names'][i]+mk:<36} {sd['locations'][i]:>4.1f} "
                  f"{sd['normal_pressure'][i]:>10.2f} {sd['drop_pressure'][i]:>10.2f} "
                  f"{sd['delta_pressure'][i]:>8.2f} {sd['suspicion_index'][i]:>8.2f}")
        print("  (* = Most Suspicious Sensor)")
    top_idx = results['top_sensor_idx']
    warn(f"Most Suspicious: {sd['names'][top_idx]} (KP {sd['locations'][top_idx]:.1f}) - Score: {sd['suspicion_index'][top_idx]:.2f}")
    return focus, critical, primary


def export_results(results, pipeline_name, focus, critical, primary, gps_mapper, final_kp, active_data):
    _leak_lat, _leak_lon, _maps_link = None, None, None
    if gps_mapper is not None:
        try:
            _leak_lat, _leak_lon = gps_mapper.get_coordinates(final_kp)
            _maps_link = gps_mapper.get_google_maps_link(final_kp, zoom=18)
        except Exception:
            pass
    export_data = {
        "Pipeline": pipeline_name,
        "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Active Sensors": len(active_data['indices']),
        "Estimated Location (KP)": f"{final_kp:.2f}",
        "Uncertainty (km)": f"{results['estimate_std']:.2f}",
        "Confidence": results['confidence'],
        "Focus Zone": f"KP {focus[0]:.1f} - {focus[1]:.1f}",
        "Critical Zone": f"KP {critical[0]:.1f} - {critical[1]:.1f}",
        "Primary Zone": f"KP {primary[0]:.1f} - {primary[1]:.1f}",
    }
    if _leak_lat is not None:
        export_data.update({
            "GPS Latitude": f"{_leak_lat:.8f}",
            "GPS Longitude": f"{_leak_lon:.8f}",
            "Google Maps Link": _maps_link,
        })
    filename = f"FOL_LeakDetection_{pipeline_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame([export_data]).to_csv(filename, index=False)
    ok(f"Hasil disimpan: {filename}")
    return filename


def select_pipeline():
    pipelines = list(PIPELINE_CONFIGS.keys())
    if RICH:
        t = Table(show_header=True, header_style="bold cyan", box=box.ROUNDED)
        t.add_column("No.", width=4, justify="right")
        t.add_column("Pipeline", width=12)
        t.add_column("Length", width=10, justify="right")
        t.add_column("Sensors", width=8, justify="right")
        t.add_column("Description", width=52)
        for i, (name, cfg) in enumerate(PIPELINE_CONFIGS.items(), 1):
            t.add_row(str(i), name, f"{cfg['total_length']} km", str(len(cfg['sensors'])), cfg['description'])
        console.print(t)
    else:
        print("\nAvailable Pipelines:")
        for i, (name, cfg) in enumerate(PIPELINE_CONFIGS.items(), 1):
            print(f"  {i}. {name:<10} | {cfg['total_length']} km | {len(cfg['sensors'])} sensors | {cfg['description']}")
    while True:
        try:
            idx = int(input(f"\nPilih pipeline [1-{len(pipelines)}]: ").strip()) - 1
            if 0 <= idx < len(pipelines):
                return pipelines[idx]
            print(f"  Pilihan harus 1-{len(pipelines)}")
        except ValueError:
            print("  Input tidak valid.")


def main():
    if RICH:
        console.print(Panel(
            "[bold cyan]FOL - FINDING OIL LOSSES[/bold cyan]\n"
            "[white]Pipeline Leak Detection System v2.0 CLI[/white]\n"
            "[dim]PT Pertamina EP Jambi Field | Research & Development[/dim]\n"
            "[dim]Behavior identik dengan Streamlit fol_leak_detect_V2.py[/dim]",
            border_style="bright_blue", padding=(1, 6)))
    else:
        print("\n" + "="*60)
        print("  FOL - FINDING OIL LOSSES v2.0 CLI")
        print("  PT Pertamina EP Jambi Field")
        print("  Behavior identik dengan Streamlit")
        print("="*60)

    while True:
        sep()
        if RICH: console.print("\n[bold cyan]PILIH PIPELINE[/bold cyan]")
        else: print("\nPILIH PIPELINE")

        selected_pipeline = select_pipeline()
        config = PIPELINE_CONFIGS[selected_pipeline]
        if RICH:
            console.print(f"\n[bold green]Pipeline:[/bold green] {selected_pipeline}  [dim]{config['description']}[/dim]")
            console.print(f"[dim]{config['total_length']} km | {len(config['sensors'])} sensors | {config['fluid_type']}[/dim]")
        else:
            print(f"\nPipeline : {selected_pipeline} | {config['description']}")

        sep()
        analyzer, gps_mapper, elev_df, elev_source = load_and_prepare_analyzer(config)

        sep()
        if RICH:
            console.print("  [1] Input manual\n  [2] Load example data")
            choice = Prompt.ask("Pilih mode", choices=["1", "2"], default="1")
        else:
            print("\n  [1] Input manual\n  [2] Load example data")
            choice = input("Pilih [1/2]: ").strip() or "1"

        normal_pressure, drop_pressure = input_pressure_data(config, use_example=(choice == "2"))

        errors = []
        valid_pairs = sum(1 for n, d in zip(normal_pressure, drop_pressure) if n is not None and d is not None)
        if valid_pairs == 0:
            errors.append("Minimal 1 sensor harus diisi (Normal & Drop pressure)")
        for i, (n, d) in enumerate(zip(normal_pressure, drop_pressure)):
            if n is not None and n < 0: errors.append(f"Sensor {i+1}: Normal pressure tidak boleh negatif")
            if d is not None and d < 0: errors.append(f"Sensor {i+1}: Drop pressure tidak boleh negatif")
        if errors:
            for e in errors: err(e)
            continue

        active_data = filter_active_sensors(config, normal_pressure, drop_pressure)
        if len(active_data["indices"]) == 0:
            err("Tidak ada sensor yang terisi dengan lengkap!")
            continue

        ok(f"{len(active_data['indices'])} sensor aktif dari {len(config['sensors'])}: "
           f"{', '.join([f'Sensor {i+1}' for i in active_data['indices']])}")

        sep()
        info(f"Analyzing with {len(active_data['indices'])} active sensor(s): {', '.join(active_data['names'])}")

        try:
            results = analyzer.predict(
                sensor_locations=active_data["locations"],
                normal_pressure=active_data["normal"],
                drop_pressure=active_data["drop"],
                sensor_names=active_data["names"]
            )
        except Exception as e:
            err(f"Prediction Error: {str(e)}")
            traceback.print_exc()
            continue

        focus, critical, primary = display_results(
            results, analyzer, config, selected_pipeline, gps_mapper, elev_source)

        sep()
        if input("Simpan hasil ke CSV? [y/N]: ").strip().lower() == 'y':
            export_results(results, selected_pipeline, focus, critical, primary,
                           gps_mapper, results['final_estimate'], active_data)

        sep()
        if input("Analisis pipeline lain? [y/N]: ").strip().lower() != 'y':
            if RICH: console.print("\n[bold cyan]FOL System selesai. Terima kasih![/bold cyan]\n")
            else: print("\nFOL System selesai. Terima kasih!\n")
            break


if __name__ == "__main__":
    main()
