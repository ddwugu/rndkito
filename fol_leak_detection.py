"""
FOL - Finding Oil Losses
Pipeline Leak Detection System - Multi-Pipeline Edition with Google Maps Integration
PT Pertamina EP Jambi Field

Developed by Research & Development
Version 2.0 - Enhanced with GPS Location
"""

import streamlit as st
import pickle
import numpy as np
import pandas as pd
from scipy import interpolate
from datetime import datetime
import os
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import folium_static
from folium import plugins

class EnhancedLeakAnalyzer:
    """
    Enhanced Leak Detection v4.0
    Auto-adjust untuk jumlah sensor berapapun
    """
    
    def __init__(self, base_config, elevation_df=None):
        self.base_config = base_config
        self.elevation_df = elevation_df
        self.has_elevation_data = elevation_df is not None
    
    def predict(self, sensor_locations, normal_pressure, drop_pressure, sensor_names=None):
        """
        MAIN PREDICTION FUNCTION
        
        Input:
            sensor_locations: array lokasi sensor (KP in km) - bisa 3, 4, 9+ sensor
            normal_pressure: array tekanan normal (psi)
            drop_pressure: array tekanan drop (psi)
            sensor_names: list nama sensor (optional)
        
        Output:
            dict dengan hasil analisis lengkap
        """
        # Convert to arrays
        locations = np.array(sensor_locations)
        normal_p = np.array(normal_pressure)
        drop_p = np.array(drop_pressure)
        n_sensors = len(locations)
        
        # Validate
        if len(normal_p) != n_sensors:
            raise ValueError(f"normal_pressure harus {n_sensors} elemen (sesuai jumlah sensor)")
        if len(drop_p) != n_sensors:
            raise ValueError(f"drop_pressure harus {n_sensors} elemen (sesuai jumlah sensor)")
        
        # Generate sensor names if not provided
        if sensor_names is None:
            sensor_names = [f'Sensor {i+1} (KP {loc:.1f})' for i, loc in enumerate(locations)]
        
        # Get elevations for these sensor locations
        if self.has_elevation_data:
            elev_interp = interpolate.interp1d(
                self.elevation_df['distance_km'],
                self.elevation_df['elevation'],
                kind='cubic',
                fill_value='extrapolate'
            )
            elevations = elev_interp(locations)
        else:
            elevations = np.zeros(n_sensors)
        
        # Calculate metrics
        delta_p = normal_p - drop_p
        abs_delta_p = np.abs(delta_p)
        
        with np.errstate(divide='ignore', invalid='ignore'):
            pressure_ratio = abs_delta_p / np.abs(normal_p) * 100
        pressure_ratio = np.nan_to_num(pressure_ratio, 0.0)
        
        # Suspicion index
        suspicion_index = self._calculate_suspicion_index(abs_delta_p, pressure_ratio, n_sensors)
        
        # All detection methods
        susp_loc = self._suspicion_method(locations, suspicion_index)
        grad_loc = self._gradient_method(locations, normal_p, drop_p)
        interp_loc = self._interpolation_method(locations, abs_delta_p)
        weighted_loc = self._weighted_method(locations, suspicion_index)
        elev_loc = self._elevation_method(locations, normal_p, drop_p, elevations, n_sensors)
        
        # Weighted final estimate
        cfg = self.base_config
        final_estimate = (
            susp_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['suspicion'] +
            interp_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['interpolation'] +
            grad_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['gradient'] +
            elev_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['elevation'] +
            weighted_loc * cfg['FINAL_ESTIMATE_WEIGHTS']['weighted']
        )
        
        estimate_std = np.std([susp_loc, interp_loc, grad_loc, elev_loc, weighted_loc])
        
        # Confidence
        if estimate_std < 2:
            confidence = "VERY HIGH (95%+)"
        elif estimate_std < 4:
            confidence = "HIGH (90-95%)"
        elif estimate_std < 6:
            confidence = "HIGH (85-90%)"
        else:
            confidence = "MEDIUM (75-85%)"
        
        # Return results
        return {
            'final_estimate': final_estimate,
            'estimate_std': estimate_std,
            'confidence': confidence,
            'zones': {
                'focus': (final_estimate - 3, final_estimate + 3),
                'critical': (final_estimate - 5, final_estimate + 5),
                'primary': (final_estimate - 10, final_estimate + 10)
            },
            'top_sensor_idx': np.argmax(suspicion_index),
            'methods': {
                'suspicion': susp_loc,
                'interpolation': interp_loc,
                'gradient': grad_loc,
                'elevation': elev_loc,
                'weighted': weighted_loc
            },
            'sensor_data': {
                'locations': locations,
                'names': sensor_names,
                'elevations': elevations,
                'normal_pressure': normal_p,
                'drop_pressure': drop_p,
                'delta_pressure': delta_p,
                'abs_delta_pressure': abs_delta_p,
                'pressure_ratio': pressure_ratio,
                'suspicion_index': suspicion_index
            }
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
        if len(locations) < 2:
            return locations[0]
        
        changes = []
        locs = []
        for i in range(len(locations) - 1):
            dist = locations[i+1] - locations[i]
            if dist > 0:
                norm_grad = (normal_p[i+1] - normal_p[i]) / dist
                drop_grad = (drop_p[i+1] - drop_p[i]) / dist
                changes.append(np.abs(norm_grad - drop_grad))
                locs.append((locations[i] + locations[i+1]) / 2)
        
        if not changes:
            return locations[0]
        
        max_idx = np.argmax(changes)
        return locs[max_idx] + cfg['UPSTREAM_BIAS_GRADIENT']
    
    def _interpolation_method(self, locations, abs_delta_p):
        cfg = self.base_config
        if len(locations) < 4:
            max_idx = np.argmax(abs_delta_p)
            return locations[max_idx] + cfg['UPSTREAM_BIAS_INTERP']
        
        try:
            f = interpolate.interp1d(locations, abs_delta_p, kind='cubic', fill_value='extrapolate')
            x_fine = np.linspace(locations.min(), locations.max(), 2000)
            y_fine = f(x_fine)
            peak_idx = np.argmax(y_fine)
            return x_fine[peak_idx] + cfg['UPSTREAM_BIAS_INTERP']
        except:
            max_idx = np.argmax(abs_delta_p)
            return locations[max_idx] + cfg['UPSTREAM_BIAS_INTERP']
    
    def _weighted_method(self, locations, suspicion_index):
        cfg = self.base_config
        total = np.sum(suspicion_index)
        if total == 0:
            return np.mean(locations)
        weighted_loc = np.sum(suspicion_index * locations) / total
        return weighted_loc + cfg['UPSTREAM_BIAS_WEIGHTED']
    
    def _elevation_method(self, locations, normal_p, drop_p, elevations, n_sensors):
        cfg = self.base_config
        
        if not self.has_elevation_data:
            return locations[np.argmax(np.abs(normal_p - drop_p))] + cfg['UPSTREAM_BIAS_PRIMARY']
        
        # Elevation-corrected pressure
        psi_per_meter = cfg['PSI_PER_METER'] * cfg['FLUID_DENSITY']
        ref_elev = elevations[0]
        elev_corr = (elevations - ref_elev) * psi_per_meter
        
        normal_corr = normal_p - elev_corr
        drop_corr = drop_p - elev_corr
        
        # Hydraulic anomaly
        anomaly_scores = np.zeros(n_sensors)
        for i in range(1, n_sensors):
            dist = locations[i] - locations[i-1]
            if dist > 0:
                exp_grad = (normal_corr[i-1] - normal_corr[i]) / dist
                act_grad = (drop_corr[i-1] - drop_corr[i]) / dist
                anom = abs(act_grad - exp_grad)
                anomaly_scores[i-1] += anom * 0.5
                anomaly_scores[i] += anom * 0.5
        
        max_idx = np.argmax(anomaly_scores)
        return locations[max_idx] + cfg['UPSTREAM_BIAS_PRIMARY']

class GPSLocationMapper:
    """
    GPS Location Mapper untuk convert KP ke Lat/Long
    """
    
    def __init__(self, elevation_df):
        """
        Initialize dengan elevation data yang sudah punya distance_km
        """
        self.elevation_df = elevation_df
        
        # Create interpolator untuk latitude dan longitude
        self.lat_interpolator = interpolate.interp1d(
            elevation_df['distance_km'],
            elevation_df['latitude'],
            kind='cubic',
            fill_value='extrapolate'
        )
        
        self.lon_interpolator = interpolate.interp1d(
            elevation_df['distance_km'],
            elevation_df['longitude'],
            kind='cubic',
            fill_value='extrapolate'
        )
    
    def get_coordinates(self, kp_km):
        """
        Get lat/long untuk KP tertentu
        
        Args:
            kp_km: Kilometer Point (float)
        
        Returns:
            tuple: (latitude, longitude)
        """
        lat = float(self.lat_interpolator(kp_km))
        lon = float(self.lon_interpolator(kp_km))
        return lat, lon
    
    def get_google_maps_link(self, kp_km, zoom=18):
        """
        Generate Google Maps link untuk KP tertentu
        
        Args:
            kp_km: Kilometer Point (float)
            zoom: Zoom level (default 18 untuk detail view)
        
        Returns:
            str: Google Maps URL
        """
        lat, lon = self.get_coordinates(kp_km)
        # Format: https://maps.google.com/?q=lat,lon&ll=lat,lon&z=zoom
        return f"https://maps.google.com/?q={lat},{lon}&ll={lat},{lon}&z={zoom}"
    
    def get_coordinates_for_zone(self, start_kp, end_kp, num_points=10):
        """
        Get multiple coordinates untuk inspection zone
        
        Args:
            start_kp: Start KP
            end_kp: End KP
            num_points: Number of points to generate
        
        Returns:
            list of tuples: [(lat1, lon1), (lat2, lon2), ...]
        """
        kp_points = np.linspace(start_kp, end_kp, num_points)
        coords = []
        for kp in kp_points:
            lat, lon = self.get_coordinates(kp)
            coords.append((lat, lon))
        return coords

def create_pressure_profile_chart(sensor_locations, normal_pressure, drop_pressure, 
                                   estimated_kp, pipeline_length, elevation_df=None,
                                   sensor_names=None, pipeline_name="Pipeline"):
    """
    Create professional Pressure Profile Comparison Chart with Elevation
    
    Args:
        sensor_locations: Array of sensor KP locations (km)
        normal_pressure: Array of normal pressures (psi)
        drop_pressure: Array of drop pressures (psi)
        estimated_kp: Estimated leak location (km)
        pipeline_length: Total pipeline length (km)
        elevation_df: DataFrame with elevation data (optional)
        sensor_names: List of sensor names (optional)
    
    Returns:
        Plotly figure object
    """
    
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Generate smooth line for entire pipeline
    kp_fine = np.linspace(0, pipeline_length, 500)
    
    # Interpolate pressure values for smooth curves
    if len(sensor_locations) >= 2:
        # Use linear interpolation for pressure (more realistic for pipeline)
        normal_interp = interpolate.interp1d(
            sensor_locations, normal_pressure, 
            kind='linear', fill_value='extrapolate'
        )
        drop_interp = interpolate.interp1d(
            sensor_locations, drop_pressure,
            kind='linear', fill_value='extrapolate'
        )
        
        normal_smooth = normal_interp(kp_fine)
        drop_smooth = drop_interp(kp_fine)
    else:
        # Single sensor - flat line
        normal_smooth = np.full_like(kp_fine, normal_pressure[0])
        drop_smooth = np.full_like(kp_fine, drop_pressure[0])
    
    # === PRIMARY Y-AXIS: PRESSURE ===
    
    # Normal Pressure Line
    fig.add_trace(
        go.Scatter(
            x=kp_fine,
            y=normal_smooth,
            mode='lines',
            name='Normal Pressure',
            line=dict(color='#00d4ff', width=3),
            hovertemplate='<b>Normal Pressure</b><br>' +
                         'KP: %{x:.2f} km<br>' +
                         'Pressure: %{y:.2f} psi<br>' +
                         '<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Drop Pressure Line
    fig.add_trace(
        go.Scatter(
            x=kp_fine,
            y=drop_smooth,
            mode='lines',
            name='Drop Pressure',
            line=dict(color='#ff3366', width=3),
            hovertemplate='<b>Drop Pressure</b><br>' +
                         'KP: %{x:.2f} km<br>' +
                         'Pressure: %{y:.2f} psi<br>' +
                         '<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Tolerance Line (97.5% of Normal) - Blue Line
    tolerance_pressure = normal_pressure * 0.975  # 2.5% tolerance
    
    if len(sensor_locations) >= 2:
        tolerance_interp = interpolate.interp1d(
            sensor_locations, tolerance_pressure,
            kind='linear', fill_value='extrapolate'
        )
        tolerance_smooth = tolerance_interp(kp_fine)
    else:
        tolerance_smooth = np.full_like(kp_fine, tolerance_pressure[0])
    
    fig.add_trace(
        go.Scatter(
            x=kp_fine,
            y=tolerance_smooth,
            mode='lines',
            name='Tolerance Line (97.5%)',
            line=dict(color="#00ff51", width=2, dash='dot'),
            hovertemplate='<b>Tolerance (97.5%)</b><br>' +
                         'KP: %{x:.2f} km<br>' +
                         'Pressure: %{y:.2f} psi<br>' +
                         '<extra></extra>'
        ),
        secondary_y=False
    )
    
    # Sensor Points - Normal
    fig.add_trace(
        go.Scatter(
            x=sensor_locations,
            y=normal_pressure,
            mode='markers',
            name='Sensors (Normal)',
            marker=dict(
                color='#00ffcc',
                size=12,
                symbol='circle',
                line=dict(color='white', width=2)
            ),
            text=sensor_names if sensor_names else [f'S{i+1}' for i in range(len(sensor_locations))],
            hovertemplate='<b>%{text}</b><br>' +
                         'KP: %{x:.2f} km<br>' +
                         'Normal: %{y:.2f} psi<br>' +
                         '<extra></extra>',
            showlegend=False
        ),
        secondary_y=False
    )
    
    # Sensor Points - Drop
    fig.add_trace(
        go.Scatter(
            x=sensor_locations,
            y=drop_pressure,
            mode='markers',
            name='Sensors (Drop)',
            marker=dict(
                color='#ff3366',
                size=12,
                symbol='square',
                line=dict(color='white', width=2)
            ),
            text=sensor_names if sensor_names else [f'S{i+1}' for i in range(len(sensor_locations))],
            hovertemplate='<b>%{text}</b><br>' +
                         'KP: %{x:.2f} km<br>' +
                         'Drop: %{y:.2f} psi<br>' +
                         '<extra></extra>',
            showlegend=False
        ),
        secondary_y=False
    )
    
    # === ELEVATION DATA (SECONDARY Y-AXIS) ===
    
    if elevation_df is not None:
        # Use elevation data directly
        elev_kp = elevation_df['distance_km'].values
        elev_values = elevation_df['elevation'].values
        
        # Add elevation area plot
        fig.add_trace(
            go.Scatter(
                x=elev_kp,
                y=elev_values,
                mode='lines',
                name='Elevation Profile',
                line=dict(color='rgba(149, 165, 166, 0.5)', width=1),
                fill='tozeroy',
                fillcolor='rgba(149, 165, 166, 0.2)',
                hovertemplate='<b>Elevation</b><br>' +
                             'KP: %{x:.2f} km<br>' +
                             'Elevation: %{y:.1f} m<br>' +
                             '<extra></extra>'
            ),
            secondary_y=True
        )
    
    # === ESTIMATED LEAK LOCATION ===
    
    # Vertical line at estimated leak location
    fig.add_vline(
        x=estimated_kp,
        line_dash="dash",
        line_color="rgba(231, 76, 60, 0.8)",
        line_width=3,
        annotation_text=f"Est. Leak<br>KP {estimated_kp:.2f}",
        annotation_position="top",
        annotation=dict(
            font=dict(size=11, color="white", family="Arial Black"),
            bgcolor="rgba(231, 76, 60, 0.9)",
            bordercolor="white",
            borderwidth=2,
            borderpad=4,
            y=0.5,  # 0.5 = middle (0=bottom, 1=top)
            yanchor="middle"
        )
    )
    
    # Shaded area around leak estimate (±1 km uncertainty)
    fig.add_vrect(
        x0=max(0, estimated_kp - 1),
        x1=min(pipeline_length, estimated_kp + 1),
        fillcolor="rgba(231, 76, 60, 0.15)",
        layer="below",
        line_width=0,
        annotation_text="",  # Remove text to avoid overlap
        annotation_position="top left"
    )
    
    # === LAYOUT CONFIGURATION ===
    
    fig.update_xaxes(
        title_text="<b>Pipeline Distance (KP in km)</b>",
        title_font=dict(size=14, color='#00d4ff'),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 168, 255, 0.15)',
        range=[0, pipeline_length],
        dtick=2 if pipeline_length <= 30 else 5,
        tickfont=dict(size=11, color='#b8d4e8')
    )
    
    fig.update_yaxes(
        title_text="<b>Pressure (psi)</b>",
        title_font=dict(size=14, color='#00d4ff'),
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0, 168, 255, 0.15)',
        tickfont=dict(size=11, color='#b8d4e8'),
        secondary_y=False
    )
    
    fig.update_yaxes(
        title_text="<b>Elevation (m)</b>",
        title_font=dict(size=14, color='rgba(122, 154, 184, 0.8)'),
        showgrid=False,
        tickfont=dict(size=11, color='rgba(122, 154, 184, 0.8)'),
        secondary_y=True
    )
    
    fig.update_layout(
        title=dict(
            text=f"<b>Pipeline Pressure Profile & Elevation Analysis Jalur {pipeline_name}</b>",
            font=dict(size=18, color='#00d4ff', family='Orbitron'),
            x=0.5,
            xanchor='center'
        ),
        plot_bgcolor='rgba(10, 14, 39, 0.95)',
        paper_bgcolor='rgba(15, 23, 41, 0.95)',
        height=550,
        hovermode='x unified',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="center",
            x=0.5,
            bgcolor="rgba(0, 0, 0, 0.7)",
            bordercolor="rgba(0, 168, 255, 0.5)",
            borderwidth=2,
            font=dict(size=11, color='#b8d4e8', family='Rajdhani')
        ),
        margin=dict(l=70, r=70, t=120, b=70)
    )
    
    return fig

def create_interactive_pipeline_map(elevation_df, gps_mapper, results, active_data, 
                                     config, pipeline_name):
    """
    Create Interactive Folium Map with Pipeline Route and Leak Detection
    
    Features:
    - Pipeline route visualization
    - Sensor position markers (with KP labels)
    - Leak location marker
    """
    
    # Get center coordinates (middle of pipeline)
    center_kp = config['total_length'] / 2
    center_lat, center_lon = gps_mapper.get_coordinates(center_kp)
    
    # Create base map with satellite view
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles=None,  # We'll add custom tiles
        control_scale=True
    )
    
    # Add multiple tile layers
    folium.TileLayer(
        'OpenStreetMap',
        name='Street Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri',
        name='Satellite View',
        overlay=False,
        control=True
    ).add_to(m)
    
    folium.TileLayer(
        'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
        attr='OpenTopoMap',
        name='Topographic Map',
        overlay=False,
        control=True
    ).add_to(m)
    
    # === 1. PIPELINE ROUTE ===
    # Get all lat/long points from elevation data
    pipeline_coords = []
    for idx, row in elevation_df.iterrows():
        pipeline_coords.append([row['latitude'], row['longitude']])
    
    # Draw pipeline route as blue line
    folium.PolyLine(
        locations=pipeline_coords,
        color='#00a8ff',
        weight=6,
        opacity=0.8,
        popup=f'<b>Pipeline: {pipeline_name}</b><br>Total Length: {config["total_length"]} km',
        tooltip=f'Pipeline Route: {pipeline_name}'
    ).add_to(m)
    
    # === 2. SENSOR MARKERS WITH KP LABELS ===
    for i, (kp, name, normal_p, drop_p) in enumerate(zip(
        active_data['locations'],
        active_data['names'],
        active_data['normal'],
        active_data['drop']
    )):
        lat, lon = gps_mapper.get_coordinates(kp)
        delta_p = normal_p - drop_p
        
        # Sensor popup content
        popup_html = f"""
        <div style="font-family: 'Rajdhani', sans-serif; min-width: 250px;">
            <h4 style="color: #0066ff; margin: 0 0 10px 0;">📍 Sensor KP {kp:.1f}</h4>
            <table style="width: 100%; font-size: 13px;">
                <tr><td><b>Name:</b></td><td>{name}</td></tr>
                <tr><td><b>GPS:</b></td><td>{lat:.6f}, {lon:.6f}</td></tr>
                <tr style="background: rgba(0,212,255,0.1);"><td><b>Normal Pressure:</b></td><td>{normal_p:.2f} psi</td></tr>
                <tr style="background: rgba(255,51,102,0.1);"><td><b>Drop Pressure:</b></td><td>{drop_p:.2f} psi</td></tr>
                <tr><td><b>ΔP:</b></td><td>{delta_p:.2f} psi</td></tr>
            </table>
        </div>
        """
        
        # Large KP marker with border
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(html=f'''
                <div style="
                    background: linear-gradient(135deg, #0066ff 0%, #00a8ff 100%);
                    color: white;
                    padding: 12px 20px;
                    border-radius: 20px;
                    font-weight: 900;
                    font-size: 16px;
                    font-family: 'Orbitron', sans-serif;
                    border: 4px solid white;
                    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4), 0 0 20px rgba(0, 168, 255, 0.6);
                    white-space: nowrap;
                    text-align: center;
                    min-width: 120px;
                    letter-spacing: 1px;
                ">
                    📍 KP {kp:.1f}
                </div>
            '''),
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f'Sensor KP {kp:.1f}'
        ).add_to(m)
        
        # Add large circle marker for better visibility
        folium.CircleMarker(
            location=[lat, lon],
            radius=12,
            color='#00d4ff',
            fillColor='#0066ff',
            fillOpacity=0.8,
            weight=4
        ).add_to(m)
    
    # === 3. LEAK LOCATION MARKER ===
    leak_kp = results['final_estimate']
    leak_lat, leak_lon = gps_mapper.get_coordinates(leak_kp)
    
    leak_popup_html = f"""
    <div style="font-family: 'Rajdhani', sans-serif; min-width: 300px;">
        <h3 style="color: #ff3366; margin: 0 0 10px 0;">🚨 ESTIMATED LEAK LOCATION</h3>
        <table style="width: 100%; font-size: 14px;">
            <tr style="background: rgba(255,51,102,0.15);"><td><b>KP Location:</b></td><td style="color: #ff3366; font-weight: bold; font-size: 18px;">KP {leak_kp:.2f}</td></tr>
            <tr><td><b>GPS Coordinates:</b></td><td>{leak_lat:.8f}<br>{leak_lon:.8f}</td></tr>
            <tr><td><b>Uncertainty:</b></td><td>± {results['estimate_std']:.2f} km</td></tr>
            <tr style="background: rgba(0,212,255,0.1);"><td><b>Confidence:</b></td><td><b>{results['confidence']}</b></td></tr>
        </table>
        <div style="margin-top: 12px; padding: 10px; background: rgba(255,51,102,0.2); border-radius: 8px; text-align: center; border: 2px solid #ff3366;">
            <b style="color: #ff3366; font-size: 15px;">⚠️ HIGH PRIORITY INSPECTION REQUIRED</b>
        </div>
    </div>
    """
    
    # Large leak KP marker with pulsing effect
    folium.Marker(
        location=[leak_lat, leak_lon],
        icon=folium.DivIcon(html=f'''
            <div style="
                background: linear-gradient(135deg, #ff3366 0%, #ff0044 100%);
                color: white;
                padding: 15px 25px;
                border-radius: 25px;
                font-weight: 900;
                font-size: 18px;
                font-family: 'Orbitron', sans-serif;
                border: 5px solid white;
                box-shadow: 
                    0 4px 16px rgba(255, 51, 102, 0.6), 
                    0 0 30px rgba(255, 51, 102, 0.8),
                    0 0 50px rgba(255, 0, 68, 0.4);
                white-space: nowrap;
                text-align: center;
                min-width: 180px;
                letter-spacing: 1px;
                animation: pulse 2s ease-in-out infinite;
            ">
                🚨 LEAK KP {leak_kp:.2f}
            </div>
            <style>
                @keyframes pulse {{
                    0%, 100% {{ transform: scale(1); opacity: 1; }}
                    50% {{ transform: scale(1.05); opacity: 0.9; }}
                }}
            </style>
        '''),
        popup=folium.Popup(leak_popup_html, max_width=350),
        tooltip='🚨 Estimated Leak Location - CLICK FOR DETAILS'
    ).add_to(m)
    
    # Add large pulsing circle for leak location
    folium.CircleMarker(
        location=[leak_lat, leak_lon],
        radius=20,
        color='#ff3366',
        fillColor='#ff0044',
        fillOpacity=0.5,
        weight=5
    ).add_to(m)
    
    # Add smaller inner circle
    folium.CircleMarker(
        location=[leak_lat, leak_lon],
        radius=10,
        color='#ff0044',
        fillColor='#ff3366',
        fillOpacity=0.8,
        weight=3
    ).add_to(m)
    
    # === 4. ADD LEGEND ===
    legend_html = '''
    <div style="
        position: fixed;
        bottom: 50px;
        right: 50px;
        width: 280px;
        background: rgba(10, 14, 39, 0.95);
        border: 2px solid rgba(0, 168, 255, 0.5);
        border-radius: 12px;
        padding: 15px;
        font-family: 'Rajdhani', sans-serif;
        color: #b8d4e8;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.5);
        z-index: 9999;
    ">
        <h4 style="margin: 0 0 10px 0; color: #00d4ff; font-family: 'Orbitron', sans-serif;">Map Legend</h4>
        <div style="margin: 8px 0;">
            <span style="background: #00a8ff; padding: 2px 8px; border-radius: 3px;">━━</span>
            <span style="margin-left: 8px;">Pipeline Route</span>
        </div>
        <div style="margin: 8px 0;">
            <span style="background: linear-gradient(135deg, #0066ff, #00a8ff); padding: 5px 12px; border-radius: 8px; color: white; font-size: 11px; font-weight: bold; border: 2px solid white;">📍 KP</span>
            <span style="margin-left: 8px;">Sensor Location</span>
        </div>
        <div style="margin: 8px 0;">
            <span style="background: linear-gradient(135deg, #ff3366, #ff0044); padding: 5px 12px; border-radius: 8px; color: white; font-size: 11px; font-weight: bold; border: 2px solid white;">🚨 KP</span>
            <span style="margin-left: 8px;">Estimated Leak</span>
        </div>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topright',
        title='Fullscreen',
        title_cancel='Exit Fullscreen',
        force_separate_button=True
    ).add_to(m)
    
    # Add measure control for distance measurement
    plugins.MeasureControl(
        position='topleft',
        primary_length_unit='kilometers',
        secondary_length_unit='meters',
        primary_area_unit='hectares',
        secondary_area_unit='sqmeters'
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl(position='topright').add_to(m)
    
    # Fit bounds to show entire pipeline with some padding
    m.fit_bounds([[lat, lon] for lat, lon in pipeline_coords])
    
    return m

# ============================================================================
# LOAD ELEVATION DATA WITH DISTANCE CALCULATION
# ============================================================================

def load_elevation_data(file_path):
    """
    Load elevation data dan calculate cumulative distance
    
    Returns:
        DataFrame dengan columns: latitude, longitude, elevation, distance_km
    """
    try:
        df = pd.read_excel(file_path)
        
        # Rename columns
        df.columns = ['latitude', 'longitude', 'elevation']
        
        # Calculate cumulative distance using Haversine formula
        distances = [0.0]  # Start dari 0
        
        for i in range(1, len(df)):
            lat1, lon1 = df.iloc[i-1]['latitude'], df.iloc[i-1]['longitude']
            lat2, lon2 = df.iloc[i]['latitude'], df.iloc[i]['longitude']
            
            # Haversine formula
            R = 6371  # Earth radius in km
            dlat = np.radians(lat2 - lat1)
            dlon = np.radians(lon2 - lon1)
            a = (np.sin(dlat/2)**2 + 
                 np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * 
                 np.sin(dlon/2)**2)
            c = 2 * np.arcsin(np.sqrt(a))
            dist = R * c
            
            distances.append(distances[-1] + dist)
        
        df['distance_km'] = distances
        
        return df, None
    
    except FileNotFoundError:
        return None, f"File '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading elevation data: {str(e)}"

# ============================================================================
# BASE CONFIGURATION
# ============================================================================
BASE_CONFIG = {
    'SUSPICION_WEIGHTS': [0.5, 0.25, 0.25],
    'UPSTREAM_BIAS_PRIMARY': -2.2,
    'UPSTREAM_BIAS_GRADIENT': -1.8,
    'UPSTREAM_BIAS_INTERP': -2.0,
    'UPSTREAM_BIAS_WEIGHTED': -1.8,
    'PSI_PER_METER': 0.433,
    'FLUID_DENSITY': 0.85,
    'FINAL_ESTIMATE_WEIGHTS': {
        'suspicion': 0.25,
        'interpolation': 0.2,
        'gradient': 0.20,
        'elevation': 0.3,
        'weighted': 0.05
    }
}

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="FOL - Pipeline Leak Detection v2.0",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - FUTURISTIC BLUE THEME
# ============================================================================
st.markdown("""
<style>
    /* Import Futuristic Font */
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;700;900&family=Rajdhani:wght@300;400;500;600;700&display=swap');
    
    /* Main theme - Cyber Blue */
    .main {
        background: linear-gradient(135deg, #0a0e27 0%, #0f1729 50%, #1a1f3a 100%);
        background-attachment: fixed;
    }
    
    /* Animated background grid */
    .main::before {
        content: '';
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        background-image: 
            linear-gradient(rgba(0, 174, 255, 0.03) 1px, transparent 1px),
            linear-gradient(90deg, rgba(0, 174, 255, 0.03) 1px, transparent 1px);
        background-size: 50px 50px;
        pointer-events: none;
        z-index: 0;
    }
    
    /* Header styling - Neon Blue */
    .header-container {
        background: linear-gradient(135deg, #0066ff 0%, #00a8ff 50%, #00d4ff 100%);
        padding: 2.5rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        box-shadow: 
            0 8px 32px rgba(0, 102, 255, 0.4),
            0 0 60px rgba(0, 168, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        border: 1px solid rgba(0, 212, 255, 0.3);
        position: relative;
        overflow: hidden;
    }
    
    .header-container::before {
        content: '';
        position: absolute;
        top: -50%;
        left: -50%;
        width: 200%;
        height: 200%;
        background: linear-gradient(
            45deg,
            transparent 30%,
            rgba(255, 255, 255, 0.1) 50%,
            transparent 70%
        );
        animation: shine 3s infinite;
    }
    
    @keyframes shine {
        0% { transform: translateX(-100%) translateY(-100%) rotate(45deg); }
        100% { transform: translateX(100%) translateY(100%) rotate(45deg); }
    }
    
    .header-title {
        color: white;
        font-size: 3.5rem;
        font-weight: 900;
        margin: 0;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 
            0 0 10px rgba(0, 212, 255, 0.8),
            0 0 20px rgba(0, 168, 255, 0.6),
            0 0 30px rgba(0, 102, 255, 0.4),
            2px 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: 2px;
        position: relative;
    }
    
    .header-subtitle {
        color: rgba(255, 255, 255, 0.95);
        font-size: 1.3rem;
        text-align: center;
        margin-top: 1rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
        text-shadow: 0 2px 8px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    /* Pipeline card - Holographic effect */
    .pipeline-card {
        background: linear-gradient(135deg, #1a2332 0%, #243447 100%);
        padding: 1.8rem;
        border-radius: 15px;
        border: 1px solid rgba(0, 168, 255, 0.3);
        box-shadow: 
            0 4px 20px rgba(0, 102, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    
    .pipeline-card::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        height: 100%;
        width: 5px;
        background: linear-gradient(180deg, #00a8ff 0%, #0066ff 100%);
        box-shadow: 0 0 15px rgba(0, 168, 255, 0.6);
    }
    
    .pipeline-name {
        color: #00d4ff;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1rem;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 10px rgba(0, 212, 255, 0.5);
    }
    
    .pipeline-info {
        color: #b8d4e8;
        font-size: 1rem;
        line-height: 1.8;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 500;
    }
    
    /* Sensor card - Tech panel style */
    .sensor-card {
        background: linear-gradient(135deg, rgba(0, 102, 255, 0.05) 0%, rgba(0, 168, 255, 0.08) 100%);
        backdrop-filter: blur(10px);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid rgba(0, 168, 255, 0.2);
        margin-bottom: 1rem;
        box-shadow: 
            0 4px 15px rgba(0, 102, 255, 0.1),
            inset 0 1px 0 rgba(255, 255, 255, 0.03);
    }
    
    /* Result card - Premium glow */
    .result-card {
        background: linear-gradient(135deg, #0d1b2a 0%, #1b2838 100%);
        padding: 2.5rem;
        border-radius: 20px;
        border: 2px solid rgba(0, 168, 255, 0.4);
        margin-top: 2rem;
        box-shadow: 
            0 8px 40px rgba(0, 102, 255, 0.3),
            0 0 60px rgba(0, 168, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        position: relative;
    }
    
    .result-card::before {
        content: '';
        position: absolute;
        inset: 0;
        border-radius: 20px;
        padding: 2px;
        background: linear-gradient(135deg, #00a8ff, #0066ff, #00d4ff);
        -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
        -webkit-mask-composite: xor;
        mask-composite: exclude;
        opacity: 0.5;
    }
    
    .result-title {
        color: #00d4ff;
        font-size: 2.5rem;
        font-weight: 900;
        text-align: center;
        margin-bottom: 2rem;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 
            0 0 20px rgba(0, 212, 255, 0.6),
            0 0 40px rgba(0, 168, 255, 0.4);
    }
    
    .result-location {
        background: linear-gradient(135deg, rgba(0, 102, 255, 0.15) 0%, rgba(0, 168, 255, 0.1) 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        margin: 2rem 0;
        border: 2px solid rgba(0, 168, 255, 0.4);
        box-shadow: 
            0 4px 30px rgba(0, 102, 255, 0.2),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
    }
    
    .result-location-value {
        color: #00d4ff;
        font-size: 4rem;
        font-weight: 900;
        margin: 0;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 
            0 0 20px rgba(0, 212, 255, 0.8),
            0 0 40px rgba(0, 168, 255, 0.5);
        animation: pulse 2s ease-in-out infinite;
    }
    
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.8; }
    }
    
    .result-location-label {
        color: #a8c5da;
        font-size: 1.1rem;
        margin-top: 0.8rem;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    /* GPS Card - Cyan gradient */
    .gps-card {
        background: linear-gradient(135deg, #006699 0%, #0099cc 50%, #00ccff 100%);
        padding: 2rem;
        border-radius: 15px;
        margin: 2rem 0;
        box-shadow: 
            0 8px 32px rgba(0, 153, 204, 0.4),
            0 0 60px rgba(0, 204, 255, 0.3);
        border: 1px solid rgba(0, 204, 255, 0.3);
    }
    
    .gps-title {
        color: white;
        font-size: 1.8rem;
        font-weight: 700;
        margin-bottom: 1.5rem;
        text-align: center;
        font-family: 'Orbitron', sans-serif;
        text-shadow: 0 0 15px rgba(0, 0, 0, 0.3);
    }
    
    /* Confidence badge - Neon style */
    .confidence-badge {
        display: inline-block;
        padding: 0.7rem 2rem;
        border-radius: 30px;
        font-weight: 700;
        font-size: 1.1rem;
        margin: 0.5rem;
        font-family: 'Rajdhani', sans-serif;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        border: 2px solid rgba(255, 255, 255, 0.2);
    }
    
    .confidence-very-high {
        background: linear-gradient(135deg, #00ff88 0%, #00cc99 100%);
        color: #001a0d;
        box-shadow: 
            0 4px 20px rgba(0, 255, 136, 0.4),
            0 0 30px rgba(0, 204, 153, 0.3);
    }
    
    .confidence-high {
        background: linear-gradient(135deg, #00d4ff 0%, #0099ff 100%);
        color: #001a33;
        box-shadow: 
            0 4px 20px rgba(0, 212, 255, 0.4),
            0 0 30px rgba(0, 153, 255, 0.3);
    }
    
    .confidence-medium {
        background: linear-gradient(135deg, #ff9d00 0%, #ff6600 100%);
        color: #1a0a00;
        box-shadow: 
            0 4px 20px rgba(255, 157, 0, 0.4),
            0 0 30px rgba(255, 102, 0, 0.3);
    }
    
    /* Zone card - Holographic borders */
    .zone-card {
        background: linear-gradient(135deg, rgba(0, 102, 255, 0.08) 0%, rgba(0, 168, 255, 0.05) 100%);
        padding: 1.2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.2);
        backdrop-filter: blur(5px);
    }
    
    .zone-focus {
        border-color: #ff3366;
        box-shadow: 
            0 4px 15px rgba(255, 51, 102, 0.3),
            inset 0 0 20px rgba(255, 51, 102, 0.1);
    }
    
    .zone-critical {
        border-color: #ff9933;
        box-shadow: 
            0 4px 15px rgba(255, 153, 51, 0.3),
            inset 0 0 20px rgba(255, 153, 51, 0.1);
    }
    
    .zone-primary {
        border-color: #ffcc00;
        box-shadow: 
            0 4px 15px rgba(255, 204, 0, 0.3),
            inset 0 0 20px rgba(255, 204, 0, 0.1);
    }
    
    .zone-title {
        font-weight: 700;
        font-size: 1.2rem;
        margin-bottom: 0.5rem;
        font-family: 'Rajdhani', sans-serif;
        color: #00d4ff;
    }
    
    .zone-range {
        color: #b8d4e8;
        font-size: 1rem;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Info box - Cyber style */
    .info-box {
        background: linear-gradient(135deg, rgba(0, 153, 255, 0.1) 0%, rgba(0, 102, 255, 0.05) 100%);
        border-left: 4px solid #0099ff;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 
            0 4px 15px rgba(0, 153, 255, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        color: #b8d4e8;
        font-family: 'Rajdhani', sans-serif;
    }
    
    .warning-box {
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.1) 0%, rgba(255, 102, 0, 0.05) 100%);
        border-left: 4px solid #ff9900;
        padding: 1.2rem;
        border-radius: 8px;
        margin: 1.5rem 0;
        box-shadow: 
            0 4px 15px rgba(255, 153, 0, 0.15),
            inset 0 1px 0 rgba(255, 255, 255, 0.05);
        color: #ffe4b3;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Button styling - Futuristic */
    .stButton>button {
        background: linear-gradient(135deg, #0066ff 0%, #00a8ff 100%);
        color: white;
        font-weight: 700;
        font-size: 1.2rem;
        padding: 1rem 2.5rem;
        border-radius: 12px;
        border: 2px solid rgba(0, 168, 255, 0.5);
        box-shadow: 
            0 4px 20px rgba(0, 102, 255, 0.4),
            0 0 30px rgba(0, 168, 255, 0.3),
            inset 0 1px 0 rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        font-family: 'Rajdhani', sans-serif;
        letter-spacing: 1px;
        text-transform: uppercase;
    }
    
    .stButton>button:hover {
        transform: translateY(-3px);
        box-shadow: 
            0 6px 30px rgba(0, 102, 255, 0.6),
            0 0 50px rgba(0, 168, 255, 0.5);
        background: linear-gradient(135deg, #0077ff 0%, #00b8ff 100%);
    }
    
    /* Link button styling */
    .stLinkButton>a {
        background: linear-gradient(135deg, #0066ff 0%, #00a8ff 100%);
        color: white !important;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.8rem 1.5rem;
        border-radius: 10px;
        border: 1px solid rgba(0, 168, 255, 0.5);
        box-shadow: 0 4px 15px rgba(0, 102, 255, 0.3);
        transition: all 0.3s ease;
        font-family: 'Rajdhani', sans-serif;
        text-decoration: none;
        display: inline-block;
    }
    
    .stLinkButton>a:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 25px rgba(0, 102, 255, 0.5);
        background: linear-gradient(135deg, #0077ff 0%, #00b8ff 100%);
        text-decoration: none;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        background: rgba(0, 102, 255, 0.05);
        border-radius: 10px;
        border: 1px solid rgba(0, 168, 255, 0.2);
    }
    
    /* Input fields */
    .stNumberInput>div>div>input {
        background: linear-gradient(135deg, rgba(0, 102, 255, 0.08) 0%, rgba(0, 168, 255, 0.05) 100%);
        border: 1px solid rgba(0, 168, 255, 0.3);
        border-radius: 8px;
        color: #00d4ff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
        font-size: 1.1rem;
    }
    
    .stNumberInput>div>div>input:focus {
        border-color: #00a8ff;
        box-shadow: 0 0 15px rgba(0, 168, 255, 0.4);
    }
    
    /* Selectbox */
    .stSelectbox>div>div>div {
        background: linear-gradient(135deg, rgba(0, 102, 255, 0.08) 0%, rgba(0, 168, 255, 0.05) 100%);
        border: 1px solid rgba(0, 168, 255, 0.3);
        border-radius: 8px;
        color: #00d4ff;
        font-family: 'Rajdhani', sans-serif;
        font-weight: 600;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0a0e27 0%, #0f1729 50%, #1a1f3a 100%);
        border-right: 2px solid rgba(0, 168, 255, 0.2);
        box-shadow: 4px 0 20px rgba(0, 102, 255, 0.15);
    }
    
    [data-testid="stSidebar"] .element-container {
        color: #b8d4e8;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Headers in content */
    h1, h2, h3 {
        color: #00d4ff !important;
        font-family: 'Orbitron', sans-serif !important;
        text-shadow: 0 0 15px rgba(0, 212, 255, 0.4);
    }
    
    /* Paragraph text */
    p {
        color: #b8d4e8;
        font-family: 'Rajdhani', sans-serif;
    }
    
    /* Footer */
    .footer {
        text-align: center;
        padding: 2.5rem;
        color: #7a9ab8;
        font-size: 1rem;
        margin-top: 3rem;
        border-top: 2px solid rgba(0, 168, 255, 0.2);
        font-family: 'Rajdhani', sans-serif;
        background: linear-gradient(135deg, rgba(0, 102, 255, 0.05) 0%, rgba(0, 168, 255, 0.03) 100%);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar - Cyber blue */
    ::-webkit-scrollbar {
        width: 12px;
        height: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: #0a0e27;
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(135deg, #0066ff 0%, #00a8ff 100%);
        border-radius: 10px;
        border: 2px solid #0a0e27;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(135deg, #0077ff 0%, #00b8ff 100%);
        box-shadow: 0 0 10px rgba(0, 168, 255, 0.5);
    }
    
    /* Success/Error/Warning messages */
    .stSuccess {
        background: linear-gradient(135deg, rgba(0, 255, 136, 0.15) 0%, rgba(0, 204, 153, 0.1) 100%);
        border-left: 4px solid #00ff88;
        color: #b3ffe6;
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(255, 51, 102, 0.15) 0%, rgba(255, 0, 68, 0.1) 100%);
        border-left: 4px solid #ff3366;
        color: #ffb3cc;
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(255, 153, 0, 0.15) 0%, rgba(255, 102, 0, 0.1) 100%);
        border-left: 4px solid #ff9900;
        color: #ffe4b3;
    }
    
    .stInfo {
        background: linear-gradient(135deg, rgba(0, 153, 255, 0.15) 0%, rgba(0, 102, 255, 0.1) 100%);
        border-left: 4px solid #0099ff;
        color: #b3d9ff;
    }
    
    /* Glowing effect for important elements */
    .glow {
        animation: glow 2s ease-in-out infinite;
    }
    
    @keyframes glow {
        0%, 100% {
            box-shadow: 0 0 20px rgba(0, 168, 255, 0.4);
        }
        50% {
            box-shadow: 0 0 40px rgba(0, 168, 255, 0.6);
        }
    }
</style>
""", unsafe_allow_html=True)

# ============================================================================
# PIPELINE CONFIGURATIONS
# ============================================================================
PIPELINE_CONFIGS = {
    "BJG-TPN": {
        "model_file": "bjg_model.sav",
        "elevation_file": "bjg_elevasi.xlsx",
        "total_length": 26.6,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 7.14 SPOT 2 FOL", "location": 7.14},
            {"name": "III. KP 15.4 SPOT 3 FOL", "location": 15.4},
            {"name": "IV. KP 19.7 SPOT 4 FOL", "location": 19.7}
        ],
        "example_normal": [136.0, 112.14, 95.4, 37.1],
        "example_drop": [133.0, 110.1, 82.5, 34.5],
        "description": "Pipeline dari Betara Jambi (BJG) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil"
    },
    "BTJ-TPN": {
        "model_file": "btj_model.sav",
        "elevation_file": "btj_elevasi.xlsx",
        "total_length": 13.8,
        "sensors": [
            {"name": "I. KP 0 SP BTJ", "location": 0.0},
            {"name": "II. KP SIMP PS Gajah 5.5", "location": 5.5},
            {"name": "III. KP Sandtrap II 9.7", "location": 9.7}
        ],
        "example_normal": [190.2, 80.0, 34.5],
        "example_drop": [185.5, 75.0, 32.1],
        "description": "Pipeline dari Betara Jambi (BTJ) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil"
    },
    "KAS-TPN": {
        "model_file": "kas_model.sav",
        "elevation_file": "kas_elevasi.xlsx",
        "total_length": 23.2,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 7.8 SPOT 2 FOL", "location": 7.8},
            {"name": "III. KP 15.4 SPOT 3 FOL", "location": 15.4},
            {"name": "IV. KP 23.2 SPOT 4 FOL", "location": 23.2}
        ],
        "example_normal": [150.727, 125.92, 84.0367, 43.2134],
        "example_drop": [143.778, 104.54, 64.3846, 40.0],
        "description": "Pipeline dari Kenali Asam (KAS) menuju Tempino (TPN)",
        "fluid_type": "Crude Oil"
    },
    "KTT-KAS": {
        "model_file": "ktt_model.sav",
        "elevation_file": "ktt_elevasi.xlsx",
        "total_length": 44.6,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 44.6 SPOT 2 FOL", "location": 44.6}
        ],
        "example_normal": [190.2, 22.0],
        "example_drop": [165.4, 20.5],
        "description": "Pipeline dari Kenali Asam Timur (KTT) menuju Kenali Asam (KAS)",
        "fluid_type": "Crude Oil"
    },
    "SG-KAS": {
        "model_file": "sg_model.sav",
        "elevation_file": "sg_elevasi.xlsx",
        "total_length": 11.2,
        "sensors": [
            {"name": "I. KP 0 SPOT 1 FOL", "location": 0.0},
            {"name": "II. KP 11.2 SPOT 2 FOL", "location": 11.2}
        ],
        "example_normal": [276.6, 20.0],
        "example_drop": [268.01, 18.0],
        "description": "Pipeline dari Sei Gelam (SG) menuju Kenali Asam (KAS)",
        "fluid_type": "Crude Oil"
    }
}

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

@st.cache_resource
def load_model_local(file_path):
    """Load model from local file"""
    try:
        with open(file_path, 'rb') as f:
            model = pickle.load(f)
        return model, None
    except FileNotFoundError:
        return None, f"Model file '{file_path}' tidak ditemukan"
    except Exception as e:
        return None, f"Error loading model: {str(e)}"

@st.cache_data
def load_pipeline_elevation(file_path):
    """Load elevation data untuk pipeline tertentu"""
    return load_elevation_data(file_path)

def validate_inputs(normal_pressure, drop_pressure):
    """Validate pressure inputs"""
    errors = []
    warnings = []
    
    valid_pairs = sum(1 for n, d in zip(normal_pressure, drop_pressure) if n is not None and d is not None)
    if valid_pairs == 0:
        errors.append("❌ Minimal 1 sensor harus diisi (Normal & Drop pressure)")
    
    for i, (n, d) in enumerate(zip(normal_pressure, drop_pressure)):
        if n is not None and n < 0:
            errors.append(f"❌ Sensor {i+1}: Normal pressure tidak boleh negatif")
        if d is not None and d < 0:
            errors.append(f"❌ Sensor {i+1}: Drop pressure tidak boleh negatif")
        
        if (n is None and d is not None) or (n is not None and d is None):
            warnings.append(f"⚠️ Sensor {i+1}: Isi kedua nilai atau kosongkan keduanya")
    
    return errors, warnings

def filter_active_sensors(config, normal_pressure, drop_pressure):
    """Filter only active sensors"""
    active_sensors = []
    active_locations = []
    active_names = []
    active_normal = []
    active_drop = []
    
    for i, (n, d) in enumerate(zip(normal_pressure, drop_pressure)):
        if n is not None and d is not None:
            active_sensors.append(i)
            active_locations.append(config["sensors"][i]["location"])
            active_names.append(config["sensors"][i]["name"])
            active_normal.append(n)
            active_drop.append(d)
    
    return {
        "indices": active_sensors,
        "locations": np.array(active_locations),
        "names": active_names,
        "normal": np.array(active_normal),
        "drop": np.array(active_drop)
    }

def format_confidence(confidence_str):
    """Format confidence with badge styling"""
    if "VERY HIGH" in confidence_str:
        badge_class = "confidence-very-high"
        icon = "🎯"
    elif "HIGH" in confidence_str:
        badge_class = "confidence-high"
        icon = "✅"
    else:
        badge_class = "confidence-medium"
        icon = "⚠️"
    
    return f'<span class="confidence-badge {badge_class}">{icon} {confidence_str}</span>'

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    # Header
    st.markdown("""
        <div class="header-container">
            <h1 class="header-title">🛢️ FOL - Finding Oil Losses v2.0</h1>
            <p class="header-subtitle">
                Pipeline Leak Detection System<br>
                PT Pertamina EP Jambi Field | <b>Research and Development Team</b>
            </p>
        </div>
    """, unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.markdown("### 📍 Select Pipeline")
        selected_pipeline = st.selectbox(
            "Pilih Jalur Pipa",
            options=list(PIPELINE_CONFIGS.keys()),
            format_func=lambda x: f"{x} ({PIPELINE_CONFIGS[x]['total_length']} km)"
        )
        
        st.markdown("---")
        st.markdown("### ℹ️ System Info")
        st.info(f"""
        **Pipeline:** {selected_pipeline}
        
        **Length:** {PIPELINE_CONFIGS[selected_pipeline]['total_length']} km
        
        **Sensors:** {len(PIPELINE_CONFIGS[selected_pipeline]['sensors'])}
        
        **Fluid:** {PIPELINE_CONFIGS[selected_pipeline]['fluid_type']}
        """)
        
        st.markdown("---")
        st.markdown("### 🎯 Features v2.0")
        st.markdown("""
        - ✅ Multi-condition prediction
        - ✅ Partial sensor data support
        - ✅ Real-time analysis
        - ✅ Elevation-corrected
        - ✅ Multiple detection methods
        - 🆕 **GPS Location & Google Maps**
        - 🆕 **Interactive Map Links**
        """)
        
        st.markdown("---")
        st.markdown(f"**Version:** 2.0 GPS Enhanced")
        st.markdown(f"**Last Updated:** {datetime.now().strftime('%d %B %Y')}")
    
    # Main content
    config = PIPELINE_CONFIGS[selected_pipeline]
    
    # Pipeline info card
    st.markdown(f"""
        <div class="pipeline-card">
            <div class="pipeline-name">📍 {selected_pipeline}</div>
            <div class="pipeline-info">
                <b>Route:</b> {config['description']}<br>
                <b>Total Length:</b> {config['total_length']} km<br>
                <b>Available Sensors:</b> {len(config['sensors'])} units<br>
                <b>Fluid Type:</b> {config['fluid_type']}<br>
                <b>GPS Data:</b> ✅ Available (Lat/Long tracking enabled)
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    # Info box
    st.markdown("""
        <div class="info-box">
            💡 <b>Cara Penggunaan:</b><br>
            • Isi <b>Normal Pressure</b> dan <b>Drop Pressure</b> untuk sensor yang aktif<br>
            • Untuk sensor yang tidak aktif/mati, <b>biarkan nilai 0</b> atau kosongkan<br>
            • Minimal 1 sensor harus terisi untuk melakukan prediksi<br>
            • Sistem akan otomatis menyesuaikan analisis berdasarkan jumlah sensor aktif<br>
            • <b>🆕 GPS coordinates dan Google Maps link akan otomatis dihasilkan!</b>
        </div>
    """, unsafe_allow_html=True)
    
    # Sensor inputs
    st.markdown("---")
    st.markdown("## 📊 Sensor Data Input")
    
    # Create button row
    col_btn1, col_btn2, col_space = st.columns([1, 1, 2])
    
    with col_btn1:
        use_example = st.button("📝 Load Example Data", use_container_width=True)
    
    with col_btn2:
        clear_data = st.button("🗑️ Clear All Data", use_container_width=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Initialize session state
    if 'sensor_values' not in st.session_state:
        st.session_state.sensor_values = {}
    
    # Create table header
    st.markdown("""
        <div style="background: rgba(255, 107, 53, 0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
            <div style="display: grid; grid-template-columns: 3fr 1fr 2fr 2fr; gap: 1rem; font-weight: 700; color: #ff6b35;">
                <div>Sensor Name</div>
                <div style="text-align: center;">KP (km)</div>
                <div style="text-align: center;">Normal Pressure (psi)</div>
                <div style="text-align: center;">Drop Pressure (psi)</div>
            </div>
        </div>
    """, unsafe_allow_html=True)
    
    normal_pressure = []
    drop_pressure = []
    
    # Sensor input rows
    for i, sensor in enumerate(config["sensors"]):
        st.markdown(f"""
            <div class="sensor-card" style="padding: 0.8rem; margin-bottom: 0.5rem;">
                <div style="color: #ff6b35; font-weight: 600; font-size: 0.9rem; margin-bottom: 0.5rem;">
                    {sensor['name']}
                </div>
            </div>
        """, unsafe_allow_html=True)
        
        col_name, col_km, col_normal, col_drop = st.columns([3, 1, 2, 2])
        
        with col_km:
            st.markdown(f"""
                <div style="text-align: center; padding: 0.5rem; background: rgba(255, 107, 53, 0.1); border-radius: 5px; margin-top: 0.3rem;">
                    <span style="font-weight: 700; color: #ff6b35; font-size: 1.1rem;">{sensor['location']}</span>
                </div>
            """, unsafe_allow_html=True)
        
        with col_normal:
            if use_example:
                default_normal = config["example_normal"][i]
            elif clear_data:
                default_normal = 0.0
            else:
                key = f"normal_{selected_pipeline}_{i}"
                default_normal = st.session_state.sensor_values.get(key, 0.0)
            
            normal_val = st.number_input(
                f"Normal {i+1}",
                min_value=0.0,
                max_value=500.0,
                value=default_normal,
                step=0.01,
                format="%.2f",
                key=f"normal_{selected_pipeline}_{i}",
                label_visibility="collapsed"
            )
            
            st.session_state.sensor_values[f"normal_{selected_pipeline}_{i}"] = normal_val
            normal_pressure.append(normal_val if normal_val > 0 else None)
        
        with col_drop:
            if use_example:
                default_drop = config["example_drop"][i]
            elif clear_data:
                default_drop = 0.0
            else:
                key = f"drop_{selected_pipeline}_{i}"
                default_drop = st.session_state.sensor_values.get(key, 0.0)
            
            drop_val = st.number_input(
                f"Drop {i+1}",
                min_value=0.0,
                max_value=500.0,
                value=default_drop,
                step=0.01,
                format="%.2f",
                key=f"drop_{selected_pipeline}_{i}",
                label_visibility="collapsed"
            )
            
            st.session_state.sensor_values[f"drop_{selected_pipeline}_{i}"] = drop_val
            drop_pressure.append(drop_val if drop_val > 0 else None)
    
    # Validate inputs
    st.markdown("---")
    validation_errors, validation_warnings = validate_inputs(normal_pressure, drop_pressure)
    
    if validation_errors:
        for error in validation_errors:
            st.error(error)
    
    if validation_warnings:
        for warning in validation_warnings:
            st.warning(warning)
    
    # Show active sensors count
    active_count = sum(1 for n, d in zip(normal_pressure, drop_pressure) if n is not None and d is not None)
    if active_count > 0:
        st.success(f"✅ {active_count} sensor aktif dari {len(config['sensors'])} sensor tersedia")
    
    # Predict button
    col_predict, col_space = st.columns([1, 3])
    with col_predict:
        predict_button = st.button(
            "🔍 DETECT LEAK LOCATION",
            disabled=len(validation_errors) > 0,
            use_container_width=True
        )
    
    # Prediction
    if predict_button and len(validation_errors) == 0:
        with st.spinner("🔄 Loading model and analyzing data..."):
            # Load model
            model_obj, error = load_model_local(config["model_file"])
            
            if model_obj is None:
                st.error(f"❌ Failed to load model: {error}")
                return
            
            # Load elevation data
            elev_df, elev_error = load_pipeline_elevation(config["elevation_file"])
            
            if elev_df is None:
                st.warning(f"⚠️ Elevation data tidak tersedia: {elev_error}")
                st.info("Prediksi akan dilanjutkan tanpa data elevasi dan GPS")
                analyzer = EnhancedLeakAnalyzer(BASE_CONFIG)
                gps_mapper = None
            else:
                analyzer = EnhancedLeakAnalyzer(BASE_CONFIG, elev_df)
                gps_mapper = GPSLocationMapper(elev_df)
                st.success("✅ Elevation data dan GPS mapping loaded successfully")
            
            # Filter active sensors
            active_data = filter_active_sensors(config, normal_pressure, drop_pressure)
            
            if len(active_data["indices"]) == 0:
                st.error("❌ Tidak ada sensor yang terisi dengan lengkap!")
                return
            
            st.info(f"""
                ℹ️ **Analyzing with {len(active_data['indices'])} active sensor(s):**
                {', '.join([f"Sensor {i+1}" for i in active_data['indices']])}
            """)
            
            # Predict
            try:
                results = analyzer.predict(
                    sensor_locations=active_data["locations"],
                    normal_pressure=active_data["normal"],
                    drop_pressure=active_data["drop"],
                    sensor_names=active_data["names"]
                )
                
                # Display results
                st.markdown("---")
                st.markdown("""
                    <div class="result-card">
                        <div class="result-title">🎯 LEAK DETECTION RESULTS</div>
                """, unsafe_allow_html=True)
                
                # Main result
                st.markdown(f"""
                    <div class="result-location">
                        <div class="result-location-value">KP {results['final_estimate']:.2f}</div>
                        <div class="result-location-label">Estimated Leak Location (± {results['estimate_std']:.2f} km)</div>
                    </div>
                """, unsafe_allow_html=True)
                
                # Confidence
                st.markdown(f"""
                    <div style="text-align: center; margin: 1.5rem 0;">
                        {format_confidence(results['confidence'])}
                    </div>
                """, unsafe_allow_html=True)
                
                st.markdown("</div>", unsafe_allow_html=True)
                
                # GPS LOCATION CARD - NEW!
                if gps_mapper is not None:
                    st.markdown("---")
                    
                    # Get GPS coordinates
                    leak_lat, leak_lon = gps_mapper.get_coordinates(results['final_estimate'])
                    maps_link = gps_mapper.get_google_maps_link(results['final_estimate'], zoom=18)
                    
                    # GPS Card Header
                    st.markdown("""
                        <div class="gps-card">
                            <div class="gps-title">📍 GPS LOCATION & GOOGLE MAPS</div>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # GPS Info in columns for better layout
                    col_gps1, col_gps2 = st.columns(2)
                    
                    with col_gps1:
                        st.markdown(f"""
                            <div style="background: rgba(46, 204, 113, 0.2); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <div style="color: #2ecc71; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                    📌 Leak Location
                                </div>
                                <div style="color: white; font-size: 1.8rem; font-weight: 800;">
                                    KP {results['final_estimate']:.2f} km
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_gps2:
                        st.markdown(f"""
                            <div style="background: rgba(46, 204, 113, 0.2); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;">
                                <div style="color: #2ecc71; font-weight: 700; font-size: 1.1rem; margin-bottom: 0.5rem;">
                                    🎯 GPS Coordinates
                                </div>
                                <div style="color: white; font-size: 1rem; font-weight: 600;">
                                    Lat: {leak_lat:.6f}<br>
                                    Lon: {leak_lon:.6f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    # Google Maps Link - using st.link_button for proper rendering
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    col_map1, col_map2, col_map3 = st.columns([1, 2, 1])
                    with col_map2:
                        st.link_button(
                            "🗺️ OPEN IN GOOGLE MAPS",
                            maps_link,
                            use_container_width=True
                        )
                    
                    st.markdown("<br>", unsafe_allow_html=True)
                    
                    # === PRESSURE PROFILE VISUALIZATION ===
                    st.markdown("---")
                    st.markdown("## 📈 Pressure Profile & Elevation Analysis")
                    
                    # Create the visualization
                    pressure_chart = create_pressure_profile_chart(
                        sensor_locations=active_data["locations"],
                        normal_pressure=active_data["normal"],
                        drop_pressure=active_data["drop"],
                        estimated_kp=results['final_estimate'],
                        pipeline_length=config['total_length'],
                        elevation_df=elev_df if elev_df is not None else None,
                        sensor_names=[f"S{i+1}" for i in range(len(active_data["locations"]))],
                        pipeline_name=selected_pipeline
                    )
                    
                    # Display the chart
                    st.plotly_chart(pressure_chart, use_container_width=True)
                    
                    # Chart interpretation
                    st.markdown("""
                        <div class="info-box">
                            <b>📊 Chart Interpretation:</b><br>
                            • <b style="color: #00d4ff;">Cyan line:</b> Normal operating pressure profile<br>
                            • <b style="color: #ff3366;">Pink line:</b> Pressure during drop/leak event<br>
                            • <b style="color: #00ff51;">green dotted line:</b> Tolerance threshold (97.5% of normal, 2.5% drop tolerance)<br>
                            • <b style="color: #7a9ab8;">Gray area:</b> Elevation profile along pipeline<br>
                            • <b style="color: #ff3366;">Dashed red line:</b> Estimated leak location<br>
                            • <b>Markers:</b> Active sensor readings (circle = normal, square = drop)<br>
                            • <b>Shaded area:</b> High confidence zone for leak detection
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # === INTERACTIVE MAP VISUALIZATION ===
                    st.markdown("---")
                    st.markdown("## 🗺️ Interactive Pipeline Map")
                    
                    # Create the interactive map
                    pipeline_map = create_interactive_pipeline_map(
                        elevation_df=elev_df,
                        gps_mapper=gps_mapper,
                        results=results,
                        active_data=active_data,
                        config=config,
                        pipeline_name=selected_pipeline
                    )
                    
                    # Display the map
                    folium_static(pipeline_map, width=1400, height=600)
                    
                    st.markdown("""
                        <div class="info-box" style="margin-top: 1rem;">
                            💡 <b>Tips:</b> Use the layer control (top-right) to switch between map views. 
                            Click on any marker or zone for detailed information. The measurement tool (top-left) 
                            allows you to calculate distances along the pipeline route.
                        </div>
                    """, unsafe_allow_html=True)
                    
                    # Zone GPS Coordinates
                    st.markdown("---")
                    st.markdown("### 🗺️ Inspection Zones GPS Coordinates")
                    
                    focus = results['zones']['focus']
                    critical = results['zones']['critical']
                    primary = results['zones']['primary']
                    
                    # Ensure zones are within pipeline bounds
                    focus = (max(0, focus[0]), min(config['total_length'], focus[1]))
                    critical = (max(0, critical[0]), min(config['total_length'], critical[1]))
                    primary = (max(0, primary[0]), min(config['total_length'], primary[1]))
                    
                    col_z1, col_z2, col_z3 = st.columns(3)
                    
                    # Focus Zone
                    with col_z1:
                        focus_start_lat, focus_start_lon = gps_mapper.get_coordinates(focus[0])
                        focus_end_lat, focus_end_lon = gps_mapper.get_coordinates(focus[1])
                        focus_link_start = gps_mapper.get_google_maps_link(focus[0])
                        focus_link_end = gps_mapper.get_google_maps_link(focus[1])
                        
                        st.markdown(f"""
                            <div class="zone-card zone-focus">
                                <div class="zone-title">🔴 FOCUS ZONE</div>
                                <div class="zone-range">KP {focus[0]:.1f} - {focus[1]:.1f}</div>
                                <div class="zone-range" style="margin-top: 0.5rem; font-size: 0.85rem;">
                                    Start: {focus_start_lat:.6f}, {focus_start_lon:.6f}<br>
                                    End: {focus_end_lat:.6f}, {focus_end_lon:.6f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.link_button("📍 Start Point", focus_link_start, use_container_width=True)
                        st.link_button("📍 End Point", focus_link_end, use_container_width=True)
                    
                    # Critical Zone
                    with col_z2:
                        crit_start_lat, crit_start_lon = gps_mapper.get_coordinates(critical[0])
                        crit_end_lat, crit_end_lon = gps_mapper.get_coordinates(critical[1])
                        crit_link_start = gps_mapper.get_google_maps_link(critical[0])
                        crit_link_end = gps_mapper.get_google_maps_link(critical[1])
                        
                        st.markdown(f"""
                            <div class="zone-card zone-critical">
                                <div class="zone-title">🟠 CRITICAL ZONE</div>
                                <div class="zone-range">KP {critical[0]:.1f} - {critical[1]:.1f}</div>
                                <div class="zone-range" style="margin-top: 0.5rem; font-size: 0.85rem;">
                                    Start: {crit_start_lat:.6f}, {crit_start_lon:.6f}<br>
                                    End: {crit_end_lat:.6f}, {crit_end_lon:.6f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.link_button("📍 Start Point", crit_link_start, use_container_width=True)
                        st.link_button("📍 End Point", crit_link_end, use_container_width=True)
                    
                    # Primary Zone
                    with col_z3:
                        prim_start_lat, prim_start_lon = gps_mapper.get_coordinates(primary[0])
                        prim_end_lat, prim_end_lon = gps_mapper.get_coordinates(primary[1])
                        prim_link_start = gps_mapper.get_google_maps_link(primary[0])
                        prim_link_end = gps_mapper.get_google_maps_link(primary[1])
                        
                        st.markdown(f"""
                            <div class="zone-card zone-primary">
                                <div class="zone-title">🟡 PRIMARY ZONE</div>
                                <div class="zone-range">KP {primary[0]:.1f} - {primary[1]:.1f}</div>
                                <div class="zone-range" style="margin-top: 0.5rem; font-size: 0.85rem;">
                                    Start: {prim_start_lat:.6f}, {prim_start_lon:.6f}<br>
                                    End: {prim_end_lat:.6f}, {prim_end_lon:.6f}
                                </div>
                            </div>
                        """, unsafe_allow_html=True)
                        
                        st.link_button("📍 Start Point", prim_link_start, use_container_width=True)
                        st.link_button("📍 End Point", prim_link_end, use_container_width=True)
                
                else:
                    # Original zones without GPS
                    st.markdown("---")
                    
                    # === PRESSURE PROFILE VISUALIZATION (Without GPS) ===
                    st.markdown("## 📈 Pressure Profile Analysis")
                    
                    # Create the visualization without elevation
                    pressure_chart = create_pressure_profile_chart(
                        sensor_locations=active_data["locations"],
                        normal_pressure=active_data["normal"],
                        drop_pressure=active_data["drop"],
                        estimated_kp=results['final_estimate'],
                        pipeline_length=config['total_length'],
                        elevation_df=None,
                        sensor_names=[f"S{i+1}" for i in range(len(active_data["locations"]))],
                        pipeline_name=selected_pipeline
                    )
                    
                    # Display the chart
                    st.plotly_chart(pressure_chart, use_container_width=True)
                    
                    # Chart interpretation
                    st.markdown("""
                        <div class="info-box">
                            <b>📊 Chart Interpretation:</b><br>
                            • <b style="color: #00d4ff;">Cyan line:</b> Normal operating pressure profile<br>
                            • <b style="color: #ff3366;">Pink line:</b> Pressure during drop/leak event<br>
                            • <b style="color: #0099ff;">Blue dotted line:</b> Tolerance threshold (97.5% of normal, 2.5% drop tolerance)<br>
                            • <b style="color: #ff3366;">Dashed red line:</b> Estimated leak location<br>
                            • <b>Markers:</b> Active sensor readings (circle = normal, square = drop)<br>
                            • <b>Shaded area:</b> High confidence zone for leak detection
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    st.markdown("## 🎯 Priority Inspection Zones")
                    
                    col_z1, col_z2, col_z3 = st.columns(3)
                    
                    focus = results['zones']['focus']
                    critical = results['zones']['critical']
                    primary = results['zones']['primary']
                    
                    with col_z1:
                        st.markdown(f"""
                            <div class="zone-card zone-focus">
                                <div class="zone-title">🔴 FOCUS ZONE</div>
                                <div class="zone-range">KP {focus[0]:.1f} - {focus[1]:.1f}</div>
                                <div class="zone-range" style="margin-top: 0.5rem; font-weight: 700;">HIGHEST PRIORITY</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_z2:
                        st.markdown(f"""
                            <div class="zone-card zone-critical">
                                <div class="zone-title">🟠 CRITICAL ZONE</div>
                                <div class="zone-range">KP {critical[0]:.1f} - {critical[1]:.1f}</div>
                                <div class="zone-range" style="margin-top: 0.5rem;">Secondary inspection</div>
                            </div>
                        """, unsafe_allow_html=True)
                    
                    with col_z3:
                        st.markdown(f"""
                            <div class="zone-card zone-primary">
                                <div class="zone-title">🟡 PRIMARY ZONE</div>
                                <div class="zone-range">KP {primary[0]:.1f} - {primary[1]:.1f}</div>
                                <div class="zone-range" style="margin-top: 0.5rem;">Tertiary inspection</div>
                            </div>
                        """, unsafe_allow_html=True)
                
                # Method breakdown
                st.markdown("---")
                st.markdown("## 📊 Detection Method Breakdown")
                
                methods_df = pd.DataFrame({
                    "Method": [
                        "Suspicion Index",
                        "Interpolation",
                        "Gradient Analysis",
                        "Elevation/Hydraulic",
                        "Weighted Average"
                    ],
                    "Location (KP)": [
                        f"{results['methods']['suspicion']:.2f}",
                        f"{results['methods']['interpolation']:.2f}",
                        f"{results['methods']['gradient']:.2f}",
                        f"{results['methods']['elevation']:.2f}",
                        f"{results['methods']['weighted']:.2f}"
                    ],
                    "Weight": ["30%", "25%", "20%", "15%", "10%"]
                })
                
                st.dataframe(methods_df, use_container_width=True, hide_index=True)
                
                # Sensor details
                st.markdown("---")
                st.markdown("## 📈 Active Sensor Details")
                
                sensor_data = results['sensor_data']
                sensor_df = pd.DataFrame({
                    "Sensor": sensor_data['names'],
                    "KP (km)": [f"{loc:.1f}" for loc in sensor_data['locations']],
                    "Elevation (m)": [f"{elev:.1f}" if analyzer.has_elevation_data else "N/A" 
                                     for elev in sensor_data['elevations']],
                    "Normal (psi)": [f"{p:.2f}" for p in sensor_data['normal_pressure']],
                    "Drop (psi)": [f"{p:.2f}" for p in sensor_data['drop_pressure']],
                    "ΔP (psi)": [f"{p:.2f}" for p in sensor_data['delta_pressure']],
                    "Suspicion Score": [f"{s:.2f}" for s in sensor_data['suspicion_index']]
                })
                
                st.dataframe(sensor_df, use_container_width=True, hide_index=True)
                
                # Most suspicious sensor
                top_idx = results['top_sensor_idx']
                st.markdown(f"""
                    <div class="warning-box">
                        ⚠️ <b>Most Suspicious Sensor:</b> {sensor_data['names'][top_idx]} 
                        (KP {sensor_data['locations'][top_idx]:.1f}) - 
                        Score: {sensor_data['suspicion_index'][top_idx]:.2f}
                    </div>
                """, unsafe_allow_html=True)
                
                # Export results
                st.markdown("---")
                st.markdown("## 💾 Export Results")
                
                # Create export data
                export_data = {
                    "Pipeline": selected_pipeline,
                    "Analysis Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "Active Sensors": len(active_data['indices']),
                    "Estimated Location (KP)": f"{results['final_estimate']:.2f}",
                    "Uncertainty (km)": f"{results['estimate_std']:.2f}",
                    "Confidence": results['confidence'],
                    "Focus Zone": f"KP {focus[0]:.1f} - {focus[1]:.1f}",
                    "Critical Zone": f"KP {critical[0]:.1f} - {critical[1]:.1f}",
                    "Primary Zone": f"KP {primary[0]:.1f} - {primary[1]:.1f}"
                }
                
                # Add GPS data if available
                if gps_mapper is not None:
                    export_data.update({
                        "GPS Latitude": f"{leak_lat:.8f}",
                        "GPS Longitude": f"{leak_lon:.8f}",
                        "Google Maps Link": maps_link
                    })
                
                export_df = pd.DataFrame([export_data])
                csv = export_df.to_csv(index=False)
                
                st.download_button(
                    label="📥 Download Results (CSV)",
                    data=csv,
                    file_name=f"FOL_LeakDetection_{selected_pipeline}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
                
            except Exception as e:
                st.error(f"❌ **Prediction Error**: {str(e)}")
                st.exception(e)
    
    # Footer
    st.markdown("---")
    st.markdown("""
        <div class="footer">
            <b>FOL - Finding Oil Losses v2.0</b> | Pipeline Leak Detection System <br>
            Developed by <b>Research and Development Team</b> - PT Pertamina EP Jambi Field<br>
            <br>
            Powered by Machine Learning, IoT Technology & GPS Mapping<br>
        </div>
    """, unsafe_allow_html=True)
 
if __name__ == "__main__":
    main()
