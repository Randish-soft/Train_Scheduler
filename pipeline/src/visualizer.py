"""
BCPC Pipeline - Interactive Route Visualizer
==========================================

This module creates interactive HTML visualizations of railway routes,
stations, train types, and infrastructure conditions including tunnels,
bridges, and ground-level sections with appropriate color coding.

Features:
- Interactive Leaflet-based maps
- Route visualization with infrastructure color coding
- Station markers with detailed popups
- Train type indicators and specifications
- Terrain complexity overlays
- Cost and demand information
- Multi-route corridor visualization
- Export to standalone HTML files

Author: BCPC Pipeline Team
License: Open Source
"""

import json
import logging
import math
import numpy as np
import pandas as pd
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple, Union, Any
from enum import Enum
from pathlib import Path
import folium
from folium import plugins
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import branca.colormap as cm
from shapely.geometry import Point, LineString, Polygon
from shapely.ops import transform
import geopandas as gpd

# Import from other BCPC modules (assuming they're available)
try:
    from cost_analysis import CostSummary, NetworkDesign, TrainType, TrackGauge
    from station_placement import CityStationNetwork, StationPlacement, StationType
    from terrain_analysis import TerrainAnalysis, TerrainComplexity, TerrainSegment
except ImportError:
    # Define minimal classes if modules not available
    class TrainType(Enum):
        DIESEL = "diesel"
        ELECTRIC_EMU = "electric_emu"
        ELECTRIC_LOCOMOTIVE = "electric_locomotive"
        HYBRID = "hybrid"
    
    class StationType(Enum):
        TERMINUS = "terminus"
        INTERMEDIATE = "intermediate"
        LOCAL = "local"
        JUNCTION = "junction"
    
    class TerrainComplexity(Enum):
        FLAT = "flat"
        ROLLING = "rolling"
        HILLY = "hilly"
        MOUNTAINOUS = "mountainous"

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class InfrastructureType(Enum):
    """Infrastructure types for color coding"""
    GROUND = "ground"
    ELEVATED = "elevated"
    BRIDGE = "bridge"
    TUNNEL = "tunnel"
    PARTIAL_TUNNEL = "partial_tunnel"
    VIADUCT = "viaduct"

@dataclass
class RouteSegmentVisualization:
    """Visualization data for a route segment"""
    start_point: Point
    end_point: Point
    infrastructure_type: InfrastructureType
    terrain_complexity: TerrainComplexity
    distance_km: float
    cost_per_km: float
    elevation_start: float
    elevation_end: float
    
@dataclass
class StationVisualization:
    """Visualization data for a station"""
    location: Point
    name: str
    station_type: StationType
    daily_passengers: int
    population_served: int
    employment_served: int
    platforms: int
    estimated_cost: float
    elevation: float
    access_modes: List[str]

@dataclass
class RouteVisualization:
    """Complete route visualization data"""
    route_id: str
    route_name: str
    segments: List[RouteSegmentVisualization]
    stations: List[StationVisualization]
    train_type: TrainType
    total_length_km: float
    total_cost: float
    overall_terrain: TerrainComplexity
    daily_passengers_total: int

class BCPCVisualizer:
    """
    Main visualization engine for BCPC railway projects
    """
    
    def __init__(self, 
                 default_center: Tuple[float, float] = (50.8503, 4.3517),  # Brussels
                 default_zoom: int = 8):
        """
        Initialize the visualizer
        
        Args:
            default_center: Default map center (lat, lon)
            default_zoom: Default zoom level
        """
        self.default_center = default_center
        self.default_zoom = default_zoom
        
        # Color schemes
        self.infrastructure_colors = {
            InfrastructureType.GROUND: '#2E8B57',        # Sea Green
            InfrastructureType.ELEVATED: '#4169E1',      # Royal Blue
            InfrastructureType.BRIDGE: '#FF6347',        # Tomato
            InfrastructureType.TUNNEL: '#8B4513',        # Saddle Brown
            InfrastructureType.PARTIAL_TUNNEL: '#D2691E', # Chocolate
            InfrastructureType.VIADUCT: '#9370DB'        # Medium Purple
        }
        
        self.train_type_colors = {
            TrainType.DIESEL: '#FF4500',           # Orange Red
            TrainType.ELECTRIC_EMU: '#32CD32',     # Lime Green
            TrainType.ELECTRIC_LOCOMOTIVE: '#0000FF', # Blue
            TrainType.HYBRID: '#FFD700'            # Gold
        }
        
        self.station_type_icons = {
            StationType.TERMINUS: 'star',
            StationType.INTERMEDIATE: 'circle',
            StationType.LOCAL: 'circle-dot',
            StationType.JUNCTION: 'diamond'
        }
        
        self.terrain_colors = {
            TerrainComplexity.FLAT: '#90EE90',     # Light Green
            TerrainComplexity.ROLLING: '#FFE135',  # Yellow
            TerrainComplexity.HILLY: '#FFA500',    # Orange
            TerrainComplexity.MOUNTAINOUS: '#FF6B6B' # Light Red
        }
    
    def create_comprehensive_visualization(self,
                                         routes: List[RouteVisualization],
                                         output_path: str = "railway_visualization.html",
                                         title: str = "BCPC Railway Route Analysis") -> None:
        """
        Create comprehensive interactive visualization
        
        Args:
            routes: List of route visualization data
            output_path: Output HTML file path
            title: Visualization title
        """
        logger.info(f"Creating comprehensive visualization for {len(routes)} routes")
        
        # Calculate map bounds
        all_points = []
        for route in routes:
            for segment in route.segments:
                all_points.extend([segment.start_point, segment.end_point])
            for station in route.stations:
                all_points.append(station.location)
        
        if all_points:
            lats = [p.y for p in all_points]
            lons = [p.x for p in all_points]
            center_lat = sum(lats) / len(lats)
            center_lon = sum(lons) / len(lons)
            map_center = (center_lat, center_lon)
        else:
            map_center = self.default_center
        
        # Create base map
        m = folium.Map(
            location=map_center,
            zoom_start=self.default_zoom,
            tiles='OpenStreetMap'
        )
        
        # Add tile layer options
        folium.TileLayer('Stamen Terrain').add_to(m)
        folium.TileLayer('CartoDB positron').add_to(m)
        
        # Add routes and stations
        for route in routes:
            self._add_route_to_map(m, route)
            self._add_stations_to_map(m, route)
        
        # Add legends and controls
        self._add_legends(m)
        self._add_layer_control(m)
        
        # Add summary information
        self._add_summary_panel(m, routes, title)
        
        # Save map
        m.save(output_path)
        logger.info(f"Visualization saved to {output_path}")
        
        # Create additional charts
        self._create_supplementary_charts(routes, output_path)
    
    def _add_route_to_map(self, map_obj: folium.Map, route: RouteVisualization) -> None:
        """Add a route with infrastructure color coding to the map"""
        
        # Create feature group for this route
        route_group = folium.FeatureGroup(name=f"Route: {route.route_name}")
        
        # Add route segments with infrastructure color coding
        for i, segment in enumerate(route.segments):
            coords = [
                [segment.start_point.y, segment.start_point.x],
                [segment.end_point.y, segment.end_point.x]
            ]
            
            color = self.infrastructure_colors[segment.infrastructure_type]
            
            # Create popup content
            popup_content = f"""
            <div style="width: 250px;">
                <h4>{route.route_name} - Segment {i+1}</h4>
                <b>Infrastructure:</b> {segment.infrastructure_type.value.title()}<br>
                <b>Terrain:</b> {segment.terrain_complexity.value.title()}<br>
                <b>Distance:</b> {segment.distance_km:.1f} km<br>
                <b>Elevation Change:</b> {segment.elevation_end - segment.elevation_start:.0f} m<br>
                <b>Cost per km:</b> €{segment.cost_per_km:,.0f}<br>
                <b>Train Type:</b> {route.train_type.value.replace('_', ' ').title()}
            </div>
            """
            
            # Line style based on infrastructure type
            if segment.infrastructure_type == InfrastructureType.TUNNEL:
                line_style = {'dash_array': '10,5'}
                weight = 6
            elif segment.infrastructure_type == InfrastructureType.BRIDGE:
                line_style = {'dash_array': '15,5,5,5'}
                weight = 5
            elif segment.infrastructure_type == InfrastructureType.PARTIAL_TUNNEL:
                line_style = {'dash_array': '8,3,3,3'}
                weight = 5
            else:
                line_style = {}
                weight = 4
            
            folium.PolyLine(
                coords,
                color=color,
                weight=weight,
                opacity=0.8,
                popup=folium.Popup(popup_content, max_width=300),
                tooltip=f"{segment.infrastructure_type.value.title()} - {segment.distance_km:.1f}km",
                **line_style
            ).add_to(route_group)
        
        # Add train type indicator at route midpoint
        if route.segments:
            mid_segment = route.segments[len(route.segments)//2]
            mid_point = [
                (mid_segment.start_point.y + mid_segment.end_point.y) / 2,
                (mid_segment.start_point.x + mid_segment.end_point.x) / 2
            ]
            
            train_icon = self._get_train_icon(route.train_type)
            
            folium.Marker(
                mid_point,
                icon=folium.Icon(
                    color='white',
                    icon_color=self.train_type_colors[route.train_type],
                    icon=train_icon,
                    prefix='fa'
                ),
                popup=f"""
                <div style="width: 200px;">
                    <h4>{route.route_name}</h4>
                    <b>Train Type:</b> {route.train_type.value.replace('_', ' ').title()}<br>
                    <b>Total Length:</b> {route.total_length_km:.1f} km<br>
                    <b>Daily Passengers:</b> {route.daily_passengers_total:,}<br>
                    <b>Total Cost:</b> €{route.total_cost:,.0f}
                </div>
                """,
                tooltip=f"{route.train_type.value.replace('_', ' ').title()} Service"
            ).add_to(route_group)
        
        route_group.add_to(map_obj)
    
    def _add_stations_to_map(self, map_obj: folium.Map, route: RouteVisualization) -> None:
        """Add stations to the map with detailed information"""
        
        station_group = folium.FeatureGroup(name=f"Stations: {route.route_name}")
        
        for station in route.stations:
            # Station marker color based on type
            if station.station_type == StationType.TERMINUS:
                color = 'red'
                icon = 'star'
            elif station.station_type == StationType.JUNCTION:
                color = 'purple'
                icon = 'exchange'
            elif station.station_type == StationType.INTERMEDIATE:
                color = 'blue'
                icon = 'train'
            else:  # LOCAL
                color = 'green'
                icon = 'circle'
            
            # Create detailed popup
            popup_content = f"""
            <div style="width: 300px;">
                <h3>{station.name}</h3>
                <table style="width:100%; font-size:12px;">
                    <tr><td><b>Type:</b></td><td>{station.station_type.value.title()}</td></tr>
                    <tr><td><b>Daily Passengers:</b></td><td>{station.daily_passengers:,}</td></tr>
                    <tr><td><b>Population Served:</b></td><td>{station.population_served:,}</td></tr>
                    <tr><td><b>Employment Served:</b></td><td>{station.employment_served:,}</td></tr>
                    <tr><td><b>Platforms:</b></td><td>{station.platforms}</td></tr>
                    <tr><td><b>Elevation:</b></td><td>{station.elevation:.0f} m</td></tr>
                    <tr><td><b>Estimated Cost:</b></td><td>€{station.estimated_cost:,.0f}</td></tr>
                    <tr><td><b>Access Modes:</b></td><td>{', '.join(station.access_modes)}</td></tr>
                </table>
                <div style="margin-top:10px;">
                    <div style="background-color:{color}; color:white; padding:3px; text-align:center; border-radius:3px;">
                        {station.station_type.value.upper()} STATION
                    </div>
                </div>
            </div>
            """
            
            # Station marker size based on passenger volume
            if station.daily_passengers > 20000:
                radius = 12
            elif station.daily_passengers > 10000:
                radius = 10
            elif station.daily_passengers > 5000:
                radius = 8
            else:
                radius = 6
            
            folium.CircleMarker(
                [station.location.y, station.location.x],
                radius=radius,
                popup=folium.Popup(popup_content, max_width=350),
                tooltip=f"{station.name} ({station.daily_passengers:,} daily passengers)",
                color='black',
                weight=2,
                fillColor=color,
                fillOpacity=0.8
            ).add_to(station_group)
            
            # Add station label
            folium.Marker(
                [station.location.y, station.location.x],
                icon=folium.DivIcon(
                    html=f'<div style="background-color:white; border:1px solid black; padding:2px; border-radius:3px; font-size:10px; font-weight:bold;">{station.name}</div>',
                    icon_size=(len(station.name)*6, 20),
                    icon_anchor=(len(station.name)*3, 25)
                )
            ).add_to(station_group)
        
        station_group.add_to(map_obj)
    
    def _add_legends(self, map_obj: folium.Map) -> None:
        """Add legends for infrastructure types, train types, and stations"""
        
        # Infrastructure legend
        infrastructure_legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: 180px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h4>Infrastructure Types</h4>
        '''
        
        for infra_type, color in self.infrastructure_colors.items():
            line_style = "solid"
            if infra_type == InfrastructureType.TUNNEL:
                line_style = "dashed"
            elif infra_type == InfrastructureType.BRIDGE:
                line_style = "dotted"
            
            infrastructure_legend_html += f'''
            <p><span style="color:{color}; font-weight:bold; text-decoration: underline; text-decoration-style: {line_style};">
            ━━━━</span> {infra_type.value.replace('_', ' ').title()}</p>
            '''
        
        infrastructure_legend_html += '</div>'
        map_obj.get_root().html.add_child(folium.Element(infrastructure_legend_html))
        
        # Train type legend
        train_legend_html = '''
        <div style="position: fixed; 
                    top: 200px; right: 10px; width: 200px; height: 140px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h4>Train Types</h4>
        '''
        
        for train_type, color in self.train_type_colors.items():
            icon = self._get_train_icon(train_type)
            train_legend_html += f'''
            <p><span style="color:{color}; font-weight:bold;">
            <i class="fa fa-{icon}"></i></span> {train_type.value.replace('_', ' ').title()}</p>
            '''
        
        train_legend_html += '</div>'
        map_obj.get_root().html.add_child(folium.Element(train_legend_html))
        
        # Station type legend
        station_legend_html = '''
        <div style="position: fixed; 
                    top: 350px; right: 10px; width: 200px; height: 120px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 10px">
        <h4>Station Types</h4>
        <p><span style="color:red;">●</span> Terminus</p>
        <p><span style="color:blue;">●</span> Intermediate</p>
        <p><span style="color:green;">●</span> Local</p>
        <p><span style="color:purple;">●</span> Junction</p>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(station_legend_html))
    
    def _add_layer_control(self, map_obj: folium.Map) -> None:
        """Add layer control to the map"""
        folium.LayerControl().add_to(map_obj)
    
    def _add_summary_panel(self, map_obj: folium.Map, routes: List[RouteVisualization], title: str) -> None:
        """Add summary information panel"""
        
        total_length = sum(route.total_length_km for route in routes)
        total_cost = sum(route.total_cost for route in routes)
        total_passengers = sum(route.daily_passengers_total for route in routes)
        total_stations = sum(len(route.stations) for route in routes)
        
        # Count infrastructure types
        infra_counts = {}
        for route in routes:
            for segment in route.segments:
                infra_type = segment.infrastructure_type.value
                infra_counts[infra_type] = infra_counts.get(infra_type, 0) + segment.distance_km
        
        summary_html = f'''
        <div style="position: fixed; 
                    top: 10px; left: 10px; width: 300px; 
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:12px; padding: 15px">
        <h3>{title}</h3>
        <table style="width:100%;">
            <tr><td><b>Total Network Length:</b></td><td>{total_length:.1f} km</td></tr>
            <tr><td><b>Total Investment:</b></td><td>€{total_cost:,.0f}</td></tr>
            <tr><td><b>Daily Passengers:</b></td><td>{total_passengers:,}</td></tr>
            <tr><td><b>Total Stations:</b></td><td>{total_stations}</td></tr>
            <tr><td><b>Number of Routes:</b></td><td>{len(routes)}</td></tr>
        </table>
        
        <h4>Infrastructure Breakdown:</h4>
        <table style="width:100%; font-size:11px;">
        '''
        
        for infra_type, length in infra_counts.items():
            percentage = (length / total_length * 100) if total_length > 0 else 0
            summary_html += f'<tr><td>{infra_type.replace("_", " ").title()}:</td><td>{length:.1f} km ({percentage:.1f}%)</td></tr>'
        
        summary_html += '''
        </table>
        </div>
        '''
        
        map_obj.get_root().html.add_child(folium.Element(summary_html))
    
    def _get_train_icon(self, train_type: TrainType) -> str:
        """Get appropriate icon for train type"""
        train_icons = {
            TrainType.DIESEL: 'truck',
            TrainType.ELECTRIC_EMU: 'flash',
            TrainType.ELECTRIC_LOCOMOTIVE: 'train',
            TrainType.HYBRID: 'leaf'
        }
        return train_icons.get(train_type, 'train')
    
    def _create_supplementary_charts(self, routes: List[RouteVisualization], base_output_path: str) -> None:
        """Create additional charts and save as separate HTML files"""
        
        # Extract data for charts
        route_names = [route.route_name for route in routes]
        route_lengths = [route.total_length_km for route in routes]
        route_costs = [route.total_cost / 1_000_000 for route in routes]  # Convert to millions
        route_passengers = [route.daily_passengers_total for route in routes]
        
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Route Lengths (km)',
                'Investment Costs (€ Millions)',
                'Daily Passengers by Route',
                'Infrastructure Distribution'
            ),
            specs=[[{"type": "bar"}, {"type": "bar"}],
                   [{"type": "bar"}, {"type": "pie"}]]
        )
        
        # Route lengths
        fig.add_trace(
            go.Bar(x=route_names, y=route_lengths, name="Length (km)", marker_color='lightblue'),
            row=1, col=1
        )
        
        # Investment costs
        fig.add_trace(
            go.Bar(x=route_names, y=route_costs, name="Cost (€M)", marker_color='lightcoral'),
            row=1, col=2
        )
        
        # Daily passengers
        fig.add_trace(
            go.Bar(x=route_names, y=route_passengers, name="Passengers", marker_color='lightgreen'),
            row=2, col=1
        )
        
        # Infrastructure distribution
        infra_totals = {}
        for route in routes:
            for segment in route.segments:
                infra_type = segment.infrastructure_type.value.replace('_', ' ').title()
                infra_totals[infra_type] = infra_totals.get(infra_type, 0) + segment.distance_km
        
        fig.add_trace(
            go.Pie(labels=list(infra_totals.keys()), values=list(infra_totals.values()), name="Infrastructure"),
            row=2, col=2
        )
        
        fig.update_layout(
            title_text="BCPC Railway Network Analysis",
            showlegend=False,
            height=800
        )
        
        # Save charts
        chart_path = base_output_path.replace('.html', '_charts.html')
        fig.write_html(chart_path)
        logger.info(f"Supplementary charts saved to {chart_path}")
    
    def create_route_from_bcpc_data(self,
                                  route_line: LineString,
                                  terrain_analysis: Any,
                                  station_network: Any,
                                  cost_summary: Any,
                                  network_design: Any,
                                  route_name: str = "Railway Route") -> RouteVisualization:
        """
        Create RouteVisualization from BCPC analysis results
        
        Args:
            route_line: Route geometry
            terrain_analysis: TerrainAnalysis object
            station_network: CityStationNetwork object
            cost_summary: CostSummary object
            network_design: NetworkDesign object
            route_name: Name for the route
            
        Returns:
            RouteVisualization object ready for mapping
        """
        
        # Convert terrain segments to visualization segments
        segments = []
        
        if hasattr(terrain_analysis, 'terrain_segments'):
            for terrain_seg in terrain_analysis.terrain_segments:
                # Determine infrastructure type based on terrain segment
                if terrain_seg.requires_tunnel and terrain_seg.tunnel_length_km > terrain_seg.bridge_length_km:
                    if terrain_seg.tunnel_length_km >= (terrain_seg.end_km - terrain_seg.start_km) * 0.8:
                        infra_type = InfrastructureType.TUNNEL
                    else:
                        infra_type = InfrastructureType.PARTIAL_TUNNEL
                elif terrain_seg.requires_bridge:
                    if terrain_seg.bridge_length_km > 2.0:  # Major bridge
                        infra_type = InfrastructureType.BRIDGE
                    else:
                        infra_type = InfrastructureType.ELEVATED
                else:
                    infra_type = InfrastructureType.GROUND
                
                # Calculate segment points
                segment_length = terrain_seg.end_km - terrain_seg.start_km
                start_distance = terrain_seg.start_km * 1000  # Convert to meters
                end_distance = terrain_seg.end_km * 1000
                
                start_point = Point(route_line.interpolate(start_distance).coords[0])
                end_point = Point(route_line.interpolate(end_distance).coords[0])
                
                # Estimate elevation (simplified)
                if hasattr(terrain_analysis, 'elevation_profile'):
                    profile = terrain_analysis.elevation_profile
                    start_elev = np.interp(terrain_seg.start_km, profile.distances, profile.elevations)
                    end_elev = np.interp(terrain_seg.end_km, profile.distances, profile.elevations)
                else:
                    start_elev = end_elev = 100.0  # Default elevation
                
                # Estimate cost per km
                base_cost_per_km = 2_500_000  # Base cost from cost_analysis
                if hasattr(cost_summary, 'cost_per_route_km'):
                    cost_per_km = cost_summary.cost_per_route_km
                else:
                    cost_per_km = base_cost_per_km * getattr(terrain_analysis, 'cost_multiplier', 1.0)
                
                segments.append(RouteSegmentVisualization(
                    start_point=start_point,
                    end_point=end_point,
                    infrastructure_type=infra_type,
                    terrain_complexity=terrain_seg.complexity,
                    distance_km=segment_length,
                    cost_per_km=cost_per_km,
                    elevation_start=start_elev,
                    elevation_end=end_elev
                ))
        
        # Convert stations to visualization format
        stations = []
        if hasattr(station_network, 'stations'):
            for station in station_network.stations:
                # Get elevation for station
                if hasattr(terrain_analysis, 'elevation_profile'):
                    station_distance_km = route_line.project(station.location) / 1000
                    profile = terrain_analysis.elevation_profile
                    elevation = np.interp(station_distance_km, profile.distances, profile.elevations)
                else:
                    elevation = 100.0
                
                stations.append(StationVisualization(
                    location=station.location,
                    name=station.name,
                    station_type=station.station_type,
                    daily_passengers=station.daily_passengers_estimate,
                    population_served=station.population_served,
                    employment_served=station.employment_served,
                    platforms=station.platform_count,
                    estimated_cost=station.estimated_cost,
                    elevation=elevation,
                    access_modes=getattr(station, 'access_modes', ['walking'])
                ))
        
        # Extract route totals
        total_length = route_line.length / 1000  # Convert to km
        total_cost = getattr(cost_summary, 'total_capex', 0)
        overall_terrain = getattr(terrain_analysis, 'overall_complexity', TerrainComplexity.FLAT)
        train_type = getattr(network_design, 'train_type', TrainType.ELECTRIC_EMU)
        daily_passengers_total = sum(s.daily_passengers for s in stations)
        
        return RouteVisualization(
            route_id=f"route_{route_name.lower().replace(' ', '_')}",
            route_name=route_name,
            segments=segments,
            stations=stations,
            train_type=train_type,
            total_length_km=total_length,
            total_cost=total_cost,
            overall_terrain=overall_terrain,
            daily_passengers_total=daily_passengers_total
        )

# Utility functions for integration

def create_visualization_from_scenario(scenario_results: Dict[str, Any],
                                     output_path: str = "scenario_visualization.html") -> None:
    """
    Create visualization from complete BCPC scenario results
    
    Args:
        scenario_results: Dictionary containing all analysis results
        output_path: Output HTML file path
    """
    
    visualizer = BCPCVisualizer()
    routes = []
    
def create_visualization_from_scenario(scenario_results: Dict[str, Any],
                                     output_path: str = "scenario_visualization.html") -> None:
    """
    Create visualization from complete BCPC scenario results
    
    Args:
        scenario_results: Dictionary containing all analysis results
        output_path: Output HTML file path
    """
    
    visualizer = BCPCVisualizer()
    routes = []
    
    # Process each city/route in the scenario
    for city_name, city_results in scenario_results.items():
        if isinstance(city_results, dict) and 'route_line' in city_results:
            route_viz = visualizer.create_route_from_bcpc_data(
                route_line=city_results['route_line'],
                terrain_analysis=city_results.get('terrain_analysis'),
                station_network=city_results.get('station_network'),
                cost_summary=city_results.get('cost_summary'),
                network_design=city_results.get('network_design'),
                route_name=city_name
            )
            routes.append(route_viz)
    
    # Create comprehensive visualization
    visualizer.create_comprehensive_visualization(
        routes=routes,
        output_path=output_path,
        title=f"BCPC Railway Analysis - {len(routes)} Routes"
    )

def create_elevation_profile_chart(terrain_analysis: Any, 
                                 route_name: str = "Route",
                                 output_path: str = "elevation_profile.html") -> None:
    """Create detailed elevation profile chart"""
    
    if not hasattr(terrain_analysis, 'elevation_profile'):
        logger.warning("No elevation profile data available")
        return
    
    profile = terrain_analysis.elevation_profile
    
    # Create elevation profile with infrastructure overlays
    fig = go.Figure()
    
    # Add elevation line
    fig.add_trace(go.Scatter(
        x=profile.distances,
        y=profile.elevations,
        mode='lines',
        name='Elevation',
        line=dict(color='blue', width=2),
        fill='tonexty',
        fillcolor='rgba(0,100,255,0.1)'
    ))
    
    # Add terrain complexity background
    if hasattr(terrain_analysis, 'terrain_segments'):
        for segment in terrain_analysis.terrain_segments:
            color_map = {
                'flat': 'rgba(144,238,144,0.3)',
                'rolling': 'rgba(255,225,53,0.3)',
                'hilly': 'rgba(255,165,0,0.3)',
                'mountainous': 'rgba(255,107,107,0.3)'
            }
            
            color = color_map.get(segment.complexity.value, 'rgba(128,128,128,0.3)')
            
            fig.add_vrect(
                x0=segment.start_km,
                x1=segment.end_km,
                fillcolor=color,
                opacity=0.3,
                layer="below",
                line_width=0,
            )
    
    # Add slope line on secondary y-axis
    fig.add_trace(go.Scatter(
        x=profile.distances[:-1],
        y=profile.slopes * 100,
        mode='lines',
        name='Slope (%)',
        line=dict(color='red', width=1),
        yaxis='y2'
    ))
    
    # Add grade limit lines
    max_grade = 2.5  # 2.5% maximum grade
    fig.add_hline(y=max_grade, line_dash="dash", line_color="red", 
                  annotation_text="Max Grade (2.5%)", yref='y2')
    fig.add_hline(y=-max_grade, line_dash="dash", line_color="red", yref='y2')
    
    # Update layout
    fig.update_layout(
        title=f'Elevation Profile - {route_name}',
        xaxis_title='Distance (km)',
        yaxis_title='Elevation (m)',
        yaxis2=dict(
            title='Slope (%)',
            overlaying='y',
            side='right',
            range=[-10, 10]
        ),
        hovermode='x unified',
        height=600
    )
    
    fig.write_html(output_path)
    logger.info(f"Elevation profile chart saved to {output_path}")

def create_cost_breakdown_chart(cost_summary: Any,
                              route_name: str = "Route",
                              output_path: str = "cost_breakdown.html") -> None:
    """Create detailed cost breakdown visualization"""
    
    # Create subplots for different cost views
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Capital Expenditure Breakdown',
            'Annual Operating Costs',
            'Cost per Kilometer',
            'Lifecycle Costs (20 years)'
        ),
        specs=[[{"type": "pie"}, {"type": "bar"}],
               [{"type": "bar"}, {"type": "waterfall"}]]
    )
    
    # CAPEX breakdown
    if hasattr(cost_summary, 'infrastructure') and hasattr(cost_summary, 'rolling_stock'):
        capex_labels = ['Track Construction', 'Electrification', 'Signaling', 'Stations', 'Rolling Stock']
        capex_values = [
            cost_summary.infrastructure.track_construction,
            cost_summary.infrastructure.electrification,
            cost_summary.infrastructure.signaling,
            cost_summary.infrastructure.stations,
            cost_summary.rolling_stock.total
        ]
        
        fig.add_trace(
            go.Pie(labels=capex_labels, values=capex_values, name="CAPEX"),
            row=1, col=1
        )
    
    # OPEX breakdown
    if hasattr(cost_summary, 'operational_annual'):
        opex_labels = ['Track Maintenance', 'Rolling Stock Maintenance', 'Energy', 'Staff']
        opex_values = [
            cost_summary.operational_annual.track_maintenance,
            cost_summary.operational_annual.rolling_stock_maintenance,
            cost_summary.operational_annual.energy,
            cost_summary.operational_annual.staff
        ]
        
        fig.add_trace(
            go.Bar(x=opex_labels, y=opex_values, name="Annual OPEX", marker_color='lightcoral'),
            row=1, col=2
        )
    
    # Cost per km
    if hasattr(cost_summary, 'cost_per_route_km'):
        cost_categories = ['Infrastructure', 'Rolling Stock', 'Total']
        cost_per_km = [
            cost_summary.infrastructure.total / (cost_summary.cost_per_route_km / cost_summary.total_capex) if hasattr(cost_summary, 'total_capex') else 0,
            cost_summary.rolling_stock.total / (cost_summary.cost_per_route_km / cost_summary.total_capex) if hasattr(cost_summary, 'total_capex') else 0,
            cost_summary.cost_per_route_km
        ]
        
        fig.add_trace(
            go.Bar(x=cost_categories, y=cost_per_km, name="Cost per km", marker_color='lightblue'),
            row=2, col=1
        )
    
    # Lifecycle costs waterfall
    if hasattr(cost_summary, 'total_capex') and hasattr(cost_summary, 'lifecycle_cost_20_years'):
        waterfall_data = [
            go.Waterfall(
                name="Lifecycle Costs",
                orientation="v",
                measure=["absolute", "relative", "total"],
                x=["Initial CAPEX", "20-Year OPEX", "Total Lifecycle"],
                textposition="outside",
                text=[f"€{cost_summary.total_capex/1e6:.1f}M", 
                      f"€{(cost_summary.lifecycle_cost_20_years - cost_summary.total_capex)/1e6:.1f}M",
                      f"€{cost_summary.lifecycle_cost_20_years/1e6:.1f}M"],
                y=[cost_summary.total_capex/1e6, 
                   (cost_summary.lifecycle_cost_20_years - cost_summary.total_capex)/1e6,
                   cost_summary.lifecycle_cost_20_years/1e6],
                connector={"line":{"color":"rgb(63, 63, 63)"}},
            )
        ]
        
        fig.add_trace(waterfall_data[0], row=2, col=2)
    
    fig.update_layout(
        title_text=f"Cost Analysis - {route_name}",
        showlegend=False,
        height=800
    )
    
    fig.write_html(output_path)
    logger.info(f"Cost breakdown chart saved to {output_path}")

def create_demand_analysis_chart(station_network: Any,
                                route_name: str = "Route",
                                output_path: str = "demand_analysis.html") -> None:
    """Create demand analysis visualization"""
    
    if not hasattr(station_network, 'stations'):
        logger.warning("No station data available for demand analysis")
        return
    
    stations = station_network.stations
    
    # Extract station data
    station_names = [s.name for s in stations]
    daily_passengers = [s.daily_passengers_estimate for s in stations]
    population_served = [s.population_served for s in stations]
    employment_served = [s.employment_served for s in stations]
    
    # Create subplots
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Daily Passengers by Station',
            'Population vs Employment Served',
            'Station Catchment Analysis',
            'Passenger Load Distribution'
        ),
        specs=[[{"type": "bar"}, {"type": "scatter"}],
               [{"type": "bar"}, {"type": "pie"}]]
    )
    
    # Daily passengers bar chart
    fig.add_trace(
        go.Bar(x=station_names, y=daily_passengers, name="Daily Passengers", 
               marker_color='lightgreen'),
        row=1, col=1
    )
    
    # Population vs Employment scatter
    fig.add_trace(
        go.Scatter(
            x=population_served,
            y=employment_served,
            mode='markers+text',
            text=station_names,
            textposition="top center",
            name="Stations",
            marker=dict(size=10, color='blue')
        ),
        row=1, col=2
    )
    
    # Catchment analysis
    total_population = [p + e for p, e in zip(population_served, employment_served)]
    fig.add_trace(
        go.Bar(x=station_names, y=total_population, name="Total Catchment",
               marker_color='orange'),
        row=2, col=1
    )
    
    # Passenger distribution pie
    fig.add_trace(
        go.Pie(labels=station_names, values=daily_passengers, name="Passenger Share"),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text=f"Demand Analysis - {route_name}",
        showlegend=False,
        height=800
    )
    
    # Update axis labels
    fig.update_xaxes(title_text="Population Served", row=1, col=2)
    fig.update_yaxes(title_text="Employment Served", row=1, col=2)
    
    fig.write_html(output_path)
    logger.info(f"Demand analysis chart saved to {output_path}")

def create_complete_dashboard(scenario_results: Dict[str, Any],
                            output_dir: str = "dashboard") -> None:
    """
    Create a complete dashboard with all visualizations
    
    Args:
        scenario_results: Complete BCPC analysis results
        output_dir: Output directory for dashboard files
    """
    
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    logger.info(f"Creating complete dashboard in {output_dir}")
    
    # Main map visualization
    create_visualization_from_scenario(
        scenario_results,
        output_path / "main_map.html"
    )
    
    # Individual analysis charts for each route
    for city_name, city_results in scenario_results.items():
        if isinstance(city_results, dict):
            # Elevation profile
            if 'terrain_analysis' in city_results:
                create_elevation_profile_chart(
                    city_results['terrain_analysis'],
                    city_name,
                    output_path / f"{city_name}_elevation.html"
                )
            
            # Cost breakdown
            if 'cost_summary' in city_results:
                create_cost_breakdown_chart(
                    city_results['cost_summary'],
                    city_name,
                    output_path / f"{city_name}_costs.html"
                )
            
            # Demand analysis
            if 'station_network' in city_results:
                create_demand_analysis_chart(
                    city_results['station_network'],
                    city_name,
                    output_path / f"{city_name}_demand.html"
                )
    
    # Create dashboard index
    create_dashboard_index(scenario_results, output_path)

def create_dashboard_index(scenario_results: Dict[str, Any], output_dir: Path) -> None:
    """Create an index.html file that links all dashboard components"""
    
    # Calculate summary statistics
    total_routes = len([k for k, v in scenario_results.items() if isinstance(v, dict)])
    total_length = sum(
        getattr(v.get('terrain_analysis', {}), 'elevation_profile', {}).get('total_length_km', 0)
        for v in scenario_results.values() if isinstance(v, dict)
    )
    total_cost = sum(
        getattr(v.get('cost_summary', {}), 'total_capex', 0)
        for v in scenario_results.values() if isinstance(v, dict)
    )
    
    index_html = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>BCPC Railway Analysis Dashboard</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }}
            .header {{
                text-align: center;
                margin-bottom: 40px;
                padding-bottom: 20px;
                border-bottom: 2px solid #e0e0e0;
            }}
            .summary-stats {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .stat-card {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 8px;
                text-align: center;
            }}
            .stat-value {{
                font-size: 2em;
                font-weight: bold;
                margin-bottom: 5px;
            }}
            .stat-label {{
                font-size: 0.9em;
                opacity: 0.9;
            }}
            .route-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(350px, 1fr));
                gap: 20px;
                margin-bottom: 40px;
            }}
            .route-card {{
                border: 1px solid #ddd;
                border-radius: 8px;
                padding: 20px;
                background-color: #f9f9f9;
            }}
            .route-title {{
                font-size: 1.2em;
                font-weight: bold;
                margin-bottom: 15px;
                color: #333;
            }}
            .route-links {{
                display: flex;
                flex-wrap: wrap;
                gap: 10px;
            }}
            .route-link {{
                display: inline-block;
                padding: 8px 15px;
                background-color: #007bff;
                color: white;
                text-decoration: none;
                border-radius: 5px;
                font-size: 0.9em;
                transition: background-color 0.3s;
            }}
            .route-link:hover {{
                background-color: #0056b3;
            }}
            .main-map-section {{
                text-align: center;
                margin: 40px 0;
            }}
            .main-map-button {{
                display: inline-block;
                padding: 15px 30px;
                background: linear-gradient(135deg, #ff6b6b 0%, #ee5a24 100%);
                color: white;
                text-decoration: none;
                border-radius: 25px;
                font-size: 1.1em;
                font-weight: bold;
                transition: transform 0.3s;
            }}
            .main-map-button:hover {{
                transform: translateY(-2px);
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🚆 BCPC Railway Analysis Dashboard</h1>
                <p>Comprehensive analysis results for railway corridor planning</p>
            </div>
            
            <div class="summary-stats">
                <div class="stat-card">
                    <div class="stat-value">{total_routes}</div>
                    <div class="stat-label">Routes Analyzed</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">{total_length:.0f} km</div>
                    <div class="stat-label">Total Network Length</div>
                </div>
                <div class="stat-card">
                    <div class="stat-value">€{total_cost/1e9:.1f}B</div>
                    <div class="stat-label">Total Investment</div>
                </div>
            </div>
            
            <div class="main-map-section">
                <a href="main_map.html" class="main-map-button">
                    🗺️ View Interactive Route Map
                </a>
            </div>
            
            <h2>📊 Detailed Analysis by Route</h2>
            <div class="route-grid">
    """
    
    # Add route cards
    for city_name, city_results in scenario_results.items():
        if isinstance(city_results, dict):
            index_html += f"""
                <div class="route-card">
                    <div class="route-title">{city_name}</div>
                    <div class="route-links">
            """
            
            # Add available analysis links
            if 'terrain_analysis' in city_results:
                index_html += f'<a href="{city_name}_elevation.html" class="route-link">📈 Elevation Profile</a>'
            
            if 'cost_summary' in city_results:
                index_html += f'<a href="{city_name}_costs.html" class="route-link">💰 Cost Analysis</a>'
            
            if 'station_network' in city_results:
                index_html += f'<a href="{city_name}_demand.html" class="route-link">👥 Demand Analysis</a>'
            
            index_html += """
                    </div>
                </div>
            """
    
    index_html += """
            </div>
            
            <div style="text-align: center; margin-top: 40px; padding-top: 20px; border-top: 1px solid #e0e0e0;">
                <p><em>Generated by BCPC Pipeline - Bringing Cities Back to the People</em></p>
            </div>
        </div>
    </body>
    </html>
    """
    
    with open(output_dir / "index.html", 'w', encoding='utf-8') as f:
        f.write(index_html)
    
    logger.info(f"Dashboard index created at {output_dir / 'index.html'}")

# Example usage and testing
if __name__ == "__main__":
    # Create example visualization data
    from shapely.geometry import LineString, Point
    
    # Example route
    example_route = LineString([
        (4.3517, 50.8466),  # Brussels
        (4.4000, 50.9000),  # Intermediate
        (4.4024, 51.2194)   # Antwerp
    ])
    
    # Example stations
    example_stations = [
        StationVisualization(
            location=Point(4.3517, 50.8466),
            name="Brussels Central",
            station_type=StationType.TERMINUS,
            daily_passengers=25000,
            population_served=180000,
            employment_served=150000,
            platforms=4,
            estimated_cost=15000000,
            elevation=56,
            access_modes=['walking', 'metro', 'bus', 'car']
        ),
        StationVisualization(
            location=Point(4.4000, 50.9000),
            name="Intermediate Station",
            station_type=StationType.INTERMEDIATE,
            daily_passengers=8000,
            population_served=45000,
            employment_served=25000,
            platforms=2,
            estimated_cost=3000000,
            elevation=45,
            access_modes=['walking', 'cycling', 'feeder_bus']
        ),
        StationVisualization(
            location=Point(4.4024, 51.2194),
            name="Antwerp Central",
            station_type=StationType.TERMINUS,
            daily_passengers=30000,
            population_served=220000,
            employment_served=180000,
            platforms=6,
            estimated_cost=20000000,
            elevation=12,
            access_modes=['walking', 'metro', 'bus', 'car', 'cycling']
        )
    ]
    
    # Example segments
    example_segments = [
        RouteSegmentVisualization(
            start_point=Point(4.3517, 50.8466),
            end_point=Point(4.3750, 50.8700),
            infrastructure_type=InfrastructureType.TUNNEL,
            terrain_complexity=TerrainComplexity.URBAN,
            distance_km=5.0,
            cost_per_km=15000000,
            elevation_start=56,
            elevation_end=45
        ),
        RouteSegmentVisualization(
            start_point=Point(4.3750, 50.8700),
            end_point=Point(4.4000, 50.9000),
            infrastructure_type=InfrastructureType.GROUND,
            terrain_complexity=TerrainComplexity.ROLLING,
            distance_km=8.0,
            cost_per_km=4000000,
            elevation_start=45,
            elevation_end=40
        ),
        RouteSegmentVisualization(
            start_point=Point(4.4000, 50.9000),
            end_point=Point(4.4024, 51.2194),
            infrastructure_type=InfrastructureType.BRIDGE,
            terrain_complexity=TerrainComplexity.FLAT,
            distance_km=12.0,
            cost_per_km=8000000,
            elevation_start=40,
            elevation_end=12
        )
    ]
    
    # Create example route visualization
    example_route_viz = RouteVisualization(
        route_id="brussels_antwerp",
        route_name="Brussels-Antwerp Railway",
        segments=example_segments,
        stations=example_stations,
        train_type=TrainType.ELECTRIC_EMU,
        total_length_km=25.0,
        total_cost=500000000,
        overall_terrain=TerrainComplexity.ROLLING,
        daily_passengers_total=63000
    )
    
    # Create visualizer and generate map
    visualizer = BCPCVisualizer()
    
    logger.info("Creating example visualization...")
    visualizer.create_comprehensive_visualization(
        routes=[example_route_viz],
        output_path="example_railway_visualization.html",
        title="Example BCPC Railway Analysis"
    )
    
    logger.info("Example visualization completed!")
    logger.info("Open 'example_railway_visualization.html' in a web browser to view the results.")