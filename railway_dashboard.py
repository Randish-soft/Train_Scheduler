"""
Railway Analysis Dashboard
Interactive visualization dashboard for railway network analysis and planning
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import json
import os
import numpy as np
from datetime import datetime
import requests
from typing import Dict, List, Any, Tuple
import folium
from streamlit_folium import st_folium
import networkx as nx
from geopy.distance import geodesic

# Page configuration
st.set_page_config(
    page_title="Railway Network Dashboard",
    page_icon="🚄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
        font-size: 1.2rem;
    }
</style>
""", unsafe_allow_html=True)

class RailwayDataFetcher:
    """Fetches existing railway data from OpenStreetMap and other sources"""
    
    def __init__(self):
        self.overpass_url = "https://overpass-api.de/api/interpreter"
        
    def fetch_existing_railways(self, country: str, bbox: Tuple[float, float, float, float] = None) -> Dict[str, Any]:
        """Fetch existing railway infrastructure from OpenStreetMap"""
        try:
            # If no bbox provided, use country bounds
            if bbox is None:
                bbox = self._get_country_bounds(country)
            
            # Overpass QL query for railways
            query = f"""
            [out:json][timeout:60];
            (
              way["railway"="rail"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
              way["railway"="light_rail"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
              way["railway"="subway"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
              way["railway"="tram"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
              node["railway"="station"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
              node["railway"="halt"]({bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]});
            );
            out body;
            >;
            out skel qt;
            """
            
            response = requests.post(
                self.overpass_url,
                data={'data': query},
                timeout=60
            )
            
            if response.status_code == 200:
                data = response.json()
                return self._process_railway_data(data)
            else:
                st.warning(f"Failed to fetch railway data: {response.status_code}")
                return {"rails": [], "stations": [], "stats": {}}
                
        except Exception as e:
            st.error(f"Error fetching railway data: {str(e)}")
            return {"rails": [], "stations": [], "stats": {}}
    
    def _get_country_bounds(self, country: str) -> Tuple[float, float, float, float]:
        """Get bounding box for a country"""
        try:
            # Use Nominatim to get country bounds
            url = f"https://nominatim.openstreetmap.org/search"
            params = {
                'country': country,
                'format': 'json',
                'limit': 1
            }
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200 and response.json():
                data = response.json()[0]
                bbox = data.get('boundingbox', [])
                if len(bbox) == 4:
                    return (float(bbox[0]), float(bbox[2]), float(bbox[1]), float(bbox[3]))
        except:
            pass
        
        # Default to a reasonable area if country bounds not found
        return (45.0, -5.0, 55.0, 15.0)  # Rough Europe bounds
    
    def _process_railway_data(self, data: Dict) -> Dict[str, Any]:
        """Process raw OSM data into structured format"""
        nodes = {n['id']: n for n in data.get('elements', []) if n['type'] == 'node'}
        ways = [w for w in data.get('elements', []) if w['type'] == 'way']
        
        rails = []
        stations = []
        
        # Process railway lines
        for way in ways:
            if 'tags' in way and 'railway' in way['tags']:
                coords = []
                for node_id in way.get('nodes', []):
                    if node_id in nodes:
                        node = nodes[node_id]
                        coords.append([node['lat'], node['lon']])
                
                if coords:
                    rails.append({
                        'id': way['id'],
                        'type': way['tags']['railway'],
                        'name': way['tags'].get('name', 'Unnamed'),
                        'electrified': way['tags'].get('electrified', 'no'),
                        'tracks': int(way['tags'].get('tracks', 1)),
                        'coordinates': coords
                    })
        
        # Process stations
        for node_id, node in nodes.items():
            if 'tags' in node and 'railway' in node['tags']:
                if node['tags']['railway'] in ['station', 'halt']:
                    stations.append({
                        'id': node_id,
                        'name': node['tags'].get('name', 'Unnamed Station'),
                        'type': node['tags']['railway'],
                        'lat': node['lat'],
                        'lon': node['lon']
                    })
        
        # Calculate statistics
        stats = {
            'total_rails': len(rails),
            'total_stations': len(stations),
            'electrified_km': sum(1 for r in rails if r['electrified'] == 'yes'),
            'rail_types': pd.DataFrame(rails)['type'].value_counts().to_dict() if rails else {}
        }
        
        return {
            'rails': rails,
            'stations': stations,
            'stats': stats
        }

class RailwayDashboard:
    """Main dashboard class"""
    
    def __init__(self):
        self.data_fetcher = RailwayDataFetcher()
        self.initialize_session_state()
        
    def initialize_session_state(self):
        """Initialize session state variables"""
        if 'analysis_data' not in st.session_state:
            st.session_state.analysis_data = None
        if 'railway_data' not in st.session_state:
            st.session_state.railway_data = None
        if 'country' not in st.session_state:
            st.session_state.country = "Belgium"
    
    def load_analysis_data(self):
        """Load analysis data from pipeline output"""
        try:
            # Try to load from multiple possible locations
            possible_paths = [
                'output/detailed_analysis.json',
                'pipeline/output/detailed_analysis.json',
                '../output/detailed_analysis.json',
                './detailed_analysis.json'
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        st.session_state.analysis_data = json.load(f)
                    return True
            
            st.warning("No analysis data found. Run the pipeline first to generate data.")
            return False
        except Exception as e:
            st.error(f"Error loading analysis data: {str(e)}")
            return False
    
    def create_railway_map(self, cities: List[Dict], railway_data: Dict = None):
        """Create interactive map with existing railways and proposed routes"""
        if not cities:
            return None
            
        # Calculate center of map
        avg_lat = np.mean([c['latitude'] for c in cities])
        avg_lon = np.mean([c['longitude'] for c in cities])
        
        # Create folium map
        m = folium.Map(
            location=[avg_lat, avg_lon],
            zoom_start=7,
            tiles='OpenStreetMap'
        )
        
        # Add existing railway lines if available
        if railway_data and railway_data.get('rails'):
            for rail in railway_data['rails']:
                color = {
                    'rail': '#ff0000',
                    'light_rail': '#ff9900',
                    'subway': '#0066cc',
                    'tram': '#00cc00'
                }.get(rail['type'], '#666666')
                
                folium.PolyLine(
                    locations=rail['coordinates'],
                    color=color,
                    weight=3,
                    opacity=0.8,
                    popup=f"{rail['name']} ({rail['type']})"
                ).add_to(m)
        
        # Add existing stations
        if railway_data and railway_data.get('stations'):
            for station in railway_data['stations']:
                folium.CircleMarker(
                    location=[station['lat'], station['lon']],
                    radius=5,
                    popup=station['name'],
                    color='#333333',
                    fill=True,
                    fillColor='#ffffff'
                ).add_to(m)
        
        # Add cities with population-based sizing
        max_pop = max([c.get('population', 100000) for c in cities])
        for city in cities:
            pop = city.get('population', 100000)
            radius = 5 + (pop / max_pop) * 20
            
            folium.CircleMarker(
                location=[city['latitude'], city['longitude']],
                radius=radius,
                popup=f"{city.get('city_name', city.get('city', 'Unknown'))}<br>Population: {pop:,}",
                color='#1f77b4',
                fill=True,
                fillColor='#1f77b4',
                fillOpacity=0.6
            ).add_to(m)
        
        # Add proposed routes (connecting cities)
        if len(cities) > 1:
            # Create a simple network connecting major cities
            G = nx.Graph()
            for i, city in enumerate(cities):
                G.add_node(i, pos=(city['latitude'], city['longitude']))
            
            # Connect cities based on distance
            for i in range(len(cities)):
                for j in range(i + 1, len(cities)):
                    dist = geodesic(
                        (cities[i]['latitude'], cities[i]['longitude']),
                        (cities[j]['latitude'], cities[j]['longitude'])
                    ).km
                    if dist < 200:  # Connect cities within 200km
                        G.add_edge(i, j, weight=dist)
            
            # Draw proposed routes
            for edge in G.edges():
                city1, city2 = cities[edge[0]], cities[edge[1]]
                folium.PolyLine(
                    locations=[
                        [city1['latitude'], city1['longitude']],
                        [city2['latitude'], city2['longitude']]
                    ],
                    color='#00ff00',
                    weight=2,
                    opacity=0.6,
                    dash_array='10',
                    popup=f"Proposed route: {city1.get('city_name', 'City')} - {city2.get('city_name', 'City')}"
                ).add_to(m)
        
        # Add legend
        legend_html = '''
        <div style="position: fixed; 
                    top: 10px; right: 10px; width: 200px; height: auto;
                    background-color: white; border:2px solid grey; z-index:9999; 
                    font-size:14px; padding: 10px">
        <p style="margin: 0;"><b>Legend</b></p>
        <p style="margin: 5px;"><span style="color: #ff0000;">━━</span> Existing Rail</p>
        <p style="margin: 5px;"><span style="color: #ff9900;">━━</span> Light Rail</p>
        <p style="margin: 5px;"><span style="color: #0066cc;">━━</span> Subway</p>
        <p style="margin: 5px;"><span style="color: #00cc00;">━━</span> Tram</p>
        <p style="margin: 5px;"><span style="color: #00ff00;">┅┅</span> Proposed Route</p>
        <p style="margin: 5px;">● Cities (size = population)</p>
        </div>
        '''
        m.get_root().html.add_child(folium.Element(legend_html))
        
        return m
    
    def create_network_graph(self, cities: List[Dict]):
        """Create network graph visualization"""
        if not cities or len(cities) < 2:
            return None, []
            
        # Create graph
        G = nx.Graph()
        
        # Add nodes
        for i, city in enumerate(cities):
            G.add_node(
                i,
                name=city.get('city_name', city.get('city', f'City {i}')),
                population=city.get('population', 100000)
            )
        
        # Add edges based on distance
        edges = []
        for i in range(len(cities)):
            for j in range(i + 1, len(cities)):
                dist = geodesic(
                    (cities[i]['latitude'], cities[i]['longitude']),
                    (cities[j]['latitude'], cities[j]['longitude'])
                ).km
                if dist < 300:  # Connect cities within 300km
                    G.add_edge(i, j, weight=dist)
                    edges.append({
                        'from': cities[i].get('city_name', f'City {i}'),
                        'to': cities[j].get('city_name', f'City {j}'),
                        'distance': dist
                    })
        
        # Create plotly figure
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=2, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        node_trace = go.Scatter(
            x=[pos[node][0] for node in G.nodes()],
            y=[pos[node][1] for node in G.nodes()],
            mode='markers+text',
            hoverinfo='text',
            text=[G.nodes[node]['name'] for node in G.nodes()],
            textposition="top center",
            marker=dict(
                size=[np.sqrt(G.nodes[node]['population']) / 50 for node in G.nodes()],
                color=[G.nodes[node]['population'] for node in G.nodes()],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(
                    thickness=15,
                    title='Population',
                    xanchor='left',
                    titleside='right'
                ),
                line_width=2
            ),
            hovertext=[f"{G.nodes[node]['name']}<br>Population: {G.nodes[node]['population']:,}" 
                      for node in G.nodes()]
        )
        
        fig = go.Figure(data=edge_trace + [node_trace])
        fig.update_layout(
            title='Railway Network Graph',
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0, l=0, r=0, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=600
        )
        
        return fig, edges
    
    def create_statistics_charts(self, analysis_data: Dict, railway_data: Dict = None):
        """Create statistical visualization charts"""
        # Population distribution
        if 'cities' in analysis_data['core_results'] and analysis_data['core_results']['cities']:
            cities_df = pd.DataFrame(analysis_data['core_results']['cities'])
            
            # Population bar chart
            fig_pop = px.bar(
                cities_df.head(20),
                x='city_name' if 'city_name' in cities_df.columns else 'city',
                y='population',
                title='Top 20 Cities by Population',
                labels={'population': 'Population', 'city_name': 'City'},
                color='population',
                color_continuous_scale='Blues'
            )
            fig_pop.update_layout(xaxis_tickangle=-45, height=400)
            
            # Population distribution pie chart
            pop_ranges = pd.cut(
                cities_df['population'],
                bins=[0, 50000, 100000, 200000, 500000, float('inf')],
                labels=['<50k', '50k-100k', '100k-200k', '200k-500k', '>500k']
            )
            fig_pie = px.pie(
                values=pop_ranges.value_counts().values,
                names=pop_ranges.value_counts().index,
                title='City Population Distribution'
            )
            
            # Existing railway statistics if available
            if railway_data and railway_data.get('stats'):
                stats = railway_data['stats']
                
                # Rail type distribution
                if stats.get('rail_types'):
                    fig_rail = px.bar(
                        x=list(stats['rail_types'].keys()),
                        y=list(stats['rail_types'].values()),
                        title='Existing Railway Infrastructure by Type',
                        labels={'x': 'Railway Type', 'y': 'Count'},
                        color=list(stats['rail_types'].values()),
                        color_continuous_scale='Reds'
                    )
                else:
                    fig_rail = None
            else:
                fig_rail = None
                
            return fig_pop, fig_pie, fig_rail
        
        return None, None, None
    
    def create_demand_heatmap(self, analysis_data: Dict):
        """Create demand heatmap between cities"""
        if 'demand' in analysis_data['core_results'] and 'demand_matrix' in analysis_data['core_results']['demand']:
            demand_matrix = analysis_data['core_results']['demand']['demand_matrix']
            
            if demand_matrix:
                # Convert to DataFrame
                cities = list(demand_matrix.keys())[:10]  # Limit to top 10 cities
                matrix = []
                for city1 in cities:
                    row = []
                    for city2 in cities:
                        if city1 in demand_matrix and city2 in demand_matrix[city1]:
                            row.append(demand_matrix[city1][city2])
                        else:
                            row.append(0)
                    matrix.append(row)
                
                fig = go.Figure(data=go.Heatmap(
                    z=matrix,
                    x=cities,
                    y=cities,
                    colorscale='YlOrRd',
                    text=matrix,
                    texttemplate='%{text:.0f}',
                    textfont={"size": 10}
                ))
                
                fig.update_layout(
                    title='Passenger Demand Heatmap (Top 10 Cities)',
                    xaxis_tickangle=-45,
                    height=600
                )
                
                return fig
        
        return None
    
    def run(self):
        """Main dashboard execution"""
        # Header
        st.markdown('<h1 class="main-header">🚄 Railway Network Analysis Dashboard</h1>', 
                   unsafe_allow_html=True)
        
        # Sidebar
        with st.sidebar:
            st.header("Configuration")
            
            # Country selection
            country = st.text_input("Country", value=st.session_state.country)
            if country != st.session_state.country:
                st.session_state.country = country
                st.session_state.railway_data = None
            
            # Load data buttons
            if st.button("Load Analysis Data"):
                self.load_analysis_data()
            
            if st.button("Fetch Railway Data"):
                with st.spinner("Fetching railway data..."):
                    st.session_state.railway_data = self.data_fetcher.fetch_existing_railways(country)
                    st.success("Railway data loaded!")
            
            # Display data status
            st.divider()
            st.subheader("Data Status")
            if st.session_state.analysis_data:
                st.success("✅ Analysis data loaded")
                metadata = st.session_state.analysis_data.get('metadata', {})
                st.text(f"Country: {metadata.get('country', 'Unknown')}")
                st.text(f"Date: {metadata.get('analysis_date', 'Unknown')[:10]}")
            else:
                st.warning("❌ No analysis data")
                
            if st.session_state.railway_data:
                stats = st.session_state.railway_data.get('stats', {})
                st.success("✅ Railway data loaded")
                st.text(f"Rails: {stats.get('total_rails', 0)}")
                st.text(f"Stations: {stats.get('total_stations', 0)}")
            else:
                st.warning("❌ No railway data")
        
        # Main content
        if not st.session_state.analysis_data:
            st.info("Please load analysis data from the sidebar to begin.")
            return
            
        # Extract data
        analysis_data = st.session_state.analysis_data
        cities = analysis_data['core_results'].get('cities', [])
        
        # Key metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Cities", len(cities))
        with col2:
            total_pop = sum(c.get('population', 0) for c in cities)
            st.metric("Total Population", f"{total_pop:,.0f}")
        with col3:
            if st.session_state.railway_data:
                st.metric("Existing Rails", 
                         st.session_state.railway_data['stats'].get('total_rails', 0))
            else:
                st.metric("Existing Rails", "N/A")
        with col4:
            if st.session_state.railway_data:
                st.metric("Existing Stations", 
                         st.session_state.railway_data['stats'].get('total_stations', 0))
            else:
                st.metric("Existing Stations", "N/A")
        
        # Tabs for different views
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🗺️ Map View", 
            "📊 Statistics", 
            "🔗 Network Analysis",
            "🔥 Demand Analysis",
            "📄 Raw Data"
        ])
        
        with tab1:
            st.subheader("Railway Network Map")
            if cities:
                railway_map = self.create_railway_map(cities, st.session_state.railway_data)
                if railway_map:
                    st_folium(railway_map, width=None, height=600)
                else:
                    st.warning("No city data available for mapping")
            else:
                st.warning("No cities found in analysis data")
        
        with tab2:
            st.subheader("Statistical Analysis")
            fig_pop, fig_pie, fig_rail = self.create_statistics_charts(
                analysis_data, 
                st.session_state.railway_data
            )
            
            if fig_pop:
                st.plotly_chart(fig_pop, use_container_width=True)
                
            col1, col2 = st.columns(2)
            with col1:
                if fig_pie:
                    st.plotly_chart(fig_pie, use_container_width=True)
            with col2:
                if fig_rail:
                    st.plotly_chart(fig_rail, use_container_width=True)
                else:
                    st.info("Load railway data to see infrastructure statistics")
        
        with tab3:
            st.subheader("Network Analysis")
            if cities and len(cities) > 1:
                fig_network, edges = self.create_network_graph(cities)
                if fig_network:
                    st.plotly_chart(fig_network, use_container_width=True)
                    
                    # Show connections table
                    if edges:
                        st.subheader("Proposed Connections")
                        edges_df = pd.DataFrame(edges)
                        edges_df['distance'] = edges_df['distance'].round(2)
                        st.dataframe(edges_df, use_container_width=True)
            else:
                st.warning("Need at least 2 cities for network analysis")
        
        with tab4:
            st.subheader("Demand Analysis")
            demand_fig = self.create_demand_heatmap(analysis_data)
            if demand_fig:
                st.plotly_chart(demand_fig, use_container_width=True)
            else:
                st.info("No demand data available")
                
            # Show demand statistics
            if 'demand' in analysis_data['core_results']:
                demand_data = analysis_data['core_results']['demand']
                if 'market_analysis' in demand_data:
                    market = demand_data['market_analysis']
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Potential Annual Trips", 
                                 f"{market.get('potential_annual_trips', 0):,.0f}")
                    with col2:
                        st.metric("Market Value (USD)", 
                                 f"${market.get('total_market_value_usd', 0):,.0f}")
                    with col3:
                        st.metric("Major Cities", 
                                 market.get('major_cities_count', 0))
        
        with tab5:
            st.subheader("Raw Data View")
            
            # Analysis data
            if st.checkbox("Show Analysis Data"):
                st.json(analysis_data)
            
            # Railway data
            if st.checkbox("Show Railway Data") and st.session_state.railway_data:
                st.json(st.session_state.railway_data)
            
            # Export options
            st.subheader("Export Data")
            col1, col2 = st.columns(2)
            with col1:
                if cities:
                    cities_df = pd.DataFrame(cities)
                    csv = cities_df.to_csv(index=False)
                    st.download_button(
                        label="Download Cities CSV",
                        data=csv,
                        file_name=f"{country}_cities.csv",
                        mime="text/csv"
                    )
            with col2:
                if st.session_state.railway_data:
                    railway_json = json.dumps(st.session_state.railway_data, indent=2)
                    st.download_button(
                        label="Download Railway Data JSON",
                        data=railway_json,
                        file_name=f"{country}_railways.json",
                        mime="application/json"
                    )

# Main execution
if __name__ == "__main__":
    dashboard = RailwayDashboard()
    dashboard.run()