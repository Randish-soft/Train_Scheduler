import logging
import numpy as np
from typing import Dict, Any, List

class StationPositioner:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def position_stations(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Position stations along proposed routes"""
        self.logger.info("Positioning stations along routes")
        
        proposed_routes = context.get('proposed_routes', [])
        demand_data = context['demand_data']
        terrain_data = context['terrain_data']
        
        for route in proposed_routes:
            stations = self._position_stations_for_route(route, demand_data, terrain_data)
            route['stations'] = stations
        
        context['station_statistics'] = self._calculate_station_statistics(proposed_routes)
        
        self.logger.info(f"Positioned stations for {len(proposed_routes)} routes")
        return context
    
    def _position_stations_for_route(self, route: Dict[str, Any], demand_data: Dict[str, Any],
                                   terrain_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Position stations for a specific route"""
        route_name = route['name']
        cities_served = route['cities_served']
        alignment = route['alignment']
        route_details = route['details']
        
        self.logger.debug(f"Positioning stations for route: {route_name}")
        
        stations = []
        
        # Add stations at endpoint cities
        for city in cities_served:
            station = self._create_city_station(city, route_details, 'major')
            stations.append(station)
        
        # Add intermediate stations based on spacing
        station_spacing = route_details['station_spacing_km']
        total_distance = alignment['total_distance_km']
        
        if len(cities_served) >= 2:
            # Calculate positions for intermediate stations
            num_intermediate = max(0, int(total_distance / station_spacing) - 1)
            
            for i in range(num_intermediate):
                position_ratio = (i + 1) / (num_intermediate + 1)
                station_name = f"Intermediate_{i+1}"
                
                station = self._create_intermediate_station(
                    station_name, position_ratio, route_details, terrain_data
                )
                stations.append(station)
        
        # Sort stations by position along route
        stations.sort(key=lambda x: x['position_km'])
        
        # Add station-specific details
        for i, station in enumerate(stations):
            station['station_id'] = f"{route_name}_ST{i+1:03d}"
            station['platform_count'] = self._determine_platform_count(station, route_details)
            station['facilities'] = self._determine_station_facilities(station, route_details)
        
        return stations
    
    def _create_city_station(self, city_name: str, route_details: Dict[str, Any], 
                           station_type: str) -> Dict[str, Any]:
        """Create a station in a major city"""
        track_type = route_details['track_type']
        
        return {
            'name': f"{city_name} Central",
            'type': station_type,
            'serves_city': city_name,
            'position_km': 0 if station_type == 'origin' else route_details['infrastructure_requirements']['earthworks_volume_cu_m'] / 10000,
            'estimated_daily_passengers': self._estimate_station_demand(city_name, track_type),
            'intermodal_connections': self._determine_intermodal_connections(city_name, station_type),
            'urban_integration_level': 'high',
            'land_requirement_hectares': self._calculate_land_requirement(station_type, track_type),
            'construction_complexity': 'high' if station_type == 'major' else 'medium'
        }
    
    def _create_intermediate_station(self, station_name: str, position_ratio: float,
                                   route_details: Dict[str, Any], terrain_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create an intermediate station"""
        total_distance = route_details['infrastructure_requirements']['earthworks_volume_cu_m'] / 10000
        position_km = total_distance * position_ratio
        
        # Determine station type based on terrain and position
        terrain_difficulty = terrain_data.get('terrain_difficulty', 0.5)
        
        if terrain_difficulty > 0.7 and position_ratio > 0.8:
            station_type = 'mountain_pass'
        elif position_ratio < 0.3 or position_ratio > 0.7:
            station_type = 'regional'
        else:
            station_type = 'local'
        
        return {
            'name': station_name,
            'type': station_type,
            'serves_city': None,
            'position_km': position_km,
            'estimated_daily_passengers': self._estimate_intermediate_demand(position_ratio, route_details),
            'intermodal_connections': ['bus', 'parking'],
            'urban_integration_level': 'medium',
            'land_requirement_hectares': self._calculate_land_requirement(station_type, route_details['track_type']),
            'construction_complexity': 'medium' if station_type == 'regional' else 'low'
        }
    
    def _estimate_station_demand(self, city_name: str, track_type: str) -> int:
        """Estimate daily passenger demand for a city station"""
        # Base demand based on track type and city size assumption
        base_demand = {
            'commuter': 5000,
            'regional': 3000,
            'high_speed': 2000,
            'mountain': 1500
        }.get(track_type, 2500)
        
        # Adjust for city importance (capital cities have higher demand)
        if 'capital' in city_name.lower():
            base_demand *= 3
        elif 'major' in city_name.lower() or 'central' in city_name.lower():
            base_demand *= 2
        
        return int(base_demand * np.random.uniform(0.8, 1.2))  # Add some randomness
    
    def _estimate_intermediate_demand(self, position_ratio: float, route_details: Dict[str, Any]) -> int:
        """Estimate demand for intermediate stations"""
        track_type = route_details['track_type']
        
        # Base demand based on track type
        base_demand = {
            'commuter': 800,
            'regional': 500,
            'high_speed': 300,
            'mountain': 200
        }.get(track_type, 400)
        
        # Stations near endpoints typically have higher demand
        position_factor = 1.0
        if position_ratio < 0.2 or position_ratio > 0.8:
            position_factor = 1.5
        elif position_ratio < 0.4 or position_ratio > 0.6:
            position_factor = 1.2
        
        return int(base_demand * position_factor * np.random.uniform(0.7, 1.3))
    
    def _determine_intermodal_connections(self, city_name: str, station_type: str) -> List[str]:
        """Determine what intermodal connections the station should have"""
        connections = ['bus']
        
        if station_type == 'major':
            connections.extend(['taxi', 'parking', 'bike_sharing'])
        
        if 'capital' in city_name.lower() or station_type == 'major':
            connections.extend(['metro', 'tram', 'regional_bus'])
        
        if station_type in ['major', 'regional']:
            connections.append('park_and_ride')
        
        return connections
    
    def _calculate_land_requirement(self, station_type: str, track_type: str) -> float:
        """Calculate land requirement in hectares"""
        base_requirements = {
            'major': 5.0,
            'regional': 2.0,
            'local': 0.5,
            'mountain_pass': 1.0
        }
        
        base_hectares = base_requirements.get(station_type, 1.0)
        
        # Adjust for track type
        if track_type == 'high_speed':
            base_hectares *= 1.5  # High-speed stations need more space
        elif track_type == 'commuter':
            base_hectares *= 1.2  # Commuter stations need parking
        
        return base_hectares
    
    def _determine_platform_count(self, station: Dict[str, Any], route_details: Dict[str, Any]) -> int:
        """Determine number of platforms needed"""
        track_type = route_details['track_type']
        station_type = station['type']
        
        if station_type == 'major':
            if track_type == 'high_speed':
                return 4
            else:
                return 3
        elif station_type == 'regional':
            return 2
        else:  # local, mountain_pass
            return 1
    
    def _determine_station_facilities(self, station: Dict[str, Any], route_details: Dict[str, Any]) -> Dict[str, Any]:
        """Determine what facilities the station should have"""
        station_type = station['type']
        track_type = route_details['track_type']
        
        facilities = {
            'waiting_room': station_type in ['major', 'regional'],
            'ticket_office': station_type == 'major',
            'vending_machines': True,
            'restrooms': station_type in ['major', 'regional'],
            'information_display': True,
            'accessibility_features': True
        }
        
        if station_type == 'major':
            facilities.update({
                'commercial_spaces': True,
                'luggage_storage': True,
                'food_beverage': True,
                'first_aid_room': True
            })
        
        if track_type == 'high_speed' and station_type in ['major', 'regional']:
            facilities.update({
                'business_lounge': True,
                'baggage_handling': True
            })
        
        return facilities
    
    def _calculate_station_statistics(self, proposed_routes: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate statistics for all stations"""
        all_stations = []
        for route in proposed_routes:
            all_stations.extend(route.get('stations', []))
        
        if not all_stations:
            return {}
        
        station_types = {}
        total_passengers = 0
        total_land_required = 0
        
        for station in all_stations:
            station_type = station['type']
            station_types[station_type] = station_types.get(station_type, 0) + 1
            total_passengers += station.get('estimated_daily_passengers', 0)
            total_land_required += station.get('land_requirement_hectares', 0)
        
        return {
            'total_stations': len(all_stations),
            'station_type_breakdown': station_types,
            'total_estimated_daily_passengers': total_passengers,
            'average_passengers_per_station': total_passengers / len(all_stations),
            'total_land_requirement_hectares': total_land_required,
            'major_stations_count': station_types.get('major', 0),
            'regional_stations_count': station_types.get('regional', 0),
            'local_stations_count': station_types.get('local', 0) + station_types.get('mountain_pass', 0)
        }