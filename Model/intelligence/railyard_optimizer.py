# File: railway_ai/intelligence/railyard_optimizer.py
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass

@dataclass
class RailyardCandidate:
    lat: float
    lon: float
    score: float
    capacity_estimate: int
    land_cost_factor: float
    accessibility_score: float
    maintenance_type: str  # 'depot', 'yard', 'terminal'
    reasons: List[str]

class RailyardOptimizer:
    def __init__(self):
        self.location_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.capacity_model = RandomForestRegressor(n_estimators=50, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        
    def learn_from_existing_railyards(self, railyard_data: List[Dict]) -> None:
        """Learn patterns from existing railyard locations and characteristics"""
        if len(railyard_data) < 10:
            print("Warning: Limited railyard data for training")
            return
            
        features = []
        location_scores = []
        capacities = []
        
        for yard in railyard_data:
            feature_vector = self._extract_railyard_features(yard)
            features.append(feature_vector)
            
            # Score based on utilization and efficiency metrics
            location_scores.append(self._calculate_yard_score(yard))
            capacities.append(yard.get('capacity', 50))  # default capacity
        
        # Train models
        features_scaled = self.scaler.fit_transform(features)
        self.location_model.fit(features_scaled, location_scores)
        self.capacity_model.fit(features_scaled, capacities)
        self.trained = True
        
    def _extract_railyard_features(self, yard: Dict) -> List[float]:
        """Extract features for ML model from railyard data"""
        return [
            yard.get('distance_to_city_center', 10),  # km
            yard.get('distance_to_major_station', 5),  # km
            yard.get('elevation', 100),  # meters
            yard.get('land_area_hectares', 20),
            yard.get('track_connections', 4),
            yard.get('industrial_proximity', 0.5),  # 0-1 score
            yard.get('highway_access', 0.8),  # 0-1 score
            yard.get('population_density_1km', 1000),  # people/km²
            yard.get('freight_demand_score', 0.6),  # 0-1 score
            yard.get('terrain_flatness', 0.9),  # 0-1 score (1=flat)
            yard.get('water_access', 0.2),  # 0-1 score
            yard.get('power_grid_distance', 2),  # km to power infrastructure
        ]
    
    def _calculate_yard_score(self, yard: Dict) -> float:
        """Calculate efficiency score for existing railyard"""
        base_score = 0.5
        
        # Positive factors
        if yard.get('utilization_rate', 0.7) > 0.8:
            base_score += 0.2
        if yard.get('on_time_performance', 0.85) > 0.9:
            base_score += 0.15
        if yard.get('cost_efficiency', 0.7) > 0.8:
            base_score += 0.15
        
        # Negative factors
        if yard.get('congestion_incidents', 5) > 10:
            base_score -= 0.1
        if yard.get('maintenance_issues', 3) > 5:
            base_score -= 0.05
            
        return max(0, min(1, base_score))
    
    def optimize_railyard_locations(self, 
                                  stations: List[Dict],
                                  terrain_data: Dict,
                                  network_demand: Dict,
                                  budget_constraints: Optional[Dict] = None) -> List[RailyardCandidate]:
        """Find optimal railyard locations for a railway network"""
        
        if not self.trained:
            print("Warning: Model not trained. Using heuristic approach.")
            return self._heuristic_optimization(stations, terrain_data, network_demand)
        
        # Generate candidate locations
        candidates = self._generate_candidate_locations(stations, terrain_data)
        
        # Score candidates using trained models
        scored_candidates = []
        for candidate in candidates:
            score = self._score_candidate_location(candidate, stations, network_demand)
            if score > 0.3:  # Minimum viability threshold
                scored_candidates.append(candidate)
        
        # Apply network optimization
        optimized_candidates = self._network_level_optimization(scored_candidates, stations)
        
        # Apply budget constraints if provided
        if budget_constraints:
            optimized_candidates = self._apply_budget_constraints(optimized_candidates, budget_constraints)
        
        return sorted(optimized_candidates, key=lambda x: x.score, reverse=True)
    
    def _generate_candidate_locations(self, stations: List[Dict], terrain_data: Dict) -> List[Dict]:
        """Generate potential railyard locations"""
        candidates = []
        
        # Strategy 1: Cluster-based locations (near station clusters)
        station_coords = [(s['lat'], s['lon']) for s in stations]
        if len(station_coords) > 3:
            kmeans = KMeans(n_clusters=min(5, len(station_coords)//3), random_state=42)
            clusters = kmeans.fit(station_coords)
            
            for center in clusters.cluster_centers_:
                candidates.append({
                    'lat': center[0],
                    'lon': center[1],
                    'type': 'cluster_center',
                    'nearby_stations': self._find_nearby_stations(center, stations, radius_km=15)
                })
        
        # Strategy 2: Major station proximity (maintenance depots)
        major_stations = [s for s in stations if s.get('category') in ['intercity', 'major']]
        for station in major_stations:
            # Place depot 5-10km from major station
            for angle in [0, 90, 180, 270]:  # Cardinal directions
                distance_km = 7
                new_lat, new_lon = self._offset_coordinates(
                    station['lat'], station['lon'], distance_km, angle
                )
                candidates.append({
                    'lat': new_lat,
                    'lon': new_lon,
                    'type': 'station_depot',
                    'parent_station': station['name'],
                    'nearby_stations': self._find_nearby_stations((new_lat, new_lon), stations, radius_km=10)
                })
        
        # Strategy 3: Network terminals (end-of-line locations)
        terminal_candidates = self._identify_terminal_locations(stations)
        candidates.extend(terminal_candidates)
        
        # Strategy 4: Freight hubs (industrial areas)
        freight_candidates = self._identify_freight_locations(stations, terrain_data)
        candidates.extend(freight_candidates)
        
        return candidates
    
    def _score_candidate_location(self, candidate: Dict, stations: List[Dict], demand: Dict) -> RailyardCandidate:
        """Score a candidate location using ML models and heuristics"""
        
        # Extract features for ML prediction
        features = self._extract_candidate_features(candidate, stations, demand)
        
        if self.trained:
            features_scaled = self.scaler.transform([features])
            location_score = self.location_model.predict(features_scaled)[0]
            capacity_estimate = int(self.capacity_model.predict(features_scaled)[0])
        else:
            location_score = self._heuristic_score(candidate, stations)
            capacity_estimate = self._estimate_capacity_heuristic(candidate, stations)
        
        # Additional scoring factors
        accessibility_score = self._calculate_accessibility(candidate, stations)
        land_cost_factor = self._estimate_land_cost(candidate)
        maintenance_type = self._determine_maintenance_type(candidate, stations)
        reasons = self._generate_reasons(candidate, stations)
        
        return RailyardCandidate(
            lat=candidate['lat'],
            lon=candidate['lon'],
            score=location_score,
            capacity_estimate=capacity_estimate,
            land_cost_factor=land_cost_factor,
            accessibility_score=accessibility_score,
            maintenance_type=maintenance_type,
            reasons=reasons
        )
    
    def _extract_candidate_features(self, candidate: Dict, stations: List[Dict], demand: Dict) -> List[float]:
        """Extract features for candidate location"""
        nearby_stations = candidate.get('nearby_stations', [])
        
        # Distance to nearest major city (approximated by largest station)
        major_stations = [s for s in stations if s.get('category') in ['intercity', 'major']]
        min_city_distance = min([
            self._haversine_distance(candidate['lat'], candidate['lon'], s['lat'], s['lon'])
            for s in major_stations
        ]) if major_stations else 20
        
        # Distance to nearest station
        min_station_distance = min([
            self._haversine_distance(candidate['lat'], candidate['lon'], s['lat'], s['lon'])
            for s in stations
        ]) if stations else 10
        
        return [
            min_city_distance,
            min_station_distance,
            100,  # elevation (would be real data)
            25,   # estimated land area
            len(nearby_stations),
            0.6,  # industrial proximity estimate
            0.7,  # highway access estimate
            800,  # population density estimate
            demand.get('freight_score', 0.5),
            0.8,  # terrain flatness estimate
            0.3,  # water access
            1.5,  # power grid distance
        ]
    
    def _heuristic_score(self, candidate: Dict, stations: List[Dict]) -> float:
        """Heuristic scoring when ML model isn't available"""
        score = 0.5  # Base score
        
        nearby_stations = candidate.get('nearby_stations', [])
        
        # Proximity to stations (good but not too close)
        if nearby_stations:
            avg_distance = sum([
                self._haversine_distance(candidate['lat'], candidate['lon'], s['lat'], s['lon'])
                for s in nearby_stations
            ]) / len(nearby_stations)
            
            # Optimal distance is 5-15km from stations
            if 5 <= avg_distance <= 15:
                score += 0.2
            elif avg_distance < 5:
                score -= 0.1  # Too close
            elif avg_distance > 20:
                score -= 0.15  # Too far
        
        # Network connectivity
        if len(nearby_stations) >= 3:
            score += 0.15
        
        # Type-specific bonuses
        if candidate.get('type') == 'cluster_center':
            score += 0.1
        elif candidate.get('type') == 'terminal':
            score += 0.05
        
        return max(0, min(1, score))
    
    def _estimate_capacity_heuristic(self, candidate: Dict, stations: List[Dict]) -> int:
        """Estimate railyard capacity based on location type"""
        base_capacity = 30
        
        nearby_count = len(candidate.get('nearby_stations', []))
        capacity = base_capacity + (nearby_count * 5)
        
        if candidate.get('type') == 'cluster_center':
            capacity *= 1.5
        elif candidate.get('type') == 'terminal':
            capacity *= 0.8
        elif candidate.get('type') == 'freight_hub':
            capacity *= 2.0
        
        return int(capacity)
    
    def _network_level_optimization(self, candidates: List[RailyardCandidate], stations: List[Dict]) -> List[RailyardCandidate]:
        """Optimize railyard network to avoid redundancy"""
        if len(candidates) <= 3:
            return candidates
        
        # Remove candidates that are too close to each other
        optimized = []
        min_distance_km = 25  # Minimum distance between railyards
        
        sorted_candidates = sorted(candidates, key=lambda x: x.score, reverse=True)
        
        for candidate in sorted_candidates:
            too_close = False
            for existing in optimized:
                distance = self._haversine_distance(
                    candidate.lat, candidate.lon,
                    existing.lat, existing.lon
                )
                if distance < min_distance_km:
                    too_close = True
                    break
            
            if not too_close:
                optimized.append(candidate)
            
            # Limit total number of railyards
            if len(optimized) >= len(stations) // 4:  # Roughly 1 railyard per 4 stations
                break
        
        return optimized
    
    def _identify_terminal_locations(self, stations: List[Dict]) -> List[Dict]:
        """Identify potential terminal/turnaround locations"""
        # Find stations with only 1-2 connections (end of lines)
        terminals = []
        
        # This would use actual network topology
        # For now, simulate by finding geographically isolated stations
        for station in stations:
            nearby = self._find_nearby_stations((station['lat'], station['lon']), stations, radius_km=20)
            if len(nearby) <= 2:  # Isolated station
                # Place terminal 3km away
                for angle in [45, 135, 225, 315]:
                    new_lat, new_lon = self._offset_coordinates(
                        station['lat'], station['lon'], 3, angle
                    )
                    terminals.append({
                        'lat': new_lat,
                        'lon': new_lon,
                        'type': 'terminal',
                        'parent_station': station['name'],
                        'nearby_stations': [station]
                    })
        
        return terminals[:3]  # Limit number of terminals
    
    def _identify_freight_locations(self, stations: List[Dict], terrain_data: Dict) -> List[Dict]:
        """Identify potential freight hub locations"""
        freight_hubs = []
        
        # Look for flat areas away from city centers
        for station in stations:
            if station.get('category') == 'regional':  # Mid-tier stations often near industrial areas
                # Place freight hub 8km away in flat terrain
                new_lat, new_lon = self._offset_coordinates(
                    station['lat'], station['lon'], 8, 180  # South of station
                )
                freight_hubs.append({
                    'lat': new_lat,
                    'lon': new_lon,
                    'type': 'freight_hub',
                    'parent_station': station['name'],
                    'nearby_stations': self._find_nearby_stations((new_lat, new_lon), stations, radius_km=12)
                })
        
        return freight_hubs[:2]  # Limit freight hubs
    
    def _calculate_accessibility(self, candidate: Dict, stations: List[Dict]) -> float:
        """Calculate accessibility score (0-1)"""
        nearby_stations = candidate.get('nearby_stations', [])
        
        if not nearby_stations:
            return 0.2
        
        # Score based on number and quality of nearby stations
        score = min(len(nearby_stations) / 5, 1.0) * 0.6
        
        # Bonus for major stations nearby
        major_nearby = [s for s in nearby_stations if s.get('category') in ['intercity', 'major']]
        score += min(len(major_nearby) / 2, 1.0) * 0.4
        
        return score
    
    def _estimate_land_cost(self, candidate: Dict) -> float:
        """Estimate relative land cost factor (1.0 = average)"""
        # Higher costs near major stations, lower in rural areas
        candidate_type = candidate.get('type', '')
        
        if candidate_type == 'station_depot':
            return 1.3  # Near stations = higher cost
        elif candidate_type == 'cluster_center':
            return 1.1  # Near urban areas
        elif candidate_type == 'terminal':
            return 0.8  # Remote areas
        elif candidate_type == 'freight_hub':
            return 0.9  # Industrial areas
        
        return 1.0
    
    def _determine_maintenance_type(self, candidate: Dict, stations: List[Dict]) -> str:
        """Determine the type of maintenance facility"""
        candidate_type = candidate.get('type', '')
        nearby_count = len(candidate.get('nearby_stations', []))
        
        if candidate_type == 'station_depot':
            return 'depot'
        elif candidate_type == 'freight_hub':
            return 'yard'
        elif candidate_type == 'terminal':
            return 'terminal'
        elif nearby_count >= 4:
            return 'yard'
        else:
            return 'depot'
    
    def _generate_reasons(self, candidate: Dict, stations: List[Dict]) -> List[str]:
        """Generate human-readable reasons for railyard placement"""
        reasons = []
        
        candidate_type = candidate.get('type', '')
        nearby_count = len(candidate.get('nearby_stations', []))
        
        if candidate_type == 'cluster_center':
            reasons.append(f"Central location serving {nearby_count} nearby stations")
        elif candidate_type == 'station_depot':
            reasons.append(f"Maintenance depot for {candidate.get('parent_station', 'major station')}")
        elif candidate_type == 'terminal':
            reasons.append("Terminal facility for end-of-line operations")
        elif candidate_type == 'freight_hub':
            reasons.append("Strategic freight operations center")
        
        if nearby_count >= 4:
            reasons.append("High connectivity to railway network")
        
        return reasons
    
    def _apply_budget_constraints(self, candidates: List[RailyardCandidate], constraints: Dict) -> List[RailyardCandidate]:
        """Apply budget constraints to candidate selection"""
        budget = constraints.get('total_budget', float('inf'))
        cost_per_facility = constraints.get('cost_per_facility', 50_000_000)  # €50M default
        
        # Sort by score/cost ratio
        for candidate in candidates:
            candidate.cost_estimate = cost_per_facility * candidate.land_cost_factor
            candidate.value_ratio = candidate.score / candidate.land_cost_factor
        
        candidates.sort(key=lambda x: x.value_ratio, reverse=True)
        
        # Select facilities within budget
        selected = []
        total_cost = 0
        
        for candidate in candidates:
            if total_cost + candidate.cost_estimate <= budget:
                selected.append(candidate)
                total_cost += candidate.cost_estimate
        
        return selected
    
    def _find_nearby_stations(self, coords: Tuple[float, float], stations: List[Dict], radius_km: float) -> List[Dict]:
        """Find stations within radius of coordinates"""
        lat, lon = coords
        nearby = []
        
        for station in stations:
            distance = self._haversine_distance(lat, lon, station['lat'], station['lon'])
            if distance <= radius_km:
                nearby.append(station)
        
        return nearby
    
    def _offset_coordinates(self, lat: float, lon: float, distance_km: float, bearing_degrees: float) -> Tuple[float, float]:
        """Calculate new coordinates offset by distance and bearing"""
        R = 6371  # Earth's radius in km
        bearing = math.radians(bearing_degrees)
        
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        
        lat2 = math.asin(math.sin(lat1) * math.cos(distance_km/R) +
                        math.cos(lat1) * math.sin(distance_km/R) * math.cos(bearing))
        
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(distance_km/R) * math.cos(lat1),
                                math.cos(distance_km/R) - math.sin(lat1) * math.sin(lat2))
        
        return math.degrees(lat2), math.degrees(lon2)
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    def _heuristic_optimization(self, stations: List[Dict], terrain_data: Dict, demand: Dict) -> List[RailyardCandidate]:
        """Fallback heuristic approach when no training data available"""
        candidates = self._generate_candidate_locations(stations, terrain_data)
        scored_candidates = []
        
        for candidate in candidates:
            score = self._heuristic_score(candidate, stations)
            if score > 0.3:
                railyard_candidate = RailyardCandidate(
                    lat=candidate['lat'],
                    lon=candidate['lon'],
                    score=score,
                    capacity_estimate=self._estimate_capacity_heuristic(candidate, stations),
                    land_cost_factor=self._estimate_land_cost(candidate),
                    accessibility_score=self._calculate_accessibility(candidate, stations),
                    maintenance_type=self._determine_maintenance_type(candidate, stations),
                    reasons=self._generate_reasons(candidate, stations)
                )
                scored_candidates.append(railyard_candidate)
        
        return self._network_level_optimization(scored_candidates, stations)