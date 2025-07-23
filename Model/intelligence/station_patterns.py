# File: model/intelligence/station_patterns.py
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from collections import Counter

@dataclass
class StationPlacement:
    lat: float
    lon: float
    station_type: str  # 'intercity', 'regional', 'local', 'suburban'
    confidence: float
    population_served: int
    transfer_potential: float
    accessibility_score: float
    reasons: List[str]

class StationPatternAnalyzer:
    def __init__(self):
        self.placement_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.spacing_predictor = GradientBoostingRegressor(n_estimators=100, random_state=42)
        self.demand_predictor = GradientBoostingRegressor(n_estimators=80, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.learned_patterns = {}
        
    def learn_from_existing_stations(self, stations: List[Dict], population_data: List[Dict] = None) -> None:
        """Learn placement patterns from existing railway stations"""
        if len(stations) < 20:
            print("Warning: Limited station data for robust pattern learning")
            
        # Extract spatial patterns
        self._analyze_spatial_distribution(stations)
        
        # Learn placement rules
        self._learn_placement_rules(stations, population_data)
        
        # Analyze station hierarchy
        self._analyze_station_hierarchy(stations)
        
        # Learn spacing patterns
        self._learn_spacing_patterns(stations)
        
        self.trained = True
        print(f"Learned patterns from {len(stations)} stations")
    
    def _analyze_spatial_distribution(self, stations: List[Dict]) -> None:
        """Analyze how stations are spatially distributed"""
        coordinates = [(s['lat'], s['lon']) for s in stations]
        
        # Cluster analysis to find station density patterns
        if len(coordinates) > 10:
            # Multiple cluster sizes to understand hierarchy
            for n_clusters in [3, 5, 8]:
                if n_clusters < len(coordinates):
                    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
                    cluster_labels = kmeans.fit_predict(coordinates)
                    
                    # Analyze cluster characteristics
                    cluster_info = {}
                    for i in range(n_clusters):
                        cluster_stations = [s for j, s in enumerate(stations) if cluster_labels[j] == i]
                        cluster_info[i] = {
                            'station_count': len(cluster_stations),
                            'avg_category': self._get_dominant_category(cluster_stations),
                            'density_score': len(cluster_stations) / max(1, self._calculate_cluster_area(cluster_stations))
                        }
                    
                    self.learned_patterns[f'clusters_{n_clusters}'] = cluster_info
        
        # Distance-based density analysis
        self._analyze_station_density(stations)
    
    def _learn_placement_rules(self, stations: List[Dict], population_data: List[Dict] = None) -> None:
        """Learn what makes a good station location"""
        features = []
        station_types = []
        
        for station in stations:
            # Extract features for each station
            feature_vector = self._extract_station_features(station, stations, population_data)
            features.append(feature_vector)
            
            # Target: station type/category
            station_types.append(station.get('category', 'regional'))
        
        if len(features) > 10:  # Minimum for training
            # Train classification model for station type prediction
            X_train, X_test, y_train, y_test = train_test_split(
                features, station_types, test_size=0.2, random_state=42
            )
            
            X_train_scaled = self.scaler.fit_transform(X_train)
            self.placement_classifier.fit(X_train_scaled, y_train)
            
            # Get feature importance
            feature_names = [
                'population_1km', 'population_5km', 'distance_to_city_center',
                'elevation', 'slope', 'nearest_station_distance', 'station_density_10km',
                'road_connectivity', 'water_proximity', 'industrial_proximity',
                'residential_density', 'commercial_density'
            ]
            
            importance_scores = dict(zip(feature_names, self.placement_classifier.feature_importances_))
            self.learned_patterns['feature_importance'] = importance_scores
    
    def _extract_station_features(self, station: Dict, all_stations: List[Dict], population_data: List[Dict] = None) -> List[float]:
        """Extract features that characterize a station location"""
        lat, lon = station['lat'], station['lon']
        
        # Population-based features
        pop_1km = self._estimate_population_radius(lat, lon, 1.0, population_data)
        pop_5km = self._estimate_population_radius(lat, lon, 5.0, population_data)
        
        # Distance to nearest major city (approximated)
        city_distance = self._estimate_city_center_distance(lat, lon, all_stations)
        
        # Topographical features
        elevation = station.get('elevation', 100)  # Would be real elevation data
        slope = self._estimate_terrain_slope(lat, lon)
        
        # Network features
        nearest_station_dist = self._distance_to_nearest_station(station, all_stations)
        station_density = self._calculate_local_station_density(station, all_stations, radius_km=10)
        
        # Infrastructure features (simplified estimates)
        road_connectivity = self._estimate_road_connectivity(lat, lon)
        water_proximity = self._estimate_water_proximity(lat, lon)
        industrial_proximity = self._estimate_industrial_proximity(lat, lon)
        residential_density = self._estimate_residential_density(lat, lon)
        commercial_density = self._estimate_commercial_density(lat, lon)
        
        return [
            pop_1km, pop_5km, city_distance, elevation, slope,
            nearest_station_dist, station_density, road_connectivity,
            water_proximity, industrial_proximity, residential_density, commercial_density
        ]
    
    def _analyze_station_hierarchy(self, stations: List[Dict]) -> None:
        """Analyze the hierarchy of station types and their relationships"""
        category_counts = Counter([s.get('category', 'unknown') for s in stations])
        
        # Calculate average distances between different station types
        hierarchy_analysis = {}
        for category in category_counts.keys():
            category_stations = [s for s in stations if s.get('category') == category]
            
            if len(category_stations) > 1:
                distances = []
                for i, station1 in enumerate(category_stations):
                    for station2 in category_stations[i+1:]:
                        dist = self._haversine_distance(
                            station1['lat'], station1['lon'],
                            station2['lat'], station2['lon']
                        )
                        distances.append(dist)
                
                hierarchy_analysis[category] = {
                    'count': len(category_stations),
                    'avg_spacing': np.mean(distances) if distances else 0,
                    'min_spacing': np.min(distances) if distances else 0,
                    'max_spacing': np.max(distances) if distances else 0,
                    'std_spacing': np.std(distances) if distances else 0
                }
        
        self.learned_patterns['hierarchy'] = hierarchy_analysis
    
    def _learn_spacing_patterns(self, stations: List[Dict]) -> None:
        """Learn optimal spacing patterns between stations"""
        spacing_data = []
        
        for i, station in enumerate(stations):
            # Find nearest stations
            distances = []
            for j, other_station in enumerate(stations):
                if i != j:
                    dist = self._haversine_distance(
                        station['lat'], station['lon'],
                        other_station['lat'], other_station['lon']
                    )
                    distances.append(dist)
            
            if distances:
                nearest_distance = min(distances)
                avg_distance_to_5_nearest = np.mean(sorted(distances)[:5])
                
                spacing_data.append({
                    'station_type': station.get('category', 'regional'),
                    'nearest_distance': nearest_distance,
                    'avg_distance_5': avg_distance_to_5_nearest,
                    'population_served': self._estimate_population_radius(station['lat'], station['lon'], 2.0),
                    'platform_count': station.get('platforms', 1)
                })
        
        # Analyze spacing patterns by station type
        spacing_patterns = {}
        for station_type in ['intercity', 'regional', 'local', 'suburban']:
            type_data = [s for s in spacing_data if s['station_type'] == station_type]
            if type_data:
                spacing_patterns[station_type] = {
                    'optimal_spacing': np.median([s['nearest_distance'] for s in type_data]),
                    'min_spacing': np.percentile([s['nearest_distance'] for s in type_data], 25),
                    'max_spacing': np.percentile([s['nearest_distance'] for s in type_data], 75),
                    'population_threshold': np.median([s['population_served'] for s in type_data])
                }
        
        self.learned_patterns['spacing'] = spacing_patterns
    
    
    def optimize_station_placement(self, 
                                 route_points: List[Tuple[float, float]],
                                 population_data: List[Dict] = None,
                                 constraints: Dict = None) -> List[StationPlacement]:
        """Optimize station placement along a route using learned patterns"""
        
        if not self.trained:
            print("Warning: Using heuristic approach - model not trained")
            return self._heuristic_station_placement(route_points, population_data, constraints)
        
        # Generate candidate locations along route
        candidates = self._generate_station_candidates(route_points, population_data)
        
        # Score candidates using learned patterns
        scored_candidates = []
        for candidate in candidates:
            placement = self._score_station_candidate(candidate, route_points, population_data)
            if placement.confidence > 0.3:  # Minimum confidence threshold
                scored_candidates.append(placement)
        
        # Apply network-level optimization
        optimized_placements = self._optimize_station_network(scored_candidates, constraints)
        
        return optimized_placements
    
    def _generate_station_candidates(self, route_points: List[Tuple[float, float]], 
                                   population_data: List[Dict] = None) -> List[Dict]:
        """Generate candidate station locations along route"""
        candidates = []
        
        # Strategy 1: Regular spacing based on learned patterns
        for station_type in ['intercity', 'regional', 'local']:
            spacing_info = self.learned_patterns.get('spacing', {}).get(station_type, {})
            optimal_spacing = spacing_info.get('optimal_spacing', 15 if station_type == 'intercity' else 8)
            
            # Place candidates along route at optimal spacing
            route_candidates = self._place_candidates_by_spacing(route_points, optimal_spacing, station_type)
            candidates.extend(route_candidates)
        
        # Strategy 2: Population-driven placement
        if population_data:
            pop_candidates = self._place_candidates_by_population(route_points, population_data)
            candidates.extend(pop_candidates)
        
        # Strategy 3: Transfer hub opportunities
        transfer_candidates = self._identify_transfer_opportunities(route_points)
        candidates.extend(transfer_candidates)
        
        # Remove duplicates (candidates too close to each other)
        candidates = self._remove_duplicate_candidates(candidates, min_distance_km=2.0)
        
        return candidates
    
    def _place_candidates_by_spacing(self, route_points: List[Tuple[float, float]], 
                                   spacing_km: float, station_type: str) -> List[Dict]:
        """Place candidates at regular spacing along route"""
        candidates = []
        
        # Calculate cumulative distances along route
        route_distances = [0]
        for i in range(1, len(route_points)):
            dist = self._haversine_distance(
                route_points[i-1][0], route_points[i-1][1],
                route_points[i][0], route_points[i][1]
            )
            route_distances.append(route_distances[-1] + dist)
        
        total_distance = route_distances[-1]
        
        # Place stations at regular intervals
        current_distance = spacing_km / 2  # Start offset
        while current_distance < total_distance - spacing_km / 2:
            # Find route point closest to current distance
            closest_idx = min(range(len(route_distances)), 
                            key=lambda i: abs(route_distances[i] - current_distance))
            
            candidates.append({
                'lat': route_points[closest_idx][0],
                'lon': route_points[closest_idx][1],
                'suggested_type': station_type,
                'placement_reason': 'optimal_spacing',
                'route_distance_km': current_distance
            })
            
            current_distance += spacing_km
        
        return candidates
    
    def _place_candidates_by_population(self, route_points: List[Tuple[float, float]], 
                                      population_data: List[Dict]) -> List[Dict]:
        """Place candidates near population centers"""
        candidates = []
        
        for point in route_points[::5]:  # Sample every 5th point to avoid too many candidates
            lat, lon = point
            
            # Check population within 5km radius
            population_5km = self._estimate_population_radius(lat, lon, 5.0, population_data)
            
            # Population thresholds for different station types
            if population_5km > 100000:
                station_type = 'intercity'
            elif population_5km > 25000:
                station_type = 'regional'
            elif population_5km > 5000:
                station_type = 'local'
            else:
                continue  # Not enough population
            
            candidates.append({
                'lat': lat,
                'lon': lon,
                'suggested_type': station_type,
                'placement_reason': 'population_center',
                'population_served': population_5km
            })
        
        return candidates
    
    def _identify_transfer_opportunities(self, route_points: List[Tuple[float, float]]) -> List[Dict]:
        """Identify potential transfer/interchange locations"""
        candidates = []
        
        # Look for route intersections or major cities (simplified)
        # In reality, this would integrate with existing transport networks
        
        for i, point in enumerate(route_points):
            if i % 10 == 0:  # Sample points
                lat, lon = point
                
                # Estimate if this could be a transfer hub
                # (In real implementation, would check for existing rail lines, airports, etc.)
                transfer_potential = self._estimate_transfer_potential(lat, lon)
                
                if transfer_potential > 0.6:
                    candidates.append({
                        'lat': lat,
                        'lon': lon,
                        'suggested_type': 'regional',  # Transfer hubs are typically regional+
                        'placement_reason': 'transfer_hub',
                        'transfer_potential': transfer_potential
                    })
        
        return candidates
    
    def _score_station_candidate(self, candidate: Dict, route_points: List[Tuple[float, float]], 
                               population_data: List[Dict] = None) -> StationPlacement:
        """Score a station candidate using learned patterns and ML models"""
        
        # Extract features for ML prediction
        features = self._extract_candidate_features(candidate, route_points, population_data)
        
        if self.trained and len(features) == 12:  # Ensure feature vector is complete
            try:
                features_scaled = self.scaler.transform([features])
                
                # Predict station type
                predicted_types = self.placement_classifier.predict_proba(features_scaled)[0]
                type_classes = self.placement_classifier.classes_
                
                # Get most likely type and confidence
                best_type_idx = np.argmax(predicted_types)
                station_type = type_classes[best_type_idx]
                confidence = predicted_types[best_type_idx]
                
            except Exception as e:
                print(f"ML prediction failed: {e}")
                station_type = candidate.get('suggested_type', 'regional')
                confidence = 0.5
        else:
            station_type = candidate.get('suggested_type', 'regional')
            confidence = self._heuristic_confidence(candidate, route_points)
        
        # Calculate additional metrics
        population_served = candidate.get('population_served', 
                                        self._estimate_population_radius(candidate['lat'], candidate['lon'], 3.0, population_data))
        
        transfer_potential = candidate.get('transfer_potential', 
                                         self._estimate_transfer_potential(candidate['lat'], candidate['lon']))
        
        accessibility_score = self._calculate_accessibility_score(candidate)
        
        reasons = self._generate_placement_reasons(candidate, features if 'features' in locals() else None)
        
        return StationPlacement(
            lat=candidate['lat'],
            lon=candidate['lon'],
            station_type=station_type,
            confidence=confidence,
            population_served=int(population_served),
            transfer_potential=transfer_potential,
            accessibility_score=accessibility_score,
            reasons=reasons
        )
    
    def _extract_candidate_features(self, candidate: Dict, route_points: List[Tuple[float, float]], 
                                  population_data: List[Dict] = None) -> List[float]:
        """Extract features for candidate location"""
        lat, lon = candidate['lat'], candidate['lon']
        
        # Use same feature extraction as training
        pop_1km = self._estimate_population_radius(lat, lon, 1.0, population_data)
        pop_5km = self._estimate_population_radius(lat, lon, 5.0, population_data)
        city_distance = self._estimate_city_center_distance(lat, lon, [])  # Simplified
        elevation = 100  # Would use real elevation data
        slope = self._estimate_terrain_slope(lat, lon)
        nearest_station_dist = 5.0  # Simplified - would use actual network data
        station_density = 0.3  # Simplified
        road_connectivity = self._estimate_road_connectivity(lat, lon)
        water_proximity = self._estimate_water_proximity(lat, lon)
        industrial_proximity = self._estimate_industrial_proximity(lat, lon)
        residential_density = self._estimate_residential_density(lat, lon)
        commercial_density = self._estimate_commercial_density(lat, lon)
        
        return [
            pop_1km, pop_5km, city_distance, elevation, slope,
            nearest_station_dist, station_density, road_connectivity,
            water_proximity, industrial_proximity, residential_density, commercial_density
        ]
    
    def _optimize_station_network(self, candidates: List[StationPlacement], 
                                constraints: Dict = None) -> List[StationPlacement]:
        """Optimize the station network to avoid redundancy and ensure coverage"""
        if not candidates:
            return []
        
        # Sort by confidence
        sorted_candidates = sorted(candidates, key=lambda x: x.confidence, reverse=True)
        
        # Apply spacing constraints
        optimized = []
        min_spacing = constraints.get('min_station_spacing_km', 3.0) if constraints else 3.0
        max_stations = constraints.get('max_stations', len(candidates)) if constraints else len(candidates)
        
        for candidate in sorted_candidates:
            # Check spacing constraints
            too_close = False
            for existing in optimized:
                distance = self._haversine_distance(
                    candidate.lat, candidate.lon,
                    existing.lat, existing.lon
                )
                if distance < min_spacing:
                    too_close = True
                    break
            
            if not too_close:
                optimized.append(candidate)
            
            if len(optimized) >= max_stations:
                break
        
        # Ensure minimum coverage (add stations if gaps are too large)
        optimized = self._ensure_minimum_coverage(optimized, constraints)
        
        return optimized
    
    def _ensure_minimum_coverage(self, stations: List[StationPlacement], 
                               constraints: Dict = None) -> List[StationPlacement]:
        """Ensure minimum station coverage along route"""
        max_gap = constraints.get('max_station_gap_km', 20.0) if constraints else 20.0
        
        if len(stations) < 2:
            return stations
        
        # Sort stations by latitude (assuming north-south route)
        sorted_stations = sorted(stations, key=lambda x: x.lat)
        
        additional_stations = []
        for i in range(len(sorted_stations) - 1):
            station1 = sorted_stations[i]
            station2 = sorted_stations[i + 1]
            
            distance = self._haversine_distance(
                station1.lat, station1.lon,
                station2.lat, station2.lon
            )
            
            # If gap is too large, add intermediate station
            if distance > max_gap:
                # Place station at midpoint
                mid_lat = (station1.lat + station2.lat) / 2
                mid_lon = (station1.lon + station2.lon) / 2
                
                additional_stations.append(StationPlacement(
                    lat=mid_lat,
                    lon=mid_lon,
                    station_type='regional',
                    confidence=0.7,
                    population_served=5000,  # Estimated
                    transfer_potential=0.3,
                    accessibility_score=0.6,
                    reasons=['coverage_gap_filling']
                ))
        
        return stations + additional_stations
    
    # Helper methods for feature estimation (simplified implementations)
    
    def _estimate_population_radius(self, lat: float, lon: float, radius_km: float, 
                                  population_data: List[Dict] = None) -> float:
        """Estimate population within radius"""
        if population_data:
            total_pop = 0
            for pop_point in population_data:
                distance = self._haversine_distance(lat, lon, pop_point['lat'], pop_point['lon'])
                if distance <= radius_km:
                    total_pop += pop_point.get('population', 1000)
            return total_pop
        
        # Simplified estimation based on coordinates
        base_pop = 10000 * radius_km * radius_km  # Base density
        urban_factor = max(0.1, 1 - abs(lat - 50) * 0.1)  # Higher near lat 50 (central Europe)
        return base_pop * urban_factor
    
    def _estimate_city_center_distance(self, lat: float, lon: float, stations: List[Dict]) -> float:
        """Estimate distance to nearest major city"""
        # Simplified - in reality would use city database
        major_cities = [(50.8503, 4.3517), (52.3676, 4.9041), (48.8566, 2.3522)]  # Brussels, Amsterdam, Paris
        
        min_distance = float('inf')
        for city_lat, city_lon in major_cities:
            distance = self._haversine_distance(lat, lon, city_lat, city_lon)
            min_distance = min(min_distance, distance)
        
        return min_distance
    
    def _estimate_terrain_slope(self, lat: float, lon: float) -> float:
        """Estimate terrain slope at location"""
        # Simplified terrain model
        return abs((lat - 50) * 2) + abs((lon - 4) * 1.5)  # Higher slope away from central coordinates
    
    def _estimate_road_connectivity(self, lat: float, lon: float) -> float:
        """Estimate road network connectivity (0-1)"""
        # Simplified - higher connectivity near populated areas
        return min(1.0, 0.5 + abs(lat - 50) * 0.1 + abs(lon - 4) * 0.05)
    
    def _estimate_water_proximity(self, lat: float, lon: float) -> float:
        """Estimate proximity to water bodies (0-1)"""
        # Simplified - random but consistent based on coordinates
        return (abs(lat * lon) % 1.0)
    
    def _estimate_industrial_proximity(self, lat: float, lon: float) -> float:
        """Estimate proximity to industrial areas (0-1)"""
        return min(1.0, abs(lat * 0.1) + abs(lon * 0.15)) % 1.0
    
    def _estimate_residential_density(self, lat: float, lon: float) -> float:
        """Estimate residential density (0-1)"""
        return min(1.0, 0.7 - abs(lat - 50.5) * 0.2)
    
    def _estimate_commercial_density(self, lat: float, lon: float) -> float:
        """Estimate commercial density (0-1)"""
        return min(1.0, 0.6 - abs(lon - 4.0) * 0.15)
    
    def _estimate_transfer_potential(self, lat: float, lon: float) -> float:
        """Estimate potential for transfer connections"""
        # Higher potential near major transport corridors
        corridor_factor = 1.0 - min(1.0, abs(lat - 50.8) + abs(lon - 4.3)) * 0.5
        return max(0.2, corridor_factor)
    
    def _calculate_accessibility_score(self, candidate: Dict) -> float:
        """Calculate accessibility score for station location"""
        base_score = 0.6
        
        if candidate.get('placement_reason') == 'population_center':
            base_score += 0.2
        elif candidate.get('placement_reason') == 'transfer_hub':
            base_score += 0.3
        
        return min(1.0, base_score)
    
    def _generate_placement_reasons(self, candidate: Dict, features: List[float] = None) -> List[str]:
        """Generate human-readable reasons for station placement"""
        reasons = []
        
        placement_reason = candidate.get('placement_reason', 'optimal_spacing')
        
        if placement_reason == 'population_center':
            pop_served = candidate.get('population_served', 0)
            reasons.append(f"Serves population center ({pop_served:,} people within 5km)")
        elif placement_reason == 'transfer_hub':
            reasons.append("Strategic transfer/interchange location")
        elif placement_reason == 'optimal_spacing':
            reasons.append("Optimal spacing for network connectivity")
        
        if features and len(features) >= 7:
            if features[6] > 0.5:  # station_density
                reasons.append("High railway network density area")
            if features[1] > 50000:  # pop_5km
                reasons.append("High population density catchment")
        
        return reasons
    
    def _heuristic_station_placement(self, route_points: List[Tuple[float, float]], 
                                   population_data: List[Dict] = None,
                                   constraints: Dict = None) -> List[StationPlacement]:
        """Fallback heuristic placement when ML models aren't available"""
        placements = []
        
        # Simple regular spacing approach
        spacing_km = constraints.get('target_spacing_km', 12.0) if constraints else 12.0
        
        # Calculate route length and place stations
        route_length = 0
        for i in range(1, len(route_points)):
            route_length += self._haversine_distance(
                route_points[i-1][0], route_points[i-1][1],
                route_points[i][0], route_points[i][1]
            )
        
        num_stations = max(2, int(route_length / spacing_km))
        
        for i in range(num_stations):
            # Find point along route
            target_distance = (i + 1) * route_length / (num_stations + 1)
            current_distance = 0
            
            for j in range(1, len(route_points)):
                segment_length = self._haversine_distance(
                    route_points[j-1][0], route_points[j-1][1],
                    route_points[j][0], route_points[j][1]
                )
                
                if current_distance + segment_length >= target_distance:
                    # Interpolate position within segment
                    ratio = (target_distance - current_distance) / segment_length
                    lat = route_points[j-1][0] + ratio * (route_points[j][0] - route_points[j-1][0])
                    lon = route_points[j-1][1] + ratio * (route_points[j][1] - route_points[j-1][1])
                    
                    placements.append(StationPlacement(
                        lat=lat,
                        lon=lon,
                        station_type='regional',
                        confidence=0.6,
                        population_served=15000,
                        transfer_potential=0.4,
                        accessibility_score=0.6,
                        reasons=['heuristic_spacing']
                    ))
                    break
                
                current_distance += segment_length
        
        return placements
    
    def _heuristic_confidence(self, candidate: Dict, route_points: List[Tuple[float, float]]) -> float:
        """Calculate confidence score using heuristics"""
        base_confidence = 0.5
        
        if candidate.get('placement_reason') == 'population_center':
            pop = candidate.get('population_served', 0)
            if pop > 50000:
                base_confidence += 0.3
            elif pop > 20000:
                base_confidence += 0.2
            elif pop > 5000:
                base_confidence += 0.1
        
        elif candidate.get('placement_reason') == 'transfer_hub':
            base_confidence += 0.25
        
        return min(1.0, base_confidence)
    
    # Utility methods
    
    def _distance_to_nearest_station(self, station: Dict, all_stations: List[Dict]) -> float:
        """Calculate distance to nearest other station"""
        min_distance = float('inf')
        for other in all_stations:
            if other != station:
                distance = self._haversine_distance(
                    station['lat'], station['lon'],
                    other['lat'], other['lon']
                )
                min_distance = min(min_distance, distance)
        return min_distance if min_distance != float('inf') else 10.0
    
    def _calculate_local_station_density(self, station: Dict, all_stations: List[Dict], radius_km: float) -> float:
        """Calculate station density within radius"""
        count = 0
        for other in all_stations:
            if other != station:
                distance = self._haversine_distance(
                    station['lat'], station['lon'],
                    other['lat'], other['lon']
                )
                if distance <= radius_km:
                    count += 1
        
        area = math.pi * radius_km * radius_km
        return count / area
    
    def _calculate_cluster_area(self, stations: List[Dict]) -> float:
        """Calculate approximate area covered by station cluster"""
        if len(stations) < 3:
            return 1.0
        
        lats = [s['lat'] for s in stations]
        lons = [s['lon'] for s in stations]
        
        # Simple bounding box area
        lat_range = max(lats) - min(lats)
        lon_range = max(lons) - min(lons)
        
        return lat_range * lon_range * 111 * 111  # Approximate km²
    
    def _get_dominant_category(self, stations: List[Dict]) -> str:
        """Get most common station category in group"""
        categories = [s.get('category', 'regional') for s in stations]
        return Counter(categories).most_common(1)[0][0]
    
    def _analyze_station_density(self, stations: List[Dict]) -> None:
        """Analyze overall station density patterns"""
        if len(stations) < 5:
            return
        
        densities = []
        for station in stations:
            density = self._calculate_local_station_density(station, stations, radius_km=15)
            densities.append(density)
        
        self.learned_patterns['density_analysis'] = {
            'avg_density': np.mean(densities),
            'density_std': np.std(densities),
            'high_density_threshold': np.percentile(densities, 75),
            'low_density_threshold': np.percentile(densities, 25)
        }
    
    def _remove_duplicate_candidates(self, candidates: List[Dict], min_distance_km: float) -> List[Dict]:
        """Remove candidates that are too close to each other"""
        if len(candidates) <= 1:
            return candidates
        
        unique_candidates = [candidates[0]]
        
        for candidate in candidates[1:]:
            too_close = False
            for existing in unique_candidates:
                distance = self._haversine_distance(
                    candidate['lat'], candidate['lon'],
                    existing['lat'], existing['lon']
                )
                if distance < min_distance_km:
                    too_close = True
                    break
            
            if not too_close:
                unique_candidates.append(candidate)
        
        return unique_candidates
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))