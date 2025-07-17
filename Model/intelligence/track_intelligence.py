# File: model/intelligence/track_intelligence.py
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Tuple, Optional
import math
from dataclasses import dataclass
from enum import Enum
import heapq

class TrackType(Enum):
    SURFACE = "surface"
    ELEVATED = "elevated" 
    TUNNEL = "tunnel"
    BRIDGE = "bridge"
    CUTTING = "cutting"  # Cut through hills
    EMBANKMENT = "embankment"  # Raised earth

@dataclass
class TrackSegment:
    start_lat: float
    start_lon: float
    end_lat: float
    end_lon: float
    track_type: TrackType
    elevation_start: float
    elevation_end: float
    gradient_percent: float
    construction_cost: float
    engineering_complexity: float
    environmental_impact: float
    reasons: List[str]

@dataclass
class RouteOption:
    segments: List[TrackSegment]
    total_cost: float
    total_length_km: float
    max_gradient: float
    avg_gradient: float
    complexity_score: float
    environmental_score: float
    construction_time_months: int

class TrackIntelligence:
    def __init__(self):
        self.gradient_predictor = RandomForestRegressor(n_estimators=100, random_state=42)
        self.track_type_classifier = GradientBoostingClassifier(n_estimators=100, random_state=42)
        self.cost_predictor = RandomForestRegressor(n_estimators=80, random_state=42)
        self.scaler = StandardScaler()
        self.trained = False
        self.learned_patterns = {}
        
        # Engineering constants learned from real railways
        self.max_gradient_standards = {
            'high_speed': 2.5,    # ICE, TGV max 2.5%
            'intercity': 3.5,     # Standard intercity max 3.5%
            'regional': 4.0,      # Regional trains max 4.0%
            'freight': 2.0        # Freight trains max 2.0%
        }
        
        self.track_costs_per_km = {
            TrackType.SURFACE: 2_000_000,      # €2M/km baseline
            TrackType.ELEVATED: 8_000_000,     # €8M/km for viaducts
            TrackType.TUNNEL: 25_000_000,      # €25M/km for tunnels
            TrackType.BRIDGE: 12_000_000,      # €12M/km for bridges
            TrackType.CUTTING: 4_000_000,      # €4M/km for cuts
            TrackType.EMBANKMENT: 3_000_000    # €3M/km for fills
        }
    
    def learn_from_existing_tracks(self, track_data: List[Dict], terrain_data: List[Dict]) -> None:
        """Learn routing patterns from existing railway tracks"""
        if len(track_data) < 50:
            print("Warning: Limited track data for robust learning")
            
        # Analyze gradient management strategies
        self._analyze_gradient_strategies(track_data, terrain_data)
        
        # Learn track type selection patterns
        self._learn_track_type_patterns(track_data, terrain_data)
        
        # Analyze cost optimization patterns
        self._analyze_cost_patterns(track_data)
        
        # Learn curve and alignment preferences
        self._analyze_alignment_patterns(track_data)
        
        # Study environmental mitigation strategies
        self._analyze_environmental_patterns(track_data, terrain_data)
        
        self.trained = True
        print(f"Learned routing intelligence from {len(track_data)} track segments")
    
    def _analyze_gradient_strategies(self, tracks: List[Dict], terrain: List[Dict]) -> None:
        """Analyze how existing railways handle gradients and elevation changes"""
        gradient_strategies = {
            'tunnel_thresholds': [],
            'bridge_thresholds': [],
            'cutting_thresholds': [],
            'spiral_usage': [],
            'switchback_usage': []
        }
        
        for track in tracks:
            elevation_profile = track.get('elevation_profile', [])
            if len(elevation_profile) < 2:
                continue
                
            # Calculate gradients along track
            gradients = []
            for i in range(1, len(elevation_profile)):
                rise = elevation_profile[i]['elevation'] - elevation_profile[i-1]['elevation']
                run = elevation_profile[i]['distance'] - elevation_profile[i-1]['distance']
                if run > 0:
                    gradient = (rise / (run * 1000)) * 100  # Convert to percentage
                    gradients.append(abs(gradient))
            
            # Analyze engineering solutions used
            track_type = track.get('engineering_type', 'surface')
            terrain_difficulty = track.get('terrain_difficulty', 'moderate')
            max_gradient = max(gradients) if gradients else 0
            
            if track_type == 'tunnel' and max_gradient > 0:
                gradient_strategies['tunnel_thresholds'].append({
                    'gradient': max_gradient,
                    'elevation_change': max(elevation_profile, key=lambda x: x['elevation'])['elevation'] - 
                                      min(elevation_profile, key=lambda x: x['elevation'])['elevation'],
                    'terrain': terrain_difficulty
                })
            
            # Analyze curve strategies for gradient management
            curves = track.get('curve_analysis', {})
            if curves and max_gradient > 2.0:
                if curves.get('spiral_curves', 0) > 0:
                    gradient_strategies['spiral_usage'].append({
                        'gradient': max_gradient,
                        'curves': curves.get('spiral_curves', 0)
                    })
        
        self.learned_patterns['gradient_strategies'] = gradient_strategies
    
    def _learn_track_type_patterns(self, tracks: List[Dict], terrain: List[Dict]) -> None:
        """Learn when to use different track types based on terrain"""
        features = []
        track_types = []
        
        for track in tracks:
            # Extract terrain features
            feature_vector = self._extract_terrain_features(track, terrain)
            if len(feature_vector) == 10:  # Ensure complete feature vector
                features.append(feature_vector)
                track_types.append(track.get('engineering_type', 'surface'))
        
        if len(features) > 20:  # Minimum for training
            # Train track type classification model
            features_scaled = self.scaler.fit_transform(features)
            self.track_type_classifier.fit(features_scaled, track_types)
            
            # Store feature importance
            feature_names = [
                'elevation_change', 'max_slope', 'avg_slope', 'terrain_roughness',
                'water_crossings', 'urban_density', 'protected_areas', 'soil_stability',
                'seismic_risk', 'weather_severity'
            ]
            
            # Get feature importance (for tree-based models)
            if hasattr(self.track_type_classifier, 'feature_importances_'):
                importance = dict(zip(feature_names, self.track_type_classifier.feature_importances_))
                self.learned_patterns['track_type_importance'] = importance
    
    def _extract_terrain_features(self, track: Dict, terrain: List[Dict]) -> List[float]:
        """Extract terrain features for track type prediction"""
        elevation_profile = track.get('elevation_profile', [])
        
        if not elevation_profile:
            return []
        
        elevations = [p['elevation'] for p in elevation_profile]
        
        # Elevation change
        elevation_change = max(elevations) - min(elevations)
        
        # Slope calculations
        slopes = []
        for i in range(1, len(elevation_profile)):
            rise = elevation_profile[i]['elevation'] - elevation_profile[i-1]['elevation']
            run = (elevation_profile[i]['distance'] - elevation_profile[i-1]['distance']) * 1000
            if run > 0:
                slopes.append(abs(rise / run) * 100)
        
        max_slope = max(slopes) if slopes else 0
        avg_slope = np.mean(slopes) if slopes else 0
        
        # Terrain roughness (elevation variance)
        terrain_roughness = np.std(elevations) if len(elevations) > 1 else 0
        
        # Environmental factors (simplified)
        water_crossings = track.get('water_crossings', 0)
        urban_density = track.get('urban_density', 0.3)
        protected_areas = track.get('protected_areas', 0.1)
        soil_stability = track.get('soil_stability', 0.8)
        seismic_risk = track.get('seismic_risk', 0.2)
        weather_severity = track.get('weather_severity', 0.4)
        
        return [
            elevation_change, max_slope, avg_slope, terrain_roughness,
            water_crossings, urban_density, protected_areas, soil_stability,
            seismic_risk, weather_severity
        ]
    
    def _analyze_cost_patterns(self, tracks: List[Dict]) -> None:
        """Analyze cost optimization patterns from existing tracks"""
        cost_analysis = {
            'cost_per_km_by_type': {},
            'cost_multipliers': {},
            'optimization_strategies': []
        }
        
        for track_type in ['surface', 'elevated', 'tunnel', 'bridge']:
            type_tracks = [t for t in tracks if t.get('engineering_type') == track_type]
            if type_tracks:
                costs = [t.get('construction_cost_per_km', 0) for t in type_tracks if t.get('construction_cost_per_km', 0) > 0]
                if costs:
                    cost_analysis['cost_per_km_by_type'][track_type] = {
                        'median': np.median(costs),
                        'mean': np.mean(costs),
                        'std': np.std(costs),
                        'samples': len(costs)
                    }
        
        self.learned_patterns['cost_analysis'] = cost_analysis
    
    def _analyze_alignment_patterns(self, tracks: List[Dict]) -> None:
        """Analyze preferred track alignments and curve strategies"""
        alignment_patterns = {
            'preferred_curve_radius': {},
            'gradient_curve_relationship': [],
            'straight_section_lengths': []
        }
        
        for track in tracks:
            geometry = track.get('geometry', [])
            if len(geometry) > 3:
                # Analyze curve radii
                curves = self._detect_curves(geometry)
                if curves:
                    track_type = track.get('train_type', 'regional')
                    if track_type not in alignment_patterns['preferred_curve_radius']:
                        alignment_patterns['preferred_curve_radius'][track_type] = []
                    
                    for curve in curves:
                        alignment_patterns['preferred_curve_radius'][track_type].append(curve['radius'])
                
                # Analyze straight sections
                straight_sections = self._detect_straight_sections(geometry)
                alignment_patterns['straight_section_lengths'].extend([s['length'] for s in straight_sections])
        
        self.learned_patterns['alignment_patterns'] = alignment_patterns
    
    def _analyze_environmental_patterns(self, tracks: List[Dict], terrain: List[Dict]) -> None:
        """Analyze environmental mitigation strategies"""
        env_patterns = {
            'protected_area_strategies': [],
            'water_crossing_methods': [],
            'noise_mitigation': [],
            'wildlife_corridors': []
        }
        
        for track in tracks:
            # Environmental challenges and solutions
            if track.get('crosses_protected_area'):
                strategy = track.get('environmental_strategy', 'unknown')
                env_patterns['protected_area_strategies'].append({
                    'strategy': strategy,
                    'track_type': track.get('engineering_type', 'surface'),
                    'success_rating': track.get('environmental_success', 0.7)
                })
            
            # Water crossing analysis
            water_crossings = track.get('water_crossings', 0)
            if water_crossings > 0:
                crossing_method = track.get('water_crossing_method', 'bridge')
                env_patterns['water_crossing_methods'].append({
                    'method': crossing_method,
                    'water_body_size': track.get('water_body_size', 'medium'),
                    'environmental_impact': track.get('water_impact_score', 0.5)
                })
        
        self.learned_patterns['environmental_patterns'] = env_patterns
    
    def optimize_track_routing(self, 
                             start_point: Tuple[float, float],
                             end_point: Tuple[float, float],
                             elevation_data: List[Dict],
                             constraints: Dict = None) -> List[RouteOption]:
        """Generate optimized track routing options between two points"""
        
        # Generate multiple route alternatives
        route_alternatives = self._generate_route_alternatives(start_point, end_point, elevation_data)
        
        # Score and optimize each alternative
        optimized_routes = []
        for alternative in route_alternatives:
            optimized_route = self._optimize_single_route(alternative, elevation_data, constraints)
            if optimized_route:
                optimized_routes.append(optimized_route)
        
        # Rank routes by overall score
        ranked_routes = self._rank_route_options(optimized_routes, constraints)
        
        return ranked_routes[:3]  # Return top 3 options
    
    def _generate_route_alternatives(self, 
                                   start: Tuple[float, float], 
                                   end: Tuple[float, float],
                                   elevation_data: List[Dict]) -> List[List[Tuple[float, float]]]:
        """Generate multiple routing alternatives using different strategies"""
        alternatives = []
        
        # Strategy 1: Direct route (baseline)
        direct_route = self._generate_direct_route(start, end)
        alternatives.append(direct_route)
        
        # Strategy 2: Elevation-conscious route (follows contours)
        contour_route = self._generate_contour_following_route(start, end, elevation_data)
        alternatives.append(contour_route)
        
        # Strategy 3: Valley-following route (seeks lower elevations)
        valley_route = self._generate_valley_route(start, end, elevation_data)
        alternatives.append(valley_route)
        
        # Strategy 4: Ridge route (higher elevation, potentially straighter)
        ridge_route = self._generate_ridge_route(start, end, elevation_data)
        alternatives.append(ridge_route)
        
        # Strategy 5: Hybrid route (combines strategies based on terrain)
        hybrid_route = self._generate_hybrid_route(start, end, elevation_data)
        alternatives.append(hybrid_route)
        
        return alternatives
    
    def _generate_direct_route(self, start: Tuple[float, float], end: Tuple[float, float]) -> List[Tuple[float, float]]:
        """Generate direct route with intermediate waypoints"""
        points = []
        num_segments = 20  # Divide route into segments for analysis
        
        for i in range(num_segments + 1):
            t = i / num_segments
            lat = start[0] + t * (end[0] - start[0])
            lon = start[1] + t * (end[1] - start[1])
            points.append((lat, lon))
        
        return points
    
    def _generate_contour_following_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                                        elevation_data: List[Dict]) -> List[Tuple[float, float]]:
        """Generate route that follows elevation contours to minimize gradients"""
        # Simplified contour following - in reality would use proper topographic analysis
        points = [start]
        current = start
        
        steps = 15
        for i in range(steps):
            progress = (i + 1) / steps
            
            # Target position if going direct
            target_lat = start[0] + progress * (end[0] - start[0])
            target_lon = start[1] + progress * (end[1] - start[1])
            
            # Find path of least elevation change
            best_point = self._find_low_gradient_point(current, (target_lat, target_lon), elevation_data)
            points.append(best_point)
            current = best_point
        
        points.append(end)
        return points
    
    def _generate_valley_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                             elevation_data: List[Dict]) -> List[Tuple[float, float]]:
        """Generate route that seeks valleys and low elevations"""
        points = [start]
        current = start
        
        steps = 12
        for i in range(steps):
            progress = (i + 1) / steps
            
            # Target area
            search_center_lat = start[0] + progress * (end[0] - start[0])
            search_center_lon = start[1] + progress * (end[1] - start[1])
            
            # Find lowest elevation point in search area
            low_point = self._find_lowest_elevation_point(
                current, (search_center_lat, search_center_lon), elevation_data, search_radius_km=5
            )
            points.append(low_point)
            current = low_point
        
        points.append(end)
        return points
    
    def _generate_ridge_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                            elevation_data: List[Dict]) -> List[Tuple[float, float]]:
        """Generate route along ridges for potentially straighter alignment"""
        points = [start]
        current = start
        
        steps = 10
        for i in range(steps):
            progress = (i + 1) / steps
            
            # Target area
            search_center_lat = start[0] + progress * (end[0] - start[0])
            search_center_lon = start[1] + progress * (end[1] - start[1])
            
            # Find moderate elevation point (not too high, not too low)
            ridge_point = self._find_ridge_point(
                current, (search_center_lat, search_center_lon), elevation_data
            )
            points.append(ridge_point)
            current = ridge_point
        
        points.append(end)
        return points
    
    def _generate_hybrid_route(self, start: Tuple[float, float], end: Tuple[float, float], 
                             elevation_data: List[Dict]) -> List[Tuple[float, float]]:
        """Generate hybrid route using ML-learned patterns"""
        if not self.trained:
            return self._generate_direct_route(start, end)
        
        # Use learned patterns to adaptively choose routing strategy
        points = [start]
        current = start
        
        steps = 16
        for i in range(steps):
            progress = (i + 1) / steps
            target_lat = start[0] + progress * (end[0] - start[0])
            target_lon = start[1] + progress * (end[1] - start[1])
            
            # Analyze local terrain to choose strategy
            terrain_features = self._analyze_local_terrain(current, (target_lat, target_lon), elevation_data)
            
            # Choose routing strategy based on terrain
            if terrain_features['elevation_variance'] > 200:  # Mountainous
                next_point = self._find_tunnel_or_bridge_route(current, (target_lat, target_lon), elevation_data)
            elif terrain_features['water_present']:  # Water crossing
                next_point = self._find_bridge_route(current, (target_lat, target_lon), elevation_data)
            else:  # Normal terrain
                next_point = self._find_low_gradient_point(current, (target_lat, target_lon), elevation_data)
            
            points.append(next_point)
            current = next_point
        
        points.append(end)
        return points
    
    def _optimize_single_route(self, route_points: List[Tuple[float, float]], 
                             elevation_data: List[Dict], constraints: Dict = None) -> Optional[RouteOption]:
        """Optimize a single route option by determining track types and costs"""
        
        if len(route_points) < 2:
            return None
        
        segments = []
        total_cost = 0
        total_length = 0
        gradients = []
        
        for i in range(len(route_points) - 1):
            start_point = route_points[i]
            end_point = route_points[i + 1]
            
            # Get elevation data for segment
            start_elevation = self._get_elevation_at_point(start_point, elevation_data)
            end_elevation = self._get_elevation_at_point(end_point, elevation_data)
            
            # Calculate segment metrics
            segment_length = self._haversine_distance(
                start_point[0], start_point[1], end_point[0], end_point[1]
            )
            
            gradient = self._calculate_gradient(start_elevation, end_elevation, segment_length)
            gradients.append(abs(gradient))
            
            # Determine optimal track type for segment
            track_type = self._determine_track_type(
                start_point, end_point, start_elevation, end_elevation, gradient, constraints
            )
            
            # Calculate costs
            segment_cost = self._calculate_segment_cost(segment_length, track_type, gradient, constraints)
            
            # Create segment
            segment = TrackSegment(
                start_lat=start_point[0],
                start_lon=start_point[1],
                end_lat=end_point[0],
                end_lon=end_point[1],
                track_type=track_type,
                elevation_start=start_elevation,
                elevation_end=end_elevation,
                gradient_percent=gradient,
                construction_cost=segment_cost,
                engineering_complexity=self._calculate_complexity(track_type, gradient),
                environmental_impact=self._calculate_environmental_impact(track_type, start_point),
                reasons=self._generate_segment_reasons(track_type, gradient, start_elevation, end_elevation)
            )
            
            segments.append(segment)
            total_cost += segment_cost
            total_length += segment_length
        
        # Calculate route-level metrics
        max_gradient = max(gradients) if gradients else 0
        avg_gradient = np.mean(gradients) if gradients else 0
        complexity_score = np.mean([s.engineering_complexity for s in segments])
        environmental_score = np.mean([s.environmental_impact for s in segments])
        construction_time = self._estimate_construction_time(segments)
        
        return RouteOption(
            segments=segments,
            total_cost=total_cost,
            total_length_km=total_length,
            max_gradient=max_gradient,
            avg_gradient=avg_gradient,
            complexity_score=complexity_score,
            environmental_score=environmental_score,
            construction_time_months=construction_time
        )
    
    def _determine_track_type(self, start_point: Tuple[float, float], end_point: Tuple[float, float],
                            start_elevation: float, end_elevation: float, gradient: float,
                            constraints: Dict = None) -> TrackType:
        """Determine optimal track type for segment"""
        
        elevation_change = abs(end_elevation - start_elevation)
        max_elevation = max(start_elevation, end_elevation)
        
        # Apply learned patterns if available
        if self.trained:
            terrain_features = [
                elevation_change, abs(gradient), abs(gradient), 100,  # terrain roughness estimate
                0, 0.3, 0.1, 0.8, 0.2, 0.4  # environmental factors (simplified)
            ]
            
            try:
                features_scaled = self.scaler.transform([terrain_features])
                predicted_type = self.track_type_classifier.predict(features_scaled)[0]
                
                # Map prediction to TrackType enum
                type_mapping = {
                    'surface': TrackType.SURFACE,
                    'elevated': TrackType.ELEVATED,
                    'tunnel': TrackType.TUNNEL,
                    'bridge': TrackType.BRIDGE,
                    'cutting': TrackType.CUTTING,
                    'embankment': TrackType.EMBANKMENT
                }
                
                if predicted_type in type_mapping:
                    return type_mapping[predicted_type]
                    
            except Exception as e:
                print(f"ML prediction failed: {e}")
        
        # Fallback to engineering rules
        train_type = constraints.get('train_type', 'regional') if constraints else 'regional'
        max_allowed_gradient = self.max_gradient_standards.get(train_type, 3.5)
        
        # Tunnel for very steep terrain or high mountains
        if abs(gradient) > max_allowed_gradient * 1.5 or max_elevation > 1500:
            return TrackType.TUNNEL
        
        # Bridge for moderate elevation but steep gradient (valley crossing)
        if abs(gradient) > max_allowed_gradient and elevation_change > 100:
            return TrackType.BRIDGE
        
        # Elevated for urban areas or moderate obstacles
        if elevation_change > 50 and abs(gradient) > max_allowed_gradient * 0.8:
            return TrackType.ELEVATED
        
        # Cutting for moderate hills
        if start_elevation > end_elevation and elevation_change > 30:
            return TrackType.CUTTING
        
        # Embankment for filling valleys
        if end_elevation > start_elevation and elevation_change > 30:
            return TrackType.EMBANKMENT
        
        # Default to surface
        return TrackType.SURFACE
    
    def _calculate_segment_cost(self, length_km: float, track_type: TrackType, 
                              gradient: float, constraints: Dict = None) -> float:
        """Calculate construction cost for track segment"""
        
        base_cost = self.track_costs_per_km[track_type] * length_km
        
        # Gradient penalty
        if abs(gradient) > 2.0:
            gradient_multiplier = 1 + (abs(gradient) - 2.0) * 0.2
            base_cost *= gradient_multiplier
        
        # Regional cost factors
        region_factor = constraints.get('regional_cost_factor', 1.0) if constraints else 1.0
        base_cost *= region_factor
        
        # Complexity multipliers
        if track_type == TrackType.TUNNEL:
            base_cost *= (1 + abs(gradient) * 0.1)  # Steeper tunnels cost more
        elif track_type == TrackType.BRIDGE:
            base_cost *= (1 + length_km * 0.05)  # Longer bridges cost more per km
        
        return base_cost
    
    def _calculate_complexity(self, track_type: TrackType, gradient: float) -> float:
        """Calculate engineering complexity score (0-1)"""
        base_complexity = {
            TrackType.SURFACE: 0.2,
            TrackType.ELEVATED: 0.5,
            TrackType.TUNNEL: 0.9,
            TrackType.BRIDGE: 0.7,
            TrackType.CUTTING: 0.4,
            TrackType.EMBANKMENT: 0.3
        }
        
        complexity = base_complexity[track_type]
        
        # Add gradient complexity
        complexity += min(0.3, abs(gradient) * 0.05)
        
        return min(1.0, complexity)
    
    def _calculate_environmental_impact(self, track_type: TrackType, location: Tuple[float, float]) -> float:
        """Calculate environmental impact score (0-1)"""
        base_impact = {
            TrackType.SURFACE: 0.6,
            TrackType.ELEVATED: 0.4,  # Less ground disturbance
            TrackType.TUNNEL: 0.3,    # Minimal surface impact
            TrackType.BRIDGE: 0.5,
            TrackType.CUTTING: 0.8,   # High excavation impact
            TrackType.EMBANKMENT: 0.7  # Significant filling impact
        }
        
        impact = base_impact[track_type]
        
        # Add location-specific factors (simplified)
        # In reality would use detailed environmental databases
        if self._is_near_water(location):
            impact += 0.2
        if self._is_protected_area(location):
            impact += 0.3
        
        return min(1.0, impact)
    
    def _generate_segment_reasons(self, track_type: TrackType, gradient: float, 
                                start_elevation: float, end_elevation: float) -> List[str]:
        """Generate human-readable reasons for track type selection"""
        reasons = []
        
        elevation_change = abs(end_elevation - start_elevation)
        
        if track_type == TrackType.TUNNEL:
            if abs(gradient) > 4.0:
                reasons.append(f"Tunnel required: gradient {gradient:.1f}% exceeds surface limits")
            if max(start_elevation, end_elevation) > 1500:
                reasons.append("Tunnel required: high mountain crossing")
            if elevation_change > 300:
                reasons.append(f"Tunnel optimal: {elevation_change:.0f}m elevation change")
                
        elif track_type == TrackType.BRIDGE:
            if elevation_change > 100:
                reasons.append(f"Bridge required: {elevation_change:.0f}m valley crossing")
            if abs(gradient) > 3.0:
                reasons.append(f"Bridge optimal: steep approach {gradient:.1f}%")
                
        elif track_type == TrackType.ELEVATED:
            reasons.append("Elevated track: moderate obstacles or urban area")
            
        elif track_type == TrackType.CUTTING:
            reasons.append(f"Cut through terrain: {elevation_change:.0f}m descent")
            
        elif track_type == TrackType.EMBANKMENT:
            reasons.append(f"Fill terrain: {elevation_change:.0f}m elevation gain")
            
        else:  # SURFACE
            reasons.append("Surface track: favorable terrain conditions")
        
        return reasons
    
    def _rank_route_options(self, routes: List[RouteOption], constraints: Dict = None) -> List[RouteOption]:
        """Rank route options by overall suitability score"""
        
        scored_routes = []
        
        for route in routes:
            score = self._calculate_route_score(route, constraints)
            route.overall_score = score
            scored_routes.append(route)
        
        return sorted(scored_routes, key=lambda x: x.overall_score, reverse=True)
    
    def _calculate_route_score(self, route: RouteOption, constraints: Dict = None) -> float:
        """Calculate overall route suitability score"""
        
        # Normalize factors (0-1, higher is better)
        
        # Cost efficiency (lower cost = higher score)
        max_reasonable_cost = route.total_length_km * 10_000_000  # €10M/km benchmark
        cost_score = max(0, 1 - (route.total_cost / max_reasonable_cost))
        
        # Gradient efficiency (lower gradients = higher score)
        max_allowed_gradient = constraints.get('max_gradient', 3.5) if constraints else 3.5
        gradient_score = max(0, 1 - (route.max_gradient / (max_allowed_gradient * 2)))
        
        # Construction complexity (lower complexity = higher score)
        complexity_score = 1 - route.complexity_score
        
        # Environmental impact (lower impact = higher score)
        environmental_score = 1 - route.environmental_score
        
        # Construction time (shorter time = higher score)
        max_reasonable_time = route.total_length_km * 6  # 6 months per km benchmark
        time_score = max(0, 1 - (route.construction_time_months / max_reasonable_time))
        
        # Length efficiency (shorter routes preferred, but not at expense of other factors)
        direct_distance = self._calculate_direct_distance(route)
        length_efficiency = direct_distance / route.total_length_km if route.total_length_km > 0 else 0
        
        # Weighted scoring
        weights = constraints.get('scoring_weights', {}) if constraints else {}
        
        total_score = (
            weights.get('cost', 0.25) * cost_score +
            weights.get('gradient', 0.20) * gradient_score +
            weights.get('complexity', 0.15) * complexity_score +
            weights.get('environmental', 0.15) * environmental_score +
            weights.get('time', 0.15) * time_score +
            weights.get('length', 0.10) * length_efficiency
        )
        
        # If no weights specified, use default weighting
        if not weights:
            total_score = (
                0.25 * cost_score +
                0.20 * gradient_score +
                0.15 * complexity_score +
                0.15 * environmental_score +
                0.15 * time_score +
                0.10 * length_efficiency
            )
        
        return total_score
    
    def _calculate_direct_distance(self, route: RouteOption) -> float:
        """Calculate direct distance for route efficiency calculation"""
        if not route.segments:
            return 0
        
        start = route.segments[0]
        end = route.segments[-1]
        
        return self._haversine_distance(
            start.start_lat, start.start_lon,
            end.end_lat, end.end_lon
        )
    
    def _estimate_construction_time(self, segments: List[TrackSegment]) -> int:
        """Estimate construction time in months"""
        total_months = 0
        
        construction_rates = {  # km per month
            TrackType.SURFACE: 2.0,
            TrackType.ELEVATED: 0.5,
            TrackType.TUNNEL: 0.2,
            TrackType.BRIDGE: 0.8,
            TrackType.CUTTING: 1.0,
            TrackType.EMBANKMENT: 1.5
        }
        
        for segment in segments:
            length = self._haversine_distance(
                segment.start_lat, segment.start_lon,
                segment.end_lat, segment.end_lon
            )
            
            rate = construction_rates[segment.track_type]
            segment_time = length / rate
            
            # Add complexity penalties
            if abs(segment.gradient_percent) > 3.0:
                segment_time *= 1.3
            
            total_months += segment_time
        
        # Add project overhead (30% for planning, mobilization, etc.)
        total_months *= 1.3
        
        return int(total_months)
    
    # Helper methods for terrain analysis
    
    def _find_low_gradient_point(self, current: Tuple[float, float], 
                               target: Tuple[float, float], 
                               elevation_data: List[Dict]) -> Tuple[float, float]:
        """Find point between current and target with lowest gradient"""
        
        # Sample points in search area
        search_points = self._generate_search_points(current, target, num_points=9)
        
        best_point = target
        best_gradient = float('inf')
        
        current_elevation = self._get_elevation_at_point(current, elevation_data)
        
        for point in search_points:
            point_elevation = self._get_elevation_at_point(point, elevation_data)
            distance = self._haversine_distance(current[0], current[1], point[0], point[1])
            
            if distance > 0:
                gradient = abs((point_elevation - current_elevation) / (distance * 1000)) * 100
                if gradient < best_gradient:
                    best_gradient = gradient
                    best_point = point
        
        return best_point
    
    def _find_lowest_elevation_point(self, current: Tuple[float, float], 
                                   center: Tuple[float, float], 
                                   elevation_data: List[Dict],
                                   search_radius_km: float = 5) -> Tuple[float, float]:
        """Find lowest elevation point in search area"""
        
        search_points = self._generate_circular_search_points(center, search_radius_km, num_points=16)
        
        best_point = center
        lowest_elevation = float('inf')
        
        for point in search_points:
            elevation = self._get_elevation_at_point(point, elevation_data)
            if elevation < lowest_elevation:
                lowest_elevation = elevation
                best_point = point
        
        return best_point
    
    def _find_ridge_point(self, current: Tuple[float, float], 
                         center: Tuple[float, float], 
                         elevation_data: List[Dict]) -> Tuple[float, float]:
        """Find moderate elevation ridge point"""
        
        search_points = self._generate_search_points(current, center, num_points=12)
        
        elevations = []
        for point in search_points:
            elevation = self._get_elevation_at_point(point, elevation_data)
            elevations.append((point, elevation))
        
        # Find point with elevation in 60th-80th percentile (ridge but not peak)
        elevations.sort(key=lambda x: x[1])
        target_idx = int(len(elevations) * 0.7)  # 70th percentile
        
        return elevations[target_idx][0] if elevations else center
    
    def _find_tunnel_or_bridge_route(self, current: Tuple[float, float], 
                                   target: Tuple[float, float], 
                                   elevation_data: List[Dict]) -> Tuple[float, float]:
        """Find optimal point for tunnel or bridge routing"""
        
        # For mountainous terrain, prefer more direct routing (tunnels/bridges handle elevation)
        # Move 80% toward target
        progress = 0.8
        result_lat = current[0] + progress * (target[0] - current[0])
        result_lon = current[1] + progress * (target[1] - current[1])
        
        return (result_lat, result_lon)
    
    def _find_bridge_route(self, current: Tuple[float, float], 
                          target: Tuple[float, float], 
                          elevation_data: List[Dict]) -> Tuple[float, float]:
        """Find optimal bridge crossing point"""
        
        # For water crossings, find narrowest crossing point
        search_points = self._generate_search_points(current, target, num_points=7)
        
        # Simplified: choose point closest to straight line
        best_point = target
        min_deviation = float('inf')
        
        for point in search_points:
            # Calculate deviation from straight line
            deviation = self._point_to_line_distance(point, current, target)
            if deviation < min_deviation:
                min_deviation = deviation
                best_point = point
        
        return best_point
    
    def _analyze_local_terrain(self, current: Tuple[float, float], 
                             target: Tuple[float, float], 
                             elevation_data: List[Dict]) -> Dict:
        """Analyze terrain characteristics in local area"""
        
        search_points = self._generate_search_points(current, target, num_points=16)
        elevations = [self._get_elevation_at_point(p, elevation_data) for p in search_points]
        
        return {
            'elevation_variance': np.var(elevations) if elevations else 0,
            'max_elevation': max(elevations) if elevations else 0,
            'min_elevation': min(elevations) if elevations else 0,
            'avg_elevation': np.mean(elevations) if elevations else 0,
            'water_present': self._check_water_presence(current, target),
            'urban_density': self._estimate_urban_density(current, target)
        }
    
    def _generate_search_points(self, start: Tuple[float, float], 
                              end: Tuple[float, float], 
                              num_points: int = 9) -> List[Tuple[float, float]]:
        """Generate search points in area between start and end"""
        
        points = []
        
        # Create grid around the direct line
        for i in range(3):
            for j in range(3):
                if num_points <= len(points):
                    break
                    
                # Progress along line (0.2, 0.5, 0.8)
                t = (i + 1) * 0.25 + 0.25
                
                # Offset perpendicular to line (-1, 0, +1 km)
                offset_km = (j - 1) * 1.0
                
                # Calculate point along line
                base_lat = start[0] + t * (end[0] - start[0])
                base_lon = start[1] + t * (end[1] - start[1])
                
                # Add perpendicular offset
                bearing = self._calculate_bearing(start, end) + 90  # Perpendicular
                offset_lat, offset_lon = self._offset_coordinates(base_lat, base_lon, offset_km, bearing)
                
                points.append((offset_lat, offset_lon))
        
        return points[:num_points]
    
    def _generate_circular_search_points(self, center: Tuple[float, float], 
                                       radius_km: float, 
                                       num_points: int = 16) -> List[Tuple[float, float]]:
        """Generate points in circle around center"""
        
        points = []
        angle_step = 360 / num_points
        
        for i in range(num_points):
            bearing = i * angle_step
            lat, lon = self._offset_coordinates(center[0], center[1], radius_km, bearing)
            points.append((lat, lon))
        
        return points
    
    def _get_elevation_at_point(self, point: Tuple[float, float], elevation_data: List[Dict]) -> float:
        """Get elevation at specific point from elevation data"""
        
        lat, lon = point
        
        # Find closest elevation data point
        min_distance = float('inf')
        closest_elevation = 100  # Default elevation
        
        for data_point in elevation_data:
            distance = self._haversine_distance(
                lat, lon, data_point['lat'], data_point['lon']
            )
            if distance < min_distance:
                min_distance = distance
                closest_elevation = data_point['elevation']
        
        return closest_elevation
    
    def _calculate_gradient(self, start_elevation: float, end_elevation: float, distance_km: float) -> float:
        """Calculate gradient percentage"""
        
        if distance_km == 0:
            return 0
        
        rise = end_elevation - start_elevation
        run = distance_km * 1000  # Convert to meters
        
        return (rise / run) * 100
    
    def _calculate_bearing(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Calculate bearing from start to end point in degrees"""
        
        lat1, lon1 = math.radians(start[0]), math.radians(start[1])
        lat2, lon2 = math.radians(end[0]), math.radians(end[1])
        
        dlon = lon2 - lon1
        
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        bearing = math.degrees(bearing)
        bearing = (bearing + 360) % 360
        
        return bearing
    
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
    
    def _point_to_line_distance(self, point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> float:
        """Calculate distance from point to line"""
        
        # Simplified distance calculation
        # In reality would use proper geodesic calculations
        
        px, py = point[1], point[0]  # lon, lat
        x1, y1 = line_start[1], line_start[0]
        x2, y2 = line_end[1], line_end[0]
        
        # Line equation: ax + by + c = 0
        a = y2 - y1
        b = x1 - x2
        c = x2 * y1 - x1 * y2
        
        if a == 0 and b == 0:
            return 0
        
        distance = abs(a * px + b * py + c) / math.sqrt(a * a + b * b)
        
        # Convert to km (approximate)
        return distance * 111  # Rough conversion for degrees to km
    
    def _detect_curves(self, geometry: List[Dict]) -> List[Dict]:
        """Detect curves in track geometry"""
        
        curves = []
        
        if len(geometry) < 3:
            return curves
        
        for i in range(1, len(geometry) - 1):
            # Calculate bearing changes
            bearing1 = self._calculate_bearing(
                (geometry[i-1]['lat'], geometry[i-1]['lon']),
                (geometry[i]['lat'], geometry[i]['lon'])
            )
            bearing2 = self._calculate_bearing(
                (geometry[i]['lat'], geometry[i]['lon']),
                (geometry[i+1]['lat'], geometry[i+1]['lon'])
            )
            
            bearing_change = abs(bearing2 - bearing1)
            if bearing_change > 180:
                bearing_change = 360 - bearing_change
            
            # If bearing change > 5 degrees, consider it a curve
            if bearing_change > 5:
                # Estimate curve radius (simplified)
                chord_length = self._haversine_distance(
                    geometry[i-1]['lat'], geometry[i-1]['lon'],
                    geometry[i+1]['lat'], geometry[i+1]['lon']
                )
                
                if bearing_change > 0:
                    radius = (chord_length * 1000) / (2 * math.sin(math.radians(bearing_change / 2)))
                    
                    curves.append({
                        'location': (geometry[i]['lat'], geometry[i]['lon']),
                        'radius': radius,
                        'bearing_change': bearing_change
                    })
        
        return curves
    
    def _detect_straight_sections(self, geometry: List[Dict]) -> List[Dict]:
        """Detect straight sections in track geometry"""
        
        straight_sections = []
        current_section_start = 0
        
        if len(geometry) < 3:
            return straight_sections
        
        for i in range(1, len(geometry) - 1):
            bearing1 = self._calculate_bearing(
                (geometry[i-1]['lat'], geometry[i-1]['lon']),
                (geometry[i]['lat'], geometry[i]['lon'])
            )
            bearing2 = self._calculate_bearing(
                (geometry[i]['lat'], geometry[i]['lon']),
                (geometry[i+1]['lat'], geometry[i+1]['lon'])
            )
            
            bearing_change = abs(bearing2 - bearing1)
            if bearing_change > 180:
                bearing_change = 360 - bearing_change
            
            # If bearing change > 3 degrees, end current straight section
            if bearing_change > 3:
                if i - current_section_start > 2:  # Minimum 3 points for a section
                    section_length = 0
                    for j in range(current_section_start, i):
                        section_length += self._haversine_distance(
                            geometry[j]['lat'], geometry[j]['lon'],
                            geometry[j+1]['lat'], geometry[j+1]['lon']
                        )
                    
                    straight_sections.append({
                        'start': current_section_start,
                        'end': i,
                        'length': section_length
                    })
                
                current_section_start = i
        
        return straight_sections
    
    def _is_near_water(self, location: Tuple[float, float]) -> bool:
        """Check if location is near water body (simplified)"""
        # Simplified water detection
        # In reality would use water body databases
        lat, lon = location
        
        # Some European rivers/coasts approximation
        if abs(lat - 51.5) < 0.1 and abs(lon - 0.1) < 0.2:  # Thames
            return True
        if abs(lat - 48.8) < 0.1 and abs(lon - 2.3) < 0.2:  # Seine
            return True
        
        return False
    
    def _is_protected_area(self, location: Tuple[float, float]) -> bool:
        """Check if location is in protected area (simplified)"""
        # Simplified protected area detection
        # In reality would use protected area databases
        lat, lon = location
        
        # Some protected areas approximation
        if 46 < lat < 47 and 7 < lon < 8:  # Swiss Alps area
            return True
        
        return False
    
    def _check_water_presence(self, start: Tuple[float, float], end: Tuple[float, float]) -> bool:
        """Check if water crossing is required between points"""
        return self._is_near_water(start) or self._is_near_water(end)
    
    def _estimate_urban_density(self, start: Tuple[float, float], end: Tuple[float, float]) -> float:
        """Estimate urban density in area (0-1)"""
        # Simplified urban density estimation
        center_lat = (start[0] + end[0]) / 2
        center_lon = (start[1] + end[1]) / 2
        
        # Higher density near major cities
        major_cities = [
            (50.8503, 4.3517),  # Brussels
            (52.3676, 4.9041),  # Amsterdam
            (48.8566, 2.3522),  # Paris
            (51.5074, -0.1278), # London
            (50.1109, 8.6821)   # Frankfurt
        ]
        
        min_distance = min([
            self._haversine_distance(center_lat, center_lon, city[0], city[1])
            for city in major_cities
        ])
        
        # Urban density decreases with distance from cities
        if min_distance < 10:
            return 0.9
        elif min_distance < 50:
            return 0.6
        elif min_distance < 100:
            return 0.3
        else:
            return 0.1
    
    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate haversine distance in km"""
        R = 6371
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        
        a = (math.sin(dlat/2)**2 + 
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * 
             math.sin(dlon/2)**2)
        
        return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    
    # Public interface methods
    
    def get_learned_patterns(self) -> Dict:
        """Get summary of learned patterns"""
        return self.learned_patterns
    
    def validate_route_feasibility(self, route: RouteOption, train_type: str = 'regional') -> Dict:
        """Validate if route is feasible for given train type"""
        
        max_allowed_gradient = self.max_gradient_standards.get(train_type, 3.5)
        
        feasibility = {
            'feasible': True,
            'issues': [],
            'recommendations': []
        }
        
        # Check gradient compliance
        if route.max_gradient > max_allowed_gradient:
            feasibility['feasible'] = False
            feasibility['issues'].append(
                f"Maximum gradient {route.max_gradient:.1f}% exceeds {train_type} limit of {max_allowed_gradient}%"
            )
            feasibility['recommendations'].append("Consider additional tunneling or more gradual routing")
        
        # Check curve radii (simplified)
        min_curve_radius = {
            'high_speed': 3000,  # meters
            'intercity': 1500,
            'regional': 800,
            'freight': 600
        }
        
        required_radius = min_curve_radius.get(train_type, 800)
        
        # Cost feasibility
        cost_per_km = route.total_cost / route.total_length_km if route.total_length_km > 0 else 0
        if cost_per_km > 15_000_000:  # €15M/km threshold
            feasibility['issues'].append(f"High construction cost: €{cost_per_km/1_000_000:.1f}M per km")
            feasibility['recommendations'].append("Consider alternative routing to reduce tunneling/bridging")
        
        # Construction time
        if route.construction_time_months > route.total_length_km * 8:  # 8 months per km threshold
            feasibility['issues'].append(f"Extended construction time: {route.construction_time_months} months")
            feasibility['recommendations'].append("Consider phased construction or simpler engineering solutions")
        
        return feasibility