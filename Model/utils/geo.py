# File: railway_ai/utils/geo.py
import math
import numpy as np
from typing import Tuple, List, Dict, Optional, Union
from dataclasses import dataclass
from enum import Enum

class CoordinateSystem(Enum):
    WGS84 = "WGS84"        # Standard GPS coordinates
    UTM = "UTM"            # Universal Transverse Mercator
    EPSG3857 = "EPSG3857"  # Web Mercator (for mapping)

@dataclass
class BoundingBox:
    west: float    # Min longitude
    south: float   # Min latitude
    east: float    # Max longitude
    north: float   # Max latitude
    
    def contains(self, lat: float, lon: float) -> bool:
        """Check if point is within bounding box"""
        return (self.south <= lat <= self.north and 
                self.west <= lon <= self.east)
    
    def expand(self, margin_km: float) -> 'BoundingBox':
        """Expand bounding box by margin in kilometers"""
        lat_margin = km_to_degrees_lat(margin_km)
        lon_margin = km_to_degrees_lon(margin_km, (self.north + self.south) / 2)
        
        return BoundingBox(
            west=self.west - lon_margin,
            south=self.south - lat_margin,
            east=self.east + lon_margin,
            north=self.north + lat_margin
        )
    
    def area_km2(self) -> float:
        """Calculate area of bounding box in km²"""
        width_km = haversine_distance(
            self.south, self.west, self.south, self.east
        )
        height_km = haversine_distance(
            self.south, self.west, self.north, self.west
        )
        return width_km * height_km

@dataclass
class GeoPoint:
    lat: float
    lon: float
    elevation: Optional[float] = None
    name: Optional[str] = None
    
    def distance_to(self, other: 'GeoPoint') -> float:
        """Calculate distance to another point in km"""
        return haversine_distance(self.lat, self.lon, other.lat, other.lon)
    
    def bearing_to(self, other: 'GeoPoint') -> float:
        """Calculate bearing to another point in degrees"""
        return calculate_bearing(self.lat, self.lon, other.lat, other.lon)
    
    def offset(self, distance_km: float, bearing_degrees: float) -> 'GeoPoint':
        """Create new point offset by distance and bearing"""
        new_lat, new_lon = offset_coordinates(
            self.lat, self.lon, distance_km, bearing_degrees
        )
        return GeoPoint(new_lat, new_lon, self.elevation)

class GeoUtils:
    """Collection of geospatial utility functions for railway planning"""
    
    # Earth parameters
    EARTH_RADIUS_KM = 6371.0
    EARTH_RADIUS_M = 6371000.0
    
    # Coordinate system constants
    WGS84_A = 6378137.0        # Semi-major axis
    WGS84_F = 1/298.257223563  # Flattening
    
    @staticmethod
    def create_bounding_box(points: List[Tuple[float, float]], margin_km: float = 5.0) -> BoundingBox:
        """Create bounding box from list of points with margin"""
        if not points:
            raise ValueError("Points list cannot be empty")
        
        lats = [p[0] for p in points]
        lons = [p[1] for p in points]
        
        bbox = BoundingBox(
            west=min(lons),
            south=min(lats),
            east=max(lons),
            north=max(lats)
        )
        
        return bbox.expand(margin_km)
    
    @staticmethod
    def interpolate_route(start: Tuple[float, float], end: Tuple[float, float], 
                         num_points: int = 50) -> List[Tuple[float, float]]:
        """Interpolate points along great circle route"""
        if num_points < 2:
            return [start, end]
        
        points = []
        for i in range(num_points):
            fraction = i / (num_points - 1)
            interpolated = great_circle_interpolate(start, end, fraction)
            points.append(interpolated)
        
        return points
    
    @staticmethod
    def simplify_route(points: List[Tuple[float, float]], tolerance_km: float = 0.5) -> List[Tuple[float, float]]:
        """Simplify route using Douglas-Peucker algorithm"""
        if len(points) <= 2:
            return points
        
        return douglas_peucker(points, tolerance_km)
    
    @staticmethod
    def calculate_route_length(points: List[Tuple[float, float]]) -> float:
        """Calculate total length of route in km"""
        if len(points) < 2:
            return 0.0
        
        total_length = 0.0
        for i in range(1, len(points)):
            segment_length = haversine_distance(
                points[i-1][0], points[i-1][1],
                points[i][0], points[i][1]
            )
            total_length += segment_length
        
        return total_length
    
    @staticmethod
    def find_closest_point(target: Tuple[float, float], 
                          candidates: List[Tuple[float, float]]) -> Tuple[int, float]:
        """Find closest point and return index and distance"""
        if not candidates:
            return -1, float('inf')
        
        min_distance = float('inf')
        closest_index = -1
        
        for i, candidate in enumerate(candidates):
            distance = haversine_distance(
                target[0], target[1], candidate[0], candidate[1]
            )
            if distance < min_distance:
                min_distance = distance
                closest_index = i
        
        return closest_index, min_distance
    
    @staticmethod
    def points_within_radius(center: Tuple[float, float], 
                           candidates: List[Tuple[float, float]], 
                           radius_km: float) -> List[Tuple[int, float]]:
        """Find all points within radius, return with indices and distances"""
        within_radius = []
        
        for i, candidate in enumerate(candidates):
            distance = haversine_distance(
                center[0], center[1], candidate[0], candidate[1]
            )
            if distance <= radius_km:
                within_radius.append((i, distance))
        
        return sorted(within_radius, key=lambda x: x[1])  # Sort by distance
    
    @staticmethod
    def calculate_area_polygon(points: List[Tuple[float, float]]) -> float:
        """Calculate area of polygon in km² using spherical excess"""
        if len(points) < 3:
            return 0.0
        
        # Convert to radians
        coords_rad = [(math.radians(lat), math.radians(lon)) for lat, lon in points]
        
        # Ensure polygon is closed
        if coords_rad[0] != coords_rad[-1]:
            coords_rad.append(coords_rad[0])
        
        # Calculate spherical excess
        area_steradians = 0.0
        n = len(coords_rad) - 1
        
        for i in range(n):
            j = (i + 1) % n
            area_steradians += coords_rad[i][1] * math.sin(coords_rad[j][0])
            area_steradians -= coords_rad[j][1] * math.sin(coords_rad[i][0])
        
        area_steradians = abs(area_steradians) / 2.0
        
        # Convert to km²
        earth_surface_area = 4 * math.pi * GeoUtils.EARTH_RADIUS_KM ** 2
        return area_steradians * earth_surface_area / (4 * math.pi)
    
    @staticmethod
    def create_grid_points(bbox: BoundingBox, spacing_km: float) -> List[Tuple[float, float]]:
        """Create regular grid of points within bounding box"""
        points = []
        
        # Calculate spacing in degrees
        lat_spacing = km_to_degrees_lat(spacing_km)
        center_lat = (bbox.north + bbox.south) / 2
        lon_spacing = km_to_degrees_lon(spacing_km, center_lat)
        
        # Generate grid
        current_lat = bbox.south
        while current_lat <= bbox.north:
            current_lon = bbox.west
            while current_lon <= bbox.east:
                points.append((current_lat, current_lon))
                current_lon += lon_spacing
            current_lat += lat_spacing
        
        return points
    
    @staticmethod
    def buffer_point(center: Tuple[float, float], radius_km: float, 
                    num_points: int = 32) -> List[Tuple[float, float]]:
        """Create circular buffer around point"""
        points = []
        angle_step = 360.0 / num_points
        
        for i in range(num_points):
            bearing = i * angle_step
            point = offset_coordinates(center[0], center[1], radius_km, bearing)
            points.append(point)
        
        # Close the polygon
        points.append(points[0])
        return points
    
    @staticmethod
    def line_intersection(line1_start: Tuple[float, float], line1_end: Tuple[float, float],
                         line2_start: Tuple[float, float], line2_end: Tuple[float, float]) -> Optional[Tuple[float, float]]:
        """Find intersection point of two lines (simplified planar approximation)"""
        x1, y1 = line1_start[1], line1_start[0]  # lon, lat
        x2, y2 = line1_end[1], line1_end[0]
        x3, y3 = line2_start[1], line2_start[0]
        x4, y4 = line2_end[1], line2_end[0]
        
        denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denominator) < 1e-10:  # Lines are parallel
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denominator
        u = -((x1 - x2) * (y1 - y3) - (y1 - y2) * (x1 - x3)) / denominator
        
        if 0 <= t <= 1 and 0 <= u <= 1:  # Intersection within both line segments
            intersection_x = x1 + t * (x2 - x1)
            intersection_y = y1 + t * (y2 - y1)
            return (intersection_y, intersection_x)  # lat, lon
        
        return None
    
    @staticmethod
    def point_to_line_distance(point: Tuple[float, float], 
                              line_start: Tuple[float, float], 
                              line_end: Tuple[float, float]) -> Tuple[float, Tuple[float, float]]:
        """Calculate shortest distance from point to line segment and closest point on line"""
        
        # Convert to local coordinate system for accuracy
        lat_center = (point[0] + line_start[0] + line_end[0]) / 3
        
        # Convert to meters in local system
        px = degrees_lon_to_meters(point[1], lat_center)
        py = degrees_lat_to_meters(point[0])
        
        x1 = degrees_lon_to_meters(line_start[1], lat_center)
        y1 = degrees_lat_to_meters(line_start[0])
        
        x2 = degrees_lon_to_meters(line_end[1], lat_center)
        y2 = degrees_lat_to_meters(line_end[0])
        
        # Calculate closest point on line
        line_length_sq = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
        if line_length_sq == 0:  # Line is actually a point
            distance_m = math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
            closest_point = line_start
        else:
            # Parameter t represents position along line (0 = start, 1 = end)
            t = max(0, min(1, ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / line_length_sq))
            
            # Closest point on line
            closest_x = x1 + t * (x2 - x1)
            closest_y = y1 + t * (y2 - y1)
            
            # Distance in meters
            distance_m = math.sqrt((px - closest_x) ** 2 + (py - closest_y) ** 2)
            
            # Convert closest point back to lat/lon
            closest_lat = meters_to_degrees_lat(closest_y)
            closest_lon = meters_to_degrees_lon(closest_x, lat_center)
            closest_point = (closest_lat, closest_lon)
        
        return distance_m / 1000.0, closest_point  # Return distance in km

# Core distance and bearing calculations

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate great circle distance between two points using Haversine formula"""
    R = 6371.0  # Earth's radius in km
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = (math.sin(dlat / 2) ** 2 + 
         math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
    
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    return R * c

def vincenty_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance using Vincenty's formula (more accurate for long distances)"""
    
    a = 6378137.0  # WGS84 semi-major axis
    f = 1 / 298.257223563  # WGS84 flattening
    b = (1 - f) * a
    
    # Convert to radians
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    
    L = lon2_rad - lon1_rad
    U1 = math.atan((1 - f) * math.tan(lat1_rad))
    U2 = math.atan((1 - f) * math.tan(lat2_rad))
    
    sin_U1 = math.sin(U1)
    cos_U1 = math.cos(U1)
    sin_U2 = math.sin(U2)
    cos_U2 = math.cos(U2)
    
    lambda_val = L
    lambda_prev = 2 * math.pi
    iteration_limit = 100
    
    while abs(lambda_val - lambda_prev) > 1e-12 and iteration_limit > 0:
        sin_lambda = math.sin(lambda_val)
        cos_lambda = math.cos(lambda_val)
        
        sin_sigma = math.sqrt((cos_U2 * sin_lambda) ** 2 + 
                             (cos_U1 * sin_U2 - sin_U1 * cos_U2 * cos_lambda) ** 2)
        
        if sin_sigma == 0:
            return 0.0  # Coincident points
        
        cos_sigma = sin_U1 * sin_U2 + cos_U1 * cos_U2 * cos_lambda
        sigma = math.atan2(sin_sigma, cos_sigma)
        
        sin_alpha = cos_U1 * cos_U2 * sin_lambda / sin_sigma
        cos2_alpha = 1 - sin_alpha ** 2
        
        if cos2_alpha == 0:
            cos_2sigma_m = 0
        else:
            cos_2sigma_m = cos_sigma - 2 * sin_U1 * sin_U2 / cos2_alpha
        
        C = f / 16 * cos2_alpha * (4 + f * (4 - 3 * cos2_alpha))
        
        lambda_prev = lambda_val
        lambda_val = (L + (1 - C) * f * sin_alpha *
                     (sigma + C * sin_sigma * (cos_2sigma_m + C * cos_sigma * 
                      (-1 + 2 * cos_2sigma_m ** 2))))
        
        iteration_limit -= 1
    
    if iteration_limit == 0:
        return float('nan')  # Failed to converge
    
    u2 = cos2_alpha * (a ** 2 - b ** 2) / (b ** 2)
    A = 1 + u2 / 16384 * (4096 + u2 * (-768 + u2 * (320 - 175 * u2)))
    B = u2 / 1024 * (256 + u2 * (-128 + u2 * (74 - 47 * u2)))
    
    delta_sigma = (B * sin_sigma * (cos_2sigma_m + B / 4 * 
                   (cos_sigma * (-1 + 2 * cos_2sigma_m ** 2) - B / 6 * cos_2sigma_m * 
                    (-3 + 4 * sin_sigma ** 2) * (-3 + 4 * cos_2sigma_m ** 2))))
    
    s = b * A * (sigma - delta_sigma)
    
    return s / 1000.0  # Convert to km

def calculate_bearing(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate initial bearing from point 1 to point 2"""
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    dlon_rad = math.radians(lon2 - lon1)
    
    y = math.sin(dlon_rad) * math.cos(lat2_rad)
    x = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
         math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(dlon_rad))
    
    bearing_rad = math.atan2(y, x)
    bearing_deg = math.degrees(bearing_rad)
    
    return (bearing_deg + 360) % 360  # Normalize to 0-360°

def offset_coordinates(lat: float, lon: float, distance_km: float, bearing_degrees: float) -> Tuple[float, float]:
    """Calculate new coordinates offset by distance and bearing"""
    R = 6371.0  # Earth's radius in km
    bearing_rad = math.radians(bearing_degrees)
    
    lat1_rad = math.radians(lat)
    lon1_rad = math.radians(lon)
    
    lat2_rad = math.asin(math.sin(lat1_rad) * math.cos(distance_km / R) +
                        math.cos(lat1_rad) * math.sin(distance_km / R) * math.cos(bearing_rad))
    
    lon2_rad = lon1_rad + math.atan2(
        math.sin(bearing_rad) * math.sin(distance_km / R) * math.cos(lat1_rad),
        math.cos(distance_km / R) - math.sin(lat1_rad) * math.sin(lat2_rad)
    )
    
    return math.degrees(lat2_rad), math.degrees(lon2_rad)

def great_circle_interpolate(start: Tuple[float, float], end: Tuple[float, float], fraction: float) -> Tuple[float, float]:
    """Interpolate point along great circle between start and end"""
    if fraction <= 0:
        return start
    if fraction >= 1:
        return end
    
    lat1_rad = math.radians(start[0])
    lon1_rad = math.radians(start[1])
    lat2_rad = math.radians(end[0])
    lon2_rad = math.radians(end[1])
    
    # Calculate angular distance
    d = math.acos(math.sin(lat1_rad) * math.sin(lat2_rad) + 
                  math.cos(lat1_rad) * math.cos(lat2_rad) * math.cos(lon2_rad - lon1_rad))
    
    if d == 0:  # Points are the same
        return start
    
    # Interpolate
    a = math.sin((1 - fraction) * d) / math.sin(d)
    b = math.sin(fraction * d) / math.sin(d)
    
    x = a * math.cos(lat1_rad) * math.cos(lon1_rad) + b * math.cos(lat2_rad) * math.cos(lon2_rad)
    y = a * math.cos(lat1_rad) * math.sin(lon1_rad) + b * math.cos(lat2_rad) * math.sin(lon2_rad)
    z = a * math.sin(lat1_rad) + b * math.sin(lat2_rad)
    
    lat_result = math.atan2(z, math.sqrt(x * x + y * y))
    lon_result = math.atan2(y, x)
    
    return math.degrees(lat_result), math.degrees(lon_result)

# Route simplification

def douglas_peucker(points: List[Tuple[float, float]], tolerance_km: float) -> List[Tuple[float, float]]:
    """Simplify line using Douglas-Peucker algorithm"""
    if len(points) <= 2:
        return points
    
    # Find point with maximum distance from line
    max_distance = 0.0
    max_index = 0
    
    start = points[0]
    end = points[-1]
    
    for i in range(1, len(points) - 1):
        distance, _ = GeoUtils.point_to_line_distance(points[i], start, end)
        if distance > max_distance:
            max_distance = distance
            max_index = i
    
    # If max distance is greater than tolerance, recursively simplify
    if max_distance > tolerance_km:
        # Recursive call
        left_points = douglas_peucker(points[:max_index + 1], tolerance_km)
        right_points = douglas_peucker(points[max_index:], tolerance_km)
        
        # Combine results (remove duplicate middle point)
        return left_points[:-1] + right_points
    else:
        # All points are within tolerance, return simplified line
        return [start, end]

# Coordinate conversion utilities

def km_to_degrees_lat(km: float) -> float:
    """Convert kilometers to degrees latitude"""
    return km / 111.32  # Approximate: 1 degree lat = 111.32 km

def km_to_degrees_lon(km: float, latitude: float) -> float:
    """Convert kilometers to degrees longitude at given latitude"""
    km_per_degree = 111.32 * math.cos(math.radians(latitude))
    return km / km_per_degree if km_per_degree > 0 else 0

def degrees_lat_to_meters(degrees: float) -> float:
    """Convert degrees latitude to meters"""
    return degrees * 111320.0

def degrees_lon_to_meters(degrees: float, latitude: float) -> float:
    """Convert degrees longitude to meters at given latitude"""
    return degrees * 111320.0 * math.cos(math.radians(latitude))

def meters_to_degrees_lat(meters: float) -> float:
    """Convert meters to degrees latitude"""
    return meters / 111320.0

def meters_to_degrees_lon(meters: float, latitude: float) -> float:
    """Convert meters to degrees longitude at given latitude"""
    return meters / (111320.0 * math.cos(math.radians(latitude)))

# Country/region utilities

def get_country_bounds(country_code: str) -> Optional[BoundingBox]:
    """Get approximate bounding box for country (simplified)"""
    country_bounds = {
        'BE': BoundingBox(2.5, 49.5, 6.4, 51.5),     # Belgium
        'NL': BoundingBox(3.3, 50.7, 7.2, 53.6),     # Netherlands
        'DE': BoundingBox(5.9, 47.3, 15.0, 55.1),    # Germany
        'FR': BoundingBox(-5.1, 41.3, 9.6, 51.1),    # France
        'CH': BoundingBox(5.9, 45.8, 10.5, 47.8),    # Switzerland
        'AT': BoundingBox(9.5, 46.4, 17.2, 49.0),    # Austria
        'IT': BoundingBox(6.6, 35.5, 18.5, 47.1),    # Italy
        'ES': BoundingBox(-9.3, 35.9, 4.3, 43.8),    # Spain
        'UK': BoundingBox(-8.6, 49.9, 1.8, 60.8),    # United Kingdom
        'PL': BoundingBox(14.1, 49.0, 24.1, 54.8),   # Poland
    }
    
    return country_bounds.get(country_code.upper())

def detect_country(lat: float, lon: float) -> Optional[str]:
    """Detect country from coordinates (simplified)"""
    for country_code, bounds in {
        'BE': BoundingBox(2.5, 49.5, 6.4, 51.5),
        'NL': BoundingBox(3.3, 50.7, 7.2, 53.6),
        'DE': BoundingBox(5.9, 47.3, 15.0, 55.1),
        'FR': BoundingBox(-5.1, 41.3, 9.6, 51.1),
        'CH': BoundingBox(5.9, 45.8, 10.5, 47.8),
        'AT': BoundingBox(9.5, 46.4, 17.2, 49.0),
    }.items():
        if bounds.contains(lat, lon):
            return country_code
    
    return None

# Example usage and testing
if __name__ == "__main__":
    # Test basic distance calculation
    brussels = (50.8503, 4.3517)
    amsterdam = (52.3676, 4.9041)
    
    distance = haversine_distance(*brussels, *amsterdam)
    print(f"Brussels to Amsterdam: {distance:.2f} km")
    
    # Test route interpolation
    route = GeoUtils.interpolate_route(brussels, amsterdam, 10)
    print(f"Interpolated route has {len(route)} points")
    
    # Test bounding box
    bbox = GeoUtils.create_bounding_box([brussels, amsterdam], margin_km=10)
    print(f"Bounding box: {bbox}")
    print(f"Area: {bbox.area_km2():.2f} km²")