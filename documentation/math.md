# Mathematical Models - Lebanon Railway Network Map

## Overview

This document outlines the mathematical concepts and calculations used in the Lebanon Railway Network Map visualization. While the current implementation uses simplified models, understanding these foundations helps with future enhancements.

---

## 1. Geographic Calculations

### 1.1 Distance Calculation (Haversine Formula)
**Used in:** Route length estimation, station spacing

```javascript
// Calculate distance between two points on Earth's surface
function haversineDistance(lat1, lon1, lat2, lon2) {
    const R = 6371; // Earth's radius in kilometers
    const φ1 = lat1 * Math.PI / 180;
    const φ2 = lat2 * Math.PI / 180;
    const Δφ = (lat2 - lat1) * Math.PI / 180;
    const Δλ = (lon2 - lon1) * Math.PI / 180;
    
    const a = Math.sin(Δφ/2) * Math.sin(Δφ/2) +
              Math.cos(φ1) * Math.cos(φ2) *
              Math.sin(Δλ/2) * Math.sin(Δλ/2);
    
    const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    
    return R * c; // Distance in kilometers
}
```

**Example:** Distance between Beirut and Tripoli ≈ 85 km

### 1.2 Linear Interpolation
**Used in:** Smooth line rendering between stations

```javascript
// Interpolate points between two coordinates
function interpolatePoints(start, end, numPoints) {
    const points = [];
    for (let i = 0; i <= numPoints; i++) {
        const t = i / numPoints;
        const lat = start[0] + t * (end[0] - start[0]);
        const lon = start[1] + t * (end[1] - start[1]);
        points.push([lat, lon]);
    }
    return points;
}
```

---

## 2. Cost Estimation Models

### 2.1 Infrastructure Cost Calculation
**Used in:** Project economics panel

```javascript
// Cost per kilometer varies by terrain type
const costModel = {
    surface: 10_000_000,      // €10M per km
    elevated: 25_000_000,     // €25M per km
    tunnel: 80_000_000,       // €80M per km
    bridge: 65_000_000        // €65M per km
};

// Total cost calculation
function calculateTotalCost(segments) {
    return segments.reduce((total, segment) => {
        return total + (segment.length * costModel[segment.type]);
    }, 0);
}
```

**Current Implementation:**
- Total network: 700 km = €18.7 billion
- Average cost: €26.7M per km

### 2.2 Gradient Calculation
**Used in:** Engineering difficulty assessment

```javascript
// Calculate gradient percentage between two points
function calculateGradient(point1, point2, distance) {
    const elevationChange = point2.elevation - point1.elevation;
    const gradient = (elevationChange / (distance * 1000)) * 100;
    return Math.abs(gradient);
}

// Gradient classification
function classifyGradient(gradient) {
    if (gradient < 1.5) return 'gentle';      // Green
    if (gradient < 2.5) return 'moderate';    // Yellow
    return 'steep';                           // Red
}
```

**Maximum gradients:**
- High-speed lines: 2.5%
- Regional lines: 3.5%
- Mountain lines: 4.0%

---

## 3. Network Optimization

### 3.1 Station Catchment Area
**Used in:** Passenger estimation

```javascript
// Circular catchment area calculation
function calculateCatchmentPopulation(station, populationDensity) {
    const radiusKm = station.type === 'mega' ? 50 : 
                     station.type === 'major' ? 30 : 15;
    
    const area = Math.PI * radiusKm * radiusKm;
    return area * populationDensity;
}
```

### 3.2 Passenger Demand Estimation
**Used in:** Daily passenger projections

```javascript
// Simplified gravity model for passenger flow
function estimatePassengerFlow(station1, station2) {
    const distance = haversineDistance(
        station1.lat, station1.lon,
        station2.lat, station2.lon
    );
    
    // Gravity model: Flow ∝ (Pop1 × Pop2) / Distance²
    const flow = (station1.population * station2.population) / 
                 (distance * distance);
    
    // Apply scaling factor
    return flow * 0.001; // Daily passengers
}
```

---

## 4. Visualization Mathematics

### 4.1 Map Projection
**Used in:** Leaflet map rendering

```javascript
// Web Mercator projection (EPSG:3857)
// Leaflet handles this automatically, but the formula is:
function latLonToMercator(lat, lon) {
    const x = lon * 20037508.34 / 180;
    const y = Math.log(Math.tan((90 + lat) * Math.PI / 360)) / 
              (Math.PI / 180) * 20037508.34 / 180;
    return [x, y];
}
```

### 4.2 Zoom Level Calculation
**Used in:** Initial map view

```javascript
// Calculate appropriate zoom level for bounding box
function calculateZoomLevel(bounds, mapDimensions) {
    const WORLD_DIM = { height: 256, width: 256 };
    const ZOOM_MAX = 18;
    
    function latRad(lat) {
        const sin = Math.sin(lat * Math.PI / 180);
        const radX2 = Math.log((1 + sin) / (1 - sin)) / 2;
        return Math.max(Math.min(radX2, Math.PI), -Math.PI) / 2;
    }
    
    function zoom(mapPx, worldPx, fraction) {
        return Math.floor(Math.log(mapPx / worldPx / fraction) / Math.LN2);
    }
    
    const latFraction = (latRad(bounds.north) - latRad(bounds.south)) / Math.PI;
    const lngDiff = bounds.east - bounds.west;
    const lngFraction = ((lngDiff < 0) ? (lngDiff + 360) : lngDiff) / 360;
    
    const latZoom = zoom(mapDimensions.height, WORLD_DIM.height, latFraction);
    const lngZoom = zoom(mapDimensions.width, WORLD_DIM.width, lngFraction);
    
    return Math.min(latZoom, lngZoom, ZOOM_MAX);
}
```

---

## 5. Engineering Calculations

### 5.1 Curve Radius
**Used in:** Track design constraints

```javascript
// Minimum curve radius based on speed
function minimumCurveRadius(speedKmh) {
    // R = V² / (127 × f)
    // where f = coefficient of friction (0.15 for railways)
    const V = speedKmh;
    const f = 0.15;
    return (V * V) / (127 * f);
}

// Examples:
// 200 km/h → 2,096m minimum radius
// 160 km/h → 1,341m minimum radius
// 120 km/h → 755m minimum radius
```

### 5.2 Tunnel Ventilation Requirements
**Used in:** Tunnel design specifications

```javascript
// Simplified ventilation calculation
function tunnelVentilationSpacing(tunnelLength, trainFrequency) {
    // Base spacing: 800m
    // Adjust for length and traffic
    const baseSpaing = 800;
    const lengthFactor = tunnelLength / 10000; // per 10km
    const trafficFactor = trainFrequency / 20;  // per 20 trains/hour
    
    return baseSpaing / (1 + lengthFactor + trafficFactor);
}
```

---

## 6. Performance Metrics

### 6.1 Network Efficiency
**Used in:** Future network optimization

```javascript
// Calculate network directness ratio
function networkEfficiency(routeDistance, straightLineDistance) {
    return straightLineDistance / routeDistance;
}

// Ideal efficiency > 0.7
// Current Beirut-Tripoli: 85km straight / 145km route = 0.59
```

### 6.2 Station Spacing Optimization
**Used in:** Station placement

```javascript
// Optimal spacing based on service type
function optimalStationSpacing(lineType, populationDensity) {
    const baseSpacing = {
        'high-speed': 50,    // km
        'regional': 20,      // km
        'commuter': 5,       // km
        'urban': 2           // km
    };
    
    // Adjust for population density
    const densityFactor = Math.log10(populationDensity) / 3;
    return baseSpacing[lineType] / densityFactor;
}
```

---

## 7. Animation & Rendering

### 7.1 Smooth Path Animation
**Used in:** Future train movement visualization

```javascript
// Bezier curve for smooth transitions
function cubicBezier(t, p0, p1, p2, p3) {
    const u = 1 - t;
    const tt = t * t;
    const uu = u * u;
    const uuu = uu * u;
    const ttt = tt * t;
    
    return uuu * p0 +      // (1-t)³ × P0
           3 * uu * t * p1 +   // 3(1-t)² × t × P1
           3 * u * tt * p2 +   // 3(1-t) × t² × P2
           ttt * p3;           // t³ × P3
}
```

### 7.2 Icon Sizing
**Used in:** Station marker scaling

```javascript
// Dynamic icon sizing based on zoom level
function calculateIconSize(baseSize, zoomLevel) {
    const minZoom = 8;
    const maxZoom = 18;
    const scaleFactor = (zoomLevel - minZoom) / (maxZoom - minZoom);
    
    return baseSize * (0.5 + 0.5 * scaleFactor);
}
```

---

## 8. Statistical Analysis

### 8.1 Passenger Distribution
**Used in:** Capacity planning

```javascript
// Normal distribution for peak hours
function passengerDistribution(hour, dailyTotal) {
    const peakMorning = 8;
    const peakEvening = 18;
    const σ = 2; // Standard deviation
    
    const morningPeak = gaussian(hour, peakMorning, σ);
    const eveningPeak = gaussian(hour, peakEvening, σ);
    
    return dailyTotal * (morningPeak + eveningPeak) / 2;
}

function gaussian(x, μ, σ) {
    return (1 / (σ * Math.sqrt(2 * Math.PI))) * 
           Math.exp(-0.5 * Math.pow((x - μ) / σ, 2));
}
```

---

## Future Mathematical Enhancements

### 1. Dijkstra's Algorithm
For finding shortest paths between stations

### 2. Voronoi Diagrams
For optimal station catchment areas

### 3. Linear Programming
For construction phase optimization

### 4. Monte Carlo Simulation
For ridership and revenue projections

### 5. Graph Theory
For network resilience analysis

---

## Summary

The current Lebanon Railway Network Map uses basic mathematical models for:
- Geographic calculations (distance, interpolation)
- Cost estimation (linear models)
- Visualization (projection, scaling)

Future enhancements will incorporate more sophisticated models for:
- Network optimization
- Demand forecasting
- Economic analysis
- Real-time operations

All mathematical functions are implemented in vanilla JavaScript within the main HTML file, making them easy to understand and modify.