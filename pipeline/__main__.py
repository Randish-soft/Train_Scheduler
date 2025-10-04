#!/usr/bin/env python3
"""
Main script to process building footprints for countries known for excellent train systems
"""

import os
import sys
import json
from datetime import datetime
from typing import List, Dict, Tuple

# Add pipeline modules to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from Train.Load_buildings import BuildingLoader
# Comment out modules that don't exist yet
# from Train.Load_train_lines import TrainLineLoader
# from Inference.Infer_country_name import CountryInference

# Countries renowned for their train systems
TRAIN_COUNTRIES = {
    "Japan": {
        "cities": [
            {"name": "Tokyo", "coords": (35.6762, 139.6503), "radius": 2000},
            {"name": "Osaka", "coords": (34.6937, 135.5023), "radius": 1500},
            {"name": "Kyoto", "coords": (35.0116, 135.7681), "radius": 1000},
        ],
        "description": "Shinkansen bullet trains, punctuality, extensive network"
    },
    
    "Switzerland": {
        "cities": [
            {"name": "Zurich", "coords": (47.3769, 8.5417), "radius": 1500},
            {"name": "Geneva", "coords": (46.2044, 6.1432), "radius": 1000},
            {"name": "Bern", "coords": (46.9481, 7.4474), "radius": 1000},
        ],
        "description": "Scenic routes, precision, mountain railways"
    },
    
    "Germany": {
        "cities": [
            {"name": "Berlin", "coords": (52.5200, 13.4050), "radius": 2000},
            {"name": "Munich", "coords": (48.1351, 11.5820), "radius": 1500},
            {"name": "Frankfurt", "coords": (50.1109, 8.6821), "radius": 1500},
        ],
        "description": "ICE high-speed trains, extensive network, reliability"
    },
    
    "France": {
        "cities": [
            {"name": "Paris", "coords": (48.8566, 2.3522), "radius": 2000},
            {"name": "Lyon", "coords": (45.7640, 4.8357), "radius": 1000},
            {"name": "Marseille", "coords": (43.2965, 5.3698), "radius": 1000},
        ],
        "description": "TGV high-speed network, extensive coverage"
    },
    
    "Netherlands": {
        "cities": [
            {"name": "Amsterdam", "coords": (52.3676, 4.9041), "radius": 1500},
            {"name": "Rotterdam", "coords": (51.9244, 4.4777), "radius": 1000},
            {"name": "Utrecht", "coords": (52.0907, 5.1214), "radius": 1000},
        ],
        "description": "Dense network, frequent service, integration"
    },
    
    "Spain": {
        "cities": [
            {"name": "Madrid", "coords": (40.4168, -3.7038), "radius": 2000},
            {"name": "Barcelona", "coords": (41.3874, 2.1686), "radius": 1500},
            {"name": "Seville", "coords": (37.3891, -5.9845), "radius": 1000},
        ],
        "description": "AVE high-speed trains, modern infrastructure"
    },
    
    "South Korea": {
        "cities": [
            {"name": "Seoul", "coords": (37.5665, 126.9780), "radius": 2000},
            {"name": "Busan", "coords": (35.1796, 129.0756), "radius": 1500},
            {"name": "Daegu", "coords": (35.8714, 128.6014), "radius": 1000},
        ],
        "description": "KTX bullet trains, modern system, efficiency"
    },
    
    "China": {
        "cities": [
            {"name": "Beijing", "coords": (39.9042, 116.4074), "radius": 2000},
            {"name": "Shanghai", "coords": (31.2304, 121.4737), "radius": 2000},
            {"name": "Guangzhou", "coords": (23.1291, 113.2644), "radius": 1500},
        ],
        "description": "World's largest high-speed network, maglev technology"
    },
    
    "United Kingdom": {
        "cities": [
            {"name": "London", "coords": (51.5074, -0.1278), "radius": 2000},
            {"name": "Manchester", "coords": (53.4808, -2.2426), "radius": 1000},
            {"name": "Edinburgh", "coords": (55.9533, -3.1883), "radius": 1000},
        ],
        "description": "Historic railway heritage, extensive network"
    },
    
    "Austria": {
        "cities": [
            {"name": "Vienna", "coords": (48.2082, 16.3738), "radius": 1500},
            {"name": "Salzburg", "coords": (47.8095, 13.0550), "radius": 1000},
            {"name": "Innsbruck", "coords": (47.2692, 11.4041), "radius": 800},
        ],
        "description": "Alpine railways, integrated transport, ÖBB Railjet"
    }
}

class TrainCountryProcessor:
    def __init__(self, output_dir: str = "data/buildings"):
        """
        Initialize the processor for countries with great train systems
        
        Parameters:
        -----------
        output_dir : str
            Directory to save the building data
        """
        self.output_dir = output_dir
        self.ensure_output_directory()
        self.results = {}
        
    def ensure_output_directory(self):
        """Create output directory structure"""
        os.makedirs(self.output_dir, exist_ok=True)
        for country in TRAIN_COUNTRIES.keys():
            country_dir = os.path.join(self.output_dir, country.replace(" ", "_"))
            os.makedirs(country_dir, exist_ok=True)
    
    def process_country(self, country: str, cities_data: Dict) -> Dict:
        """
        Process building data for a specific country
        
        Parameters:
        -----------
        country : str
            Country name
        cities_data : dict
            Dictionary containing cities and their data
        
        Returns:
        --------
        dict : Processing results
        """
        print(f"\n{'='*60}")
        print(f"Processing {country}")
        print(f"Description: {cities_data['description']}")
        print(f"{'='*60}")
        
        country_results = {
            "country": country,
            "description": cities_data["description"],
            "cities": [],
            "total_buildings": 0,
            "processing_time": None
        }
        
        start_time = datetime.now()
        country_dir = os.path.join(self.output_dir, country.replace(" ", "_"))
        
        for city_info in cities_data["cities"]:
            city_result = self.process_city(
                city_info["name"], 
                city_info["coords"], 
                city_info["radius"],
                country_dir
            )
            country_results["cities"].append(city_result)
            country_results["total_buildings"] += city_result.get("building_count", 0)
        
        country_results["processing_time"] = str(datetime.now() - start_time)
        
        # Save country summary
        summary_path = os.path.join(country_dir, "summary.json")
        with open(summary_path, 'w') as f:
            json.dump(country_results, f, indent=2)
        
        print(f"\n✓ Completed {country}: {country_results['total_buildings']} buildings")
        
        return country_results
    
    def process_city(self, city_name: str, coords: Tuple[float, float], 
                     radius: int, output_dir: str) -> Dict:
        """
        Process building data for a specific city
        
        Parameters:
        -----------
        city_name : str
            Name of the city
        coords : tuple
            (latitude, longitude) coordinates
        radius : int
            Radius in meters for data extraction
        output_dir : str
            Directory to save the data
        
        Returns:
        --------
        dict : City processing results
        """
        print(f"\n  Processing {city_name}...")
        
        city_result = {
            "name": city_name,
            "coordinates": coords,
            "radius": radius,
            "building_count": 0,
            "status": "pending"
        }
        
        try:
            # Try to import the building extraction function
            try:
                from Train.Load_buildings import extract_buildings_for_location
                
                # Define output paths
                city_filename = city_name.replace(" ", "_").lower()
                geojson_path = os.path.join(output_dir, f"{city_filename}_buildings.geojson")
                json_path = os.path.join(output_dir, f"{city_filename}_buildings.json")
                
                # Extract buildings
                building_count = extract_buildings_for_location(
                    location=coords,
                    radius=radius,
                    output_geojson=geojson_path,
                    output_json=json_path
                )
                
                city_result["building_count"] = building_count
                city_result["status"] = "success"
                city_result["files"] = {
                    "geojson": geojson_path,
                    "json": json_path
                }
                
                print(f"    ✓ {city_name}: {building_count} buildings extracted")
                
            except ImportError:
                # Fall back to using BuildingLoader class directly
                from Train.Load_buildings import BuildingLoader
                
                loader = BuildingLoader()
                buildings = loader.extract_buildings(coords, radius)
                
                if buildings is not None and not buildings.empty:
                    buildings = loader.process_buildings(buildings)
                    
                    city_filename = city_name.replace(" ", "_").lower()
                    geojson_path = os.path.join(output_dir, f"{city_filename}_buildings.geojson")
                    json_path = os.path.join(output_dir, f"{city_filename}_buildings.json")
                    
                    loader.save_as_geojson(buildings, geojson_path)
                    loader.save_as_json(buildings, json_path)
                    
                    city_result["building_count"] = len(buildings)
                    city_result["status"] = "success"
                    city_result["files"] = {
                        "geojson": geojson_path,
                        "json": json_path
                    }
                    
                    print(f"    ✓ {city_name}: {len(buildings)} buildings extracted")
                else:
                    raise Exception("No buildings found")
            
        except Exception as e:
            print(f"    ✗ Error processing {city_name}: {str(e)}")
            city_result["status"] = "error"
            city_result["error"] = str(e)
        
        return city_result
    
    def process_all_countries(self, countries: List[str] = None) -> Dict:
        """
        Process all or selected countries
        
        Parameters:
        -----------
        countries : list
            List of country names to process. If None, process all.
        
        Returns:
        --------
        dict : Complete processing results
        """
        if countries is None:
            countries = list(TRAIN_COUNTRIES.keys())
        
        print(f"\n{'#'*60}")
        print(f"PROCESSING COUNTRIES WITH EXCELLENT TRAIN SYSTEMS")
        print(f"Countries to process: {len(countries)}")
        print(f"{'#'*60}")
        
        start_time = datetime.now()
        
        for country in countries:
            if country in TRAIN_COUNTRIES:
                self.results[country] = self.process_country(
                    country, 
                    TRAIN_COUNTRIES[country]
                )
            else:
                print(f"\n⚠ Warning: {country} not in list of train countries")
        
        # Save overall summary
        summary = {
            "processing_date": datetime.now().isoformat(),
            "total_processing_time": str(datetime.now() - start_time),
            "countries_processed": len(self.results),
            "total_buildings": sum(r["total_buildings"] for r in self.results.values()),
            "results": self.results
        }
        
        summary_path = os.path.join(self.output_dir, "processing_summary.json")
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"\n{'#'*60}")
        print(f"PROCESSING COMPLETE")
        print(f"Total time: {summary['total_processing_time']}")
        print(f"Total buildings: {summary['total_buildings']}")
        print(f"Summary saved to: {summary_path}")
        print(f"{'#'*60}\n")
        
        return summary
    
    def generate_report(self) -> str:
        """Generate a formatted report of the processing results"""
        report = []
        report.append("\n" + "="*60)
        report.append("TRAIN COUNTRIES BUILDING EXTRACTION REPORT")
        report.append("="*60 + "\n")
        
        for country, data in self.results.items():
            report.append(f"\n{country.upper()}")
            report.append(f"  Description: {data['description']}")
            report.append(f"  Total Buildings: {data['total_buildings']:,}")
            report.append(f"  Processing Time: {data['processing_time']}")
            report.append(f"  Cities Processed:")
            
            for city in data['cities']:
                status_symbol = "✓" if city['status'] == 'success' else "✗"
                report.append(f"    {status_symbol} {city['name']}: {city['building_count']:,} buildings")
        
        report_str = "\n".join(report)
        
        # Save report
        report_path = os.path.join(self.output_dir, "report.txt")
        with open(report_path, 'w') as f:
            f.write(report_str)
        
        return report_str

def main():
    """Main execution function"""
    
    # Initialize processor
    processor = TrainCountryProcessor()
    
    # Option 1: Process all countries
    # results = processor.process_all_countries()
    
    # Option 2: Process specific countries known for the BEST train systems
    top_train_countries = [
        "Japan",         # Shinkansen, punctuality
        "Switzerland",   # Scenic, precision
        "Germany",       # ICE, extensive network
        "France",        # TGV pioneer
        "Netherlands"    # Dense, integrated
    ]
    
    results = processor.process_all_countries(top_train_countries)
    
    # Generate and print report
    report = processor.generate_report()
    print(report)
    
    return results

if __name__ == "__main__":
    # Check Python version
    if sys.version_info < (3, 6):
        print("Error: Python 3.6+ required")
        sys.exit(1)
    
    # Run main
    results = main()