"""
Data Input Processing Module
============================

Handles automatic city data retrieval from multiple sources including:
- OpenStreetMap Nominatim API
- Built-in city databases
- Manual input fallback

Author: Miguel Ibrahim E
"""

import requests
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class CityData:
    """Data class for city information."""
    name: str
    population: int
    latitude: float
    longitude: float
    elevation: Optional[float] = None
    region: Optional[str] = None
    is_capital: bool = False
    economic_importance: Optional[str] = None


class InputProcessor:
    """Main class for processing and retrieving city data."""
    
    def __init__(self, country: str):
        self.country = country
        self.logger = logging.getLogger(__name__)
        
    def process(self) -> List[Dict[str, Any]]:
        """
        Main processing method that retrieves city data.
        
        Returns:
            List of city dictionaries with standardized format
        """
        self.logger.info(f"🌍 Processing city data for {self.country}...")
        
        # Try multiple data sources
        cities = []
        
        # Method 1: Try built-in database first (fastest)
        cities.extend(self._get_builtin_cities())
        
        # Method 2: Try OpenStreetMap Nominatim if no built-in data
        if not cities:
            cities.extend(self._fetch_from_nominatim())
        
        # Method 3: Manual input as last resort
        if not cities:
            cities.extend(self._get_manual_input())
        
        if cities:
            self.logger.info(f"✅ Retrieved {len(cities)} cities for {self.country}")
            # Convert CityData objects to dictionaries and sort by population
            city_dicts = [self._city_to_dict(city) for city in cities]
            return sorted(city_dicts, key=lambda x: x['population'], reverse=True)
        else:
            self.logger.warning(f"⚠️ Could not retrieve city data for {self.country}")
            return []
    
    def _get_builtin_cities(self) -> List[CityData]:
        """Get cities from built-in database."""
        builtin_data = {
            "ghana": [
                CityData("Accra", 2291352, 5.6037, -0.1870, 61, "Greater Accra", True),
                CityData("Kumasi", 2069350, 6.6885, -1.6244, 287, "Ashanti"),
                CityData("Tamale", 371351, 9.4008, -0.8393, 183, "Northern"),
                CityData("Takoradi", 232919, 4.8845, -1.7537, 9, "Western"),
                CityData("Cape Coast", 169894, 5.1053, -1.2466, 17, "Central"),
                CityData("Obuasi", 175043, 6.2022, -1.6640, 244, "Ashanti"),
                CityData("Tema", 161612, 5.6698, -0.0166, 16, "Greater Accra"),
                CityData("Koforidua", 120971, 6.0840, -0.2540, 167, "Eastern"),
                CityData("Sunyani", 248496, 7.3386, -2.3265, 310, "Brong-Ahafo"),
                CityData("Ho", 69998, 6.6110, 0.4713, 158, "Volta")
            ],
            "nigeria": [
                CityData("Lagos", 15388000, 6.5244, 3.3792, 41, "Lagos", False),
                CityData("Kano", 4103000, 12.0022, 8.5920, 472, "Kano"),
                CityData("Ibadan", 3649000, 7.3775, 3.9470, 150, "Oyo"),
                CityData("Abuja", 3278000, 9.0579, 7.4951, 840, "FCT", True),
                CityData("Port Harcourt", 1865000, 4.8156, 7.0498, 16, "Rivers"),
                CityData("Benin City", 1782000, 6.3350, 5.6037, 87, "Edo"),
                CityData("Maiduguri", 1197000, 11.8311, 13.1510, 354, "Borno"),
                CityData("Zaria", 975153, 11.1110, 7.7240, 610, "Kaduna"),
                CityData("Aba", 897560, 5.1066, 7.3667, 122, "Abia"),
                CityData("Jos", 900000, 9.9288, 8.8921, 1238, "Plateau"),
                CityData("Ilorin", 847582, 8.4966, 4.5426, 273, "Kwara"),
                CityData("Oyo", 736072, 7.8460, 3.9314, 355, "Oyo"),
                CityData("Enugu", 688862, 6.4414, 7.4989, 223, "Enugu"),
                CityData("Abeokuta", 593100, 7.1475, 3.3619, 67, "Ogun"),
                CityData("Osogbo", 570126, 7.7667, 4.5667, 156, "Osun")
            ],
            "kenya": [
                CityData("Nairobi", 4397073, -1.2921, 36.8219, 1795, "Nairobi", True),
                CityData("Mombasa", 1208333, -4.0435, 39.6682, 17, "Mombasa"),
                CityData("Kisumu", 409928, -0.1022, 34.7617, 1131, "Kisumu"),
                CityData("Nakuru", 570674, -0.3031, 36.0800, 1850, "Nakuru"),
                CityData("Eldoret", 475716, 0.5143, 35.2698, 2085, "Uasin Gishu"),
                CityData("Thika", 136576, -1.0332, 37.0692, 1631, "Kiambu"),
                CityData("Malindi", 207253, -3.2175, 40.1169, 23, "Kilifi"),
                CityData("Kitale", 106187, 1.0157, 35.0062, 1900, "Trans-Nzoia"),
                CityData("Garissa", 119696, -0.4536, 39.6401, 164, "Garissa"),
                CityData("Kakamega", 107227, 0.2827, 34.7519, 1535, "Kakamega"),
                CityData("Kisii", 124834, -0.6817, 34.7680, 1477, "Kisii"),
                CityData("Nyeri", 119272, -0.4167, 36.9500, 1759, "Nyeri")
            ],
            "rwanda": [
                CityData("Kigali", 1132686, -1.9441, 30.0619, 1567, "Kigali", True),
                CityData("Butare", 89600, -2.5967, 29.7391, 1768, "Southern"),
                CityData("Gitarama", 87613, -2.0756, 29.7564, 1849, "Southern"),
                CityData("Musanze", 86685, -1.5008, 29.6336, 1850, "Northern"),
                CityData("Gisenyi", 83623, -1.7025, 29.2562, 1486, "Western"),
                CityData("Byumba", 70593, -1.5761, 30.0678, 1873, "Northern"),
                CityData("Cyangugu", 63883, -2.4843, 28.9075, 1547, "Western"),
                CityData("Kibungo", 46240, -2.1547, 30.7342, 1245, "Eastern")
            ],
            "uganda": [
                CityData("Kampala", 1507080, 0.3476, 32.5825, 1190, "Central", True),
                CityData("Gulu", 152276, 2.7856, 32.2998, 1104, "Northern"),
                CityData("Lira", 119323, 2.2421, 32.8954, 1063, "Northern"),
                CityData("Mbarara", 97500, -0.6103, 30.6481, 1420, "Western"),
                CityData("Jinja", 93061, 0.4244, 33.2043, 1134, "Eastern"),
                CityData("Mbale", 88500, 1.0827, 34.1795, 1154, "Eastern"),
                CityData("Mukono", 67269, 0.3531, 32.7553, 1181, "Central"),
                CityData("Kasese", 58427, 0.1833, 30.0833, 950, "Western"),
                CityData("Masaka", 45200, -0.3397, 31.7313, 1328, "Central"),
                CityData("Entebbe", 42763, 0.0590, 32.4467, 1155, "Central")
            ],
            "tanzania": [
                CityData("Dar es Salaam", 4364541, -6.7924, 39.2083, 74, "Dar es Salaam"),
                CityData("Dodoma", 410956, -6.1630, 35.7516, 1119, "Dodoma", True),
                CityData("Mwanza", 706543, -2.5164, 32.9175, 1140, "Mwanza"),
                CityData("Zanzibar", 403658, -6.1659, 39.2026, 17, "Zanzibar"),
                CityData("Arusha", 416442, -3.3869, 36.6830, 1400, "Arusha"),
                CityData("Mbeya", 385279, -8.9094, 33.4534, 1700, "Mbeya"),
                CityData("Morogoro", 315866, -6.8235, 37.6614, 526, "Morogoro"),
                CityData("Tanga", 273332, -5.0693, 39.0993, 49, "Tanga")
            ],
            "ethiopia": [
                CityData("Addis Ababa", 3040740, 8.9806, 38.7578, 2355, "Addis Ababa", True),
                CityData("Dire Dawa", 252279, 9.5931, 41.8661, 1213, "Dire Dawa"),
                CityData("Mek'ele", 215546, 13.4967, 39.4700, 2084, "Tigray"),
                CityData("Gondar", 207044, 12.6090, 37.4700, 2133, "Amhara"),
                CityData("Awasa", 318618, 7.0469, 38.4762, 1708, "SNNP"),
                CityData("Bahir Dar", 221991, 11.5742, 37.3611, 1801, "Amhara"),
                CityData("Dessie", 151174, 11.1300, 39.6333, 2470, "Amhara"),
                CityData("Jimma", 207573, 7.6667, 36.8333, 1780, "Oromia")
            ]
        }
        
        country_key = self.country.lower()
        if country_key in builtin_data:
            self.logger.info(f"✅ Using built-in city database for {self.country}")
            return builtin_data[country_key]
        else:
            self.logger.info(f"ℹ️  No built-in data for {self.country}, trying external sources...")
            return []
    
    def _fetch_from_nominatim(self) -> List[CityData]:
        """Fetch cities from OpenStreetMap Nominatim API."""
        try:
            self.logger.info("🌐 Fetching city data from OpenStreetMap Nominatim...")
            
            # Search for major cities in the country
            url = "https://nominatim.openstreetmap.org/search"
            params = {
                'q': f"city in {self.country}",
                'format': 'json',
                'limit': 50,
                'addressdetails': 1,
                'extratags': 1
            }
            
            headers = {'User-Agent': 'Railway-Raster-Pipeline/1.0'}
            response = requests.get(url, params=params, headers=headers, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                cities = []
                
                for item in data:
                    try:
                        name = item.get('display_name', '').split(',')[0]
                        lat = float(item.get('lat', 0))
                        lon = float(item.get('lon', 0))
                        
                        # Estimate population based on place importance and type
                        importance = float(item.get('importance', 0))
                        place_type = item.get('type', '')
                        
                        # Better population estimation based on place type and importance
                        if place_type == 'city':
                            population = int(importance * 2000000) if importance > 0.5 else 500000
                        elif place_type == 'town':
                            population = int(importance * 500000) if importance > 0.3 else 100000
                        else:
                            population = int(importance * 1000000) if importance > 0 else 50000
                        
                        # Get region/state from address
                        address = item.get('address', {})
                        region = address.get('state') or address.get('region') or address.get('county', '')
                        
                        city = CityData(
                            name=name,
                            population=max(population, 10000),  # Minimum 10k population
                            latitude=lat,
                            longitude=lon,
                            region=region,
                            is_capital=('capital' in name.lower() or 'capital' in item.get('display_name', '').lower())
                        )
                        cities.append(city)
                        
                    except (ValueError, KeyError, TypeError) as e:
                        self.logger.debug(f"Skipping malformed city data: {e}")
                        continue
                
                # Remove duplicates and sort by population
                unique_cities = {}
                for city in cities:
                    if city.name not in unique_cities or city.population > unique_cities[city.name].population:
                        unique_cities[city.name] = city
                
                final_cities = list(unique_cities.values())
                self.logger.info(f"✅ Retrieved {len(final_cities)} cities from OpenStreetMap")
                return final_cities[:20]  # Return top 20 cities
                
        except requests.RequestException as e:
            self.logger.warning(f"🌐 OpenStreetMap API request failed: {e}")
            return []
        except Exception as e:
            self.logger.warning(f"🌐 OpenStreetMap data processing failed: {e}")
            return []
    
    def _get_manual_input(self) -> List[CityData]:
        """Get city data through manual user input."""
        self.logger.info("🙋 Automatic data retrieval failed. Requesting manual input...")
        
        cities = []
        
        print(f"\n🏙️  Could not automatically retrieve city data for {self.country}")
        print("Please provide information for major cities manually.")
        print("Press Enter with empty city name to finish.")
        print("\nFormat: city_name,population,latitude,longitude")
        print("Example: Accra,2291352,5.6037,-0.1870")
        print("\n💡 You can find coordinates at: https://www.latlong.net/")
        
        while True:
            try:
                user_input = input(f"\nEnter city data (or press Enter to finish): ").strip()
                
                if not user_input:
                    break
                
                parts = [p.strip() for p in user_input.split(',')]
                if len(parts) >= 4:
                    name = parts[0]
                    population = int(parts[1])
                    latitude = float(parts[2])
                    longitude = float(parts[3])
                    
                    # Validate coordinates
                    if not (-90 <= latitude <= 90):
                        print("❌ Invalid latitude. Must be between -90 and 90.")
                        continue
                    if not (-180 <= longitude <= 180):
                        print("❌ Invalid longitude. Must be between -180 and 180.")
                        continue
                    
                    city = CityData(
                        name=name,
                        population=population,
                        latitude=latitude,
                        longitude=longitude
                    )
                    cities.append(city)
                    print(f"✅ Added {name} (population: {population:,})")
                else:
                    print("❌ Invalid format. Please use: city_name,population,latitude,longitude")
                    
            except (ValueError, IndexError) as e:
                print(f"❌ Error parsing input: {e}")
                print("Please check the format and try again.")
                continue
            except KeyboardInterrupt:
                print("\n⚠️ Input cancelled by user")
                break
        
        if cities:
            self.logger.info(f"✅ Collected {len(cities)} cities from manual input")
        else:
            self.logger.warning("⚠️ No cities provided via manual input")
            
        return cities
    
    def _city_to_dict(self, city: CityData) -> Dict[str, Any]:
        """Convert CityData object to dictionary."""
        return {
            'city_name': city.name,
            'population': city.population,
            'latitude': city.latitude,
            'longitude': city.longitude,
            'elevation': city.elevation,
            'region': city.region,
            'is_capital': city.is_capital,
            'economic_importance': city.economic_importance
        }
    
    def validate_city_data(self, cities: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and clean city data."""
        valid_cities = []
        
        for city in cities:
            # Check required fields
            required_fields = ['city_name', 'population', 'latitude', 'longitude']
            if not all(field in city and city[field] is not None for field in required_fields):
                self.logger.warning(f"Skipping city with missing required fields: {city.get('city_name', 'Unknown')}")
                continue
            
            # Validate data types and ranges
            try:
                city['population'] = int(city['population'])
                city['latitude'] = float(city['latitude'])
                city['longitude'] = float(city['longitude'])
                
                # Validate ranges
                if not (-90 <= city['latitude'] <= 90):
                    self.logger.warning(f"Invalid latitude for {city['city_name']}: {city['latitude']}")
                    continue
                    
                if not (-180 <= city['longitude'] <= 180):
                    self.logger.warning(f"Invalid longitude for {city['city_name']}: {city['longitude']}")
                    continue
                    
                if city['population'] <= 0:
                    self.logger.warning(f"Invalid population for {city['city_name']}: {city['population']}")
                    continue
                
                valid_cities.append(city)
                
            except (ValueError, TypeError) as e:
                self.logger.warning(f"Data type error for {city.get('city_name', 'Unknown')}: {e}")
                continue
        
        self.logger.info(f"✅ Validated {len(valid_cities)} out of {len(cities)} cities")
        return valid_cities