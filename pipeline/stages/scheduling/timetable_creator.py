import logging
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, Any, List

class TimetableCreator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def create_timetables(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Create timetables for all routes"""
        self.logger.info("Creating train timetables")
        
        optimized_routes = context.get('optimized_routes', [])
        train_selection = context.get('train_selection', {})
        demand_data = context['demand_data']
        
        timetables = {}
        
        for route in optimized_routes:
            route_name = route['name']
            if route_name in train_selection:
                timetable = self._create_timetable_for_route(route, train_selection[route_name], demand_data)
                timetables[route_name] = timetable
        
        context['timetables'] = timetables
        context['timetable_analysis'] = self._analyze_timetables(timetables)
        
        self.logger.info(f"Created timetables for {len(timetables)} routes")
        return context
    
    def _create_timetable_for_route(self, route: Dict[str, Any], train_selection: Dict[str, Any],
                                  demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create timetable for a specific route"""
        route_name = route['name']
        self.logger.debug(f"Creating timetable for route: {route_name}")
        
        stations = route.get('stations', [])
        route_details = route['details']
        fleet_requirements = train_selection['fleet_requirements']
        
        if not stations:
            self.logger.warning(f"No stations found for route {route_name}")
            return {}
        
        # Sort stations by position
        stations_sorted = sorted(stations, key=lambda x: x['position_km'])
        
        # Calculate station times
        station_times = self._calculate_station_times(stations_sorted, route_details, train_selection)
        
        # Create daily schedule
        daily_schedule = self._create_daily_schedule(station_times, fleet_requirements, demand_data)
        
        # Create service patterns
        service_patterns = self._create_service_patterns(daily_schedule, stations_sorted)
        
        return {
            'route_name': route_name,
            'stations': stations_sorted,
            'station_times': station_times,
            'daily_schedule': daily_schedule,
            'service_patterns': service_patterns,
            'operating_hours': self._determine_operating_hours(demand_data),
            'timetable_metrics': self._calculate_timetable_metrics(daily_schedule, station_times)
        }
    
    def _calculate_station_times(self, stations: List[Dict[str, Any]], route_details: Dict[str, Any],
                               train_selection: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Calculate travel times between stations"""
        station_times = []
        train = train_selection['selected_train']
        
        for i in range(len(stations)):
            current_station = stations[i]
            
            if i == 0:
                # First station
                station_time = {
                    'station_name': current_station['name'],
                    'distance_from_start_km': 0,
                    'cumulative_time_minutes': 0,
                    'dwell_time_minutes': self._calculate_dwell_time(current_station),
                    'departure_time_offset': 0
                }
            else:
                # Calculate travel time from previous station
                prev_station = stations[i-1]
                distance = current_station['position_km'] - prev_station['position_km']
                travel_time = self._calculate_travel_time(distance, route_details, train)
                
                prev_station_time = station_times[-1]
                cumulative_time = prev_station_time['cumulative_time_minutes'] + travel_time
                
                station_time = {
                    'station_name': current_station['name'],
                    'distance_from_start_km': current_station['position_km'],
                    'travel_time_from_previous_minutes': travel_time,
                    'cumulative_time_minutes': cumulative_time,
                    'dwell_time_minutes': self._calculate_dwell_time(current_station),
                    'departure_time_offset': cumulative_time + self._calculate_dwell_time(current_station)
                }
            
            station_times.append(station_time)
        
        return station_times
    
    def _calculate_dwell_time(self, station: Dict[str, Any]) -> float:
        """Calculate dwell time at station based on station type"""
        station_type = station['type']
        dwell_times = {
            'major': 3.0,      # 3 minutes at major stations
            'regional': 2.0,   # 2 minutes at regional stations
            'local': 1.0,      # 1 minute at local stations
            'mountain_pass': 1.5  # 1.5 minutes at mountain passes
        }
        return dwell_times.get(station_type, 1.5)
    
    def _calculate_travel_time(self, distance_km: float, route_details: Dict[str, Any],
                             train: Dict[str, Any]) -> float:
        """Calculate travel time for a segment"""
        max_speed = min(route_details['max_design_speed_kmh'], train['max_speed_kmh'])
        avg_speed = max_speed * 0.8  # Assume 80% of max speed for scheduling
        
        # Account for acceleration and deceleration
        acceleration_time = train['max_speed_kmh'] / (train['acceleration_ms2'] * 3.6)  # Convert to seconds
        deceleration_time = train['max_speed_kmh'] / (train['deceleration_ms2'] * 3.6)
        
        # Calculate base travel time
        base_time = (distance_km / avg_speed) * 60  # Convert to minutes
        
        # Add acceleration/deceleration penalty for short segments
        if distance_km < 10:
            penalty = (acceleration_time + deceleration_time) / 60  # Convert to minutes
            base_time += penalty
        
        return max(1.0, base_time)  # Minimum 1 minute
    
    def _create_daily_schedule(self, station_times: List[Dict[str, Any]], fleet_requirements: Dict[str, Any],
                             demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create daily schedule with frequencies"""
        operating_hours = self._determine_operating_hours(demand_data)
        total_route_time = station_times[-1]['departure_time_offset'] if station_times else 0
        
        # Calculate frequencies based on demand and fleet
        peak_frequency = fleet_requirements['peak_frequency_trains_per_hour']
        off_peak_frequency = fleet_requirements['off_peak_frequency_trains_per_hour']
        
        # Create schedule periods
        schedule_periods = self._create_schedule_periods(operating_hours, peak_frequency, off_peak_frequency)
        
        # Generate train departures
        departures = self._generate_train_departures(schedule_periods, total_route_time)
        
        return {
            'operating_hours': operating_hours,
            'schedule_periods': schedule_periods,
            'train_departures': departures,
            'daily_train_trips': len(departures),
            'average_headway_minutes': self._calculate_average_headway(schedule_periods)
        }
    
    def _determine_operating_hours(self, demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Determine operating hours based on demand patterns"""
        return {
            'start_time': '05:00',
            'end_time': '23:00',
            'peak_hours_morning': ['07:00', '09:00'],
            'peak_hours_evening': ['17:00', '19:00'],
            'total_operating_hours': 18
        }
    
    def _create_schedule_periods(self, operating_hours: Dict[str, Any], peak_frequency: int,
                               off_peak_frequency: int) -> List[Dict[str, Any]]:
        """Create different schedule periods throughout the day"""
        periods = []
        
        # Early morning (pre-peak)
        periods.append({
            'period': 'early_morning',
            'start_time': '05:00',
            'end_time': '07:00',
            'frequency_trains_per_hour': max(1, off_peak_frequency // 2),
            'headway_minutes': 60 // max(1, off_peak_frequency // 2)
        })
        
        # Morning peak
        periods.append({
            'period': 'morning_peak',
            'start_time': '07:00',
            'end_time': '09:00',
            'frequency_trains_per_hour': peak_frequency,
            'headway_minutes': 60 // peak_frequency
        })
        
        # Mid-day off-peak
        periods.append({
            'period': 'midday',
            'start_time': '09:00',
            'end_time': '16:00',
            'frequency_trains_per_hour': off_peak_frequency,
            'headway_minutes': 60 // off_peak_frequency
        })
        
        # Evening peak
        periods.append({
            'period': 'evening_peak',
            'start_time': '16:00',
            'end_time': '19:00',
            'frequency_trains_per_hour': peak_frequency,
            'headway_minutes': 60 // peak_frequency
        })
        
        # Evening off-peak
        periods.append({
            'period': 'evening',
            'start_time': '19:00',
            'end_time': '23:00',
            'frequency_trains_per_hour': off_peak_frequency,
            'headway_minutes': 60 // off_peak_frequency
        })
        
        return periods
    
    def _generate_train_departures(self, schedule_periods: List[Dict[str, Any]], 
                                 total_route_time: float) -> List[Dict[str, Any]]:
        """Generate train departure times"""
        departures = []
        current_time = datetime.strptime('05:00', '%H:%M')
        
        for period in schedule_periods:
            period_start = datetime.strptime(period['start_time'], '%H:%M')
            period_end = datetime.strptime(period['end_time'], '%H:%M')
            frequency = period['frequency_trains_per_hour']
            headway = timedelta(minutes=period['headway_minutes'])
            
            # Generate departures for this period
            departure_time = period_start
            while departure_time < period_end:
                departures.append({
                    'departure_time': departure_time.strftime('%H:%M'),
                    'period': period['period'],
                    'trip_duration_minutes': total_route_time,
                    'estimated_arrival_time': (departure_time + timedelta(minutes=total_route_time)).strftime('%H:%M')
                })
                departure_time += headway
        
        return departures
    
    def _calculate_average_headway(self, schedule_periods: List[Dict[str, Any]]) -> float:
        """Calculate average headway across all periods"""
        total_minutes = 0
        total_weight = 0
        
        for period in schedule_periods:
            start = datetime.strptime(period['start_time'], '%H:%M')
            end = datetime.strptime(period['end_time'], '%H:%M')
            duration_minutes = (end - start).total_seconds() / 60
            headway = period['headway_minutes']
            
            total_minutes += duration_minutes * headway
            total_weight += duration_minutes
        
        return total_minutes / total_weight if total_weight > 0 else 30
    
    def _create_service_patterns(self, daily_schedule: Dict[str, Any], 
                               stations: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create different service patterns (express, local, etc.)"""
        if len(stations) < 4:
            # Only local service for short routes
            return {
                'local': {
                    'description': 'All stations service',
                    'stations_served': [s['name'] for s in stations],
                    'estimated_time_saving_minutes': 0,
                    'applicable_periods': ['all']
                }
            }
        
        # Create express pattern skipping some stations
        major_stations = [s for s in stations if s['type'] in ['major', 'regional']]
        if len(major_stations) >= 2:
            express_pattern = {
                'express': {
                    'description': 'Express service serving major stations only',
                    'stations_served': [s['name'] for s in major_stations],
                    'estimated_time_saving_minutes': self._calculate_express_savings(stations, major_stations),
                    'applicable_periods': ['morning_peak', 'evening_peak']
                }
            }
        else:
            express_pattern = {}
        
        # Create local pattern (all stations)
        local_pattern = {
            'local': {
                'description': 'All stations service',
                'stations_served': [s['name'] for s in stations],
                'estimated_time_saving_minutes': 0,
                'applicable_periods': ['all']
            }
        }
        
        return {**local_pattern, **express_pattern}
    
    def _calculate_express_savings(self, all_stations: List[Dict[str, Any]], 
                                 express_stations: List[Dict[str, Any]]) -> float:
        """Calculate time savings for express service"""
        total_local_time = sum(station.get('travel_time_from_previous_minutes', 0) 
                             for station in all_stations[1:])
        
        # Calculate express time (only between express stations)
        express_time = 0
        for i in range(1, len(express_stations)):
            start_pos = express_stations[i-1]['position_km']
            end_pos = express_stations[i]['position_km']
            distance = end_pos - start_pos
            # Estimate express travel time (faster due to fewer stops)
            express_time += (distance / 100) * 60  # Assume 100 km/h average
        
        return max(0, total_local_time - express_time)
    
    def _calculate_timetable_metrics(self, daily_schedule: Dict[str, Any],
                                  station_times: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Calculate timetable performance metrics"""
        total_departures = len(daily_schedule['train_departures'])
        operating_hours = daily_schedule['operating_hours']['total_operating_hours']
        
        average_frequency = total_departures / operating_hours if operating_hours > 0 else 0
        total_route_time = station_times[-1]['departure_time_offset'] if station_times else 0
        
        return {
            'daily_departures': total_departures,
            'average_frequency_trains_per_hour': average_frequency,
            'total_route_time_minutes': total_route_time,
            'average_speed_kmh': self._calculate_average_speed(station_times),
            'punctuality_estimate': 0.95,  # 95% on-time performance estimate
            'transfer_opportunities': self._assess_transfer_opportunities(station_times)
        }
    
    def _calculate_average_speed(self, station_times: List[Dict[str, Any]]) -> float:
        """Calculate average speed along the route"""
        if not station_times:
            return 0
        
        total_distance = station_times[-1]['distance_from_start_km']
        total_time = station_times[-1]['cumulative_time_minutes'] / 60  # Convert to hours
        
        return total_distance / total_time if total_time > 0 else 0
    
    def _assess_transfer_opportunities(self, station_times: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Assess transfer opportunities at stations"""
        # This would typically analyze connections with other routes
        # For now, return basic assessment
        transfers = []
        
        for station_time in station_times:
            station_name = station_time['station_name']
            if 'Central' in station_name or 'major' in station_name.lower():
                transfers.append({
                    'station': station_name,
                    'transfer_type': 'major_hub',
                    'estimated_wait_time_minutes': 10,
                    'connection_quality': 'excellent'
                })
        
        return transfers
    
    def _analyze_timetables(self, timetables: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze all timetables for system-wide metrics"""
        total_departures = 0
        total_operating_hours = 0
        route_metrics = {}
        
        for route_name, timetable in timetables.items():
            metrics = timetable['timetable_metrics']
            total_departures += metrics['daily_departures']
            total_operating_hours += timetable['daily_schedule']['operating_hours']['total_operating_hours']
            
            route_metrics[route_name] = {
                'daily_trips': metrics['daily_departures'],
                'average_frequency': metrics['average_frequency_trains_per_hour'],
                'route_duration_minutes': metrics['total_route_time_minutes']
            }
        
        return {
            'total_daily_departures': total_departures,
            'total_operating_hours': total_operating_hours,
            'system_average_frequency': total_departures / total_operating_hours if total_operating_hours > 0 else 0,
            'route_metrics': route_metrics,
            'system_efficiency_score': self._calculate_system_efficiency(route_metrics)
        }
    
    def _calculate_system_efficiency(self, route_metrics: Dict[str, Any]) -> float:
        """Calculate overall system efficiency score"""
        if not route_metrics:
            return 0.0
        
        efficiency_scores = []
        
        for route_metric in route_metrics.values():
            frequency = route_metric['average_frequency']
            duration = route_metric['route_duration_minutes']
            
            # Score based on frequency and reasonable journey times
            freq_score = min(1.0, frequency / 4)  # 4 trains/hour is ideal
            time_score = 1.0 if duration < 120 else 0.7  # 2 hours max for good score
            
            efficiency_scores.append((freq_score + time_score) / 2)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0