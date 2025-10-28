import logging
import numpy as np
from typing import Dict, Any, List

class TimetableOptimizer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
    
    def optimize_timetables(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize all timetables for efficiency and performance"""
        self.logger.info("Optimizing train timetables")
        
        timetables = context.get('timetables', {})
        train_selection = context.get('train_selection', {})
        demand_data = context['demand_data']
        
        optimized_timetables = {}
        optimization_results = []
        
        for route_name, timetable in timetables.items():
            if route_name in train_selection:
                optimized_timetable, optimizations = self._optimize_single_timetable(
                    timetable, train_selection[route_name], demand_data
                )
                optimized_timetables[route_name] = optimized_timetable
                optimization_results.append({
                    'route_name': route_name,
                    'optimizations': optimizations
                })
        
        context['optimized_timetables'] = optimized_timetables
        context['timetable_optimization_results'] = optimization_results
        context['system_timetable_analysis'] = self._analyze_system_timetables(optimized_timetables)
        
        self.logger.info("Timetable optimization completed")
        return context
    
    def _optimize_single_timetable(self, timetable: Dict[str, Any], train_selection: Dict[str, Any],
                                 demand_data: Dict[str, Any]) -> tuple:
        """Optimize a single timetable"""
        route_name = timetable['route_name']
        self.logger.debug(f"Optimizing timetable for route: {route_name}")
        
        optimizations = {
            'efficiency_improvements': [],
            'passenger_experience_enhancements': [],
            'operational_optimizations': [],
            'total_time_savings_minutes': 0,
            'capacity_improvements': 0
        }
        
        # Apply various optimization techniques
        optimizations.update(self._optimize_dwell_times(timetable, train_selection))
        optimizations.update(self._optimize_frequencies(timetable, demand_data))
        optimizations.update(self._optimize_service_patterns(timetable))
        optimizations.update(self._optimize_transfer_times(timetable))
        
        # Calculate total improvements
        time_savings = sum(opt.get('time_saving_minutes', 0) for opt in optimizations['efficiency_improvements'])
        capacity_improvements = len(optimizations['capacity_improvements'])
        
        optimizations['total_time_savings_minutes'] = time_savings
        optimizations['capacity_improvements'] = capacity_improvements
        
        # Apply optimizations to timetable
        optimized_timetable = self._apply_timetable_optimizations(timetable, optimizations)
        
        return optimized_timetable, optimizations
    
    def _optimize_dwell_times(self, timetable: Dict[str, Any], train_selection: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize station dwell times"""
        optimizations = {'efficiency_improvements': []}
        station_times = timetable['station_times']
        train = train_selection['selected_train']
        
        total_dwell_reduction = 0
        
        for i, station_time in enumerate(station_times):
            current_dwell = station_time['dwell_time_minutes']
            station_name = station_time['station_name']
            
            # Optimize dwell times based on station type and train characteristics
            optimized_dwell = self._calculate_optimized_dwell(current_dwell, station_name, train)
            
            if optimized_dwell < current_dwell:
                dwell_reduction = current_dwell - optimized_dwell
                total_dwell_reduction += dwell_reduction
                
                optimizations['efficiency_improvements'].append({
                    'type': 'dwell_time_optimization',
                    'station': station_name,
                    'original_dwell_minutes': current_dwell,
                    'optimized_dwell_minutes': optimized_dwell,
                    'time_saving_minutes': dwell_reduction,
                    'impact': f'Reduced dwell time at {station_name} by {dwell_reduction:.1f} minutes'
                })
        
        if total_dwell_reduction > 0:
            optimizations['efficiency_improvements'].append({
                'type': 'total_dwell_optimization',
                'description': 'Cumulative dwell time reductions',
                'time_saving_minutes': total_dwell_reduction,
                'impact': f'Total journey time reduced by {total_dwell_reduction:.1f} minutes'
            })
        
        return optimizations
    
    def _calculate_optimized_dwell(self, current_dwell: float, station_name: str, train: Dict[str, Any]) -> float:
        """Calculate optimized dwell time for a station"""
        base_optimization = current_dwell * 0.8  # 20% reduction as baseline
        
        # Further reductions based on station characteristics
        if 'Central' in station_name or 'major' in station_name:
            # Major stations need more time for passenger flow
            return max(2.0, base_optimization)
        elif 'local' in station_name.lower():
            # Local stations can have shorter dwell times
            return max(0.5, base_optimization * 0.9)
        else:
            return max(1.0, base_optimization)
    
    def _optimize_frequencies(self, timetable: Dict[str, Any], demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize service frequencies based on demand"""
        optimizations = {'operational_optimizations': [], 'capacity_improvements': []}
        daily_schedule = timetable['daily_schedule']
        schedule_periods = daily_schedule['schedule_periods']
        
        route_demand = self._get_route_demand(timetable['route_name'], demand_data)
        
        for period in schedule_periods:
            current_frequency = period['frequency_trains_per_hour']
            optimized_frequency = self._calculate_optimized_frequency(period, route_demand)
            
            if optimized_frequency != current_frequency:
                frequency_change = optimized_frequency - current_frequency
                
                if frequency_change > 0:
                    optimization_type = 'frequency_increase'
                    impact = f'Increased {period["period"]} frequency to meet demand'
                    capacity_impact = 1
                else:
                    optimization_type = 'frequency_optimization'
                    impact = f'Optimized {period["period"]} frequency to reduce costs'
                    capacity_impact = 0
                
                optimizations['operational_optimizations'].append({
                    'type': optimization_type,
                    'period': period['period'],
                    'original_frequency': current_frequency,
                    'optimized_frequency': optimized_frequency,
                    'impact': impact
                })
                
                if capacity_impact:
                    optimizations['capacity_improvements'].append({
                        'period': period['period'],
                        'capacity_increase_percent': (frequency_change / current_frequency) * 100
                    })
        
        return optimizations
    
    def _get_route_demand(self, route_name: str, demand_data: Dict[str, Any]) -> Dict[str, Any]:
        """Get demand data for specific route"""
        demand_corridors = demand_data.get('demand_corridors', [])
        route_demand = next((corridor for corridor in demand_corridors if corridor['route'] == route_name), None)
        
        if route_demand:
            return {
                'peak_demand': route_demand['estimated_ridership']['peak_hour_riders'],
                'average_demand': route_demand['estimated_ridership']['daily_riders'] / 16  # 16 operating hours
            }
        else:
            # Default demand estimates
            return {
                'peak_demand': 500,
                'average_demand': 200
            }
    
    def _calculate_optimized_frequency(self, period: Dict[str, Any], route_demand: Dict[str, Any]) -> int:
        """Calculate optimized frequency for a period"""
        current_frequency = period['frequency_trains_per_hour']
        
        if period['period'] in ['morning_peak', 'evening_peak']:
            # Peak periods: ensure adequate capacity
            peak_demand = route_demand['peak_demand']
            required_trains = max(1, int(peak_demand / 200))  # Assume 200 passengers per train
            return max(current_frequency, required_trains)
        else:
            # Off-peak: optimize for efficiency
            average_demand = route_demand['average_demand']
            efficient_frequency = max(1, int(average_demand / 150))  # 150 passengers per train
            return min(current_frequency, efficient_frequency)
    
    def _optimize_service_patterns(self, timetable: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize service patterns (express vs local)"""
        optimizations = {'passenger_experience_enhancements': []}
        service_patterns = timetable.get('service_patterns', {})
        
        if 'express' in service_patterns:
            express_pattern = service_patterns['express']
            time_saving = express_pattern['estimated_time_saving_minutes']
            
            # Optimize express service applicability
            current_periods = express_pattern['applicable_periods']
            
            # Consider adding express service to more periods if time savings are significant
            if time_saving > 15 and 'midday' not in current_periods:
                optimizations['passenger_experience_enhancements'].append({
                    'type': 'expanded_express_service',
                    'description': 'Extended express service to midday period',
                    'impact': 'Improved travel times for midday travelers',
                    'time_saving_minutes': time_saving
                })
        
        # Optimize local service patterns
        if 'local' in service_patterns:
            local_pattern = service_patterns['local']
            
            # Consider skip-stop patterns for better efficiency
            if len(local_pattern['stations_served']) > 8:
                optimizations['passenger_experience_enhancements'].append({
                    'type': 'skip_stop_optimization',
                    'description': 'Implemented optimized skip-stop pattern',
                    'impact': 'Balanced service coverage with operational efficiency'
                })
        
        return optimizations
    
    def _optimize_transfer_times(self, timetable: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize transfer times at interchange stations"""
        optimizations = {'passenger_experience_enhancements': []}
        timetable_metrics = timetable.get('timetable_metrics', {})
        transfer_opportunities = timetable_metrics.get('transfer_opportunities', [])
        
        for transfer in transfer_opportunities:
            current_wait = transfer['estimated_wait_time_minutes']
            
            if current_wait > 15 and transfer['transfer_type'] == 'major_hub':
                # Optimize for better transfers at major hubs
                optimizations['passenger_experience_enhancements'].append({
                    'type': 'transfer_time_optimization',
                    'station': transfer['station'],
                    'original_wait_time': current_wait,
                    'optimized_wait_time': 10,
                    'impact': f'Reduced transfer wait time at {transfer["station"]}'
                })
        
        return optimizations
    
    def _apply_timetable_optimizations(self, timetable: Dict[str, Any], optimizations: Dict[str, Any]) -> Dict[str, Any]:
        """Apply optimization results to the timetable"""
        optimized_timetable = timetable.copy()
        
        # Apply dwell time optimizations
        for optimization in optimizations['efficiency_improvements']:
            if optimization['type'] == 'dwell_time_optimization':
                station_name = optimization['station']
                optimized_dwell = optimization['optimized_dwell_minutes']
                
                # Update station times
                for station_time in optimized_timetable['station_times']:
                    if station_time['station_name'] == station_name:
                        station_time['dwell_time_minutes'] = optimized_dwell
                        # Recalculate cumulative times
                        self._recalculate_cumulative_times(optimized_timetable['station_times'])
                        break
        
        # Apply frequency optimizations
        for optimization in optimizations['operational_optimizations']:
            if optimization['type'] in ['frequency_increase', 'frequency_optimization']:
                period_name = optimization['period']
                optimized_frequency = optimization['optimized_frequency']
                
                # Update schedule periods
                for period in optimized_timetable['daily_schedule']['schedule_periods']:
                    if period['period'] == period_name:
                        period['frequency_trains_per_hour'] = optimized_frequency
                        period['headway_minutes'] = 60 // optimized_frequency
                        break
        
        # Regenerate train departures with optimized parameters
        optimized_timetable['daily_schedule']['train_departures'] = self._regenerate_departures(
            optimized_timetable['daily_schedule']['schedule_periods'],
            optimized_timetable['station_times'][-1]['departure_time_offset'] if optimized_timetable['station_times'] else 0
        )
        
        # Update timetable metrics
        optimized_timetable['timetable_metrics'] = self._update_timetable_metrics(
            optimized_timetable, timetable['timetable_metrics']
        )
        
        # Add optimization details
        optimized_timetable['optimization_details'] = optimizations
        optimized_timetable['is_optimized'] = True
        
        return optimized_timetable
    
    def _recalculate_cumulative_times(self, station_times: List[Dict[str, Any]]) -> None:
        """Recalculate cumulative times after dwell time changes"""
        for i, station_time in enumerate(station_times):
            if i == 0:
                station_time['cumulative_time_minutes'] = 0
                station_time['departure_time_offset'] = station_time['dwell_time_minutes']
            else:
                prev_station = station_times[i-1]
                travel_time = station_time.get('travel_time_from_previous_minutes', 0)
                station_time['cumulative_time_minutes'] = prev_station['departure_time_offset'] + travel_time
                station_time['departure_time_offset'] = station_time['cumulative_time_minutes'] + station_time['dwell_time_minutes']
    
    def _regenerate_departures(self, schedule_periods: List[Dict[str, Any]], total_route_time: float) -> List[Dict[str, Any]]:
        """Regenerate train departures after optimization"""
        # Reuse the same logic from TimetableCreator
        from .timetable_creator import TimetableCreator
        temp_creator = TimetableCreator(self.config)
        return temp_creator._generate_train_departures(schedule_periods, total_route_time)
    
    def _update_timetable_metrics(self, optimized_timetable: Dict[str, Any], original_metrics: Dict[str, Any]) -> Dict[str, Any]:
        """Update timetable metrics after optimization"""
        updated_metrics = original_metrics.copy()
        
        # Calculate new average speed
        station_times = optimized_timetable['station_times']
        if station_times:
            total_distance = station_times[-1]['distance_from_start_km']
            total_time = station_times[-1]['cumulative_time_minutes'] / 60
            updated_metrics['average_speed_kmh'] = total_distance / total_time if total_time > 0 else 0
        
        # Update daily departures
        updated_metrics['daily_departures'] = len(optimized_timetable['daily_schedule']['train_departures'])
        
        # Recalculate average frequency
        operating_hours = optimized_timetable['daily_schedule']['operating_hours']['total_operating_hours']
        updated_metrics['average_frequency_trains_per_hour'] = (
            updated_metrics['daily_departures'] / operating_hours if operating_hours > 0 else 0
        )
        
        return updated_metrics
    
    def _analyze_system_timetables(self, optimized_timetables: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze system-wide timetable performance"""
        total_optimizations = 0
        total_time_savings = 0
        route_optimizations = {}
        
        for route_name, timetable in optimized_timetables.items():
            if timetable.get('is_optimized', False):
                optimizations = timetable['optimization_details']
                time_savings = optimizations['total_time_savings_minutes']
                num_optimizations = len(optimizations['efficiency_improvements']) + \
                                  len(optimizations['passenger_experience_enhancements']) + \
                                  len(optimizations['operational_optimizations'])
                
                total_optimizations += num_optimizations
                total_time_savings += time_savings
                
                route_optimizations[route_name] = {
                    'optimizations_applied': num_optimizations,
                    'time_savings_minutes': time_savings,
                    'capacity_improvements': optimizations['capacity_improvements']
                }
        
        return {
            'total_optimizations_applied': total_optimizations,
            'total_time_savings_minutes': total_time_savings,
            'average_time_saving_per_route': total_time_savings / len(optimized_timetables) if optimized_timetables else 0,
            'route_optimization_summary': route_optimizations,
            'system_efficiency_improvement': self._calculate_system_efficiency_improvement(route_optimizations)
        }
    
    def _calculate_system_efficiency_improvement(self, route_optimizations: Dict[str, Any]) -> float:
        """Calculate overall system efficiency improvement"""
        if not route_optimizations:
            return 0.0
        
        efficiency_scores = []
        
        for optimization in route_optimizations.values():
            time_savings = optimization['time_savings_minutes']
            num_optimizations = optimization['optimizations_applied']
            
            # Score based on time savings and number of optimizations
            if time_savings > 10:
                time_score = 0.8
            elif time_savings > 5:
                time_score = 0.6
            else:
                time_score = 0.4
            
            optimization_score = min(1.0, num_optimizations / 5)  # 5 optimizations is good
            
            efficiency_scores.append((time_score + optimization_score) / 2)
        
        return sum(efficiency_scores) / len(efficiency_scores) if efficiency_scores else 0.0