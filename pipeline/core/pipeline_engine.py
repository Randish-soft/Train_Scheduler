import logging
from typing import Dict, Any, Optional
from ..stages.initialization.input_processor import InputProcessor
from ..stages.analysis.demand_analyzer import DemandAnalyzer
from ..stages.analysis.constraint_analyzer import ConstraintAnalyzer
from ..stages.plotting.route_plotter import RoutePlotter
from ..stages.analysis.nimby_analyzer import NIMBYAnalyzer
from ..stages.scheduling.timetable_creator import TimetableCreator
from .cache_manager import CacheManager
from .epoch_manager import EpochManager

class BCPCPipeline:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_manager = CacheManager(config)
        self.epoch_manager = EpochManager(config)
        
        # Initialize pipeline stages
        self.stages = {
            'input_processing': InputProcessor(config),
            'demand_analysis': DemandAnalyzer(config),
            'constraint_analysis': ConstraintAnalyzer(config),
            'route_plotting': RoutePlotter(config),
            'nimby_analysis': NIMBYAnalyzer(config),
            'timetable_creation': TimetableCreator(config)
        }
    
    def run(self, country_data: Dict[str, Any], user_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the complete BCPC pipeline"""
        self.logger.info("Starting BCPC Pipeline")
        
        # Check cache first
        cache_key = self._generate_cache_key(country_data, user_input)
        cached_result = self.cache_manager.get(cache_key)
        if cached_result:
            self.logger.info("Found cached result")
            return cached_result
        
        # Initialize processing
        context = {
            'country_data': country_data,
            'user_input': user_input,
            'terrain_data': None,
            'reference_data': None,
            'demand_data': None,
            'constraints': None,
            'routes': [],
            'timetables': []
        }
        
        # Execute pipeline stages
        try:
            # Stage 1: Initialization
            context = self.stages['input_processing'].process(context)
            context = self._process_terrain(context)
            context = self._reference_similar_country(context)
            
            # Stage 2: Analysis
            context = self.stages['demand_analysis'].analyze(context)
            context = self.stages['constraint_analysis'].analyze(context)
            
            # Stage 3: Plotting with NIMBY consideration
            context = self.stages['route_plotting'].plot_routes(context)
            context = self.stages['nimby_analysis'].analyze(context)
            
            # Stage 4: Scheduling
            context = self.stages['timetable_creation'].create_timetables(context)
            
            # Stage 5: Optimization
            context = self._optimize_railyards(context)
            
            # Cache the result
            self.cache_manager.set(cache_key, context)
            
            # Create new epoch
            self.epoch_manager.create_epoch(context)
            
            self.logger.info("Pipeline execution completed successfully")
            return context
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {e}")
            raise
    
    def _process_terrain(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Process terrain data"""
        from ..stages.initialization.terrain_processor import TerrainProcessor
        terrain_processor = TerrainProcessor(self.config)
        return terrain_processor.process(context)
    
    def _reference_similar_country(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Reference similar countries"""
        from ..stages.initialization.country_reference import CountryReference
        country_reference = CountryReference(self.config)
        return country_reference.find_similar_countries(context)
    
    def _optimize_railyards(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Optimize railyard placement"""
        from ..stages.optimization.railyard_optimizer import RailyardOptimizer
        railyard_optimizer = RailyardOptimizer(self.config)
        return railyard_optimizer.optimize(context)
    
    def _generate_cache_key(self, country_data: Dict[str, Any], user_input: Dict[str, Any]) -> str:
        """Generate cache key from input data"""
        country_name = country_data.get('name', 'unknown')
        budget = user_input.get('budget', 0)
        priority_areas = ','.join(user_input.get('priority_areas', []))
        return f"{country_name}_{budget}_{priority_areas}"