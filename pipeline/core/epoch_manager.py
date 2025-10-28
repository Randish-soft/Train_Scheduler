import logging
import json
from datetime import datetime
from typing import Dict, Any, List
from pathlib import Path

class EpochManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.epochs_dir = Path(config.get('epochs', {}).get('directory', 'epochs'))
        self.epochs_dir.mkdir(exist_ok=True)
        self.max_epochs = config.get('epochs', {}).get('max_epochs', 50)
    
    def create_epoch(self, context: Dict[str, Any]) -> str:
        """Create a new epoch from pipeline context"""
        epoch_id = f"epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        epoch_data = {
            'id': epoch_id,
            'timestamp': datetime.now().isoformat(),
            'country': context['country_data'].get('name'),
            'budget': context['user_input'].get('budget'),
            'routes_count': len(context.get('routes', [])),
            'summary': self._generate_epoch_summary(context)
        }
        
        epoch_file = self.epochs_dir / f"{epoch_id}.json"
        
        try:
            with open(epoch_file, 'w') as f:
                json.dump(epoch_data, f, indent=2)
            self.logger.info(f"Created epoch: {epoch_id}")
        except Exception as e:
            self.logger.error(f"Failed to create epoch {epoch_id}: {e}")
        
        self._cleanup_old_epochs()
        return epoch_id
    
    def get_epoch(self, epoch_id: str) -> Dict[str, Any]:
        """Get epoch data by ID"""
        epoch_file = self.epochs_dir / f"{epoch_id}.json"
        
        if epoch_file.exists():
            try:
                with open(epoch_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self.logger.error(f"Failed to load epoch {epoch_id}: {e}")
        
        return {}
    
    def list_epochs(self) -> List[Dict[str, Any]]:
        """List all available epochs"""
        epochs = []
        for epoch_file in self.epochs_dir.glob("*.json"):
            try:
                with open(epoch_file, 'r') as f:
                    epoch_data = json.load(f)
                    epochs.append(epoch_data)
            except Exception as e:
                self.logger.warning(f"Failed to load epoch file {epoch_file}: {e}")
        
        # Sort by timestamp descending
        epochs.sort(key=lambda x: x.get('timestamp', ''), reverse=True)
        return epochs
    
    def _generate_epoch_summary(self, context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate summary of epoch data"""
        routes = context.get('routes', [])
        total_distance = sum(route.get('distance', 0) for route in routes)
        total_stations = sum(len(route.get('stations', [])) for route in routes)
        
        return {
            'total_routes': len(routes),
            'total_distance_km': total_distance,
            'total_stations': total_stations,
            'estimated_cost': context.get('estimated_cost', 0),
            'construction_time_months': context.get('construction_time', 0)
        }
    
    def _cleanup_old_epochs(self) -> None:
        """Remove old epochs if exceeding maximum count"""
        epochs = list(self.epochs_dir.glob("*.json"))
        if len(epochs) > self.max_epochs:
            # Sort by modification time and remove oldest
            epochs.sort(key=lambda x: x.stat().st_mtime)
            for epoch_file in epochs[:-self.max_epochs]:
                epoch_file.unlink()
                self.logger.debug(f"Removed old epoch: {epoch_file.name}")