import logging
import json
import yaml
import csv
from pathlib import Path
from typing import Dict, Any, List, Optional

class FileLoader:
    def __init__(self, base_path: str = "data"):
        self.base_path = Path(base_path)
        self.logger = logging.getLogger(__name__)
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Ensure required directories exist"""
        directories = [
            'input',
            'output', 
            'cache',
            'exports',
            'temp'
        ]
        
        for directory in directories:
            (self.base_path / directory).mkdir(parents=True, exist_ok=True)
    
    def load_country_data(self, country_name: str) -> Optional[Dict[str, Any]]:
        """Load country data from JSON file"""
        file_path = self.base_path / 'input' / 'countries' / f"{country_name.lower()}.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded country data from {file_path}")
                return data
            else:
                self.logger.warning(f"Country data file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load country data from {file_path}: {e}")
            return None
    
    def load_terrain_data(self, country_name: str) -> Optional[Dict[str, Any]]:
        """Load terrain data for a country"""
        file_path = self.base_path / 'input' / 'terrain' / f"{country_name.lower()}_terrain.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded terrain data from {file_path}")
                return data
            else:
                self.logger.warning(f"Terrain data file not found: {file_path}")
                return self._generate_default_terrain_data(country_name)
        except Exception as e:
            self.logger.error(f"Failed to load terrain data from {file_path}: {e}")
            return self._generate_default_terrain_data(country_name)
    
    def _generate_default_terrain_data(self, country_name: str) -> Dict[str, Any]:
        """Generate default terrain data when no file exists"""
        self.logger.info(f"Generating default terrain data for {country_name}")
        
        # Simple default terrain based on country name patterns
        # In a real implementation, this would use GIS data or APIs
        if any(word in country_name.lower() for word in ['mountain', 'alps', 'himalaya', 'andes']):
            terrain_type = 'mountainous'
        elif any(word in country_name.lower() for word in ['island', 'coast', 'sea', 'ocean']):
            terrain_type = 'coastal'
        elif any(word in country_name.lower() for word in ['plain', 'flat', 'lowland']):
            terrain_type = 'flat'
        else:
            terrain_type = 'mixed'
        
        return {
            'type': terrain_type,
            'elevation_profile': self._generate_synthetic_elevation(terrain_type),
            'difficulty_score': 0.5,  # Default medium difficulty
            'water_bodies': [],
            'obstacles': []
        }
    
    def _generate_synthetic_elevation(self, terrain_type: str) -> List[float]:
        """Generate synthetic elevation data based on terrain type"""
        import numpy as np
        
        if terrain_type == 'mountainous':
            return (np.random.normal(500, 300, 100) + 200).tolist()
        elif terrain_type == 'flat':
            return np.random.normal(50, 10, 100).tolist()
        elif terrain_type == 'coastal':
            return (np.random.normal(100, 50, 100) + 20).tolist()
        else:  # mixed
            return (np.random.normal(200, 150, 100) + 50).tolist()
    
    def load_config(self, config_name: str) -> Optional[Dict[str, Any]]:
        """Load configuration from YAML file"""
        file_path = self.base_path / 'config' / f"{config_name}.yaml"
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                self.logger.info(f"Loaded config from {file_path}")
                return data
            else:
                self.logger.warning(f"Config file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load config from {file_path}: {e}")
            return None
    
    def load_reference_data(self, data_type: str) -> Optional[Dict[str, Any]]:
        """Load reference data (train types, cost factors, etc.)"""
        file_path = self.base_path / 'input' / 'reference' / f"{data_type}.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded reference data from {file_path}")
                return data
            else:
                self.logger.warning(f"Reference data file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load reference data from {file_path}: {e}")
            return None
    
    def save_pipeline_output(self, project_name: str, output_data: Dict[str, Any], 
                           output_type: str = 'full') -> str:
        """Save pipeline output to file"""
        timestamp = self._get_timestamp()
        filename = f"{project_name}_{output_type}_{timestamp}.json"
        file_path = self.base_path / 'output' / filename
        
        try:
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved pipeline output to {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save pipeline output to {file_path}: {e}")
            raise
    
    def save_optimization_results(self, project_name: str, results: Dict[str, Any]) -> str:
        """Save optimization results to file"""
        timestamp = self._get_timestamp()
        filename = f"{project_name}_optimization_{timestamp}.json"
        file_path = self.base_path / 'output' / 'optimizations' / filename
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved optimization results to {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save optimization results to {file_path}: {e}")
            raise
    
    def load_previous_output(self, project_name: str, output_type: str = 'latest') -> Optional[Dict[str, Any]]:
        """Load previous pipeline output"""
        if output_type == 'latest':
            # Find the most recent file for this project
            output_dir = self.base_path / 'output'
            pattern = f"{project_name}_*.json"
            files = list(output_dir.glob(pattern))
            
            if not files:
                self.logger.warning(f"No previous output found for project {project_name}")
                return None
            
            latest_file = max(files, key=lambda x: x.stat().st_mtime)
            file_path = latest_file
        else:
            file_path = self.base_path / 'output' / f"{project_name}_{output_type}.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"Loaded previous output from {file_path}")
                return data
            else:
                self.logger.warning(f"Previous output file not found: {file_path}")
                return None
        except Exception as e:
            self.logger.error(f"Failed to load previous output from {file_path}: {e}")
            return None
    
    def save_cache(self, cache_key: str, data: Any) -> str:
        """Save data to cache"""
        import hashlib
        
        # Create hash of cache key for filename
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        file_path = self.base_path / 'cache' / f"{cache_hash}.json"
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            self.logger.debug(f"Saved cache entry: {cache_key} -> {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save cache entry {cache_key}: {e}")
            raise
    
    def load_cache(self, cache_key: str) -> Optional[Any]:
        """Load data from cache"""
        import hashlib
        
        cache_hash = hashlib.md5(cache_key.encode()).hexdigest()
        file_path = self.base_path / 'cache' / f"{cache_hash}.json"
        
        try:
            if file_path.exists():
                with open(file_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.debug(f"Loaded cache entry: {cache_key}")
                return data
            else:
                return None
        except Exception as e:
            self.logger.error(f"Failed to load cache entry {cache_key}: {e}")
            return None
    
    def clear_cache(self, older_than_days: int = 7):
        """Clear cache files older than specified days"""
        import time
        from datetime import datetime, timedelta
        
        cache_dir = self.base_path / 'cache'
        cutoff_time = time.time() - (older_than_days * 24 * 60 * 60)
        
        try:
            deleted_count = 0
            for cache_file in cache_dir.glob("*.json"):
                if cache_file.stat().st_mtime < cutoff_time:
                    cache_file.unlink()
                    deleted_count += 1
            
            self.logger.info(f"Cleared {deleted_count} cache files older than {older_than_days} days")
        except Exception as e:
            self.logger.error(f"Failed to clear cache: {e}")
    
    def export_to_csv(self, data: List[Dict[str, Any]], filename: str) -> str:
        """Export data to CSV format"""
        file_path = self.base_path / 'exports' / f"{filename}.csv"
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            if not data:
                self.logger.warning("No data to export to CSV")
                return str(file_path)
            
            # Get all unique keys from the data
            fieldnames = set()
            for item in data:
                fieldnames.update(item.keys())
            fieldnames = sorted(fieldnames)
            
            with open(file_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(data)
            
            self.logger.info(f"Exported {len(data)} rows to CSV: {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to export to CSV {file_path}: {e}")
            raise
    
    def load_csv_data(self, filename: str) -> List[Dict[str, Any]]:
        """Load data from CSV file"""
        file_path = self.base_path / 'input' / 'csv' / f"{filename}.csv"
        
        try:
            if not file_path.exists():
                self.logger.warning(f"CSV file not found: {file_path}")
                return []
            
            with open(file_path, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                data = [row for row in reader]
            
            self.logger.info(f"Loaded {len(data)} rows from CSV: {file_path}")
            return data
        except Exception as e:
            self.logger.error(f"Failed to load CSV data from {file_path}: {e}")
            return []
    
    def save_visualization_data(self, project_name: str, viz_data: Dict[str, Any], 
                              viz_type: str) -> str:
        """Save data specifically for visualization purposes"""
        timestamp = self._get_timestamp()
        filename = f"{project_name}_{viz_type}_{timestamp}.json"
        file_path = self.base_path / 'output' / 'visualizations' / filename
        
        try:
            file_path.parent.mkdir(parents=True, exist_ok=True)
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(viz_data, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"Saved visualization data to {file_path}")
            return str(file_path)
        except Exception as e:
            self.logger.error(f"Failed to save visualization data to {file_path}: {e}")
            raise
    
    def _get_timestamp(self) -> str:
        """Get current timestamp for filenames"""
        from datetime import datetime
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    def list_projects(self) -> List[str]:
        """List all available projects in output directory"""
        output_dir = self.base_path / 'output'
        projects = set()
        
        for file_path in output_dir.glob("*.json"):
            filename = file_path.stem
            # Extract project name (part before first underscore)
            project_name = filename.split('_')[0]
            projects.add(project_name)
        
        return sorted(list(projects))
    
    def get_project_files(self, project_name: str) -> List[Dict[str, Any]]:
        """Get all files for a specific project"""
        output_dir = self.base_path / 'output'
        project_files = []
        
        for file_path in output_dir.glob(f"{project_name}_*.json"):
            stat = file_path.stat()
            project_files.append({
                'filename': file_path.name,
                'file_path': str(file_path),
                'size_bytes': stat.st_size,
                'modified_time': stat.st_mtime,
                'file_type': self._classify_file_type(file_path.name)
            })
        
        return sorted(project_files, key=lambda x: x['modified_time'], reverse=True)
    
    def _classify_file_type(self, filename: str) -> str:
        """Classify file type based on filename"""
        if 'optimization' in filename:
            return 'optimization'
        elif 'visualization' in filename:
            return 'visualization'
        elif 'full' in filename:
            return 'full_output'
        else:
            return 'output'
    
    def cleanup_temp_files(self, older_than_hours: int = 24):
        """Clean up temporary files"""
        import time
        from datetime import datetime, timedelta
        
        temp_dir = self.base_path / 'temp'
        cutoff_time = time.time() - (older_than_hours * 60 * 60)
        
        try:
            deleted_count = 0
            for temp_file in temp_dir.glob("*"):
                if temp_file.stat().st_mtime < cutoff_time:
                    if temp_file.is_file():
                        temp_file.unlink()
                    else:
                        import shutil
                        shutil.rmtree(temp_file)
                    deleted_count += 1
            
            self.logger.info(f"Cleaned up {deleted_count} temporary files older than {older_than_hours} hours")
        except Exception as e:
            self.logger.error(f"Failed to clean up temporary files: {e}")