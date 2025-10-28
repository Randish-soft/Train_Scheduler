import logging
import pickle
import hashlib
from typing import Dict, Any, Optional
from pathlib import Path

class CacheManager:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger(__name__)
        self.cache_dir = Path(config.get('cache', {}).get('directory', 'cache'))
        self.cache_dir.mkdir(exist_ok=True)
        self.max_size = config.get('cache', {}).get('max_size_mb', 100)
    
    def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get cached result by key"""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'rb') as f:
                    cached_data = pickle.load(f)
                self.logger.debug(f"Cache hit for key: {key}")
                return cached_data
            except Exception as e:
                self.logger.warning(f"Failed to load cache for key {key}: {e}")
        
        return None
    
    def set(self, key: str, data: Dict[str, Any]) -> None:
        """Cache result with key"""
        cache_file = self.cache_dir / f"{self._hash_key(key)}.pkl"
        
        try:
            with open(cache_file, 'wb') as f:
                pickle.dump(data, f)
            self.logger.debug(f"Cached result for key: {key}")
        except Exception as e:
            self.logger.warning(f"Failed to cache data for key {key}: {e}")
    
    def clear(self) -> None:
        """Clear all cached data"""
        for cache_file in self.cache_dir.glob("*.pkl"):
            cache_file.unlink()
        self.logger.info("Cache cleared")
    
    def _hash_key(self, key: str) -> str:
        """Create hash of cache key"""
        return hashlib.md5(key.encode()).hexdigest()
    
    def _cleanup_old_cache(self) -> None:
        """Clean up old cache files if cache size exceeds limit"""
        # Implementation for cache size management
        pass