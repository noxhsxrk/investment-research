"""Cache management for stock analysis system.

This module provides a caching system for storing retrieved data with time-based
expiration, cache invalidation strategies, and size management.
"""

import os
import json
import time
import shutil
import pickle
import logging
import threading
from typing import Dict, Any, Optional, List, Tuple, Callable, Union
from datetime import datetime, timedelta
from pathlib import Path

from stock_analysis.utils.config import config
from stock_analysis.utils.exceptions import StockAnalysisError
from stock_analysis.utils.logging import get_logger

logger = get_logger(__name__)


class CacheError(StockAnalysisError):
    """Raised when cache operations fail."""
    
    def __init__(
        self, 
        message: str, 
        cache_key: Optional[str] = None,
        cache_type: Optional[str] = None,
        **kwargs
    ):
        context = kwargs.get('context', {})
        if cache_key:
            context['cache_key'] = cache_key
        if cache_type:
            context['cache_type'] = cache_type
        
        super().__init__(
            message, 
            error_code="CACHE_ERROR",
            context=context,
            **kwargs
        )


class CacheEntry:
    """Represents a single cache entry with metadata."""
    
    def __init__(self, key: str, value: Any, expiry: Optional[float] = None, 
                 data_type: str = "general", tags: Optional[List[str]] = None):
        """Initialize a cache entry.
        
        Args:
            key: Unique identifier for the cache entry
            value: The data to cache
            expiry: Timestamp when the entry expires (None for no expiration)
            data_type: Type of data for categorization (e.g., "stock_info", "historical")
            tags: List of tags for additional categorization
        """
        self.key = key
        self.value = value
        self.expiry = expiry
        self.created_at = time.time()
        self.last_accessed = self.created_at
        self.access_count = 0
        self.data_type = data_type
        self.tags = tags or []
        self.size_bytes = self._estimate_size()
    
    def is_expired(self) -> bool:
        """Check if the cache entry has expired.
        
        Returns:
            True if expired, False otherwise
        """
        if self.expiry is None:
            return False
        return time.time() > self.expiry
    
    def access(self) -> None:
        """Update access metadata when entry is accessed."""
        self.last_accessed = time.time()
        self.access_count += 1
    
    def _estimate_size(self) -> int:
        """Estimate the size of the cached value in bytes.
        
        Returns:
            Estimated size in bytes
        """
        try:
            # Use pickle to get a rough estimate of object size
            return len(pickle.dumps(self.value, protocol=pickle.HIGHEST_PROTOCOL))
        except Exception:
            # Fallback to a conservative estimate
            return 1024  # 1KB default


class CacheManager:
    """Manages caching of retrieved data with expiration and invalidation strategies."""
    
    _instance = None
    _lock = threading.RLock()
    
    def __new__(cls, *args, **kwargs):
        """Create a singleton instance of the cache manager."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(CacheManager, cls).__new__(cls)
                cls._instance._initialized = False
            return cls._instance
    
    def __init__(self, cache_dir: Optional[str] = None):
        """Initialize the cache manager.
        
        Args:
            cache_dir: Directory to store persistent cache files
        """
        # Only initialize once (singleton pattern)
        with self._lock:
            if self._initialized:
                return
            
            self._initialized = True
            self._memory_cache: Dict[str, CacheEntry] = {}
            self._disk_cache_enabled = config.get('stock_analysis.cache.use_disk_cache', True)
            self._cache_dir = cache_dir or config.get('stock_analysis.cache.directory', './.cache')
            self._max_memory_size = config.get('stock_analysis.cache.max_memory_size', 100 * 1024 * 1024)  # 100MB default
            self._max_disk_size = config.get('stock_analysis.cache.max_disk_size', 1024 * 1024 * 1024)  # 1GB default
            self._current_memory_size = 0
            self._cleanup_thread = None
            self._cleanup_interval = config.get('stock_analysis.cache.cleanup_interval', 3600)  # 1 hour default
            self._running = False
            
            # Default expiration times for different data types (in seconds)
            self._default_expiry = {
                "stock_info": 3600,  # 1 hour
                "historical_data": 86400,  # 24 hours
                "financial_statements": 86400 * 7,  # 1 week
                "news": 1800,  # 30 minutes
                "peer_data": 86400 * 7,  # 1 week
                "general": 3600  # 1 hour default
            }
            
            # Create cache directory if using disk cache
            if self._disk_cache_enabled:
                os.makedirs(self._cache_dir, exist_ok=True)
            
            # Register for configuration changes
            config.register_callback(self._on_config_change)
            
            # Start cleanup thread
            self.start_cleanup_thread()
    
    def _on_config_change(self) -> None:
        """Handle configuration changes."""
        with self._lock:
            # Update cache settings from config
            self._disk_cache_enabled = config.get('stock_analysis.cache.use_disk_cache', True)
            self._max_memory_size = config.get('stock_analysis.cache.max_memory_size', 100 * 1024 * 1024)
            self._max_disk_size = config.get('stock_analysis.cache.max_disk_size', 1024 * 1024 * 1024)
            self._cleanup_interval = config.get('stock_analysis.cache.cleanup_interval', 3600)
            
            # Update expiration times
            for data_type, default_time in self._default_expiry.items():
                config_key = f'stock_analysis.cache.expiry.{data_type}'
                self._default_expiry[data_type] = config.get(config_key, default_time)
            
            # Create cache directory if needed
            if self._disk_cache_enabled:
                os.makedirs(self._cache_dir, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get a value from the cache.
        
        Args:
            key: Cache key
            default: Default value if key not found or expired
            
        Returns:
            Cached value or default
        """
        with self._lock:
            # Check memory cache first
            if key in self._memory_cache:
                entry = self._memory_cache[key]
                
                # Check if expired
                if entry.is_expired():
                    self.invalidate(key)
                    return default
                
                # Update access metadata
                entry.access()
                return entry.value
            
            # Check disk cache if enabled
            if self._disk_cache_enabled:
                try:
                    disk_entry = self._load_from_disk(key)
                    if disk_entry:
                        # Check if expired
                        if disk_entry.is_expired():
                            self._remove_from_disk(key)
                            return default
                        
                        # Update access metadata and move to memory cache
                        disk_entry.access()
                        self._add_to_memory_cache(disk_entry)
                        return disk_entry.value
                except Exception as e:
                    logger.warning(f"Error loading from disk cache: {str(e)}")
            
            return default
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None, 
            data_type: str = "general", tags: Optional[List[str]] = None) -> None:
        """Set a value in the cache.
        
        Args:
            key: Cache key
            value: Value to cache
            ttl: Time to live in seconds (None for default based on data_type)
            data_type: Type of data for categorization
            tags: List of tags for additional categorization
        """
        with self._lock:
            # Calculate expiry time
            expiry = None
            if ttl is not None:
                expiry = time.time() + ttl
            elif data_type in self._default_expiry:
                expiry = time.time() + self._default_expiry[data_type]
            
            # Create cache entry
            entry = CacheEntry(key, value, expiry, data_type, tags)
            
            # Add to memory cache
            self._add_to_memory_cache(entry)
            
            # Add to disk cache if enabled
            if self._disk_cache_enabled:
                try:
                    self._save_to_disk(entry)
                except Exception as e:
                    logger.warning(f"Error saving to disk cache: {str(e)}")
    
    def _add_to_memory_cache(self, entry: CacheEntry) -> None:
        """Add an entry to the memory cache with size management.
        
        Args:
            entry: Cache entry to add
        """
        # If key already exists, remove its size first
        if entry.key in self._memory_cache:
            self._current_memory_size -= self._memory_cache[entry.key].size_bytes
        
        # Check if we need to make room
        if self._current_memory_size + entry.size_bytes > self._max_memory_size:
            self._evict_entries(entry.size_bytes)
        
        # Add to memory cache and update size
        self._memory_cache[entry.key] = entry
        self._current_memory_size += entry.size_bytes
    
    def _evict_entries(self, required_space: int) -> None:
        """Evict entries from memory cache to make room.
        
        Args:
            required_space: Amount of space needed in bytes
        """
        if not self._memory_cache:
            return
        
        # Sort entries by last accessed time (oldest first)
        entries = sorted(
            self._memory_cache.values(),
            key=lambda e: (e.last_accessed, -e.access_count)
        )
        
        # Evict entries until we have enough space
        space_freed = 0
        for entry in entries:
            if self._current_memory_size - space_freed + required_space <= self._max_memory_size:
                break
            
            space_freed += entry.size_bytes
            del self._memory_cache[entry.key]
            logger.debug(f"Evicted cache entry: {entry.key} ({entry.size_bytes} bytes)")
        
        self._current_memory_size -= space_freed
    
    def _save_to_disk(self, entry: CacheEntry) -> None:
        """Save a cache entry to disk.
        
        Args:
            entry: Cache entry to save
        """
        if not self._disk_cache_enabled:
            return
        
        # Create file path
        file_path = os.path.join(self._cache_dir, f"{entry.key}.cache")
        
        # Ensure directory exists
        os.makedirs(self._cache_dir, exist_ok=True)
        
        # Create subdirectories if key contains path separators
        dir_path = os.path.dirname(file_path)
        if dir_path and dir_path != self._cache_dir:
            os.makedirs(dir_path, exist_ok=True)
        
        # Save entry to disk
        with open(file_path, 'wb') as f:
            pickle.dump(entry, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Check disk cache size and clean up if needed
        self._check_disk_cache_size()
    
    def _load_from_disk(self, key: str) -> Optional[CacheEntry]:
        """Load a cache entry from disk.
        
        Args:
            key: Cache key
            
        Returns:
            Cache entry or None if not found
        """
        if not self._disk_cache_enabled:
            return None
        
        file_path = os.path.join(self._cache_dir, f"{key}.cache")
        
        if not os.path.exists(file_path):
            return None
        
        try:
            with open(file_path, 'rb') as f:
                return pickle.load(f)
        except Exception as e:
            logger.warning(f"Failed to load cache entry {key} from disk: {str(e)}")
            # Remove corrupted cache file
            try:
                os.remove(file_path)
            except Exception:
                pass
            return None
    
    def _remove_from_disk(self, key: str) -> None:
        """Remove a cache entry from disk.
        
        Args:
            key: Cache key
        """
        if not self._disk_cache_enabled:
            return
        
        file_path = os.path.join(self._cache_dir, f"{key}.cache")
        
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                logger.warning(f"Failed to remove cache file {file_path}: {str(e)}")
    
    def _check_disk_cache_size(self) -> None:
        """Check disk cache size and clean up if needed."""
        if not self._disk_cache_enabled or not os.path.exists(self._cache_dir):
            return
        
        try:
            # Get total size of cache directory
            total_size = sum(
                os.path.getsize(os.path.join(dirpath, filename))
                for dirpath, _, filenames in os.walk(self._cache_dir)
                for filename in filenames
                if filename.endswith('.cache')
            )
            
            # If over limit, clean up oldest files
            if total_size > self._max_disk_size:
                self._clean_disk_cache(total_size - self._max_disk_size)
        except Exception as e:
            logger.warning(f"Error checking disk cache size: {str(e)}")
    
    def _clean_disk_cache(self, bytes_to_free: int) -> None:
        """Clean up disk cache to free space.
        
        Args:
            bytes_to_free: Number of bytes to free
        """
        if not self._disk_cache_enabled:
            return
        
        try:
            # Get all cache files with their modification times
            cache_files = []
            for dirpath, _, filenames in os.walk(self._cache_dir):
                for filename in filenames:
                    if filename.endswith('.cache'):
                        file_path = os.path.join(dirpath, filename)
                        mtime = os.path.getmtime(file_path)
                        size = os.path.getsize(file_path)
                        cache_files.append((file_path, mtime, size))
            
            # Sort by modification time (oldest first)
            cache_files.sort(key=lambda x: x[1])
            
            # Delete files until we've freed enough space
            freed = 0
            for file_path, _, size in cache_files:
                if freed >= bytes_to_free:
                    break
                
                try:
                    os.remove(file_path)
                    freed += size
                    logger.debug(f"Removed cache file: {file_path} ({size} bytes)")
                except Exception as e:
                    logger.warning(f"Failed to remove cache file {file_path}: {str(e)}")
        except Exception as e:
            logger.warning(f"Error cleaning disk cache: {str(e)}")
    
    def invalidate(self, key: str) -> None:
        """Invalidate a specific cache entry.
        
        Args:
            key: Cache key to invalidate
        """
        with self._lock:
            # Remove from memory cache
            if key in self._memory_cache:
                self._current_memory_size -= self._memory_cache[key].size_bytes
                del self._memory_cache[key]
            
            # Remove from disk cache
            self._remove_from_disk(key)
    
    def invalidate_by_pattern(self, pattern: str) -> int:
        """Invalidate cache entries by key pattern.
        
        Args:
            pattern: Pattern to match against keys
            
        Returns:
            Number of entries invalidated
        """
        import fnmatch
        
        with self._lock:
            # Find matching keys in memory cache
            memory_keys = [k for k in self._memory_cache.keys() if fnmatch.fnmatch(k, pattern)]
            
            # Invalidate matching memory cache entries
            for key in memory_keys:
                self._current_memory_size -= self._memory_cache[key].size_bytes
                del self._memory_cache[key]
            
            # Find and invalidate matching disk cache entries
            disk_count = 0
            if self._disk_cache_enabled and os.path.exists(self._cache_dir):
                for dirpath, _, filenames in os.walk(self._cache_dir):
                    for filename in filenames:
                        if filename.endswith('.cache'):
                            # Get relative path from cache dir
                            rel_path = os.path.relpath(os.path.join(dirpath, filename), self._cache_dir)
                            # Extract key (remove .cache extension)
                            key = rel_path[:-6] if rel_path.endswith('.cache') else rel_path
                            if fnmatch.fnmatch(key, pattern):
                                file_path = os.path.join(dirpath, filename)
                                try:
                                    os.remove(file_path)
                                    disk_count += 1
                                except Exception as e:
                                    logger.warning(f"Failed to remove cache file {file_path}: {str(e)}")
            
            return len(memory_keys) + disk_count
    
    def invalidate_by_type(self, data_type: str) -> int:
        """Invalidate cache entries by data type.
        
        Args:
            data_type: Data type to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            # Find matching keys in memory cache
            memory_keys = [
                k for k, v in self._memory_cache.items() 
                if v.data_type == data_type
            ]
            
            # Invalidate matching memory cache entries
            for key in memory_keys:
                self._current_memory_size -= self._memory_cache[key].size_bytes
                del self._memory_cache[key]
            
            # Find and invalidate matching disk cache entries
            disk_count = 0
            if self._disk_cache_enabled and os.path.exists(self._cache_dir):
                for dirpath, _, filenames in os.walk(self._cache_dir):
                    for filename in filenames:
                        if filename.endswith('.cache'):
                            file_path = os.path.join(dirpath, filename)
                            try:
                                with open(file_path, 'rb') as f:
                                    entry = pickle.load(f)
                                    if entry.data_type == data_type:
                                        os.remove(file_path)
                                        disk_count += 1
                            except Exception:
                                # Skip corrupted files
                                pass
            
            return len(memory_keys) + disk_count
    
    def invalidate_by_tag(self, tag: str) -> int:
        """Invalidate cache entries by tag.
        
        Args:
            tag: Tag to invalidate
            
        Returns:
            Number of entries invalidated
        """
        with self._lock:
            # Find matching keys in memory cache
            memory_keys = [
                k for k, v in self._memory_cache.items() 
                if tag in v.tags
            ]
            
            # Invalidate matching memory cache entries
            for key in memory_keys:
                self._current_memory_size -= self._memory_cache[key].size_bytes
                del self._memory_cache[key]
            
            # Find and invalidate matching disk cache entries
            disk_count = 0
            if self._disk_cache_enabled and os.path.exists(self._cache_dir):
                for dirpath, _, filenames in os.walk(self._cache_dir):
                    for filename in filenames:
                        if filename.endswith('.cache'):
                            file_path = os.path.join(dirpath, filename)
                            try:
                                with open(file_path, 'rb') as f:
                                    entry = pickle.load(f)
                                    if tag in entry.tags:
                                        os.remove(file_path)
                                        disk_count += 1
                            except Exception:
                                # Skip corrupted files
                                pass
            
            return len(memory_keys) + disk_count
    
    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            # Clear memory cache
            self._memory_cache.clear()
            self._current_memory_size = 0
            
            # Clear disk cache
            if self._disk_cache_enabled and os.path.exists(self._cache_dir):
                try:
                    for filename in os.listdir(self._cache_dir):
                        if filename.endswith('.cache'):
                            file_path = os.path.join(self._cache_dir, filename)
                            try:
                                os.remove(file_path)
                            except Exception:
                                pass
                except Exception as e:
                    logger.warning(f"Error clearing disk cache: {str(e)}")
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics.
        
        Returns:
            Dictionary with cache statistics
        """
        with self._lock:
            memory_count = len(self._memory_cache)
            memory_size = self._current_memory_size
            
            # Count entries by data type
            type_counts = {}
            for entry in self._memory_cache.values():
                type_counts[entry.data_type] = type_counts.get(entry.data_type, 0) + 1
            
            # Calculate disk cache stats
            disk_count = 0
            disk_size = 0
            if self._disk_cache_enabled and os.path.exists(self._cache_dir):
                try:
                    for dirpath, _, filenames in os.walk(self._cache_dir):
                        for filename in filenames:
                            if filename.endswith('.cache'):
                                disk_count += 1
                                file_path = os.path.join(dirpath, filename)
                                disk_size += os.path.getsize(file_path)
                except Exception as e:
                    logger.warning(f"Error calculating disk cache stats: {str(e)}")
            
            return {
                "memory_entries": memory_count,
                "memory_size_bytes": memory_size,
                "memory_size_mb": memory_size / (1024 * 1024),
                "disk_entries": disk_count,
                "disk_size_bytes": disk_size,
                "disk_size_mb": disk_size / (1024 * 1024),
                "entries_by_type": type_counts
            }
    
    def start_cleanup_thread(self) -> None:
        """Start the background cleanup thread."""
        if self._running:
            return
        
        self._running = True
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True)
        self._cleanup_thread.start()
    
    def stop_cleanup_thread(self) -> None:
        """Stop the background cleanup thread."""
        self._running = False
        if self._cleanup_thread:
            self._cleanup_thread.join(timeout=1.0)
            self._cleanup_thread = None
    
    def _cleanup_loop(self) -> None:
        """Background thread for periodic cache cleanup."""
        while self._running:
            try:
                self._cleanup_expired()
            except Exception as e:
                logger.warning(f"Error in cache cleanup: {str(e)}")
            
            # Sleep for cleanup interval
            for _ in range(self._cleanup_interval):
                if not self._running:
                    break
                time.sleep(1)
    
    def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        with self._lock:
            # Clean up memory cache
            expired_keys = []
            for key, entry in self._memory_cache.items():
                if entry.is_expired():
                    expired_keys.append(key)
            
            for key in expired_keys:
                self._current_memory_size -= self._memory_cache[key].size_bytes
                del self._memory_cache[key]
            
            if expired_keys:
                logger.debug(f"Cleaned up {len(expired_keys)} expired memory cache entries")
            
            # Clean up disk cache
            if self._disk_cache_enabled and os.path.exists(self._cache_dir):
                try:
                    disk_expired = 0
                    for dirpath, _, filenames in os.walk(self._cache_dir):
                        for filename in filenames:
                            if filename.endswith('.cache'):
                                file_path = os.path.join(dirpath, filename)
                                try:
                                    with open(file_path, 'rb') as f:
                                        entry = pickle.load(f)
                                        if entry.is_expired():
                                            os.remove(file_path)
                                            disk_expired += 1
                                except Exception:
                                    # Remove corrupted files
                                    try:
                                        os.remove(file_path)
                                    except Exception:
                                        pass
                    
                    if disk_expired:
                        logger.debug(f"Cleaned up {disk_expired} expired disk cache entries")
                except Exception as e:
                    logger.warning(f"Error cleaning up disk cache: {str(e)}")


# Global cache manager instance
cache_manager = CacheManager()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager instance.
    
    Returns:
        Global CacheManager instance
    """
    return cache_manager