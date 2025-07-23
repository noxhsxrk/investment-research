"""Tests for enhanced cache manager functionality."""

import os
import time
import unittest
from unittest.mock import patch, MagicMock
import tempfile
import shutil

from stock_analysis.utils.cache_manager import CacheManager, CacheEntry


class TestEnhancedCacheManager(unittest.TestCase):
    """Test cases for enhanced cache manager functionality."""
    
    def setUp(self):
        """Set up test environment."""
        # Create temporary directory for cache
        self.temp_dir = tempfile.mkdtemp()
        
        # Reset singleton instance
        CacheManager._instance = None
        CacheManager._initialized = False
        
        # Create cache manager with test directory
        with patch('stock_analysis.utils.cache_manager.config') as mock_config:
            mock_config.get.side_effect = self._mock_config_get
            self.cache_manager = CacheManager(self.temp_dir)
    
    def tearDown(self):
        """Clean up test environment."""
        # Remove temporary directory
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def _mock_config_get(self, key, default=None):
        """Mock config.get method."""
        config_values = {
            'stock_analysis.cache.use_disk_cache': True,
            'stock_analysis.cache.max_memory_size': 1024 * 1024,  # 1MB
            'stock_analysis.cache.max_disk_size': 10 * 1024 * 1024,  # 10MB
            'stock_analysis.cache.cleanup_interval': 1,
            'stock_analysis.cache.hot_cache_size': 512 * 1024,  # 512KB
            'stock_analysis.cache.hot_cache_threshold': 3,
            'stock_analysis.cache.hot_cache_ttl_multiplier': 2.0,
            'stock_analysis.cache.access_tracking_window': 60
        }
        return config_values.get(key, default)
    
    def test_hot_cache_tracking(self):
        """Test hot cache access tracking."""
        # Set some values
        self.cache_manager.set('key1', 'value1', ttl=10)
        self.cache_manager.set('key2', 'value2', ttl=10)
        
        # Access key1 multiple times to make it hot
        for _ in range(5):
            self.cache_manager.get('key1')
        
        # Access key2 only once
        self.cache_manager.get('key2')
        
        # Verify access counts
        self.assertEqual(len(self.cache_manager._access_counts.get('key1', [])), 5)
        self.assertEqual(len(self.cache_manager._access_counts.get('key2', [])), 1)
    
    def test_hot_cache_ttl_extension(self):
        """Test TTL extension for hot cache items."""
        # Set a value with short TTL
        self.cache_manager.set('hot_key', 'hot_value', ttl=10)
        
        # Get original expiry time
        original_expiry = self.cache_manager._memory_cache['hot_key'].expiry
        
        # Access multiple times to make it hot
        for _ in range(5):
            self.cache_manager.get('hot_key')
        
        # Get new expiry time
        new_expiry = self.cache_manager._memory_cache['hot_key'].expiry
        
        # Verify TTL was extended
        self.assertGreater(new_expiry, original_expiry)
    
    def test_eviction_preserves_hot_items(self):
        """Test that eviction preserves hot items when possible."""
        # Create a large item that will trigger eviction
        large_data = 'x' * 400 * 1024  # 400KB
        
        # Set some small values and access them to make them hot
        for i in range(5):
            key = f'hot_key_{i}'
            self.cache_manager.set(key, f'hot_value_{i}', ttl=60)
            
            # Access multiple times to make it hot
            for _ in range(5):
                self.cache_manager.get(key)
        
        # Set some cold values
        for i in range(5):
            self.cache_manager.set(f'cold_key_{i}', f'cold_value_{i}', ttl=60)
        
        # Get current cache size before adding large item
        initial_size = self.cache_manager._current_memory_size
        
        # Add large item to trigger eviction
        self.cache_manager.set('large_key', large_data, ttl=60)
        
        # Verify some items were evicted
        self.assertLess(self.cache_manager._current_memory_size, initial_size + len(large_data))
        
        # Check if hot items are still in cache
        hot_items_remaining = sum(1 for i in range(5) if f'hot_key_{i}' in self.cache_manager._memory_cache)
        cold_items_remaining = sum(1 for i in range(5) if f'cold_key_{i}' in self.cache_manager._memory_cache)
        
        # Hot items should be preserved over cold items
        self.assertGreater(hot_items_remaining, cold_items_remaining)
    
    def test_access_tracking_window(self):
        """Test that access tracking respects the time window."""
        # Set a value
        self.cache_manager.set('window_key', 'window_value', ttl=60)
        
        # Mock time to simulate access at different times
        with patch('time.time') as mock_time:
            # Set current time
            current_time = time.time()
            mock_time.return_value = current_time
            
            # Access once
            self.cache_manager.get('window_key')
            
            # Access again 30 seconds later (within window)
            mock_time.return_value = current_time + 30
            self.cache_manager.get('window_key')
            
            # Access again 90 seconds later (outside window)
            mock_time.return_value = current_time + 90
            self.cache_manager.get('window_key')
            
            # Verify only accesses within window are counted
            self.assertEqual(len(self.cache_manager._access_counts['window_key']), 2)
            
            # Verify oldest access was removed
            self.assertGreaterEqual(min(self.cache_manager._access_counts['window_key']), current_time + 30)
    
    def test_cache_stats_includes_hot_cache_info(self):
        """Test that cache stats include hot cache information."""
        # Set some values
        self.cache_manager.set('hot_key', 'hot_value', ttl=60)
        self.cache_manager.set('cold_key', 'cold_value', ttl=60)
        
        # Access hot_key multiple times to make it hot
        for _ in range(5):
            self.cache_manager.get('hot_key')
        
        # Get cache stats
        stats = self.cache_manager.get_stats()
        
        # Verify hot cache stats are included
        self.assertIn('hot_cache_entries', stats)
        self.assertIn('hot_cache_size_bytes', stats)
        self.assertIn('hot_cache_size_mb', stats)
        self.assertIn('hot_cache_threshold', stats)
        self.assertIn('hot_cache_ttl_multiplier', stats)
        
        # Verify hot cache entries count
        self.assertEqual(stats['hot_cache_entries'], 1)
    
    def test_clear_removes_access_tracking(self):
        """Test that clear method removes access tracking data."""
        # Set a value and access it
        self.cache_manager.set('test_key', 'test_value', ttl=60)
        self.cache_manager.get('test_key')
        
        # Verify access tracking exists
        self.assertIn('test_key', self.cache_manager._access_counts)
        
        # Clear cache
        self.cache_manager.clear()
        
        # Verify access tracking was cleared
        self.assertNotIn('test_key', self.cache_manager._access_counts)
    
    def test_invalidate_removes_access_tracking(self):
        """Test that invalidate method removes access tracking data."""
        # Set a value and access it
        self.cache_manager.set('test_key', 'test_value', ttl=60)
        self.cache_manager.get('test_key')
        
        # Verify access tracking exists
        self.assertIn('test_key', self.cache_manager._access_counts)
        
        # Invalidate key
        self.cache_manager.invalidate('test_key')
        
        # Verify access tracking was removed
        self.assertNotIn('test_key', self.cache_manager._access_counts)


if __name__ == '__main__':
    unittest.main()