"""Unit tests for the cache manager."""

import os
import time
import shutil
import unittest
from unittest.mock import patch, MagicMock
import pandas as pd
import numpy as np
from datetime import datetime

from stock_analysis.utils.cache_manager import CacheManager, CacheEntry, get_cache_manager
from stock_analysis.utils.config import ConfigManager


class TestCacheEntry(unittest.TestCase):
    """Test the CacheEntry class."""
    
    def test_init(self):
        """Test initialization of cache entry."""
        entry = CacheEntry("test_key", "test_value", time.time() + 60, "test_type", ["tag1", "tag2"])
        self.assertEqual(entry.key, "test_key")
        self.assertEqual(entry.value, "test_value")
        self.assertEqual(entry.data_type, "test_type")
        self.assertEqual(entry.tags, ["tag1", "tag2"])
        self.assertGreater(entry.size_bytes, 0)
    
    def test_is_expired(self):
        """Test expiration check."""
        # Not expired
        entry = CacheEntry("test_key", "test_value", time.time() + 60)
        self.assertFalse(entry.is_expired())
        
        # Expired
        entry = CacheEntry("test_key", "test_value", time.time() - 60)
        self.assertTrue(entry.is_expired())
        
        # No expiration
        entry = CacheEntry("test_key", "test_value", None)
        self.assertFalse(entry.is_expired())
    
    def test_access(self):
        """Test access tracking."""
        entry = CacheEntry("test_key", "test_value")
        initial_access_time = entry.last_accessed
        initial_count = entry.access_count
        
        time.sleep(0.01)  # Small delay to ensure time difference
        entry.access()
        
        self.assertGreater(entry.last_accessed, initial_access_time)
        self.assertEqual(entry.access_count, initial_count + 1)


class TestCacheManager(unittest.TestCase):
    """Test the CacheManager class."""
    
    def setUp(self):
        """Set up test environment."""
        # Create a test cache directory
        self.test_cache_dir = "./.test_cache"
        os.makedirs(self.test_cache_dir, exist_ok=True)
        
        # Create a test config
        self.config_mock = MagicMock()
        self.config_mock.get.side_effect = self._mock_config_get
        
        # Create a cache manager with test configuration
        with patch('stock_analysis.utils.cache_manager.config', self.config_mock):
            self.cache_manager = CacheManager(self.test_cache_dir)
            # Reset the singleton instance for testing
            CacheManager._instance = None
    
    def tearDown(self):
        """Clean up after tests."""
        # Stop cleanup thread
        self.cache_manager.stop_cleanup_thread()
        
        # Remove test cache directory
        if os.path.exists(self.test_cache_dir):
            shutil.rmtree(self.test_cache_dir)
    
    def _mock_config_get(self, key, default=None):
        """Mock config.get method."""
        config_values = {
            'stock_analysis.cache.use_disk_cache': True,
            'stock_analysis.cache.directory': self.test_cache_dir,
            'stock_analysis.cache.max_memory_size': 1024 * 1024,  # 1MB
            'stock_analysis.cache.max_disk_size': 10 * 1024 * 1024,  # 10MB
            'stock_analysis.cache.cleanup_interval': 1,  # 1 second for faster testing
            'stock_analysis.cache.expiry.stock_info': 60,  # 1 minute
            'stock_analysis.cache.expiry.historical_data': 300,  # 5 minutes
            'stock_analysis.cache.expiry.financial_statements': 600,  # 10 minutes
            'stock_analysis.cache.expiry.news': 30,  # 30 seconds
            'stock_analysis.cache.expiry.peer_data': 600,  # 10 minutes
            'stock_analysis.cache.expiry.general': 60,  # 1 minute
        }
        return config_values.get(key, default)
    
    def test_singleton_pattern(self):
        """Test that CacheManager is a singleton."""
        with patch('stock_analysis.utils.cache_manager.config', self.config_mock):
            cache1 = CacheManager()
            cache2 = CacheManager()
            self.assertIs(cache1, cache2)
    
    def test_get_cache_manager(self):
        """Test get_cache_manager function."""
        with patch('stock_analysis.utils.cache_manager.config', self.config_mock):
            cache = get_cache_manager()
            self.assertIsInstance(cache, CacheManager)
    
    def test_set_and_get(self):
        """Test setting and getting values."""
        # Set a value
        self.cache_manager.set("test_key", "test_value")
        
        # Get the value
        value = self.cache_manager.get("test_key")
        self.assertEqual(value, "test_value")
        
        # Get a non-existent key
        value = self.cache_manager.get("non_existent_key")
        self.assertIsNone(value)
        
        # Get a non-existent key with default
        value = self.cache_manager.get("non_existent_key", "default_value")
        self.assertEqual(value, "default_value")
    
    def test_expiration(self):
        """Test cache entry expiration."""
        # Set a value with short expiration
        self.cache_manager.set("expiring_key", "expiring_value", ttl=1)  # 1 second
        
        # Get the value immediately
        value = self.cache_manager.get("expiring_key")
        self.assertEqual(value, "expiring_value")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Get the value after expiration
        value = self.cache_manager.get("expiring_key")
        self.assertIsNone(value)
    
    def test_invalidate(self):
        """Test invalidating cache entries."""
        # Set some values
        self.cache_manager.set("key1", "value1")
        self.cache_manager.set("key2", "value2")
        
        # Invalidate one key
        self.cache_manager.invalidate("key1")
        
        # Check values
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertEqual(self.cache_manager.get("key2"), "value2")
    
    def test_invalidate_by_pattern(self):
        """Test invalidating cache entries by pattern."""
        # Set some values
        self.cache_manager.set("prefix_key1", "value1")
        self.cache_manager.set("prefix_key2", "value2")
        self.cache_manager.set("other_key", "value3")
        
        # Invalidate by pattern
        count = self.cache_manager.invalidate_by_pattern("prefix_*")
        
        # Check values
        self.assertIsNone(self.cache_manager.get("prefix_key1"))
        self.assertIsNone(self.cache_manager.get("prefix_key2"))
        self.assertEqual(self.cache_manager.get("other_key"), "value3")
        self.assertEqual(count, 2)
    
    def test_invalidate_by_type(self):
        """Test invalidating cache entries by data type."""
        # Set some values with different types
        self.cache_manager.set("key1", "value1", data_type="type1")
        self.cache_manager.set("key2", "value2", data_type="type2")
        self.cache_manager.set("key3", "value3", data_type="type1")
        
        # Invalidate by type
        count = self.cache_manager.invalidate_by_type("type1")
        
        # Check values
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertEqual(self.cache_manager.get("key2"), "value2")
        self.assertIsNone(self.cache_manager.get("key3"))
        self.assertEqual(count, 2)
    
    def test_invalidate_by_tag(self):
        """Test invalidating cache entries by tag."""
        # Set some values with different tags
        self.cache_manager.set("key1", "value1", tags=["tag1", "tag2"])
        self.cache_manager.set("key2", "value2", tags=["tag2", "tag3"])
        self.cache_manager.set("key3", "value3", tags=["tag3", "tag4"])
        
        # Invalidate by tag
        count = self.cache_manager.invalidate_by_tag("tag2")
        
        # Check values
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))
        self.assertEqual(self.cache_manager.get("key3"), "value3")
        self.assertEqual(count, 2)
    
    def test_clear(self):
        """Test clearing all cache entries."""
        # Set some values
        self.cache_manager.set("key1", "value1")
        self.cache_manager.set("key2", "value2")
        
        # Clear cache
        self.cache_manager.clear()
        
        # Check values
        self.assertIsNone(self.cache_manager.get("key1"))
        self.assertIsNone(self.cache_manager.get("key2"))
    
    def test_get_stats(self):
        """Test getting cache statistics."""
        # Set some values with different types
        self.cache_manager.set("key1", "value1", data_type="type1")
        self.cache_manager.set("key2", "value2", data_type="type2")
        self.cache_manager.set("key3", "value3", data_type="type1")
        
        # Get stats
        stats = self.cache_manager.get_stats()
        
        # Check stats
        self.assertEqual(stats["memory_entries"], 3)
        self.assertGreater(stats["memory_size_bytes"], 0)
        self.assertEqual(stats["entries_by_type"]["type1"], 2)
        self.assertEqual(stats["entries_by_type"]["type2"], 1)
    
    def test_memory_eviction(self):
        """Test memory cache eviction when size limit is reached."""
        # Create a large value (approximately 500KB)
        large_value = "x" * 500000
        
        # Set values to fill cache
        self.cache_manager.set("key1", large_value)
        self.cache_manager.set("key2", large_value)
        
        # Access key1 to make it more recently used
        self.cache_manager.get("key1")
        
        # Add another large value to trigger eviction
        self.cache_manager.set("key3", large_value)
        
        # Check values (key2 should be evicted as least recently used)
        self.assertEqual(self.cache_manager.get("key1"), large_value)
        self.assertIsNone(self.cache_manager.get("key2"))
        self.assertEqual(self.cache_manager.get("key3"), large_value)
    
    def test_disk_cache(self):
        """Test disk cache functionality."""
        # Set a value that should be saved to disk
        self.cache_manager.set("disk_key", "disk_value")
        
        # Check that file exists
        cache_file = os.path.join(self.test_cache_dir, "disk_key.cache")
        self.assertTrue(os.path.exists(cache_file))
        
        # Create a new cache manager (simulating restart)
        with patch('stock_analysis.utils.cache_manager.config', self.config_mock):
            CacheManager._instance = None
            new_cache = CacheManager(self.test_cache_dir)
        
        # Memory cache should be empty in new instance
        self.assertEqual(len(new_cache._memory_cache), 0)
        
        # Get value (should load from disk)
        value = new_cache.get("disk_key")
        self.assertEqual(value, "disk_value")
        
        # Now it should be in memory cache
        self.assertEqual(len(new_cache._memory_cache), 1)
    
    def test_cleanup_expired(self):
        """Test cleanup of expired entries."""
        # Set values with short expiration
        self.cache_manager.set("expiring_key1", "value1", ttl=1)
        self.cache_manager.set("expiring_key2", "value2", ttl=1)
        self.cache_manager.set("non_expiring_key", "value3")
        
        # Wait for expiration
        time.sleep(1.1)
        
        # Manually trigger cleanup
        self.cache_manager._cleanup_expired()
        
        # Check values
        self.assertIsNone(self.cache_manager.get("expiring_key1"))
        self.assertIsNone(self.cache_manager.get("expiring_key2"))
        self.assertEqual(self.cache_manager.get("non_expiring_key"), "value3")
    
    def test_complex_data_types(self):
        """Test caching complex data types."""
        # Create a pandas DataFrame
        df = pd.DataFrame({
            'A': np.random.rand(100),
            'B': np.random.rand(100),
            'C': pd.date_range('2023-01-01', periods=100)
        })
        
        # Set in cache
        self.cache_manager.set("dataframe_key", df)
        
        # Get from cache
        cached_df = self.cache_manager.get("dataframe_key")
        
        # Check that it's the same
        pd.testing.assert_frame_equal(df, cached_df)
        
        # Test with a dictionary containing mixed types
        complex_dict = {
            'string': 'value',
            'number': 123.456,
            'date': datetime.now(),
            'list': [1, 2, 3, 'four'],
            'nested': {'a': 1, 'b': 2},
            'dataframe': df
        }
        
        # Set in cache
        self.cache_manager.set("complex_key", complex_dict)
        
        # Get from cache
        cached_dict = self.cache_manager.get("complex_key")
        
        # Check key elements
        self.assertEqual(cached_dict['string'], complex_dict['string'])
        self.assertEqual(cached_dict['number'], complex_dict['number'])
        self.assertEqual(cached_dict['list'], complex_dict['list'])
        self.assertEqual(cached_dict['nested'], complex_dict['nested'])
        pd.testing.assert_frame_equal(cached_dict['dataframe'], complex_dict['dataframe'])


if __name__ == '__main__':
    unittest.main()