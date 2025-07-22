"""Unit tests for cache manager."""

import pytest
import time
import os
from unittest.mock import patch, MagicMock

from stock_analysis.utils.cache_manager import CacheManager, CacheEntry


@pytest.fixture
def cache_manager():
    """Create a cache manager instance for testing."""
    manager = CacheManager()
    manager.clear()  # Start with a clean cache
    return manager


class TestCacheManager:
    """Test cases for CacheManager."""
    
    def test_singleton_instance(self):
        """Test that CacheManager is a singleton."""
        manager1 = CacheManager()
        manager2 = CacheManager()
        assert manager1 is manager2
    
    def test_set_and_get(self, cache_manager):
        """Test basic set and get operations."""
        cache_manager.set("test_key", "test_value")
        assert cache_manager.get("test_key") == "test_value"
    
    def test_get_nonexistent(self, cache_manager):
        """Test getting nonexistent key."""
        assert cache_manager.get("nonexistent") is None
    
    def test_set_with_ttl(self, cache_manager):
        """Test setting value with TTL."""
        cache_manager.set("test_key", "test_value", ttl=0.1)  # 100ms TTL
        assert cache_manager.get("test_key") == "test_value"
        
        time.sleep(0.2)  # Wait for TTL to expire
        assert cache_manager.get("test_key") is None
    
    def test_exists(self, cache_manager):
        """Test key existence check."""
        cache_manager.set("test_key", "test_value")
        assert cache_manager.exists("test_key") is True
        assert cache_manager.exists("nonexistent") is False
    
    def test_delete(self, cache_manager):
        """Test deleting cache entry."""
        cache_manager.set("test_key", "test_value")
        cache_manager.delete("test_key")
        assert cache_manager.get("test_key") is None
    
    def test_clear(self, cache_manager):
        """Test clearing all cache entries."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.clear()
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
    
    def test_set_with_tags(self, cache_manager):
        """Test setting value with tags."""
        cache_manager.set("key1", "value1", tags=["tag1", "tag2"])
        cache_manager.set("key2", "value2", tags=["tag2", "tag3"])
        
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"
    
    def test_invalidate_by_tag(self, cache_manager):
        """Test invalidating entries by tag."""
        cache_manager.set("key1", "value1", tags=["tag1", "tag2"])
        cache_manager.set("key2", "value2", tags=["tag2", "tag3"])
        
        cache_manager.invalidate_by_tag("tag2")
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
    
    def test_invalidate_by_pattern(self, cache_manager):
        """Test invalidating entries by pattern."""
        cache_manager.set("test:key1", "value1")
        cache_manager.set("test:key2", "value2")
        cache_manager.set("other:key3", "value3")
        
        cache_manager.invalidate_by_pattern("test:*")
        assert cache_manager.get("test:key1") is None
        assert cache_manager.get("test:key2") is None
        assert cache_manager.get("other:key3") == "value3"
    
    def test_invalidate_by_type(self, cache_manager):
        """Test invalidating entries by type."""
        cache_manager.set("stock:AAPL", "value1", type="stock_info")
        cache_manager.set("stock:MSFT", "value2", type="stock_info")
        cache_manager.set("news:AAPL", "value3", type="news")
        
        cache_manager.invalidate_by_type("stock_info")
        assert cache_manager.get("stock:AAPL") is None
        assert cache_manager.get("stock:MSFT") is None
        assert cache_manager.get("news:AAPL") == "value3"
    
    def test_get_by_pattern(self, cache_manager):
        """Test getting entries by pattern."""
        cache_manager.set("test:key1", "value1")
        cache_manager.set("test:key2", "value2")
        cache_manager.set("other:key3", "value3")
        
        test_entries = cache_manager.get_by_pattern("test:*")
        assert len(test_entries) == 2
        assert "value1" in [entry.value for entry in test_entries]
        assert "value2" in [entry.value for entry in test_entries]
    
    def test_get_by_tag(self, cache_manager):
        """Test getting entries by tag."""
        cache_manager.set("key1", "value1", tags=["tag1", "tag2"])
        cache_manager.set("key2", "value2", tags=["tag2", "tag3"])
        
        tag2_entries = cache_manager.get_by_tag("tag2")
        assert len(tag2_entries) == 2
        assert "value1" in [entry.value for entry in tag2_entries]
        assert "value2" in [entry.value for entry in tag2_entries]
    
    def test_get_by_type(self, cache_manager):
        """Test getting entries by type."""
        cache_manager.set("stock:AAPL", "value1", type="stock_info")
        cache_manager.set("stock:MSFT", "value2", type="stock_info")
        cache_manager.set("news:AAPL", "value3", type="news")
        
        stock_entries = cache_manager.get_by_type("stock_info")
        assert len(stock_entries) == 2
        assert "value1" in [entry.value for entry in stock_entries]
        assert "value2" in [entry.value for entry in stock_entries]
    
    def test_get_stats(self, cache_manager):
        """Test getting cache statistics."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.get("key1")  # One hit
        cache_manager.get("nonexistent")  # One miss
        
        stats = cache_manager.get_stats()
        assert stats['total_entries'] == 2
        assert stats['hits'] == 1
        assert stats['misses'] == 1
    
    def test_cleanup_expired(self, cache_manager):
        """Test cleaning up expired entries."""
        cache_manager.set("key1", "value1", ttl=0.1)  # 100ms TTL
        cache_manager.set("key2", "value2")  # No TTL
        
        time.sleep(0.2)  # Wait for key1 to expire
        cache_manager.cleanup_expired()
        
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") == "value2"
    
    def test_set_many(self, cache_manager):
        """Test setting multiple entries at once."""
        entries = {
            "key1": "value1",
            "key2": "value2",
            "key3": "value3"
        }
        
        cache_manager.set_many(entries)
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"
        assert cache_manager.get("key3") == "value3"
    
    def test_get_many(self, cache_manager):
        """Test getting multiple entries at once."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")
        
        values = cache_manager.get_many(["key1", "key2", "nonexistent"])
        assert values == {"key1": "value1", "key2": "value2", "nonexistent": None}
    
    def test_delete_many(self, cache_manager):
        """Test deleting multiple entries at once."""
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        cache_manager.set("key3", "value3")
        
        cache_manager.delete_many(["key1", "key2"])
        assert cache_manager.get("key1") is None
        assert cache_manager.get("key2") is None
        assert cache_manager.get("key3") == "value3"
    
    def test_cache_entry_serialization(self, cache_manager):
        """Test cache entry serialization."""
        entry = CacheEntry(
            key="test_key",
            value="test_value",
            created_at=time.time(),
            ttl=60,
            type="test_type",
            tags=["tag1", "tag2"]
        )
        
        # Should be able to serialize and deserialize
        serialized = entry.to_dict()
        deserialized = CacheEntry.from_dict(serialized)
        
        assert deserialized.key == entry.key
        assert deserialized.value == entry.value
        assert deserialized.type == entry.type
        assert deserialized.tags == entry.tags
    
    def test_cache_persistence(self, cache_manager, tmp_path):
        """Test cache persistence to disk."""
        cache_file = tmp_path / "test_cache.json"
        
        # Set some test data
        cache_manager.set("key1", "value1")
        cache_manager.set("key2", "value2")
        
        # Save to disk
        cache_manager.save_to_disk(str(cache_file))
        
        # Clear memory cache
        cache_manager.clear()
        
        # Load from disk
        cache_manager.load_from_disk(str(cache_file))
        
        assert cache_manager.get("key1") == "value1"
        assert cache_manager.get("key2") == "value2"