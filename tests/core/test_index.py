"""Tests for the Index class."""

import pytest
from tensor_network_library.core.index import Index


class TestIndexCreation:
    """Test Index creation and basic properties."""
    
    def test_index_creation(self):
        """Test creating an Index with basic parameters."""
        idx = Index(dim=10, name="test_index", tags=frozenset({"tag1", "tag2"}), prime=0)
        assert idx.dim == 10
        assert idx.name == "test_index"
        assert idx.tags == frozenset({"tag1", "tag2"})
        assert idx.prime == 0
        assert idx.id is not None
    
    def test_index_default_values(self):
        """Test Index with default parameters."""
        idx = Index(dim=5)
        assert idx.dim == 5
        assert idx.name == "Index"
        assert idx.tags == frozenset()
        assert idx.prime == 0
        assert idx.id is not None
    
    def test_index_uniqueness(self):
        """Test that two independently created indices have different ids."""
        idx1 = Index(dim=5, name="idx")
        idx2 = Index(dim=5, name="idx")
        assert idx1.id != idx2.id

class TestIndexEquality:
    """Test equality based on id, tags, and prime."""
    
    def test_equality_same_id(self):
        """Two indices with same id are equal."""
        idx1 = Index(dim=5, name="idx")
        idx2 = Index(dim=5, name="other_name", prime=0, id=idx1.id)
        assert idx1 == idx2
    
    def test_inequality_different_id(self):
        """Two indices with different ids are not equal."""
        idx1 = Index(dim=5)
        idx2 = Index(dim=5)
        assert idx1 != idx2
    
    def test_inequality_different_prime(self):
        """Two indices with same id but different prime are not equal."""
        idx1 = Index(dim=5, name="idx")
        idx2 = idx1.prime_id(increment=1)
        assert idx1 != idx2
    
    def test_inequality_different_tags(self):
        """Two indices with same id but different tags are not equal."""
        idx1 = Index(dim=5, name="idx", tags=frozenset({"tag1"}))
        idx2 = Index(dim=5, name="idx", tags=frozenset({"tag1", "tag2"}), prime=0, id=idx1.id
        )
        assert idx1 != idx2

class TestPriming:
    """Test priming operations."""
    
    def test_prime_increment(self):
        """Test prime level increment."""
        idx = Index(dim=5, name="idx", prime=0)
        idx_primed = idx.prime_id(increment=1)
        assert idx_primed.prime == 1
        assert idx_primed.dim == 5
        assert idx_primed.name == "idx"
        assert idx_primed.id == idx.id
    
    def test_prime_multiple_increments(self):
        """Test multiple prime increments."""
        idx = Index(dim=5, name="idx", prime=0)
        idx_pp = idx.prime_id(increment=2)
        assert idx_pp.prime == 2
    
    def test_noprime(self):
        """Test removing prime."""
        idx = Index(dim=5, name="idx", prime=3)
        idx_noprime = idx.no_prime_id()
        assert idx_noprime.prime == 0
        assert idx_noprime.id == idx.id
    
    def test_negative_prime_clamp(self):
        """Test that negative prime is clamped to 0."""
        idx = Index(dim=5, prime=0)
        idx_neg = idx.prime_id(increment=-5)
        assert idx_neg.prime == 0

class TestTags:
    """Test tag operations."""
    
    def test_add_tags(self):
        """Test adding tags to an index."""
        idx = Index(dim=5, tags=frozenset({"tag1"}))
        idx_tagged = idx.add_tags("tag2", "tag3")
        assert idx_tagged.tags == frozenset({"tag1", "tag2", "tag3"})
        assert idx_tagged.id == idx.id
    
    def test_remove_tags(self):
        """Test removing tags from an index."""
        idx = Index(dim=5, tags=frozenset({"tag1", "tag2", "tag3"}))
        idx_removed = idx.remove_tags("tag2")
        assert idx_removed.tags == frozenset({"tag1", "tag3"})
        assert idx_removed.id == idx.id
    
    def test_remove_nonexistent_tag(self):
        """Test removing a tag that doesn't exist."""
        idx = Index(dim=5, tags=frozenset({"tag1"}))
        idx_removed = idx.remove_tags("nonexistent")
        assert idx_removed.tags == frozenset({"tag1"})   

class TestHashing:
    """Test that indices can be used as dictionary keys."""
    
    def test_hash_in_dict(self):
        """Test using Index as dictionary key."""
        idx = Index(dim=5, name="test")
        d = {idx: "value"}
        assert d[idx] == "value"
    
    def test_hash_in_set(self):
        """Test using Index in a set."""
        idx1 = Index(dim=5, name="test")
        idx2 = idx1.prime_id()
        
        s = {idx1, idx2}
        assert len(s) == 2
        assert idx1 in s
        assert idx2 in s

class TestSimilarIndex:
    """Test creating similar indices with fresh ids."""
    
    def test_sim(self):
        """Test sim() creates a new index with same properties but different id."""
        idx = Index(dim=5, name="test", tags=frozenset({"tag"}), prime=1)
        idx_sim = idx.sim()
        
        assert idx_sim.dim == idx.dim
        assert idx_sim.name == idx.name
        assert idx_sim.tags == idx.tags
        assert idx_sim.prime == idx.prime
        assert idx_sim.id != idx.id
    
    def test_sim_inequality(self):
        """Test that sim'd index is not equal to original."""
        idx = Index(dim=5, name="test")
        idx_sim = idx.sim()
        assert idx != idx_sim
        
if __name__ == "__main__":
    pytest.main([__file__, "-v"])