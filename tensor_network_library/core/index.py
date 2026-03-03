"""
Index objects for tensor network indices for the connectivity of thensors.

Here we follow Schollwöck's convention for the MPSs and MPOs:
    - MPS site tensor: (left_bond, physical, right_bond)
    - MPO site tensor: (left_bond, physical_in, physical_out, right_bond)

Index object will carry identity (id), dimension, tags and prime level.
Two index objects are equal if and only if they share the same id, tags and prime.
"""

from __future__ import annotations
from dataclasses import dataclass , field
from typing import FrozenSet
import uuid

@dataclass(frozen=True)
class Index:
    """
    An immutable index object representing a vector space leg in a tensor.

    Attributes:
        dim (int): Dimesnion of this index (size of vector space)
        name (str): human-readable name (e.g., "Site", "Link", "Left")
        tags (frozenset[str]): Additional tags for categorization (e.g., {"t=0", "qubit"}).
        prime (int): Prime level (0 = no prime, 1 = i.e. i', 2 = i'', etc.)
                     Used to avoid accidental contractions.
        id(str): Unique identifier (UUID). Two indices are equal iff they share the same id, tags and prime.
    """

    dim: int
    name: str = "Index"
    tags: FrozenSet[str] = field(default_factory=frozenset)
    prime: int = 0
    id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def __eq__(self, other: object) -> bool:
        """
        Two indices are equal if they share the same id, tags and prime.
        """
        if not isinstance(other, Index):
            return NotImplemented
        
        return self.id == other.id and self.tags == other.tags and self.prime == other.prime
    
    def __hash__(self) -> int:
        """
        Hash based on id, tags and prime (forzen, so hashable).
        """
        return hash((self.id, self.tags, self.prime))
    
    def __repr__(self) -> str:
        """
        Print index information.
        """

        tags_str = ",".join(sorted(self.tags)) if self.tags else ""

        prime_str = "'" * self.prime

        return f"Index(dim = {self.dim}, name = {self.name}, tags = {{{tags_str}}}, prime = {prime_str})"
    
    def prime_id(self, increment: int = 1) -> Index:
        """
        Return a new Index with prime level incremented.

        Args:
            increment: Amount to increase prime level (default 1). Can be negative to decrease.

        Returns:
            New Index with updated prime level.
        """

        return Index(dim = self.dim, 
                     name = self.name,
                     tags = self.tags,
                     prime = max(0, self.prime + increment),
                     id = self.id)
    
    def no_prime_id(self) -> Index:
        """
        Return a new Index with prime level set to 0.

        Returns: 
            New index with prime = 0
        """

        return Index(dim = self.dim,
                     name = self.name,
                     tags = self.tags,
                     prime = 0,
                     id = self.id)
    
    def add_tags(self, *new_tags: str) -> Index:
        """Return a new Index with additional tags.
        
        Args:
            *new_tags: Tags to add to this index.
        
        Returns:
            New Index with merged tags.
        """

        merged_tags = self.tags | frozenset(new_tags)

        return Index(dim = self.dim,
                     name = self.name,
                     tags = merged_tags,
                     prime = self.prime,
                     id = self.id)
    
    def remove_tags(self, *tags_to_remove: str) -> Index:
        """Return a new Index with specified tags removed.
        
        Args:
            *tags_to_remove: Tags to remove from this index.
        
        Returns:
            New Index with reduced tags.
        """

        remaining_tags = self.tags - frozenset(tags_to_remove)

        return Index(dim = self.dim,
                     name = self.name,
                     tags = remaining_tags,
                     prime = self.prime,
                     id = self.id)

    def sim(self) -> Index:
        """
        Return a similar index but with a different id (for creating a distinct space).

        Useful when you want an index with the same dim/name/tags but different identity, so one cannot contract it automatically.

        Returns:
            New index with same properties but fresh id.
        """

        return Index(dim = self.dim,
                     name = self.name,
                     tags = self.tags,
                     prime = self.prime,
                     id = str(uuid.uuid4()))