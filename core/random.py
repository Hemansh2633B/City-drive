"""
Deterministic Random Number Generator

Provides seed-based deterministic random number generation that can be
saved/restored for exact replay capability.
"""

import struct
from typing import Any, Optional, Union
import numpy as np


class DeterministicRNG:
    """
    A deterministic random number generator based on PCG64.
    
    Features:
    - Same seed always produces same sequence
    - State can be serialized/deserialized for replays
    - Supports multiple independent streams via sequence
    - Compatible with numpy random interface
    """
    
    def __init__(self, seed: Optional[Union[int, str, bytes]] = None, 
                 sequence: int = 0):
        """
        Initialize the RNG.
        
        Args:
            seed: Seed value (int, string, or bytes). If None, uses system entropy.
            sequence: Sequence number for independent streams (default 0)
        """
        if seed is None:
            # Use system entropy for non-deterministic seed
            seed_bytes = np.random.bytes(16)
            seed = int.from_bytes(seed_bytes, 'little')
        elif isinstance(seed, str):
            # Hash string to integer
            seed = self._hash_string(seed)
        elif isinstance(seed, bytes):
            seed = int.from_bytes(seed[:16], 'little')
        
        # Ensure seed is a regular Python int
        seed = int(seed)
        
        # Combine seed and sequence for stream independence
        # Use a simple but effective mixing
        combined_seed = (seed ^ ((sequence + 1) * 0x9e3779b97f4a7c15)) & 0xFFFFFFFFFFFFFFFF
        
        self._state = np.random.PCG64(seed=combined_seed)
        self._bit_generator = np.random.Generator(self._state)
        self._initial_seed = seed
        self._sequence = sequence
    
    @staticmethod
    def _hash_string(s: str) -> int:
        """Hash a string to a 64-bit integer using FNV-1a."""
        hash_val = 0xcbf29ce484222325  # FNV offset basis
        for byte in s.encode('utf-8'):
            hash_val ^= byte
            hash_val *= 0x100000001b3  # FNV prime
            hash_val &= 0xFFFFFFFFFFFFFFFF
        return hash_val
    
    @property
    def seed(self) -> int:
        """Return the initial seed."""
        return self._initial_seed
    
    @property
    def sequence(self) -> int:
        """Return the sequence number."""
        return self._sequence
    
    def get_state(self) -> dict:
        """Get serializable state for replay."""
        state = self._state.state
        inner_state = state['state']
        return {
            'state': str(inner_state['state']),
            'inc': str(inner_state['inc']),
            'seed': self._initial_seed,
            'sequence': self._sequence
        }
    
    def set_state(self, state_dict: dict) -> None:
        """Restore state from serialized dict."""
        self._initial_seed = state_dict['seed']
        self._sequence = state_dict['sequence']
        # Restore PCG64 state - parse strings back to int
        inner_state = {'state': int(state_dict['state']),
                       'inc': int(state_dict['inc'])}
        new_state = {'bit_generator': 'PCG64', 'state': inner_state, 'has_uint32': 0, 'uinteger': 0}
        self._state = np.random.PCG64()
        self._state.state = new_state
        self._bit_generator = np.random.Generator(self._state)
    
    def advance(self, steps: int) -> None:
        """Advance the RNG by n steps (for jumping ahead)."""
        for _ in range(steps):
            self._bit_generator.random()
    
    # Numpy-compatible interface
    def random(self) -> float:
        """Return random float in [0, 1)."""
        return self._bit_generator.random()
    
    def random_raw(self) -> int:
        """Return raw 64-bit integer."""
        return self._bit_generator.integers(0, 2**64, dtype=np.uint64)
    
    def integers(self, low: int, high: Optional[int] = None, 
                 size: Optional[Union[int, tuple]] = None, 
                 dtype: np.dtype = np.int64) -> np.ndarray:
        """Return random integers in [low, high)."""
        return self._bit_generator.integers(low, high, size, dtype)
    
    def choice(self, a: Union[int, np.ndarray], size: Optional[Union[int, tuple]] = None,
               replace: bool = True, p: Optional[np.ndarray] = None) -> np.ndarray:
        """Random choice from array or range."""
        return self._bit_generator.choice(a, size, replace, p)
    
    def normal(self, loc: float = 0.0, scale: float = 1.0,
               size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from normal distribution."""
        return self._bit_generator.normal(loc, scale, size)
    
    def uniform(self, low: float = 0.0, high: float = 1.0,
                size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from uniform distribution."""
        return self._bit_generator.uniform(low, high, size)
    
    def shuffle(self, arr: np.ndarray) -> None:
        """Shuffle array in-place."""
        self._bit_generator.shuffle(arr)
    
    def permutation(self, x: Union[int, np.ndarray]) -> np.ndarray:
        """Return permuted range or array."""
        return self._bit_generator.permutation(x)
    
    def beta(self, a: float, b: float, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from beta distribution."""
        return self._bit_generator.beta(a, b, size)
    
    def gamma(self, shape: float, scale: float = 1.0,
              size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from gamma distribution."""
        return self._bit_generator.gamma(shape, scale, size)
    
    def exponential(self, scale: float = 1.0,
                    size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from exponential distribution."""
        return self._bit_generator.exponential(scale, size)
    
    def dirichlet(self, alpha: np.ndarray, size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from Dirichlet distribution."""
        return self._bit_generator.dirichlet(alpha, size)
    
    def multinomial(self, n: int, pvals: np.ndarray,
                    size: Optional[Union[int, tuple]] = None) -> np.ndarray:
        """Return random samples from multinomial distribution."""
        return self._bit_generator.multinomial(n, pvals, size)


class SeededRandomStreams:
    """
    Manages multiple independent random streams from a master seed.
    
    Useful for separating randomness for different systems (track gen, physics, AI)
    while maintaining overall determinism.
    """
    
    def __init__(self, master_seed: Union[int, str, bytes]):
        self._master_seed = master_seed
        self._streams: dict[str, DeterministicRNG] = {}
        # Create master RNG for deriving stream seeds
        self._master_rng = DeterministicRNG(master_seed)
    
    def get_stream(self, name: str) -> DeterministicRNG:
        """Get or create a named random stream."""
        if name not in self._streams:
            # Derive stream seed from master
            stream_seed = self._master_rng.random_raw()
            # Use hash of name for sequence, keep it in reasonable range
            seq = abs(hash(name)) % 10000
            self._streams[name] = DeterministicRNG(stream_seed, sequence=seq)
        return self._streams[name]
    
    def get_state(self) -> dict:
        """Get state of all streams."""
        return {
            'master_seed': self._master_seed,
            'streams': {name: rng.get_state() for name, rng in self._streams.items()}
        }
    
    def set_state(self, state_dict: dict) -> None:
        """Restore all streams from state."""
        self._master_seed = state_dict['master_seed']
        self._master_rng = DeterministicRNG(self._master_seed)
        self._streams = {}
        for name, stream_state in state_dict['streams'].items():
            rng = DeterministicRNG(0)  # Dummy, will be replaced
            rng.set_state(stream_state)
            self._streams[name] = rng


def create_rng(seed: Optional[Union[int, str, bytes]] = None) -> DeterministicRNG:
    """Factory function to create a deterministic RNG."""
    return DeterministicRNG(seed)


def seed_from_string(s: str) -> int:
    """Convert a string to a deterministic integer seed."""
    return DeterministicRNG._hash_string(s)