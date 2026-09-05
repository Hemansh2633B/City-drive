"""
Track Package

Procedural track generation and track representation.
"""

from track.track import Track, TrackSegment, Checkpoint, RacingLinePoint
from track.generator import TrackGenerator, TrackGeneratorConfig

__all__ = [
    'Track', 'TrackSegment', 'Checkpoint', 'RacingLinePoint',
    'TrackGenerator', 'TrackGeneratorConfig'
]