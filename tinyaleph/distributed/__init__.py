"""
Distributed Computing Primitives for TinyAleph.

Provides transport layers and backends for distributed prime-state
computation across multiple nodes.

Components:
- Transport: Abstract transport interface with implementations
- Ray Backend: Ray-based distributed executor (optional)
"""

from tinyaleph.distributed.transport import (
    Transport,
    LocalTransport,
    Message,
    MessageType,
)

__all__ = [
    "Transport",
    "LocalTransport", 
    "Message",
    "MessageType",
]