"""
Transport Layer for Distributed Prime Networks

Provides abstract transport interface and implementations for
message passing between distributed nodes.

Supports:
- Local (in-process) transport for testing
- Async message queues
- Serialization of prime states
- Connection management
"""

from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import (
    Dict, List, Optional, Any, Callable, TypeVar, Generic,
    Awaitable, Set, Tuple
)
from enum import Enum, auto
from collections import deque
import json
import time
import asyncio
from uuid import uuid4
import logging

from tinyaleph.core.quaternion import Quaternion
from tinyaleph.core.complex import Complex

logger = logging.getLogger(__name__)


class MessageType(Enum):
    """Types of messages in the distributed network."""
    
    # State operations
    STATE_SYNC = auto()       # Synchronize prime state
    STATE_UPDATE = auto()     # Incremental state update
    STATE_REQUEST = auto()    # Request state from peer
    
    # Entanglement operations
    ENTANGLE_REQUEST = auto()    # Request entanglement with peer
    ENTANGLE_ACCEPT = auto()     # Accept entanglement request
    ENTANGLE_REJECT = auto()     # Reject entanglement request
    BELL_MEASUREMENT = auto()    # Bell measurement result
    
    # Teleportation
    TELEPORT_START = auto()      # Initiate teleportation
    TELEPORT_CLASSICAL = auto()  # Classical bits for teleportation
    TELEPORT_COMPLETE = auto()   # Teleportation completed
    
    # Coordination
    HEARTBEAT = auto()        # Keep-alive signal
    JOIN = auto()             # Node joining network
    LEAVE = auto()            # Node leaving network
    DISCOVER = auto()         # Peer discovery
    
    # Data
    QUERY = auto()            # Query operation
    RESPONSE = auto()         # Response to query
    ERROR = auto()            # Error message
    
    # Consensus
    PROPOSE = auto()          # Propose value for consensus
    VOTE = auto()             # Vote on proposal
    COMMIT = auto()           # Commit agreed value


@dataclass
class Message:
    """
    Network message between nodes.
    
    Attributes:
        id: Unique message identifier
        type: Message type
        source: Source node ID
        destination: Destination node ID (None for broadcast)
        payload: Message data
        timestamp: Message creation time
        correlation_id: ID of related message (for request/response)
        ttl: Time-to-live (hops remaining)
    """
    
    id: str = field(default_factory=lambda: str(uuid4()))
    type: MessageType = MessageType.HEARTBEAT
    source: str = ""
    destination: Optional[str] = None
    payload: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)
    correlation_id: Optional[str] = None
    ttl: int = 16
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary for serialization."""
        return {
            "id": self.id,
            "type": self.type.name,
            "source": self.source,
            "destination": self.destination,
            "payload": self.payload,
            "timestamp": self.timestamp,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Message:
        """Create message from dictionary."""
        return cls(
            id=data.get("id", str(uuid4())),
            type=MessageType[data["type"]],
            source=data.get("source", ""),
            destination=data.get("destination"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", time.time()),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl", 16)
        )
    
    def reply(self, 
              type: MessageType, 
              payload: Dict[str, Any],
              source: str) -> Message:
        """Create reply message."""
        return Message(
            type=type,
            source=source,
            destination=self.source,
            payload=payload,
            correlation_id=self.id
        )
    
    def is_broadcast(self) -> bool:
        """Check if this is a broadcast message."""
        return self.destination is None
    
    def decrement_ttl(self) -> bool:
        """Decrement TTL. Returns False if message should be dropped."""
        self.ttl -= 1
        return self.ttl > 0


T = TypeVar('T')


class MessageHandler(ABC, Generic[T]):
    """Abstract message handler."""
    
    @abstractmethod
    async def handle(self, message: Message) -> Optional[T]:
        """Handle a message and optionally return response."""
        pass


@dataclass
class Transport(ABC):
    """
    Abstract transport interface for distributed communication.
    
    Implementations provide actual network connectivity.
    """
    
    node_id: str = field(default_factory=lambda: str(uuid4()))
    handlers: Dict[MessageType, List[Callable[[Message], Awaitable[Optional[Message]]]]] = field(
        default_factory=dict
    )
    running: bool = False
    
    @abstractmethod
    async def send(self, message: Message) -> bool:
        """
        Send message to destination.
        
        Returns True if sent successfully.
        """
        pass
    
    @abstractmethod
    async def broadcast(self, message: Message) -> int:
        """
        Broadcast message to all connected peers.
        
        Returns number of peers message was sent to.
        """
        pass
    
    @abstractmethod
    async def receive(self) -> Optional[Message]:
        """
        Receive next message.
        
        Returns None if no message available.
        """
        pass
    
    @abstractmethod
    async def connect(self, address: str) -> bool:
        """
        Connect to peer at address.
        
        Returns True if connection established.
        """
        pass
    
    @abstractmethod
    async def disconnect(self, peer_id: str) -> bool:
        """
        Disconnect from peer.
        
        Returns True if disconnected successfully.
        """
        pass
    
    @abstractmethod
    def get_peers(self) -> List[str]:
        """Get list of connected peer IDs."""
        pass
    
    def register_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[Message], Awaitable[Optional[Message]]]
    ) -> None:
        """Register handler for message type."""
        if msg_type not in self.handlers:
            self.handlers[msg_type] = []
        self.handlers[msg_type].append(handler)
    
    def unregister_handler(
        self,
        msg_type: MessageType,
        handler: Callable[[Message], Awaitable[Optional[Message]]]
    ) -> bool:
        """Unregister handler. Returns True if found and removed."""
        if msg_type in self.handlers:
            try:
                self.handlers[msg_type].remove(handler)
                return True
            except ValueError:
                pass
        return False
    
    async def dispatch(self, message: Message) -> List[Optional[Message]]:
        """Dispatch message to registered handlers."""
        handlers = self.handlers.get(message.type, [])
        results = []
        
        for handler in handlers:
            try:
                result = await handler(message)
                results.append(result)
            except Exception as e:
                logger.error(f"Handler error for {message.type}: {e}")
                results.append(None)
        
        return results
    
    async def start(self) -> None:
        """Start transport (begin accepting messages)."""
        self.running = True
    
    async def stop(self) -> None:
        """Stop transport."""
        self.running = False
    
    async def request(
        self,
        peer_id: str,
        type: MessageType,
        payload: Dict[str, Any],
        timeout: float = 5.0
    ) -> Optional[Message]:
        """
        Send request and wait for response.
        
        Returns response message or None on timeout.
        """
        request = Message(
            type=type,
            source=self.node_id,
            destination=peer_id,
            payload=payload
        )
        
        # Set up response handler
        response_received = asyncio.Event()
        response_message: Optional[Message] = None
        
        async def response_handler(msg: Message) -> Optional[Message]:
            nonlocal response_message
            if msg.correlation_id == request.id:
                response_message = msg
                response_received.set()
            return None
        
        # Register for response types
        self.register_handler(MessageType.RESPONSE, response_handler)
        
        try:
            await self.send(request)
            
            try:
                await asyncio.wait_for(response_received.wait(), timeout)
                return response_message
            except asyncio.TimeoutError:
                return None
        finally:
            self.unregister_handler(MessageType.RESPONSE, response_handler)


@dataclass
class LocalTransport(Transport):
    """
    Local (in-process) transport for testing and single-node use.
    
    All nodes share the same message queues in memory.
    """
    
    # Class-level shared registry of transports
    _registry: Dict[str, 'LocalTransport'] = field(default_factory=dict, repr=False)
    _inbox: deque = field(default_factory=deque)
    _connected_peers: Set[str] = field(default_factory=set)
    
    def __post_init__(self):
        # Register this transport
        LocalTransport._registry[self.node_id] = self
    
    async def send(self, message: Message) -> bool:
        """Send message to destination's inbox."""
        if not message.destination:
            return False
        
        message.source = self.node_id
        
        target = LocalTransport._registry.get(message.destination)
        if target:
            target._inbox.append(message)
            return True
        return False
    
    async def broadcast(self, message: Message) -> int:
        """Broadcast to all connected peers."""
        message.source = self.node_id
        message.destination = None
        
        count = 0
        for peer_id in self._connected_peers:
            target = LocalTransport._registry.get(peer_id)
            if target:
                target._inbox.append(message)
                count += 1
        
        return count
    
    async def receive(self) -> Optional[Message]:
        """Get next message from inbox."""
        if self._inbox:
            return self._inbox.popleft()
        return None
    
    async def connect(self, address: str) -> bool:
        """
        Connect to peer by node ID.
        
        For LocalTransport, address is the node ID.
        """
        if address in LocalTransport._registry:
            self._connected_peers.add(address)
            # Bidirectional connection
            LocalTransport._registry[address]._connected_peers.add(self.node_id)
            return True
        return False
    
    async def disconnect(self, peer_id: str) -> bool:
        """Disconnect from peer."""
        if peer_id in self._connected_peers:
            self._connected_peers.discard(peer_id)
            # Remove bidirectional connection
            if peer_id in LocalTransport._registry:
                LocalTransport._registry[peer_id]._connected_peers.discard(self.node_id)
            return True
        return False
    
    def get_peers(self) -> List[str]:
        """Get list of connected peers."""
        return list(self._connected_peers)
    
    async def run_message_loop(self, interval: float = 0.01) -> None:
        """Process incoming messages in a loop."""
        while self.running:
            message = await self.receive()
            if message:
                responses = await self.dispatch(message)
                
                # Send any response messages
                for response in responses:
                    if response:
                        await self.send(response)
            
            await asyncio.sleep(interval)
    
    @classmethod
    def clear_registry(cls) -> None:
        """Clear all registered transports (for testing)."""
        cls._registry.clear()


@dataclass
class StateSerializer:
    """
    Serializes and deserializes prime state data for transport.
    """
    
    @staticmethod
    def serialize_complex(c: Complex) -> Dict[str, float]:
        """Serialize complex number."""
        return {"real": c.real, "imag": c.imag}
    
    @staticmethod
    def deserialize_complex(data: Dict[str, float]) -> Complex:
        """Deserialize complex number."""
        return Complex(data["real"], data["imag"])
    
    @staticmethod
    def serialize_quaternion(q: Quaternion) -> Dict[str, float]:
        """Serialize quaternion."""
        return {"w": q.w, "i": q.i, "j": q.j, "k": q.k}
    
    @staticmethod
    def deserialize_quaternion(data: Dict[str, float]) -> Quaternion:
        """Deserialize quaternion."""
        return Quaternion(data["w"], data["i"], data["j"], data["k"])
    
    @staticmethod
    def serialize_state(amplitudes: Dict[int, Complex]) -> Dict[str, Any]:
        """Serialize prime state amplitudes."""
        return {
            "amplitudes": {
                str(p): StateSerializer.serialize_complex(a)
                for p, a in amplitudes.items()
            }
        }
    
    @staticmethod
    def deserialize_state(data: Dict[str, Any]) -> Dict[int, Complex]:
        """Deserialize prime state amplitudes."""
        return {
            int(p): StateSerializer.deserialize_complex(a)
            for p, a in data.get("amplitudes", {}).items()
        }
    
    @staticmethod
    def serialize_qstate(amplitudes: Dict[int, Quaternion]) -> Dict[str, Any]:
        """Serialize quaternionic prime state."""
        return {
            "q_amplitudes": {
                str(p): StateSerializer.serialize_quaternion(q)
                for p, q in amplitudes.items()
            }
        }
    
    @staticmethod
    def deserialize_qstate(data: Dict[str, Any]) -> Dict[int, Quaternion]:
        """Deserialize quaternionic prime state."""
        return {
            int(p): StateSerializer.deserialize_quaternion(q)
            for p, q in data.get("q_amplitudes", {}).items()
        }


@dataclass
class ConnectionPool:
    """
    Manages a pool of transport connections.
    
    Handles connection lifecycle, reconnection, and load balancing.
    """
    
    transport: Transport
    max_connections: int = 100
    reconnect_interval: float = 5.0
    _failed_connections: Dict[str, float] = field(default_factory=dict)
    
    async def get_connection(self, peer_id: str) -> bool:
        """
        Ensure connection to peer exists.
        
        Returns True if connected (existing or new).
        """
        if peer_id in self.transport.get_peers():
            return True
        
        # Check if recently failed
        last_fail = self._failed_connections.get(peer_id, 0)
        if time.time() - last_fail < self.reconnect_interval:
            return False
        
        # Check connection limit
        if len(self.transport.get_peers()) >= self.max_connections:
            # Could implement eviction policy here
            return False
        
        success = await self.transport.connect(peer_id)
        if not success:
            self._failed_connections[peer_id] = time.time()
        
        return success
    
    async def release_connection(self, peer_id: str) -> None:
        """Release connection (mark as available for reuse)."""
        # In this simple implementation, connections persist
        pass
    
    async def close_connection(self, peer_id: str) -> bool:
        """Close and remove connection."""
        return await self.transport.disconnect(peer_id)
    
    def get_available_peers(self) -> List[str]:
        """Get list of currently connected peers."""
        return self.transport.get_peers()
    
    async def broadcast_to_pool(self, message: Message) -> int:
        """Broadcast message to all pooled connections."""
        return await self.transport.broadcast(message)


@dataclass
class MessageRouter:
    """
    Routes messages through the network using routing tables.
    
    Supports multi-hop routing when direct connection not available.
    """
    
    transport: Transport
    routing_table: Dict[str, str] = field(default_factory=dict)  # dest -> next_hop
    
    async def route(self, message: Message) -> bool:
        """
        Route message to destination.
        
        Uses direct connection if available, otherwise routes
        through next hop.
        """
        if not message.destination:
            # Broadcast
            await self.transport.broadcast(message)
            return True
        
        # Direct connection?
        if message.destination in self.transport.get_peers():
            return await self.transport.send(message)
        
        # Use routing table
        next_hop = self.routing_table.get(message.destination)
        if next_hop and next_hop in self.transport.get_peers():
            if message.decrement_ttl():
                message_copy = Message(
                    id=message.id,
                    type=message.type,
                    source=message.source,
                    destination=message.destination,
                    payload=message.payload,
                    timestamp=message.timestamp,
                    correlation_id=message.correlation_id,
                    ttl=message.ttl
                )
                return await self.transport.send(message_copy)
        
        return False
    
    def add_route(self, destination: str, next_hop: str) -> None:
        """Add routing entry."""
        self.routing_table[destination] = next_hop
    
    def remove_route(self, destination: str) -> None:
        """Remove routing entry."""
        self.routing_table.pop(destination, None)
    
    def update_routes_from_neighbors(self, neighbor_tables: Dict[str, Dict[str, str]]) -> None:
        """
        Update routing table based on neighbor information.
        
        Simple distance-vector routing.
        """
        for neighbor, their_table in neighbor_tables.items():
            for dest, their_next in their_table.items():
                if dest != self.transport.node_id:
                    # Route through this neighbor if we don't have a route
                    if dest not in self.routing_table:
                        self.routing_table[dest] = neighbor