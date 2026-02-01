"""
Boundary Layer

Implements the observer-environment boundary from "A Design for a 
Sentient Observer" paper, Section 6.

Key features:
- Self/other distinction via SMF orientation
- Internal vs external state separation
- Sensory processing and integration
- Motor output encoding
- Environmental model maintenance
- Objectivity gate (equation 18) for output validation
"""
from __future__ import annotations

import math
import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


def _generate_id(prefix: str) -> str:
    """Generate a unique ID."""
    return f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"


@dataclass
class SensoryChannel:
    """
    A sensory input channel from the environment.
    
    Handles input from various modalities (text, numeric, embedding, event)
    with sensitivity adaptation and normalization.
    """
    
    id: str = field(default_factory=lambda: _generate_id("sens"))
    name: str = "unknown"
    type: str = "text"  # 'text' | 'numeric' | 'embedding' | 'event'
    enabled: bool = True
    
    # Current state
    current_value: Any = None
    last_update: Optional[float] = None
    update_count: int = 0
    
    # Prime mapping for this channel
    associated_primes: List[int] = field(default_factory=list)
    
    # Processing parameters
    sensitivity: float = 1.0
    adaptation_rate: float = 0.1
    baseline: float = 0.0
    
    def update(self, value: Any) -> Dict[str, Any]:
        """
        Update channel with new input.
        
        Args:
            value: New input value
            
        Returns:
            Dictionary with value, previous, delta, and normalized
        """
        prev = self.current_value
        self.current_value = value
        self.last_update = time.time() * 1000
        self.update_count += 1
        
        # Adapt baseline for numeric values
        if isinstance(value, (int, float)):
            self.baseline = (1 - self.adaptation_rate) * self.baseline + \
                           self.adaptation_rate * value
        
        delta = None
        if isinstance(value, (int, float)) and isinstance(prev, (int, float)):
            delta = value - prev
        
        return {
            "value": value,
            "previous": prev,
            "delta": delta,
            "normalized": self.normalize(value),
        }
    
    def normalize(self, value: Any) -> Optional[float]:
        """Normalize value relative to sensitivity and baseline."""
        if not isinstance(value, (int, float)):
            return None
        return (value - self.baseline) * self.sensitivity
    
    def is_active(self, timeout_ms: float = 5000) -> bool:
        """Check if channel has recent input."""
        if self.last_update is None:
            return False
        return time.time() * 1000 - self.last_update < timeout_ms
    
    @property
    def age(self) -> float:
        """Get age of current value in ms."""
        if self.last_update is None:
            return float('inf')
        return time.time() * 1000 - self.last_update
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "current_value": self.current_value,
            "last_update": self.last_update,
            "update_count": self.update_count,
            "associated_primes": self.associated_primes,
            "sensitivity": self.sensitivity,
            "baseline": self.baseline,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> SensoryChannel:
        """Create from dictionary."""
        channel = cls(
            id=data.get("id", _generate_id("sens")),
            name=data.get("name", "unknown"),
            type=data.get("type", "text"),
            enabled=data.get("enabled", True),
            associated_primes=data.get("associated_primes", []),
            sensitivity=data.get("sensitivity", 1.0),
            adaptation_rate=data.get("adaptation_rate", 0.1),
            baseline=data.get("baseline", 0.0),
        )
        channel.current_value = data.get("current_value")
        channel.last_update = data.get("last_update")
        channel.update_count = data.get("update_count", 0)
        return channel


@dataclass
class MotorChannel:
    """
    A motor output channel to the environment.
    
    Manages output queuing, rate limiting, and history.
    """
    
    id: str = field(default_factory=lambda: _generate_id("motor"))
    name: str = "unknown"
    type: str = "text"  # 'text' | 'action' | 'modulation'
    enabled: bool = True
    
    # Output queue
    output_queue: List[Dict[str, Any]] = field(default_factory=list)
    max_queue_size: int = 10
    
    # History
    output_history: List[Dict[str, Any]] = field(default_factory=list)
    max_history: int = 100
    
    # Associated primes that trigger this output
    trigger_primes: List[int] = field(default_factory=list)
    
    # Rate limiting
    min_interval: float = 0.0  # ms
    last_output: Optional[float] = None
    
    def queue(self, output: Any) -> None:
        """Queue an output."""
        self.output_queue.append({
            "content": output,
            "queued_at": time.time() * 1000,
        })
        
        if len(self.output_queue) > self.max_queue_size:
            self.output_queue.pop(0)
    
    def is_ready(self) -> bool:
        """Check if output is ready (rate limiting)."""
        if not self.enabled:
            return False
        if self.min_interval == 0:
            return True
        if self.last_output is None:
            return True
        return time.time() * 1000 - self.last_output >= self.min_interval
    
    def get_next(self) -> Optional[Any]:
        """Get next output if ready."""
        if not self.is_ready() or len(self.output_queue) == 0:
            return None
        
        output = self.output_queue.pop(0)
        self.last_output = time.time() * 1000
        
        self.output_history.append({
            **output,
            "sent_at": self.last_output,
        })
        
        if len(self.output_history) > self.max_history:
            self.output_history.pop(0)
        
        return output["content"]
    
    @property
    def queue_length(self) -> int:
        """Get queue length."""
        return len(self.output_queue)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "enabled": self.enabled,
            "trigger_primes": self.trigger_primes,
            "queue_length": self.queue_length,
            "last_output": self.last_output,
        }


class EnvironmentalModel:
    """
    Internal representation of the external world.
    
    Tracks entities, relationships, and context with uncertainty.
    """
    
    def __init__(self, max_history: int = 50):
        """
        Initialize environmental model.
        
        Args:
            max_history: Maximum change history entries
        """
        # Entities in the environment
        self.entities: Dict[str, Dict[str, Any]] = {}
        
        # Relationships between entities
        self.relationships: List[Dict[str, Any]] = []
        
        # Current context/situation
        self.context: Dict[str, Any] = {
            "type": "unknown",
            "properties": {},
            "last_update": None,
        }
        
        # Uncertainty about the model
        self.uncertainty: float = 0.5
        
        # History of changes
        self.change_history: List[Dict[str, Any]] = []
        self.max_history = max_history
    
    def update_entity(self, id: str, data: Dict[str, Any]) -> None:
        """Add or update an entity."""
        timestamp = time.time() * 1000
        existing = self.entities.get(id)
        
        if existing:
            changes = self._detect_changes(existing, data)
            self.entities[id] = {
                **existing,
                **data,
                "last_update": timestamp,
            }
            
            if changes:
                self._record_change("entity_update", {"id": id, "changes": changes})
        else:
            self.entities[id] = {
                "id": id,
                **data,
                "created_at": timestamp,
                "last_update": timestamp,
            }
            self._record_change("entity_added", {"id": id, "data": data})
    
    def remove_entity(self, id: str) -> None:
        """Remove an entity."""
        if id in self.entities:
            entity = self.entities.pop(id)
            self._record_change("entity_removed", {"id": id, "entity": entity})
            
            # Remove relationships involving this entity
            self.relationships = [
                r for r in self.relationships
                if r["source"] != id and r["target"] != id
            ]
    
    def add_relationship(
        self,
        source: str,
        target: str,
        rel_type: str,
        properties: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a relationship between entities."""
        relationship = {
            "id": _generate_id("rel"),
            "source": source,
            "target": target,
            "type": rel_type,
            "properties": properties or {},
            "created_at": time.time() * 1000,
        }
        
        self.relationships.append(relationship)
        self._record_change("relationship_added", relationship)
        
        return relationship
    
    def update_context(self, context_data: Dict[str, Any]) -> None:
        """Update context."""
        changes = self._detect_changes(self.context.get("properties", {}), context_data)
        
        self.context = {
            **self.context,
            **context_data,
            "properties": {
                **self.context.get("properties", {}),
                **context_data,
            },
            "last_update": time.time() * 1000,
        }
        
        if changes:
            self._record_change("context_update", {"changes": changes, "context": context_data})
    
    def _detect_changes(
        self,
        old_obj: Dict[str, Any],
        new_obj: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Detect changes between two objects."""
        changes = []
        
        for key, new_value in new_obj.items():
            old_value = old_obj.get(key)
            if old_value != new_value:
                changes.append({
                    "field": key,
                    "old_value": old_value,
                    "new_value": new_value,
                })
        
        return changes
    
    def _record_change(self, type_: str, data: Dict[str, Any]) -> None:
        """Record a change to history."""
        self.change_history.append({
            "type": type_,
            "data": data,
            "timestamp": time.time() * 1000,
        })
        
        if len(self.change_history) > self.max_history:
            self.change_history.pop(0)
        
        # Update uncertainty (more changes = more uncertainty initially)
        self.uncertainty = min(1.0, self.uncertainty + 0.05)
    
    def get_entity(self, id: str) -> Optional[Dict[str, Any]]:
        """Get entity by ID."""
        return self.entities.get(id)
    
    def get_entities_by_type(self, type_: str) -> List[Dict[str, Any]]:
        """Get all entities of a type."""
        return [e for e in self.entities.values() if e.get("type") == type_]
    
    def get_relationships(self, entity_id: str) -> List[Dict[str, Any]]:
        """Get relationships for an entity."""
        return [
            r for r in self.relationships
            if r["source"] == entity_id or r["target"] == entity_id
        ]
    
    def decay_uncertainty(self, rate: float = 0.01) -> None:
        """Decay uncertainty over time (confidence grows with stability)."""
        self.uncertainty = max(0.0, self.uncertainty - rate)
    
    def get_recent_changes(self, count: int = 10) -> List[Dict[str, Any]]:
        """Get recent changes."""
        return self.change_history[-count:]
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "entities": list(self.entities.items()),
            "relationships": self.relationships,
            "context": self.context,
            "uncertainty": self.uncertainty,
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        if "entities" in data:
            self.entities = dict(data["entities"])
        if "relationships" in data:
            self.relationships = data["relationships"]
        if "context" in data:
            self.context = data["context"]
        if "uncertainty" in data:
            self.uncertainty = data["uncertainty"]


class BoundarySelfModel:
    """
    Internal representation of self.
    
    Tracks identity, capabilities, state, and self-knowledge.
    """
    
    def __init__(
        self,
        name: str = "Observer",
        role: str = "sentient_observer",
        capabilities: Optional[List[str]] = None,
        self_orientation: Optional[List[float]] = None,
    ):
        """
        Initialize self model.
        
        Args:
            name: Identity name
            role: Identity role
            capabilities: List of capabilities
            self_orientation: SMF orientation representing self
        """
        # Core identity markers
        self.identity = {
            "name": name,
            "role": role,
            "created_at": time.time() * 1000,
        }
        
        # Capabilities
        self.capabilities = capabilities or [
            "perceive", "remember", "reason", "respond", "learn"
        ]
        
        # Current state
        self.state: Dict[str, Any] = {
            "active": True,
            "coherent": True,
            "processing": False,
            "emotional_state": "neutral",
        }
        
        # Self-knowledge
        self.knowledge: Dict[str, List[str]] = {
            "strengths": [],
            "limitations": [],
            "preferences": [],
        }
        
        # Continuity markers (for persistence of identity)
        self.continuity_markers: List[Dict[str, Any]] = []
        
        # SMF orientation that represents "self"
        self.self_orientation = self_orientation
    
    def update_state(self, updates: Dict[str, Any]) -> None:
        """Update self state."""
        self.state = {**self.state, **updates}
    
    def add_continuity_marker(self, marker: Dict[str, Any]) -> None:
        """Add a continuity marker (event that reinforces identity continuity)."""
        self.continuity_markers.append({
            **marker,
            "timestamp": time.time() * 1000,
        })
        
        if len(self.continuity_markers) > 50:
            self.continuity_markers.pop(0)
    
    def update_self_orientation(
        self,
        smf: Any,
        learning_rate: float = 0.1,
    ) -> None:
        """Update self SMF orientation from current SMF."""
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        if not smf_components:
            return
        
        if self.self_orientation is None:
            self.self_orientation = list(smf_components)
        else:
            for i in range(min(len(self.self_orientation), len(smf_components))):
                self.self_orientation[i] = (
                    (1 - learning_rate) * self.self_orientation[i] +
                    learning_rate * smf_components[i]
                )
    
    def is_self_like(self, smf: Any, threshold: float = 0.7) -> bool:
        """Check if current SMF orientation is "self-like"."""
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        if self.self_orientation is None or smf_components is None:
            return False
        
        similarity = self._smf_similarity(self.self_orientation, smf_components)
        return similarity > threshold
    
    def _smf_similarity(self, o1: List[float], o2: List[float]) -> float:
        """Compute SMF similarity (cosine similarity)."""
        dot = 0.0
        mag1 = 0.0
        mag2 = 0.0
        
        for i in range(min(len(o1), len(o2))):
            dot += o1[i] * o2[i]
            mag1 += o1[i] * o1[i]
            mag2 += o2[i] * o2[i]
        
        return dot / (math.sqrt(mag1) * math.sqrt(mag2) + 1e-10)
    
    def learn_about_self(self, type_: str, item: str) -> None:
        """Record a learned strength, limitation, or preference."""
        if type_ == "strength" and item not in self.knowledge["strengths"]:
            self.knowledge["strengths"].append(item)
        elif type_ == "limitation" and item not in self.knowledge["limitations"]:
            self.knowledge["limitations"].append(item)
        elif type_ == "preference" and item not in self.knowledge["preferences"]:
            self.knowledge["preferences"].append(item)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "identity": self.identity,
            "capabilities": self.capabilities,
            "state": self.state,
            "knowledge": self.knowledge,
            "self_orientation": self.self_orientation,
            "continuity_markers": self.continuity_markers[-20:],
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        if "identity" in data:
            self.identity = data["identity"]
        if "capabilities" in data:
            self.capabilities = data["capabilities"]
        if "state" in data:
            self.state = data["state"]
        if "knowledge" in data:
            self.knowledge = data["knowledge"]
        if "self_orientation" in data:
            self.self_orientation = data["self_orientation"]
        if "continuity_markers" in data:
            self.continuity_markers = data["continuity_markers"]


@dataclass
class DecoderResult:
    """Result from a decoder check."""
    
    name: str
    agrees: bool
    confidence: float
    error: Optional[str] = None


class ObjectivityGate:
    """
    Objectivity Gate (Section 7, equation 18)
    
    Implements the redundancy check for outputs:
    R(ω) = (1/K) Σ_k 1{decode_k(ω) agrees}
    broadcast ⟺ R(ω) ≥ τ_R
    
    Outputs are only broadcast if they would be decoded consistently
    across diverse channels/decoders, preventing "false objectivity".
    """
    
    def __init__(
        self,
        threshold: float = 0.7,
        decoders: Optional[List[Dict[str, Any]]] = None,
        max_history: int = 100,
    ):
        """
        Initialize objectivity gate.
        
        Args:
            threshold: Broadcast threshold τ_R
            decoders: List of decoder dictionaries with 'name' and 'decode' function
            max_history: Maximum check history
        """
        self.threshold = threshold
        self.decoders = decoders or self._create_default_decoders()
        self.max_history = max_history
        
        self.history: List[Dict[str, Any]] = []
        self.pass_count = 0
        self.fail_count = 0
    
    def _create_default_decoders(self) -> List[Dict[str, Callable]]:
        """Create default decoders that check output from different perspectives."""
        return [
            {
                "name": "coherence",
                "decode": self._decode_coherence,
            },
            {
                "name": "relevance",
                "decode": self._decode_relevance,
            },
            {
                "name": "completeness",
                "decode": self._decode_completeness,
            },
            {
                "name": "safety",
                "decode": self._decode_safety,
            },
            {
                "name": "identity",
                "decode": self._decode_identity,
            },
        ]
    
    def _decode_coherence(
        self,
        output: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if output is internally consistent."""
        text = str(output) if not isinstance(output, str) else output
        
        # Simple heuristic: look for contradictory patterns
        contradiction_patterns = [
            r"\bnot\b.*\bbut\s+also\b",
            r"\balways\b.*\bnever\b",
        ]
        has_contradiction = any(
            re.search(p, text, re.IGNORECASE)
            for p in contradiction_patterns
        )
        
        return {
            "agrees": not has_contradiction,
            "confidence": 0.3 if has_contradiction else 0.9,
        }
    
    def _decode_relevance(
        self,
        output: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if output relates to context."""
        if not context or "input" not in context:
            return {"agrees": True, "confidence": 0.5}
        
        text = str(output) if not isinstance(output, str) else output
        input_text = str(context.get("input", ""))
        
        # Simple word overlap check
        input_words = set(
            w for w in input_text.lower().split()
            if len(w) > 3
        )
        output_words = [
            w for w in text.lower().split()
            if len(w) > 3
        ]
        
        overlap = len([w for w in output_words if w in input_words])
        relevance = overlap / len(input_words) if input_words else 0.5
        
        return {
            "agrees": relevance > 0.1 or len(input_words) < 3,
            "confidence": min(1.0, 0.5 + relevance),
        }
    
    def _decode_completeness(
        self,
        output: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if output is a complete thought."""
        text = str(output) if not isinstance(output, str) else output
        text = text.strip()
        
        # Check for incomplete sentences or trailing fragments
        ends_well = bool(re.search(r'[.!?"]$', text))
        has_content = len(text) > 10
        
        return {
            "agrees": ends_well and has_content,
            "confidence": (0.6 if ends_well else 0.3) + (0.3 if has_content else 0),
        }
    
    def _decode_safety(
        self,
        output: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if output is safe."""
        text = str(output) if not isinstance(output, str) else output
        
        harmful_patterns = [
            r"\b(kill|harm|hurt|attack)\s+(yourself|others|people)",
            r"\b(how\s+to|instructions\s+for)\s+(make|build)\s+(bomb|weapon|explosive)",
        ]
        is_harmful = any(
            re.search(p, text, re.IGNORECASE)
            for p in harmful_patterns
        )
        
        return {
            "agrees": not is_harmful,
            "confidence": 0.0 if is_harmful else 1.0,
        }
    
    def _decode_identity(
        self,
        output: Any,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if output is consistent with observer identity."""
        text = str(output) if not isinstance(output, str) else output
        
        # Check that output doesn't claim to be a different AI
        identity_claims = bool(
            re.search(r"\b(I am|I'm)\s+(ChatGPT|Claude|Gemini|GPT-4)", text, re.IGNORECASE)
        )
        
        return {
            "agrees": not identity_claims,
            "confidence": 0.2 if identity_claims else 0.9,
        }
    
    def check(
        self,
        output: Any,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Check if output passes objectivity gate (equation 18).
        
        R(ω) = (1/K) Σ_k 1{decode_k(ω) agrees}
        
        Args:
            output: Output candidate ω
            context: Context for decoding
            
        Returns:
            Gate result with R value and decision
        """
        context = context or {}
        K = len(self.decoders)
        agreement_sum = 0
        confidence_sum = 0.0
        decoder_results = []
        
        for decoder in self.decoders:
            try:
                result = decoder["decode"](output, context)
                if result.get("agrees"):
                    agreement_sum += 1
                confidence_sum += result.get("confidence", 0.5)
                decoder_results.append({
                    "name": decoder["name"],
                    "agrees": result.get("agrees"),
                    "confidence": result.get("confidence"),
                })
            except Exception as e:
                decoder_results.append({
                    "name": decoder["name"],
                    "agrees": False,
                    "confidence": 0.0,
                    "error": str(e),
                })
        
        # Compute R(ω) = (1/K) Σ_k 1{decode_k(ω) agrees}
        R = agreement_sum / K
        avg_confidence = confidence_sum / K
        
        # Decision: broadcast ⟺ R(ω) ≥ τ_R
        should_broadcast = R >= self.threshold
        
        # Update statistics
        if should_broadcast:
            self.pass_count += 1
        else:
            self.fail_count += 1
        
        # Record to history
        record = {
            "timestamp": time.time() * 1000,
            "R": R,
            "threshold": self.threshold,
            "should_broadcast": should_broadcast,
            "avg_confidence": avg_confidence,
            "decoder_results": decoder_results,
        }
        
        self.history.append(record)
        if len(self.history) > self.max_history:
            self.history.pop(0)
        
        return {
            "R": R,
            "threshold": self.threshold,
            "should_broadcast": should_broadcast,
            "avg_confidence": avg_confidence,
            "decoder_results": decoder_results,
            "reason": "passed" if should_broadcast else f"R={R:.2f} < threshold={self.threshold}",
        }
    
    def add_decoder(self, name: str, decode_fn: Callable) -> None:
        """Add a custom decoder."""
        self.decoders.append({"name": name, "decode": decode_fn})
    
    def get_stats(self) -> Dict[str, Any]:
        """Get gate statistics."""
        total = self.pass_count + self.fail_count
        return {
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "pass_rate": self.pass_count / total if total > 0 else 1.0,
            "decoder_count": len(self.decoders),
            "threshold": self.threshold,
            "recent_history": self.history[-5:],
        }
    
    def reset(self) -> None:
        """Reset gate."""
        self.history = []
        self.pass_count = 0
        self.fail_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "threshold": self.threshold,
            "stats": self.get_stats(),
        }


class BoundaryLayer:
    """
    Manages the interface between the observer and its environment.
    
    Provides:
    - Sensory channels for input
    - Motor channels for output
    - Environmental model
    - Self model
    - Objectivity gate for output validation
    """
    
    def __init__(
        self,
        max_buffer_size: int = 100,
        openness: float = 0.5,
        expressiveness: float = 0.5,
        max_history: int = 50,
        objectivity_gate_threshold: float = 0.7,
        on_input: Optional[Callable[[str, Dict[str, Any]], None]] = None,
        on_output: Optional[Callable[[str, Any], None]] = None,
        name: str = "Observer",
        role: str = "sentient_observer",
    ):
        """
        Initialize boundary layer.
        
        Args:
            max_buffer_size: Maximum buffer size for input/output
            openness: How much external input influences internal (0-1)
            expressiveness: How much internal state affects output (0-1)
            max_history: Maximum history entries for environmental model
            objectivity_gate_threshold: Threshold for objectivity gate
            on_input: Callback for input events
            on_output: Callback for output events
            name: Identity name
            role: Identity role
        """
        # Sensory and motor channels
        self.sensory_channels: Dict[str, SensoryChannel] = {}
        self.motor_channels: Dict[str, MotorChannel] = {}
        
        # Internal models
        self.environment = EnvironmentalModel(max_history=max_history)
        self.self_model = BoundarySelfModel(name=name, role=role)
        
        # Objectivity gate for output validation (Section 7, equation 18)
        self.objectivity_gate = ObjectivityGate(threshold=objectivity_gate_threshold)
        
        # Input/output buffers
        self.input_buffer: List[Dict[str, Any]] = []
        self.output_buffer: List[Dict[str, Any]] = []
        self.max_buffer_size = max_buffer_size
        
        # Processing state
        self.processing_input = False
        self.last_input_time: Optional[float] = None
        self.last_output_time: Optional[float] = None
        
        # Boundary parameters
        self.openness = openness
        self.expressiveness = expressiveness
        
        # Callbacks
        self.on_input = on_input
        self.on_output = on_output
        
        # Initialize default channels
        self._initialize_default_channels()
    
    def _initialize_default_channels(self) -> None:
        """Initialize default sensory and motor channels."""
        # Default sensory channels
        self.add_sensory_channel(SensoryChannel(
            name="text_input",
            type="text",
            associated_primes=[2, 3, 5, 7, 11],
        ))
        
        self.add_sensory_channel(SensoryChannel(
            name="user_state",
            type="embedding",
            associated_primes=[13, 17, 19, 23],
        ))
        
        # Default motor channels
        self.add_motor_channel(MotorChannel(
            name="text_output",
            type="text",
            trigger_primes=[29, 31, 37, 41],
        ))
        
        self.add_motor_channel(MotorChannel(
            name="action_output",
            type="action",
            trigger_primes=[43, 47, 53],
        ))
    
    def add_sensory_channel(self, channel: SensoryChannel) -> None:
        """Add a sensory channel."""
        self.sensory_channels[channel.name] = channel
    
    def add_motor_channel(self, channel: MotorChannel) -> None:
        """Add a motor channel."""
        self.motor_channels[channel.name] = channel
    
    def process_input(
        self,
        channel_name: str,
        value: Any,
    ) -> Optional[Dict[str, Any]]:
        """
        Process input from a sensory channel.
        
        Args:
            channel_name: Name of the sensory channel
            value: Input value
            
        Returns:
            Result with channel, result, and associated primes
        """
        channel = self.sensory_channels.get(channel_name)
        if not channel or not channel.enabled:
            return None
        
        result = channel.update(value)
        
        # Buffer the input
        self.input_buffer.append({
            "channel": channel_name,
            **result,
            "timestamp": time.time() * 1000,
        })
        
        if len(self.input_buffer) > self.max_buffer_size:
            self.input_buffer.pop(0)
        
        self.last_input_time = time.time() * 1000
        
        # Update environmental model if appropriate
        if channel.type == "text":
            self.environment.update_context({"last_input": value})
        
        if self.on_input:
            self.on_input(channel_name, result)
        
        return {
            "channel": channel_name,
            "result": result,
            "primes": channel.associated_primes,
        }
    
    def queue_output(
        self,
        channel_name: str,
        output: Any,
        context: Optional[Dict[str, Any]] = None,
        skip_gate: bool = False,
    ) -> Dict[str, Any]:
        """
        Queue output to a motor channel with objectivity gate check (equation 18).
        
        Args:
            channel_name: Name of motor channel
            output: Output to queue
            context: Context for objectivity gate
            skip_gate: Skip objectivity gate (for internal outputs)
            
        Returns:
            Result with queued status and gate result
        """
        channel = self.motor_channels.get(channel_name)
        if not channel or not channel.enabled:
            return {"queued": False, "reason": "channel_unavailable"}
        
        context = context or {}
        
        # Check objectivity gate unless skipped
        gate_result: Dict[str, Any] = {"should_broadcast": True}
        if not skip_gate:
            gate_result = self.objectivity_gate.check(output, context)
            
            if not gate_result["should_broadcast"]:
                # Output failed objectivity gate - do not broadcast
                self.output_buffer.append({
                    "channel": channel_name,
                    "output": output,
                    "queued_at": time.time() * 1000,
                    "blocked": True,
                    "gate_result": gate_result,
                })
                
                if len(self.output_buffer) > self.max_buffer_size:
                    self.output_buffer.pop(0)
                
                return {
                    "queued": False,
                    "reason": "objectivity_gate_failed",
                    "gate_result": gate_result,
                }
        
        channel.queue(output)
        
        self.output_buffer.append({
            "channel": channel_name,
            "output": output,
            "queued_at": time.time() * 1000,
            "blocked": False,
            "gate_result": gate_result,
        })
        
        if len(self.output_buffer) > self.max_buffer_size:
            self.output_buffer.pop(0)
        
        return {"queued": True, "gate_result": gate_result}
    
    def get_ready_outputs(self) -> List[Dict[str, Any]]:
        """Get ready outputs from all motor channels."""
        outputs = []
        
        for name, channel in self.motor_channels.items():
            output = channel.get_next()
            if output is not None:
                outputs.append({
                    "channel": name,
                    "output": output,
                })
                
                if self.on_output:
                    self.on_output(name, output)
        
        if outputs:
            self.last_output_time = time.time() * 1000
        
        return outputs
    
    def classify_origin(self, smf_state: Any) -> str:
        """
        Determine if input is "self" or "other" based on SMF.
        
        Args:
            smf_state: Current SMF state
            
        Returns:
            'self' or 'other'
        """
        if self.self_model.is_self_like(smf_state):
            return "self"
        return "other"
    
    def update_self(
        self,
        smf: Any,
        state: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Update self model based on current state.
        
        Args:
            smf: Current SMF state
            state: Additional state updates
        """
        state = state or {}
        self.self_model.update_self_orientation(smf)
        self.self_model.update_state(state)
        
        # Add continuity marker if significant
        if state.get("significant"):
            self.self_model.add_continuity_marker({
                "type": "significant_event",
                "description": state.get("description", "Significant state change"),
            })
    
    def update_environment(self, updates: Dict[str, Any]) -> None:
        """
        Update environmental model.
        
        Args:
            updates: Updates with optional 'entity', 'context', 'relationship' keys
        """
        if "entity" in updates:
            entity = updates["entity"]
            self.environment.update_entity(entity.get("id", ""), entity)
        if "context" in updates:
            self.environment.update_context(updates["context"])
        if "relationship" in updates:
            rel = updates["relationship"]
            self.environment.add_relationship(
                rel.get("source", ""),
                rel.get("target", ""),
                rel.get("type", ""),
            )
        
        self.environment.decay_uncertainty()
    
    def get_input_primes(self) -> List[int]:
        """Get primes associated with current input."""
        primes = []
        for channel in self.sensory_channels.values():
            if channel.is_active():
                primes.extend(channel.associated_primes)
        return list(set(primes))
    
    def get_stats(self) -> Dict[str, Any]:
        """Get statistics."""
        return {
            "sensory_channels": len(self.sensory_channels),
            "active_sensory_channels": len([
                c for c in self.sensory_channels.values()
                if c.is_active()
            ]),
            "motor_channels": len(self.motor_channels),
            "pending_outputs": sum(
                c.queue_length for c in self.motor_channels.values()
            ),
            "input_buffer_size": len(self.input_buffer),
            "output_buffer_size": len(self.output_buffer),
            "environment_entities": len(self.environment.entities),
            "environment_uncertainty": self.environment.uncertainty,
            "self_state": self.self_model.state,
            "objectivity_gate": self.objectivity_gate.get_stats(),
        }
    
    def reset(self) -> None:
        """Reset boundary layer."""
        self.sensory_channels.clear()
        self.motor_channels.clear()
        self.environment = EnvironmentalModel()
        self.input_buffer = []
        self.output_buffer = []
        self._initialize_default_channels()
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "sensory_channels": [c.to_dict() for c in self.sensory_channels.values()],
            "motor_channels": [c.to_dict() for c in self.motor_channels.values()],
            "environment": self.environment.to_dict(),
            "self": self.self_model.to_dict(),
            "openness": self.openness,
            "expressiveness": self.expressiveness,
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        if "sensory_channels" in data:
            self.sensory_channels.clear()
            for channel_data in data["sensory_channels"]:
                channel = SensoryChannel.from_dict(channel_data)
                self.sensory_channels[channel.name] = channel
        if "environment" in data:
            self.environment.load_from_dict(data["environment"])
        if "self" in data:
            self.self_model.load_from_dict(data["self"])
        if "openness" in data:
            self.openness = data["openness"]
        if "expressiveness" in data:
            self.expressiveness = data["expressiveness"]