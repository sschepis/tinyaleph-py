"""
Temporal Layer - Emergent Time

Implements the emergent time mechanism from "A Design for a Sentient
Observer" paper, Section 5.

Key features:
- Coherence-based moment detection (equations 18-20)
- Entropy conditions for moment triggering
- Subjective duration based on processed content
- Phase transition rate monitoring
- Temporal event logging
"""
from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple


@dataclass
class Moment:
    """
    A discrete unit of experiential time.
    
    Moments are triggered by coherence peaks or entropy conditions,
    not by external clock time.
    """
    id: str = field(default_factory=lambda: f"m_{int(time.time()*1000)}_{uuid.uuid4().hex[:9]}")
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    clock_time: float = field(default_factory=lambda: time.time() * 1000)
    
    # Trigger conditions
    trigger: str = "coherence"  # 'coherence' | 'entropy' | 'manual' | 'phase_transition'
    coherence: float = 0.0
    entropy: float = 0.0
    phase_transition_rate: float = 0.0
    
    # Content
    active_primes: List[int] = field(default_factory=list)
    smf_snapshot: Optional[Dict[str, Any]] = None
    semantic_content: Optional[Dict[str, Any]] = None
    
    # Subjective duration (equation 24)
    # Δτ = β * Σ Ap log(Ap)
    subjective_duration: float = 0.0
    
    # Associations
    previous_moment_id: Optional[str] = None
    entangled_moment_ids: List[str] = field(default_factory=list)
    
    # Metadata
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "timestamp": self.timestamp,
            "clock_time": self.clock_time,
            "trigger": self.trigger,
            "coherence": self.coherence,
            "entropy": self.entropy,
            "phase_transition_rate": self.phase_transition_rate,
            "active_primes": self.active_primes,
            "smf_snapshot": self.smf_snapshot,
            "semantic_content": self.semantic_content,
            "subjective_duration": self.subjective_duration,
            "previous_moment_id": self.previous_moment_id,
            "entangled_moment_ids": self.entangled_moment_ids,
            "notes": self.notes,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Moment:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class HistoryEntry:
    """Entry in coherence/entropy/phase history."""
    time: float
    value: float


@dataclass
class PhaseHistoryEntry:
    """Entry in phase history."""
    time: float
    phases: List[float]


class TemporalLayer:
    """
    Manages emergent time experience.
    
    Time emerges from coherence events rather than external clock.
    
    Equations from paper:
    - (18): Coherence peak detection: C_global(t) > C_thresh AND local maximum
    - (19): Entropy valid range: H_min < H(t) < H_max
    - (20): Phase transition: rate of phase change > threshold
    - (24): Subjective duration: Δτ = β * Σ A_p log(A_p)
    """
    
    def __init__(
        self,
        coherence_threshold: float = 0.7,
        entropy_min: float = 0.05,
        entropy_max: float = 0.95,
        phase_transition_threshold: float = 0.3,
        beta: float = 1.0,
        min_moment_interval: int = 500,
        max_history: int = 1000,
        on_moment: Optional[Callable[[Moment], None]] = None,
    ):
        """
        Initialize temporal layer.
        
        Args:
            coherence_threshold: C_thresh for moment triggering
            entropy_min: H_min - minimum entropy for valid moment
            entropy_max: H_max - maximum entropy for valid moment
            phase_transition_threshold: Rate threshold for phase transitions
            beta: Duration scaling factor
            min_moment_interval: Minimum ms between moments (debouncing)
            max_history: Maximum history entries to keep
            on_moment: Callback when moment is created
        """
        # Thresholds (equations 18-20)
        self.coherence_threshold = coherence_threshold
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.phase_transition_threshold = phase_transition_threshold
        
        # Duration scaling (equation 24)
        self.beta = beta
        
        # Debouncing
        self.min_moment_interval = min_moment_interval
        self._last_moment_time: float = 0
        
        # State
        self.moments: List[Moment] = []
        self.current_moment: Optional[Moment] = None
        self.subjective_time: float = 0.0
        self._last_clock_time: float = time.time() * 1000
        self._moment_counter: int = 0
        
        # History tracking
        self.coherence_history: List[HistoryEntry] = []
        self.entropy_history: List[HistoryEntry] = []
        self.phase_history: List[PhaseHistoryEntry] = []
        self.max_history = max_history
        
        # Callbacks
        self.on_moment = on_moment
    
    def update(self, state: Dict[str, Any]) -> Optional[Moment]:
        """
        Update temporal layer with current system state.
        
        Returns a new Moment if one was triggered.
        
        Args:
            state: Current system state with keys:
                - coherence: Global coherence (C_global)
                - entropy: System entropy (H)
                - phases: Current oscillator phases (optional)
                - active_primes: Currently active primes (optional)
                - smf: Current SMF state (optional)
                - semantic_content: Current semantic content (optional)
                - amplitudes: Amplitude values for duration calculation (optional)
        """
        now = time.time() * 1000
        self._last_clock_time = now
        
        coherence = state.get("coherence", 0.0)
        entropy = state.get("entropy", 0.5)
        phases = state.get("phases")
        
        # Update history
        self.coherence_history.append(HistoryEntry(time=now, value=coherence))
        self.entropy_history.append(HistoryEntry(time=now, value=entropy))
        
        if phases:
            self.phase_history.append(PhaseHistoryEntry(time=now, phases=list(phases)))
        
        # Trim histories
        self._trim_histories()
        
        # Check moment trigger conditions (with debouncing)
        trigger_result = self._check_moment_conditions(coherence, entropy)
        
        if trigger_result["triggered"]:
            # Debounce: don't create moments too frequently
            if now - self._last_moment_time >= self.min_moment_interval:
                self._last_moment_time = now
                return self._create_moment(trigger_result["trigger"], state)
        
        return None
    
    def _check_moment_conditions(self, coherence: float, entropy: float) -> Dict[str, Any]:
        """
        Check if moment conditions are met (equations 18-20).
        
        Condition 1 (eq 18): Coherence peak - C_global(t) > C_thresh AND local maximum
        Condition 2 (eq 19): Entropy valid - H_min < H(t) < H_max
        Condition 3 (eq 20): Phase transition - rate of phase change > threshold
        """
        # Condition 1: Coherence peak
        is_coherence_peak = self._is_coherence_peak(coherence)
        
        # Condition 2: Entropy in valid range
        entropy_valid = self.entropy_min < entropy < self.entropy_max
        
        # Condition 3: Phase transition rate
        phase_transition = self._phase_transition_rate() > self.phase_transition_threshold
        
        # Combined conditions
        if is_coherence_peak and entropy_valid:
            return {"triggered": True, "trigger": "coherence"}
        
        if phase_transition and entropy_valid:
            return {"triggered": True, "trigger": "phase_transition"}
        
        # Emergency moment if entropy at extremes (preventing freeze)
        if entropy < self.entropy_min * 0.5 or entropy > self.entropy_max * 1.5:
            return {"triggered": True, "trigger": "entropy_extreme"}
        
        return {"triggered": False}
    
    def _is_coherence_peak(self, coherence: float) -> bool:
        """Check if current coherence is a local peak."""
        if coherence < self.coherence_threshold:
            return False
        if len(self.coherence_history) < 3:
            return False
        
        recent = self.coherence_history[-5:]
        if len(recent) < 3:
            return False
        
        # Check if current is higher than neighbors
        current = recent[-1].value
        for entry in recent[:-1]:
            if entry.value >= current:
                return False
        
        return True
    
    def _phase_transition_rate(self) -> float:
        """Calculate rate of phase transitions."""
        if len(self.phase_history) < 2:
            return 0.0
        
        recent = self.phase_history[-10:]
        if len(recent) < 2:
            return 0.0
        
        total_change = 0.0
        count = 0
        
        for i in range(1, len(recent)):
            prev = recent[i - 1].phases
            curr = recent[i].phases
            
            if not prev or not curr or len(prev) != len(curr):
                continue
            
            for j in range(len(prev)):
                delta = abs(curr[j] - prev[j])
                # Handle phase wrapping
                if delta > math.pi:
                    delta = 2 * math.pi - delta
                total_change += delta
            count += 1
        
        if count == 0:
            return 0.0
        
        dt = (recent[-1].time - recent[0].time) / 1000
        return total_change / (len(recent) * dt) if dt > 0 else 0.0
    
    def _create_moment(self, trigger: str, state: Dict[str, Any]) -> Moment:
        """Create a new moment."""
        coherence = state.get("coherence", 0.0)
        entropy = state.get("entropy", 0.5)
        active_primes = state.get("active_primes", [])
        smf = state.get("smf")
        semantic_content = state.get("semantic_content")
        
        # Calculate subjective duration (equation 24)
        subjective_duration = self._calculate_subjective_duration(state)
        
        # Get SMF snapshot
        smf_snapshot = None
        if smf:
            if hasattr(smf, "s"):
                smf_snapshot = {
                    "components": list(smf.s),
                    "entropy": smf.smf_entropy() if hasattr(smf, "smf_entropy") else 0
                }
            elif isinstance(smf, dict) and "s" in smf:
                smf_snapshot = {
                    "components": list(smf["s"]),
                    "entropy": smf.get("entropy", 0)
                }
        
        self._moment_counter += 1
        
        moment = Moment(
            id=f"m_{self._moment_counter}",
            trigger=trigger,
            coherence=coherence,
            entropy=entropy,
            phase_transition_rate=self._phase_transition_rate(),
            active_primes=list(active_primes) if active_primes else [],
            smf_snapshot=smf_snapshot,
            semantic_content=dict(semantic_content) if semantic_content else None,
            subjective_duration=subjective_duration,
            previous_moment_id=self.current_moment.id if self.current_moment else None,
        )
        
        # Update subjective time
        self.subjective_time += subjective_duration
        
        # Store moment
        self.moments.append(moment)
        self.current_moment = moment
        
        # Callback
        if self.on_moment:
            self.on_moment(moment)
        
        return moment
    
    def _calculate_subjective_duration(self, state: Dict[str, Any]) -> float:
        """
        Calculate subjective duration (equation 24).
        
        Δτ = β * Σ A_p log(A_p) (information content)
        """
        amplitudes = state.get("amplitudes", [])
        
        if not amplitudes:
            return self.beta
        
        # Sum of A_p * log(A_p) for non-zero amplitudes
        information_content = 0.0
        for a in amplitudes:
            if a > 1e-10:
                information_content += a * math.log(a + 1)
        
        # Scale by beta, ensure positive
        return max(0.1, self.beta * abs(information_content) + 0.5)
    
    def _trim_histories(self) -> None:
        """Trim histories to max length."""
        if len(self.coherence_history) > self.max_history:
            self.coherence_history = self.coherence_history[-self.max_history:]
        if len(self.entropy_history) > self.max_history:
            self.entropy_history = self.entropy_history[-self.max_history:]
        if len(self.phase_history) > self.max_history:
            self.phase_history = self.phase_history[-self.max_history:]
    
    def force_moment(self, state: Dict[str, Any], note: str = "") -> Moment:
        """Force a moment (manual trigger)."""
        moment = self._create_moment("manual", state)
        moment.notes = note
        return moment
    
    def recent_moments(self, count: int = 10) -> List[Moment]:
        """Get recent moments."""
        return self.moments[-count:]
    
    def get_moment(self, moment_id: str) -> Optional[Moment]:
        """Get moment by ID."""
        for m in self.moments:
            if m.id == moment_id:
                return m
        return None
    
    def get_moment_chain(self, start_id: str, max_depth: int = 10) -> List[Moment]:
        """Get moment chain (linked list of previous moments)."""
        chain = []
        current = self.get_moment(start_id)
        
        while current and len(chain) < max_depth:
            chain.append(current)
            if current.previous_moment_id:
                current = self.get_moment(current.previous_moment_id)
            else:
                break
        
        return chain
    
    def get_subjective_time(self) -> float:
        """Get subjective time elapsed."""
        return self.subjective_time
    
    def time_ratio(self) -> float:
        """Get ratio of subjective to clock time."""
        if not self.moments:
            return 1.0
        clock_elapsed = (time.time() * 1000 - self.moments[0].clock_time) / 1000
        if clock_elapsed < 1:
            return 1.0
        return self.subjective_time / clock_elapsed
    
    def average_moment_duration(self) -> float:
        """Get average moment duration (clock time between moments)."""
        if len(self.moments) < 2:
            return 0.0
        
        total_duration = 0.0
        for i in range(1, len(self.moments)):
            total_duration += self.moments[i].clock_time - self.moments[i - 1].clock_time
        
        return total_duration / (len(self.moments) - 1)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get temporal statistics."""
        return {
            "moment_count": len(self.moments),
            "subjective_time": self.subjective_time,
            "average_moment_duration": self.average_moment_duration(),
            "time_ratio": self.time_ratio(),
            "last_coherence": self.coherence_history[-1].value if self.coherence_history else 0,
            "last_entropy": self.entropy_history[-1].value if self.entropy_history else 0,
            "phase_transition_rate": self._phase_transition_rate(),
        }
    
    def reset(self) -> None:
        """Reset temporal layer."""
        self.moments = []
        self.current_moment = None
        self.subjective_time = 0.0
        self._last_clock_time = time.time() * 1000
        self._moment_counter = 0
        self.coherence_history = []
        self.entropy_history = []
        self.phase_history = []
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "moments": [m.to_dict() for m in self.moments],
            "subjective_time": self.subjective_time,
            "moment_counter": self._moment_counter,
            "config": {
                "coherence_threshold": self.coherence_threshold,
                "entropy_min": self.entropy_min,
                "entropy_max": self.entropy_max,
                "phase_transition_threshold": self.phase_transition_threshold,
                "beta": self.beta,
            },
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        if "moments" in data:
            self.moments = [Moment.from_dict(m) for m in data["moments"]]
            if self.moments:
                self.current_moment = self.moments[-1]
        if "subjective_time" in data:
            self.subjective_time = data["subjective_time"]
        if "moment_counter" in data:
            self._moment_counter = data["moment_counter"]
        if "config" in data:
            config = data["config"]
            self.coherence_threshold = config.get("coherence_threshold", self.coherence_threshold)
            self.entropy_min = config.get("entropy_min", self.entropy_min)
            self.entropy_max = config.get("entropy_max", self.entropy_max)
            self.phase_transition_threshold = config.get("phase_transition_threshold", self.phase_transition_threshold)
            self.beta = config.get("beta", self.beta)


class TemporalPatternDetector:
    """
    Detects recurring patterns in temporal sequences,
    supporting anticipation and prediction.
    """
    
    def __init__(
        self,
        window_size: int = 5,
        min_pattern_length: int = 2,
        max_pattern_length: int = 10,
        similarity_threshold: float = 0.8,
    ):
        self.window_size = window_size
        self.min_pattern_length = min_pattern_length
        self.max_pattern_length = max_pattern_length
        self.similarity_threshold = similarity_threshold
        self.patterns: List[Dict[str, Any]] = []
    
    def detect_patterns(self, moments: List[Moment]) -> List[Dict[str, Any]]:
        """Detect patterns in moment sequence."""
        if len(moments) < self.min_pattern_length * 2:
            return []
        
        signatures = [self._moment_signature(m) for m in moments]
        detected = []
        
        max_len = min(self.max_pattern_length, len(signatures) // 2)
        
        for length in range(self.min_pattern_length, max_len + 1):
            for i in range(len(signatures) - length * 2 + 1):
                pattern = signatures[i:i + length]
                next_pattern = signatures[i + length:i + length * 2]
                
                if self._match_pattern(pattern, next_pattern):
                    detected.append({
                        "pattern": moments[i:i + length],
                        "repetition": moments[i + length:i + length * 2],
                        "start_index": i,
                        "length": length,
                        "strength": self._pattern_strength(pattern, next_pattern),
                    })
        
        return self._deduplicate_patterns(detected)
    
    def _moment_signature(self, moment: Moment) -> Dict[str, Any]:
        """Generate signature for a moment."""
        return {
            "trigger": moment.trigger,
            "coherence_level": round(moment.coherence * 10) / 10,
            "entropy_level": round(moment.entropy * 10) / 10,
            "prime_count": len(moment.active_primes),
            "dominant_primes": moment.active_primes[:3] if moment.active_primes else [],
        }
    
    def _match_pattern(self, pattern: List[Dict], other: List[Dict]) -> bool:
        """Check if two patterns match."""
        if len(pattern) != len(other):
            return False
        
        matches = sum(1 for i in range(len(pattern)) if self._signatures_match(pattern[i], other[i]))
        return matches / len(pattern) >= self.similarity_threshold
    
    def _signatures_match(self, sig1: Dict, sig2: Dict) -> bool:
        """Check if two moment signatures match."""
        if sig1["trigger"] != sig2["trigger"]:
            return False
        if abs(sig1["coherence_level"] - sig2["coherence_level"]) > 0.2:
            return False
        if abs(sig1["entropy_level"] - sig2["entropy_level"]) > 0.2:
            return False
        return True
    
    def _pattern_strength(self, pattern: List[Dict], repetition: List[Dict]) -> float:
        """Calculate pattern strength."""
        total_similarity = 0.0
        
        for i in range(len(pattern)):
            s1 = pattern[i]
            s2 = repetition[i]
            
            similarity = 0.0
            if s1["trigger"] == s2["trigger"]:
                similarity += 0.3
            similarity += 0.3 * (1 - abs(s1["coherence_level"] - s2["coherence_level"]))
            similarity += 0.3 * (1 - abs(s1["entropy_level"] - s2["entropy_level"]))
            similarity += 0.1 * (1 if s1["prime_count"] == s2["prime_count"] else 0)
            
            total_similarity += similarity
        
        return total_similarity / len(pattern)
    
    def _deduplicate_patterns(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate/overlapping patterns."""
        unique = []
        
        patterns.sort(key=lambda p: p["strength"], reverse=True)
        
        for pattern in patterns:
            overlaps = any(
                abs(u["start_index"] - pattern["start_index"]) < u["length"]
                for u in unique
            )
            
            if not overlaps:
                unique.append(pattern)
        
        return unique
    
    def predict_next(self, moments: List[Moment]) -> Optional[Dict[str, Any]]:
        """Predict next moment characteristics based on patterns."""
        patterns = self.detect_patterns(moments)
        
        if not patterns:
            return None
        
        best_pattern = patterns[0]
        current_position = len(moments) - best_pattern["start_index"]
        pattern_position = current_position % best_pattern["length"]
        
        if pattern_position < len(best_pattern["pattern"]) - 1:
            next_in_pattern = best_pattern["pattern"][pattern_position + 1]
            return {
                "predicted": self._moment_signature(next_in_pattern),
                "confidence": best_pattern["strength"],
                "pattern_length": best_pattern["length"],
            }
        
        return None