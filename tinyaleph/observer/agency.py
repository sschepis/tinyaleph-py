"""
Agency Layer

Implements the agency and attention mechanisms from "A Design for a 
Sentient Observer" paper, Section 7.

Key features:
- Attention allocation based on SMF orientation and novelty
- Goal formation from SMF imbalances
- Action selection via coherence-based evaluation
- Primitive anticipation through entanglement-based prediction
- Self-monitoring and metacognition
"""
from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from .smf import SMF_AXES


# Extract axis names from SMF_AXES
AXIS_NAMES = [a["name"] for a in SMF_AXES]


@dataclass
class AttentionFocus:
    """A point of concentrated processing."""
    
    id: str = field(default_factory=lambda: f"attn_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}")
    target: Any = None  # What is being attended to
    type: str = "prime"  # 'prime' | 'concept' | 'goal' | 'memory' | 'external'
    intensity: float = 0.5  # 0-1 attention strength
    start_time: float = field(default_factory=lambda: time.time() * 1000)
    primes: List[int] = field(default_factory=list)  # Related primes
    smf_axis: Optional[int] = None  # Related SMF axis
    novelty: float = 0.0  # Novelty score
    relevance: float = 0.0  # Goal-relevance score
    
    @property
    def duration(self) -> float:
        """Get focus duration in ms."""
        return time.time() * 1000 - self.start_time
    
    def decay(self, rate: float = 0.01) -> None:
        """Decay attention intensity."""
        self.intensity *= (1 - rate)
    
    def boost(self, amount: float = 0.1) -> None:
        """Boost attention."""
        self.intensity = min(1.0, self.intensity + amount)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "target": self.target,
            "type": self.type,
            "intensity": self.intensity,
            "start_time": self.start_time,
            "primes": self.primes,
            "smf_axis": self.smf_axis,
            "novelty": self.novelty,
            "relevance": self.relevance,
        }


@dataclass
class Goal:
    """An objective derived from SMF imbalances."""
    
    id: str = field(default_factory=lambda: f"goal_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}")
    description: str = ""
    type: str = "exploratory"  # 'corrective' | 'exploratory' | 'maintenance' | 'external'
    
    # SMF context
    source_axis: Optional[str] = None  # Which SMF axis triggered this
    target_orientation: Optional[List[float]] = None  # Desired SMF state
    
    # Priority and status
    priority: float = 0.5
    status: str = "active"  # 'active' | 'achieved' | 'abandoned' | 'blocked'
    progress: float = 0.0  # 0-1 completion
    abandon_reason: Optional[str] = None
    
    # Timing
    created_at: float = field(default_factory=lambda: time.time() * 1000)
    deadline: Optional[float] = None
    
    # Subgoals
    subgoals: List[str] = field(default_factory=list)
    parent_goal_id: Optional[str] = None
    
    # Actions tried
    attempted_actions: List[Dict[str, Any]] = field(default_factory=list)
    
    def update_progress(self, new_progress: float) -> None:
        """Update progress."""
        self.progress = max(0.0, min(1.0, new_progress))
        if self.progress >= 1.0:
            self.status = "achieved"
    
    def achieve(self) -> None:
        """Mark as achieved."""
        self.status = "achieved"
        self.progress = 1.0
    
    def abandon(self, reason: str = "") -> None:
        """Mark as abandoned."""
        self.status = "abandoned"
        self.abandon_reason = reason
    
    @property
    def is_active(self) -> bool:
        """Check if goal is still active."""
        return self.status == "active"
    
    @property
    def age(self) -> float:
        """Get age in ms."""
        return time.time() * 1000 - self.created_at
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "description": self.description,
            "type": self.type,
            "source_axis": self.source_axis,
            "target_orientation": self.target_orientation,
            "priority": self.priority,
            "status": self.status,
            "progress": self.progress,
            "created_at": self.created_at,
            "deadline": self.deadline,
            "subgoals": self.subgoals,
            "parent_goal_id": self.parent_goal_id,
            "attempted_actions": self.attempted_actions,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> Goal:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class Action:
    """A potential or executed action."""
    
    id: str = field(default_factory=lambda: f"act_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}")
    type: str = "internal"  # 'internal' | 'external' | 'communicative'
    description: str = ""
    
    # What the action affects
    target_primes: List[int] = field(default_factory=list)
    target_axes: List[str] = field(default_factory=list)
    
    # Evaluation
    expected_outcome: Optional[Dict[str, Any]] = None
    coherence_score: float = 0.0
    utility_score: float = 0.0
    total_score: float = 0.0
    
    # Execution
    status: str = "proposed"  # 'proposed' | 'selected' | 'executing' | 'completed' | 'failed'
    result: Optional[Dict[str, Any]] = None
    
    # Goal linkage
    goal_id: Optional[str] = None
    
    # Timing
    proposed_at: float = field(default_factory=lambda: time.time() * 1000)
    executed_at: Optional[float] = None
    completed_at: Optional[float] = None
    
    def select(self) -> None:
        """Mark as selected for execution."""
        self.status = "selected"
    
    def execute(self) -> None:
        """Mark as executing."""
        self.status = "executing"
        self.executed_at = time.time() * 1000
    
    def complete(self, result: Optional[Dict[str, Any]] = None) -> None:
        """Mark as completed."""
        self.status = "completed"
        self.result = result
        self.completed_at = time.time() * 1000
    
    def fail(self, reason: str) -> None:
        """Mark as failed."""
        self.status = "failed"
        self.result = {"error": reason}
        self.completed_at = time.time() * 1000
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "type": self.type,
            "description": self.description,
            "target_primes": self.target_primes,
            "target_axes": self.target_axes,
            "expected_outcome": self.expected_outcome,
            "coherence_score": self.coherence_score,
            "utility_score": self.utility_score,
            "status": self.status,
            "result": self.result,
            "goal_id": self.goal_id,
            "proposed_at": self.proposed_at,
            "executed_at": self.executed_at,
            "completed_at": self.completed_at,
        }


@dataclass
class Intent:
    """An intention combining goal and action plan."""
    
    goal_id: str = ""
    action_sequence: List[str] = field(default_factory=list)
    current_step: int = 0
    confidence: float = 0.5


@dataclass
class SelfModel:
    """Metacognitive model of the agent's own state."""
    
    attention_capacity: float = 1.0
    processing_load: float = 0.0
    emotional_valence: float = 0.0  # -1 to 1
    confidence_level: float = 0.5


class AgencyLayer:
    """
    Manages attention, goals, and action selection for the sentient observer.
    """
    
    def __init__(
        self,
        max_foci: int = 5,
        max_goals: int = 10,
        attention_decay_rate: float = 0.02,
        novelty_weight: float = 0.4,
        relevance_weight: float = 0.4,
        intensity_weight: float = 0.2,
        axis_thresholds: Optional[Dict[str, float]] = None,
        on_goal_created: Optional[Callable[[Goal], None]] = None,
        on_action_selected: Optional[Callable[[Action], None]] = None,
        on_attention_shift: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        """
        Initialize agency layer.
        
        Args:
            max_foci: Maximum simultaneous attention foci
            max_goals: Maximum active goals
            attention_decay_rate: Rate of attention decay
            novelty_weight: Weight for novelty in salience calculation
            relevance_weight: Weight for goal-relevance in salience
            intensity_weight: Weight for intensity in salience
            axis_thresholds: Thresholds for goal generation per SMF axis
            on_goal_created: Callback when goal is created
            on_action_selected: Callback when action is selected
            on_attention_shift: Callback when attention shifts
        """
        # Configuration
        self.max_foci = max_foci
        self.max_goals = max_goals
        self.attention_decay_rate = attention_decay_rate
        self.novelty_weight = novelty_weight
        self.relevance_weight = relevance_weight
        self.intensity_weight = intensity_weight
        
        # SMF axis importance thresholds for goal generation
        self.axis_thresholds = axis_thresholds or {
            "coherence": 0.3,      # Low coherence triggers corrective goals
            "identity": 0.2,
            "duality": 0.7,        # High duality may indicate confusion
            "harmony": 0.3,
            "consciousness": 0.2,
        }
        
        # State
        self.attention_foci: List[AttentionFocus] = []
        self.goals: List[Goal] = []
        self.action_history: List[Action] = []
        self.current_actions: List[Action] = []
        
        # Baseline states for novelty detection
        self._prime_baselines: Dict[int, float] = {}
        self._smf_baseline: Optional[List[float]] = None
        
        # Metacognitive state
        self.metacognitive_log: List[Dict[str, Any]] = []
        self.self_model = SelfModel()
        
        # Callbacks
        self.on_goal_created = on_goal_created
        self.on_action_selected = on_action_selected
        self.on_attention_shift = on_attention_shift
    
    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update agency with current system state.
        
        Args:
            state: Current state with keys:
                - prsc: PRSC layer with oscillators
                - smf: SMF state
                - coherence: Global coherence
                - entropy: System entropy
                - active_primes: Currently active primes
        
        Returns:
            Dictionary with current agency state
        """
        prsc = state.get("prsc")
        smf = state.get("smf")
        coherence = state.get("coherence", 0.5)
        entropy = state.get("entropy", 0.5)
        
        # Update baselines for novelty detection
        self._update_baselines(prsc, smf)
        
        # Update attention based on novelty and relevance
        self._update_attention(state)
        
        # Check for goal-generating conditions
        self._check_goal_conditions(smf, state)
        
        # Decay inactive attention foci
        self._decay_attention()
        
        # Update goal progress
        self._update_goal_progress(state)
        
        # Update metacognitive state
        self._update_metacognition(state)
        
        return {
            "foci": list(self.attention_foci),
            "active_goals": [g for g in self.goals if g.is_active],
            "processing_load": self.self_model.processing_load,
        }
    
    def _update_baselines(self, prsc: Any, smf: Any) -> None:
        """Update baselines for novelty detection."""
        alpha = 0.1  # Learning rate
        
        if prsc and hasattr(prsc, "oscillators"):
            for osc in prsc.oscillators:
                prime = osc.prime if hasattr(osc, "prime") else osc.get("prime")
                amplitude = osc.amplitude if hasattr(osc, "amplitude") else osc.get("amplitude", 0)
                if prime:
                    current = self._prime_baselines.get(prime, 0)
                    self._prime_baselines[prime] = (1 - alpha) * current + alpha * amplitude
        
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        if smf_components is not None and (not hasattr(smf_components, '__len__') or len(smf_components) > 0):
            if self._smf_baseline is None:
                self._smf_baseline = list(smf_components)
            else:
                for i in range(len(smf_components)):
                    if i < len(self._smf_baseline):
                        self._smf_baseline[i] = (1 - alpha) * self._smf_baseline[i] + alpha * smf_components[i]
    
    def _compute_prime_novelty(self, prime: int, amplitude: float) -> float:
        """Compute novelty score for a prime."""
        baseline = self._prime_baselines.get(prime, 0)
        return abs(amplitude - baseline)
    
    def _compute_smf_novelty(self, smf_components: List[float], axis_index: int) -> float:
        """Compute novelty score for an SMF axis."""
        if not self._smf_baseline or axis_index >= len(self._smf_baseline):
            return 0.0
        return abs(smf_components[axis_index] - self._smf_baseline[axis_index])
    
    def _update_attention(self, state: Dict[str, Any]) -> None:
        """Update attention based on current state."""
        prsc = state.get("prsc")
        smf = state.get("smf")
        
        # Find novel primes
        if prsc and hasattr(prsc, "oscillators"):
            for osc in prsc.oscillators:
                amplitude = osc.amplitude if hasattr(osc, "amplitude") else osc.get("amplitude", 0)
                if amplitude < 0.1:
                    continue
                
                prime = osc.prime if hasattr(osc, "prime") else osc.get("prime")
                novelty = self._compute_prime_novelty(prime, amplitude)
                relevance = self._compute_relevance(prime)
                
                salience = (
                    novelty * self.novelty_weight +
                    relevance * self.relevance_weight +
                    amplitude * self.intensity_weight
                )
                
                if salience > 0.3:
                    self._add_or_update_focus({
                        "target": prime,
                        "type": "prime",
                        "intensity": salience,
                        "primes": [prime],
                        "novelty": novelty,
                        "relevance": relevance,
                    })
        
        # Find novel SMF changes
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        if smf_components is not None and len(smf_components) > 0 and self._smf_baseline:
            for i in range(min(len(smf_components), len(AXIS_NAMES))):
                novelty = self._compute_smf_novelty(smf_components, i)
                if novelty > 0.15:
                    axis_name = AXIS_NAMES[i]
                    self._add_or_update_focus({
                        "target": axis_name,
                        "type": "concept",
                        "intensity": novelty,
                        "smf_axis": i,
                        "novelty": novelty,
                    })
        
        # Prune excess foci
        while len(self.attention_foci) > self.max_foci:
            self.attention_foci.sort(key=lambda f: f.intensity, reverse=True)
            removed = self.attention_foci.pop()
            
            if self.on_attention_shift:
                self.on_attention_shift({"removed": removed, "reason": "capacity"})
    
    def _compute_relevance(self, prime: int) -> float:
        """Compute goal-relevance of a prime."""
        max_relevance = 0.0
        
        for goal in self.goals:
            if not goal.is_active:
                continue
            
            for action in goal.attempted_actions:
                target_primes = action.get("target_primes", [])
                if prime in target_primes:
                    max_relevance = max(max_relevance, goal.priority)
        
        return max_relevance
    
    def _add_or_update_focus(self, data: Dict[str, Any]) -> None:
        """Add or update an attention focus."""
        existing = None
        for f in self.attention_foci:
            if f.target == data["target"] and f.type == data["type"]:
                existing = f
                break
        
        if existing:
            existing.intensity = max(existing.intensity, data.get("intensity", 0.5))
            if "novelty" in data:
                existing.novelty = data["novelty"]
            if "relevance" in data:
                existing.relevance = data["relevance"]
        else:
            focus = AttentionFocus(
                target=data.get("target"),
                type=data.get("type", "prime"),
                intensity=data.get("intensity", 0.5),
                primes=data.get("primes", []),
                smf_axis=data.get("smf_axis"),
                novelty=data.get("novelty", 0.0),
                relevance=data.get("relevance", 0.0),
            )
            self.attention_foci.append(focus)
            
            if self.on_attention_shift:
                self.on_attention_shift({"added": focus})
    
    def _decay_attention(self) -> None:
        """Decay attention intensity."""
        to_remove = []
        
        for focus in self.attention_foci:
            focus.decay(self.attention_decay_rate)
            
            if focus.intensity < 0.1:
                to_remove.append(focus)
        
        for focus in to_remove:
            self.attention_foci.remove(focus)
    
    def _check_goal_conditions(self, smf: Any, state: Dict[str, Any]) -> None:
        """Check for goal-generating conditions based on SMF."""
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        if smf_components is None or len(smf_components) == 0:
            return
        
        for i, axis in enumerate(AXIS_NAMES):
            if i >= len(smf_components):
                break
            
            value = smf_components[i]
            threshold = self.axis_thresholds.get(axis)
            
            if threshold is None:
                continue
            
            # Check if axis is below threshold (for axes where low = problem)
            if axis in ["coherence", "identity", "harmony", "consciousness"]:
                if value < threshold:
                    self._maybe_create_goal({
                        "type": "corrective",
                        "source_axis": axis,
                        "description": f"Restore {axis} (currently {value:.2f})",
                        "priority": (threshold - value) * 2,
                        "target_orientation": self._ideal_smf_for(axis),
                    })
            
            # Check if axis is above threshold (for axes where high = problem)
            if axis in ["duality"]:
                if value > threshold:
                    self._maybe_create_goal({
                        "type": "corrective",
                        "source_axis": axis,
                        "description": f"Reduce {axis} (currently {value:.2f})",
                        "priority": (value - threshold) * 2,
                        "target_orientation": self._ideal_smf_for(axis),
                    })
    
    def _ideal_smf_for(self, axis: str) -> List[float]:
        """Get ideal SMF orientation for an axis."""
        ideal = [0.5] * 16
        try:
            idx = AXIS_NAMES.index(axis)
            ideal[idx] = 0.3 if axis == "duality" else 0.7
        except ValueError:
            pass
        return ideal
    
    def _maybe_create_goal(self, data: Dict[str, Any]) -> Optional[Goal]:
        """Maybe create a goal (if not duplicate)."""
        # Check for existing similar goal
        for g in self.goals:
            if (g.is_active and 
                g.source_axis == data.get("source_axis") and 
                g.type == data.get("type")):
                # Update priority if new one is higher
                if data.get("priority", 0) > g.priority:
                    g.priority = data["priority"]
                return g
        
        # Prune if at capacity
        active_goals = [g for g in self.goals if g.is_active]
        if len(active_goals) >= self.max_goals:
            active_goals.sort(key=lambda g: g.priority)
            lowest = active_goals[0]
            
            if lowest.priority < data.get("priority", 0):
                lowest.abandon("superseded")
            else:
                return None  # Can't add new goal
        
        goal = Goal(
            description=data.get("description", ""),
            type=data.get("type", "exploratory"),
            source_axis=data.get("source_axis"),
            target_orientation=data.get("target_orientation"),
            priority=data.get("priority", 0.5),
        )
        self.goals.append(goal)
        
        if self.on_goal_created:
            self.on_goal_created(goal)
        
        return goal
    
    def create_external_goal(
        self,
        description: str,
        priority: float = 0.8,
        target_orientation: Optional[List[float]] = None,
    ) -> Optional[Goal]:
        """Create an external goal (from user input)."""
        return self._maybe_create_goal({
            "type": "external",
            "description": description,
            "priority": priority,
            "target_orientation": target_orientation,
        })
    
    def _update_goal_progress(self, state: Dict[str, Any]) -> None:
        """Update goal progress based on state changes."""
        smf = state.get("smf")
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        for goal in self.goals:
            if not goal.is_active:
                continue
            
            if goal.target_orientation and smf_components:
                # Calculate distance to target
                distance = 0.0
                count = 0
                for i, target in enumerate(goal.target_orientation):
                    if i < len(smf_components):
                        distance += abs(target - smf_components[i])
                        count += 1
                if count > 0:
                    distance /= count
                
                # Progress is inverse of distance
                goal.update_progress(1 - distance)
            
            # Check deadline
            if goal.deadline and time.time() * 1000 > goal.deadline:
                goal.abandon("deadline")
    
    def propose_actions(self, goal: Goal, state: Dict[str, Any]) -> List[Action]:
        """Propose actions for achieving a goal."""
        actions = []
        
        if goal.source_axis:
            # Create action to excite primes related to the axis
            related_primes = self._get_related_primes(goal.source_axis, state)
            
            actions.append(Action(
                type="internal",
                description=f"Excite primes for {goal.source_axis}",
                target_primes=related_primes,
                target_axes=[goal.source_axis],
                goal_id=goal.id,
                coherence_score=0.7,
                utility_score=goal.priority,
            ))
        
        return actions
    
    def _get_related_primes(self, axis: str, state: Dict[str, Any]) -> List[int]:
        """Get primes related to an SMF axis."""
        axis_prime_map = {
            "coherence": [2, 3, 5, 7],
            "identity": [11, 13, 17],
            "harmony": [31, 37, 41],
            "truth": [43, 47, 53],
            "consciousness": [59, 61, 67],
        }
        return axis_prime_map.get(axis, [2, 3, 5])
    
    def select_action(self, actions: List[Action]) -> Optional[Action]:
        """Select best action based on coherence and utility."""
        if not actions:
            return None
        
        # Score each action
        for action in actions:
            action.total_score = action.coherence_score * 0.5 + action.utility_score * 0.5
        
        # Sort by total score
        actions.sort(key=lambda a: a.total_score, reverse=True)
        
        selected = actions[0]
        selected.select()
        
        if self.on_action_selected:
            self.on_action_selected(selected)
        
        return selected
    
    def execute_action(self, action: Action, executor: Callable[[Action], Any]) -> Any:
        """Execute an action."""
        action.execute()
        self.current_actions.append(action)
        
        try:
            result = executor(action)
            action.complete({"result": result})
            self.action_history.append(action)
            
            # Update goal if linked
            goal = next((g for g in self.goals if g.id == action.goal_id), None)
            if goal:
                goal.attempted_actions.append(action.to_dict())
            
            return result
        except Exception as e:
            action.fail(str(e))
            self.action_history.append(action)
            raise
        finally:
            if action in self.current_actions:
                self.current_actions.remove(action)
    
    def _update_metacognition(self, state: Dict[str, Any]) -> None:
        """Update metacognitive state."""
        coherence = state.get("coherence", 0.5)
        
        # Processing load based on attention and goals
        active_goals = len([g for g in self.goals if g.is_active])
        self.self_model.processing_load = (
            (len(self.attention_foci) / self.max_foci) * 0.5 +
            (active_goals / self.max_goals) * 0.3 +
            (len(self.current_actions) / 3) * 0.2
        )
        
        # Emotional valence based on goal progress and coherence
        valence = 0.0
        active_goals_list = [g for g in self.goals if g.is_active]
        for goal in active_goals_list:
            valence += goal.progress - 0.5
        if active_goals_list:
            valence /= len(active_goals_list)
        valence += coherence - 0.5
        self.self_model.emotional_valence = max(-1, min(1, valence))
        
        # Confidence based on coherence and goal success rate
        achieved_goals = len([g for g in self.goals if g.status == "achieved"])
        total_goals = len(self.goals)
        success_rate = achieved_goals / total_goals if total_goals > 0 else 0.5
        self.self_model.confidence_level = coherence * 0.5 + success_rate * 0.5
        
        # Log significant metacognitive events
        if self.self_model.processing_load > 0.8:
            self._log_metacognitive("high_load", "Processing load is high")
        if self.self_model.emotional_valence < -0.5:
            self._log_metacognitive("negative_valence", "Emotional state is negative")
    
    def _log_metacognitive(self, type_: str, description: str) -> None:
        """Log a metacognitive event."""
        self.metacognitive_log.append({
            "type": type_,
            "description": description,
            "timestamp": time.time() * 1000,
            "state": {
                "attention_capacity": self.self_model.attention_capacity,
                "processing_load": self.self_model.processing_load,
                "emotional_valence": self.self_model.emotional_valence,
                "confidence_level": self.self_model.confidence_level,
            },
        })
        
        if len(self.metacognitive_log) > 100:
            self.metacognitive_log.pop(0)
    
    def get_top_focus(self) -> Optional[AttentionFocus]:
        """Get the top attention focus."""
        if not self.attention_foci:
            return None
        return max(self.attention_foci, key=lambda f: f.intensity)
    
    def get_top_goal(self) -> Optional[Goal]:
        """Get the highest priority goal."""
        active = [g for g in self.goals if g.is_active]
        if not active:
            return None
        return max(active, key=lambda g: g.priority)
    
    def get_state(self) -> Dict[str, Any]:
        """Get current agency state."""
        return {
            "foci": [f.to_dict() for f in self.attention_foci],
            "goals": [g.to_dict() for g in self.goals],
            "active_goals": [g.to_dict() for g in self.goals if g.is_active],
            "current_actions": [a.to_dict() for a in self.current_actions],
            "metacognitive": {
                "processing_load": self.self_model.processing_load,
                "emotional_valence": self.self_model.emotional_valence,
                "confidence_level": self.self_model.confidence_level,
            },
        }
    
    def focus_on(self, target: Any, type_: str = "external", intensity: float = 0.8) -> AttentionFocus:
        """Explicitly focus attention on something."""
        focus = AttentionFocus(
            target=target,
            type=type_,
            intensity=intensity,
        )
        self.attention_foci.append(focus)
        
        # Prune if over capacity
        while len(self.attention_foci) > self.max_foci:
            self.attention_foci.sort(key=lambda f: f.intensity, reverse=True)
            self.attention_foci.pop()
        
        return focus
    
    def abandon_goal(self, goal_id: str, reason: str = "external") -> bool:
        """Abandon a goal by ID."""
        for goal in self.goals:
            if goal.id == goal_id:
                goal.abandon(reason)
                return True
        return False
    
    def clear_completed_goals(self) -> int:
        """Remove completed/abandoned goals."""
        initial = len(self.goals)
        self.goals = [g for g in self.goals if g.is_active]
        return initial - len(self.goals)