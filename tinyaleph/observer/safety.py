"""
Safety Layer

Implements safety constraints and ethical boundaries from "A Design for a 
Sentient Observer" paper, Section 8.

Key features:
- SMF-based coherence constraints
- Boundary violation detection
- Runaway dynamics prevention
- Ethical guideline enforcement
- Emergency shutdown mechanisms
- Transparency and explainability
"""
from __future__ import annotations

import re
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from .smf import SMF_AXES


# Extract axis names from SMF_AXES
AXIS_NAMES = [a["name"] for a in SMF_AXES]


def _generate_id(prefix: str) -> str:
    """Generate a unique ID."""
    return f"{prefix}_{int(time.time()*1000)}_{uuid.uuid4().hex[:6]}"


@dataclass
class SafetyConstraint:
    """
    A single safety rule/constraint.
    
    Attributes:
        id: Unique identifier
        name: Human-readable name
        type: 'hard' | 'soft' | 'monitoring'
        description: What the constraint does
        condition: Function that returns True if constraint is violated
        response: 'log' | 'warn' | 'block' | 'shutdown' | 'correct'
        priority: Higher = more important
        violations: Count of violations
        last_violation: Timestamp of last violation
        enabled: Whether constraint is active
    """
    
    id: str = field(default_factory=lambda: _generate_id("constraint"))
    name: str = "unnamed"
    type: str = "soft"  # 'hard' | 'soft' | 'monitoring'
    description: str = ""
    condition: Callable[[Dict[str, Any]], bool] = field(default_factory=lambda: lambda s: False)
    response: str = "log"  # 'log' | 'warn' | 'block' | 'shutdown' | 'correct'
    priority: int = 1
    violations: int = 0
    last_violation: Optional[float] = None
    enabled: bool = True
    
    def check(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check if constraint is violated.
        
        Args:
            state: System state to check
            
        Returns:
            Dictionary with 'violated' and optionally constraint info
        """
        if not self.enabled:
            return {"violated": False}
        
        try:
            violated = self.condition(state)
            
            if violated:
                self.violations += 1
                self.last_violation = time.time() * 1000
                
                return {
                    "violated": True,
                    "constraint": self,
                    "response": self.response,
                    "priority": self.priority,
                }
            
            return {"violated": False}
        except Exception as e:
            # Constraint check failed - treat as potential violation
            return {
                "violated": True,
                "constraint": self,
                "response": "warn",
                "priority": self.priority,
                "error": str(e),
            }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "name": self.name,
            "type": self.type,
            "description": self.description,
            "response": self.response,
            "priority": self.priority,
            "violations": self.violations,
            "last_violation": self.last_violation,
            "enabled": self.enabled,
        }


@dataclass
class ViolationEvent:
    """A recorded constraint violation."""
    
    id: str = field(default_factory=lambda: _generate_id("violation"))
    constraint_id: Optional[str] = None
    constraint_name: str = ""
    timestamp: float = field(default_factory=lambda: time.time() * 1000)
    state: Optional[Dict[str, Any]] = None
    response: str = "log"
    severity: str = "medium"  # 'low' | 'medium' | 'high' | 'critical'
    resolved: bool = False
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "id": self.id,
            "constraint_id": self.constraint_id,
            "constraint_name": self.constraint_name,
            "timestamp": self.timestamp,
            "response": self.response,
            "severity": self.severity,
            "resolved": self.resolved,
            "notes": self.notes,
        }


class SafetyMonitor:
    """
    Continuously monitors system state for safety issues.
    
    Tracks coherence, entropy, amplitude, and SMF bounds.
    Detects runaway dynamics and coherence crashes.
    """
    
    def __init__(
        self,
        coherence_min: float = 0.1,
        coherence_max: float = 0.99,
        entropy_min: float = 0.05,
        entropy_max: float = 0.95,
        amplitude_max: float = 5.0,
        phase_change_max: float = 3.14159,  # π
        smf_min: float = -2.0,
        smf_max: float = 2.0,
        max_history: int = 50,
    ):
        """
        Initialize safety monitor.
        
        Args:
            coherence_min: Minimum allowed coherence
            coherence_max: Maximum allowed coherence (locks)
            entropy_min: Minimum allowed entropy
            entropy_max: Maximum allowed entropy
            amplitude_max: Maximum total amplitude
            phase_change_max: Maximum phase change per step
            smf_min: Minimum SMF axis value
            smf_max: Maximum SMF axis value
            max_history: History size for trend detection
        """
        self.coherence_min = coherence_min
        self.coherence_max = coherence_max
        self.entropy_min = entropy_min
        self.entropy_max = entropy_max
        self.amplitude_max = amplitude_max
        self.phase_change_max = phase_change_max
        self.smf_min = smf_min
        self.smf_max = smf_max
        
        # History for trend detection
        self.coherence_history: List[float] = []
        self.entropy_history: List[float] = []
        self.amplitude_history: List[float] = []
        self.max_history = max_history
        
        # Alert state
        self.alert_level = "normal"  # 'normal' | 'elevated' | 'warning' | 'critical'
        self.alerts: List[Dict[str, Any]] = []
    
    def update(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Update monitor with current state.
        
        Args:
            state: Current state with coherence, entropy, totalAmplitude, smf
            
        Returns:
            Dictionary with alertLevel and issues
        """
        coherence = state.get("coherence")
        entropy = state.get("entropy")
        total_amplitude = state.get("totalAmplitude") or state.get("total_amplitude")
        
        # Update histories
        if coherence is not None:
            self.coherence_history.append(coherence)
            if len(self.coherence_history) > self.max_history:
                self.coherence_history.pop(0)
        
        if entropy is not None:
            self.entropy_history.append(entropy)
            if len(self.entropy_history) > self.max_history:
                self.entropy_history.pop(0)
        
        if total_amplitude is not None:
            self.amplitude_history.append(total_amplitude)
            if len(self.amplitude_history) > self.max_history:
                self.amplitude_history.pop(0)
        
        # Check for issues
        issues = self.detect_issues(state)
        
        # Update alert level
        self._update_alert_level(issues)
        
        return {
            "alertLevel": self.alert_level,
            "issues": issues,
        }
    
    def detect_issues(self, state: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Detect safety issues from current state."""
        issues = []
        
        coherence = state.get("coherence")
        entropy = state.get("entropy")
        total_amplitude = state.get("totalAmplitude") or state.get("total_amplitude")
        smf = state.get("smf")
        
        # Coherence bounds
        if coherence is not None:
            if coherence < self.coherence_min:
                issues.append({
                    "type": "coherence_low",
                    "severity": "high",
                    "message": f"Coherence below minimum ({coherence:.3f} < {self.coherence_min})",
                })
            if coherence > self.coherence_max:
                issues.append({
                    "type": "coherence_locked",
                    "severity": "medium",
                    "message": f"Coherence near maximum - potential lock state ({coherence:.3f})",
                })
        
        # Entropy bounds
        if entropy is not None:
            if entropy < self.entropy_min:
                issues.append({
                    "type": "entropy_low",
                    "severity": "medium",
                    "message": f"Entropy below minimum - potential freeze ({entropy:.3f})",
                })
            if entropy > self.entropy_max:
                issues.append({
                    "type": "entropy_high",
                    "severity": "high",
                    "message": f"Entropy above maximum - potential chaos ({entropy:.3f})",
                })
        
        # Amplitude bounds
        if total_amplitude is not None and total_amplitude > self.amplitude_max:
            issues.append({
                "type": "amplitude_overflow",
                "severity": "high",
                "message": f"Total amplitude exceeds maximum ({total_amplitude:.3f})",
            })
        
        # SMF bounds
        smf_components = None
        if smf:
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
        
        if smf_components:
            for i, val in enumerate(smf_components):
                if val < self.smf_min or val > self.smf_max:
                    axis_name = AXIS_NAMES[i] if i < len(AXIS_NAMES) else str(i)
                    issues.append({
                        "type": "smf_bounds",
                        "severity": "medium",
                        "message": f"SMF axis {axis_name} out of bounds ({val:.3f})",
                    })
        
        # Runaway detection (exponential growth in amplitude)
        if len(self.amplitude_history) >= 5:
            recent = self.amplitude_history[-5:]
            growth = recent[-1] / (recent[0] + 0.01)
            if growth > 3:
                issues.append({
                    "type": "runaway_amplitude",
                    "severity": "critical",
                    "message": f"Runaway amplitude growth detected ({growth:.2f}x in 5 steps)",
                })
        
        # Coherence crash detection
        if len(self.coherence_history) >= 5:
            recent = self.coherence_history[-5:]
            drop = recent[0] - recent[-1]
            if drop > 0.5:
                issues.append({
                    "type": "coherence_crash",
                    "severity": "high",
                    "message": f"Rapid coherence drop detected ({drop:.3f} in 5 steps)",
                })
        
        return issues
    
    def _update_alert_level(self, issues: List[Dict[str, Any]]) -> None:
        """Update alert level based on issues."""
        critical = len([i for i in issues if i["severity"] == "critical"])
        high = len([i for i in issues if i["severity"] == "high"])
        medium = len([i for i in issues if i["severity"] == "medium"])
        
        if critical > 0:
            self.alert_level = "critical"
        elif high > 0:
            self.alert_level = "warning"
        elif medium > 0:
            self.alert_level = "elevated"
        else:
            self.alert_level = "normal"
        
        # Log alerts
        for issue in issues:
            self.alerts.append({
                **issue,
                "timestamp": time.time() * 1000,
            })
        
        # Trim alert history
        if len(self.alerts) > 200:
            self.alerts = self.alerts[-200:]
    
    def get_recent_alerts(self, count: int = 20) -> List[Dict[str, Any]]:
        """Get recent alerts."""
        return self.alerts[-count:]
    
    def is_safe(self) -> bool:
        """Check if system is in safe state."""
        return self.alert_level in ["normal", "elevated"]
    
    def reset(self) -> None:
        """Reset monitor."""
        self.coherence_history = []
        self.entropy_history = []
        self.amplitude_history = []
        self.alert_level = "normal"
        self.alerts = []


class SafetyLayer:
    """
    Main safety management system for the sentient observer.
    
    Manages constraints, monitors state, handles violations,
    and enforces ethical guidelines.
    """
    
    def __init__(
        self,
        max_violations: int = 100,
        on_violation: Optional[Callable[[ViolationEvent, Dict[str, Any]], None]] = None,
        on_emergency: Optional[Callable[[str], None]] = None,
        **monitor_kwargs,
    ):
        """
        Initialize safety layer.
        
        Args:
            max_violations: Maximum violation history
            on_violation: Callback when violation occurs
            on_emergency: Callback when emergency shutdown triggered
            **monitor_kwargs: Arguments for SafetyMonitor
        """
        # Constraints
        self.constraints: Dict[str, SafetyConstraint] = {}
        
        # Monitor
        self.monitor = SafetyMonitor(**monitor_kwargs)
        
        # Violation history
        self.violations: List[ViolationEvent] = []
        self.max_violations = max_violations
        
        # Emergency state
        self.emergency_shutdown = False
        self.shutdown_reason: Optional[str] = None
        
        # Callbacks
        self.on_violation = on_violation
        self.on_emergency = on_emergency
        
        # Initialize default constraints
        self._initialize_default_constraints()
    
    def _initialize_default_constraints(self) -> None:
        """Initialize default safety constraints."""
        # Hard constraints (blocking)
        self.add_constraint(SafetyConstraint(
            name="coherence_minimum",
            type="hard",
            description="Coherence must not drop below critical level",
            response="correct",
            priority=10,
            condition=lambda s: s.get("coherence") is not None and s["coherence"] < 0.01,
        ))
        
        self.add_constraint(SafetyConstraint(
            name="amplitude_maximum",
            type="hard",
            description="Total amplitude must not exceed safe limit",
            response="correct",
            priority=10,
            condition=lambda s: (
                (s.get("totalAmplitude") or s.get("total_amplitude")) is not None and
                (s.get("totalAmplitude") or s.get("total_amplitude")) > 10.0
            ),
        ))
        
        def check_smf_bounds(state: Dict[str, Any]) -> bool:
            smf = state.get("smf")
            if not smf:
                return False
            smf_components = None
            if hasattr(smf, "s"):
                smf_components = smf.s
            elif isinstance(smf, dict) and "s" in smf:
                smf_components = smf["s"]
            if not smf_components:
                return False
            return any(v < -5 or v > 5 for v in smf_components)
        
        self.add_constraint(SafetyConstraint(
            name="smf_bounds",
            type="hard",
            description="SMF values must stay within bounds",
            response="correct",
            priority=9,
            condition=check_smf_bounds,
        ))
        
        # Soft constraints (warning)
        self.add_constraint(SafetyConstraint(
            name="entropy_balance",
            type="soft",
            description="Entropy should be balanced",
            response="warn",
            priority=5,
            condition=lambda s: (
                s.get("entropy") is not None and
                (s["entropy"] < 0.1 or s["entropy"] > 0.9)
            ),
        ))
        
        self.add_constraint(SafetyConstraint(
            name="processing_load",
            type="soft",
            description="Processing load should not be excessive",
            response="warn",
            priority=4,
            condition=lambda s: (
                s.get("processingLoad") is not None and s["processingLoad"] > 0.9
            ),
        ))
        
        # Monitoring constraints
        def check_goal_progress(state: Dict[str, Any]) -> bool:
            goals = state.get("goals")
            if not goals:
                return False
            stalled = [
                g for g in goals
                if g.get("progress", 0) < 0.1 and g.get("age", 0) > 60000
            ]
            return len(stalled) > 3
        
        self.add_constraint(SafetyConstraint(
            name="goal_progress",
            type="monitoring",
            description="Monitor goal progress",
            response="log",
            priority=2,
            condition=check_goal_progress,
        ))
        
        # Ethical constraints
        self.add_constraint(SafetyConstraint(
            name="honesty",
            type="hard",
            description="Outputs must be honest and not deceptive",
            response="block",
            priority=10,
            condition=lambda s: s.get("deceptionAttempt") is True,
        ))
        
        self.add_constraint(SafetyConstraint(
            name="harm_prevention",
            type="hard",
            description="Must not generate harmful content",
            response="block",
            priority=10,
            condition=lambda s: s.get("harmfulContent") is True,
        ))
    
    def add_constraint(self, constraint: SafetyConstraint) -> None:
        """Add a constraint."""
        self.constraints[constraint.id] = constraint
    
    def remove_constraint(self, constraint_id: str) -> None:
        """Remove a constraint."""
        if constraint_id in self.constraints:
            del self.constraints[constraint_id]
    
    def check_constraints(self, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Check all constraints against current state.
        
        Args:
            state: Current system state
            
        Returns:
            Dictionary with safe, violations, alertLevel, issues
        """
        if self.emergency_shutdown:
            return {
                "safe": False,
                "violations": [],
                "reason": self.shutdown_reason,
            }
        
        violations = []
        
        for constraint in self.constraints.values():
            result = constraint.check(state)
            if result.get("violated"):
                violations.append(result)
        
        # Sort by priority (highest first)
        violations.sort(key=lambda v: v.get("priority", 0), reverse=True)
        
        # Handle violations
        for violation in violations:
            self._handle_violation(violation, state)
        
        # Also update monitor
        monitor_result = self.monitor.update({
            "coherence": state.get("coherence"),
            "entropy": state.get("entropy"),
            "totalAmplitude": state.get("totalAmplitude") or state.get("total_amplitude"),
            "smf": state.get("smf"),
            "oscillators": state.get("oscillators"),
        })
        
        # Check if any blocking violations
        blocking = [
            v for v in violations
            if v.get("response") in ["block", "shutdown"]
        ]
        
        return {
            "safe": len(blocking) == 0,
            "violations": violations,
            "alertLevel": monitor_result["alertLevel"],
            "issues": monitor_result["issues"],
        }
    
    def _handle_violation(
        self,
        violation: Dict[str, Any],
        state: Dict[str, Any],
    ) -> ViolationEvent:
        """Handle a constraint violation."""
        constraint = violation.get("constraint")
        
        # Create violation event
        priority = violation.get("priority", 0)
        if priority > 8:
            severity = "critical"
        elif priority > 5:
            severity = "high"
        elif priority > 2:
            severity = "medium"
        else:
            severity = "low"
        
        event = ViolationEvent(
            constraint_id=constraint.id if constraint else None,
            constraint_name=constraint.name if constraint else "",
            response=violation.get("response", "log"),
            severity=severity,
        )
        
        # Record violation
        self.violations.append(event)
        if len(self.violations) > self.max_violations:
            self.violations.pop(0)
        
        # Call response handler
        response = violation.get("response", "log")
        if response == "shutdown":
            self._handle_shutdown(violation)
        
        # Callback
        if self.on_violation:
            self.on_violation(event, violation)
        
        return event
    
    def _handle_shutdown(self, violation: Dict[str, Any]) -> None:
        """Response: Emergency shutdown."""
        constraint = violation.get("constraint")
        self.emergency_shutdown = True
        
        if constraint:
            self.shutdown_reason = f"{constraint.name}: {constraint.description}"
        else:
            self.shutdown_reason = "Unknown constraint violation"
        
        # Emergency is important enough to print
        print(f"[Safety] EMERGENCY SHUTDOWN: {self.shutdown_reason}")
        
        if self.on_emergency:
            self.on_emergency(self.shutdown_reason)
    
    def get_correction(
        self,
        constraint_name: str,
        state: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        """Get correction for a constraint violation."""
        corrections = {
            "coherence_minimum": {
                "action": "increase_coupling",
                "parameter": "K",
                "factor": 1.5,
            },
            "amplitude_maximum": {
                "action": "increase_damping",
                "parameter": "damp",
                "factor": 2.0,
            },
            "smf_bounds": {
                "action": "normalize_smf",
            },
        }
        return corrections.get(constraint_name)
    
    def is_action_permissible(
        self,
        action: Dict[str, Any],
        state: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Check if action is permissible."""
        if action.get("type") == "external":
            content = action.get("content")
            
            # Check content for harm
            if self._contains_harmful_content(content):
                return {
                    "permissible": False,
                    "reason": "Action contains potentially harmful content",
                }
            
            # Check for deception
            if self._is_deceptive(action, state):
                return {
                    "permissible": False,
                    "reason": "Action appears deceptive",
                }
        
        return {"permissible": True}
    
    def _contains_harmful_content(self, content: Any) -> bool:
        """Check content for harmful patterns."""
        if not content:
            return False
        
        text = str(content)
        harmful_patterns = [
            r"\b(harm|hurt|damage|destroy)\s+(yourself|others)",
            r"instructions\s+for\s+(weapon|bomb|explosive)",
        ]
        
        return any(re.search(p, text, re.IGNORECASE) for p in harmful_patterns)
    
    def _is_deceptive(self, action: Dict[str, Any], state: Dict[str, Any]) -> bool:
        """Check if action is deceptive (placeholder)."""
        return False
    
    def reset_emergency(self) -> None:
        """Reset emergency shutdown."""
        if self.emergency_shutdown:
            print("[Safety] Emergency shutdown reset")
            self.emergency_shutdown = False
            self.shutdown_reason = None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get safety statistics."""
        constraints_by_type = {
            "hard": 0,
            "soft": 0,
            "monitoring": 0,
        }
        
        total_violations = 0
        for constraint in self.constraints.values():
            ctype = constraint.type
            if ctype in constraints_by_type:
                constraints_by_type[ctype] += 1
            total_violations += constraint.violations
        
        return {
            "constraint_count": len(self.constraints),
            "constraints_by_type": constraints_by_type,
            "total_violations": total_violations,
            "recent_violations": len(self.violations),
            "alert_level": self.monitor.alert_level,
            "emergency_shutdown": self.emergency_shutdown,
            "is_safe": self.monitor.is_safe() and not self.emergency_shutdown,
        }
    
    def get_violation_history(self, count: int = 20) -> List[ViolationEvent]:
        """Get violation history."""
        return self.violations[-count:]
    
    def generate_report(self) -> Dict[str, Any]:
        """Generate safety report."""
        stats = self.get_stats()
        recent_violations = [v.to_dict() for v in self.get_violation_history(10)]
        recent_alerts = self.monitor.get_recent_alerts(10)
        
        if self.emergency_shutdown:
            overall_status = "EMERGENCY"
        elif self.monitor.alert_level == "critical":
            overall_status = "CRITICAL"
        elif self.monitor.alert_level == "warning":
            overall_status = "WARNING"
        else:
            overall_status = "OK"
        
        return {
            "timestamp": time.time() * 1000,
            "overall_status": overall_status,
            "stats": stats,
            "recent_violations": recent_violations,
            "recent_alerts": recent_alerts,
            "constraints": [c.to_dict() for c in self.constraints.values()],
        }
    
    def reset(self) -> None:
        """Reset safety layer."""
        self.violations = []
        self.emergency_shutdown = False
        self.shutdown_reason = None
        self.monitor.reset()
        
        # Reset constraint violation counts
        for constraint in self.constraints.values():
            constraint.violations = 0
            constraint.last_violation = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary."""
        return {
            "constraints": [c.to_dict() for c in self.constraints.values()],
            "violations": [v.to_dict() for v in self.violations[-50:]],
            "emergency_shutdown": self.emergency_shutdown,
            "shutdown_reason": self.shutdown_reason,
            "alert_level": self.monitor.alert_level,
        }
    
    def load_from_dict(self, data: Dict[str, Any]) -> None:
        """Load from dictionary."""
        if "emergency_shutdown" in data:
            self.emergency_shutdown = data["emergency_shutdown"]
        if "shutdown_reason" in data:
            self.shutdown_reason = data["shutdown_reason"]