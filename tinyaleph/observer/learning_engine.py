from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
import json
import os
import time
import math
import random

from tinyaleph.observer.semantic_prime_mapper import SemanticPrimeMapper


@dataclass
class LearningGoal:
    id: str
    type: str  # define_symbol | find_relationship | expand_concept
    description: str
    priority: float
    status: str = "pending"  # pending | in_progress | completed | failed
    target_prime: Optional[int] = None
    concepts: Optional[List[str]] = None
    created_at: float = field(default_factory=time.time)
    completed_at: Optional[float] = None
    result: Any = None


@dataclass
class LearningSession:
    id: str
    started_at: float
    goals_completed: int = 0
    symbols_learned: int = 0
    relationships_discovered: int = 0
    is_active: bool = True


@dataclass
class LearningState:
    learning_queue: List[LearningGoal] = field(default_factory=list)
    learned_symbols: Dict[int, Dict[str, Any]] = field(default_factory=dict)
    learned_relationships: List[Dict[str, Any]] = field(default_factory=list)
    current_session: Optional[LearningSession] = None
    total_goals_completed: int = 0
    is_learning: bool = False
    last_learning_action: Optional[str] = None


class LearningEngine:
    """
    Python port of the TS LearningEngine.

    Responsibilities:
      - Manage a queue of learning goals (define_symbol, find_relationship, expand_concept).
      - Call a pluggable chaperone_fn(goal_context) -> result to execute goals.
      - Integrate results into SemanticPrimeMapper and track stats.
      - Persist state to a JSON file (storage_path).
    """

    def __init__(
        self,
        mapper: SemanticPrimeMapper,
        chaperone_fn: Callable[[Dict[str, Any]], Any],
        storage_path: str = "runs/sentient/learning_state.json",
    ):
        self.mapper = mapper
        self.chaperone_fn = chaperone_fn
        self.storage_path = storage_path
        self.state = LearningState()
        self._load()

    # ---------- persistence ----------
    def _load(self) -> None:
        if not os.path.exists(self.storage_path):
            return
        try:
            with open(self.storage_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self._from_dict(data)
        except Exception:
            pass

    def _save(self) -> None:
        os.makedirs(os.path.dirname(self.storage_path), exist_ok=True)
        with open(self.storage_path, "w", encoding="utf-8") as f:
            json.dump(self._to_dict(), f, ensure_ascii=False, indent=2)

    def _to_dict(self) -> Dict[str, Any]:
        return {
            "learning_queue": [g.__dict__ for g in self.state.learning_queue],
            "learned_symbols": self.state.learned_symbols,
            "learned_relationships": self.state.learned_relationships,
            "current_session": self.state.current_session.__dict__ if self.state.current_session else None,
            "total_goals_completed": self.state.total_goals_completed,
            "last_learning_action": self.state.last_learning_action,
        }

    def _from_dict(self, data: Dict[str, Any]) -> None:
        q = []
        for g in data.get("learning_queue", []):
            q.append(LearningGoal(**g))
        self.state.learning_queue = q
        self.state.learned_symbols = data.get("learned_symbols", {})
        self.state.learned_relationships = data.get("learned_relationships", [])
        cs = data.get("current_session")
        if cs:
            self.state.current_session = LearningSession(**cs)
        self.state.total_goals_completed = data.get("total_goals_completed", 0)
        self.state.last_learning_action = data.get("last_learning_action")

    # ---------- public API ----------
    def start_session(self) -> None:
        if self.state.current_session and self.state.current_session.is_active:
            return
        self.state.current_session = LearningSession(id=f"session_{int(time.time())}", started_at=time.time())
        self._identify_opportunities()
        self._save()

    def stop_session(self) -> None:
        if self.state.current_session:
            self.state.current_session.is_active = False
        self._save()

    def add_goal(self, goal_type: str, description: str, priority: float = 0.7, target_prime: Optional[int] = None, concepts: Optional[List[str]] = None) -> None:
        new_goal = LearningGoal(
            id=f"goal_{int(time.time())}_{random.randint(0,9999)}",
            type=goal_type,
            description=description,
            priority=priority,
            target_prime=target_prime,
            concepts=concepts,
        )
        self.state.learning_queue.append(new_goal)
        self.state.learning_queue.sort(key=lambda g: g.priority, reverse=True)
        self._save()

    def process_next_goal(self) -> None:
        if self.state.is_learning:
            return
        next_goal = next((g for g in self.state.learning_queue if g.status == "pending"), None)
        if not next_goal:
            self._identify_opportunities()
            return
        self.state.is_learning = True
        next_goal.status = "in_progress"
        self.state.last_learning_action = f"Learning: {next_goal.description}"
        self._save()
        try:
            result = self._execute_goal(next_goal)
            next_goal.status = "completed"
            next_goal.completed_at = time.time()
            next_goal.result = result
            self.state.total_goals_completed += 1
            if self.state.current_session:
                self.state.current_session.goals_completed += 1
            self._integrate_results(next_goal.type, result, next_goal)
            self.state.last_learning_action = f"Learned: {self._summarize(next_goal.type, result)}"
        except Exception as e:
            next_goal.status = "failed"
            self.state.last_learning_action = f"Failed: {next_goal.description} ({e})"
        self.state.is_learning = False
        cutoff = time.time() - 5 * 60
        self.state.learning_queue = [g for g in self.state.learning_queue if g.status in ("pending", "in_progress") or (g.completed_at and g.completed_at > cutoff)]
        self._save()

    def get_stats(self) -> Dict[str, Any]:
        return {
            "symbols_learned": len(self.state.learned_symbols),
            "relationships_discovered": len(self.state.learned_relationships),
            "goals_completed": self.state.total_goals_completed,
            "queue_length": len([g for g in self.state.learning_queue if g.status == "pending"]),
            "is_active": bool(self.state.current_session and self.state.current_session.is_active),
            "last_action": self.state.last_learning_action,
        }

    # ---------- internals ----------
    def _identify_opportunities(self) -> None:
        uncatalogued = self.mapper.get_uncatalogued_primes()
        for prime in uncatalogued[:3]:
            already = any(g.target_prime == prime for g in self.state.learning_queue)
            if not already:
                self.add_goal("define_symbol", f"Learn meaning for prime {prime}", priority=0.8, target_prime=prime)
        self._save()

    def _execute_goal(self, goal: LearningGoal) -> Any:
        context = {
            "type": goal.type,
            "target_prime": goal.target_prime,
            "concepts": goal.concepts,
            "description": goal.description,
            "known_symbols": list(self.state.learned_symbols.values())[:20],
        }
        return self.chaperone_fn(context)

    def _integrate_results(self, goal_type: str, result: Any, goal: LearningGoal) -> None:
        if goal_type == "define_symbol" and result:
            prime = result.get("prime") or goal.target_prime
            meaning = result.get("meaning") or result.get("definition")
            if prime and meaning:
                if prime in self.state.learned_symbols:
                    return
                self.state.learned_symbols[prime] = {
                    "prime": prime,
                    "meaning": meaning,
                    "category": result.get("category"),
                    "confidence": result.get("confidence", 0.7),
                    "learned_at": time.time(),
                }
                self.mapper.add_learned_meaning(prime, meaning, result.get("confidence", 0.7), result.get("category"))
                if self.state.current_session:
                    self.state.current_session.symbols_learned += 1

        if goal_type == "find_relationship" and result:
            rels = result.get("relationships") or []
            for rel in rels:
                pa = rel.get("primeA")
                pb = rel.get("primeB")
                if pa and pb:
                    rec = {
                        "primeA": pa,
                        "primeB": pb,
                        "relationship_type": rel.get("relationshipType", "resonates_with"),
                        "strength": rel.get("strength", 0.5),
                        "explanation": rel.get("explanation", ""),
                        "learned_at": time.time(),
                    }
                    exists = any((r["primeA"], r["primeB"]) == (pa, pb) or (r["primeA"], r["primeB"]) == (pb, pa) for r in self.state.learned_relationships)
                    if not exists:
                        self.state.learned_relationships.append(rec)
                        if self.state.current_session:
                            self.state.current_session.relationships_discovered += 1

        if goal_type == "expand_concept" and result:
            suggestions = result.get("suggestions") or []
            uncatalogued = self.mapper.get_uncatalogued_primes()
            for sug in suggestions[:3]:
                target = None
                for p in uncatalogued:
                    if not any(g.target_prime == p for g in self.state.learning_queue):
                        target = p
                        break
                if target:
                    self.add_goal("define_symbol", f"Define '{sug.get('concept', 'concept')}' for prime {target}", priority=0.7, target_prime=target, concepts=[sug.get("concept")])

    def _summarize(self, goal_type: str, result: Any) -> str:
        if goal_type == "define_symbol":
            if not result:
                return "no definition"
            return f"{result.get('prime')} → {result.get('meaning')}"
        if goal_type == "find_relationship":
            rels = result.get("relationships") or []
            return f"{len(rels)} relationships"
        if goal_type == "expand_concept":
            sugs = result.get("suggestions") or []
            return f"{len(sugs)} suggestions"
        return "done"

