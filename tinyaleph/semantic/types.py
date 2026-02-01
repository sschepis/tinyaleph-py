"""
Formal Type System for Prime-Based Compositional Languages

Implements a typed term calculus for prime-indexed semantics:
- N(p): noun/subject term indexed by prime p
- A(p): adjective/operator term indexed by prime p
- S: sentence-level proposition

Key features:
- Ordered operator application with p < q constraint
- Triadic fusion FUSE(p, q, r) where p+q+r is prime
- Sentence composition (◦) and implication (⇒)
- Full typing judgments Γ ⊢ e : T
"""

from typing import List, Optional, Dict, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import json

from ..core.primes import is_prime, first_n_primes


class Types(Enum):
    """Type constants for the type system."""
    NOUN = 'N'
    ADJECTIVE = 'A'
    SENTENCE = 'S'


class Term:
    """Base class for all typed terms."""
    
    def __init__(self, term_type: Types):
        self.type = term_type
    
    def is_well_formed(self) -> bool:
        """Check if term is well-formed."""
        raise NotImplementedError("Must be implemented by subclass")
    
    def signature(self) -> str:
        """Get the semantic signature."""
        raise NotImplementedError("Must be implemented by subclass")
    
    def clone(self) -> 'Term':
        """Clone the term."""
        raise NotImplementedError("Must be implemented by subclass")
    
    def __str__(self) -> str:
        return self.signature()
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.signature()})"


class NounTerm(Term):
    """
    NounTerm - N(p)
    
    Represents a noun/subject indexed by prime p.
    Semantically denotes the prime itself: ⟦N(p)⟧ = p
    """
    
    def __init__(self, prime: int):
        super().__init__(Types.NOUN)
        if not is_prime(prime):
            raise TypeError(f"NounTerm requires prime number, got {prime}")
        self.prime = prime
    
    def is_well_formed(self) -> bool:
        return is_prime(self.prime)
    
    def signature(self) -> str:
        return f"N({self.prime})"
    
    def clone(self) -> 'NounTerm':
        return NounTerm(self.prime)
    
    def interpret(self) -> int:
        """Semantic interpretation: ⟦N(p)⟧ = p"""
        return self.prime
    
    def equals(self, other: 'Term') -> bool:
        """Check equality with another noun term."""
        return isinstance(other, NounTerm) and self.prime == other.prime
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'N', 'prime': self.prime}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NounTerm':
        return cls(data['prime'])


class AdjTerm(Term):
    """
    AdjTerm - A(p)
    
    Represents an adjective/operator indexed by prime p.
    Semantically denotes a partial function: ⟦A(p)⟧ = f_p : D ⇀ D
    where dom(f_p) ⊆ {q ∈ D : p < q}
    """
    
    def __init__(self, prime: int):
        super().__init__(Types.ADJECTIVE)
        if not is_prime(prime):
            raise TypeError(f"AdjTerm requires prime number, got {prime}")
        self.prime = prime
    
    def is_well_formed(self) -> bool:
        return is_prime(self.prime)
    
    def signature(self) -> str:
        return f"A({self.prime})"
    
    def clone(self) -> 'AdjTerm':
        return AdjTerm(self.prime)
    
    def can_apply_to(self, noun: 'NounTerm') -> bool:
        """Check if this adjective can apply to a noun (p < q constraint)."""
        if not isinstance(noun, NounTerm):
            raise TypeError('Can only apply to NounTerm')
        return self.prime < noun.prime
    
    def apply(self, noun: 'NounTerm') -> 'ChainTerm':
        """
        Apply this adjective to a noun term.
        Returns a ChainTerm for type safety.
        """
        if not self.can_apply_to(noun):
            raise TypeError(
                f"Application constraint violated: {self.prime} must be < {noun.prime}"
            )
        return ChainTerm([self], noun)
    
    def equals(self, other: 'Term') -> bool:
        """Check equality with another adjective term."""
        return isinstance(other, AdjTerm) and self.prime == other.prime
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'A', 'prime': self.prime}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AdjTerm':
        return cls(data['prime'])


class ChainTerm(Term):
    """
    ChainTerm - A(p₁), A(p₂), ..., A(pₖ), N(q)
    
    Represents an operator chain applied to a noun.
    Semantically: ⟦chain⟧ = f_p₁(f_p₂(...f_pₖ(q)...))
    """
    
    def __init__(self, operators: List[AdjTerm], noun: NounTerm):
        super().__init__(Types.NOUN)
        if not isinstance(operators, list):
            raise TypeError('Operators must be a list')
        if not isinstance(noun, NounTerm):
            raise TypeError('Noun must be a NounTerm')
        
        self.operators = operators
        self.noun = noun
    
    def is_well_formed(self) -> bool:
        """Check well-formedness: all operators must satisfy p < q constraint."""
        if len(self.operators) == 0:
            return self.noun.is_well_formed()
        
        # Check innermost constraint: last operator's prime < noun's prime
        last = self.operators[-1]
        if last.prime >= self.noun.prime:
            return False
        
        return True
    
    def signature(self) -> str:
        ops = ', '.join(o.signature() for o in self.operators)
        return f"{ops}, {self.noun.signature()}" if ops else self.noun.signature()
    
    def clone(self) -> 'ChainTerm':
        return ChainTerm(
            [o.clone() for o in self.operators],
            self.noun.clone()
        )
    
    def prepend(self, operator: AdjTerm) -> 'ChainTerm':
        """Prepend an operator to this chain."""
        return ChainTerm([operator] + self.operators, self.noun)
    
    @property
    def length(self) -> int:
        """Get the chain length (number of operators)."""
        return len(self.operators)
    
    def get_all_primes(self) -> List[int]:
        """Get all primes in the chain (operators + noun)."""
        return [o.prime for o in self.operators] + [self.noun.prime]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'chain',
            'operators': [o.to_dict() for o in self.operators],
            'noun': self.noun.to_dict()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ChainTerm':
        return cls(
            [AdjTerm.from_dict(o) for o in data['operators']],
            NounTerm.from_dict(data['noun'])
        )


class FusionTerm(Term):
    """
    FusionTerm - FUSE(p, q, r)
    
    Represents triadic prime fusion.
    Well-formed when: p, q, r are distinct odd primes and p+q+r is prime.
    Semantically: ⟦FUSE(p,q,r)⟧ = p + q + r
    """
    
    def __init__(self, p: int, q: int, r: int):
        super().__init__(Types.NOUN)
        self.p = p
        self.q = q
        self.r = r
    
    def is_well_formed(self) -> bool:
        """
        Check well-formedness:
        1. p, q, r are distinct
        2. p, q, r are odd primes (> 2)
        3. p + q + r is prime
        """
        # Check distinctness
        if self.p == self.q or self.q == self.r or self.p == self.r:
            return False
        
        # Check all are odd primes (> 2)
        if self.p == 2 or self.q == 2 or self.r == 2:
            return False
        
        if not is_prime(self.p) or not is_prime(self.q) or not is_prime(self.r):
            return False
        
        # Check sum is prime
        return is_prime(self.p + self.q + self.r)
    
    def get_fused_prime(self) -> int:
        """Get the fused prime value."""
        if not self.is_well_formed():
            raise ValueError('Cannot get fused prime from ill-formed fusion')
        return self.p + self.q + self.r
    
    def signature(self) -> str:
        return f"FUSE({self.p}, {self.q}, {self.r})"
    
    def clone(self) -> 'FusionTerm':
        return FusionTerm(self.p, self.q, self.r)
    
    def to_noun_term(self) -> NounTerm:
        """Convert to equivalent NounTerm (after reduction)."""
        return NounTerm(self.get_fused_prime())
    
    def canonical(self) -> 'FusionTerm':
        """Get canonical form (sorted primes)."""
        sorted_primes = sorted([self.p, self.q, self.r])
        return FusionTerm(sorted_primes[0], sorted_primes[1], sorted_primes[2])
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'FUSE', 'p': self.p, 'q': self.q, 'r': self.r}
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FusionTerm':
        return cls(data['p'], data['q'], data['r'])
    
    @staticmethod
    def find_triads(target: int, limit: int = 100) -> List['FusionTerm']:
        """
        Find valid fusion triads for a target prime.
        
        Args:
            target: Target prime (sum of three primes)
            limit: Maximum prime to consider
            
        Returns:
            List of valid FusionTerms
        """
        if not is_prime(target):
            return []
        
        triads = []
        primes = [p for p in first_n_primes(min(limit, 100)) 
                  if p > 2 and p < target]
        
        for i, p in enumerate(primes):
            for j in range(i + 1, len(primes)):
                q = primes[j]
                r = target - p - q
                if r > q and is_prime(r) and r != 2:
                    triads.append(FusionTerm(p, q, r))
        
        return triads


class SentenceTerm(Term):
    """
    SentenceTerm - S
    Base class for sentence-level expressions.
    """
    
    def __init__(self):
        super().__init__(Types.SENTENCE)
    
    def get_discourse_state(self) -> List[int]:
        """Get discourse state (sequence of primes)."""
        raise NotImplementedError("Must be implemented by subclass")


class NounSentence(SentenceTerm):
    """
    NounSentence - Sentence from noun term.
    
    A noun-denoting expression treated as a one-token discourse state.
    ⟦e : N⟧_S = [⟦e⟧] ∈ D*
    """
    
    def __init__(self, noun_expr):
        super().__init__()
        if not isinstance(noun_expr, (NounTerm, ChainTerm, FusionTerm)):
            raise TypeError('NounSentence requires noun-typed expression')
        self.expr = noun_expr
    
    def is_well_formed(self) -> bool:
        return self.expr.is_well_formed()
    
    def signature(self) -> str:
        return f"S({self.expr.signature()})"
    
    def clone(self) -> 'NounSentence':
        return NounSentence(self.expr.clone())
    
    def get_discourse_state(self) -> List[int]:
        """Get discourse state (sequence of primes)."""
        if isinstance(self.expr, NounTerm):
            return [self.expr.prime]
        elif isinstance(self.expr, ChainTerm):
            return self.expr.get_all_primes()
        elif isinstance(self.expr, FusionTerm):
            return [self.expr.get_fused_prime()]
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        return {'type': 'NounSentence', 'expr': self.expr.to_dict()}


class SeqSentence(SentenceTerm):
    """
    SeqSentence - s₁ ◦ s₂
    
    Sequential composition of sentences.
    Semantically: ⟦s₁ ◦ s₂⟧ = ⟦s₁⟧ · ⟦s₂⟧ (concatenation in D*)
    """
    
    def __init__(self, left: SentenceTerm, right: SentenceTerm):
        super().__init__()
        if not isinstance(left, SentenceTerm) or not isinstance(right, SentenceTerm):
            raise TypeError('SeqSentence requires two SentenceTerms')
        
        self.left = left
        self.right = right
    
    def is_well_formed(self) -> bool:
        return self.left.is_well_formed() and self.right.is_well_formed()
    
    def signature(self) -> str:
        return f"({self.left.signature()} ◦ {self.right.signature()})"
    
    def clone(self) -> 'SeqSentence':
        return SeqSentence(self.left.clone(), self.right.clone())
    
    def get_discourse_state(self) -> List[int]:
        """Get combined discourse state (concatenation)."""
        return self.left.get_discourse_state() + self.right.get_discourse_state()
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'SeqSentence',
            'left': self.left.to_dict(),
            'right': self.right.to_dict()
        }


class ImplSentence(SentenceTerm):
    """
    ImplSentence - s₁ ⇒ s₂
    
    Implication/entailment between sentences.
    Semantically: M ⊨ (s₁ ⇒ s₂) iff ⟦s₁⟧ ⪯ ⟦s₂⟧ (prefix entailment)
    """
    
    def __init__(self, antecedent: SentenceTerm, consequent: SentenceTerm):
        super().__init__()
        if not isinstance(antecedent, SentenceTerm) or not isinstance(consequent, SentenceTerm):
            raise TypeError('ImplSentence requires two SentenceTerms')
        
        self.antecedent = antecedent
        self.consequent = consequent
    
    def is_well_formed(self) -> bool:
        return self.antecedent.is_well_formed() and self.consequent.is_well_formed()
    
    def signature(self) -> str:
        return f"({self.antecedent.signature()} ⇒ {self.consequent.signature()})"
    
    def clone(self) -> 'ImplSentence':
        return ImplSentence(self.antecedent.clone(), self.consequent.clone())
    
    def get_discourse_state(self) -> List[int]:
        """For implications, return consequent discourse state."""
        return self.consequent.get_discourse_state()
    
    def holds(self) -> bool:
        """
        Check if implication holds (prefix entailment).
        """
        ante = self.antecedent.get_discourse_state()
        cons = self.consequent.get_discourse_state()
        
        # Prefix entailment: antecedent is prefix of consequent
        if len(ante) > len(cons):
            return False
        
        for i in range(len(ante)):
            if ante[i] != cons[i]:
                return False
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'type': 'ImplSentence',
            'antecedent': self.antecedent.to_dict(),
            'consequent': self.consequent.to_dict()
        }


@dataclass
class TypingContext:
    """
    TypingContext - Γ
    A context for typing judgments.
    """
    bindings: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    
    def bind(self, name: str, term_type: Types, term: Optional[Term] = None) -> 'TypingContext':
        """Bind a variable name to a type."""
        self.bindings[name] = {'type': term_type, 'term': term}
        return self
    
    def get_type(self, name: str) -> Optional[Types]:
        """Get type of a variable."""
        binding = self.bindings.get(name)
        return binding['type'] if binding else None
    
    def get_term(self, name: str) -> Optional[Term]:
        """Get term for a variable."""
        binding = self.bindings.get(name)
        return binding['term'] if binding else None
    
    def has(self, name: str) -> bool:
        """Check if variable is bound."""
        return name in self.bindings
    
    def clone(self) -> 'TypingContext':
        """Clone the context."""
        ctx = TypingContext()
        ctx.bindings = {k: dict(v) for k, v in self.bindings.items()}
        return ctx
    
    def __str__(self) -> str:
        entries = [f"{name}: {b['type'].value}" for name, b in self.bindings.items()]
        return f"Γ = {{{', '.join(entries)}}}"


@dataclass
class TypingJudgment:
    """
    TypingJudgment - Γ ⊢ e : T
    Represents a typing judgment.
    """
    context: TypingContext
    term: Term
    term_type: Types
    
    def is_valid(self) -> bool:
        """Check if this judgment is valid."""
        if not self.term.is_well_formed():
            return False
        return self.term.type == self.term_type
    
    def __str__(self) -> str:
        return f"{self.context} ⊢ {self.term} : {self.term_type.value}"


class TypeChecker:
    """
    TypeChecker - Implements typing rules from the formal system.
    """
    
    def __init__(self):
        self.context = TypingContext()
    
    def infer_type(self, term: Term) -> Optional[Types]:
        """
        Infer type of a term.
        
        Args:
            term: The term to type
            
        Returns:
            The inferred type, or None if ill-typed
        """
        if isinstance(term, NounTerm):
            return Types.NOUN if term.is_well_formed() else None
        
        if isinstance(term, AdjTerm):
            return Types.ADJECTIVE if term.is_well_formed() else None
        
        if isinstance(term, ChainTerm):
            return Types.NOUN if term.is_well_formed() else None
        
        if isinstance(term, FusionTerm):
            return Types.NOUN if term.is_well_formed() else None
        
        if isinstance(term, SentenceTerm):
            return Types.SENTENCE if term.is_well_formed() else None
        
        return None
    
    def check_type(self, term: Term, expected_type: Types) -> bool:
        """
        Check if a term has a specific type.
        
        Args:
            term: The term to check
            expected_type: The expected type
            
        Returns:
            True if term has expected type
        """
        inferred = self.infer_type(term)
        return inferred == expected_type
    
    def derive(self, term: Term) -> Optional[TypingJudgment]:
        """
        Derive typing judgment.
        
        Args:
            term: The term to type
            
        Returns:
            The judgment, or None if ill-typed
        """
        term_type = self.infer_type(term)
        if term_type is None:
            return None
        return TypingJudgment(self.context.clone(), term, term_type)
    
    def check_application(self, adj: AdjTerm, noun: NounTerm) -> Dict[str, Any]:
        """
        Check application well-formedness.
        
        Args:
            adj: Adjective term
            noun: Noun term
            
        Returns:
            Dict with 'valid' and optional 'reason'
        """
        if not self.check_type(adj, Types.ADJECTIVE):
            return {'valid': False, 'reason': 'Adjective ill-typed'}
        
        if not self.check_type(noun, Types.NOUN):
            return {'valid': False, 'reason': 'Noun ill-typed'}
        
        if not adj.can_apply_to(noun):
            return {
                'valid': False,
                'reason': f"Ordering constraint violated: {adj.prime} ≮ {noun.prime}"
            }
        
        return {'valid': True}
    
    def check_fusion(self, fusion: FusionTerm) -> Dict[str, Any]:
        """
        Check fusion well-formedness.
        
        Args:
            fusion: Fusion term
            
        Returns:
            Dict with 'valid', optional 'reason', and 'result'
        """
        if not fusion.is_well_formed():
            return {
                'valid': False,
                'reason': f"Fusion ill-formed: primes {fusion.p}, {fusion.q}, {fusion.r} don't satisfy constraints"
            }
        return {'valid': True, 'result': fusion.get_fused_prime()}


# =============================================================================
# TERM BUILDERS
# =============================================================================

def N(prime: int) -> NounTerm:
    """Build a noun term from prime."""
    return NounTerm(prime)


def A(prime: int) -> AdjTerm:
    """Build an adjective term from prime."""
    return AdjTerm(prime)


def FUSE(p: int, q: int, r: int) -> FusionTerm:
    """Build a fusion term."""
    return FusionTerm(p, q, r)


def CHAIN(operators, noun) -> ChainTerm:
    """Build a chain term."""
    ops = [A(p) if isinstance(p, int) else p for p in operators]
    n = N(noun) if isinstance(noun, int) else noun
    return ChainTerm(ops, n)


def SENTENCE(expr) -> NounSentence:
    """Build a sentence from noun expression."""
    if isinstance(expr, int):
        expr = N(expr)
    return NounSentence(expr)


def SEQ(s1, s2) -> SeqSentence:
    """Sequential composition of sentences."""
    left = s1 if isinstance(s1, SentenceTerm) else SENTENCE(s1)
    right = s2 if isinstance(s2, SentenceTerm) else SENTENCE(s2)
    return SeqSentence(left, right)


def IMPL(s1, s2) -> ImplSentence:
    """Implication of sentences."""
    ante = s1 if isinstance(s1, SentenceTerm) else SENTENCE(s1)
    cons = s2 if isinstance(s2, SentenceTerm) else SENTENCE(s2)
    return ImplSentence(ante, cons)