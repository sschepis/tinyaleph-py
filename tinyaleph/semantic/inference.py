"""
Inference Module for TinyAleph.

Implements semantic inference and compound building:
- CompoundBuilder: Build complex semantic structures from primitives
- SemanticInference: Make inferences from semantic knowledge
- EntityExtractor: Extract entities and relations from text

Mathematical Foundation:
    Semantic inference uses prime-based composition where:
    - Atomic concepts are primes
    - Compound concepts are products
    - Relations are prime-indexed operations
    - Inference follows from factorization
"""

from typing import List, Dict, Tuple, Optional, Any, Callable, Set, Union
from dataclasses import dataclass, field
from enum import Enum
import math
from functools import lru_cache
import random
import re


# =============================================================================
# Utilities
# =============================================================================

def is_prime(n: int) -> bool:
    """Check if n is prime."""
    if n < 2:
        return False
    if n == 2:
        return True
    if n % 2 == 0:
        return False
    for i in range(3, int(math.sqrt(n)) + 1, 2):
        if n % i == 0:
            return False
    return True


def nth_prime(n: int) -> int:
    """Get the nth prime number (1-indexed)."""
    if n <= 0:
        return 2
    count = 0
    candidate = 2
    while count < n:
        if is_prime(candidate):
            count += 1
            if count == n:
                return candidate
        candidate += 1
    return candidate


@lru_cache(maxsize=1024)
def factorize(n: int) -> Dict[int, int]:
    """Prime factorization as {prime: exponent}."""
    if n <= 1:
        return {}
    factors = {}
    d = 2
    while d * d <= n:
        while n % d == 0:
            factors[d] = factors.get(d, 0) + 1
            n //= d
        d += 1
    if n > 1:
        factors[n] = factors.get(n, 0) + 1
    return factors


def gcd(a: int, b: int) -> int:
    """Greatest common divisor."""
    while b:
        a, b = b, a % b
    return a


def lcm(a: int, b: int) -> int:
    """Least common multiple."""
    return a * b // gcd(a, b)


# =============================================================================
# Semantic Primitives
# =============================================================================

class SemanticCategory(Enum):
    """Categories of semantic primitives."""
    ENTITY = "entity"
    ACTION = "action"
    PROPERTY = "property"
    RELATION = "relation"
    QUANTIFIER = "quantifier"
    MODIFIER = "modifier"
    CONNECTIVE = "connective"


@dataclass
class SemanticPrimitive:
    """
    A primitive semantic unit.
    
    Represented by a prime number with associated meaning.
    """
    prime: int
    label: str
    category: SemanticCategory
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return self.prime
    
    def __eq__(self, other):
        if isinstance(other, SemanticPrimitive):
            return self.prime == other.prime
        return False
    
    @property
    def is_atomic(self) -> bool:
        """All primitives are atomic by definition."""
        return True


# Standard semantic primes (based on Wierzbicka's Natural Semantic Metalanguage)
SEMANTIC_PRIMES = {
    # Substantives (primes 2-7)
    2: SemanticPrimitive(2, "I", SemanticCategory.ENTITY),
    3: SemanticPrimitive(3, "YOU", SemanticCategory.ENTITY),
    5: SemanticPrimitive(5, "SOMEONE", SemanticCategory.ENTITY),
    7: SemanticPrimitive(7, "SOMETHING", SemanticCategory.ENTITY),
    
    # Determiners (primes 11-17)
    11: SemanticPrimitive(11, "THIS", SemanticCategory.MODIFIER),
    13: SemanticPrimitive(13, "THE_SAME", SemanticCategory.MODIFIER),
    17: SemanticPrimitive(17, "OTHER", SemanticCategory.MODIFIER),
    
    # Quantifiers (primes 19-29)
    19: SemanticPrimitive(19, "ONE", SemanticCategory.QUANTIFIER),
    23: SemanticPrimitive(23, "TWO", SemanticCategory.QUANTIFIER),
    29: SemanticPrimitive(29, "SOME", SemanticCategory.QUANTIFIER),
    31: SemanticPrimitive(31, "ALL", SemanticCategory.QUANTIFIER),
    37: SemanticPrimitive(37, "MANY", SemanticCategory.QUANTIFIER),
    
    # Mental predicates (primes 41-59)
    41: SemanticPrimitive(41, "THINK", SemanticCategory.ACTION),
    43: SemanticPrimitive(43, "KNOW", SemanticCategory.ACTION),
    47: SemanticPrimitive(47, "WANT", SemanticCategory.ACTION),
    53: SemanticPrimitive(53, "FEEL", SemanticCategory.ACTION),
    59: SemanticPrimitive(59, "SEE", SemanticCategory.ACTION),
    61: SemanticPrimitive(61, "HEAR", SemanticCategory.ACTION),
    
    # Speech (primes 67-73)
    67: SemanticPrimitive(67, "SAY", SemanticCategory.ACTION),
    71: SemanticPrimitive(71, "WORDS", SemanticCategory.ENTITY),
    73: SemanticPrimitive(73, "TRUE", SemanticCategory.PROPERTY),
    
    # Actions (primes 79-97)
    79: SemanticPrimitive(79, "DO", SemanticCategory.ACTION),
    83: SemanticPrimitive(83, "HAPPEN", SemanticCategory.ACTION),
    89: SemanticPrimitive(89, "MOVE", SemanticCategory.ACTION),
    97: SemanticPrimitive(97, "TOUCH", SemanticCategory.ACTION),
    
    # Existence (primes 101-107)
    101: SemanticPrimitive(101, "THERE_IS", SemanticCategory.RELATION),
    103: SemanticPrimitive(103, "BE", SemanticCategory.RELATION),
    107: SemanticPrimitive(107, "HAVE", SemanticCategory.RELATION),
    
    # Life (primes 109-113)
    109: SemanticPrimitive(109, "LIVE", SemanticCategory.ACTION),
    113: SemanticPrimitive(113, "DIE", SemanticCategory.ACTION),
    
    # Space (primes 127-149)
    127: SemanticPrimitive(127, "WHERE", SemanticCategory.RELATION),
    131: SemanticPrimitive(131, "HERE", SemanticCategory.MODIFIER),
    137: SemanticPrimitive(137, "ABOVE", SemanticCategory.RELATION),
    139: SemanticPrimitive(139, "BELOW", SemanticCategory.RELATION),
    149: SemanticPrimitive(149, "NEAR", SemanticCategory.RELATION),
    
    # Time (primes 151-173)
    151: SemanticPrimitive(151, "WHEN", SemanticCategory.RELATION),
    157: SemanticPrimitive(157, "NOW", SemanticCategory.MODIFIER),
    163: SemanticPrimitive(163, "BEFORE", SemanticCategory.RELATION),
    167: SemanticPrimitive(167, "AFTER", SemanticCategory.RELATION),
    173: SemanticPrimitive(173, "LONG", SemanticCategory.PROPERTY),
    
    # Quality (primes 179-197)
    179: SemanticPrimitive(179, "BIG", SemanticCategory.PROPERTY),
    181: SemanticPrimitive(181, "SMALL", SemanticCategory.PROPERTY),
    191: SemanticPrimitive(191, "GOOD", SemanticCategory.PROPERTY),
    193: SemanticPrimitive(193, "BAD", SemanticCategory.PROPERTY),
    197: SemanticPrimitive(197, "KIND", SemanticCategory.PROPERTY),
    
    # Logical (primes 199-229)
    199: SemanticPrimitive(199, "NOT", SemanticCategory.CONNECTIVE),
    211: SemanticPrimitive(211, "MAYBE", SemanticCategory.CONNECTIVE),
    223: SemanticPrimitive(223, "CAN", SemanticCategory.CONNECTIVE),
    227: SemanticPrimitive(227, "BECAUSE", SemanticCategory.CONNECTIVE),
    229: SemanticPrimitive(229, "IF", SemanticCategory.CONNECTIVE),
}


def get_primitive(prime: int) -> Optional[SemanticPrimitive]:
    """Get semantic primitive by prime."""
    return SEMANTIC_PRIMES.get(prime)


def get_primitive_by_label(label: str) -> Optional[SemanticPrimitive]:
    """Get semantic primitive by label."""
    for prim in SEMANTIC_PRIMES.values():
        if prim.label == label.upper():
            return prim
    return None


# =============================================================================
# CompoundBuilder - Build Complex Semantic Structures
# =============================================================================

@dataclass
class CompoundConcept:
    """
    A compound concept built from primitives.
    
    Represented as product of primes with structure.
    """
    value: int  # Product of constituent primes
    components: List[SemanticPrimitive]
    structure: str  # Structural description
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def factors(self) -> Dict[int, int]:
        """Prime factorization of value."""
        return factorize(self.value)
    
    @property
    def complexity(self) -> int:
        """Number of prime factors (with multiplicity)."""
        return sum(self.factors.values())
    
    def contains(self, primitive: SemanticPrimitive) -> bool:
        """Check if compound contains a primitive."""
        return self.value % primitive.prime == 0
    
    def overlap(self, other: 'CompoundConcept') -> int:
        """GCD gives overlap between compounds."""
        return gcd(self.value, other.value)
    
    def combine(self, other: 'CompoundConcept') -> 'CompoundConcept':
        """LCM combines compounds (union of primitives)."""
        combined_value = lcm(self.value, other.value)
        combined_components = list(set(self.components + other.components))
        return CompoundConcept(
            value=combined_value,
            components=combined_components,
            structure=f"({self.structure} ∪ {other.structure})"
        )


class CompoundBuilder:
    """
    Builds compound semantic structures from primitives.
    
    Operations:
    - Conjunction: A ∧ B = A * B (product)
    - Modification: Mod(A) = A * modifier_prime
    - Relation: R(A, B) = A * relation_prime * B
    - Negation: ¬A = NOT_prime * A
    """
    
    def __init__(self, custom_primes: Optional[Dict[int, SemanticPrimitive]] = None):
        """
        Initialize builder.
        
        Args:
            custom_primes: Additional semantic primes to use
        """
        self.primes = dict(SEMANTIC_PRIMES)
        if custom_primes:
            self.primes.update(custom_primes)
        
        # Cache for built compounds
        self.cache: Dict[int, CompoundConcept] = {}
    
    def primitive(self, label_or_prime: Union[str, int]) -> CompoundConcept:
        """Create compound from single primitive."""
        if isinstance(label_or_prime, int):
            prim = get_primitive(label_or_prime)
        else:
            prim = get_primitive_by_label(label_or_prime)
        
        if prim is None:
            raise ValueError(f"Unknown primitive: {label_or_prime}")
        
        return CompoundConcept(
            value=prim.prime,
            components=[prim],
            structure=prim.label
        )
    
    def conjoin(self, *concepts: CompoundConcept) -> CompoundConcept:
        """
        Conjoin multiple concepts: A ∧ B ∧ C = A * B * C.
        
        Represents concepts that hold simultaneously.
        """
        if not concepts:
            raise ValueError("Need at least one concept")
        
        result = concepts[0]
        for c in concepts[1:]:
            result = result.combine(c)
        
        result.structure = " ∧ ".join(c.structure for c in concepts)
        return result
    
    def modify(self, concept: CompoundConcept,
               modifier: Union[str, int, SemanticPrimitive]) -> CompoundConcept:
        """
        Apply modifier to concept.
        
        E.g., BIG(SOMETHING) = SOMETHING * BIG
        """
        if isinstance(modifier, SemanticPrimitive):
            mod_prim = modifier
        elif isinstance(modifier, int):
            mod_prim = get_primitive(modifier)
        else:
            mod_prim = get_primitive_by_label(modifier)
        
        if mod_prim is None:
            raise ValueError(f"Unknown modifier: {modifier}")
        
        return CompoundConcept(
            value=concept.value * mod_prim.prime,
            components=concept.components + [mod_prim],
            structure=f"{mod_prim.label}({concept.structure})"
        )
    
    def relate(self, subject: CompoundConcept,
               relation: Union[str, int, SemanticPrimitive],
               object_: CompoundConcept) -> CompoundConcept:
        """
        Create relational structure: R(A, B) = A * R * B.
        
        E.g., ABOVE(BIRD, TREE) = BIRD * ABOVE * TREE
        """
        if isinstance(relation, SemanticPrimitive):
            rel_prim = relation
        elif isinstance(relation, int):
            rel_prim = get_primitive(relation)
        else:
            rel_prim = get_primitive_by_label(relation)
        
        if rel_prim is None:
            raise ValueError(f"Unknown relation: {relation}")
        
        return CompoundConcept(
            value=subject.value * rel_prim.prime * object_.value,
            components=subject.components + [rel_prim] + object_.components,
            structure=f"{rel_prim.label}({subject.structure}, {object_.structure})"
        )
    
    def negate(self, concept: CompoundConcept) -> CompoundConcept:
        """
        Negate concept: ¬A = NOT * A.
        """
        not_prim = get_primitive(199)  # NOT
        if not_prim is None:
            raise ValueError("NOT primitive not found")
        
        return CompoundConcept(
            value=concept.value * not_prim.prime,
            components=[not_prim] + concept.components,
            structure=f"NOT({concept.structure})"
        )
    
    def conditional(self, antecedent: CompoundConcept,
                    consequent: CompoundConcept) -> CompoundConcept:
        """
        Create conditional: IF A THEN B.
        
        Represented as IF * A * B.
        """
        if_prim = get_primitive(229)  # IF
        if if_prim is None:
            raise ValueError("IF primitive not found")
        
        return CompoundConcept(
            value=antecedent.value * if_prim.prime * consequent.value,
            components=[if_prim] + antecedent.components + consequent.components,
            structure=f"IF({antecedent.structure}, {consequent.structure})"
        )
    
    def quantify(self, quantifier: Union[str, int],
                 concept: CompoundConcept) -> CompoundConcept:
        """
        Apply quantifier: ALL(X), SOME(X), etc.
        """
        if isinstance(quantifier, int):
            quant_prim = get_primitive(quantifier)
        else:
            quant_prim = get_primitive_by_label(quantifier)
        
        if quant_prim is None:
            raise ValueError(f"Unknown quantifier: {quantifier}")
        
        return CompoundConcept(
            value=quant_prim.prime * concept.value,
            components=[quant_prim] + concept.components,
            structure=f"{quant_prim.label}({concept.structure})"
        )
    
    def analyze(self, value: int) -> CompoundConcept:
        """
        Analyze an integer as a semantic compound.
        
        Factorizes and maps factors to primitives.
        """
        factors = factorize(value)
        components = []
        labels = []
        
        for prime, exp in factors.items():
            prim = get_primitive(prime)
            if prim:
                for _ in range(exp):
                    components.append(prim)
                labels.append(f"{prim.label}" + (f"^{exp}" if exp > 1 else ""))
            else:
                labels.append(f"p_{prime}" + (f"^{exp}" if exp > 1 else ""))
        
        return CompoundConcept(
            value=value,
            components=components,
            structure=" * ".join(labels)
        )
    
    def similarity(self, c1: CompoundConcept, c2: CompoundConcept) -> float:
        """
        Compute semantic similarity between compounds.
        
        Based on shared prime factors.
        """
        overlap = c1.overlap(c2)
        if overlap <= 1:
            return 0.0
        
        # Jaccard-like similarity
        union = lcm(c1.value, c2.value)
        
        overlap_complexity = sum(factorize(overlap).values())
        union_complexity = sum(factorize(union).values())
        
        return overlap_complexity / union_complexity if union_complexity > 0 else 0.0


# =============================================================================
# SemanticInference - Make Inferences from Semantic Knowledge
# =============================================================================

@dataclass
class InferenceRule:
    """A rule for semantic inference."""
    name: str
    premises: List[int]  # Pattern values that must divide
    conclusion: int  # Pattern to add
    confidence: float = 1.0
    
    def applies_to(self, knowledge: Set[int]) -> bool:
        """Check if rule can fire on given knowledge."""
        for premise in self.premises:
            # Check if any knowledge item contains this pattern
            if not any(k % premise == 0 for k in knowledge):
                return False
        return True
    
    def apply(self, knowledge: Set[int]) -> Set[int]:
        """Apply rule and return new knowledge."""
        if not self.applies_to(knowledge):
            return set()
        
        # Find matching items for each premise
        matches = []
        for premise in self.premises:
            for k in knowledge:
                if k % premise == 0:
                    matches.append(k)
                    break
        
        if not matches:
            return set()
        
        # Generate conclusion by combining matches with conclusion pattern
        base = matches[0]
        for m in matches[1:]:
            base = lcm(base, m)
        
        return {lcm(base, self.conclusion)}


class SemanticInference:
    """
    Semantic inference engine.
    
    Uses prime-based knowledge representation for:
    - Forward chaining inference
    - Pattern matching
    - Subsumption checking
    - Analogical reasoning
    """
    
    def __init__(self, builder: Optional[CompoundBuilder] = None):
        """
        Initialize inference engine.
        
        Args:
            builder: CompoundBuilder for constructing concepts
        """
        self.builder = builder or CompoundBuilder()
        self.knowledge: Set[int] = set()
        self.rules: List[InferenceRule] = []
        
        # Initialize with default rules
        self._init_default_rules()
    
    def _init_default_rules(self):
        """Initialize default inference rules."""
        # If X is GOOD and X is ACTION, then WANT(I, X)
        # GOOD * DO → WANT * I
        good = 191  # GOOD
        do = 79     # DO
        want = 47   # WANT
        i = 2       # I
        
        self.rules.append(InferenceRule(
            name="good_action_want",
            premises=[good * do],
            conclusion=want * i,
            confidence=0.8
        ))
        
        # If X HAPPENS and X is BAD, then FEEL BAD
        bad = 193
        happen = 83
        feel = 53
        
        self.rules.append(InferenceRule(
            name="bad_event_feeling",
            premises=[bad * happen],
            conclusion=feel * bad,
            confidence=0.7
        ))
        
        # Transitivity: ABOVE(A, B) and ABOVE(B, C) → ABOVE(A, C)
        above = 137
        self.rules.append(InferenceRule(
            name="above_transitive",
            premises=[above],  # Simplified
            conclusion=above,
            confidence=1.0
        ))
    
    def add_rule(self, rule: InferenceRule):
        """Add inference rule."""
        self.rules.append(rule)
    
    def assert_knowledge(self, concept: CompoundConcept):
        """Assert a concept as known."""
        self.knowledge.add(concept.value)
    
    def assert_value(self, value: int):
        """Assert a value directly."""
        self.knowledge.add(value)
    
    def query(self, pattern: int) -> List[int]:
        """
        Query knowledge base for items matching pattern.
        
        Pattern matches if it divides the knowledge item.
        """
        return [k for k in self.knowledge if k % pattern == 0]
    
    def subsumes(self, general: int, specific: int) -> bool:
        """
        Check if general concept subsumes specific.
        
        A subsumes B if A divides B (A's factors are subset of B's).
        """
        return specific % general == 0
    
    def forward_chain(self, max_steps: int = 100) -> Set[int]:
        """
        Perform forward chaining inference.
        
        Applies all applicable rules until fixed point.
        """
        new_knowledge = set()
        
        for _ in range(max_steps):
            fired = False
            
            for rule in self.rules:
                if rule.applies_to(self.knowledge):
                    inferred = rule.apply(self.knowledge)
                    
                    # Add only genuinely new knowledge
                    for k in inferred:
                        if k not in self.knowledge:
                            new_knowledge.add(k)
                            self.knowledge.add(k)
                            fired = True
            
            if not fired:
                break
        
        return new_knowledge
    
    def abductive_explain(self, observation: int) -> List[Tuple[int, float]]:
        """
        Abductive inference: find explanations for observation.
        
        Returns list of (explanation, confidence) tuples.
        """
        explanations = []
        
        for rule in self.rules:
            # Check if conclusion matches observation
            if observation % rule.conclusion == 0:
                # Premises could explain this
                for premise in rule.premises:
                    explanations.append((premise, rule.confidence))
        
        # Also add factorization-based explanations
        factors = factorize(observation)
        for prime, exp in factors.items():
            if prime in SEMANTIC_PRIMES:
                explanations.append((prime, 1.0 / exp))
        
        return explanations
    
    def analogical_inference(self, source: int, target: int,
                             source_property: int) -> Optional[int]:
        """
        Analogical reasoning: transfer property from source to target.
        
        If source has property, and target is similar, infer target might have property.
        """
        # Compute similarity via shared factors
        shared = gcd(source, target)
        if shared <= 1:
            return None
        
        # If source has property
        if source % source_property == 0:
            # Infer target might have property (with analogy marker)
            return target * source_property
        
        return None
    
    def generalize(self, instances: List[int]) -> int:
        """
        Generalize from multiple instances.
        
        GCD gives most general concept subsuming all instances.
        """
        if not instances:
            return 1
        
        result = instances[0]
        for instance in instances[1:]:
            result = gcd(result, instance)
        
        return result
    
    def specialize(self, concept: int, with_property: int) -> int:
        """
        Specialize concept by adding property.
        
        LCM gives most specific concept with both.
        """
        return lcm(concept, with_property)
    
    def contradiction_check(self) -> List[Tuple[int, int]]:
        """
        Check for contradictions in knowledge base.
        
        Contradiction: both X and NOT(X) are known.
        """
        not_prime = 199  # NOT
        contradictions = []
        
        for k1 in self.knowledge:
            for k2 in self.knowledge:
                if k1 != k2:
                    # Check if k2 = NOT(k1)
                    if k2 == k1 * not_prime or k1 == k2 * not_prime:
                        contradictions.append((k1, k2))
        
        return contradictions
    
    def get_all_inferences(self) -> Dict[str, Any]:
        """Get summary of all possible inferences."""
        return {
            'knowledge_count': len(self.knowledge),
            'knowledge_items': list(self.knowledge),
            'applicable_rules': [
                r.name for r in self.rules if r.applies_to(self.knowledge)
            ],
            'contradictions': self.contradiction_check()
        }


# =============================================================================
# EntityExtractor - Extract Entities from Text
# =============================================================================

@dataclass
class ExtractedEntity:
    """An extracted entity with semantic representation."""
    text: str
    start: int
    end: int
    entity_type: str
    prime_encoding: int
    confidence: float
    
    def __repr__(self):
        return f"Entity({self.text!r}, type={self.entity_type}, p={self.prime_encoding})"


@dataclass
class ExtractedRelation:
    """An extracted relation between entities."""
    subject: ExtractedEntity
    predicate: str
    object_: ExtractedEntity
    prime_encoding: int
    confidence: float


class EntityExtractor:
    """
    Extracts entities and relations from text.
    
    Uses pattern matching and prime-based encoding.
    """
    
    def __init__(self, builder: Optional[CompoundBuilder] = None):
        """
        Initialize extractor.
        
        Args:
            builder: CompoundBuilder for semantic encoding
        """
        self.builder = builder or CompoundBuilder()
        
        # Entity type to prime mapping
        self.entity_primes = {
            'PERSON': 2,      # I
            'THING': 7,       # SOMETHING
            'PLACE': 127,     # WHERE
            'TIME': 151,      # WHEN
            'ACTION': 79,     # DO
            'PROPERTY': 179,  # BIG (representative property)
        }
        
        # Patterns for extraction (simplified)
        self.patterns = {
            'PERSON': [
                r'\b[A-Z][a-z]+\b',  # Capitalized words
                r'\b(he|she|they|we|I)\b',
            ],
            'THING': [
                r'\b(the|a|an)\s+\w+\b',
            ],
            'ACTION': [
                r'\b\w+(s|ed|ing)\b',  # Verb patterns
            ],
        }
        
        # Relation patterns
        self.relation_patterns = [
            (r'(\w+)\s+is\s+(\w+)', 'IS'),
            (r'(\w+)\s+has\s+(\w+)', 'HAS'),
            (r'(\w+)\s+(?:wants?|wanted)\s+(\w+)', 'WANT'),
            (r'(\w+)\s+(?:sees?|saw)\s+(\w+)', 'SEE'),
            (r'(\w+)\s+above\s+(\w+)', 'ABOVE'),
            (r'(\w+)\s+before\s+(\w+)', 'BEFORE'),
        ]
    
    def encode_entity_type(self, entity_type: str) -> int:
        """Get prime encoding for entity type."""
        return self.entity_primes.get(entity_type, 7)  # Default to SOMETHING
    
    def extract_entities(self, text: str) -> List[ExtractedEntity]:
        """
        Extract entities from text.
        
        Args:
            text: Input text
            
        Returns:
            List of extracted entities
        """
        entities = []
        
        for entity_type, patterns in self.patterns.items():
            for pattern in patterns:
                for match in re.finditer(pattern, text, re.IGNORECASE):
                    entity_text = match.group()
                    
                    # Generate unique prime encoding
                    # Base prime for type * hash-derived modifier
                    base = self.encode_entity_type(entity_type)
                    modifier = hash(entity_text.lower()) % 1000 + 233  # Offset to primes
                    
                    # Find next prime after modifier
                    while not is_prime(modifier):
                        modifier += 1
                    
                    encoding = base * modifier
                    
                    entities.append(ExtractedEntity(
                        text=entity_text,
                        start=match.start(),
                        end=match.end(),
                        entity_type=entity_type,
                        prime_encoding=encoding,
                        confidence=0.7
                    ))
        
        # Deduplicate overlapping entities (keep higher confidence)
        entities = self._deduplicate_entities(entities)
        
        return entities
    
    def _deduplicate_entities(self, entities: List[ExtractedEntity]) -> List[ExtractedEntity]:
        """Remove overlapping entities, keeping highest confidence."""
        if not entities:
            return []
        
        # Sort by start position, then by confidence (descending)
        entities.sort(key=lambda e: (e.start, -e.confidence))
        
        result = []
        last_end = -1
        
        for entity in entities:
            if entity.start >= last_end:
                result.append(entity)
                last_end = entity.end
        
        return result
    
    def extract_relations(self, text: str,
                          entities: Optional[List[ExtractedEntity]] = None
                          ) -> List[ExtractedRelation]:
        """
        Extract relations between entities.
        
        Args:
            text: Input text
            entities: Pre-extracted entities (optional)
            
        Returns:
            List of extracted relations
        """
        if entities is None:
            entities = self.extract_entities(text)
        
        relations = []
        
        for pattern, rel_type in self.relation_patterns:
            for match in re.finditer(pattern, text, re.IGNORECASE):
                groups = match.groups()
                if len(groups) >= 2:
                    subj_text = groups[0]
                    obj_text = groups[1]
                    
                    # Find matching entities
                    subj_entity = self._find_entity(subj_text, entities)
                    obj_entity = self._find_entity(obj_text, entities)
                    
                    if subj_entity and obj_entity:
                        # Get relation prime
                        rel_prim = get_primitive_by_label(rel_type)
                        rel_prime = rel_prim.prime if rel_prim else 103  # Default to BE
                        
                        encoding = subj_entity.prime_encoding * rel_prime * obj_entity.prime_encoding
                        
                        relations.append(ExtractedRelation(
                            subject=subj_entity,
                            predicate=rel_type,
                            object_=obj_entity,
                            prime_encoding=encoding,
                            confidence=0.6
                        ))
        
        return relations
    
    def _find_entity(self, text: str,
                     entities: List[ExtractedEntity]) -> Optional[ExtractedEntity]:
        """Find entity matching text."""
        text_lower = text.lower().strip()
        
        for entity in entities:
            if entity.text.lower().strip() == text_lower:
                return entity
            if text_lower in entity.text.lower():
                return entity
        
        # Create ad-hoc entity if not found
        return ExtractedEntity(
            text=text,
            start=0,
            end=len(text),
            entity_type='THING',
            prime_encoding=7 * (hash(text) % 1000 + 233),
            confidence=0.5
        )
    
    def to_semantic_graph(self, text: str) -> Dict[str, Any]:
        """
        Convert text to semantic graph representation.
        
        Returns:
            Dict with nodes (entities) and edges (relations)
        """
        entities = self.extract_entities(text)
        relations = self.extract_relations(text, entities)
        
        nodes = [
            {
                'id': i,
                'text': e.text,
                'type': e.entity_type,
                'prime': e.prime_encoding,
                'confidence': e.confidence
            }
            for i, e in enumerate(entities)
        ]
        
        # Map entities to node IDs
        entity_to_id = {e.text.lower(): i for i, e in enumerate(entities)}
        
        edges = []
        for rel in relations:
            subj_id = entity_to_id.get(rel.subject.text.lower(), -1)
            obj_id = entity_to_id.get(rel.object_.text.lower(), -1)
            
            if subj_id >= 0 and obj_id >= 0:
                edges.append({
                    'source': subj_id,
                    'target': obj_id,
                    'relation': rel.predicate,
                    'prime': rel.prime_encoding,
                    'confidence': rel.confidence
                })
        
        return {
            'nodes': nodes,
            'edges': edges,
            'text': text
        }
    
    def semantic_fingerprint(self, text: str) -> int:
        """
        Generate semantic fingerprint of text.
        
        Product of all entity primes gives unique semantic signature.
        """
        entities = self.extract_entities(text)
        
        if not entities:
            return 1
        
        fingerprint = 1
        for entity in entities:
            fingerprint *= entity.prime_encoding
        
        return fingerprint


# =============================================================================
# Integration Functions
# =============================================================================

def build_compound(labels: List[str]) -> CompoundConcept:
    """
    Build compound from list of primitive labels.
    
    Args:
        labels: List of primitive labels like ["GOOD", "THING"]
        
    Returns:
        CompoundConcept representing conjunction
    """
    builder = CompoundBuilder()
    concepts = [builder.primitive(label) for label in labels]
    return builder.conjoin(*concepts)


def analyze_semantic_value(value: int) -> Dict[str, Any]:
    """
    Analyze a semantic value (product of primes).
    
    Args:
        value: Integer representing semantic compound
        
    Returns:
        Analysis dictionary
    """
    builder = CompoundBuilder()
    compound = builder.analyze(value)
    
    # Get primitive labels
    labels = [c.label for c in compound.components]
    categories = [c.category.value for c in compound.components]
    
    return {
        'value': value,
        'structure': compound.structure,
        'complexity': compound.complexity,
        'primitives': labels,
        'categories': categories,
        'factors': dict(compound.factors)
    }


def infer_from_knowledge(knowledge: List[int],
                         query: int) -> Dict[str, Any]:
    """
    Run inference on knowledge base and query.
    
    Args:
        knowledge: List of known semantic values
        query: Pattern to query
        
    Returns:
        Inference results
    """
    engine = SemanticInference()
    
    for k in knowledge:
        engine.assert_value(k)
    
    # Forward chain
    new_inferences = engine.forward_chain()
    
    # Query
    matches = engine.query(query)
    
    # Explanations
    explanations = engine.abductive_explain(query)
    
    return {
        'knowledge_added': knowledge,
        'new_inferences': list(new_inferences),
        'query_matches': matches,
        'explanations': explanations,
        'total_knowledge': len(engine.knowledge)
    }


def extract_semantic_structure(text: str) -> Dict[str, Any]:
    """
    Full semantic extraction from text.
    
    Args:
        text: Input text
        
    Returns:
        Complete semantic analysis
    """
    extractor = EntityExtractor()
    
    entities = extractor.extract_entities(text)
    relations = extractor.extract_relations(text, entities)
    graph = extractor.to_semantic_graph(text)
    fingerprint = extractor.semantic_fingerprint(text)
    
    return {
        'text': text,
        'entities': [
            {
                'text': e.text,
                'type': e.entity_type,
                'prime': e.prime_encoding,
                'span': (e.start, e.end)
            }
            for e in entities
        ],
        'relations': [
            {
                'subject': r.subject.text,
                'predicate': r.predicate,
                'object': r.object_.text,
                'prime': r.prime_encoding
            }
            for r in relations
        ],
        'graph': graph,
        'fingerprint': fingerprint,
        'fingerprint_analysis': analyze_semantic_value(fingerprint) if fingerprint > 1 else None
    }


def semantic_similarity(text1: str, text2: str) -> float:
    """
    Compute semantic similarity between two texts.
    
    Uses prime-based fingerprints.
    """
    extractor = EntityExtractor()
    
    fp1 = extractor.semantic_fingerprint(text1)
    fp2 = extractor.semantic_fingerprint(text2)
    
    # GCD gives shared semantic content
    shared = gcd(fp1, fp2)
    
    if shared <= 1:
        return 0.0
    
    # LCM gives combined content
    combined = lcm(fp1, fp2)
    
    # Similarity based on factorization
    shared_factors = sum(factorize(shared).values())
    combined_factors = sum(factorize(combined).values())
    
    return shared_factors / combined_factors if combined_factors > 0 else 0.0