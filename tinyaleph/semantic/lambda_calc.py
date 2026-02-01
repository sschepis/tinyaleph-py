"""
λ-Calculus Translation Layer for Prime-Indexed Semantic Calculi

Provides a denotational semantics via λ-calculus:
- τ (tau) translation function from typed terms to λ-expressions
- Compositional semantics via function application
- Type preservation during translation
- Denotational semantics bridge
- PRQS Lexicon for concept interpretation

The translation τ maps:
- N(p) → λx.p (constant function returning prime p)
- A(p) → λf.λx.f(p,x) (operator awaiting application)
- FUSE(p,q,r) → λx.(p+q+r) (fusion to sum)
- A(p)N(q) → (τ(A(p)))(τ(N(q))) = p ⊕ q
- S₁ ◦ S₂ → (τ(S₁), τ(S₂))
- S₁ ⇒ S₂ → τ(S₁) → τ(S₂)
"""

from typing import List, Optional, Dict, Any, Set, Tuple
from dataclasses import dataclass
from abc import ABC, abstractmethod
import math

from ..core.primes import is_prime, first_n_primes
from .types import (
    Term, NounTerm, AdjTerm, ChainTerm, FusionTerm,
    NounSentence, SeqSentence, ImplSentence,
    N, A, FUSE, CHAIN
)
from .reduction import ReductionSystem, DEFAULT_OPERATOR


# =============================================================================
# λ-EXPRESSION AST
# =============================================================================

class LambdaExpr(ABC):
    """Base class for λ-expressions."""
    
    @abstractmethod
    def get_type(self) -> Optional[Dict[str, Any]]:
        """Get the type of this expression."""
        pass
    
    @abstractmethod
    def __str__(self) -> str:
        """Convert to string representation."""
        pass
    
    def is_value(self) -> bool:
        """Check if this is a value (fully evaluated)."""
        return False
    
    @abstractmethod
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        """Substitute variable x with expression e."""
        pass
    
    def free_vars(self) -> Set[str]:
        """Get free variables."""
        return set()
    
    def alpha_equals(self, other: 'LambdaExpr') -> bool:
        """Alpha-equivalence check."""
        return str(self) == str(other)


class VarExpr(LambdaExpr):
    """Variable expression."""
    
    def __init__(self, name: str, var_type: Optional[Dict[str, Any]] = None):
        self.name = name
        self.var_type = var_type
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        return self.var_type
    
    def __str__(self) -> str:
        return self.name
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        if self.name == x:
            return e
        return self
    
    def free_vars(self) -> Set[str]:
        return {self.name}


class ConstExpr(LambdaExpr):
    """Constant (prime number) expression."""
    
    def __init__(self, value: int):
        self.value = value
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        return {'kind': 'const', 'value': self.value}
    
    def __str__(self) -> str:
        return str(self.value)
    
    def is_value(self) -> bool:
        return True
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        return self


class LamExpr(LambdaExpr):
    """Lambda abstraction: λx.body"""
    
    def __init__(self, param: str, body: LambdaExpr, 
                 param_type: Optional[Dict[str, Any]] = None):
        self.param = param
        self.body = body
        self.param_type = param_type
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        return {
            'kind': 'function',
            'param_type': self.param_type,
            'return_type': self.body.get_type()
        }
    
    def __str__(self) -> str:
        type_ann = f":{self.param_type}" if self.param_type else ''
        return f"(λ{self.param}{type_ann}.{self.body})"
    
    def is_value(self) -> bool:
        return True  # Lambdas are values
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        if self.param == x:
            # x is bound here, no substitution in body
            return self
        
        # Check for variable capture
        free_in_e = e.free_vars()
        if self.param in free_in_e:
            # Alpha-rename to avoid capture
            fresh = self._fresh_var(self.param)
            renamed_body = self.body.substitute(self.param, VarExpr(fresh))
            return LamExpr(fresh, renamed_body.substitute(x, e), self.param_type)
        
        return LamExpr(self.param, self.body.substitute(x, e), self.param_type)
    
    def free_vars(self) -> Set[str]:
        vars_set = self.body.free_vars()
        vars_set.discard(self.param)
        return vars_set
    
    def _fresh_var(self, base: str) -> str:
        return f"{base}'"


class AppExpr(LambdaExpr):
    """Application expression: e1 e2"""
    
    def __init__(self, func: LambdaExpr, arg: LambdaExpr):
        self.func = func
        self.arg = arg
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        func_type = self.func.get_type()
        if func_type and func_type.get('kind') == 'function':
            return func_type.get('return_type')
        return None
    
    def __str__(self) -> str:
        return f"({self.func} {self.arg})"
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        return AppExpr(
            self.func.substitute(x, e),
            self.arg.substitute(x, e)
        )
    
    def free_vars(self) -> Set[str]:
        return self.func.free_vars() | self.arg.free_vars()


class PairExpr(LambdaExpr):
    """Pair expression: (e1, e2) for sentence composition."""
    
    def __init__(self, left: LambdaExpr, right: LambdaExpr):
        self.left = left
        self.right = right
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        return {
            'kind': 'pair',
            'left_type': self.left.get_type(),
            'right_type': self.right.get_type()
        }
    
    def __str__(self) -> str:
        return f"⟨{self.left}, {self.right}⟩"
    
    def is_value(self) -> bool:
        return self.left.is_value() and self.right.is_value()
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        return PairExpr(
            self.left.substitute(x, e),
            self.right.substitute(x, e)
        )
    
    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


class ImplExpr(LambdaExpr):
    """Implication expression: e1 → e2"""
    
    def __init__(self, antecedent: LambdaExpr, consequent: LambdaExpr):
        self.antecedent = antecedent
        self.consequent = consequent
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        return {
            'kind': 'implication',
            'ante_type': self.antecedent.get_type(),
            'cons_type': self.consequent.get_type()
        }
    
    def __str__(self) -> str:
        return f"({self.antecedent} → {self.consequent})"
    
    def is_value(self) -> bool:
        return self.antecedent.is_value() and self.consequent.is_value()
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        return ImplExpr(
            self.antecedent.substitute(x, e),
            self.consequent.substitute(x, e)
        )
    
    def free_vars(self) -> Set[str]:
        return self.antecedent.free_vars() | self.consequent.free_vars()


class PrimOpExpr(LambdaExpr):
    """Primitive operator application: ⊕(p, q)"""
    
    def __init__(self, op: str, left: LambdaExpr, right: LambdaExpr):
        self.op = op
        self.left = left
        self.right = right
    
    def get_type(self) -> Optional[Dict[str, Any]]:
        return {'kind': 'prime'}
    
    def __str__(self) -> str:
        return f"({self.left} ⊕ {self.right})"
    
    def substitute(self, x: str, e: 'LambdaExpr') -> 'LambdaExpr':
        return PrimOpExpr(
            self.op,
            self.left.substitute(x, e),
            self.right.substitute(x, e)
        )
    
    def free_vars(self) -> Set[str]:
        return self.left.free_vars() | self.right.free_vars()


# =============================================================================
# τ TRANSLATION FUNCTION
# =============================================================================

class Translator:
    """
    Translator - Implements the τ function.
    
    Translates typed terms to λ-expressions.
    """
    
    def __init__(self, operator=None):
        self.operator = operator or DEFAULT_OPERATOR
        self.var_counter = 0
    
    def fresh_var(self) -> str:
        """Generate a fresh variable name."""
        self.var_counter += 1
        return f"x{self.var_counter}"
    
    def translate(self, term: Term) -> LambdaExpr:
        """
        τ: Term → LambdaExpr
        Main translation function.
        """
        # τ(N(p)) = p (constant)
        if isinstance(term, NounTerm):
            return ConstExpr(term.prime)
        
        # τ(A(p)) = λx.⊕(p, x)
        if isinstance(term, AdjTerm):
            x = self.fresh_var()
            return LamExpr(
                x,
                PrimOpExpr('⊕', ConstExpr(term.prime), VarExpr(x)),
                {'kind': 'prime'}
            )
        
        # τ(FUSE(p,q,r)) = p+q+r (constant)
        if isinstance(term, FusionTerm):
            return ConstExpr(term.get_fused_prime())
        
        # τ(A(p₁)...A(pₖ)N(q)) = ((τ(A(p₁)) ... (τ(A(pₖ)) τ(N(q))))
        if isinstance(term, ChainTerm):
            # Start with the noun
            result = self.translate(term.noun)
            
            # Apply operators from innermost to outermost
            for i in range(len(term.operators) - 1, -1, -1):
                op = term.operators[i]
                op_lambda = self.translate(op)
                result = AppExpr(op_lambda, result)
            
            return result
        
        # τ(NounSentence(e)) = τ(e)
        if isinstance(term, NounSentence):
            return self.translate(term.expr)
        
        # τ(S₁ ◦ S₂) = ⟨τ(S₁), τ(S₂)⟩
        if isinstance(term, SeqSentence):
            return PairExpr(
                self.translate(term.left),
                self.translate(term.right)
            )
        
        # τ(S₁ ⇒ S₂) = τ(S₁) → τ(S₂)
        if isinstance(term, ImplSentence):
            return ImplExpr(
                self.translate(term.antecedent),
                self.translate(term.consequent)
            )
        
        raise ValueError(f"Cannot translate: {term}")
    
    def translate_with_trace(self, term: Term) -> Dict[str, Any]:
        """Translate and show the translation steps."""
        result = self.translate(term)
        return {
            'source': term.signature(),
            'target': str(result),
            'type': result.get_type()
        }


class TypeDirectedTranslator(Translator):
    """TypeDirectedTranslator - Translation with type constraints."""
    
    def translate_typed(self, term: Term, context=None) -> Dict[str, Any]:
        """Translate with explicit type annotations."""
        lambda_expr = self.translate(term)
        
        return {
            'expr': lambda_expr,
            'source_type': term.__class__.__name__,
            'target_type': lambda_expr.get_type(),
            'context': context
        }
    
    def check_type_preservation(self, term: Term, context=None) -> Dict[str, Any]:
        """
        Check type preservation.
        τ preserves typing: Γ ⊢ e : T implies τ(Γ) ⊢ τ(e) : τ(T)
        """
        translation = self.translate_typed(term, context)
        
        return {
            'preserved': translation['target_type'] is not None,
            'source': translation['source_type'],
            'target': translation['target_type']
        }


# =============================================================================
# λ-CALCULUS EVALUATOR
# =============================================================================

class LambdaEvaluator:
    """Call-by-value evaluator for λ-expressions."""
    
    def __init__(self, operator=None):
        self.operator = operator or DEFAULT_OPERATOR
        self.max_steps = 1000
    
    def step(self, expr: LambdaExpr) -> Optional[LambdaExpr]:
        """Evaluate one step (small-step semantics)."""
        # Application of lambda to value: (λx.e) v → e[x := v]
        if isinstance(expr, AppExpr):
            # First evaluate function to value
            if not expr.func.is_value():
                new_func = self.step(expr.func)
                if new_func:
                    return AppExpr(new_func, expr.arg)
            
            # Then evaluate argument to value
            if not expr.arg.is_value():
                new_arg = self.step(expr.arg)
                if new_arg:
                    return AppExpr(expr.func, new_arg)
            
            # Both are values, perform β-reduction
            if isinstance(expr.func, LamExpr) and expr.arg.is_value():
                return expr.func.body.substitute(expr.func.param, expr.arg)
        
        # Primitive operator application
        if isinstance(expr, PrimOpExpr):
            # Evaluate operands first
            if not expr.left.is_value():
                new_left = self.step(expr.left)
                if new_left:
                    return PrimOpExpr(expr.op, new_left, expr.right)
            
            if not expr.right.is_value():
                new_right = self.step(expr.right)
                if new_right:
                    return PrimOpExpr(expr.op, expr.left, new_right)
            
            # Both values, apply operator
            if isinstance(expr.left, ConstExpr) and isinstance(expr.right, ConstExpr):
                p = expr.left.value
                q = expr.right.value
                if self.operator.can_apply(p, q):
                    result = self.operator.apply(p, q)
                    return ConstExpr(result)
                elif self.operator.can_apply(q, p):
                    result = self.operator.apply(q, p)
                    return ConstExpr(result)
                # Fallback: return the larger value
                return ConstExpr(max(p, q))
        
        # Pair reduction
        if isinstance(expr, PairExpr):
            if not expr.left.is_value():
                new_left = self.step(expr.left)
                if new_left:
                    return PairExpr(new_left, expr.right)
            
            if not expr.right.is_value():
                new_right = self.step(expr.right)
                if new_right:
                    return PairExpr(expr.left, new_right)
        
        # Implication reduction
        if isinstance(expr, ImplExpr):
            if not expr.antecedent.is_value():
                new_ante = self.step(expr.antecedent)
                if new_ante:
                    return ImplExpr(new_ante, expr.consequent)
            
            if not expr.consequent.is_value():
                new_cons = self.step(expr.consequent)
                if new_cons:
                    return ImplExpr(expr.antecedent, new_cons)
        
        return None
    
    def evaluate(self, expr: LambdaExpr) -> Dict[str, Any]:
        """Fully evaluate expression to value."""
        current = expr
        steps = 0
        
        while steps < self.max_steps:
            next_expr = self.step(current)
            if not next_expr:
                break
            current = next_expr
            steps += 1
        
        return {
            'result': current,
            'steps': steps,
            'is_value': current.is_value()
        }


# =============================================================================
# COMPOSITIONAL SEMANTICS
# =============================================================================

class Semantics:
    """Denotational semantics via λ-calculus interpretation."""
    
    def __init__(self, operator=None):
        self.translator = Translator(operator)
        self.evaluator = LambdaEvaluator(operator)
        self.reducer = ReductionSystem(operator)
    
    def denote(self, term: Term) -> LambdaExpr:
        """
        Get the denotation of a term.
        [[e]] = evaluate(τ(e))
        """
        lambda_expr = self.translator.translate(term)
        result = self.evaluator.evaluate(lambda_expr)
        return result['result']
    
    def equivalent(self, term1: Term, term2: Term) -> bool:
        """
        Check semantic equivalence.
        [[e₁]] = [[e₂]]
        """
        d1 = self.denote(term1)
        d2 = self.denote(term2)
        
        if isinstance(d1, ConstExpr) and isinstance(d2, ConstExpr):
            return d1.value == d2.value
        
        return str(d1) == str(d2)
    
    def verify_semantic_equivalence(self, term: Term) -> Dict[str, Any]:
        """
        Verify operational and denotational semantics agree.
        Theorem: If e →* v in operational semantics, then [[e]] = v
        """
        # Get operational result
        op_result = self.reducer.evaluate(term)
        
        # Get denotational result
        den_result = self.denote(term)
        
        # Compare
        equivalent = False
        if isinstance(op_result, NounTerm) and isinstance(den_result, ConstExpr):
            equivalent = op_result.prime == den_result.value
        
        return {
            'term': term.signature(),
            'operational': op_result.prime if isinstance(op_result, NounTerm) else op_result.signature(),
            'denotational': den_result.value if isinstance(den_result, ConstExpr) else str(den_result),
            'equivalent': equivalent
        }


# =============================================================================
# PRQS LEXICON
# =============================================================================

# PRQS (Prime-Indexed Resonant Quantum Semantics) Lexicon
# From TriadicPrimeFusion paper: Core semantic primes form a minimal basis
PRQS_LEXICON = {
    'nouns': {
        2: {'concept': 'duality', 'category': 'foundation', 'role': 'split/pair'},
        3: {'concept': 'structure', 'category': 'foundation', 'role': 'form/frame'},
        5: {'concept': 'change', 'category': 'dynamic', 'role': 'motion/flow'},
        7: {'concept': 'identity', 'category': 'foundation', 'role': 'self/same'},
        11: {'concept': 'complexity', 'category': 'structural', 'role': 'layers/depth'},
        13: {'concept': 'relation', 'category': 'relational', 'role': 'link/bond'},
        17: {'concept': 'boundary', 'category': 'relational', 'role': 'edge/limit'},
        19: {'concept': 'observer', 'category': 'dynamic', 'role': 'witness/measure'},
        23: {'concept': 'time', 'category': 'dynamic', 'role': 'sequence/duration'},
        29: {'concept': 'space', 'category': 'structural', 'role': 'extent/place'},
        31: {'concept': 'energy', 'category': 'dynamic', 'role': 'force/potential'},
        37: {'concept': 'information', 'category': 'structural', 'role': 'pattern/signal'},
        41: {'concept': 'pattern', 'category': 'structural', 'role': 'repeat/form'},
        43: {'concept': 'recursion', 'category': 'dynamic', 'role': 'self-reference'},
        47: {'concept': 'emergence', 'category': 'relational', 'role': 'arising/novelty'},
        53: {'concept': 'coherence', 'category': 'relational', 'role': 'unity/harmony'},
        59: {'concept': 'entropy', 'category': 'dynamic', 'role': 'disorder/spread'},
        61: {'concept': 'symmetry', 'category': 'structural', 'role': 'invariance'},
        67: {'concept': 'causation', 'category': 'relational', 'role': 'origin/effect'},
        71: {'concept': 'memory', 'category': 'structural', 'role': 'retention/trace'},
        73: {'concept': 'intention', 'category': 'dynamic', 'role': 'aim/purpose'},
        79: {'concept': 'context', 'category': 'relational', 'role': 'surround/frame'},
        83: {'concept': 'resonance', 'category': 'dynamic', 'role': 'vibration/echo'},
        89: {'concept': 'transformation', 'category': 'dynamic', 'role': 'change-of-form'},
        97: {'concept': 'closure', 'category': 'relational', 'role': 'complete/whole'},
        101: {'concept': 'consciousness', 'category': 'dynamic', 'role': 'awareness'},
        103: {'concept': 'meaning', 'category': 'relational', 'role': 'significance'},
        107: {'concept': 'truth', 'category': 'foundation', 'role': 'correspondence'},
        109: {'concept': 'beauty', 'category': 'structural', 'role': 'harmony/form'},
        113: {'concept': 'value', 'category': 'relational', 'role': 'worth/good'}
    },
    'adjectives': {
        2: {'concept': 'dual', 'category': 'foundation', 'intensifies': False},
        3: {'concept': 'structured', 'category': 'structural', 'intensifies': True},
        5: {'concept': 'dynamic', 'category': 'dynamic', 'intensifies': True},
        7: {'concept': 'essential', 'category': 'foundation', 'intensifies': False},
        11: {'concept': 'complex', 'category': 'structural', 'intensifies': True},
        13: {'concept': 'relational', 'category': 'relational', 'intensifies': False},
        17: {'concept': 'bounded', 'category': 'relational', 'intensifies': False},
        19: {'concept': 'observed', 'category': 'dynamic', 'intensifies': True},
        23: {'concept': 'temporal', 'category': 'dynamic', 'intensifies': False},
        29: {'concept': 'spatial', 'category': 'structural', 'intensifies': False},
        31: {'concept': 'energetic', 'category': 'dynamic', 'intensifies': True},
        37: {'concept': 'informational', 'category': 'structural', 'intensifies': True},
        41: {'concept': 'patterned', 'category': 'structural', 'intensifies': True},
        43: {'concept': 'recursive', 'category': 'dynamic', 'intensifies': True},
        47: {'concept': 'emergent', 'category': 'relational', 'intensifies': True},
        53: {'concept': 'coherent', 'category': 'relational', 'intensifies': True},
        59: {'concept': 'entropic', 'category': 'dynamic', 'intensifies': False},
        61: {'concept': 'symmetric', 'category': 'structural', 'intensifies': True},
        67: {'concept': 'causal', 'category': 'relational', 'intensifies': True},
        71: {'concept': 'remembered', 'category': 'structural', 'intensifies': False},
        73: {'concept': 'intentional', 'category': 'dynamic', 'intensifies': True},
        79: {'concept': 'contextual', 'category': 'relational', 'intensifies': False},
        83: {'concept': 'resonant', 'category': 'dynamic', 'intensifies': True},
        89: {'concept': 'transformative', 'category': 'dynamic', 'intensifies': True},
        97: {'concept': 'closed', 'category': 'relational', 'intensifies': False}
    }
}


def classify_prime(p: int) -> Dict[str, Any]:
    """
    Semantic category classification based on prime properties.
    Categories emerge from quadratic residue character.
    """
    mod4 = p % 4
    mod6 = p % 6
    
    # Primary classification
    if mod4 == 1:
        primary = 'structural'  # Can be expressed as sum of two squares
    elif mod4 == 3:
        primary = 'dynamic'  # Cannot be expressed as sum of two squares
    else:
        primary = 'foundation'  # p = 2
    
    # Secondary classification
    if mod6 == 1:
        secondary = 'relational'
    elif mod6 == 5:
        secondary = 'foundational'
    else:
        secondary = 'neutral'  # p = 2 or p = 3
    
    return {'primary': primary, 'secondary': secondary, 'mod4': mod4, 'mod6': mod6}


class ConceptInterpreter:
    """
    ConceptInterpreter - Maps primes to semantic concepts.
    Enhanced with PRQS Lexicon for richer semantic interpretation.
    """
    
    def __init__(self, lexicon: Dict = None):
        self.lexicon = lexicon or PRQS_LEXICON
        self.noun_concepts = {
            p: data['concept'] 
            for p, data in self.lexicon.get('nouns', {}).items()
        }
        self.adj_concepts = {
            p: data['concept']
            for p, data in self.lexicon.get('adjectives', {}).items()
        }
        self.noun_metadata = self.lexicon.get('nouns', {})
        self.adj_metadata = self.lexicon.get('adjectives', {})
    
    def get_category(self, prime: int) -> str:
        """Get semantic category for a prime."""
        if prime in self.noun_metadata:
            return self.noun_metadata[prime]['category']
        if prime in self.adj_metadata:
            return self.adj_metadata[prime]['category']
        
        # Derive from number-theoretic properties
        return classify_prime(prime)['primary']
    
    def get_role(self, prime: int) -> Optional[str]:
        """Get semantic role for a prime (if known)."""
        if prime in self.noun_metadata:
            return self.noun_metadata[prime].get('role')
        return None
    
    def interpret_noun(self, term: NounTerm) -> str:
        """Interpret a noun term as concept."""
        p = term.prime
        if p in self.noun_concepts:
            return self.noun_concepts[p]
        
        # Generate description based on prime properties
        category = classify_prime(p)
        return f"{category['primary']}_concept_{p}"
    
    def interpret_noun_full(self, term: NounTerm) -> Dict[str, Any]:
        """Interpret a noun term with full metadata."""
        p = term.prime
        concept = self.interpret_noun(term)
        category = self.get_category(p)
        role = self.get_role(p)
        classification = classify_prime(p)
        
        return {
            'prime': p,
            'concept': concept,
            'category': category,
            'role': role,
            'classification': classification,
            'is_core': p in self.noun_metadata
        }
    
    def interpret_adj(self, term: AdjTerm) -> str:
        """Interpret an adjective term as modifier."""
        p = term.prime
        if p in self.adj_concepts:
            return self.adj_concepts[p]
        
        category = classify_prime(p)
        return f"{category['primary']}_modifier_{p}"
    
    def is_intensifier(self, prime: int) -> bool:
        """Check if an adjective intensifies meaning."""
        if prime in self.adj_metadata:
            return self.adj_metadata[prime].get('intensifies', False)
        # Default: structural adjectives intensify
        return classify_prime(prime)['primary'] == 'structural'
    
    def interpret_chain(self, term: ChainTerm) -> str:
        """Interpret a chain as modified concept."""
        noun = self.interpret_noun(term.noun)
        adjs = [self.interpret_adj(op) for op in term.operators]
        
        # Build phrase: adj1 adj2 ... noun
        return ' '.join(adjs + [noun])
    
    def interpret(self, term: Term) -> str:
        """Full interpretation of any term."""
        if isinstance(term, NounTerm):
            return self.interpret_noun(term)
        if isinstance(term, AdjTerm):
            return self.interpret_adj(term)
        if isinstance(term, ChainTerm):
            return self.interpret_chain(term)
        if isinstance(term, FusionTerm):
            return f"fusion({term.p},{term.q},{term.r})"
        if isinstance(term, NounSentence):
            return f"[{self.interpret(term.expr)}]"
        if isinstance(term, SeqSentence):
            return f"{self.interpret(term.left)} and {self.interpret(term.right)}"
        if isinstance(term, ImplSentence):
            return f"if {self.interpret(term.antecedent)} then {self.interpret(term.consequent)}"
        
        return str(term)
    
    def add_noun_concept(self, prime: int, concept: str,
                         metadata: Optional[Dict[str, Any]] = None):
        """Add custom concept mappings."""
        self.noun_concepts[prime] = concept
        if metadata:
            self.noun_metadata[prime] = {'concept': concept, **metadata}
    
    def add_adj_concept(self, prime: int, concept: str,
                        metadata: Optional[Dict[str, Any]] = None):
        """Add custom adjective mappings."""
        self.adj_concepts[prime] = concept
        if metadata:
            self.adj_metadata[prime] = {'concept': concept, **metadata}
    
    def get_core_primes(self) -> Dict[str, List[int]]:
        """Get all core semantic primes (those with explicit mappings)."""
        nouns = list(self.noun_concepts.keys())
        adjs = list(self.adj_concepts.keys())
        return {'nouns': nouns, 'adjs': adjs, 'all': list(set(nouns + adjs))}
    
    def analyze_compatibility(self, p1: int, p2: int) -> Dict[str, Any]:
        """
        Analyze semantic compatibility between two primes.
        Compatible primes have complementary categories.
        """
        cat1 = classify_prime(p1)
        cat2 = classify_prime(p2)
        
        # Complementary categories have higher compatibility
        complementary = cat1['primary'] != cat2['primary']
        same_secondary = cat1['secondary'] == cat2['secondary']
        
        # Compute compatibility score
        score = 0.5  # Base score
        if complementary:
            score += 0.25  # Different primary = good
        if same_secondary:
            score += 0.15  # Same secondary = good
        if (p1 + p2) % 4 == 0:
            score += 0.1  # Sum divisible by 4
        
        return {
            'p1': p1, 'p2': p2,
            'cat1': cat1, 'cat2': cat2,
            'complementary': complementary,
            'same_secondary': same_secondary,
            'score': score,
            'interpretation': ('highly_compatible' if score > 0.7 else
                              'compatible' if score > 0.5 else
                              'weakly_compatible')
        }
    
    def interpret_fusion_semantic(self, p: int, q: int, r: int) -> Dict[str, Any]:
        """
        Generate semantic blend for a fusion.
        Fusion creates emergent meaning.
        """
        concepts = [
            self.noun_concepts.get(prime, f"concept_{prime}")
            for prime in [p, q, r]
        ]
        
        categories = [classify_prime(prime) for prime in [p, q, r]]
        
        # Count category types
        cat_counts = {'structural': 0, 'dynamic': 0, 'relational': 0, 'foundation': 0}
        for cat in categories:
            cat_counts[cat['primary']] = cat_counts.get(cat['primary'], 0) + 1
        
        # Dominant category
        dominant = max(cat_counts.items(), key=lambda x: x[1])[0]
        
        # Emergent meaning description
        emergent = f"{dominant}_fusion({'+'.join(concepts)})"
        
        return {
            'components': concepts,
            'categories': categories,
            'dominant': dominant,
            'fused_prime': p + q + r,
            'emergent': emergent,
            'description': f"Emergent {dominant} concept from {', '.join(concepts)}"
        }