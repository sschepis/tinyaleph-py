"""
PrimeTokenizer: Text-to-Prime Mapping

Maps characters/tokens to prime numbers, leveraging the unique factorization
property of primes for semantic compositionality.

Key Features:
- Character-level or BPE-style tokenization
- Prime-indexed vocabulary (each token → unique prime)
- Special tokens: PAD, UNK, BOS, EOS mapped to small primes
- Composite number encoding for multi-token concepts
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple
from collections import Counter
import math
import sys
sys.path.insert(0, '../..')

from tinyaleph.core.primes import nth_prime, is_prime, prime_sieve


@dataclass
class PrimeTokenizer:
    """
    Tokenizer that maps tokens to prime numbers.
    
    Each unique token is assigned a unique prime, enabling:
    - Unique factorization for composed concepts
    - Prime-resonant attention weighting
    - Sparse representation in H_P
    
    Attributes:
        vocab: Dict mapping token -> prime
        inverse_vocab: Dict mapping prime -> token
        special_tokens: Dict of special token names -> primes
    """
    
    vocab: Dict[str, int] = field(default_factory=dict)
    inverse_vocab: Dict[int, str] = field(default_factory=dict)
    special_tokens: Dict[str, int] = field(default_factory=dict)
    char_level: bool = True
    
    # Special token constants
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    BOS_TOKEN = "<BOS>"
    EOS_TOKEN = "<EOS>"
    
    def __post_init__(self):
        """Initialize special tokens if vocab is empty."""
        if not self.vocab:
            self._init_special_tokens()
    
    def _init_special_tokens(self):
        """Assign small primes to special tokens."""
        # Use smallest primes for special tokens
        self.special_tokens = {
            self.PAD_TOKEN: 2,
            self.UNK_TOKEN: 3,
            self.BOS_TOKEN: 5,
            self.EOS_TOKEN: 7,
        }
        
        for token, prime in self.special_tokens.items():
            self.vocab[token] = prime
            self.inverse_vocab[prime] = token
    
    @classmethod
    def from_corpus(cls, texts: List[str], 
                    min_freq: int = 1,
                    max_vocab_size: Optional[int] = None,
                    char_level: bool = True) -> PrimeTokenizer:
        """
        Build tokenizer from corpus of texts.
        
        Args:
            texts: List of text strings
            min_freq: Minimum frequency for token inclusion
            max_vocab_size: Maximum vocabulary size (None for unlimited)
            char_level: If True, tokenize by character; else by whitespace
            
        Returns:
            Configured PrimeTokenizer
        """
        tokenizer = cls(char_level=char_level)
        
        # Count token frequencies
        counter = Counter()
        for text in texts:
            if char_level:
                tokens = list(text)
            else:
                tokens = text.split()
            counter.update(tokens)
        
        # Filter by frequency
        tokens = [t for t, c in counter.most_common() if c >= min_freq]
        
        # Limit vocab size
        if max_vocab_size is not None:
            # Reserve 4 slots for special tokens
            tokens = tokens[:max_vocab_size - 4]
        
        # Assign primes (starting after special tokens)
        next_prime_idx = 5  # Skip 2, 3, 5, 7 (special tokens)
        for token in tokens:
            if token not in tokenizer.vocab:
                prime = nth_prime(next_prime_idx)
                tokenizer.vocab[token] = prime
                tokenizer.inverse_vocab[prime] = token
                next_prime_idx += 1
        
        return tokenizer
    
    @classmethod
    def from_chars(cls, chars: str) -> PrimeTokenizer:
        """
        Create tokenizer from explicit character set.
        
        Args:
            chars: String of characters to include
            
        Returns:
            Configured PrimeTokenizer
        """
        tokenizer = cls(char_level=True)
        
        # Assign primes to each character
        next_prime_idx = 5  # Skip special tokens
        for char in chars:
            if char not in tokenizer.vocab:
                prime = nth_prime(next_prime_idx)
                tokenizer.vocab[char] = prime
                tokenizer.inverse_vocab[prime] = char
                next_prime_idx += 1
        
        return tokenizer
    
    @classmethod
    def default_ascii(cls) -> PrimeTokenizer:
        """Create tokenizer for printable ASCII characters."""
        chars = ''.join(chr(i) for i in range(32, 127))
        return cls.from_chars(chars)
    
    @property
    def vocab_size(self) -> int:
        """Return vocabulary size."""
        return len(self.vocab)
    
    @property
    def max_prime(self) -> int:
        """Return largest prime in vocabulary."""
        return max(self.vocab.values()) if self.vocab else 2
    
    def encode(self, text: str, 
               add_bos: bool = False,
               add_eos: bool = False) -> List[int]:
        """
        Encode text to list of prime indices.
        
        Args:
            text: Input text string
            add_bos: Add BOS token at start
            add_eos: Add EOS token at end
            
        Returns:
            List of prime numbers
        """
        if self.char_level:
            tokens = list(text)
        else:
            tokens = text.split()
        
        primes = []
        
        if add_bos:
            primes.append(self.special_tokens[self.BOS_TOKEN])
        
        for token in tokens:
            if token in self.vocab:
                primes.append(self.vocab[token])
            else:
                primes.append(self.special_tokens[self.UNK_TOKEN])
        
        if add_eos:
            primes.append(self.special_tokens[self.EOS_TOKEN])
        
        return primes
    
    def decode(self, primes: List[int], 
               skip_special: bool = True) -> str:
        """
        Decode list of primes to text.
        
        Args:
            primes: List of prime numbers
            skip_special: If True, skip special tokens in output
            
        Returns:
            Decoded text string
        """
        tokens = []
        special_primes = set(self.special_tokens.values())
        
        for prime in primes:
            if skip_special and prime in special_primes:
                continue
            
            if prime in self.inverse_vocab:
                tokens.append(self.inverse_vocab[prime])
            else:
                tokens.append(self.UNK_TOKEN)
        
        if self.char_level:
            return ''.join(tokens)
        else:
            return ' '.join(tokens)
    
    def encode_batch(self, texts: List[str],
                     max_length: Optional[int] = None,
                     padding: bool = True,
                     truncation: bool = True) -> Tuple[List[List[int]], List[int]]:
        """
        Encode batch of texts with optional padding/truncation.
        
        Args:
            texts: List of text strings
            max_length: Maximum sequence length
            padding: Pad shorter sequences
            truncation: Truncate longer sequences
            
        Returns:
            Tuple of (encoded_sequences, lengths)
        """
        encoded = [self.encode(text, add_bos=True, add_eos=True) for text in texts]
        lengths = [len(seq) for seq in encoded]
        
        if max_length is None:
            max_length = max(lengths)
        
        pad_prime = self.special_tokens[self.PAD_TOKEN]
        
        result = []
        for seq in encoded:
            if truncation and len(seq) > max_length:
                seq = seq[:max_length]
            
            if padding and len(seq) < max_length:
                seq = seq + [pad_prime] * (max_length - len(seq))
            
            result.append(seq)
        
        return result, lengths
    
    def prime_to_index(self, prime: int) -> int:
        """
        Convert prime to 0-indexed vocabulary index.
        
        Useful for embedding lookups.
        """
        # Sort vocab by prime
        sorted_primes = sorted(self.vocab.values())
        if prime in sorted_primes:
            return sorted_primes.index(prime)
        return 1  # UNK index
    
    def index_to_prime(self, index: int) -> int:
        """Convert vocabulary index to prime."""
        sorted_primes = sorted(self.vocab.values())
        if 0 <= index < len(sorted_primes):
            return sorted_primes[index]
        return 3  # UNK prime
    
    def save(self, path: str):
        """Save tokenizer to file."""
        import json
        data = {
            'vocab': self.vocab,
            'special_tokens': self.special_tokens,
            'char_level': self.char_level,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load(cls, path: str) -> PrimeTokenizer:
        """Load tokenizer from file."""
        import json
        with open(path, 'r') as f:
            data = json.load(f)
        
        tokenizer = cls(
            vocab=data['vocab'],
            special_tokens=data['special_tokens'],
            char_level=data.get('char_level', True),
        )
        
        # Rebuild inverse vocab
        tokenizer.inverse_vocab = {v: k for k, v in tokenizer.vocab.items()}
        
        return tokenizer
    
    def __repr__(self) -> str:
        return f"PrimeTokenizer(vocab_size={self.vocab_size}, max_prime={self.max_prime})"


def create_shakespeare_tokenizer() -> PrimeTokenizer:
    """
    Create tokenizer suitable for Shakespeare-like text.
    
    Includes ASCII letters, common punctuation, newlines.
    """
    chars = (
        "abcdefghijklmnopqrstuvwxyz"
        "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        "0123456789"
        " .,!?;:'-\"\n"
    )
    return PrimeTokenizer.from_chars(chars)


def create_code_tokenizer() -> PrimeTokenizer:
    """
    Create tokenizer suitable for source code.
    
    Includes all printable ASCII plus common escapes.
    """
    chars = ''.join(chr(i) for i in range(32, 127)) + '\n\t'
    return PrimeTokenizer.from_chars(chars)