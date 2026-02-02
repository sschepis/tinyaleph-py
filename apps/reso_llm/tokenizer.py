"""
ResoLLM Tokenizer.

Provides two tokenizer options:
1. ResoBPETokenizer: GPT-2 based BPE tokenizer (recommended for production)
2. ResoLLMTokenizer: Character-level prime tokenizer (legacy)

The BPE tokenizer uses HuggingFace's GPT-2 tokenizer for robust subword tokenization.
"""
import sys
import os
from typing import List, Dict, Optional

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

# Try to import HuggingFace tokenizers
try:
    from transformers import AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    HAS_TRANSFORMERS = False

# Legacy character-level tokenizer
from apps.resoformer.tokenizer import PrimeTokenizer, create_shakespeare_tokenizer, create_code_tokenizer


class ResoBPETokenizer:
    """
    BPE Tokenizer wrapper (GPT-2 based) compatible with ResoLLM.
    
    This is the recommended tokenizer for training and inference.
    Uses HuggingFace's GPT-2 tokenizer for robust subword tokenization.
    """
    
    def __init__(self, model_name: str = "gpt2"):
        if not HAS_TRANSFORMERS:
            raise ImportError(
                "transformers package required for ResoBPETokenizer. "
                "Install with: pip install transformers"
            )
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        # GPT-2 has no PAD token, set it to EOS
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Suppress token length warnings during dataset processing
        # We handle sequence length truncation manually in the dataset/model
        self.tokenizer.model_max_length = 1_000_000_000

        self.special_tokens = {
            "<PAD>": self.tokenizer.pad_token_id,
            "<UNK>": self.tokenizer.unk_token_id or self.tokenizer.eos_token_id,
            "<BOS>": self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
            "<EOS>": self.tokenizer.eos_token_id,
        }

        self.vocab_size = self.tokenizer.vocab_size
        
        # Compatibility attributes
        self.char_level = False
        self.vocab = {}  # Not used for BPE but provided for compatibility
        self.inverse_vocab = {}  # Not used for BPE but provided for compatibility

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text)
        if add_bos and self.special_tokens["<BOS>"] != self.special_tokens["<EOS>"]:
            ids.insert(0, self.special_tokens["<BOS>"])
        if add_eos:
            ids.append(self.special_tokens["<EOS>"])
        return ids

    def decode(self, tokens: List[int], skip_special: bool = True) -> str:
        """Decode token IDs to text."""
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)
    
    def batch_encode(self, texts: List[str], max_length: int = 512, padding: bool = True) -> Dict:
        """Batch encode texts with padding."""
        return self.tokenizer(
            texts, 
            max_length=max_length, 
            padding=padding, 
            truncation=True, 
            return_tensors="pt"
        )

    def save(self, path: str):
        """Save tokenizer to path."""
        self.tokenizer.save_pretrained(path)

    @classmethod
    def load(cls, path: str) -> 'ResoBPETokenizer':
        """Load tokenizer from path."""
        return cls(path)
    
    def prime_to_index(self, prime: int) -> Optional[int]:
        """Compatibility method for legacy code. Returns None for BPE tokenizer."""
        return None


class ResoLLMTokenizer(PrimeTokenizer):
    """
    Legacy character-level tokenizer for ResoLLM.
    
    Extends PrimeTokenizer with specific utilities for the Reso-LLM architecture.
    Use ResoBPETokenizer for production training.
    """
    pass


def create_bpe_tokenizer(model_name: str = "gpt2") -> ResoBPETokenizer:
    """Create a GPT-2 based BPE tokenizer (recommended)."""
    return ResoBPETokenizer(model_name)


def create_default_tokenizer() -> ResoBPETokenizer:
    """
    Create the default tokenizer for ResoLLM.
    
    Returns ResoBPETokenizer (GPT-2 based) if transformers is available,
    otherwise falls back to character-level tokenizer.
    """
    if HAS_TRANSFORMERS:
        return ResoBPETokenizer()
    else:
        # Fallback to character-level tokenizer
        chars = ''.join(chr(i) for i in range(32, 127)) + "\n"
        base = PrimeTokenizer.from_chars(chars)
        return ResoLLMTokenizer(
            vocab=base.vocab,
            inverse_vocab=base.inverse_vocab,
            special_tokens=base.special_tokens,
            char_level=base.char_level
        )


def create_char_tokenizer(vocab_size: int = 10000) -> ResoLLMTokenizer:
    """
    Create a character-level tokenizer (legacy, for small experiments).
    
    Args:
        vocab_size: Target vocab size. For compatibility with legacy models,
                   we create a tokenizer with at least this many tokens.
                   
    Returns:
        ResoLLMTokenizer instance
    """
    # Use extended ASCII plus common chars for a larger vocab
    chars = ''.join(chr(i) for i in range(32, 127)) + "\n"
    base = PrimeTokenizer.from_chars(chars)
    return ResoLLMTokenizer(
        vocab=base.vocab,
        inverse_vocab=base.inverse_vocab,
        special_tokens=base.special_tokens,
        char_level=base.char_level
    )


def create_tokenizer_for_vocab_size(vocab_size: int):
    """
    Create an appropriate tokenizer for a given vocab size.
    
    Args:
        vocab_size: The vocab size to match
        
    Returns:
        Tokenizer instance (BPE or character-level depending on vocab size)
    """
    if vocab_size == 50257:
        # GPT-2 vocab size - use BPE tokenizer
        return create_bpe_tokenizer()
    else:
        # Legacy vocab size - use character-level tokenizer
        return create_char_tokenizer(vocab_size)
