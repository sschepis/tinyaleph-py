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


# Chat template special tokens used in training data
CHAT_SPECIAL_TOKENS = [
    "<|user|>",
    "<|endofuser|>",
    "<|assistant|>",
    "<|endofassistant|>",
    "<|system|>",
    "<|endofsystem|>",
    "<|endofconversation|>",
]


class ResoBPETokenizer:
    """
    BPE Tokenizer wrapper (GPT-2 based) compatible with ResoLLM.
    
    This is the recommended tokenizer for training and inference.
    Uses HuggingFace's GPT-2 tokenizer for robust subword tokenization.
    
    IMPORTANT: Chat template special tokens (<|user|>, <|assistant|>, etc.)
    are added to the vocabulary so they tokenize as single tokens, preventing
    fragmentation during training and inference.
    """
    
    def __init__(self, model_name: str = "gpt2", add_chat_tokens: bool = True):
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
        
        # Add chat template special tokens to prevent fragmentation
        # These tokens will be tokenized as single units, not broken up
        if add_chat_tokens:
            num_added = self.tokenizer.add_tokens(CHAT_SPECIAL_TOKENS, special_tokens=True)
            if num_added > 0:
                # Note: Model embeddings will need to be resized to match new vocab
                pass

        self.special_tokens = {
            "<PAD>": self.tokenizer.pad_token_id,
            "<UNK>": self.tokenizer.unk_token_id or self.tokenizer.eos_token_id,
            "<BOS>": self.tokenizer.bos_token_id or self.tokenizer.eos_token_id,
            "<EOS>": self.tokenizer.eos_token_id,
        }
        
        # Add chat tokens to special_tokens dict for easy access
        for token in CHAT_SPECIAL_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                self.special_tokens[token] = token_id

        self.vocab_size = len(self.tokenizer)  # Use len() to get updated vocab size with added tokens
        
        # Compatibility attributes
        self.char_level = False
        self.vocab = {}  # Not used for BPE but provided for compatibility
        self.inverse_vocab = {}  # Not used for BPE but provided for compatibility
        
        # Store chat token IDs for efficient lookup
        self._chat_token_ids = set()
        for token in CHAT_SPECIAL_TOKENS:
            token_id = self.tokenizer.convert_tokens_to_ids(token)
            if token_id != self.tokenizer.unk_token_id:
                self._chat_token_ids.add(token_id)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> List[int]:
        """Encode text to token IDs."""
        ids = self.tokenizer.encode(text)
        if add_bos and self.special_tokens["<BOS>"] != self.special_tokens["<EOS>"]:
            ids.insert(0, self.special_tokens["<BOS>"])
        if add_eos:
            ids.append(self.special_tokens["<EOS>"])
        return ids

    def decode(self, tokens: List[int], skip_special: bool = False) -> str:
        """
        Decode token IDs to text.
        
        Note: skip_special defaults to False to preserve chat template tokens
        in the decoded output for training data inspection.
        """
        return self.tokenizer.decode(tokens, skip_special_tokens=skip_special)
    
    def is_chat_token(self, token_id: int) -> bool:
        """Check if a token ID is a chat template special token."""
        return token_id in self._chat_token_ids
    
    def get_chat_token_id(self, token: str) -> Optional[int]:
        """Get the token ID for a chat template token, or None if not found."""
        return self.special_tokens.get(token)
    
    def find_conversation_boundaries(self, token_ids: List[int]) -> List[tuple]:
        """
        Find the start and end indices of complete conversations in a token sequence.
        
        A complete conversation starts with <|user|> and ends with <|endofassistant|>.
        
        Returns:
            List of (start_idx, end_idx) tuples for each complete conversation
        """
        user_token = self.special_tokens.get("<|user|>")
        endofassistant_token = self.special_tokens.get("<|endofassistant|>")
        
        if user_token is None or endofassistant_token is None:
            return []
        
        boundaries = []
        current_start = None
        
        for i, token_id in enumerate(token_ids):
            if token_id == user_token and current_start is None:
                current_start = i
            elif token_id == endofassistant_token and current_start is not None:
                boundaries.append((current_start, i + 1))
                current_start = None  # Look for next conversation
        
        return boundaries
    
    def extract_complete_conversations(self, token_ids: List[int], max_count: int = 5) -> List[str]:
        """
        Extract complete conversations from a token sequence.
        
        Returns decoded text for up to max_count complete conversations.
        """
        boundaries = self.find_conversation_boundaries(token_ids)
        conversations = []
        
        for start, end in boundaries[:max_count]:
            conv_tokens = token_ids[start:end]
            conv_text = self.decode(conv_tokens, skip_special=False)
            conversations.append(conv_text)
        
        return conversations
    
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


def create_bpe_tokenizer(model_name: str = "gpt2", add_chat_tokens: bool = True) -> ResoBPETokenizer:
    """
    Create a GPT-2 based BPE tokenizer (recommended).
    
    Args:
        model_name: HuggingFace model name for tokenizer
        add_chat_tokens: Whether to add chat template special tokens.
                        Set to False for models trained without these tokens.
    """
    return ResoBPETokenizer(model_name, add_chat_tokens=add_chat_tokens)


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


# GPT-2 vocab size (50257) + 7 chat special tokens = 50264
GPT2_BASE_VOCAB_SIZE = 50257
GPT2_EXTENDED_VOCAB_SIZE = GPT2_BASE_VOCAB_SIZE + len(CHAT_SPECIAL_TOKENS)


def create_tokenizer_for_vocab_size(vocab_size: int):
    """
    Create an appropriate tokenizer for a given vocab size.
    
    Args:
        vocab_size: The vocab size to match
        
    Returns:
        Tokenizer instance (BPE or character-level depending on vocab size)
    """
    if vocab_size in (GPT2_BASE_VOCAB_SIZE, GPT2_EXTENDED_VOCAB_SIZE):
        # GPT-2 vocab size (with or without chat tokens) - use BPE tokenizer
        return create_bpe_tokenizer()
    else:
        # Legacy vocab size - use character-level tokenizer
        return create_char_tokenizer(vocab_size)
