"""
Dataset handling for Reso-LLM training.

Handles loading text data, tokenizing with PrimeTokenizer, and creating
batches for training.

Supports:
- Local text files
- HuggingFace datasets with automatic format detection
- Multiple datasets combined for training (MultiDataset)

Supported HuggingFace Datasets:
- timdettmers/openassistant-guanaco (### Human: / ### Assistant:)
- databricks/databricks-dolly-15k (instruction/context/response)
- Open-Orca/OpenOrca (system_prompt/question/response)
- tatsu-lab/alpaca (instruction/input/output)
- WizardLM/WizardLM_evol_instruct_70k (instruction/input/output)
- Anthropic/hh-rlhf (chosen/rejected conversations)
- OpenAssistant/oasst2 (text/role tree structure)
- lmsys/chatbot_arena_conversations (conversation_a/b)
- sahil2801/CodeAlpaca-20k (code instruction/input/output)
- TokenBender/code_instructions_122k_alpaca_style (code instruction/input/output)
- knkarthick/dialogsum (dialogue/summary)
- lonestar108/sexygpt (user/assistant pairs)
- lonestar108/enlightenedllm (instruction/output pairs)
- lonestar108/rawdata (input/output pairs)
- Any dataset with instruction/output, prompt/response, input/output, or question/answer pairs

Unified Format:
All datasets are converted to match the chat template in inference.py:
    <|user|>
    {question/prompt}
    <|endofuser|>
    <|assistant|>
    {answer/response}
    <|endofassistant|>

This ensures training data matches the inference format exactly.

Multi-Dataset Support:
Use MultiDataset to load and combine multiple datasets:

    from apps.reso_llm.dataset import MultiDataset
    
    multi_ds = MultiDataset(
        dataset_names=[
            "timdettmers/openassistant-guanaco",
            "databricks/databricks-dolly-15k",
            "tatsu-lab/alpaca"
        ],
        tokenizer=tokenizer,
        seq_len=256,
        batch_size=32
    )
    
    # Validate combined output
    report = multi_ds.validate()
    
    # Get sample conversations
    samples = multi_ds.get_sample_conversations(n=5)
"""
import sys
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Generator, Optional, Dict

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.tokenizer import ResoLLMTokenizer
from tinyaleph.ml.resoformer import Tensor


@dataclass
class DatasetStats:
    """Statistics for dataset loading and formatting."""
    total_examples: int = 0
    valid_examples: int = 0
    skipped_empty: int = 0
    skipped_no_user: int = 0
    skipped_no_assistant: int = 0
    skipped_blank_parts: int = 0
    skipped_language_filter: int = 0
    
    def summary(self) -> str:
        """Return a summary of dataset statistics."""
        valid_pct = (self.valid_examples / max(1, self.total_examples)) * 100
        lines = [
            f"Dataset Statistics:",
            f"  Total examples processed: {self.total_examples:,}",
            f"  Valid examples: {self.valid_examples:,} ({valid_pct:.1f}%)",
            f"  Skipped (empty): {self.skipped_empty:,}",
            f"  Skipped (no <|user|> markers): {self.skipped_no_user:,}",
            f"  Skipped (no <|assistant|> markers): {self.skipped_no_assistant:,}",
            f"  Skipped (blank question/answer): {self.skipped_blank_parts:,}",
        ]
        if self.skipped_language_filter > 0:
            lines.append(f"  Skipped (language filter): {self.skipped_language_filter:,}")
        return "\n".join(lines)


def validate_format(text: str, require_newlines: bool = False) -> Tuple[bool, str]:
    """
    Validate that text follows the chat template format.
    
    Expected format:
        <|user|>
        {content}
        <|endofuser|>
        <|assistant|>
        {content}
        <|endofassistant|>
    
    Args:
        text: Formatted text to validate
        require_newlines: If True, require newlines around markers (ignored, always required)
        
    Returns:
        (is_valid, reason) tuple
    """
    if not text or not text.strip():
        return False, "empty"
    
    text = text.strip()
    
    # Check for user markers
    if "<|user|>" not in text or "<|endofuser|>" not in text:
        return False, "no_user"
    
    # Check for assistant markers
    if "<|assistant|>" not in text or "<|endofassistant|>" not in text:
        return False, "no_assistant"
    
    # Extract and validate parts are non-empty
    try:
        # Find marker positions
        user_start = text.find("<|user|>")
        user_end = text.find("<|endofuser|>")
        assistant_start = text.find("<|assistant|>")
        assistant_end = text.find("<|endofassistant|>")
        
        # Correct order: user_start < user_end < assistant_start < assistant_end
        if not (user_start < user_end < assistant_start < assistant_end):
            return False, "wrong_order"
        
        # Extract user content
        user_content = text[user_start + len("<|user|>"):user_end].strip()
        
        # Extract assistant content
        assistant_content = text[assistant_start + len("<|assistant|>"):assistant_end].strip()
        
        # Check for blank parts
        if not user_content:
            return False, "blank_user"
        if not assistant_content:
            return False, "blank_assistant"
        
        return True, "valid"
        
    except Exception:
        return False, "parse_error"


# Default dataset for Reso-LLM training
DEFAULT_DATASET = "timdettmers/openassistant-guanaco"


class TextDataset:
    """
    Simple text dataset loader.
    """
    
    def __init__(self, 
                 file_path: str, 
                 tokenizer: ResoLLMTokenizer, 
                 seq_len: int = 128,
                 batch_size: int = 32):
        self.file_path = file_path
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        
        self.data: List[int] = []
        self._load_data()
        
    def _load_data(self):
        """Load and tokenize data."""
        if not os.path.exists(self.file_path):
            # Create dummy data if file doesn't exist (for testing)
            print(f"File {self.file_path} not found. Creating dummy data.")
            dummy_text = "The quick brown fox jumps over the lazy dog. " * 1000
            self.data = self.tokenizer.encode(dummy_text)
        else:
            with open(self.file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            self.data = self.tokenizer.encode(text)
            
        print(f"Loaded dataset with {len(self.data)} tokens.")
        
    def __len__(self) -> int:
        """Number of batches per epoch."""
        # We can extract (len(data) - 1) // seq_len sequences
        # Then divide by batch_size
        n_sequences = (len(self.data) - 1) // self.seq_len
        return max(1, n_sequences // self.batch_size)
        
    def get_batch(self) -> Tuple[Tensor, Tensor]:
        """
        Get a random batch of data.
        
        Returns:
            x: Input tensor (batch_size, seq_len)
            y: Target tensor (batch_size, seq_len) - shifted by 1
        """
        # Random sampling
        max_idx = len(self.data) - self.seq_len - 1
        
        batch_x = []
        batch_y = []
        
        for _ in range(self.batch_size):
            idx = random.randint(0, max_idx)
            x_seq = self.data[idx : idx + self.seq_len]
            y_seq = self.data[idx + 1 : idx + self.seq_len + 1]
            
            batch_x.extend(x_seq)
            batch_y.extend(y_seq)
            
        # Create tensors with correct shape
        x_tensor = Tensor(batch_x, (self.batch_size, self.seq_len))
        y_tensor = Tensor(batch_y, (self.batch_size, self.seq_len))
        
        return x_tensor, y_tensor
    
    def iterate_batches(self) -> Generator[Tuple[Tensor, Tensor], None, None]:
        """Iterate over all batches sequentially (for validation)."""
        n_sequences = (len(self.data) - 1) // self.seq_len
        n_batches = n_sequences // self.batch_size
        
        for i in range(n_batches):
            start_seq_idx = i * self.batch_size
            
            batch_x = []
            batch_y = []
            
            for j in range(self.batch_size):
                seq_idx = start_seq_idx + j
                # Calculate token index
                idx = seq_idx * self.seq_len
                
                if idx + self.seq_len + 1 > len(self.data):
                    break
                    
                x_seq = self.data[idx : idx + self.seq_len]
                y_seq = self.data[idx + 1 : idx + self.seq_len + 1]
                
                batch_x.extend(x_seq)
                batch_y.extend(y_seq)
            
            if len(batch_x) == self.batch_size * self.seq_len:
                x_tensor = Tensor(batch_x, (self.batch_size, self.seq_len))
                y_tensor = Tensor(batch_y, (self.batch_size, self.seq_len))
                yield x_tensor, y_tensor


# Datasets that require English-only filtering
ENGLISH_FILTER_DATASETS = {
    "OpenAssistant/oasst1",
    "OpenAssistant/oasst2",
}


class HuggingFaceDataset(TextDataset):
    """
    Dataset loader for HuggingFace datasets.
    
    Supports multiple conversation and instruction formats:
    
    Instruction-following datasets:
    - databricks/databricks-dolly-15k (instruction/context/response)
    - Open-Orca/OpenOrca (system_prompt/question/response)
    - tatsu-lab/alpaca (instruction/input/output)
    - WizardLM/WizardLM_evol_instruct_70k (instruction/input/output)
    
    Conversation datasets:
    - timdettmers/openassistant-guanaco (### Human: / ### Assistant:)
    - Anthropic/hh-rlhf (chosen/rejected conversations)
    - OpenAssistant/oasst2 (text/role tree structure) - auto-filters to English
    - lmsys/chatbot_arena_conversations (conversation_a/b)
    - lonestar108/sexygpt (user/assistant pairs)
    
    Code instruction datasets:
    - sahil2801/CodeAlpaca-20k (instruction/input/output)
    - TokenBender/code_instructions_122k_alpaca_style (instruction/input/output)
    
    Other formats:
    - knkarthick/dialogsum (dialogue/summary)
    - Any instruction/output, prompt/response, or question/answer pairs
    
    All formats are unified to the chat template format:
        <|user|>
        {question/prompt}
        <|endofuser|>
        <|assistant|>
        {answer/response}
        <|endofassistant|>
    
    Validation ensures proper structure with non-blank content.
    
    Language Filtering:
    - OpenAssistant datasets (oasst1, oasst2) automatically filter to lang="en"
    - Use language_filter parameter to override or apply to other datasets
    """
    
    def __init__(self,
                 dataset_name: str = DEFAULT_DATASET,
                 tokenizer: ResoLLMTokenizer = None,
                 seq_len: int = 128,
                 batch_size: int = 32,
                 split: str = "train",
                 max_tokens: int = 10_000_000,
                 validate_format: bool = True,
                 language_filter: Optional[str] = None):
        """
        Initialize a HuggingFace dataset loader.
        
        Args:
            dataset_name: HuggingFace dataset name or path
            tokenizer: Tokenizer to use for encoding
            seq_len: Sequence length for training
            batch_size: Batch size
            split: Dataset split to use (default: "train")
            max_tokens: Maximum tokens to load
            validate_format: Whether to validate output format
            language_filter: Language code to filter by (e.g., "en").
                           Auto-set to "en" for OpenAssistant datasets.
        """
        self.dataset_name = dataset_name
        self.split = split
        self.max_tokens = max_tokens
        self.validate_format = validate_format
        
        # Auto-set language filter for OpenAssistant datasets
        if language_filter is None and dataset_name in ENGLISH_FILTER_DATASETS:
            self.language_filter = "en"
        else:
            self.language_filter = language_filter
            
        # Don't call super().__init__ with file path - we'll load differently
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.data: List[int] = []
        self.stats: DatasetStats = DatasetStats()
        self._load_data()

    def _format_guanaco_example(self, text: str) -> str:
        """
        Format timdettmers/openassistant-guanaco examples.
        
        The Guanaco dataset uses format:
        ### Human: <question>
        ### Assistant: <answer>
        
        We convert to the chat template format matching inference.py:
        <|user|>
        {question}
        <|endofuser|>
        <|assistant|>
        {answer}
        <|endofassistant|>
        """
        if not isinstance(text, str):
            return ""
        
        import re
        
        # Split on ### Human: and ### Assistant: markers
        # Pattern: ### Human: <content> ### Assistant: <content>
        parts = re.split(r'###\s*(Human|Assistant):\s*', text.strip())
        
        # parts will be like: ['', 'Human', '<question>', 'Assistant', '<answer>', ...]
        result_parts = []
        i = 1  # Skip first empty part
        while i < len(parts) - 1:
            role = parts[i].strip().lower()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            
            if role == 'human' and content:
                result_parts.append(f"<|user|>\n{content}\n<|endofuser|>")
            elif role == 'assistant' and content:
                result_parts.append(f"<|assistant|>\n{content}\n<|endofassistant|>")
            
            i += 2
        
        return "\n".join(result_parts) if result_parts else ""

    def _format_dialogsum_example(self, item: dict) -> str:
        """Format knkarthick/dialogsum examples using chat template."""
        dialogue = item.get("dialogue", "")
        summary = item.get("summary", "")
        if not isinstance(dialogue, str) or not isinstance(summary, str):
            return ""
        dialogue = dialogue.strip()
        summary = summary.strip()
        if not dialogue or not summary:
            return ""
        return (
            f"<|user|>\nSummarize the following dialogue.\n\nDialog:\n{dialogue}\n<|endofuser|>\n"
            f"<|assistant|>\n{summary}\n<|endofassistant|>"
        )

    def _format_dolly_example(self, item: dict) -> str:
        """
        Format databricks/databricks-dolly-15k examples.
        
        Dolly has: instruction, context (optional), response, category
        """
        instruction = item.get("instruction", "").strip()
        context = item.get("context", "").strip()
        response = item.get("response", "").strip()
        
        if not instruction or not response:
            return ""
        
        # Include context if present
        if context:
            prompt = f"{instruction}\n\nContext:\n{context}"
        else:
            prompt = instruction
            
        return (
            f"<|user|>\n{prompt}\n<|endofuser|>\n"
            f"<|assistant|>\n{response}\n<|endofassistant|>"
        )
    
    def _format_orca_example(self, item: dict) -> str:
        """
        Format Open-Orca/OpenOrca examples.
        
        OpenOrca has: system_prompt, question, response
        """
        system_prompt = item.get("system_prompt", "").strip()
        question = item.get("question", "").strip()
        response = item.get("response", "").strip()
        
        if not question or not response:
            return ""
        
        # Include system prompt in the user message if present
        if system_prompt and system_prompt != "You are an AI assistant.":
            prompt = f"[System: {system_prompt}]\n\n{question}"
        else:
            prompt = question
            
        return (
            f"<|user|>\n{prompt}\n<|endofuser|>\n"
            f"<|assistant|>\n{response}\n<|endofassistant|>"
        )
    
    def _format_alpaca_example(self, item: dict) -> str:
        """
        Format tatsu-lab/alpaca and similar examples.
        
        Alpaca has: instruction, input (optional), output
        """
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        
        if not instruction or not output:
            return ""
        
        # Include input if present
        if input_text:
            prompt = f"{instruction}\n\nInput:\n{input_text}"
        else:
            prompt = instruction
            
        return (
            f"<|user|>\n{prompt}\n<|endofuser|>\n"
            f"<|assistant|>\n{output}\n<|endofassistant|>"
        )
    
    def _format_hh_rlhf_example(self, item: dict) -> str:
        """
        Format Anthropic/hh-rlhf examples.
        
        hh-rlhf has: chosen, rejected (we use chosen for training)
        Format is: Human: ... Assistant: ... Human: ... Assistant: ...
        """
        chosen = item.get("chosen", "").strip()
        if not chosen:
            return ""
        
        import re
        # Split on Human: and Assistant: markers
        parts = re.split(r'\n\n(Human|Assistant):\s*', chosen)
        
        result_parts = []
        i = 1  # Skip first empty part
        while i < len(parts) - 1:
            role = parts[i].strip().lower()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            
            if role == 'human' and content:
                result_parts.append(f"<|user|>\n{content}\n<|endofuser|>")
            elif role == 'assistant' and content:
                result_parts.append(f"<|assistant|>\n{content}\n<|endofassistant|>")
            
            i += 2
        
        return "\n".join(result_parts) if result_parts else ""
    
    def _format_oasst_example(self, item: dict, dataset=None, idx: int = 0) -> str:
        """
        Format OpenAssistant/oasst2 examples.
        
        oasst2 has a tree structure where each row is a single message.
        We need to find parent-child pairs to form complete conversations.
        
        For efficiency, we only process assistant messages and look up their parent (prompter).
        
        Note: Language filtering is applied at the caller level using self.language_filter.
        This method is called only for items that pass the language filter.
        """
        text = item.get("text", "").strip()
        role = item.get("role", "").strip().lower()
        parent_id = item.get("parent_id")
        
        if not text:
            return ""
        
        # Only process assistant messages that have a parent
        if role == "assistant" and parent_id and dataset is not None:
            # Try to find the parent message (should be a prompter)
            # This is expensive, but oasst2 is structured this way
            try:
                # Look for parent in nearby messages (tree structure usually keeps them close)
                search_range = min(100, len(dataset))
                start_idx = max(0, idx - search_range)
                
                for i in range(start_idx, min(idx + search_range, len(dataset))):
                    if i == idx:
                        continue
                    candidate = dataset[i]
                    if candidate.get("message_id") == parent_id:
                        # Check that parent also passes language filter
                        if self.language_filter:
                            parent_lang = candidate.get("lang", "").strip().lower()
                            if parent_lang != self.language_filter.lower():
                                return ""
                        
                        parent_text = candidate.get("text", "").strip()
                        parent_role = candidate.get("role", "").strip().lower()
                        
                        if parent_text and parent_role == "prompter":
                            return (
                                f"<|user|>\n{parent_text}\n<|endofuser|>\n"
                                f"<|assistant|>\n{text}\n<|endofassistant|>"
                            )
                        break
            except Exception:
                pass
            return ""
        
        # Skip prompter messages (they'll be included when we process their child assistant)
        return ""
    
    def _format_chatbot_arena_example(self, item: dict) -> str:
        """
        Format lmsys/chatbot_arena_conversations examples.
        
        Has conversation_a, conversation_b with turns.
        """
        # Use conversation_a as the primary conversation
        conversation = item.get("conversation_a", [])
        if not conversation:
            conversation = item.get("conversation_b", [])
        
        if not conversation:
            return ""
        
        result_parts = []
        for turn in conversation:
            role = turn.get("role", "").lower()
            content = turn.get("content", "").strip()
            
            if not content:
                continue
                
            if role in ("user", "human"):
                result_parts.append(f"<|user|>\n{content}\n<|endofuser|>")
            elif role in ("assistant", "model"):
                result_parts.append(f"<|assistant|>\n{content}\n<|endofassistant|>")
        
        return "\n".join(result_parts) if result_parts else ""
    
    def _format_code_alpaca_example(self, item: dict) -> str:
        """
        Format sahil2801/CodeAlpaca-20k and TokenBender/code_instructions examples.
        
        Uses instruction, input (optional), output format.
        """
        instruction = item.get("instruction", "").strip()
        input_text = item.get("input", "").strip()
        output = item.get("output", "").strip()
        
        if not instruction or not output:
            return ""
        
        # Include input if present
        if input_text:
            prompt = f"{instruction}\n\n```\n{input_text}\n```"
        else:
            prompt = instruction
            
        return (
            f"<|user|>\n{prompt}\n<|endofuser|>\n"
            f"<|assistant|>\n{output}\n<|endofassistant|>"
        )

    def _format_pair_example(self, item: dict, prompt_key: str, response_key: str) -> str:
        """
        Format instruction/output or prompt/response pairs.
        
        Uses the chat template format matching inference.py:
        <|user|>
        {prompt}
        <|endofuser|>
        <|assistant|>
        {response}
        <|endofassistant|>
        """
        prompt = item.get(prompt_key, "")
        response = item.get(response_key, "")
        if not isinstance(prompt, str) or not isinstance(response, str):
            return ""
        prompt = prompt.strip()
        response = response.strip()
        if not prompt or not response:
            return ""
        return (
            f"<|user|>\n{prompt}\n<|endofuser|>\n"
            f"<|assistant|>\n{response}\n<|endofassistant|>"
        )
        
    def _load_data(self):
        """Load and tokenize data from HuggingFace."""
        try:
            from datasets import load_dataset
            
            print(f"Loading dataset '{self.dataset_name}' ({self.split})...")
            
            # Try to load dataset, with fallback for schema mismatch errors
            try:
                dataset = load_dataset(self.dataset_name, split=self.split)
            except Exception as schema_error:
                error_str = str(schema_error).lower()
                error_type = type(schema_error).__name__
                
                # Check underlying cause if available (often wrapped in DatasetGenerationError)
                cause_str = ""
                cause_type = ""
                if hasattr(schema_error, '__cause__') and schema_error.__cause__:
                    cause_str = str(schema_error.__cause__).lower()
                    cause_type = type(schema_error.__cause__).__name__.lower()
                
                is_schema_error = (
                    "column names don't match" in error_str or
                    "casterror" in error_type.lower() or
                    "couldn't cast" in error_str or
                    "schema" in error_str or
                    "column names don't match" in cause_str or
                    "casterror" in cause_type or
                    "couldn't cast" in cause_str or
                    "schema" in cause_str
                )
                if is_schema_error:
                    # Schema mismatch - try loading from cached parquet directly
                    print(f"  Schema mismatch detected, trying to load from cached parquet...")
                    try:
                        import glob
                        from pathlib import Path
                        
                        # Find the cached parquet file
                        cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
                        dataset_cache_name = f"datasets--{self.dataset_name.replace('/', '--')}"
                        dataset_cache_dir = cache_dir / dataset_cache_name
                        
                        # Find parquet files in the cache
                        parquet_files = list(dataset_cache_dir.glob("**/data/*.parquet"))
                        
                        if parquet_files:
                            parquet_file = str(parquet_files[0])
                            print(f"  Found cached parquet: {parquet_file}")
                            # Load directly without schema enforcement
                            dataset = load_dataset("parquet", data_files=parquet_file, split="train")
                            print(f"  ✓ Loaded from cached parquet file")
                        else:
                            # Try downloading fresh via huggingface_hub
                            from huggingface_hub import hf_hub_download, list_repo_files
                            
                            # List files in the dataset repo to find parquet files
                            files = list_repo_files(self.dataset_name, repo_type="dataset")
                            parquet_files = [f for f in files if f.endswith('.parquet')]
                            
                            if parquet_files:
                                parquet_file = hf_hub_download(
                                    repo_id=self.dataset_name,
                                    filename=parquet_files[0],
                                    repo_type="dataset"
                                )
                                dataset = load_dataset("parquet", data_files=parquet_file, split="train")
                                print(f"  ✓ Loaded from downloaded parquet file")
                            else:
                                raise schema_error
                    except Exception as e2:
                        print(f"  Alternative load methods failed: {e2}")
                        raise schema_error
                else:
                    raise schema_error
            
            print(f"Dataset columns: {dataset.column_names}")
            print(f"Dataset size: {len(dataset)} examples")
            
            # Detect dataset format
            columns = dataset.column_names
            
            # timdettmers/openassistant-guanaco has a "text" column with ### Human: / ### Assistant: format
            is_guanaco = self.dataset_name == "timdettmers/openassistant-guanaco" or (
                "text" in columns and
                len(dataset) > 0 and
                "### Human:" in str(dataset[0].get("text", ""))
            )
            
            is_dialogsum = "dialogue" in columns and "summary" in columns
            has_instruction = "instruction" in columns and "output" in columns
            has_prompt_response = "prompt" in columns and "response" in columns
            has_qa = "question" in columns and "answer" in columns
            has_user_assistant = "user" in columns and "assistant" in columns
            has_input_output = "input" in columns and "output" in columns and "instruction" not in columns
            
            # Specific dataset format detection
            is_dolly = (self.dataset_name == "databricks/databricks-dolly-15k" or
                       ("instruction" in columns and "context" in columns and "response" in columns))
            is_orca = (self.dataset_name == "Open-Orca/OpenOrca" or
                      ("system_prompt" in columns and "question" in columns and "response" in columns))
            is_alpaca = (self.dataset_name == "tatsu-lab/alpaca" or
                        self.dataset_name == "WizardLM/WizardLM_evol_instruct_70k" or
                        ("instruction" in columns and "input" in columns and "output" in columns))
            is_hh_rlhf = (self.dataset_name == "Anthropic/hh-rlhf" or
                         ("chosen" in columns and "rejected" in columns))
            is_oasst = (self.dataset_name in ("OpenAssistant/oasst1", "OpenAssistant/oasst2") or
                       ("text" in columns and "role" in columns and "parent_id" in columns and "lang" in columns))
            is_chatbot_arena = (self.dataset_name == "lmsys/chatbot_arena_conversations" or
                               ("conversation_a" in columns and "conversation_b" in columns))
            is_code_alpaca = (self.dataset_name in ["sahil2801/CodeAlpaca-20k",
                                                     "TokenBender/code_instructions_122k_alpaca_style"] or
                             (self.dataset_name.startswith("code") and
                              "instruction" in columns and "output" in columns))
            
            # Determine text column
            text_column = None
            for col in ["text", "content", "dialogue", "dialog", "conversation", "messages"]:
                if col in columns:
                    text_column = col
                    break
                    
            if is_guanaco:
                print("Detected Guanaco format (### Human: / ### Assistant:)")
            elif is_dolly:
                print("Detected Dolly format (instruction/context/response)")
            elif is_orca:
                print("Detected OpenOrca format (system_prompt/question/response)")
            elif is_alpaca:
                print("Detected Alpaca format (instruction/input/output)")
            elif is_hh_rlhf:
                print("Detected Anthropic hh-rlhf format (chosen/rejected)")
            elif is_oasst:
                print("Detected OpenAssistant oasst2 format (text/role tree)")
                if self.language_filter:
                    print(f"  Language filter: {self.language_filter} only")
            elif is_chatbot_arena:
                print("Detected Chatbot Arena format (conversation_a/b)")
            elif is_code_alpaca:
                print("Detected Code Alpaca format (instruction/input/output)")
            elif is_dialogsum:
                print("Detected DialogSum format")
            elif has_instruction:
                print("Detected instruction/output format")
            elif has_prompt_response:
                print("Detected prompt/response format")
            elif has_qa:
                print("Detected question/answer format")
            elif has_user_assistant:
                print("Detected user/assistant format (e.g., lonestar108/sexygpt)")
            elif has_input_output:
                print("Detected input/output format (e.g., lonestar108/rawdata)")
            elif text_column:
                print(f"Using text column: '{text_column}'")
            else:
                print("No standard format detected, will concatenate all string fields")
            
            # Process in chunks to avoid memory issues
            current_chunk = []
            chunk_size = 1000  # Process 1000 examples at a time
            
            for i, item in enumerate(dataset):
                text_to_add = ""

                if is_guanaco:
                    # Guanaco format: text column contains ### Human: / ### Assistant:
                    text_to_add = self._format_guanaco_example(item.get("text", ""))
                elif is_dolly:
                    text_to_add = self._format_dolly_example(item)
                elif is_orca:
                    text_to_add = self._format_orca_example(item)
                elif is_alpaca:
                    text_to_add = self._format_alpaca_example(item)
                elif is_hh_rlhf:
                    text_to_add = self._format_hh_rlhf_example(item)
                elif is_oasst:
                    # Apply language filter for OpenAssistant datasets
                    if self.language_filter:
                        item_lang = item.get("lang", "").strip().lower()
                        if item_lang != self.language_filter.lower():
                            # Skip non-matching language items
                            self.stats.skipped_language_filter += 1
                            continue
                    text_to_add = self._format_oasst_example(item, dataset=dataset, idx=i)
                elif is_chatbot_arena:
                    text_to_add = self._format_chatbot_arena_example(item)
                elif is_code_alpaca:
                    text_to_add = self._format_code_alpaca_example(item)
                elif is_dialogsum:
                    text_to_add = self._format_dialogsum_example(item)
                elif has_instruction:
                    text_to_add = self._format_pair_example(item, "instruction", "output")
                elif has_prompt_response:
                    text_to_add = self._format_pair_example(item, "prompt", "response")
                elif has_qa:
                    text_to_add = self._format_pair_example(item, "question", "answer")
                elif has_user_assistant:
                    text_to_add = self._format_pair_example(item, "user", "assistant")
                elif has_input_output:
                    text_to_add = self._format_pair_example(item, "input", "output")
                elif text_column:
                    content = item.get(text_column, "")
                
                    # Handle list of strings (e.g. daily_dialog)
                    if isinstance(content, list):
                        if content and isinstance(content[0], str):
                            # Convert list of strings to chat template alternating format
                            conversation_parts = []
                            for idx, line in enumerate(content):
                                line = line.strip()
                                if line:
                                    if idx % 2 == 0:
                                        conversation_parts.append(f"<|user|>\n{line}\n<|endofuser|>")
                                    else:
                                        conversation_parts.append(f"<|assistant|>\n{line}\n<|endofassistant|>")
                            text_to_add = "\n".join(conversation_parts) if conversation_parts else ""
                        elif content and isinstance(content[0], dict):
                            # Handle list of dicts (e.g. messages: [{'role':..., 'content':...}])
                            conversation = []
                            for msg in content:
                                if 'content' in msg:
                                    role = msg.get('role', 'unknown').lower()
                                    msg_content = msg['content'].strip() if isinstance(msg['content'], str) else str(msg['content'])
                                    if msg_content:
                                        if role in ['human', 'user']:
                                            conversation.append(f"<|user|>\n{msg_content}\n<|endofuser|>")
                                        elif role in ['assistant', 'bot', 'ai']:
                                            conversation.append(f"<|assistant|>\n{msg_content}\n<|endofassistant|>")
                            text_to_add = "\n".join(conversation)
                    
                    # Handle simple string - try to detect and convert format
                    elif isinstance(content, str):
                        content = content.strip()
                        if content:
                            # Check if already in chat template format
                            if "<|user|>" in content and "<|assistant|>" in content:
                                text_to_add = content
                            elif "### Human:" in content and "### Assistant:" in content:
                                # Convert Guanaco-style format
                                text_to_add = self._format_guanaco_example(content)
                            else:
                                # Plain text - skip as we can't determine Q/A structure
                                text_to_add = ""
                
                # Track stats and validate
                self.stats.total_examples += 1
                
                if not text_to_add:
                    self.stats.skipped_empty += 1
                elif self.validate_format:
                    is_valid, reason = validate_format(text_to_add)
                    if is_valid:
                        self.stats.valid_examples += 1
                        current_chunk.append(text_to_add)
                    else:
                        if reason == "no_user":
                            self.stats.skipped_no_user += 1
                        elif reason == "no_assistant":
                            self.stats.skipped_no_assistant += 1
                        elif reason in ("blank_user", "blank_assistant"):
                            self.stats.skipped_blank_parts += 1
                        else:
                            self.stats.skipped_empty += 1
                else:
                    # No validation - just add if non-empty
                    self.stats.valid_examples += 1
                    current_chunk.append(text_to_add)
                    
                # Encode chunk if full
                if len(current_chunk) >= chunk_size:
                    chunk_text = "\n\n".join(current_chunk)
                    self.data.extend(self.tokenizer.encode(chunk_text))
                    current_chunk = []
                    
                    if i % 5000 == 0:
                        print(f"  Processed {i} examples, {len(self.data)} tokens, "
                              f"{self.stats.valid_examples} valid...")
                        
                    if len(self.data) > self.max_tokens:
                        print(f"Reached {self.max_tokens // 1_000_000}M token limit, stopping load.")
                        break
                        
            # Encode remaining
            if current_chunk:
                chunk_text = "\n\n".join(current_chunk)
                self.data.extend(self.tokenizer.encode(chunk_text))
            
            print(f"\nLoaded {len(self.data):,} tokens from '{self.dataset_name}'")
            print(self.stats.summary())
            
        except ImportError:
            print("Error: 'datasets' library not found. Please install with: pip install datasets")
            self.data = []
        except Exception as e:
            print(f"Error loading HuggingFace dataset: {e}")
            import traceback
            traceback.print_exc()
            self.data = []


class GuanacoDataset(HuggingFaceDataset):
    """
    Convenience class for loading the timdettmers/openassistant-guanaco dataset.
    
    This is the recommended dataset for Reso-LLM training.
    
    The Guanaco dataset contains ~10K high-quality conversation examples
    derived from OpenAssistant, formatted as:
    
    ### Human: <user message>
    ### Assistant: <assistant response>
    
    Example usage:
        tokenizer = create_default_tokenizer()
        dataset = GuanacoDataset(tokenizer, seq_len=256, batch_size=32)
        trainer = ResoLLMTrainer(model, dataset)
    """
    
    def __init__(
        self,
        tokenizer: ResoLLMTokenizer,
        seq_len: int = 256,
        batch_size: int = 32,
        split: str = "train",
        max_tokens: int = 10_000_000
    ):
        super().__init__(
            dataset_name="timdettmers/openassistant-guanaco",
            tokenizer=tokenizer,
            seq_len=seq_len,
            batch_size=batch_size,
            split=split,
            max_tokens=max_tokens
        )


def create_guanaco_dataset(
    tokenizer: ResoLLMTokenizer,
    seq_len: int = 256,
    batch_size: int = 32,
    split: str = "train"
) -> GuanacoDataset:
    """
    Create a dataset from timdettmers/openassistant-guanaco.
    
    This is the recommended function for creating training datasets.
    
    Args:
        tokenizer: The tokenizer to use
        seq_len: Sequence length for training
        batch_size: Batch size
        split: Dataset split ("train" is the only one available)
        
    Returns:
        GuanacoDataset instance
    """
    return GuanacoDataset(
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        split=split
    )


def verify_dataset_fusion(
    dataset: HuggingFaceDataset,
    num_samples: int = 5,
    verbose: bool = True
) -> Dict[str, any]:
    """
    Verify that dataset fusion produced valid training data.
    
    Checks that:
    1. The dataset has tokens loaded
    2. Each decoded sample contains User: and Assistant: parts
    3. Neither the question nor answer parts are blank
    4. Proper newline separation between role markers
    5. End-of-conversation markers present
    
    Args:
        dataset: The HuggingFaceDataset to verify
        num_samples: Number of random samples to check and display
        verbose: Whether to print sample outputs
        
    Returns:
        Dictionary with verification results:
        - valid: bool indicating if dataset is valid
        - token_count: total tokens loaded
        - stats: DatasetStats object
        - sample_checks: list of sample verification results
    """
    import re
    
    results = {
        "valid": True,
        "token_count": len(dataset.data),
        "stats": dataset.stats,
        "sample_checks": [],
        "issues": [],
        "format_checks": {
            "proper_newlines": 0,
            "inline_markers": 0,
            "has_eoc_marker": 0,
            "complete_conversations": 0
        }
    }
    
    # Check 1: We have data
    if len(dataset.data) == 0:
        results["valid"] = False
        results["issues"].append("No tokens loaded in dataset")
        return results
    
    # Check 2: Validation stats look reasonable
    if dataset.stats.valid_examples == 0:
        results["valid"] = False
        results["issues"].append("No valid examples in dataset")
        return results
    
    valid_pct = dataset.stats.valid_examples / max(1, dataset.stats.total_examples)
    if valid_pct < 0.5:
        results["issues"].append(f"Warning: Only {valid_pct*100:.1f}% of examples were valid")
    
    # Check 3: Sample complete conversations by finding EOC markers
    if verbose:
        print("\n" + "=" * 60)
        print("Dataset Fusion Verification")
        print("=" * 60)
        print(f"Total tokens: {len(dataset.data):,}")
        print(dataset.stats.summary())
        print("\n" + "-" * 60)
        print(f"Checking {num_samples} complete conversation samples:")
        print("-" * 60)
    
    # Decode a large chunk to find complete conversations
    import random
    
    # Try to find complete conversations by looking for EOC markers
    chunk_size = min(50000, len(dataset.data))  # Decode up to 50k tokens
    start_pos = random.randint(0, max(0, len(dataset.data) - chunk_size))
    chunk_tokens = dataset.data[start_pos:start_pos + chunk_size]
    
    try:
        chunk_text = dataset.tokenizer.decode(chunk_tokens)
    except Exception as e:
        chunk_text = ""
        results["issues"].append(f"Failed to decode sample chunk: {e}")
    
    # Split by EOC marker to get individual conversations
    conversations = chunk_text.split("<|endofconversation|>")
    valid_conversations = []
    
    for conv in conversations:
        conv = conv.strip()
        if not conv:
            continue
        
        # Check if this is a complete conversation
        has_user = "\nUser:" in conv or conv.startswith("User:")
        has_assistant = "\nAssistant:" in conv
        
        # Check for inline markers (bad formatting)
        has_inline = bool(re.search(r'[^\n]User:', conv)) or bool(re.search(r'[^\n]Assistant:', conv))
        
        if has_user and has_assistant:
            valid_conversations.append({
                "text": conv,
                "has_inline": has_inline
            })
            results["format_checks"]["complete_conversations"] += 1
            if has_inline:
                results["format_checks"]["inline_markers"] += 1
            else:
                results["format_checks"]["proper_newlines"] += 1
    
    # Show samples
    samples_shown = 0
    for i, conv_data in enumerate(valid_conversations[:num_samples]):
        conv = conv_data["text"]
        has_inline = conv_data["has_inline"]
        
        # Find User and Assistant content
        user_match = re.search(r'User:\s*(.+?)(?=\n*Assistant:|$)', conv, re.DOTALL)
        assistant_match = re.search(r'Assistant:\s*(.+?)$', conv, re.DOTALL)
        
        user_content = user_match.group(1).strip()[:100] if user_match else "[not found]"
        assistant_content = assistant_match.group(1).strip()[:100] if assistant_match else "[not found]"
        
        sample_result = {
            "index": i,
            "has_inline_markers": has_inline,
            "user_preview": user_content,
            "assistant_preview": assistant_content,
            "properly_formatted": not has_inline and bool(user_match) and bool(assistant_match)
        }
        results["sample_checks"].append(sample_result)
        
        if verbose:
            status = "⚠️  INLINE MARKERS" if has_inline else "✓ Properly formatted"
            print(f"\n[Conversation {i+1}] {status}")
            print(f"  User: {user_content}...")
            print(f"  Assistant: {assistant_content}...")
        
        samples_shown += 1
    
    # Report inline marker issues
    if results["format_checks"]["inline_markers"] > 0:
        pct = results["format_checks"]["inline_markers"] / max(1, results["format_checks"]["complete_conversations"]) * 100
        results["issues"].append(
            f"Warning: {results['format_checks']['inline_markers']} conversations "
            f"({pct:.1f}%) have inline markers without proper newlines"
        )
    
    if verbose:
        print("\n" + "=" * 60)
        if results["issues"]:
            print("Issues found:")
            for issue in results["issues"]:
                print(f"  - {issue}")
        else:
            print("✓ Dataset fusion verification passed!")
        print("=" * 60)
    
    return results


def create_multi_dataset(
    dataset_names: List[str],
    tokenizer: ResoLLMTokenizer,
    seq_len: int = 256,
    batch_size: int = 32,
    max_tokens_per_dataset: int = 5_000_000
) -> HuggingFaceDataset:
    """
    Create a fused dataset from multiple HuggingFace datasets.
    
    Each dataset is loaded and converted to the unified User:/Assistant: format,
    then combined into a single dataset for training.
    
    Args:
        dataset_names: List of HuggingFace dataset names
        tokenizer: The tokenizer to use
        seq_len: Sequence length for training
        batch_size: Batch size
        max_tokens_per_dataset: Maximum tokens to load from each dataset
        
    Returns:
        A HuggingFaceDataset with combined data from all sources
    """
    combined_data: List[int] = []
    combined_stats = DatasetStats()
    
    print(f"Loading {len(dataset_names)} datasets for fusion...")
    
    for name in dataset_names:
        print(f"\nLoading: {name}")
        try:
            ds = HuggingFaceDataset(
                dataset_name=name,
                tokenizer=tokenizer,
                seq_len=seq_len,
                batch_size=batch_size,
                max_tokens=max_tokens_per_dataset
            )
            
            # Merge data
            combined_data.extend(ds.data)
            
            # Merge stats
            combined_stats.total_examples += ds.stats.total_examples
            combined_stats.valid_examples += ds.stats.valid_examples
            combined_stats.skipped_empty += ds.stats.skipped_empty
            combined_stats.skipped_no_user += ds.stats.skipped_no_user
            combined_stats.skipped_no_assistant += ds.stats.skipped_no_assistant
            combined_stats.skipped_blank_parts += ds.stats.skipped_blank_parts
            
            print(f"  Added {len(ds.data):,} tokens, {ds.stats.valid_examples:,} valid examples")
            
        except Exception as e:
            print(f"  Error loading {name}: {e}")
            continue
    
    # Create a result dataset with combined data
    result = HuggingFaceDataset.__new__(HuggingFaceDataset)
    result.dataset_name = f"multi:{'+'.join(dataset_names)}"
    result.split = "train"
    result.max_tokens = max_tokens_per_dataset * len(dataset_names)
    result.validate_format = True
    result.tokenizer = tokenizer
    result.seq_len = seq_len
    result.batch_size = batch_size
    result.data = combined_data
    result.stats = combined_stats
    
    print(f"\n{'='*60}")
    print(f"Multi-Dataset Fusion Complete")
    print(f"{'='*60}")
    print(f"Total tokens: {len(combined_data):,}")
    print(combined_stats.summary())
    
    return result


@dataclass
class DatasetInfo:
    """Information about a single dataset in a multi-dataset collection."""
    name: str
    token_count: int = 0
    valid_examples: int = 0
    total_examples: int = 0
    error: Optional[str] = None
    
    @property
    def valid_percentage(self) -> float:
        """Percentage of valid examples."""
        return (self.valid_examples / max(1, self.total_examples)) * 100


class MultiDataset(TextDataset):
    """
    Dataset that combines multiple HuggingFace datasets for training.
    
    Loads multiple datasets, processes each through the unified format pipeline,
    validates all output, and combines into a single training dataset.
    
    Features:
    - Per-dataset statistics tracking
    - Combined validation with detailed reporting
    - Shuffle/interleave options for balanced sampling
    - Sample inspection for quality verification
    
    Example:
        tokenizer = create_default_tokenizer()
        multi_ds = MultiDataset(
            dataset_names=[
                "timdettmers/openassistant-guanaco",
                "databricks/databricks-dolly-15k",
                "tatsu-lab/alpaca"
            ],
            tokenizer=tokenizer,
            seq_len=256,
            batch_size=32
        )
        
        # Check per-dataset stats
        for info in multi_ds.dataset_info:
            print(f"{info.name}: {info.token_count:,} tokens")
        
        # Validate the combined dataset
        report = multi_ds.validate()
        print(report)
    """
    
    def __init__(
        self,
        dataset_names: List[str],
        tokenizer: ResoLLMTokenizer,
        seq_len: int = 256,
        batch_size: int = 32,
        max_tokens_per_dataset: int = 5_000_000,
        shuffle: bool = True,
        interleave: bool = False,
        validate_output: bool = True
    ):
        """
        Initialize a multi-dataset loader.
        
        Args:
            dataset_names: List of HuggingFace dataset names to load
            tokenizer: The tokenizer to use for encoding
            seq_len: Sequence length for training batches
            batch_size: Batch size for training
            max_tokens_per_dataset: Maximum tokens to load from each dataset
            shuffle: Whether to shuffle the combined data
            interleave: If True, interleave tokens from datasets instead of concatenating
            validate_output: Whether to validate the combined output
        """
        self.dataset_names = dataset_names
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.max_tokens_per_dataset = max_tokens_per_dataset
        self.shuffle_data = shuffle
        self.interleave = interleave
        self.validate_output = validate_output
        
        # Per-dataset tracking
        self.dataset_info: List[DatasetInfo] = []
        self.individual_datasets: List[HuggingFaceDataset] = []
        
        # Combined stats
        self.stats = DatasetStats()
        self.data: List[int] = []
        
        # Load all datasets
        self._load_all_datasets()
    
    def _load_all_datasets(self):
        """Load and combine all specified datasets."""
        print(f"\n{'='*60}")
        print(f"MultiDataset: Loading {len(self.dataset_names)} datasets")
        print(f"{'='*60}")
        
        all_data: List[List[int]] = []
        
        for name in self.dataset_names:
            print(f"\n[Loading] {name}")
            info = DatasetInfo(name=name)
            
            try:
                ds = HuggingFaceDataset(
                    dataset_name=name,
                    tokenizer=self.tokenizer,
                    seq_len=self.seq_len,
                    batch_size=self.batch_size,
                    max_tokens=self.max_tokens_per_dataset,
                    validate_format=True
                )
                
                info.token_count = len(ds.data)
                info.valid_examples = ds.stats.valid_examples
                info.total_examples = ds.stats.total_examples
                
                # Aggregate stats
                self.stats.total_examples += ds.stats.total_examples
                self.stats.valid_examples += ds.stats.valid_examples
                self.stats.skipped_empty += ds.stats.skipped_empty
                self.stats.skipped_no_user += ds.stats.skipped_no_user
                self.stats.skipped_no_assistant += ds.stats.skipped_no_assistant
                self.stats.skipped_blank_parts += ds.stats.skipped_blank_parts
                
                all_data.append(ds.data)
                self.individual_datasets.append(ds)
                
                print(f"  ✓ {info.token_count:,} tokens, "
                      f"{info.valid_examples:,} valid examples "
                      f"({info.valid_percentage:.1f}%)")
                
            except Exception as e:
                info.error = str(e)
                print(f"  ✗ Error: {e}")
            
            self.dataset_info.append(info)
        
        # Combine data
        if self.interleave and len(all_data) > 1:
            self.data = self._interleave_data(all_data)
        else:
            for tokens in all_data:
                self.data.extend(tokens)
        
        # Shuffle if requested
        if self.shuffle_data and self.data:
            self._shuffle_sequences()
        
        # Validate combined output
        if self.validate_output:
            self._run_validation()
        
        # Print summary
        self._print_summary()
    
    def _interleave_data(self, all_data: List[List[int]]) -> List[int]:
        """
        Interleave tokens from multiple datasets.
        
        This helps balance training when datasets have different sizes.
        """
        result = []
        chunk_size = self.seq_len * 10  # Interleave in chunks
        
        # Calculate chunks per dataset
        chunks = []
        for data in all_data:
            ds_chunks = [data[i:i+chunk_size] for i in range(0, len(data), chunk_size)]
            chunks.append(ds_chunks)
        
        # Round-robin interleave
        max_chunks = max(len(c) for c in chunks) if chunks else 0
        for i in range(max_chunks):
            for ds_chunks in chunks:
                if i < len(ds_chunks):
                    result.extend(ds_chunks[i])
        
        return result
    
    def _shuffle_sequences(self):
        """Shuffle data at the sequence level to mix datasets."""
        # Chunk data into sequences
        sequences = []
        for i in range(0, len(self.data) - self.seq_len, self.seq_len):
            sequences.append(self.data[i:i+self.seq_len])
        
        # Shuffle sequences
        random.shuffle(sequences)
        
        # Flatten back
        self.data = []
        for seq in sequences:
            self.data.extend(seq)
    
    def _run_validation(self):
        """Run validation on the combined dataset."""
        print("\n[Validating] Combined dataset output...")
        
        # Decode sample sections and validate format
        sample_size = min(10000, len(self.data))
        sample_tokens = self.data[:sample_size]
        
        try:
            sample_text = self.tokenizer.decode(sample_tokens)
            
            # Check for required markers
            has_user = "<|user|>" in sample_text
            has_endofuser = "<|endofuser|>" in sample_text
            has_assistant = "<|assistant|>" in sample_text
            has_endofassistant = "<|endofassistant|>" in sample_text
            
            if has_user and has_endofuser and has_assistant and has_endofassistant:
                print("  ✓ Chat template format validated")
            else:
                missing = []
                if not has_user: missing.append("<|user|>")
                if not has_endofuser: missing.append("<|endofuser|>")
                if not has_assistant: missing.append("<|assistant|>")
                if not has_endofassistant: missing.append("<|endofassistant|>")
                print(f"  ⚠ Missing markers: {', '.join(missing)}")
                
        except Exception as e:
            print(f"  ⚠ Validation decode error: {e}")
    
    def _print_summary(self):
        """Print a summary of the loaded datasets."""
        print(f"\n{'='*60}")
        print("MultiDataset Summary")
        print(f"{'='*60}")
        
        # Per-dataset breakdown
        print("\nPer-Dataset Statistics:")
        print(f"{'Dataset':<45} {'Tokens':>12} {'Valid':>8} {'%':>6}")
        print("-" * 75)
        
        for info in self.dataset_info:
            if info.error:
                print(f"{info.name:<45} {'ERROR':>12} {'-':>8} {'-':>6}")
            else:
                print(f"{info.name:<45} {info.token_count:>12,} "
                      f"{info.valid_examples:>8,} {info.valid_percentage:>5.1f}%")
        
        print("-" * 75)
        print(f"{'TOTAL':<45} {len(self.data):>12,} "
              f"{self.stats.valid_examples:>8,} "
              f"{(self.stats.valid_examples/max(1,self.stats.total_examples))*100:>5.1f}%")
        
        # Aggregated stats
        print(f"\n{self.stats.summary()}")
        print(f"{'='*60}\n")
    
    def validate(self, num_samples: int = 5, verbose: bool = True) -> Dict[str, any]:
        """
        Perform comprehensive validation of the combined dataset.
        
        Args:
            num_samples: Number of samples to display and check
            verbose: Whether to print detailed output
            
        Returns:
            Dictionary with validation results including:
            - valid: Overall validity boolean
            - per_dataset: List of per-dataset validation results
            - combined: Combined dataset validation metrics
            - samples: Sample conversation previews
        """
        results = {
            "valid": True,
            "per_dataset": [],
            "combined": {
                "total_tokens": len(self.data),
                "total_examples": self.stats.total_examples,
                "valid_examples": self.stats.valid_examples,
                "valid_percentage": (self.stats.valid_examples /
                                    max(1, self.stats.total_examples)) * 100
            },
            "samples": [],
            "issues": []
        }
        
        # Check each dataset
        for info in self.dataset_info:
            ds_result = {
                "name": info.name,
                "valid": info.error is None,
                "token_count": info.token_count,
                "valid_examples": info.valid_examples,
                "error": info.error
            }
            results["per_dataset"].append(ds_result)
            
            if info.error:
                results["valid"] = False
                results["issues"].append(f"Dataset '{info.name}' failed to load: {info.error}")
        
        # Check we have data
        if len(self.data) == 0:
            results["valid"] = False
            results["issues"].append("No tokens loaded in combined dataset")
            return results
        
        # Sample and validate conversations
        try:
            # Decode random sections
            for i in range(num_samples):
                start_pos = random.randint(0, max(0, len(self.data) - 2000))
                sample_tokens = self.data[start_pos:start_pos + 2000]
                sample_text = self.tokenizer.decode(sample_tokens)
                
                # Find a complete conversation
                # Look for <|user|>...<|endofassistant|> pattern
                import re
                pattern = r'<\|user\|>\s*(.+?)\s*<\|endofuser\|>\s*<\|assistant\|>\s*(.+?)\s*<\|endofassistant\|>'
                match = re.search(pattern, sample_text, re.DOTALL)
                
                if match:
                    user_text = match.group(1).strip()[:100]
                    assistant_text = match.group(2).strip()[:100]
                    
                    results["samples"].append({
                        "user_preview": user_text,
                        "assistant_preview": assistant_text,
                        "valid": bool(user_text and assistant_text)
                    })
                    
                    if verbose:
                        print(f"\n[Sample {i+1}]")
                        print(f"  User: {user_text}...")
                        print(f"  Assistant: {assistant_text}...")
        except Exception as e:
            results["issues"].append(f"Sample validation error: {e}")
        
        if verbose:
            print(f"\n{'='*60}")
            print("Validation Complete")
            print(f"{'='*60}")
            print(f"Valid: {'✓ YES' if results['valid'] else '✗ NO'}")
            print(f"Total tokens: {results['combined']['total_tokens']:,}")
            print(f"Valid examples: {results['combined']['valid_examples']:,} "
                  f"({results['combined']['valid_percentage']:.1f}%)")
            
            if results["issues"]:
                print("\nIssues:")
                for issue in results["issues"]:
                    print(f"  - {issue}")
            print(f"{'='*60}")
        
        return results
    
    def get_sample_conversations(self, n: int = 5) -> List[Dict[str, str]]:
        """
        Extract sample conversations from the dataset.
        
        Args:
            n: Number of sample conversations to extract
            
        Returns:
            List of dictionaries with 'user' and 'assistant' keys
        """
        import re
        samples = []
        
        # Decode a large chunk
        chunk_size = min(100000, len(self.data))
        chunk_tokens = self.data[:chunk_size]
        
        try:
            chunk_text = self.tokenizer.decode(chunk_tokens)
            
            # Find all complete conversations
            pattern = r'<\|user\|>\s*(.+?)\s*<\|endofuser\|>\s*<\|assistant\|>\s*(.+?)\s*<\|endofassistant\|>'
            matches = re.findall(pattern, chunk_text, re.DOTALL)
            
            for user_text, assistant_text in matches[:n]:
                samples.append({
                    "user": user_text.strip(),
                    "assistant": assistant_text.strip()
                })
                
        except Exception:
            pass
        
        return samples
    
    def __len__(self) -> int:
        """Number of batches per epoch."""
        n_sequences = (len(self.data) - 1) // self.seq_len
        return max(1, n_sequences // self.batch_size)
    
    def get_batch(self) -> Tuple[Tensor, Tensor]:
        """Get a random batch of data."""
        max_idx = len(self.data) - self.seq_len - 1
        
        batch_x = []
        batch_y = []
        
        for _ in range(self.batch_size):
            idx = random.randint(0, max_idx)
            x_seq = self.data[idx : idx + self.seq_len]
            y_seq = self.data[idx + 1 : idx + self.seq_len + 1]
            
            batch_x.extend(x_seq)
            batch_y.extend(y_seq)
        
        x_tensor = Tensor(batch_x, (self.batch_size, self.seq_len))
        y_tensor = Tensor(batch_y, (self.batch_size, self.seq_len))
        
        return x_tensor, y_tensor


def create_recommended_multi_dataset(
    tokenizer: ResoLLMTokenizer,
    seq_len: int = 256,
    batch_size: int = 32,
    max_tokens_per_dataset: int = 2_000_000
) -> MultiDataset:
    """
    Create a multi-dataset with recommended high-quality datasets.
    
    Uses a curated selection of diverse, high-quality datasets:
    - timdettmers/openassistant-guanaco (general conversation)
    - databricks/databricks-dolly-15k (instruction following)
    - tatsu-lab/alpaca (instruction tuning)
    
    Args:
        tokenizer: The tokenizer to use
        seq_len: Sequence length for training
        batch_size: Batch size
        max_tokens_per_dataset: Max tokens per dataset
        
    Returns:
        MultiDataset instance with combined data
    """
    recommended_datasets = [
        "timdettmers/openassistant-guanaco",
        "databricks/databricks-dolly-15k",
        "tatsu-lab/alpaca",
    ]
    
    return MultiDataset(
        dataset_names=recommended_datasets,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        max_tokens_per_dataset=max_tokens_per_dataset
    )
