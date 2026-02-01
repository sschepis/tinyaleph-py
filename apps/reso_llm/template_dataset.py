"""
Template-based Dataset Integration for ResoLLM

This module bridges the input_templates system with the existing dataset infrastructure,
providing a unified way to load, format, and tokenize data using templates.

Usage:
    from apps.reso_llm.template_dataset import TemplateBasedDataset, create_templated_dataset
    from apps.reso_llm.input_templates import get_template
    from apps.reso_llm.tokenizer import ResoBPETokenizer
    
    # Create tokenizer and template
    tokenizer = ResoBPETokenizer()
    template = get_template("chat")
    
    # Load dataset with template formatting
    dataset = TemplateBasedDataset(
        dataset_name="timdettmers/openassistant-guanaco",
        template=template,
        tokenizer=tokenizer,
        seq_len=512,
        batch_size=8,
    )
    
    # Get training batches with proper loss masking
    batch = dataset.get_batch_with_masks()
"""

import sys
import os
import random
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any, Union, Generator

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.input_templates import (
    InputTemplate, TemplateOutput, TokenizedOutput,
    ChatTemplate, InstructionTemplate, QATemplate, CompletionTemplate,
    DialogueTemplate, CodeTemplate, PreferenceTemplate,
    TemplateRegistry, get_template, SpecialTokens,
)

try:
    from tinyaleph.ml.resoformer import Tensor
except ImportError:
    # Fallback if tinyaleph not available
    class Tensor:
        def __init__(self, data, shape):
            self.data = data
            self.shape = shape


@dataclass
class BatchWithMasks:
    """A training batch with loss masks for selective gradient computation."""
    input_ids: Any  # Tensor or list
    target_ids: Any  # Tensor or list
    attention_mask: Any  # Tensor or list
    loss_mask: Any  # Tensor or list - 1 where to compute loss, 0 otherwise
    segment_ids: Optional[Any] = None  # Optional segment indicators
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "input_ids": self.input_ids,
            "target_ids": self.target_ids,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
            "segment_ids": self.segment_ids,
        }


class DatasetFormatConverter:
    """
    Converts raw HuggingFace dataset items to template-compatible formats.
    
    This class handles the various formats found in common datasets and
    converts them to the format expected by each template type.
    """
    
    # Dataset name to template type mapping
    DATASET_TEMPLATES = {
        "timdettmers/openassistant-guanaco": "dialogue",
        "databricks/databricks-dolly-15k": "instruction",
        "Open-Orca/OpenOrca": "instruction",
        "tatsu-lab/alpaca": "instruction",
        "WizardLM/WizardLM_evol_instruct_70k": "instruction",
        "Anthropic/hh-rlhf": "preference",
        "sahil2801/CodeAlpaca-20k": "code",
        "TokenBender/code_instructions_122k_alpaca_style": "code",
        "knkarthick/dialogsum": "qa",
    }
    
    @classmethod
    def detect_template(cls, dataset_name: str, columns: List[str]) -> str:
        """Detect the appropriate template type for a dataset."""
        # Check known datasets first
        if dataset_name in cls.DATASET_TEMPLATES:
            return cls.DATASET_TEMPLATES[dataset_name]
        
        # Detect by column structure
        if "messages" in columns:
            return "chat"
        elif "chosen" in columns and "rejected" in columns:
            return "preference"
        elif "instruction" in columns and "output" in columns:
            if "code" in dataset_name.lower():
                return "code"
            return "instruction"
        elif "question" in columns and "answer" in columns:
            return "qa"
        elif "prompt" in columns and "completion" in columns:
            return "completion"
        elif "dialogue" in columns or "dialog" in columns:
            return "dialogue"
        elif "text" in columns:
            # Could be guanaco-style or plain text
            return "dialogue"  # Assume dialogue, will be converted
        else:
            return "completion"  # Fallback
    
    @classmethod
    def convert_guanaco(cls, text: str) -> Dict[str, Any]:
        """Convert Guanaco format (### Human:/### Assistant:) to dialogue format."""
        import re
        turns = []
        parts = re.split(r'###\s*(Human|Assistant):\s*', text.strip())
        
        i = 1
        while i < len(parts) - 1:
            speaker = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            if content:
                turns.append({"speaker": speaker, "text": content})
            i += 2
        
        return {"turns": turns}
    
    @classmethod
    def convert_alpaca(cls, item: Dict) -> Dict[str, Any]:
        """Convert Alpaca format to instruction format."""
        return {
            "instruction": item.get("instruction", ""),
            "input": item.get("input", ""),
            "output": item.get("output", ""),
        }
    
    @classmethod
    def convert_dolly(cls, item: Dict) -> Dict[str, Any]:
        """Convert Dolly format to instruction format."""
        instruction = item.get("instruction", "")
        context = item.get("context", "")
        if context:
            instruction = f"{instruction}\n\nContext: {context}"
        return {
            "instruction": instruction,
            "output": item.get("response", ""),
        }
    
    @classmethod
    def convert_orca(cls, item: Dict) -> Dict[str, Any]:
        """Convert OpenOrca format to instruction format."""
        system = item.get("system_prompt", "")
        question = item.get("question", "")
        if system and system != "You are an AI assistant.":
            instruction = f"[System: {system}]\n\n{question}"
        else:
            instruction = question
        return {
            "instruction": instruction,
            "output": item.get("response", ""),
        }
    
    @classmethod
    def convert_hh_rlhf(cls, item: Dict) -> Dict[str, Any]:
        """Convert Anthropic hh-rlhf format to preference format."""
        import re
        
        def extract_last_exchange(text: str) -> Tuple[str, str]:
            parts = re.split(r'\n\n(Human|Assistant):\s*', text)
            prompt, response = "", ""
            for i in range(len(parts) - 2, -1, -2):
                if parts[i].lower() == "assistant":
                    response = parts[i + 1].strip() if i + 1 < len(parts) else ""
                    # Get preceding human turn as prompt
                    if i >= 2 and parts[i - 2].lower() == "human":
                        prompt = parts[i - 1].strip()
                    break
            return prompt, response
        
        chosen_prompt, chosen_response = extract_last_exchange(item.get("chosen", ""))
        _, rejected_response = extract_last_exchange(item.get("rejected", ""))
        
        return {
            "prompt": chosen_prompt,
            "chosen": chosen_response,
            "rejected": rejected_response,
        }
    
    @classmethod
    def convert_code_alpaca(cls, item: Dict) -> Dict[str, Any]:
        """Convert code instruction format."""
        instruction = item.get("instruction", "")
        input_code = item.get("input", "")
        if input_code:
            instruction = f"{instruction}\n\n```\n{input_code}\n```"
        return {
            "instruction": instruction,
            "code": item.get("output", ""),
            "language": "python",  # Default assumption
        }
    
    @classmethod
    def convert_dialogsum(cls, item: Dict) -> Dict[str, Any]:
        """Convert DialogSum format to QA format."""
        return {
            "context": item.get("dialogue", ""),
            "question": "Summarize the above dialogue.",
            "answer": item.get("summary", ""),
        }
    
    @classmethod
    def convert_messages(cls, messages: List[Dict]) -> Dict[str, Any]:
        """Convert messages list format to chat format."""
        return {"messages": messages}
    
    @classmethod
    def convert_item(
        cls,
        item: Dict[str, Any],
        dataset_name: str,
        columns: List[str],
    ) -> Tuple[str, Dict[str, Any]]:
        """
        Convert a dataset item to template-compatible format.
        
        Returns:
            (template_name, converted_data) tuple
        """
        template_name = cls.detect_template(dataset_name, columns)
        
        # Handle specific dataset formats
        if dataset_name == "timdettmers/openassistant-guanaco":
            text = item.get("text", "")
            if "### Human:" in text:
                return "dialogue", cls.convert_guanaco(text)
        
        if dataset_name == "databricks/databricks-dolly-15k":
            return "instruction", cls.convert_dolly(item)
        
        if dataset_name == "Open-Orca/OpenOrca":
            return "instruction", cls.convert_orca(item)
        
        if dataset_name in ["tatsu-lab/alpaca", "WizardLM/WizardLM_evol_instruct_70k"]:
            return "instruction", cls.convert_alpaca(item)
        
        if dataset_name == "Anthropic/hh-rlhf":
            return "preference", cls.convert_hh_rlhf(item)
        
        if dataset_name in ["sahil2801/CodeAlpaca-20k", "TokenBender/code_instructions_122k_alpaca_style"]:
            return "code", cls.convert_code_alpaca(item)
        
        if dataset_name == "knkarthick/dialogsum":
            return "qa", cls.convert_dialogsum(item)
        
        # Generic handling by column structure
        if "messages" in columns:
            messages = item.get("messages", [])
            return "chat", cls.convert_messages(messages)
        
        if "instruction" in columns and "output" in columns:
            return "instruction", cls.convert_alpaca(item)
        
        if "question" in columns and "answer" in columns:
            return "qa", {
                "question": item.get("question", ""),
                "answer": item.get("answer", ""),
                "context": item.get("context", ""),
            }
        
        if "prompt" in columns and "completion" in columns:
            return "completion", {
                "prompt": item.get("prompt", ""),
                "completion": item.get("completion", ""),
            }
        
        if "text" in columns:
            text = item.get("text", "")
            if "### Human:" in text:
                return "dialogue", cls.convert_guanaco(text)
            return "completion", {"text": text}
        
        # Last resort: concatenate all string fields
        text_parts = []
        for key, value in item.items():
            if isinstance(value, str) and value.strip():
                text_parts.append(value.strip())
        return "completion", {"text": " ".join(text_parts)}


class TemplateBasedDataset:
    """
    Dataset that uses templates for consistent formatting and tokenization.
    
    This class loads data from HuggingFace datasets and applies template
    formatting with proper tokenization, attention masks, and loss masks.
    """
    
    def __init__(
        self,
        dataset_name: str,
        template: Union[InputTemplate, str],
        tokenizer: Any,
        seq_len: int = 512,
        batch_size: int = 8,
        split: str = "train",
        max_examples: int = 100_000,
        cache_tokenized: bool = True,
        auto_detect_template: bool = True,
    ):
        """
        Initialize a template-based dataset.
        
        Args:
            dataset_name: HuggingFace dataset name
            template: InputTemplate instance or template name string
            tokenizer: Tokenizer with encode/decode methods
            seq_len: Maximum sequence length
            batch_size: Batch size for training
            split: Dataset split to load
            max_examples: Maximum number of examples to load
            cache_tokenized: Whether to cache tokenized outputs
            auto_detect_template: If True, auto-detect template from data format
        """
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.split = split
        self.max_examples = max_examples
        self.cache_tokenized = cache_tokenized
        self.auto_detect_template = auto_detect_template
        
        # Template handling
        if isinstance(template, str):
            self.template = get_template(template)
        else:
            self.template = template
        
        # Storage
        self.examples: List[Dict[str, Any]] = []
        self.tokenized_cache: List[TokenizedOutput] = []
        self.columns: List[str] = []
        
        # Load data
        self._load_dataset()
    
    def _load_dataset(self):
        """Load and process the dataset."""
        try:
            from datasets import load_dataset
            
            print(f"Loading dataset '{self.dataset_name}' ({self.split})...")
            dataset = load_dataset(self.dataset_name, split=self.split)
            self.columns = dataset.column_names
            print(f"  Columns: {self.columns}")
            print(f"  Size: {len(dataset)} examples")
            
            # Optionally auto-detect template
            if self.auto_detect_template:
                detected = DatasetFormatConverter.detect_template(
                    self.dataset_name, self.columns
                )
                print(f"  Detected template type: {detected}")
            
            # Process examples
            processed = 0
            skipped = 0
            
            for i, item in enumerate(dataset):
                if processed >= self.max_examples:
                    break
                
                try:
                    # Convert item to template format
                    template_name, converted = DatasetFormatConverter.convert_item(
                        dict(item), self.dataset_name, self.columns
                    )
                    
                    # Use detected template if auto-detect is on
                    if self.auto_detect_template:
                        template = get_template(template_name)
                    else:
                        template = self.template
                    
                    # Validate
                    is_valid, error = template.validate(converted)
                    if not is_valid:
                        skipped += 1
                        continue
                    
                    # Store example
                    self.examples.append({
                        "data": converted,
                        "template_name": template_name,
                    })
                    
                    # Optionally cache tokenized output
                    if self.cache_tokenized:
                        output = template.format(converted)
                        tokenized = template.tokenize(output, self.tokenizer)
                        self.tokenized_cache.append(tokenized)
                    
                    processed += 1
                    
                    if (i + 1) % 1000 == 0:
                        print(f"  Processed {i + 1} items, {processed} valid, {skipped} skipped...")
                        
                except Exception as e:
                    skipped += 1
                    continue
            
            print(f"  Final: {processed} valid examples, {skipped} skipped")
            
        except ImportError:
            print("Error: 'datasets' library not found. Install with: pip install datasets")
        except Exception as e:
            print(f"Error loading dataset: {e}")
            import traceback
            traceback.print_exc()
    
    def __len__(self) -> int:
        """Number of examples."""
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> TokenizedOutput:
        """Get a tokenized example."""
        if self.cache_tokenized and idx < len(self.tokenized_cache):
            return self.tokenized_cache[idx]
        
        example = self.examples[idx]
        template = get_template(example["template_name"])
        output = template.format(example["data"])
        return template.tokenize(output, self.tokenizer)
    
    def get_batch(self, random_sample: bool = True) -> Tuple[Any, Any]:
        """
        Get a batch of input/target pairs for training.
        
        Returns:
            (x_tensor, y_tensor) tuple where:
            - x_tensor: Input token IDs (batch_size, seq_len)
            - y_tensor: Target token IDs (batch_size, seq_len) - shifted by 1
        """
        batch_x = []
        batch_y = []
        
        if random_sample:
            indices = random.sample(range(len(self.examples)), min(self.batch_size, len(self.examples)))
        else:
            indices = list(range(min(self.batch_size, len(self.examples))))
        
        for idx in indices:
            tokenized = self[idx]
            input_ids = list(tokenized.input_ids)
            
            # Pad or truncate to seq_len + 1 (for target shift)
            if len(input_ids) < self.seq_len + 1:
                pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
                input_ids.extend([pad_id] * (self.seq_len + 1 - len(input_ids)))
            else:
                input_ids = input_ids[:self.seq_len + 1]
            
            batch_x.extend(input_ids[:self.seq_len])
            batch_y.extend(input_ids[1:self.seq_len + 1])
        
        x_tensor = Tensor(batch_x, (self.batch_size, self.seq_len))
        y_tensor = Tensor(batch_y, (self.batch_size, self.seq_len))
        
        return x_tensor, y_tensor
    
    def get_batch_with_masks(self, random_sample: bool = True) -> BatchWithMasks:
        """
        Get a batch with attention and loss masks.
        
        This allows for selective loss computation, only training on
        assistant/output tokens rather than the full sequence.
        
        Returns:
            BatchWithMasks with all tensors for training
        """
        batch_input_ids = []
        batch_target_ids = []
        batch_attention_mask = []
        batch_loss_mask = []
        batch_segment_ids = []
        
        if random_sample:
            indices = random.sample(range(len(self.examples)), min(self.batch_size, len(self.examples)))
        else:
            indices = list(range(min(self.batch_size, len(self.examples))))
        
        for idx in indices:
            tokenized = self[idx]
            
            input_ids = list(tokenized.input_ids)
            attention_mask = list(tokenized.attention_mask)
            loss_mask = list(tokenized.loss_mask)
            segment_ids = list(tokenized.segment_ids) if tokenized.segment_ids else [0] * len(input_ids)
            
            # Pad or truncate
            pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
            
            if len(input_ids) < self.seq_len + 1:
                pad_len = self.seq_len + 1 - len(input_ids)
                input_ids.extend([pad_id] * pad_len)
                attention_mask.extend([0] * pad_len)
                loss_mask.extend([0] * pad_len)
                segment_ids.extend([0] * pad_len)
            else:
                input_ids = input_ids[:self.seq_len + 1]
                attention_mask = attention_mask[:self.seq_len + 1]
                loss_mask = loss_mask[:self.seq_len + 1]
                segment_ids = segment_ids[:self.seq_len + 1]
            
            batch_input_ids.extend(input_ids[:self.seq_len])
            batch_target_ids.extend(input_ids[1:self.seq_len + 1])
            batch_attention_mask.extend(attention_mask[:self.seq_len])
            batch_loss_mask.extend(loss_mask[1:self.seq_len + 1])  # Shift to match targets
            batch_segment_ids.extend(segment_ids[:self.seq_len])
        
        return BatchWithMasks(
            input_ids=Tensor(batch_input_ids, (self.batch_size, self.seq_len)),
            target_ids=Tensor(batch_target_ids, (self.batch_size, self.seq_len)),
            attention_mask=Tensor(batch_attention_mask, (self.batch_size, self.seq_len)),
            loss_mask=Tensor(batch_loss_mask, (self.batch_size, self.seq_len)),
            segment_ids=Tensor(batch_segment_ids, (self.batch_size, self.seq_len)),
        )
    
    def iterate_batches(
        self, 
        with_masks: bool = False
    ) -> Generator[Union[Tuple[Any, Any], BatchWithMasks], None, None]:
        """
        Iterate over all batches sequentially.
        
        Args:
            with_masks: If True, yield BatchWithMasks; otherwise yield (x, y) tuples
        """
        n_batches = len(self.examples) // self.batch_size
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            indices = list(range(start_idx, start_idx + self.batch_size))
            
            if with_masks:
                batch = self._get_batch_from_indices(indices, with_masks=True)
                yield batch
            else:
                batch = self._get_batch_from_indices(indices, with_masks=False)
                yield batch
    
    def _get_batch_from_indices(
        self,
        indices: List[int],
        with_masks: bool = False
    ) -> Union[Tuple[Any, Any], BatchWithMasks]:
        """Helper to get a batch from specific indices."""
        if with_masks:
            # Implementation similar to get_batch_with_masks but with specific indices
            batch_input_ids = []
            batch_target_ids = []
            batch_attention_mask = []
            batch_loss_mask = []
            
            for idx in indices:
                tokenized = self[idx]
                input_ids = list(tokenized.input_ids)
                attention_mask = list(tokenized.attention_mask)
                loss_mask = list(tokenized.loss_mask)
                
                pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
                if len(input_ids) < self.seq_len + 1:
                    pad_len = self.seq_len + 1 - len(input_ids)
                    input_ids.extend([pad_id] * pad_len)
                    attention_mask.extend([0] * pad_len)
                    loss_mask.extend([0] * pad_len)
                else:
                    input_ids = input_ids[:self.seq_len + 1]
                    attention_mask = attention_mask[:self.seq_len + 1]
                    loss_mask = loss_mask[:self.seq_len + 1]
                
                batch_input_ids.extend(input_ids[:self.seq_len])
                batch_target_ids.extend(input_ids[1:self.seq_len + 1])
                batch_attention_mask.extend(attention_mask[:self.seq_len])
                batch_loss_mask.extend(loss_mask[1:self.seq_len + 1])
            
            return BatchWithMasks(
                input_ids=Tensor(batch_input_ids, (len(indices), self.seq_len)),
                target_ids=Tensor(batch_target_ids, (len(indices), self.seq_len)),
                attention_mask=Tensor(batch_attention_mask, (len(indices), self.seq_len)),
                loss_mask=Tensor(batch_loss_mask, (len(indices), self.seq_len)),
            )
        else:
            batch_x = []
            batch_y = []
            
            for idx in indices:
                tokenized = self[idx]
                input_ids = list(tokenized.input_ids)
                
                pad_id = getattr(self.tokenizer, 'pad_token_id', 0)
                if len(input_ids) < self.seq_len + 1:
                    input_ids.extend([pad_id] * (self.seq_len + 1 - len(input_ids)))
                else:
                    input_ids = input_ids[:self.seq_len + 1]
                
                batch_x.extend(input_ids[:self.seq_len])
                batch_y.extend(input_ids[1:self.seq_len + 1])
            
            return (
                Tensor(batch_x, (len(indices), self.seq_len)),
                Tensor(batch_y, (len(indices), self.seq_len)),
            )


def create_templated_dataset(
    dataset_name: str,
    tokenizer: Any,
    template: Union[str, InputTemplate] = "auto",
    seq_len: int = 512,
    batch_size: int = 8,
    split: str = "train",
    max_examples: int = 100_000,
) -> TemplateBasedDataset:
    """
    Create a template-based dataset with automatic format detection.
    
    Args:
        dataset_name: HuggingFace dataset name
        tokenizer: Tokenizer with encode/decode methods
        template: Template name, InputTemplate instance, or "auto" for auto-detection
        seq_len: Maximum sequence length
        batch_size: Batch size
        split: Dataset split
        max_examples: Maximum examples to load
        
    Returns:
        TemplateBasedDataset ready for training
    """
    auto_detect = template == "auto"
    
    if auto_detect:
        template_instance = get_template("completion")  # Default, will be auto-detected
    elif isinstance(template, str):
        template_instance = get_template(template)
    else:
        template_instance = template
    
    return TemplateBasedDataset(
        dataset_name=dataset_name,
        template=template_instance,
        tokenizer=tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        split=split,
        max_examples=max_examples,
        auto_detect_template=auto_detect,
    )


# Convenience functions for common datasets
def load_guanaco(tokenizer: Any, **kwargs) -> TemplateBasedDataset:
    """Load OpenAssistant Guanaco dataset with dialogue template."""
    return create_templated_dataset(
        "timdettmers/openassistant-guanaco",
        tokenizer,
        template="dialogue",
        **kwargs
    )


def load_alpaca(tokenizer: Any, **kwargs) -> TemplateBasedDataset:
    """Load Stanford Alpaca dataset with instruction template."""
    return create_templated_dataset(
        "tatsu-lab/alpaca",
        tokenizer,
        template="instruction",
        **kwargs
    )


def load_dolly(tokenizer: Any, **kwargs) -> TemplateBasedDataset:
    """Load Databricks Dolly dataset with instruction template."""
    return create_templated_dataset(
        "databricks/databricks-dolly-15k",
        tokenizer,
        template="instruction",
        **kwargs
    )


def load_code_alpaca(tokenizer: Any, **kwargs) -> TemplateBasedDataset:
    """Load Code Alpaca dataset with code template."""
    return create_templated_dataset(
        "sahil2801/CodeAlpaca-20k",
        tokenizer,
        template="code",
        **kwargs
    )
