"""
Templatized Structured Data Input System for ResoLLM

This module provides a flexible template system for structuring model inputs,
including tokenization, formatting, attention masks, and loss masks.

Template Types:
- ChatTemplate: Multi-turn conversation format
- InstructionTemplate: Single instruction-response format
- QATemplate: Question-answer format
- CompletionTemplate: Simple text completion
- DialogueTemplate: Two-party dialogue format
- CodeTemplate: Code generation/completion format
- PreferenceTemplate: DPO/preference learning format
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union, Tuple
from enum import Enum
import re


# ==============================================================================
# Core Data Structures
# ==============================================================================

class Role(Enum):
    """Standard role identifiers for conversation participants."""
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    FUNCTION = "function"
    TOOL = "tool"


@dataclass
class Message:
    """Represents a single message in a conversation."""
    role: Role
    content: str
    name: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"role": self.role.value, "content": self.content}
        if self.name:
            result["name"] = self.name
        if self.metadata:
            result["metadata"] = self.metadata
        return result
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        role = Role(data["role"]) if isinstance(data["role"], str) else data["role"]
        return cls(
            role=role,
            content=data["content"],
            name=data.get("name"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TemplateOutput:
    """Output from template formatting, ready for tokenization."""
    text: str
    input_text: str
    target_text: str
    segments: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def get_loss_mask_positions(self) -> List[Tuple[int, int]]:
        """Return (start, end) positions for loss computation."""
        return [(s["start"], s["end"]) for s in self.segments if s.get("compute_loss", True)]


@dataclass 
class TokenizedOutput:
    """Fully tokenized output with all masks and ids."""
    input_ids: List[int]
    attention_mask: List[int]
    loss_mask: List[int]
    segment_ids: Optional[List[int]] = None
    position_ids: Optional[List[int]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        result = {
            "input_ids": self.input_ids,
            "attention_mask": self.attention_mask,
            "loss_mask": self.loss_mask,
        }
        if self.segment_ids:
            result["segment_ids"] = self.segment_ids
        if self.position_ids:
            result["position_ids"] = self.position_ids
        result["metadata"] = self.metadata
        return result


# ==============================================================================
# Special Token Definitions
# ==============================================================================

@dataclass
class SpecialTokens:
    """Collection of special tokens for a template."""
    bos: str = "<|bos|>"
    eos: str = "<|eos|>"
    pad: str = "<|pad|>"
    sep: str = "<|sep|>"
    system_start: str = "<|system|>"
    system_end: str = "<|/system|>"
    user_start: str = "<|user|>"
    user_end: str = "<|/user|>"
    assistant_start: str = "<|assistant|>"
    assistant_end: str = "<|/assistant|>"
    think_start: str = "<|think|>"
    think_end: str = "<|/think|>"
    code_start: str = "<|code|>"
    code_end: str = "<|/code|>"
    
    def all_tokens(self) -> List[str]:
        return [
            self.bos, self.eos, self.pad, self.sep,
            self.system_start, self.system_end,
            self.user_start, self.user_end,
            self.assistant_start, self.assistant_end,
            self.think_start, self.think_end,
            self.code_start, self.code_end,
        ]
    
    def get_role_tokens(self, role: Role) -> Tuple[str, str]:
        mapping = {
            Role.SYSTEM: (self.system_start, self.system_end),
            Role.USER: (self.user_start, self.user_end),
            Role.ASSISTANT: (self.assistant_start, self.assistant_end),
            Role.FUNCTION: (self.code_start, self.code_end),
            Role.TOOL: (self.code_start, self.code_end),
        }
        return mapping.get(role, (self.sep, self.sep))


DEFAULT_SPECIAL_TOKENS = SpecialTokens()


# ==============================================================================
# Abstract Base Template
# ==============================================================================

class InputTemplate(ABC):
    """Abstract base class for all input templates."""
    
    def __init__(
        self,
        name: str,
        special_tokens: Optional[SpecialTokens] = None,
        max_length: int = 2048,
        truncation_side: str = "left",
        add_bos: bool = True,
        add_eos: bool = True,
    ):
        self.name = name
        self.special_tokens = special_tokens or DEFAULT_SPECIAL_TOKENS
        self.max_length = max_length
        self.truncation_side = truncation_side
        self.add_bos = add_bos
        self.add_eos = add_eos
    
    @abstractmethod
    def format(self, data: Any) -> TemplateOutput:
        """Format raw data into a TemplateOutput."""
        pass
    
    @abstractmethod
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        """Validate input data for this template."""
        pass
    
    def tokenize(
        self,
        output: TemplateOutput,
        tokenizer: Any,
        return_tensors: Optional[str] = None,
    ) -> TokenizedOutput:
        """Tokenize a TemplateOutput using the provided tokenizer."""
        full_text = output.text
        if self.add_bos:
            full_text = self.special_tokens.bos + full_text
        if self.add_eos:
            full_text = full_text + self.special_tokens.eos
            
        if hasattr(tokenizer, 'encode'):
            input_ids = tokenizer.encode(full_text)
        else:
            input_ids = list(tokenizer(full_text))
            
        if len(input_ids) > self.max_length:
            if self.truncation_side == "left":
                input_ids = input_ids[-self.max_length:]
            else:
                input_ids = input_ids[:self.max_length]
        
        attention_mask = [1] * len(input_ids)
        loss_mask = self._compute_loss_mask(output, input_ids, tokenizer)
        segment_ids = self._compute_segment_ids(output, input_ids, tokenizer)
        
        result = TokenizedOutput(
            input_ids=input_ids,
            attention_mask=attention_mask,
            loss_mask=loss_mask,
            segment_ids=segment_ids,
            metadata=output.metadata,
        )
        
        if return_tensors == "pt":
            import torch
            result.input_ids = torch.tensor(result.input_ids)
            result.attention_mask = torch.tensor(result.attention_mask)
            result.loss_mask = torch.tensor(result.loss_mask)
            if result.segment_ids:
                result.segment_ids = torch.tensor(result.segment_ids)
        elif return_tensors == "np":
            import numpy as np
            result.input_ids = np.array(result.input_ids)
            result.attention_mask = np.array(result.attention_mask)
            result.loss_mask = np.array(result.loss_mask)
            if result.segment_ids:
                result.segment_ids = np.array(result.segment_ids)
        
        return result
    
    def _compute_loss_mask(
        self,
        output: TemplateOutput,
        input_ids: List[int],
        tokenizer: Any,
    ) -> List[int]:
        loss_mask = [0] * len(input_ids)
        if not output.segments:
            input_len = len(tokenizer.encode(output.input_text)) if hasattr(tokenizer, 'encode') else len(output.input_text)
            for i in range(min(input_len, len(loss_mask)), len(loss_mask)):
                loss_mask[i] = 1
        else:
            for segment in output.segments:
                if segment.get("compute_loss", True):
                    start = segment.get("token_start", 0)
                    end = segment.get("token_end", len(loss_mask))
                    for i in range(start, min(end, len(loss_mask))):
                        loss_mask[i] = 1
        return loss_mask
    
    def _compute_segment_ids(
        self,
        output: TemplateOutput,
        input_ids: List[int],
        tokenizer: Any,
    ) -> Optional[List[int]]:
        if not output.segments:
            return None
        segment_ids = [0] * len(input_ids)
        for i, segment in enumerate(output.segments):
            start = segment.get("token_start", 0)
            end = segment.get("token_end", len(segment_ids))
            for j in range(start, min(end, len(segment_ids))):
                segment_ids[j] = i
        return segment_ids
    
    def __call__(self, data: Any, tokenizer: Any = None) -> Union[TemplateOutput, TokenizedOutput]:
        is_valid, error = self.validate(data)
        if not is_valid:
            raise ValueError(f"Invalid data for template '{self.name}': {error}")
        output = self.format(data)
        if tokenizer is not None:
            return self.tokenize(output, tokenizer)
        return output


# ==============================================================================
# Concrete Template Implementations
# ==============================================================================

class ChatTemplate(InputTemplate):
    """Template for multi-turn chat conversations."""
    
    def __init__(self, name: str = "chat", train_on_input: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.train_on_input = train_on_input
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        if "messages" not in data:
            return False, "Data must contain 'messages' key"
        if not isinstance(data["messages"], list) or len(data["messages"]) == 0:
            return False, "'messages' must be a non-empty list"
        for i, msg in enumerate(data["messages"]):
            if not isinstance(msg, dict) or "role" not in msg or "content" not in msg:
                return False, f"Message {i} must have 'role' and 'content'"
        return True, None
    
    def format(self, data: Dict[str, Any]) -> TemplateOutput:
        messages = data["messages"]
        system_prompt = data.get("system_prompt")
        parts, segments, input_parts, target_parts = [], [], [], []
        current_pos = 0
        
        if system_prompt:
            system_text = f"{self.special_tokens.system_start}{system_prompt}{self.special_tokens.system_end}\n"
            parts.append(system_text)
            input_parts.append(system_text)
            segments.append({"role": "system", "text": system_text, "start": current_pos, "end": current_pos + len(system_text), "compute_loss": False})
            current_pos += len(system_text)
        
        for msg in messages:
            role_str = msg["role"] if isinstance(msg["role"], str) else msg["role"].value
            role = Role(role_str) if isinstance(role_str, str) else msg["role"]
            start_tok, end_tok = self.special_tokens.get_role_tokens(role)
            msg_text = f"{start_tok}{msg['content']}{end_tok}\n"
            parts.append(msg_text)
            is_assistant = role == Role.ASSISTANT
            compute_loss = is_assistant or (self.train_on_input and role == Role.USER)
            (target_parts if is_assistant else input_parts).append(msg_text)
            segments.append({"role": role_str, "text": msg_text, "start": current_pos, "end": current_pos + len(msg_text), "compute_loss": compute_loss})
            current_pos += len(msg_text)
        
        return TemplateOutput(text="".join(parts), input_text="".join(input_parts), target_text="".join(target_parts), segments=segments, metadata={"template": self.name, "num_turns": len(messages)})


class InstructionTemplate(InputTemplate):
    """Template for instruction-following format."""
    
    DEFAULT_SYSTEM = "You are a helpful assistant that follows instructions carefully."
    
    def __init__(self, name: str = "instruction", include_system: bool = True, **kwargs):
        super().__init__(name=name, **kwargs)
        self.include_system = include_system
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        if "instruction" not in data or "output" not in data:
            return False, "Data must contain 'instruction' and 'output' keys"
        return True, None
    
    def format(self, data: Dict[str, Any]) -> TemplateOutput:
        instruction, input_text, output = data["instruction"], data.get("input", ""), data["output"]
        system = data.get("system", self.DEFAULT_SYSTEM if self.include_system else "")
        parts, segments, current_pos = [], [], 0
        
        if system:
            system_text = f"{self.special_tokens.system_start}{system}{self.special_tokens.system_end}\n"
            parts.append(system_text)
            segments.append({"role": "system", "text": system_text, "start": current_pos, "end": current_pos + len(system_text), "compute_loss": False})
            current_pos += len(system_text)
        
        user_content = f"{instruction}\n\nInput: {input_text}" if input_text else instruction
        user_text = f"{self.special_tokens.user_start}{user_content}{self.special_tokens.user_end}\n"
        parts.append(user_text)
        segments.append({"role": "user", "text": user_text, "start": current_pos, "end": current_pos + len(user_text), "compute_loss": False})
        current_pos += len(user_text)
        
        assistant_text = f"{self.special_tokens.assistant_start}{output}{self.special_tokens.assistant_end}\n"
        parts.append(assistant_text)
        segments.append({"role": "assistant", "text": assistant_text, "start": current_pos, "end": current_pos + len(assistant_text), "compute_loss": True})
        
        return TemplateOutput(text="".join(parts), input_text="".join(parts[:-1]), target_text=assistant_text, segments=segments, metadata={"template": self.name, "has_input": bool(input_text)})


class QATemplate(InputTemplate):
    """Template for question-answering format."""
    
    def __init__(self, name: str = "qa", context_prefix: str = "Context: ", question_prefix: str = "Question: ", answer_prefix: str = "Answer: ", **kwargs):
        super().__init__(name=name, **kwargs)
        self.context_prefix, self.question_prefix, self.answer_prefix = context_prefix, question_prefix, answer_prefix
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, dict) or "question" not in data or "answer" not in data:
            return False, "Data must contain 'question' and 'answer' keys"
        return True, None
    
    def format(self, data: Dict[str, Any]) -> TemplateOutput:
        question, answer, context = data["question"], data["answer"], data.get("context", "")
        parts, segments, current_pos = [], [], 0
        
        if context:
            context_text = f"{self.context_prefix}{context}\n\n"
            parts.append(context_text)
            segments.append({"role": "context", "text": context_text, "start": current_pos, "end": current_pos + len(context_text), "compute_loss": False})
            current_pos += len(context_text)
        
        question_text = f"{self.question_prefix}{question}\n"
        parts.append(question_text)
        segments.append({"role": "question", "text": question_text, "start": current_pos, "end": current_pos + len(question_text), "compute_loss": False})
        current_pos += len(question_text)
        
        answer_text = f"{self.answer_prefix}{answer}\n"
        parts.append(answer_text)
        segments.append({"role": "answer", "text": answer_text, "start": current_pos, "end": current_pos + len(answer_text), "compute_loss": True})
        
        return TemplateOutput(text="".join(parts), input_text="".join(parts[:-1]) + self.answer_prefix, target_text=answer + "\n", segments=segments, metadata={"template": self.name, "has_context": bool(context)})


class CompletionTemplate(InputTemplate):
    """Simple text completion template."""
    
    def __init__(self, name: str = "completion", prompt_suffix: str = "", **kwargs):
        super().__init__(name=name, **kwargs)
        self.prompt_suffix = prompt_suffix
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if isinstance(data, str):
            return True, None
        if isinstance(data, dict) and ("text" in data or ("prompt" in data and "completion" in data)):
            return True, None
        return False, "Data must be string or dict with 'text' or 'prompt'/'completion'"
    
    def format(self, data: Union[str, Dict[str, Any]]) -> TemplateOutput:
        if isinstance(data, str):
            return TemplateOutput(text=data, input_text="", target_text=data, segments=[{"role": "text", "text": data, "start": 0, "end": len(data), "compute_loss": True}], metadata={"template": self.name, "mode": "pure"})
        
        prompt = data.get("prompt", "")
        completion = data.get("completion", data.get("text", ""))
        prompt_with_suffix = prompt + self.prompt_suffix
        full_text = prompt_with_suffix + completion
        segments = []
        if prompt:
            segments.append({"role": "prompt", "text": prompt_with_suffix, "start": 0, "end": len(prompt_with_suffix), "compute_loss": False})
        segments.append({"role": "completion", "text": completion, "start": len(prompt_with_suffix), "end": len(full_text), "compute_loss": True})
        return TemplateOutput(text=full_text, input_text=prompt_with_suffix, target_text=completion, segments=segments, metadata={"template": self.name, "mode": "prompt_completion"})


class DialogueTemplate(InputTemplate):
    """Template for two-party dialogue (Human/Assistant)."""
    
    def __init__(self, name: str = "dialogue", human_prefix: str = "### Human: ", assistant_prefix: str = "### Assistant: ", train_on_human: bool = False, **kwargs):
        super().__init__(name=name, **kwargs)
        self.human_prefix, self.assistant_prefix, self.train_on_human = human_prefix, assistant_prefix, train_on_human
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if isinstance(data, str) or (isinstance(data, dict) and ("turns" in data or "text" in data)):
            return True, None
        return False, "Data must be string or dict with 'turns' or 'text'"
    
    def _parse_text(self, text: str) -> List[Dict[str, str]]:
        turns, pattern = [], rf"({re.escape(self.human_prefix)}|{re.escape(self.assistant_prefix)})"
        parts, current_speaker = re.split(pattern, text), None
        for part in parts:
            if part == self.human_prefix:
                current_speaker = "Human"
            elif part == self.assistant_prefix:
                current_speaker = "Assistant"
            elif current_speaker and part.strip():
                turns.append({"speaker": current_speaker, "text": part.strip()})
        return turns
    
    def format(self, data: Union[str, Dict[str, Any]]) -> TemplateOutput:
        turns = self._parse_text(data if isinstance(data, str) else data.get("text", "")) if isinstance(data, str) or "text" in data else data["turns"]
        parts, segments, current_pos = [], [], 0
        
        for turn in turns:
            speaker, text = turn["speaker"], turn["text"]
            prefix = self.human_prefix if speaker == "Human" else self.assistant_prefix
            compute_loss = speaker != "Human" or self.train_on_human
            turn_text = f"{prefix}{text}\n"
            parts.append(turn_text)
            segments.append({"role": speaker.lower(), "text": turn_text, "start": current_pos, "end": current_pos + len(turn_text), "compute_loss": compute_loss})
            current_pos += len(turn_text)
        
        input_parts = [p for p, s in zip(parts, segments) if not s["compute_loss"]]
        target_parts = [p for p, s in zip(parts, segments) if s["compute_loss"]]
        return TemplateOutput(text="".join(parts), input_text="".join(input_parts), target_text="".join(target_parts), segments=segments, metadata={"template": self.name, "num_turns": len(turns)})


class CodeTemplate(InputTemplate):
    """Template for code generation and completion."""
    
    def __init__(self, name: str = "code", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, dict) or "code" not in data:
            return False, "Data must contain 'code' key"
        return True, None
    
    def format(self, data: Dict[str, Any]) -> TemplateOutput:
        instruction = data.get("instruction", "Write the following code:")
        code, language, context = data["code"], data.get("language", ""), data.get("context", "")
        parts, segments, current_pos = [], [], 0
        
        instruction_text = f"{self.special_tokens.user_start}{instruction}{self.special_tokens.user_end}\n"
        parts.append(instruction_text)
        segments.append({"role": "instruction", "text": instruction_text, "start": current_pos, "end": current_pos + len(instruction_text), "compute_loss": False})
        current_pos += len(instruction_text)
        
        if context:
            context_text = f"Context:\n{context}\n\n"
            parts.append(context_text)
            segments.append({"role": "context", "text": context_text, "start": current_pos, "end": current_pos + len(context_text), "compute_loss": False})
            current_pos += len(context_text)
        
        lang_marker = f"```{language}\n" if language else f"{self.special_tokens.code_start}\n"
        lang_end = "```\n" if language else f"{self.special_tokens.code_end}\n"
        code_text = f"{lang_marker}{code}\n{lang_end}"
        parts.append(code_text)
        segments.append({"role": "code", "text": code_text, "start": current_pos, "end": current_pos + len(code_text), "compute_loss": True})
        
        return TemplateOutput(text="".join(parts), input_text="".join(parts[:-1]) + lang_marker, target_text=code + "\n" + lang_end, segments=segments, metadata={"template": self.name, "language": language})


class PreferenceTemplate(InputTemplate):
    """Template for preference learning (DPO, RLHF)."""
    
    def __init__(self, name: str = "preference", **kwargs):
        super().__init__(name=name, **kwargs)
    
    def validate(self, data: Any) -> Tuple[bool, Optional[str]]:
        if not isinstance(data, dict):
            return False, "Data must be a dictionary"
        if "prompt" not in data or ("chosen" not in data and "rejected" not in data):
            return False, "Data must contain 'prompt' and at least one of 'chosen'/'rejected'"
        return True, None
    
    def format(self, data: Dict[str, Any]) -> TemplateOutput:
        prompt, chosen, rejected = data["prompt"], data.get("chosen", ""), data.get("rejected", "")
        prompt_text = f"{self.special_tokens.user_start}{prompt}{self.special_tokens.user_end}\n"
        
        segments = [{"role": "prompt", "text": prompt_text, "start": 0, "end": len(prompt_text), "compute_loss": False}]
        if chosen:
            chosen_text = f"{self.special_tokens.assistant_start}{chosen}{self.special_tokens.assistant_end}"
            segments.append({"role": "chosen", "text": chosen_text, "start": len(prompt_text), "end": len(prompt_text) + len(chosen_text), "compute_loss": True, "preference": "chosen"})
        
        return TemplateOutput(text=prompt_text + (f"{self.special_tokens.assistant_start}{chosen}{self.special_tokens.assistant_end}" if chosen else ""), input_text=prompt_text, target_text=chosen, segments=segments, metadata={"template": self.name, "has_chosen": bool(chosen), "has_rejected": bool(rejected), "rejected": rejected})


# ==============================================================================
# Template Registry and Factory
# ==============================================================================

class TemplateRegistry:
    """Registry for managing template types."""
    
    _templates: Dict[str, type] = {}
    _instances: Dict[str, InputTemplate] = {}
    
    @classmethod
    def register(cls, name: str, template_class: type) -> None:
        cls._templates[name] = template_class
    
    @classmethod
    def get_class(cls, name: str) -> type:
        if name not in cls._templates:
            raise KeyError(f"Template '{name}' not registered. Available: {list(cls._templates.keys())}")
        return cls._templates[name]
    
    @classmethod
    def create(cls, name: str, **kwargs) -> InputTemplate:
        template_class = cls.get_class(name)
        return template_class(name=name, **kwargs)
    
    @classmethod
    def get_or_create(cls, name: str, **kwargs) -> InputTemplate:
        cache_key = f"{name}_{hash(frozenset(kwargs.items()))}"
        if cache_key not in cls._instances:
            cls._instances[cache_key] = cls.create(name, **kwargs)
        return cls._instances[cache_key]
    
    @classmethod
    def list_templates(cls) -> List[str]:
        return list(cls._templates.keys())


# Register built-in templates
TemplateRegistry.register("chat", ChatTemplate)
TemplateRegistry.register("instruction", InstructionTemplate)
TemplateRegistry.register("qa", QATemplate)
TemplateRegistry.register("completion", CompletionTemplate)
TemplateRegistry.register("dialogue", DialogueTemplate)
TemplateRegistry.register("code", CodeTemplate)
TemplateRegistry.register("preference", PreferenceTemplate)


def get_template(name: str, **kwargs) -> InputTemplate:
    """Convenience function to get a template by name."""
    return TemplateRegistry.get_or_create(name, **kwargs)


# ==============================================================================
# Dataset Integration Helpers
# ==============================================================================

class TemplatedDataset:
    """Wrapper to apply templates to datasets."""
    
    def __init__(self, dataset: Any, template: InputTemplate, tokenizer: Any = None):
        self.dataset = dataset
        self.template = template
        self.tokenizer = tokenizer
    
    def __len__(self) -> int:
        return len(self.dataset)
    
    def __getitem__(self, idx: int) -> Union[TemplateOutput, TokenizedOutput]:
        item = self.dataset[idx]
        return self.template(item, self.tokenizer)
    
    def __iter__(self):
        for item in self.dataset:
            yield self.template(item, self.tokenizer)


def apply_template_to_batch(batch: List[Any], template: InputTemplate, tokenizer: Any = None, pad_to_max: bool = True) -> Dict[str, List]:
    """Apply template to a batch and collate results."""
    outputs = [template(item, tokenizer) for item in batch]
    
    if tokenizer is None:
        return {"outputs": outputs}
    
    max_len = max(len(o.input_ids) for o in outputs)
    pad_id = tokenizer.pad_token_id if hasattr(tokenizer, 'pad_token_id') else 0
    
    result = {"input_ids": [], "attention_mask": [], "loss_mask": []}
    for o in outputs:
        pad_len = max_len - len(o.input_ids) if pad_to_max else 0
        result["input_ids"].append(list(o.input_ids) + [pad_id] * pad_len)
        result["attention_mask"].append(list(o.attention_mask) + [0] * pad_len)
        result["loss_mask"].append(list(o.loss_mask) + [0] * pad_len)
    
    return result
