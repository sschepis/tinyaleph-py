"""
Training Template Configuration for ResoLLM

This module provides the mapping between datasets and training templates,
as well as the unified special tokens that match the inference format.

Template Types:
- completion: Raw text completion (no structure)
- chat: Multi-turn conversations (matches inference.py format)
- instruction: Instruction-following (Alpaca, Dolly, etc.)

The templates use the SAME special tokens as inference.py to ensure
training data matches inference format exactly.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum


# =============================================================================
# Training Template Type Enum
# =============================================================================

class TrainingTemplateType(Enum):
    """Types of training templates available."""
    COMPLETION = "completion"  # Raw text, no structure
    CHAT = "chat"              # Multi-turn conversation
    INSTRUCTION = "instruction"  # Instruction following


# =============================================================================
# Special Tokens (MUST match inference.py exactly)
# =============================================================================

# These tokens match the format used in inference.py:
# - CHAT_TEMPLATE, USER_TEMPLATE, ASSISTANT_TEMPLATE

SPECIAL_TOKENS = {
    # System tokens
    "system_start": "<|system|>",
    "system_end": "<|endofsystem|>",
    
    # User tokens
    "user_start": "<|user|>",
    "user_end": "<|endofuser|>",
    
    # Assistant tokens
    "assistant_start": "<|assistant|>",
    "assistant_end": "<|endofassistant|>",
    
    # End of text
    "eos": "<|endoftext|>",
}


# =============================================================================
# Training Template Base
# =============================================================================

@dataclass
class TrainingTemplate:
    """
    A training template that defines how data is formatted for training.
    """
    template_type: TrainingTemplateType
    system_prompt: str = "You are a helpful, harmless, and honest AI assistant."
    compute_loss_on_input: bool = False  # If True, compute loss on entire sequence
    
    def format_example(self, data: Dict[str, Any]) -> str:
        """Format a single training example."""
        if self.template_type == TrainingTemplateType.COMPLETION:
            return self._format_completion(data)
        elif self.template_type == TrainingTemplateType.CHAT:
            return self._format_chat(data)
        elif self.template_type == TrainingTemplateType.INSTRUCTION:
            return self._format_instruction(data)
        else:
            raise ValueError(f"Unknown template type: {self.template_type}")
    
    def _format_completion(self, data: Dict[str, Any]) -> str:
        """Format raw text completion - no structure."""
        text = data.get("text", data.get("content", ""))
        if isinstance(text, str):
            return text.strip()
        return str(text)
    
    def _format_chat(self, data: Dict[str, Any]) -> str:
        """
        Format chat/dialogue data.
        
        Uses the SAME format as inference.py:
        <|system|>
        {system_prompt}
        <|endofsystem|>
        <|user|>
        {user_message}
        <|endofuser|>
        <|assistant|>
        {assistant_message}
        <|endofassistant|>
        """
        parts = []
        
        # Add system prompt
        parts.append(f"{SPECIAL_TOKENS['system_start']}")
        parts.append(self.system_prompt)
        parts.append(f"{SPECIAL_TOKENS['system_end']}")
        
        # Handle different input formats
        if "messages" in data:
            # Chat messages format
            for msg in data["messages"]:
                role = msg.get("role", "user").lower()
                content = msg.get("content", "")
                
                if role in ("user", "human"):
                    parts.append(f"{SPECIAL_TOKENS['user_start']}")
                    parts.append(content)
                    parts.append(f"{SPECIAL_TOKENS['user_end']}")
                elif role in ("assistant", "bot", "gpt"):
                    parts.append(f"{SPECIAL_TOKENS['assistant_start']}")
                    parts.append(content)
                    parts.append(f"{SPECIAL_TOKENS['assistant_end']}")
        
        elif "turns" in data:
            # Dialogue turns format
            for turn in data["turns"]:
                speaker = turn.get("speaker", "").lower()
                text = turn.get("text", "")
                
                if speaker == "human":
                    parts.append(f"{SPECIAL_TOKENS['user_start']}")
                    parts.append(text)
                    parts.append(f"{SPECIAL_TOKENS['user_end']}")
                elif speaker == "assistant":
                    parts.append(f"{SPECIAL_TOKENS['assistant_start']}")
                    parts.append(text)
                    parts.append(f"{SPECIAL_TOKENS['assistant_end']}")
        
        elif "user" in data and "assistant" in data:
            # Simple user/assistant pair
            parts.append(f"{SPECIAL_TOKENS['user_start']}")
            parts.append(data["user"])
            parts.append(f"{SPECIAL_TOKENS['user_end']}")
            parts.append(f"{SPECIAL_TOKENS['assistant_start']}")
            parts.append(data["assistant"])
            parts.append(f"{SPECIAL_TOKENS['assistant_end']}")
        
        parts.append(SPECIAL_TOKENS['eos'])
        return "\n".join(parts)
    
    def _format_instruction(self, data: Dict[str, Any]) -> str:
        """
        Format instruction-following data.
        
        Uses chat format for consistency with inference:
        <|system|>
        {system_prompt}
        <|endofsystem|>
        <|user|>
        {instruction}
        [Input: {input}]  (if present)
        <|endofuser|>
        <|assistant|>
        {output}
        <|endofassistant|>
        """
        parts = []
        
        # System prompt
        parts.append(f"{SPECIAL_TOKENS['system_start']}")
        parts.append(self.system_prompt)
        parts.append(f"{SPECIAL_TOKENS['system_end']}")
        
        # User instruction
        instruction = data.get("instruction", data.get("prompt", data.get("question", "")))
        input_text = data.get("input", data.get("context", ""))
        
        parts.append(f"{SPECIAL_TOKENS['user_start']}")
        if input_text:
            parts.append(f"{instruction}\n\nInput: {input_text}")
        else:
            parts.append(instruction)
        parts.append(f"{SPECIAL_TOKENS['user_end']}")
        
        # Assistant output
        output = data.get("output", data.get("response", data.get("answer", "")))
        parts.append(f"{SPECIAL_TOKENS['assistant_start']}")
        parts.append(output)
        parts.append(f"{SPECIAL_TOKENS['assistant_end']}")
        
        parts.append(SPECIAL_TOKENS['eos'])
        return "\n".join(parts)


# =============================================================================
# Dataset to Template Mapping
# =============================================================================

# Map dataset names to their appropriate template type
DATASET_TEMPLATE_MAP: Dict[str, TrainingTemplateType] = {
    # Chat/Dialogue datasets
    "timdettmers/openassistant-guanaco": TrainingTemplateType.CHAT,
    "OpenAssistant/oasst2": TrainingTemplateType.CHAT,
    "lmsys/chatbot_arena_conversations": TrainingTemplateType.CHAT,
    "Anthropic/hh-rlhf": TrainingTemplateType.CHAT,
    "lonestar108/sexygpt": TrainingTemplateType.CHAT,
    
    # Instruction-following datasets
    "databricks/databricks-dolly-15k": TrainingTemplateType.INSTRUCTION,
    "Open-Orca/OpenOrca": TrainingTemplateType.INSTRUCTION,
    "tatsu-lab/alpaca": TrainingTemplateType.INSTRUCTION,
    "WizardLM/WizardLM_evol_instruct_70k": TrainingTemplateType.INSTRUCTION,
    "lonestar108/enlightenedllm": TrainingTemplateType.INSTRUCTION,
    
    # Code instruction datasets
    "sahil2801/CodeAlpaca-20k": TrainingTemplateType.INSTRUCTION,
    "TokenBender/code_instructions_122k_alpaca_style": TrainingTemplateType.INSTRUCTION,
    
    # QA/Summary datasets (treated as instruction)
    "knkarthick/dialogsum": TrainingTemplateType.INSTRUCTION,
    
    # Raw text datasets
    "lonestar108/rawdata": TrainingTemplateType.COMPLETION,
}


def get_template_for_dataset(dataset_name: str) -> TrainingTemplateType:
    """
    Get the appropriate template type for a dataset.
    
    Args:
        dataset_name: HuggingFace dataset name
        
    Returns:
        TrainingTemplateType enum value
    """
    if dataset_name in DATASET_TEMPLATE_MAP:
        return DATASET_TEMPLATE_MAP[dataset_name]
    
    # Auto-detect by name patterns
    name_lower = dataset_name.lower()
    
    if any(x in name_lower for x in ["chat", "convers", "dialog", "oasst"]):
        return TrainingTemplateType.CHAT
    
    if any(x in name_lower for x in ["instruct", "alpaca", "dolly", "orca", "wizardlm"]):
        return TrainingTemplateType.INSTRUCTION
    
    # Default to instruction for structured datasets
    return TrainingTemplateType.INSTRUCTION


def create_template(
    template_type: TrainingTemplateType,
    system_prompt: Optional[str] = None,
) -> TrainingTemplate:
    """
    Create a TrainingTemplate instance.
    
    Args:
        template_type: Type of template to create
        system_prompt: Optional custom system prompt
        
    Returns:
        TrainingTemplate instance
    """
    template = TrainingTemplate(template_type=template_type)
    if system_prompt:
        template.system_prompt = system_prompt
    return template


# =============================================================================
# Dataset Report Generation
# =============================================================================

@dataclass
class DatasetTemplateInfo:
    """Information about a dataset's template mapping."""
    dataset_name: str
    template_type: TrainingTemplateType
    is_mapped: bool  # True if explicitly in DATASET_TEMPLATE_MAP
    description: str


def generate_template_report(dataset_names: List[str]) -> List[DatasetTemplateInfo]:
    """
    Generate a report of template mappings for a list of datasets.
    
    Args:
        dataset_names: List of dataset names
        
    Returns:
        List of DatasetTemplateInfo objects
    """
    report = []
    
    for name in dataset_names:
        template_type = get_template_for_dataset(name)
        is_mapped = name in DATASET_TEMPLATE_MAP
        
        if template_type == TrainingTemplateType.CHAT:
            desc = "Multi-turn chat format with <|user|>/<|assistant|> tokens"
        elif template_type == TrainingTemplateType.INSTRUCTION:
            desc = "Instruction format with <|user|>/<|assistant|> tokens"
        else:
            desc = "Raw text completion (no special formatting)"
        
        report.append(DatasetTemplateInfo(
            dataset_name=name,
            template_type=template_type,
            is_mapped=is_mapped,
            description=desc,
        ))
    
    return report


def print_template_report(dataset_names: List[str]) -> str:
    """
    Print a formatted template report.
    
    Args:
        dataset_names: List of dataset names
        
    Returns:
        Formatted report string
    """
    report = generate_template_report(dataset_names)
    
    lines = []
    lines.append("=" * 80)
    lines.append("TRAINING TEMPLATE REPORT")
    lines.append("=" * 80)
    lines.append("")
    
    # Group by template type
    by_type: Dict[TrainingTemplateType, List[DatasetTemplateInfo]] = {}
    for info in report:
        if info.template_type not in by_type:
            by_type[info.template_type] = []
        by_type[info.template_type].append(info)
    
    # Print each group
    for template_type in [TrainingTemplateType.CHAT, TrainingTemplateType.INSTRUCTION, TrainingTemplateType.COMPLETION]:
        if template_type not in by_type:
            continue
            
        datasets = by_type[template_type]
        lines.append(f"\n{template_type.value.upper()} TEMPLATE ({len(datasets)} datasets)")
        lines.append("-" * 60)
        
        if template_type == TrainingTemplateType.CHAT:
            lines.append("Format: Multi-turn chat conversations")
            lines.append("Tokens: <|user|>, <|endofuser|>, <|assistant|>, <|endofassistant|>")
            lines.append("Loss: Computed on assistant responses only")
        elif template_type == TrainingTemplateType.INSTRUCTION:
            lines.append("Format: Instruction-response pairs")
            lines.append("Tokens: <|user|>, <|endofuser|>, <|assistant|>, <|endofassistant|>")
            lines.append("Loss: Computed on response only")
        else:
            lines.append("Format: Raw text completion")
            lines.append("Tokens: None (plain text)")
            lines.append("Loss: Computed on entire sequence")
        
        lines.append("")
        for info in datasets:
            mapped_str = "✓" if info.is_mapped else "~"
            lines.append(f"  {mapped_str} {info.dataset_name}")
    
    lines.append("")
    lines.append("=" * 80)
    lines.append("Legend: ✓ = Explicitly mapped, ~ = Auto-detected")
    lines.append("=" * 80)
    
    return "\n".join(lines)


# =============================================================================
# Recommended Datasets
# =============================================================================

# Curated list of recommended datasets for training
RECOMMENDED_DATASETS = [
    "timdettmers/openassistant-guanaco",  # High-quality chat
    "databricks/databricks-dolly-15k",     # Instruction following
    "tatsu-lab/alpaca",                     # Instruction tuning
]

# Extended list for more comprehensive training
EXTENDED_DATASETS = [
    # Chat
    "timdettmers/openassistant-guanaco",
    "lonestar108/sexygpt",
    
    # Instruction
    "databricks/databricks-dolly-15k",
    "tatsu-lab/alpaca",
    "Open-Orca/OpenOrca",
    
    # Code
    "sahil2801/CodeAlpaca-20k",
]


# =============================================================================
# CLI Helper
# =============================================================================

def main():
    """Print template report for recommended datasets."""
    print("\n" + "=" * 80)
    print("Available Training Template Types")
    print("=" * 80)
    print("""
1. COMPLETION Template
   - Raw text without structure
   - Used for: Plain text files, books, articles
   - Loss: Entire sequence

2. CHAT Template
   - Multi-turn conversations
   - Used for: OpenAssistant, chatbot data
   - Loss: Assistant responses only

3. INSTRUCTION Template
   - Instruction → Response format
   - Used for: Alpaca, Dolly, code instructions
   - Loss: Response only
""")
    
    print("\n" + print_template_report(RECOMMENDED_DATASETS))
    
    print("\n\nExtended Dataset List:")
    print(print_template_report(EXTENDED_DATASETS))


if __name__ == "__main__":
    main()
