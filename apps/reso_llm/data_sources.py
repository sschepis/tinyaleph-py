"""
Data sources for training the ResoLLM model.

Provides:
- HuggingFace dataset loading (OpenAssistant-Guanaco, etc.)
- LMStudio/OpenAI-compatible teacher API
- Preference data generation for RLHF/DPO
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import requests


# =============================================================================
# HuggingFace Dataset Loader
# =============================================================================

@dataclass
class ConversationTurn:
    """A single turn in a conversation."""
    role: str  # "user", "assistant", "system"
    content: str


@dataclass
class Conversation:
    """A multi-turn conversation."""
    turns: List[ConversationTurn]
    
    def to_chat_format(self) -> str:
        """Convert to chat format string for training."""
        parts = []
        for turn in self.turns:
            if turn.role == "user":
                parts.append(f"<|user|>\n{turn.content}\n<|endofuser|>")
            elif turn.role == "assistant":
                parts.append(f"<|assistant|>\n{turn.content}\n<|endofassistant|>")
            elif turn.role == "system":
                parts.append(f"<|system|>\n{turn.content}\n<|endofsystem|>")
        return "\n".join(parts)


def load_guanaco_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: str = "data/hf_cache"
) -> Generator[Conversation, None, None]:
    """
    Load OpenAssistant-Guanaco dataset from HuggingFace.
    
    Args:
        split: Dataset split ("train" or "test")
        max_samples: Maximum samples to load (None for all)
        cache_dir: Directory to cache downloaded data
    
    Yields:
        Conversation objects
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(
        "timdettmers/openassistant-guanaco",
        split=split,
        cache_dir=cache_dir
    )
    
    count = 0
    for item in dataset:
        if max_samples and count >= max_samples:
            break
        
        # Parse the text format (### Human: ... ### Assistant: ...)
        text = item.get("text", "")
        turns = _parse_guanaco_format(text)
        
        if turns:
            yield Conversation(turns=turns)
            count += 1


def _parse_guanaco_format(text: str) -> List[ConversationTurn]:
    """Parse the Guanaco ### Human: / ### Assistant: format."""
    turns = []
    
    # Split on markers
    parts = text.split("###")
    
    for part in parts:
        part = part.strip()
        if not part:
            continue
        
        if part.startswith("Human:"):
            content = part[6:].strip()
            turns.append(ConversationTurn(role="user", content=content))
        elif part.startswith("Assistant:"):
            content = part[10:].strip()
            turns.append(ConversationTurn(role="assistant", content=content))
    
    return turns


def load_oasst2_dataset(
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: str = "data/hf_cache",
    lang: str = "en"
) -> Generator[Conversation, None, None]:
    """
    Load OpenAssistant/oasst2 dataset from HuggingFace.
    
    Args:
        split: Dataset split ("train" or "validation")
        max_samples: Maximum samples to load (None for all)
        cache_dir: Directory to cache downloaded data
        lang: Language filter (default "en" for English only)
    
    Yields:
        Conversation objects (English only by default)
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Install datasets: pip install datasets")
    
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load the dataset
    dataset = load_dataset(
        "OpenAssistant/oasst2",
        split=split,
        cache_dir=cache_dir
    )
    
    count = 0
    for item in dataset:
        if max_samples and count >= max_samples:
            break
        
        # Filter by language
        item_lang = item.get("lang", "")
        if item_lang != lang:
            continue
        
        # oasst2 format: has 'text', 'role', 'lang' fields
        # Each row is a single message, need to group by conversation
        # For simplicity, treat each message as a single-turn conversation
        text = item.get("text", "")
        role = item.get("role", "user")
        
        if not text:
            continue
        
        # Map oasst2 roles to our format
        if role == "prompter":
            role = "user"
        
        turns = [ConversationTurn(role=role, content=text)]
        
        if turns:
            yield Conversation(turns=turns)
            count += 1


def load_conversation_dataset(
    dataset_name: str = "timdettmers/openassistant-guanaco",
    split: str = "train",
    max_samples: Optional[int] = None,
    cache_dir: str = "data/hf_cache"
) -> Generator[Conversation, None, None]:
    """
    Generic loader for conversation datasets.
    
    Supports:
    - timdettmers/openassistant-guanaco
    - OpenAssistant/oasst1
    - OpenAssistant/oasst2 (filters to lang=en only)
    - Other compatible datasets
    """
    if "guanaco" in dataset_name.lower():
        yield from load_guanaco_dataset(split, max_samples, cache_dir)
    elif "oasst2" in dataset_name.lower():
        yield from load_oasst2_dataset(split, max_samples, cache_dir, lang="en")
    else:
        # Generic HF dataset loading
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError("Install datasets: pip install datasets")
        
        os.makedirs(cache_dir, exist_ok=True)
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        
        count = 0
        for item in dataset:
            if max_samples and count >= max_samples:
                break
            
            # Try different formats
            turns = []
            
            # Format 1: 'messages' field with list of dicts
            if "messages" in item:
                for msg in item["messages"]:
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    turns.append(ConversationTurn(role=role, content=content))
            
            # Format 2: 'text' field
            elif "text" in item:
                text = item["text"]
                turns = _parse_guanaco_format(text)
            
            # Format 3: 'instruction' + 'output' fields
            elif "instruction" in item and "output" in item:
                turns = [
                    ConversationTurn(role="user", content=item["instruction"]),
                    ConversationTurn(role="assistant", content=item["output"]),
                ]
            
            # Format 4: 'user' + 'assistant' fields (e.g., lonestar108/companion)
            elif "user" in item and "assistant" in item:
                turns = [
                    ConversationTurn(role="user", content=item["user"]),
                    ConversationTurn(role="assistant", content=item["assistant"]),
                ]
            
            if turns:
                yield Conversation(turns=turns)
                count += 1


# =============================================================================
# LMStudio / OpenAI-Compatible Teacher
# =============================================================================

@dataclass
class LMStudioConfig:
    """Configuration for LMStudio API connection."""
    base_url: str = "http://localhost:1234"  # Base URL without /v1
    model: str = "llama-3.1-8b-lexi-uncensored-v2"  # Default to smaller model
    api_key: str = ""  # LMStudio token (optional)
    temperature: float = 0.7
    max_tokens: int = 512
    timeout: int = 120
    context_length: int = 4096


class LMStudioTeacher:
    """
    Teacher model using LMStudio's native /api/v1/chat API.
    
    Uses the new LMStudio API format (not OpenAI-compatible).
    
    Can be used for:
    - Generating training examples
    - Providing feedback for RLHF
    - Evaluating student outputs
    """
    
    def __init__(self, config: Optional[LMStudioConfig] = None):
        self.config = config or LMStudioConfig()
        self._headers = {
            "Content-Type": "application/json",
        }
        if self.config.api_key:
            self._headers["Authorization"] = f"Bearer {self.config.api_key}"
    
    def is_available(self) -> bool:
        """Check if LMStudio server is running."""
        try:
            # Try the LMStudio native models endpoint
            response = requests.get(
                f"{self.config.base_url}/api/v1/models",
                headers=self._headers,
                timeout=5
            )
            if response.status_code == 200:
                return True
            # Fall back to OpenAI-compatible endpoint
            response = requests.get(
                f"{self.config.base_url}/v1/models",
                headers=self._headers,
                timeout=5
            )
            return response.status_code == 200
        except Exception:
            return False
    
    def get_loaded_models(self) -> List[str]:
        """Get list of currently loaded models."""
        try:
            response = requests.get(
                f"{self.config.base_url}/api/v1/models",
                headers=self._headers,
                timeout=5
            )
            if response.status_code == 200:
                data = response.json()
                return [m.get("id", "") for m in data.get("data", [])]
        except Exception:
            pass
        return []
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        retry_count: int = 2,
    ) -> str:
        """Generate text from the teacher model using LMStudio native API."""
        # LMStudio native API format: POST /api/v1/chat
        payload = {
            "model": self.config.model,
            "input": prompt,
            "temperature": temperature if temperature is not None else self.config.temperature,
            "max_output_tokens": max_tokens or self.config.max_tokens,
            "context_length": self.config.context_length,
            "store": False,  # Don't store for training use
        }
        
        if system_prompt:
            payload["system_prompt"] = system_prompt
        
        last_error = None
        for attempt in range(retry_count + 1):
            try:
                response = requests.post(
                    f"{self.config.base_url}/api/v1/chat",
                    headers=self._headers,
                    json=payload,
                    timeout=self.config.timeout
                )
                
                if response.status_code == 200:
                    data = response.json()
                    # Extract content from LMStudio response format
                    # Response: {"output": [{"type": "message", "content": "..."}]}
                    outputs = data.get("output", [])
                    for item in outputs:
                        if item.get("type") == "message":
                            return item.get("content", "")
                    # Fallback: try to find any content
                    if outputs and isinstance(outputs[0], dict):
                        return outputs[0].get("content", str(outputs[0]))
                    return str(outputs)
                
                # Parse error response
                error_msg = response.text
                try:
                    error_data = response.json()
                    error_msg = error_data.get("error", response.text)
                except Exception:
                    pass
                
                # Model crash or loading error - may be recoverable with retry
                if "crashed" in str(error_msg).lower() or response.status_code in (400, 500, 503):
                    last_error = f"Model error (attempt {attempt + 1}): {error_msg}"
                    print(f"[LMStudio] {last_error}")
                    if attempt < retry_count:
                        import time
                        time.sleep(2)  # Wait before retry
                        continue
                
                raise RuntimeError(f"LMStudio API error: {response.status_code} - {error_msg}")
                
            except requests.exceptions.Timeout:
                last_error = f"Request timeout (attempt {attempt + 1})"
                print(f"[LMStudio] {last_error}")
                if attempt < retry_count:
                    continue
                raise RuntimeError(f"LMStudio request timed out after {self.config.timeout}s")
            
            except requests.exceptions.ConnectionError as e:
                raise RuntimeError(f"Cannot connect to LMStudio at {self.config.base_url}: {e}")
        
        raise RuntimeError(f"LMStudio failed after {retry_count + 1} attempts: {last_error}")
    
    def generate_training_shards(
        self,
        symbol_name: str,
        symbol_description: str,
        num_examples: int = 3,
    ) -> List[Dict[str, str]]:
        """Generate training shards for a symbol concept."""
        system_prompt = """You are a helpful teacher generating training data for an AI language model.
Generate diverse question-answer pairs about the given concept.
Return a JSON array of objects with 'kind', 'input_text', and 'target_text' fields.
Kinds should be: 'label', 'definition', 'example', 'analogy', or 'comparison'."""

        prompt = f"""Generate {num_examples} training examples for the concept "{symbol_name}".
Description: {symbol_description}

Return only valid JSON array, no other text."""

        try:
            response = self.generate(prompt, system_prompt=system_prompt, temperature=0.8)
            # Try to parse JSON from response
            response = response.strip()
            if response.startswith("```json"):
                response = response[7:]
            if response.startswith("```"):
                response = response[3:]
            if response.endswith("```"):
                response = response[:-3]
            
            shards = json.loads(response)
            return shards
        except Exception as e:
            # Fallback to simple format
            return [
                {"kind": "definition", "input_text": f"What is {symbol_name}?", "target_text": symbol_description},
                {"kind": "example", "input_text": f"Give an example of {symbol_name}.", "target_text": f"An example of {symbol_name} in action..."},
                {"kind": "label", "input_text": f"Name this concept: {symbol_description}", "target_text": symbol_name},
            ]
    
    def score_response(
        self,
        prompt: str,
        response: str,
        criteria: str = "helpfulness, accuracy, and clarity"
    ) -> Tuple[float, str]:
        """
        Score a response for RLHF/DPO.
        
        Returns:
            (score 0-10, explanation)
        """
        system_prompt = f"""You are evaluating AI responses based on {criteria}.
Rate the response from 0 to 10, where 10 is perfect.
Respond with JSON: {{"score": <number>, "explanation": "<text>"}}"""

        scoring_prompt = f"""Question/Prompt: {prompt}

AI Response: {response}

Rate this response:"""

        try:
            result = self.generate(scoring_prompt, system_prompt=system_prompt, temperature=0.3)
            # Parse JSON
            result = result.strip()
            if result.startswith("```json"):
                result = result[7:]
            if result.endswith("```"):
                result = result[:-3]
            
            data = json.loads(result)
            return float(data.get("score", 5)), data.get("explanation", "")
        except Exception:
            return 5.0, "Could not parse score"
    
    def generate_preference_pair(
        self,
        prompt: str,
        bad_response: str,
    ) -> str:
        """
        Generate a better response for DPO training.
        
        Args:
            prompt: The original prompt
            bad_response: The student's (potentially poor) response
        
        Returns:
            A better response from the teacher
        """
        system_prompt = """You are a helpful AI assistant. 
Provide a high-quality, accurate, and helpful response to the user's question."""

        return self.generate(prompt, system_prompt=system_prompt, temperature=0.5)


# =============================================================================
# RLHF / DPO Data Generator
# =============================================================================

@dataclass
class PreferencePair:
    """A preference pair for DPO/RLHF training."""
    prompt: str
    chosen: str  # Preferred response
    rejected: str  # Less preferred response
    chosen_score: float = 0.0
    rejected_score: float = 0.0


class PreferenceDataGenerator:
    """
    Generate preference data for DPO/RLHF training.
    
    Uses the teacher model to:
    1. Generate prompts
    2. Score student responses
    3. Generate better alternatives
    """
    
    def __init__(
        self,
        teacher: LMStudioTeacher,
        student_generator: Callable[[str], str],
    ):
        self.teacher = teacher
        self.student_generator = student_generator
    
    def generate_pair(self, prompt: str) -> PreferencePair:
        """Generate a preference pair for a given prompt."""
        # Get student response
        student_response = self.student_generator(prompt)
        
        # Score it
        student_score, _ = self.teacher.score_response(prompt, student_response)
        
        # Get teacher response (assumed better)
        teacher_response = self.teacher.generate_preference_pair(prompt, student_response)
        teacher_score, _ = self.teacher.score_response(prompt, teacher_response)
        
        # Determine chosen/rejected
        if teacher_score >= student_score:
            return PreferencePair(
                prompt=prompt,
                chosen=teacher_response,
                rejected=student_response,
                chosen_score=teacher_score,
                rejected_score=student_score,
            )
        else:
            return PreferencePair(
                prompt=prompt,
                chosen=student_response,
                rejected=teacher_response,
                chosen_score=student_score,
                rejected_score=teacher_score,
            )
    
    def generate_batch(
        self,
        prompts: List[str],
        min_score_diff: float = 1.0,
    ) -> List[PreferencePair]:
        """
        Generate a batch of preference pairs.
        
        Only keeps pairs where the score difference exceeds min_score_diff.
        """
        pairs = []
        for prompt in prompts:
            pair = self.generate_pair(prompt)
            if abs(pair.chosen_score - pair.rejected_score) >= min_score_diff:
                pairs.append(pair)
        return pairs


# =============================================================================
# Teacher Factory for UI
# =============================================================================

def create_teacher_generator(
    use_lmstudio: bool = False,
    lmstudio_config: Optional[LMStudioConfig] = None,
    lmstudio_url: Optional[str] = None,
    lmstudio_model: Optional[str] = None,
) -> Callable[[str], str]:
    """
    Create a teacher generator function for the training loop.
    
    Args:
        use_lmstudio: Whether to use LMStudio API
        lmstudio_config: LMStudio configuration (overrides url/model)
        lmstudio_url: LMStudio API base URL (e.g., http://localhost:1234)
        lmstudio_model: Model name to use
    
    Returns:
        Generator function that takes a prompt and returns training shards JSON
    """
    if use_lmstudio:
        # Build config from either explicit config or URL/model params
        if lmstudio_config is None:
            # Normalize URL: remove /v1 suffix if present (new LMStudio API uses /api/v1)
            base_url = lmstudio_url or "http://localhost:1234"
            if base_url.endswith("/v1"):
                base_url = base_url[:-3]
            if base_url.endswith("/"):
                base_url = base_url[:-1]
            
            lmstudio_config = LMStudioConfig(
                base_url=base_url,
                model=lmstudio_model or "llama-3.1-8b-lexi-uncensored-v2",
            )
        
        teacher = LMStudioTeacher(lmstudio_config)
        
        if not teacher.is_available():
            print("[LMStudio] Warning: Server not available, falling back to static generator")
            return _static_teacher_generator
        
        # Check what models are loaded
        loaded = teacher.get_loaded_models()
        if loaded:
            print(f"[LMStudio] Available models: {loaded}")
        
        # Test actual generation to ensure model works
        try:
            test_response = teacher.generate("Say OK", max_tokens=5, temperature=0.1, retry_count=1)
            print(f"[LMStudio] Connected to {lmstudio_config.base_url}, model: {lmstudio_config.model}")
            print(f"[LMStudio] Test response: {test_response[:50]}")
        except Exception as e:
            print(f"[LMStudio] Model test failed: {e}")
            print("[LMStudio] Falling back to static generator (model crashed or unavailable)")
            return _static_teacher_generator
        
        def lmstudio_gen(prompt: str) -> str:
            """
            Generate training shards JSON for the sentient LLM loop.
            
            Expected prompt format from LocalTeacher._prompt_for_symbol:
            "Generate training shards for a student model learning a new concept.
             Symbol token: ⟦{symbol_id}⟧
             stability=X.XXXX novelty=X.XXXX
             prime_basis=[...]
             Return JSON array. Each item: {kind,input_text,target_text}.
             kind ∈ [label,definition,example,qa]."
            """
            try:
                print(f"[LMStudio Gen] Received prompt: {prompt[:100]}...", flush=True)
                
                # Extract symbol info from the prompt
                symbol_name = "concept"
                
                # Look for "Symbol token: ⟦XXX⟧"
                if "Symbol token:" in prompt:
                    start = prompt.find("⟦") + 1
                    end = prompt.find("⟧")
                    if start > 0 and end > start:
                        symbol_name = prompt[start:end]
                
                # Use teacher.generate() directly with the prompt
                # Let the LLM follow the instructions in the prompt
                system_prompt = """You are generating training data for an AI language model.
Generate diverse question-answer pairs as requested.
ALWAYS return ONLY a valid JSON array with objects containing: kind, input_text, target_text.
Kinds: label, definition, example, qa
Example format:
[{"kind": "definition", "input_text": "What is X?", "target_text": "X is..."}]"""
                
                response = teacher.generate(
                    prompt,
                    system_prompt=system_prompt,
                    temperature=0.8,
                    max_tokens=1024,
                )
                
                print(f"[LMStudio Gen] Got response: {len(response)} chars", flush=True)
                print(f"[LMStudio Gen] Response preview: {response[:200]}...", flush=True)
                
                # Try to ensure it's valid JSON
                response = response.strip()
                
                # Try to parse to validate
                try:
                    json.loads(response)
                    return response
                except json.JSONDecodeError:
                    # Try to extract JSON array from response
                    json_str = None
                    
                    if "```json" in response:
                        start = response.find("```json") + 7
                        end = response.find("```", start)
                        if end > start:
                            json_str = response[start:end].strip()
                        else:
                            # No closing ```, try to find the array anyway
                            json_str = response[start:].strip()
                    elif "```" in response:
                        # Generic code block
                        start = response.find("```") + 3
                        # Skip optional language identifier
                        if response[start:start+1].isalpha():
                            start = response.find("\n", start) + 1
                        end = response.find("```", start)
                        if end > start:
                            json_str = response[start:end].strip()
                    
                    if json_str is None and "[" in response:
                        start = response.find("[")
                        end = response.rfind("]") + 1
                        if end > start:
                            json_str = response[start:end]
                    
                    if json_str:
                        try:
                            json.loads(json_str)
                            return json_str
                        except json.JSONDecodeError:
                            # JSON may be truncated - try to repair by closing arrays/objects
                            repaired = _repair_truncated_json(json_str)
                            if repaired:
                                return repaired
                
                # If we get here, couldn't parse - return as-is, let teacher_local.py handle it
                return response
                
            except Exception as e:
                import traceback
                print(f"[LMStudio Gen] Error: {e}", flush=True)
                print(traceback.format_exc(), flush=True)
                return _static_teacher_generator(prompt)
        
        return lmstudio_gen
    
    return _static_teacher_generator


def _repair_truncated_json(json_str: str) -> Optional[str]:
    """
    Attempt to repair truncated JSON arrays.
    
    LLMs sometimes output truncated JSON when hitting token limits.
    This tries to close open arrays/objects to salvage partial data.
    """
    if not json_str or not json_str.strip():
        return None
    
    json_str = json_str.strip()
    
    # Count open brackets/braces
    open_brackets = json_str.count('[') - json_str.count(']')
    open_braces = json_str.count('{') - json_str.count('}')
    
    # If nothing is unclosed, can't repair
    if open_brackets <= 0 and open_braces <= 0:
        return None
    
    # Try to find a good truncation point (after a complete object in an array)
    # Look for }, or }] patterns
    repaired = json_str
    
    # If we're in the middle of a string, find the last complete item
    # Look backwards for a closing brace that could be a complete object
    last_complete = -1
    for i in range(len(json_str) - 1, -1, -1):
        if json_str[i] == '}':
            # Check if this could be a complete object
            test = json_str[:i+1]
            # Add closing brackets
            test += ']' * (test.count('[') - test.count(']'))
            try:
                parsed = json.loads(test)
                if isinstance(parsed, list) and len(parsed) > 0:
                    return test
            except json.JSONDecodeError:
                continue
    
    # Simple approach: just close everything
    repaired = json_str.rstrip()
    # Remove trailing comma if present
    if repaired.endswith(','):
        repaired = repaired[:-1]
    # Remove incomplete string
    if repaired.count('"') % 2 == 1:
        # Find last complete string
        last_quote = repaired.rfind('"')
        repaired = repaired[:last_quote+1]
    
    # Close braces then brackets
    repaired += '}' * open_braces
    repaired += ']' * open_brackets
    
    try:
        parsed = json.loads(repaired)
        if isinstance(parsed, list) and len(parsed) > 0:
            print(f"[JSON Repair] Recovered {len(parsed)} items from truncated JSON", flush=True)
            return repaired
    except json.JSONDecodeError:
        pass
    
    return None


def _static_teacher_generator(prompt: str) -> str:
    """Static fallback teacher generator."""
    return json.dumps([
        {
            "kind": "label",
            "input_text": "Name the concept symbol.",
            "target_text": "A canonical symbol in the lexicon.",
        },
        {
            "kind": "definition",
            "input_text": "Define the symbol and its role.",
            "target_text": "A stable semantic concept used for continual learning.",
        },
        {
            "kind": "example",
            "input_text": "Use the symbol in context.",
            "target_text": "In practice, this symbol anchors meaning across episodes.",
        },
    ])
