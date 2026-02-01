"""
Inference Engine for ResoLLM.

Supports both:
- Standard mode: Simple generation matching jupyter2.py
- Extended mode: Full monitoring with stability, agency, and multi-agent coordination

Standard Mode Features:
- Text generation with temperature, top-k, top-p sampling
- Repetition penalty
- Basic entropy tracking
- Chat formatting using jupyter2.py template format

Extended Mode Features (when extensions enabled):
- Predictive Lyapunov stability analysis
- Dynamic temperature adjustment
- Agency-guided generation with goal tracking
- Multi-agent coordination through entanglement
- Stochastic resonance for escaping repetition
"""
import sys
import os
import math
import time
from typing import List, Optional, Callable, Dict, Any, Tuple, Union

# =============================================================================
# Chat Template Format (matches training templates exactly)
# =============================================================================

# Import training template tokens to ensure consistency
try:
    from apps.reso_llm.training_templates import SPECIAL_TOKENS
    _SYSTEM_START = SPECIAL_TOKENS['system_start']
    _SYSTEM_END = SPECIAL_TOKENS['system_end']
    _USER_START = SPECIAL_TOKENS['user_start']
    _USER_END = SPECIAL_TOKENS['user_end']
    _ASSISTANT_START = SPECIAL_TOKENS['assistant_start']
    _ASSISTANT_END = SPECIAL_TOKENS['assistant_end']
    _EOS = SPECIAL_TOKENS['eos']
except ImportError:
    # Fallback if training_templates not available
    _SYSTEM_START = "<|system|>"
    _SYSTEM_END = "<|endofsystem|>"
    _USER_START = "<|user|>"
    _USER_END = "<|endofuser|>"
    _ASSISTANT_START = "<|assistant|>"
    _ASSISTANT_END = "<|endofassistant|>"
    _EOS = "<|endoftext|>"

# Chat prompt template for proper instruction-following format
CHAT_TEMPLATE = f"""{_SYSTEM_START}
You are a helpful, harmless, and honest AI assistant.
{_SYSTEM_END}
{{conversation}}{_EOS}"""

USER_TEMPLATE = f"{_USER_START}\n{{message}}\n{_USER_END}"
ASSISTANT_TEMPLATE = f"{_ASSISTANT_START}\n{{message}}\n{_ASSISTANT_END}"

# Stop tokens for chat generation
CHAT_STOP_TOKENS = [_ASSISTANT_END, _USER_START, _EOS]


def format_conversation(messages: List[Dict[str, str]]) -> str:
    """
    Format a list of messages into proper chat format.
    
    Args:
        messages: List of dicts with 'role' and 'content' keys
    
    Returns:
        Formatted conversation string
    """
    formatted_parts = []
    for msg in messages:
        role = msg.get('role', 'user').lower()
        content = msg.get('content', '')
        
        if role in ('user', 'human'):
            formatted_parts.append(USER_TEMPLATE.format(message=content))
        elif role in ('assistant', 'bot', 'gpt'):
            formatted_parts.append(ASSISTANT_TEMPLATE.format(message=content))
    
    return '\n'.join(formatted_parts)


def build_chat_prompt(messages: List[Dict[str, str]], add_assistant_prompt: bool = True) -> str:
    """
    Build a full chat prompt from message history.
    
    Args:
        messages: List of message dicts with 'role' and 'content'
        add_assistant_prompt: Whether to add the assistant prompt prefix
        
    Returns:
        Formatted prompt string ready for generation
    """
    conversation = format_conversation(messages)
    prompt = CHAT_TEMPLATE.format(conversation=conversation)
    
    # Remove end token for continuation
    prompt = prompt.replace('<|endoftext|>', '')
    
    if add_assistant_prompt:
        prompt += "\n<|assistant|>\n"
    
    return prompt


def extract_assistant_response(text: str) -> str:
    """
    Extract the assistant's response from generated text.
    
    Args:
        text: Raw generated text
        
    Returns:
        Cleaned response text
    """
    response = text
    
    # Stop at end markers
    for marker in CHAT_STOP_TOKENS:
        if marker in response:
            response = response.split(marker)[0]
    
    return response.strip()

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

import torch
import torch.nn.functional as F

from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import ResoLLMTokenizer, ResoBPETokenizer
from apps.reso_llm.config import ResoLLMConfig, GenerationConfig

# Type alias for any tokenizer type
TokenizerType = Union[ResoLLMTokenizer, ResoBPETokenizer]


class GenerationResult:
    """
    Result from text generation.
    
    In standard mode, contains text and basic metrics.
    In extended mode, contains full monitoring data.
    """
    
    def __init__(
        self,
        text: str,
        stability: str = "unknown",
        lyapunov: float = 0.0,
        entropy_trace: List[float] = None,
        lyapunov_trace: List[float] = None,
        coherence_trace: List[float] = None,
        temperature_trace: List[float] = None,
        goals_achieved: List[str] = None,
        attention_summary: Dict[str, float] = None,
        warnings: List[str] = None,
        generation_time_ms: float = 0.0,
        tokens_generated: int = 0,
    ):
        self.text = text
        self.stability = stability
        self.lyapunov = lyapunov
        self.entropy_trace = entropy_trace or []
        self.lyapunov_trace = lyapunov_trace or []
        self.coherence_trace = coherence_trace or []
        self.temperature_trace = temperature_trace or []
        self.goals_achieved = goals_achieved or []
        self.attention_summary = attention_summary or {}
        self.warnings = warnings or []
        self.generation_time_ms = generation_time_ms
        self.tokens_generated = tokens_generated
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "text": self.text,
            "stability": self.stability,
            "lyapunov": self.lyapunov,
            "entropy_trace": self.entropy_trace,
            "lyapunov_trace": self.lyapunov_trace,
            "coherence_trace": self.coherence_trace,
            "temperature_trace": self.temperature_trace,
            "goals_achieved": self.goals_achieved,
            "attention_summary": self.attention_summary,
            "warnings": self.warnings,
            "generation_time_ms": self.generation_time_ms,
            "tokens_generated": self.tokens_generated,
        }
        
    def is_stable(self) -> bool:
        """Check if generation was stable."""
        return self.stability in ["stable", "periodic", "unknown"]
        
    def has_warnings(self) -> bool:
        """Check if any warnings were generated."""
        return len(self.warnings) > 0
        
    def __repr__(self) -> str:
        return (
            f"GenerationResult(tokens={self.tokens_generated}, "
            f"stability={self.stability}, time={self.generation_time_ms:.1f}ms)"
        )


class ResoLLMInference:
    """
    Inference Engine for ResoLLM.
    
    Standard Mode (default):
        Simple text generation with sampling strategies.
        Compatible with models trained using jupyter2.py.
    
    Extended Mode (when model has extensions enabled):
        Full stability monitoring, agency guidance, and multi-agent coordination.
    """
    
    def __init__(
        self,
        model: ResoLLMModel,
        tokenizer: TokenizerType,
        temperature: float = 0.8,
        max_length: int = 100,
        top_k: int = 50,
        top_p: float = 0.95,
        repetition_penalty: float = 1.1,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = model.config
        
        # Generation parameters
        self.temperature = temperature
        self.max_length = max_length
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        
        # State tracking
        self._current_temperature = temperature
        self._generation_step = 0
        self._warnings: List[str] = []
        self._connected_agents: List[str] = []
        
        # Extended mode components (lazy-loaded)
        self._entropy_tracker = None
        self._stability_class = None
        
        # Check if extended mode
        self._extended_mode = not self.config.standard_mode and self.config.has_extensions_enabled()
        
        if self._extended_mode:
            self._init_extended_tracking()
    
    def _init_extended_tracking(self):
        """Initialize extended mode tracking components."""
        try:
            from tinyaleph.physics.entropy import EntropyTracker, StabilityClass
            self._entropy_tracker = EntropyTracker(
                window_size=self.config.stability.entropy_window
            )
            self._stability_class = StabilityClass
        except ImportError:
            self._extended_mode = False
            print("Warning: Extended tracking unavailable - running in standard mode")
    
    def apply_repetition_penalty(
        self,
        logits: torch.Tensor,
        generated_tokens: List[int],
        window: int = 50
    ) -> torch.Tensor:
        """Apply repetition penalty to recent tokens."""
        if self.repetition_penalty == 1.0 or not generated_tokens:
            return logits
        
        recent_tokens = generated_tokens[-window:]
        
        for token_id in set(recent_tokens):
            if token_id < logits.size(-1):
                if logits[token_id] < 0:
                    logits[token_id] *= self.repetition_penalty
                else:
                    logits[token_id] /= self.repetition_penalty
        return logits
    
    def sample(self, logits: torch.Tensor, temperature: float = None) -> int:
        """Sample next token with top-k/top-p filtering."""
        if temperature is None:
            temperature = self._current_temperature
            
        if logits.dim() > 1:
            logits = logits.squeeze()
        
        # Get last position if needed
        if logits.dim() > 1:
            logits = logits[-1]
        
        # Temperature
        if temperature > 0:
            logits = logits / temperature
        
        # Top-K
        if self.top_k > 0:
            top_k = min(self.top_k, logits.size(-1))
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[-1]] = float('-inf')
        
        # Top-P (Nucleus)
        if self.top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > self.top_p
            sorted_indices_to_remove[1:] = sorted_indices_to_remove[:-1].clone()
            sorted_indices_to_remove[0] = 0
            
            indices_to_remove = torch.zeros_like(logits, dtype=torch.bool).scatter_(
                0, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')
        
        # Sample
        probs = F.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        return next_token.item()
    
    def compute_entropy(self, logits: torch.Tensor) -> float:
        """Compute entropy of the distribution."""
        probs = F.softmax(logits, dim=-1)
        log_probs = F.log_softmax(logits, dim=-1)
        entropy = -(probs * log_probs).sum().item()
        return entropy
    
    def generate(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = None,
        stop_on_instability: bool = True,
        goal: Optional[str] = None,
        auto_temperature: bool = True,
        callback: Optional[Callable[[str, int, float], None]] = None,
    ) -> GenerationResult:
        """
        Generate text from prompt.
        
        Args:
            prompt: Input text
            max_length: Maximum tokens to generate
            temperature: Initial sampling temperature
            stop_on_instability: Whether to halt if system becomes divergent (extended mode)
            goal: Optional goal description for agency-guided generation (extended mode)
            auto_temperature: Whether to automatically adjust temperature (extended mode)
            callback: Optional callback(text, step, coherence) for each token
            
        Returns:
            GenerationResult with text and metrics
        """
        start_time = time.time()
        max_length = max_length or self.max_length
        self._current_temperature = temperature or self.temperature
        self._generation_step = 0
        self._warnings = []
        
        # Device
        device = next(self.model.parameters()).device
        
        # Encode prompt
        input_ids = self.tokenizer.encode(prompt)
        generated_tokens = []
        
        # Tracking
        entropy_trace = []
        lyapunov_trace = []
        coherence_trace = []
        temperature_trace = []
        
        # Create goal if specified (extended mode)
        active_goal = None
        if goal and hasattr(self.model, 'create_goal') and self.model.agency:
            active_goal = self.model.create_goal(goal, goal_type="exploratory")
        
        self.model.eval()
        with torch.no_grad():
            for step in range(max_length):
                self._generation_step = step
                
                # Prepare context
                context = input_ids + generated_tokens
                context_tensor = torch.tensor([context], device=device)
                
                # Truncate if needed
                if context_tensor.size(1) > self.config.max_seq_len:
                    context_tensor = context_tensor[:, -self.config.max_seq_len:]
                
                # Forward pass
                outputs = self.model(context_tensor)
                
                # Get logits for last position
                if outputs.dim() == 3:
                    next_token_logits = outputs[0, -1, :]
                else:
                    next_token_logits = outputs[-self.config.vocab_size:]
                
                # Apply repetition penalty
                next_token_logits = self.apply_repetition_penalty(
                    next_token_logits, generated_tokens
                )
                
                # Compute entropy and coherence
                entropy = self.compute_entropy(next_token_logits)
                max_entropy = math.log(self.config.vocab_size)
                coherence = 1.0 - (entropy / max_entropy)
                
                entropy_trace.append(entropy)
                coherence_trace.append(coherence)
                temperature_trace.append(self._current_temperature)
                
                # Extended mode processing
                should_stop = False
                if self._extended_mode and self._entropy_tracker:
                    disorder = 1.0 - coherence
                    self._entropy_tracker.record(disorder)
                    
                    lyap = self._entropy_tracker.lyapunov_exponent()
                    lyapunov_trace.append(lyap)
                    
                    stability = self._entropy_tracker.stability()
                    
                    # Update repetition tracking
                    if generated_tokens:
                        self.model.update_repetition_score(generated_tokens[-1])
                    
                    # Predictive stability check
                    if self.model.stability_monitor:
                        will_destabilize, confidence, steps_until = (
                            self.model.stability_monitor.predict_instability()
                        )
                        if will_destabilize and confidence > 0.7:
                            self._warnings.append(
                                f"Step {step}: Predicted instability in {steps_until} steps"
                            )
                    
                    # Auto temperature adjustment
                    if auto_temperature and self.config.stability.auto_temperature_adjust:
                        new_temp = self.model.get_suggested_temperature(self._current_temperature)
                        if abs(new_temp - self._current_temperature) > 0.05:
                            self._warnings.append(
                                f"Step {step}: Temp {self._current_temperature:.2f}→{new_temp:.2f}"
                            )
                            self._current_temperature = new_temp
                    
                    # Stability-based stopping
                    if stop_on_instability and stability == self._stability_class.DIVERGENT:
                        self._warnings.append(f"Step {step}: Halted due to instability")
                        should_stop = True
                    
                    # Goal progress
                    if active_goal and self.model.agency:
                        progress = min(1.0, step / max_length)
                        active_goal.update_progress(progress * 0.8)
                
                # Sample next token
                next_token = self.sample(next_token_logits)
                generated_tokens.append(next_token)
                
                # Callback
                if callback:
                    current_text = self.tokenizer.decode(generated_tokens)
                    callback(current_text, step, coherence)

                # Early stop on chat stop markers
                current_text = self.tokenizer.decode(generated_tokens)
                for marker in CHAT_STOP_TOKENS:
                    if marker in current_text:
                        # Trim to before the marker and stop
                        current_text = current_text.split(marker)[0]
                        generated_tokens = self.tokenizer.encode(current_text)
                        should_stop = True
                        break

                # Stop conditions
                if should_stop:
                    break
                
                # Check for EOS
                eos_id = getattr(self.tokenizer, 'eos_id', None)
                if eos_id is None:
                    special = getattr(self.tokenizer, 'special_tokens', {})
                    eos_id = special.get('<EOS>') or special.get('<eos>')
                if eos_id is not None and next_token == eos_id:
                    break
        
        # Decode output
        generated_text = self.tokenizer.decode(generated_tokens)
        
        # Final analysis (extended mode)
        final_stability = "unknown"
        final_lyap = 0.0
        if self._extended_mode and self._entropy_tracker:
            final_lyap = self._entropy_tracker.lyapunov_exponent()
            final_stability = self._entropy_tracker.stability().value
            
            # Complete goal
            if active_goal:
                if final_stability in ["stable", "collapsed"]:
                    active_goal.achieve()
                else:
                    active_goal.abandon("Unstable generation")
        
        # Update memory (if available)
        if hasattr(self.model, 'update_memory'):
            self.model.update_memory(prompt + generated_text, importance=0.8)
        
        # Compile attention summary (extended mode)
        attention_summary = {}
        if hasattr(self.model, 'get_attention_foci') and self.model.agency:
            for focus in self.model.get_attention_foci():
                key = f"{focus.type}:{focus.target}"
                attention_summary[key] = focus.intensity
        
        # Compile goals achieved (extended mode)
        goals_achieved = []
        if hasattr(self.model, 'agency') and self.model.agency:
            for g in self.model.agency.goals:
                if g.status == "achieved":
                    goals_achieved.append(g.description)
        
        generation_time = (time.time() - start_time) * 1000
        
        return GenerationResult(
            text=generated_text,
            stability=final_stability,
            lyapunov=final_lyap,
            entropy_trace=entropy_trace,
            lyapunov_trace=lyapunov_trace,
            coherence_trace=coherence_trace,
            temperature_trace=temperature_trace,
            goals_achieved=goals_achieved,
            attention_summary=attention_summary,
            warnings=self._warnings,
            generation_time_ms=generation_time,
            tokens_generated=len(generated_tokens),
        )

    def generate_simple(
        self,
        prompt: str,
        max_length: int = None,
        temperature: float = None,
    ) -> str:
        """
        Simple generation returning just text.
        
        Convenience method for standard mode usage.
        """
        result = self.generate(
            prompt,
            max_length=max_length,
            temperature=temperature,
            auto_temperature=False,
        )
        return result.text

    def chat(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 200,
        goal: Optional[str] = None,
    ) -> str:
        """
        Generate a response in a chat context using jupyter2.py format.
        
        Uses proper chat tokens: <|system|>, <|user|>, <|assistant|>, etc.
        This matches the training format used in jupyter2.py/Colab.
        
        Args:
            user_input: The user's message
            history: List of message dicts with 'role' and 'content' keys
            max_length: Maximum response length
            goal: Optional goal for response generation (extended mode)
            
        Returns:
            The model's response text (cleaned)
        """
        history = history or []
        
        # Build messages list
        messages = list(history)  # Copy to avoid modifying original
        messages.append({'role': 'user', 'content': user_input})
        
        # Build prompt using jupyter2.py format
        prompt = build_chat_prompt(messages, add_assistant_prompt=True)
        
        # Update memory if available
        if hasattr(self.model, 'update_memory'):
            self.model.update_memory(user_input, importance=0.9)
        
        result = self.generate(prompt, max_length=max_length, goal=goal)
        
        # Extract and clean the response
        response = extract_assistant_response(result.text)
        return response
        
    def chat_with_result(
        self,
        user_input: str,
        history: Optional[List[Dict[str, str]]] = None,
        max_length: int = 200,
        goal: Optional[str] = None,
    ) -> Tuple[str, GenerationResult]:
        """
        Generate a chat response with full result data.
        
        Uses proper chat tokens matching jupyter2.py format.
        """
        history = history or []
        
        # Build messages list
        messages = list(history)
        messages.append({'role': 'user', 'content': user_input})
        
        # Build prompt using jupyter2.py format
        prompt = build_chat_prompt(messages, add_assistant_prompt=True)
        
        if hasattr(self.model, 'update_memory'):
            self.model.update_memory(user_input, importance=0.9)
        
        result = self.generate(prompt, max_length=max_length, goal=goal)
        
        # Extract and clean the response
        response = extract_assistant_response(result.text)
        return response, result

    def interactive_session(self, show_metrics: bool = False):
        """
        Run an interactive CLI session.
        
        Uses jupyter2.py chat format with proper chat tokens.
        
        Args:
            show_metrics: Whether to display generation metrics
        """
        mode_str = "Extended" if self._extended_mode else "Standard"
        print(f"Reso-LLM Interactive Session ({mode_str} Mode)")
        print("Type 'exit' or 'quit' to end session")
        if self._extended_mode:
            print("Commands: 'goal <description>', 'status', 'metrics'")
        print("-" * 50)
        
        # History as list of dicts with 'role' and 'content' keys
        # This matches jupyter2.py chat format
        history: List[Dict[str, str]] = []
        show_metrics_mode = show_metrics
        current_goal = None
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if not user_input:
                    continue
                    
                if user_input.lower() in ["exit", "quit"]:
                    print("Goodbye!")
                    break
                    
                if user_input.lower() == "status" and self._extended_mode:
                    self._print_status()
                    continue
                    
                if user_input.lower() == "metrics":
                    show_metrics_mode = not show_metrics_mode
                    print(f"Metrics display: {'ON' if show_metrics_mode else 'OFF'}")
                    continue
                    
                if user_input.lower().startswith("goal ") and self._extended_mode:
                    current_goal = user_input[5:].strip()
                    print(f"Goal set: {current_goal}")
                    continue
                
                # Generate response using proper chat format
                response, result = self.chat_with_result(
                    user_input,
                    history=history,
                    goal=current_goal
                )
                
                # Update history with proper role/content format
                history.append({'role': 'user', 'content': user_input})
                history.append({'role': 'assistant', 'content': response})
                
                # Keep history manageable (10 exchanges = 20 messages)
                if len(history) > 20:
                    history = history[-20:]
                
                print(f"\nAssistant: {response}")
                
                if show_metrics_mode:
                    print(f"\n  [Tokens: {result.tokens_generated}, "
                          f"Time: {result.generation_time_ms:.1f}ms, "
                          f"Stability: {result.stability}]")
                    if result.warnings:
                        for w in result.warnings[-3:]:
                            print(f"  [Warning: {w}]")
                            
            except KeyboardInterrupt:
                print("\n\nInterrupted. Goodbye!")
                break
            except Exception as e:
                print(f"\nError: {e}")
    
    def _print_status(self):
        """Print current model status (extended mode)."""
        print("\n--- Model Status ---")
        
        if self._entropy_tracker:
            stability = self._entropy_tracker.stability()
            lyap = self._entropy_tracker.lyapunov_exponent()
            print(f"Stability: {stability.value}")
            print(f"Lyapunov: {lyap:.4f}")
        
        if hasattr(self.model, 'get_stability'):
            print(f"Model Stability: {self.model.get_stability():.4f}")
        
        if hasattr(self.model, 'get_active_goals') and self.model.agency:
            goals = self.model.get_active_goals()
            print(f"Active Goals: {len(goals)}")
            for g in goals[:3]:
                print(f"  - {g.description} ({g.progress:.1%})")
        
        if hasattr(self.model, 'get_attention_foci') and self.model.agency:
            foci = self.model.get_attention_foci()
            print(f"Attention Foci: {len(foci)}")
        
        print("-" * 20)
