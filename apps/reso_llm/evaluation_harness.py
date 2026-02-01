import torch
import numpy as np
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import statistics

from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.inference import ResoLLMInference
from apps.reso_llm.config import ResoLLMConfig

@dataclass
class EvaluationMetric:
    name: str
    value: float
    unit: str
    description: str
    metadata: Dict[str, Any] = None

class ResoEvaluationHarness:
    """
    Evaluation harness for Reso-LLM extensions.
    
    Provides specialized metrics for:
    - Agency (Goal completion, attention)
    - PRSC (Semantic coherence)
    - Stability (Lyapunov analysis)
    - Memory (Recall, decay)
    """
    
    def __init__(self, model: ResoLLMModel, tokenizer):
        self.model = model
        self.tokenizer = tokenizer
        self.inference = ResoLLMInference(model, tokenizer)
        
    def evaluate_agency(self, num_trials: int = 3) -> List[EvaluationMetric]:
        """Evaluate Agency Layer performance."""
        if not getattr(self.model, 'agency', None):
            return []
            
        metrics = []
        goals_achieved = 0
        avg_focus_intensity = []
        
        print(f"Evaluating Agency Layer ({num_trials} trials)...")
        
        for i in range(num_trials):
            # Test Goal Achievement
            goal_desc = f"Test goal {i}"
            
            # Manually inject a goal if method exists
            if hasattr(self.model, 'create_goal'):
                goal = self.model.create_goal(goal_desc, goal_type="exploratory")
                
                # Run a short generation
                self.inference.generate(f"Working on {goal_desc}", max_length=20)
                
                if goal.status == "achieved" or goal.progress > 0.8:
                    goals_achieved += 1
            
            # Check attention
            if hasattr(self.model, 'get_attention_foci'):
                foci = self.model.get_attention_foci()
                if foci:
                    avg_focus_intensity.append(max(f.intensity for f in foci))
                
        metrics.append(EvaluationMetric(
            name="goal_completion_rate",
            value=goals_achieved / num_trials,
            unit="rate",
            description="Proportion of goals achieved or significantly progressed"
        ))
        
        if avg_focus_intensity:
            metrics.append(EvaluationMetric(
                name="mean_max_attention",
                value=statistics.mean(avg_focus_intensity),
                unit="intensity",
                description="Average maximum attention intensity during tasks"
            ))
            
        return metrics

    def evaluate_prsc(self) -> List[EvaluationMetric]:
        """Evaluate PRSC semantic coherence."""
        if not getattr(self.model, 'prsc', None):
            return []
            
        print("Evaluating PRSC...")
        metrics = []
        
        # Generate some text to populate PRSC
        self.inference.generate("The quick brown fox jumps over the lazy dog", max_length=20)
        
        if hasattr(self.model.prsc, 'global_coherence'):
            metrics.append(EvaluationMetric(
                name="global_semantic_coherence",
                value=self.model.prsc.global_coherence,
                unit="score",
                description="Global semantic coherence of the PRSC layer"
            ))
            
        return metrics

    def evaluate_stability(self, prompt: str = "The nature of reality is", max_length: int = 50) -> List[EvaluationMetric]:
        """Evaluate Stability Monitor accuracy."""
        if not getattr(self.model, 'stability_monitor', None):
            return []
            
        print("Evaluating Stability...")
        metrics = []
        
        result = self.inference.generate(prompt, max_length=max_length)
        
        metrics.append(EvaluationMetric(
            name="final_lyapunov_exponent",
            value=result.lyapunov,
            unit="lambda",
            description="Lyapunov exponent at end of generation"
        ))
        
        # Check if stability matches lyapunov roughly
        # collapsed < -0.1, stable < 0, chaotic > 0.1
        consistent = False
        if result.stability == "collapsed" and result.lyapunov < -0.1: consistent = True
        elif result.stability == "stable" and result.lyapunov < 0: consistent = True
        elif result.stability == "divergent" and result.lyapunov > 0.1: consistent = True
        elif result.stability in ["critical", "metastable"]: consistent = True # fuzzy
        elif result.stability == "unknown": consistent = True # fallback
        
        metrics.append(EvaluationMetric(
            name="stability_consistency",
            value=1.0 if consistent else 0.0,
            unit="bool",
            description="Whether stability class matches Lyapunov value"
        ))
        
        return metrics

    def evaluate_memory(self) -> List[EvaluationMetric]:
        """Evaluate Temporal SMF memory."""
        if not getattr(self.model, 'smf', None):
            return []
            
        print("Evaluating Memory...")
        metrics = []
        
        # Inject memory
        fact = "The secret code is 42."
        if hasattr(self.model, 'update_memory'):
            self.model.update_memory(fact, importance=1.0)
        
        # Query SMF directly
        if hasattr(self.model.smf, 'recall'):
            recalled = self.model.smf.recall("secret code")
            success = any("42" in m.content for m in recalled if m.content)
            
            metrics.append(EvaluationMetric(
                name="memory_recall_success",
                value=1.0 if success else 0.0,
                unit="bool",
                description="Successfully recalled injected fact"
            ))
            
            if recalled:
                metrics.append(EvaluationMetric(
                    name="top_memory_coherence",
                    value=recalled[0].coherence,
                    unit="score",
                    description="Coherence of top recalled memory"
                ))
                
        return metrics

    def run_all(self) -> Dict[str, float]:
        """Run all evaluations and return summary dict."""
        all_metrics = []
        all_metrics.extend(self.evaluate_agency())
        all_metrics.extend(self.evaluate_prsc())
        all_metrics.extend(self.evaluate_stability())
        all_metrics.extend(self.evaluate_memory())
        
        summary = {m.name: m.value for m in all_metrics}
        return summary
