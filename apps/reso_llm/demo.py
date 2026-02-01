"""
Reso-LLM Demo.

Demonstrates the capabilities of the Resonant LLM architecture:
1. Sparse Prime Embeddings
2. Holographic Memory (SMF)
3. Dynamic Context (Kuramoto)
4. Stability Analysis (Lyapunov)
"""
import sys
import os
import random

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.inference import ResoLLMInference

def main():
    print("Initializing Reso-LLM...")
    
    # 1. Configuration
    config = ResoLLMConfig(
        vocab_size=1000, # Small vocab for demo
        dim=64,          # Small dim for speed
        num_layers=2,
        num_heads=4,
        use_smf_memory=True,
        use_kuramoto_dynamics=True
    )
    
    # 2. Tokenizer
    tokenizer = create_default_tokenizer()
    
    # 3. Model
    model = ResoLLMModel(config)
    
    # 4. Inference Engine
    engine = ResoLLMInference(model, tokenizer)
    
    print("\n=== Architecture Overview ===")
    print(f"Embedding: SparsePrimeEmbedding (Dim: {config.dim})")
    print(f"Memory: SedenionMemoryField (Dim: {config.smf_dim})")
    print(f"Dynamics: KuramotoModel (Coupling: {config.kuramoto_coupling})")
    print(f"Safety: Lyapunov Stability Analysis")
    
    # Demo 1: Memory Injection
    print("\n=== Demo 1: Holographic Memory Injection ===")
    context = "The secret code is 42."
    print(f"Injecting context: '{context}'")
    model.update_memory(context, importance=1.0)
    
    # Show memory state
    if model.smf:
        print(f"Memory Coherence: {model.smf.mean_coherence:.3f}")
        print(f"Dominant Axes: {[a['name'] for a in model.smf.dominant_axes(2)]}")
        
    # Demo 2: Generation with Stability
    print("\n=== Demo 2: Generation with Stability Analysis ===")
    prompt = "What is the secret?"
    print(f"Prompt: '{prompt}'")
    
    # Since the model is untrained, output will be random, but we can observe the dynamics
    result = engine.generate(prompt, max_length=20)
    
    print(f"Generated: '{result['text']}'")
    print(f"Stability: {result['stability']}")
    print(f"Lyapunov Exponent: {result['lyapunov']:.3f}")
    
    # Demo 3: Dynamic Synchronization
    print("\n=== Demo 3: Kuramoto Synchronization ===")
    if model.kuramoto:
        initial_sync = model.kuramoto.synchronization()
        print(f"Initial Synchronization: {initial_sync:.3f}")
        
        # Simulate processing steps
        print("Processing tokens...")
        for _ in range(5):
            model.kuramoto.step()
            
        final_sync = model.kuramoto.synchronization()
        print(f"Final Synchronization: {final_sync:.3f}")
        
    print("\nDemo completed successfully.")

if __name__ == "__main__":
    main()
