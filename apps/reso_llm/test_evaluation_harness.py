import unittest
import torch
from apps.reso_llm.config import ResoLLMConfig
from apps.reso_llm.model import ResoLLMModel
from apps.reso_llm.tokenizer import create_default_tokenizer
from apps.reso_llm.evaluation_harness import ResoEvaluationHarness

class TestResoEvaluationHarness(unittest.TestCase):
    def setUp(self):
        # Create a small extended config for testing
        self.config = ResoLLMConfig.tiny(standard=False)
        self.config.enable_extensions()
        self.model = ResoLLMModel(self.config)
        self.tokenizer = create_default_tokenizer()
        self.harness = ResoEvaluationHarness(self.model, self.tokenizer)

    def test_agency_evaluation(self):
        metrics = self.harness.evaluate_agency(num_trials=1)
        self.assertTrue(len(metrics) >= 0) # Should run without error
        for m in metrics:
            print(f"Agency Metric: {m.name} = {m.value}")

    def test_prsc_evaluation(self):
        metrics = self.harness.evaluate_prsc()
        self.assertTrue(len(metrics) >= 0)
        for m in metrics:
            print(f"PRSC Metric: {m.name} = {m.value}")

    def test_stability_evaluation(self):
        metrics = self.harness.evaluate_stability(max_length=10)
        self.assertTrue(len(metrics) >= 0)
        for m in metrics:
            print(f"Stability Metric: {m.name} = {m.value}")

    def test_memory_evaluation(self):
        metrics = self.harness.evaluate_memory()
        self.assertTrue(len(metrics) >= 0)
        for m in metrics:
            print(f"Memory Metric: {m.name} = {m.value}")

    def test_run_all(self):
        summary = self.harness.run_all()
        self.assertIsInstance(summary, dict)
        print("Run All Summary:", summary)

if __name__ == '__main__':
    unittest.main()
