#!/usr/bin/env python3
"""
Dataset Fusion Verification Script for Reso-LLM.

This script verifies that dataset fusion properly:
1. Accounts for each dataset's format (Guanaco, DialogSum, instruction/output, etc.)
2. Produces a unified training format (User:/Assistant:) that is valid
3. Ensures question and answer parts are not blank
4. Maintains uniform formatting across all datasets

Run with:
    python -m apps.reso_llm.verify_fusion

Or:
    python apps/reso_llm/verify_fusion.py
"""
import sys
import os
import re
from typing import Dict, List, Tuple

# Ensure we can import from parent directories
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

from apps.reso_llm.tokenizer import ResoLLMTokenizer, create_default_tokenizer
from apps.reso_llm.dataset import (
    HuggingFaceDataset,
    validate_format,
    verify_dataset_fusion,
    DatasetStats
)


def check_conversation_formatting(text: str) -> Dict[str, any]:
    """
    Detailed check of a single conversation's formatting.
    
    Returns dict with:
    - valid: bool
    - issues: list of specific issues found
    - user_content: extracted user content
    - assistant_content: extracted assistant content
    """
    result = {
        "valid": True,
        "issues": [],
        "user_content": None,
        "assistant_content": None,
        "structure": []
    }
    
    if not text or not text.strip():
        result["valid"] = False
        result["issues"].append("Empty conversation")
        return result
    
    text = text.strip()
    
    # Check for User: marker
    if "User:" not in text:
        result["valid"] = False
        result["issues"].append("Missing 'User:' marker")
    
    # Check for Assistant: marker
    if "Assistant:" not in text:
        result["valid"] = False
        result["issues"].append("Missing 'Assistant:' marker")
    
    if not result["valid"]:
        return result
    
    # Check for inline markers (not preceded by newline or start of string)
    user_positions = [(m.start(), m.group()) for m in re.finditer(r'User:', text)]
    assistant_positions = [(m.start(), m.group()) for m in re.finditer(r'Assistant:', text)]
    
    for pos, marker in user_positions:
        if pos > 0 and text[pos-1] != '\n':
            result["issues"].append(f"'User:' at position {pos} not preceded by newline (char before: '{text[pos-1]}')")
    
    for pos, marker in assistant_positions:
        if pos > 0 and text[pos-1] != '\n':
            result["issues"].append(f"'Assistant:' at position {pos} not preceded by newline (char before: '{text[pos-1]}')")
    
    # Extract content
    user_match = re.search(r'User:\s*(.+?)(?=\nAssistant:|$)', text, re.DOTALL)
    assistant_match = re.search(r'Assistant:\s*(.+?)(?=\n<\|endofconversation\|>|$)', text, re.DOTALL)
    
    if user_match:
        result["user_content"] = user_match.group(1).strip()
        if not result["user_content"]:
            result["issues"].append("User content is blank")
            result["valid"] = False
    else:
        result["issues"].append("Could not extract User content")
        result["valid"] = False
    
    if assistant_match:
        result["assistant_content"] = assistant_match.group(1).strip()
        if not result["assistant_content"]:
            result["issues"].append("Assistant content is blank")
            result["valid"] = False
    else:
        result["issues"].append("Could not extract Assistant content")
        result["valid"] = False
    
    # Document structure
    result["structure"] = user_positions + assistant_positions
    result["structure"].sort(key=lambda x: x[0])
    
    if result["issues"]:
        result["valid"] = False
    
    return result


def verify_raw_formatting(dataset: HuggingFaceDataset, num_samples: int = 10) -> Dict[str, any]:
    """
    Verify raw formatting by decoding tokens and checking structure.
    
    This performs a more thorough check than the standard verify_dataset_fusion.
    """
    results = {
        "total_conversations_checked": 0,
        "valid_conversations": 0,
        "invalid_conversations": 0,
        "issues_found": {},
        "samples": []
    }
    
    if len(dataset.data) == 0:
        print("ERROR: No data in dataset")
        return results
    
    # Decode a substantial chunk
    chunk_size = min(100000, len(dataset.data))
    chunk_tokens = dataset.data[:chunk_size]
    
    try:
        chunk_text = dataset.tokenizer.decode(chunk_tokens)
    except Exception as e:
        print(f"ERROR: Failed to decode tokens: {e}")
        return results
    
    # Debug: show what's in the decoded text
    print(f"Decoded {len(chunk_text)} characters")
    print(f"Sample of decoded text (first 500 chars):")
    print(repr(chunk_text[:500]))
    print()
    
    # Count multi-turn conversations
    eoc_segments = chunk_text.split("<|endofconversation|>")
    single_turn = 0
    multi_turn = 0
    multi_turn_total_turns = 0
    
    for seg in eoc_segments:
        if not seg.strip():
            continue
        user_count = seg.count("User:")
        if user_count == 1:
            single_turn += 1
        elif user_count > 1:
            multi_turn += 1
            multi_turn_total_turns += user_count
    
    total_convs = single_turn + multi_turn
    results["multi_turn_stats"] = {
        "single_turn": single_turn,
        "multi_turn": multi_turn,
        "multi_turn_percentage": (multi_turn / max(1, total_convs)) * 100,
        "avg_turns_in_multi": multi_turn_total_turns / max(1, multi_turn)
    }
    
    print(f"Conversation structure:")
    print(f"  Single-turn: {single_turn}")
    print(f"  Multi-turn: {multi_turn} ({results['multi_turn_stats']['multi_turn_percentage']:.1f}%)")
    if multi_turn > 0:
        print(f"  Avg turns in multi-turn: {results['multi_turn_stats']['avg_turns_in_multi']:.1f}")
    print()
    
    # Find complete User:/Assistant: pairs
    # Pattern: User: ... Assistant: ... (until next User: or EOC or end)
    # For multi-turn conversations, we extract each User-Assistant pair individually
    
    pair_pattern = r'User:\s*(.+?)\n+Assistant:\s*(.+?)(?=\n\nUser:|\n<\|endofconversation\|>|$)'
    pair_matches = list(re.finditer(pair_pattern, chunk_text, re.DOTALL))
    
    print(f"Found {len(pair_matches)} complete User/Assistant pairs")
    
    # Extract conversations - each complete pair is a valid conversation
    conversations = []
    
    for match in pair_matches[:50]:  # Check first 50 complete pairs
        user_content = match.group(1).strip()
        assistant_content = match.group(2).strip()
        
        # Reconstruct the conversation text for validation
        conv = f"User: {user_content}\nAssistant: {assistant_content}"
        if conv:
            conversations.append(conv)
    
    print(f"\nFound {len(conversations)} conversation segments in first {chunk_size:,} tokens")
    print("-" * 60)
    
    for i, conv in enumerate(conversations):
        conv = conv.strip()
        if not conv:
            continue
        
        results["total_conversations_checked"] += 1
        check = check_conversation_formatting(conv)
        
        if check["valid"]:
            results["valid_conversations"] += 1
        else:
            results["invalid_conversations"] += 1
            for issue in check["issues"]:
                results["issues_found"][issue] = results["issues_found"].get(issue, 0) + 1
        
        # Collect samples (both valid and invalid for review)
        if len(results["samples"]) < num_samples:
            results["samples"].append({
                "index": i,
                "valid": check["valid"],
                "issues": check["issues"],
                "user_preview": (check["user_content"] or "")[:150],
                "assistant_preview": (check["assistant_content"] or "")[:150],
                "raw_text": conv[:300] if not check["valid"] else None  # Only show raw for invalid
            })
    
    return results


def format_test_cases() -> List[Tuple[str, str, bool, bool]]:
    """
    Return test cases for format validation.
    
    Each test case is: (name, text, expected_valid, require_newlines)
    """
    return [
        # Valid cases (no newline requirement)
        (
            "Simple valid conversation",
            "User: Hello\nAssistant: Hi there!",
            True,
            False
        ),
        (
            "Valid with EOC marker",
            "User: What is 2+2?\nAssistant: 4\n<|endofconversation|>",
            True,
            False
        ),
        (
            "Multi-line valid",
            "User: Tell me a story\nAssistant: Once upon a time...\nThere was a kingdom.",
            True,
            False
        ),
        # Invalid cases
        (
            "Inline Assistant marker (strict)",
            "User: Hello?Assistant: Hi",
            False,  # Should be invalid when require_newlines=True
            True    # require_newlines
        ),
        (
            "Inline Assistant marker (lenient)",
            "User: Hello?Assistant: Hi",
            True,   # Valid when require_newlines=False (just checks order and content)
            False   # don't require newlines
        ),
        (
            "Missing User",
            "Assistant: Hi there!",
            False,
            False
        ),
        (
            "Missing Assistant",
            "User: Hello there",
            False,
            False
        ),
        (
            "Blank User content",
            "User:\nAssistant: Response",
            False,
            False
        ),
        (
            "Blank Assistant content",
            "User: Question\nAssistant:",
            False,
            False
        ),
        (
            "Empty string",
            "",
            False,
            False
        ),
        (
            "Wrong order",
            "Assistant: Hi\nUser: Hello",
            False,
            False
        ),
    ]


def run_format_unit_tests():
    """Run unit tests on format validation."""
    print("\n" + "=" * 60)
    print("Format Validation Unit Tests")
    print("=" * 60)
    
    test_cases = format_test_cases()
    passed = 0
    failed = 0
    
    for name, text, expected, require_newlines in test_cases:
        is_valid, reason = validate_format(text, require_newlines=require_newlines)
        
        if is_valid == expected:
            status = "✓ PASS"
            passed += 1
        else:
            status = "✗ FAIL"
            failed += 1
        
        mode = " (strict)" if require_newlines else ""
        print(f"{status}: {name}{mode}")
        if is_valid != expected:
            print(f"       Expected: {expected}, Got: {is_valid} (reason: {reason})")
            print(f"       Text: {repr(text[:50])}")
    
    print("-" * 60)
    print(f"Results: {passed} passed, {failed} failed out of {len(test_cases)} tests")
    
    return failed == 0


def run_guanaco_format_test():
    """Test the Guanaco format conversion specifically."""
    print("\n" + "=" * 60)
    print("Guanaco Format Conversion Test")
    print("=" * 60)
    
    # Simulate what _format_guanaco_example does
    test_inputs = [
        # Standard Guanaco format
        "### Human: What is the meaning of life?### Assistant: The meaning of life is a profound question.",
        # With existing newlines
        "### Human: Hello!\n### Assistant: Hi there!",
        # With inline markers (problematic case)
        "### Human: Como estas?### Assistant: Muy bien, gracias!",
        # Multi-turn
        "### Human: First question### Assistant: First answer### Human: Follow-up### Assistant: Follow-up answer",
    ]
    
    def format_guanaco(text: str) -> str:
        """Replicate the formatting logic from dataset.py"""
        if not isinstance(text, str):
            return ""
        
        # Replace Guanaco markers with our format
        text = text.replace("### Human:", "\n\nUser: ")
        text = text.replace("### Assistant:", "\n\nAssistant: ")
        
        # Ensure newlines before markers
        text = re.sub(r'(?<!\n)User:', '\nUser:', text)
        text = re.sub(r'(?<!\n)Assistant:', '\nAssistant:', text)
        
        # Clean up
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'\n +', '\n', text)
        text = text.strip()
        
        if text:
            text = text + "\n<|endofconversation|>\n"
        
        return text
    
    all_passed = True
    
    for i, input_text in enumerate(test_inputs):
        print(f"\nTest {i+1}: Input = {repr(input_text[:60])}...")
        
        output = format_guanaco(input_text)
        check = check_conversation_formatting(output)
        
        print(f"  Output structure check: {'✓ VALID' if check['valid'] else '✗ INVALID'}")
        
        if check["issues"]:
            print(f"  Issues: {check['issues']}")
            all_passed = False
        
        if check["user_content"]:
            print(f"  User: {check['user_content'][:50]}...")
        if check["assistant_content"]:
            print(f"  Assistant: {check['assistant_content'][:50]}...")
        
        # Show raw output for debugging
        print(f"  Raw output (first 100 chars): {repr(output[:100])}")
    
    return all_passed


def main():
    """Main verification routine."""
    print("=" * 60)
    print("Reso-LLM Dataset Fusion Verification")
    print("=" * 60)
    
    # Run unit tests first
    format_tests_passed = run_format_unit_tests()
    guanaco_tests_passed = run_guanaco_format_test()
    
    if not format_tests_passed:
        print("\n⚠️  Some format unit tests failed!")
    
    if not guanaco_tests_passed:
        print("\n⚠️  Some Guanaco format tests failed!")
    
    # Now test with real dataset
    print("\n" + "=" * 60)
    print("Loading Real Dataset for Verification")
    print("=" * 60)
    
    try:
        tokenizer = create_default_tokenizer()
        
        # Load a small sample
        dataset = HuggingFaceDataset(
            dataset_name="timdettmers/openassistant-guanaco",
            tokenizer=tokenizer,
            seq_len=256,
            batch_size=32,
            max_tokens=500_000,  # Small sample for testing
            validate_format=True
        )
        
        # Run standard verification
        print("\n--- Standard Verification ---")
        standard_results = verify_dataset_fusion(dataset, num_samples=5, verbose=True)
        
        # Run detailed verification
        print("\n--- Detailed Format Verification ---")
        detailed_results = verify_raw_formatting(dataset, num_samples=10)
        
        print("\n" + "=" * 60)
        print("Detailed Verification Summary")
        print("=" * 60)
        print(f"Total conversations checked: {detailed_results['total_conversations_checked']}")
        print(f"Valid: {detailed_results['valid_conversations']}")
        print(f"Invalid: {detailed_results['invalid_conversations']}")
        
        if detailed_results["issues_found"]:
            print("\nIssues found:")
            for issue, count in sorted(detailed_results["issues_found"].items(), key=lambda x: -x[1]):
                print(f"  - {issue}: {count} occurrences")
        
        print("\n--- Sample Conversations ---")
        for sample in detailed_results["samples"][:5]:
            status = "✓ VALID" if sample["valid"] else "✗ INVALID"
            print(f"\n[Conversation {sample['index']}] {status}")
            if sample["issues"]:
                print(f"  Issues: {sample['issues']}")
            print(f"  User: {sample['user_preview'][:80]}...")
            print(f"  Assistant: {sample['assistant_preview'][:80]}...")
            if sample["raw_text"]:
                print(f"  Raw: {repr(sample['raw_text'][:100])}...")
        
        # Final summary
        print("\n" + "=" * 60)
        print("FINAL VERIFICATION SUMMARY")
        print("=" * 60)
        
        all_good = True
        
        if not format_tests_passed:
            print("✗ Format unit tests: FAILED")
            all_good = False
        else:
            print("✓ Format unit tests: PASSED")
        
        if not guanaco_tests_passed:
            print("✗ Guanaco format tests: FAILED")
            all_good = False
        else:
            print("✓ Guanaco format tests: PASSED")
        
        if standard_results["valid"]:
            print("✓ Standard dataset verification: PASSED")
        else:
            print("✗ Standard dataset verification: FAILED")
            all_good = False
        
        valid_pct = detailed_results["valid_conversations"] / max(1, detailed_results["total_conversations_checked"]) * 100
        if valid_pct >= 95:
            print(f"✓ Detailed verification: {valid_pct:.1f}% valid (PASSED)")
        elif valid_pct >= 80:
            print(f"⚠️  Detailed verification: {valid_pct:.1f}% valid (WARNING)")
        else:
            print(f"✗ Detailed verification: {valid_pct:.1f}% valid (FAILED)")
            all_good = False
        
        if all_good:
            print("\n✓✓✓ ALL VERIFICATIONS PASSED ✓✓✓")
        else:
            print("\n⚠️  SOME VERIFICATIONS FAILED - Review issues above")
        
        return 0 if all_good else 1
        
    except ImportError as e:
        print(f"Error: Missing dependency - {e}")
        print("Install with: pip install datasets")
        return 1
    except Exception as e:
        import traceback
        print(f"Error during verification: {e}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
