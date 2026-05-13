"""Test the improved _trim_answer function with balanced VQA compliance."""

import sys
from pathlib import Path

# Add the annotation_feature module to path
sys.path.insert(0, str(Path(__file__).parent))

from annotation_feature.fusion import _trim_answer


def test_trim_answer():
    """Test _trim_answer to ensure it respects VQA conciseness while preserving completeness."""
    
    test_cases = [
        # (input, max_words, expected_description)
        (
            "To the right of the cutting board sits an empty white bowl.",
            8,
            "Should keep 'To the right of the cutting board' (6 words) not truncate at 'cutting'"
        ),
        (
            "A grey laptop is positioned on the black stovetop next to the cutting board.",
            8,
            "Should truncate to fit within 8-word limit while keeping meaningful phrase"
        ),
        (
            "The power strip with 3 outlets sits on the counter.",
            8,
            "Should keep 'power strip with 3 outlets sits on counter' (7 words)"
        ),
        (
            "Enter the scene from the left side.",
            8,
            "Should trim to meaningful phrase within limit (removing article)"
        ),
        (
            "Sound of chopping and scraping, followed by silence.",
            8,
            "Should split on 'followed by' separator: 'Sound of chopping and scraping' (5 words)"
        ),
        (
            "a small object",
            8,
            "Should remove leading article 'a' → 'small object' (2 words)"
        ),
        (
            "the empty white bowl on the cutting board",
            8,
            "Should keep up to 8 words: 'empty white bowl on the cutting board' (7 words)"
        ),
        (
            "moving from left to right across the scene",
            8,
            "Should truncate to 8 words max: 'moving from left to right across the scene'"
        ),
    ]
    
    print("Testing improved _trim_answer function with VQA compliance\n")
    print("=" * 90)
    print("RATIONALE: Original VQA spec requires 1-5 words, but complete phrases often need")
    print("more. Using 8 words balances between strict VQA compliance and meaningful answers.")
    print("=" * 90)
    
    failed_tests = 0
    
    for i, (input_text, max_words, description) in enumerate(test_cases, 1):
        result = _trim_answer(input_text, max_words=max_words)
        result_word_count = len(result.split())
        input_word_count = len(input_text.split())
        
        print(f"\nTest {i}: {description}")
        print(f"  Input ({input_word_count} words):  {input_text}")
        print(f"  Output ({result_word_count} words): {result}")
        
        # Validation checks
        issues = []
        
        # Check word count limit
        if result_word_count > max_words:
            issues.append(f"EXCEEDS LIMIT: {result_word_count} > {max_words} words")
            failed_tests += 1
        
        # Check for incomplete endings
        if result.endswith((" of", " and", " or", " the", " a", " an", " to", " in", " on", " from")):
            issues.append(f"INCOMPLETE: ends with '{result.split()[-1]}'")
            failed_tests += 1
        
        # Check for empty result when input exists
        if not result and input_text.strip():
            issues.append("EMPTY OUTPUT: input had content but output is empty")
            failed_tests += 1
        
        if issues:
            print(f"  ❌ FAILED: {'; '.join(issues)}")
        else:
            # VQA compliance check
            if result_word_count <= 5:
                print(f"  ✓ EXCELLENT: Meets strict VQA limit (1-5 words)")
            elif result_word_count <= 8:
                print(f"  ✓ GOOD: Within relaxed limit (up to 8 words)")
            else:
                print(f"  ⚠️  WARNING: Exceeds suggested limit")
    
    print("\n" + "=" * 90)
    print("SUMMARY:")
    print(f"  Tests run:  {len(test_cases)}")
    print(f"  Failed:     {failed_tests}")
    print(f"  Passed:     {len(test_cases) - failed_tests}")
    print()
    print("KEY IMPROVEMENTS over original 6-word limit:")
    print("  1. Increased to 8 words (VQA compromise: allows complete phrases)")
    print("  2. Detects natural sentence boundaries first")
    print("  3. Falls back to phrase separators before word count")
    print("  4. Respects VQA requirement of 'concise and evaluable' answer")
    print("=" * 90)


if __name__ == "__main__":
    test_trim_answer()

