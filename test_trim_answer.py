#!/usr/bin/env python
from annotation_feature.fusion import _trim_answer

# Test cases
test_cases = [
    "To the right of the cutting board sits an empty white bowl",
    "To the right of the cutting",
    "laptop is positioned about 1 step to the left of the white cutting board",
    "The white bowl is placed roughly 0.5 steps to the right of the cutting board",
    "empty white bowl",
    "At the start, the carrot lies between the peeler and the knife on the cutting board",
]

print("=== Testing _trim_answer() Function ===\n")

for test in test_cases:
    result = _trim_answer(test)
    word_count_in = len(test.split())
    word_count_out = len(result.split())
    print(f"Input ({word_count_in} words): {test}")
    print(f"Output ({word_count_out} words): {result}")
    print()
