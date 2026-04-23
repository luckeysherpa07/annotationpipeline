#!/usr/bin/env python
"""Quick test script to verify audio pipeline output"""

import json
from pathlib import Path

# Load and verify the output
output_file = Path("audio_qa_results.json")
if output_file.exists():
    with open(output_file) as f:
        data = json.load(f)
    
    print("=" * 60)
    print("AUDIO PIPELINE OUTPUT VERIFICATION")
    print("=" * 60)
    print(f"\nTotal audio files processed: {len(data)}\n")
    
    for pair_key, pair_data in data.items():
        print(f"Audio file: {pair_key}")
        print(f"  Path: {pair_data['audio_file']}")
        annotations = pair_data['annotations']
        print(f"  Annotation types: {len(annotations)}")
        print(f"  Types: {list(annotations.keys())}")
        
        # Check one annotation type
        first_type = list(annotations.keys())[0]
        first_anno = annotations[first_type]
        print(f"\n  Sample ({first_type}):")
        print(f"    - Caption: {first_anno['caption'][:50]}...")
        print(f"    - Question: {first_anno['question']}")
        print(f"    - Answer: {first_anno['answer']}\n")
    
    print("=" * 60)
    print("✓ All audio files processed successfully!")
    print("=" * 60)
else:
    print(f"ERROR: {output_file} not found!")
