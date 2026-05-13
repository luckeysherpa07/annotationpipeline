#!/usr/bin/env python
import json

# Load the detailed results to see which sentences ended up in which sections
with open('fused_qa_results.json') as f:
    data = json.load(f)

print('=== Q/A Distribution Analysis ===\n')

sample_data = data['cut_carrot']
section_qas = sample_data['section_evidence_qas']

for section, qas_list in section_qas.items():
    print(f'\n[{section.upper()}] - {len(qas_list)} items')
    print('-' * 80)
    
    for i, qa in enumerate(qas_list, 1):
        caption = qa['caption']
        answer = qa['answer']
        annotation_key = qa.get('annotation_key', 'N/A')
        source_modality = qa.get('source_modality', 'N/A')
        qa_source = qa.get('qa_source', 'N/A')
        
        # Show first 100 chars of caption
        caption_preview = (caption[:100] + '...') if len(caption) > 100 else caption
        
        print(f'{i}. [{annotation_key}] {qa_source}')
        print(f'   Caption: {caption_preview}')
        print(f'   Q: {qa["question"]}')
        print(f'   A: {answer}')
        print(f'   Source: {source_modality}')
        print()

# Summary of issues
print('\n=== Issue Analysis ===')
print('Problem sentences in wrong sections:')

# Scene Overview should have high-level scene descriptions, not spatial layout
scene_qa = section_qas.get('scene_overview', [])[0]
if 'board' in scene_qa['caption'] and 'right' in scene_qa['caption']:
    print(f"✗ Scene Overview item 1 seems like layout, not scene: '{scene_qa['caption'][:60]}...'")
    print(f'  → This should probably be in visible_objects_and_layout')
    print(f'  → Root cause: Keyword "cutting" matched scene_overview keywords')
