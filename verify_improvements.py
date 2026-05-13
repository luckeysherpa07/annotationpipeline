#!/usr/bin/env python
import time
import json
from pathlib import Path
from annotation_feature.fusion import run_late_fusion
from annotation_feature.analyze_fusion_outputs import build_reports

print('Running fusion with improved patterns and Q/A fallback...')
start = time.time()
run_late_fusion(relax_filters=True, collect_diagnostics=True)
elapsed = time.time() - start
print(f'✓ Fusion completed in {elapsed:.2f}s')

print('Generating reports...')
build_reports(
    fused_results_path=Path('fused_qa_results.json'),
    diagnostics_path=Path('fusion_diagnostics.json'),
    output_json_path=Path('fusion_section_qa_stats.json'),
    output_csv_path=Path('fusion_section_qa_rows.csv')
)
print('✓ Reports generated')
print()

print('=== Checking Improved Q/A Entries ===')
with open('fused_qa_results.json') as f:
    data = json.load(f)

# Check scene_overview entries
print('\n[Scene Overview - First Entry]')
first_qa = data['cut_carrot']['section_evidence_qas']['scene_overview'][0]
print(f'Question: {first_qa["question"]}')
print(f'Answer: {first_qa["answer"]}')
print(f'QA Source: {first_qa.get("qa_source", "N/A")}')
print(f'Caption: {first_qa["caption"]}')

# Check visible_objects_and_layout
print('\n[Visible Objects - First Entry]')
layout_qa = data['cut_carrot']['section_evidence_qas']['visible_objects_and_layout'][0]
print(f'Question: {layout_qa["question"]}')
print(f'Answer: {layout_qa["answer"]}')
print(f'QA Source: {layout_qa.get("qa_source", "N/A")}')
print(f'Caption: {layout_qa["caption"]}')

# Count qa_source distribution
print('\n=== Q/A Source Distribution ===')
source_counts = {'extracted': 0, 'original_modality': 0}
for section_qas in data.values():
    for qas_list in section_qas['section_evidence_qas'].values():
        for qa in qas_list:
            source = qa.get('qa_source', 'unknown')
            if source in source_counts:
                source_counts[source] += 1

print(f'Extracted (pattern-based): {source_counts["extracted"]}')
print(f'Original Modality (fallback): {source_counts["original_modality"]}')
print(f'Total: {sum(source_counts.values())}')
