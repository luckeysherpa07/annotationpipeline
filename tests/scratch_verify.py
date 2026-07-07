import json
import sys
import traceback
import faulthandler
from annotation_feature.aligned_multimodal_caption_pipeline import _validate_caption_schema, CaptionValidationError

faulthandler.enable()

def main():
    with open('outputs/temp_bike_8_plus_2_gemini_3.5_flash.json') as f:
        d = json.load(f)
    
    cap = d['items'][0]['caption']
    cap['schema_version'] = 'cross_modal_disambiguation_caption_v10'
    
    frames = set()
    for k in ('video1_analysis', 'video2_analysis'):
        for a in cap.get(k, {}).get('information_atoms', []):
            frames.update(a.get('frame_keys', []))
            
    try:
        print("Starting validation...")
        res = _validate_caption_schema(cap, frames, 'rgb', 'event')
        print("Success!")
        print("Warnings:", res.get('validation_warnings', []))
    except Exception as e:
        print("Failed validation (this is expected for the old JSON):")
        traceback.print_exc()

if __name__ == '__main__':
    import threading, os, time
    def timeout():
        time.sleep(2)
        print("Timeout! Dumping traceback:")
        faulthandler.dump_traceback()
        os._exit(1)
    threading.Thread(target=timeout, daemon=True).start()
    main()
