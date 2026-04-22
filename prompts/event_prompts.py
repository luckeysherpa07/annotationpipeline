EVENT_PROMPTS = {
    "event_object_recognition": {
        "caption_prompt": """I am specifically focusing on event-based visual recognition related captioning. Please describe the video by emphasizing motion, object boundaries, temporal changes, and visually salient activity patterns captured by the event stream. Focus on objects, scene structure, and changes over time that are strongly indicated by event activity. Do not rely on color, texture, or fine RGB appearance unless it is clearly inferable. Give a detailed caption that enumerates all identifiable moving or changing objects, regions, and their temporal behavior.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates fine-grained understanding of visible objects and their motion. Avoid questions about color, texture, illumination, or static appearance. Focus only on motion-based, event-visible objects and regions. Make sure the question can be answered in a few words and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the question, generate a short and factual answer. Use only visible motion- and event-based cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable (ideally 1–5 words)."""
    },


    "event_spatial_reasoning": {
        "caption_prompt": """I am specifically focusing on spatial relationships inferred from the event stream. Please caption the video by emphasizing motion-based spatial cues, object interactions, relative positions, occlusion boundaries, and changes in layout that can be inferred from event activity. Describe which objects are closer, farther, overlapping, entering, or leaving the field of view when possible. Do not describe color or texture. Give more than 4 object pairs if available.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates fine-grained understanding of spatial relations between objects. Avoid questions about color, texture, or illumination. Focus only on motion-based distance, occlusion, and relative positions. Make sure the question can be answered in a few words and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the question, generate a short and factual answer. Use only motion-based spatial cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable (ideally 1–5 words)."""
    },


    "event_scene_sequence": {
        "caption_prompt": """I am specifically focusing on scene sequence from the event stream. Please describe the order of scene transitions, motion changes, and environment changes as the camera moves through the video. Emphasize what appears first, what changes next, what disappears, and how the visual event activity evolves across the sequence. Do not rely on RGB-style appearance details. Give a detailed temporal caption of the full sequence.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates understanding of the full scene sequence. Avoid questions that can be answered from a few frames. Ask about the order of scenes, transitions, or environment changes inferred from motion and event activity. Make sure the question can be answered in a few words and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the question, generate a short and factual answer. Use only motion-based and temporal cues from the full sequence and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable."""
    },


    "event_counting": {
        "caption_prompt": """I am specifically focusing on counting objects or scene elements using the event stream. Please identify more than 5 useful objects or regions that can be counted from motion or event activity, and estimate their number over the full video sequence. Focus on objects that are clearly separable by movement, appearance change, or event boundaries. Do not rely on color or texture.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 8 first-person questions that evaluate counting of objects or regions. Avoid questions that can be answered from a few frames; instead, ask about the total number of items, objects, or regions over the full video. Focus on motion-based evidence and avoid questions about color, texture, or illumination. Make sure each question can be answered with a precise number or short phrase.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the questions, generate short and factual answers. Use only motion-based and event-based counting cues and avoid mentioning color, texture, or illumination. Keep each answer concise and evaluable (e.g., a number or short phrase)."""
    },


    "event_dynamic_counting": {
        "caption_prompt": """I am specifically focusing on counting moving humans or objects in the event stream. Please describe all moving entities except the camera holder, and estimate how many of each appear, move, pass by, or disappear over time. Include the temporal span of each moving entity when possible. Focus only on motion-based evidence.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 8 first-person questions that evaluate counting of dynamic objects or humans. Avoid questions about the camera holder and avoid questions regarding illumination. Focus on questions like ‘How many cars approach me?’, ‘How many people pass by?’, or ‘How many times does an object move?’. Each question should rely on motion-based evidence and be answerable with a short number or phrase.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the questions, generate short and factual answers. Use only motion-based temporal cues and avoid answering about illumination or static appearance. Keep each answer concise and evaluable (e.g., a number or short phrase)."""
    },


    "event_navigation": {
        "caption_prompt": """I am specifically focusing on navigation cues from the event stream. Please describe turns, direction changes, steps, motion paths, and scene progression that can support navigation. Emphasize where the camera goes, how the viewpoint changes, and what landmarks or motion cues indicate the route. Do not rely on RGB-style appearance details. Give a detailed navigation-oriented caption.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates understanding of navigation. Avoid questions that can be answered from a few frames; instead, ask about the path taken, the best route, or a sequence of directions inferred from motion and event activity. Make sure the question can be answered with a short, precise answer and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the question, generate a short and factual navigation answer. Use only motion-based viewpoint and path cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable."""
    },


    "event_dynamic_recognition": {
        "caption_prompt": """I am specifically focusing on recognizing dynamic humans or objects in the event stream. Please describe each moving entity, when it appears, how it moves, and when it leaves, including start and end frame if possible. Focus on motion-based identification and timing, not on appearance colors or textures.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 8 precise first-person questions that evaluate understanding of dynamic objects or humans. Avoid questions about the camera holder and about illumination. Focus on questions like ‘Did a car pass by?’, ‘When did a person appear?’, or ‘Has the door fully closed?’. Make sure each question relies on motion-based evidence and can be answered shortly.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the questions, generate short and factual answers. Use only motion-based timing and behavior cues and avoid answering about illumination or static appearance. Keep each answer concise and evaluable."""
    },


    "event_action": {
        "caption_prompt": """I am specifically focusing on the subject's action as inferred from the event stream. Please describe the camera holder's actions, object interactions, motion intention, and possible purpose or consequence based on temporal change and movement. Focus on what the subject is doing and how the scene responds over time. Do not rely on static RGB appearance.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 8 first-person questions that evaluate understanding of the subject’s actions. Avoid questions about illumination or dynamic objects/humans other than those interacting with the subject. Focus on questions about what the subject is doing, which object they interact with, and how long or how the motion evolves. Make sure each question can be answered with a short, precise answer grounded in motion and event cues.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the questions, generate short and factual answers. Use only motion-based action and interaction cues and avoid mentioning illumination or RGB appearance. Keep each answer concise and evaluable."""
    },


    "event_non_common": {
        "caption_prompt": """I am specifically focusing on unusual or physically implausible situations visible in the event stream. Please identify strange motion patterns, unrealistic object behavior, odd layouts, clipping, floating objects, or other violations of common sense that can be inferred from event activity. If nothing unusual is present, say so clearly. Do not discuss color or texture.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the event stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates understanding of non‑common‑sense or physically implausible situations. Focus only on motion, layout, and simulator‑related issues that can be inferred from the event activity. Avoid questions about illumination, color, or texture. Make sure the question can be answered with a short, factual response.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the event stream, the caption, and the question, generate a short and factual answer. Use only motion-based and layout-based cues and avoid mentioning illumination, color, or texture. Keep the answer concise and evaluable."""
    }
}