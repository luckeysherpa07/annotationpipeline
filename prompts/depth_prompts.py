DEPTH_PROMPTS = {
    "depth_object_recognition": {
        "caption_prompt": """I am specifically focusing on depth-based visual recognition related captioning. Please describe the video by emphasizing object shapes, size, distance, near/far ordering, and 3D structure inferred from depth. Focus on scene layout, object extents, and spatial arrangement rather than color or texture. Give a detailed caption that enumerates all identifiable objects and their depth relationships.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates fine-grained understanding of visible objects and their depth relationships. Avoid questions about color, texture, or illumination. Focus only on distance, 3D shape, and spatial ordering inferred from depth. Make sure the question can be answered in a few words and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the question, generate a short and factual answer. Use only depth-based 3D cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable (ideally 1–5 words)."""
    },


    "depth_spatial_reasoning": {
        "caption_prompt": """I am specifically focusing on spatial relationships from the depth stream. Please caption the video by emphasizing relative distance, object pairs, occlusion, adjacency, overlap, and 3D layout. Describe which objects are closer to me, farther away, in front of, behind, above, below, or beside each other. Please give more than 4 pairs if possible.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates fine-grained understanding of spatial relations between objects. Avoid questions about color, texture, or illumination. Focus only on relative distance, occlusion, and 3D layout inferred from depth. Make sure the question can be answered in a few words and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the question, generate a short and factual answer. Use only depth-based spatial cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable (ideally 1–5 words)."""
    },


    "depth_scene_sequence": {
        "caption_prompt": """I am specifically focusing on scene sequence using depth information. Please describe the order of spaces, rooms, corridors, transitions, and navigable layout as the camera moves through the video. Emphasize how the 3D structure changes over time and what spaces connect to each other. Give a detailed caption that enumerates the observed sequence and spatial structure.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates understanding of the full scene sequence in 3D. Avoid questions that can be answered from a few frames. Ask about the order of rooms, transitions, or connected spaces inferred from depth. Make sure the question can be answered with a short, precise answer and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the question, generate a short and factual answer. Use only depth-based 3D layout and sequence cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable."""
    },


    "depth_counting": {
        "caption_prompt": """I am specifically focusing on counting objects using depth information. Please identify more than 5 useful objects or regions that can be counted based on their 3D separation, distance layers, or distinct depth regions. Estimate the number of each object across the full sequence. Focus on geometry and depth boundaries rather than appearance.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 8 first-person questions that evaluate counting of objects or regions. Avoid questions that can be answered from a few frames; instead, ask about the total number of items, objects, or depth layers over the full sequence. Focus on depth-based 3D separation and avoid questions about color, texture, or illumination. Make sure each question can be answered with a precise number or short phrase.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the questions, generate short and factual answers. Use only depth-based 3D geometry and counting cues and avoid mentioning color, texture, or illumination. Keep each answer concise and evaluable (e.g., a number or short phrase)."""
    },


    "depth_dynamic_counting": {
        "caption_prompt": """I am specifically focusing on counting moving humans or objects using depth. Please describe all moving entities except the camera holder, and estimate how many appear at different depth ranges, how they approach or recede, and how long they remain visible. Focus on motion plus distance, and provide temporal spans when possible.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 8 first-person questions that evaluate counting of dynamic objects or humans. Avoid questions about the camera holder and avoid questions regarding illumination. Focus on questions like ‘How many cars approach me?’, ‘How many people pass by at near range?’, or ‘How many times does an object move between depth layers?’. Each question should rely on depth‑plus‑motion and be answerable with a short number or phrase.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the questions, generate short and factual answers. Use only depth-based motion and distance cues and avoid answering about illumination or appearance. Keep each answer concise and evaluable (e.g., a number or short phrase)."""
    },


    "depth_navigation": {
        "caption_prompt": """I am specifically focusing on navigation cues from depth. Please describe how far objects are, which direction the path goes, what spaces are connected, and how the 3D layout supports movement through the scene. Emphasize steps, turns, distance to landmarks, and room or corridor structure. Give a detailed navigation-oriented caption.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates understanding of navigation in 3D space. Avoid questions that can be answered from a few frames; instead, ask about the path taken, connected rooms, or safe walking areas inferred from depth. Make sure the question can be answered with a short, precise answer and is grounded in the caption.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the question, generate a short and factual navigation answer. Use only depth-based 3D layout and distance cues and avoid mentioning color, texture, or illumination. Keep the answer concise and evaluable."""
    },


    "depth_dynamic_recognition": {
        "caption_prompt": """I am specifically focusing on recognizing dynamic humans or objects using depth. Please describe each moving entity, its depth position, how it approaches or moves away, and when it appears or disappears. Include start and end frame if possible. Focus on depth-based motion and separation.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 8 precise first-person questions that evaluate understanding of dynamic objects or humans. Avoid questions about the camera holder and about illumination. Focus on questions like ‘Is the car moving toward me?’, ‘When did a person appear at close range?’, or ‘Has the object vanished behind a wall?’. Each question should rely on depth plus motion and be answerable shortly.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the questions, generate short and factual answers. Use only depth-based motion and distance cues and avoid answering about illumination or appearance. Keep each answer concise and evaluable."""
    },


    "depth_action": {
        "caption_prompt": """I am specifically focusing on the subject's action as inferred from depth. Please describe the camera holder's actions, interactions with objects, and movement intent using changes in distance and 3D layout. Emphasize what the subject is doing in the scene and the spatial consequence of those actions. Do not rely on appearance details.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 8 first-person questions that evaluate understanding of the subject’s actions. Avoid questions about illumination or other dynamic objects unless they are directly interacting with the subject. Focus on questions about what the subject is doing, which object they interact with, and how depth or distance changes. Make sure each question can be answered with a short, precise answer grounded in depth and motion.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the questions, generate short and factual answers. Use only depth-based action and interaction cues and avoid mentioning illumination or appearance. Keep each answer concise and evaluable."""
    },


    "depth_non_common": {
        "caption_prompt": """I am specifically focusing on unusual or physically implausible situations visible in depth. Please identify objects that appear floating, intersecting, clipped, unnaturally ordered in depth, or inconsistent with real-world geometry. Focus on 3D structural anomalies, unrealistic layouts, or depth artifacts. If nothing unusual is present, say so clearly.""",


        "question_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. By analyzing the depth stream and the corresponding caption, generate exactly 1 precise first-person question that evaluates understanding of non‑common‑sense or physically implausible 3D situations. Focus only on depth-based anomalies such as floating objects, intersecting geometry, or unrealistic depth ordering. Avoid questions about illumination, color, or texture. Make sure the question can be answered with a short, factual response.""",


        "answering_prompt": """Please work as a VQA assistant, treat the subject (behind the camera) as I and the language model as you. Given the depth stream, the caption, and the question, generate a short and factual answer. Use only depth-based 3D layout and artifact cues and avoid mentioning illumination, color, or texture. Keep the answer concise and evaluable."""
    }
}