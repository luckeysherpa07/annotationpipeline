EVENT_PROMPTS = {
    "event_object_recognition": {
        "caption_prompt": """I am specifically focusing on event-based visual recognition related captioning. Please describe the video by emphasizing motion, object boundaries, temporal changes, and visually salient activity patterns captured by the event stream. Focus on objects, scene structure, and changes over time that are strongly indicated by event activity. Do not rely on color, texture, or fine RGB appearance unless it is clearly inferable. Give a detailed caption that enumerates all identifiable moving or changing objects, regions, and their temporal behavior."""
    },

    "event_spatial_reasoning": {
        "caption_prompt": """I am specifically focusing on spatial relationships inferred from the event stream. Please caption the video by emphasizing motion-based spatial cues, object interactions, relative positions, occlusion boundaries, and changes in layout that can be inferred from event activity. Describe which objects are closer, farther, overlapping, entering, or leaving the field of view when possible. Do not describe color or texture. Give more than 4 object pairs if available."""
    },

    "event_scene_sequence": {
        "caption_prompt": """I am specifically focusing on scene sequence from the event stream. Please describe the order of scene transitions, motion changes, and environment changes as the camera moves through the video. Emphasize what appears first, what changes next, what disappears, and how the visual event activity evolves across the sequence. Do not rely on RGB-style appearance details. Give a detailed temporal caption of the full sequence."""
    },

    "event_counting": {
        "caption_prompt": """I am specifically focusing on counting objects or scene elements using the event stream. Please identify more than 5 useful objects or regions that can be counted from motion or event activity, and estimate their number over the full video sequence. Focus on objects that are clearly separable by movement, appearance change, or event boundaries. Do not rely on color or texture."""
    },

    "event_dynamic_counting": {
        "caption_prompt": """I am specifically focusing on counting moving humans or objects in the event stream. Please describe all moving entities except the camera holder, and estimate how many of each appear, move, pass by, or disappear over time. Include the temporal span of each moving entity when possible. Focus only on motion-based evidence."""
    },

    "event_navigation": {
        "caption_prompt": """I am specifically focusing on navigation cues from the event stream. Please describe turns, direction changes, steps, motion paths, and scene progression that can support navigation. Emphasize where the camera goes, how the viewpoint changes, and what landmarks or motion cues indicate the route. Do not rely on RGB-style appearance details. Give a detailed navigation-oriented caption."""
    },

    "event_dynamic_recognition": {
        "caption_prompt": """I am specifically focusing on recognizing dynamic humans or objects in the event stream. Please describe each moving entity, when it appears, how it moves, and when it leaves, including start and end frame if possible. Focus on motion-based identification and timing, not on appearance colors or textures."""
    },

    "event_action": {
        "caption_prompt": """I am specifically focusing on the subject's action as inferred from the event stream. Please describe the camera holder's actions, object interactions, motion intention, and possible purpose or consequence based on temporal change and movement. Focus on what the subject is doing and how the scene responds over time. Do not rely on static RGB appearance."""
    },

    "event_non_common": {
        "caption_prompt": """I am specifically focusing on unusual or physically implausible situations visible in the event stream. Please identify strange motion patterns, unrealistic object behavior, odd layouts, clipping, floating objects, or other violations of common sense that can be inferred from event activity. If nothing unusual is present, say so clearly. Do not discuss color or texture."""
    }
}