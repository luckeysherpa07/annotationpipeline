DEPTH_PROMPTS = {
    "depth_object_recognition": {
        "caption_prompt": """I am specifically focusing on depth-based visual recognition related captioning. Please describe the video by emphasizing object shapes, size, distance, near/far ordering, and 3D structure inferred from depth. Focus on scene layout, object extents, and spatial arrangement rather than color or texture. Give a detailed caption that enumerates all identifiable objects and their depth relationships."""
    },

    "depth_spatial_reasoning": {
        "caption_prompt": """I am specifically focusing on spatial relationships from the depth stream. Please caption the video by emphasizing relative distance, object pairs, occlusion, adjacency, overlap, and 3D layout. Describe which objects are closer to me, farther away, in front of, behind, above, below, or beside each other. Please give more than 4 pairs if possible."""
    },

    "depth_scene_sequence": {
        "caption_prompt": """I am specifically focusing on scene sequence using depth information. Please describe the order of spaces, rooms, corridors, transitions, and navigable layout as the camera moves through the video. Emphasize how the 3D structure changes over time and what spaces connect to each other. Give a detailed caption that enumerates the observed sequence and spatial structure."""
    },

    "depth_counting": {
        "caption_prompt": """I am specifically focusing on counting objects using depth information. Please identify more than 5 useful objects or regions that can be counted based on their 3D separation, distance layers, or distinct depth regions. Estimate the number of each object across the full sequence. Focus on geometry and depth boundaries rather than appearance."""
    },

    "depth_dynamic_counting": {
        "caption_prompt": """I am specifically focusing on counting moving humans or objects using depth. Please describe all moving entities except the camera holder, and estimate how many appear at different depth ranges, how they approach or recede, and how long they remain visible. Focus on motion plus distance, and provide temporal spans when possible."""
    },

    "depth_navigation": {
        "caption_prompt": """I am specifically focusing on navigation cues from depth. Please describe how far objects are, which direction the path goes, what spaces are connected, and how the 3D layout supports movement through the scene. Emphasize steps, turns, distance to landmarks, and room or corridor structure. Give a detailed navigation-oriented caption."""
    },

    "depth_dynamic_recognition": {
        "caption_prompt": """I am specifically focusing on recognizing dynamic humans or objects using depth. Please describe each moving entity, its depth position, how it approaches or moves away, and when it appears or disappears. Include start and end frame if possible. Focus on depth-based motion and separation."""
    },

    "depth_action": {
        "caption_prompt": """I am specifically focusing on the subject's action as inferred from depth. Please describe the camera holder's actions, interactions with objects, and movement intent using changes in distance and 3D layout. Emphasize what the subject is doing in the scene and the spatial consequence of those actions. Do not rely on appearance details."""
    },

    "depth_non_common": {
        "caption_prompt": """I am specifically focusing on unusual or physically implausible situations visible in depth. Please identify objects that appear floating, intersecting, clipped, unnaturally ordered in depth, or inconsistent with real-world geometry. Focus on 3D structural anomalies, unrealistic layouts, or depth artifacts. If nothing unusual is present, say so clearly."""
    }
}