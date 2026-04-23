AUDIO_PROMPTS = {
    "sound_recognition": {
        "caption_prompt": """I am specifically focusing on audio event recognition, thus can caption my input audio paying attention to the sounds that are closely related to audible events, for example, voices, footsteps, doors, vehicles, machines, animals, appliances, and other environmental sounds. Please give a detailed caption that enumerate everything you can identify.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate a question that evaluates fine-grained understanding of sound events. Focus on what can be heard, such as sound sources, event types, materials, and audible actions. Avoid questions about visual appearance, illumination, or humans' faces. Please provide only 1 question, and make it precise, objective, and answerable with a few words.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the question, generate an answer that is consistent with the audio content and easy to evaluate. Keep the answer short, objective, and unambiguous. Avoid vague phrasing."""
    },

    "sound_counting": {
        "caption_prompt": """Now I am focusing on counting audible events. Please identify countable sound events in the audio and count how many times each important sound appears in the full sequence. Focus on useful daily-life sounds such as knocks, beeps, claps, alarms, footsteps, door sounds, and vehicle sounds.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of counting sound events. Avoid questions that can be answered from a short fragment; instead, design questions that require understanding the full audio. Please provide 8 questions. Each question should be precise and answerable with an exact number.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are exact numbers or short count phrases. Make sure the answers are consistent with the audio and easy to evaluate. Avoid vague responses."""
    },

    "sound_sequence": {
        "caption_prompt": """I am specifically focusing on the sequence of sounds in the audio. Please describe the order of important audible events, including what sound happened first, next, and last. Enumerate the sequence clearly.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of sound sequence. Avoid questions that can be answered from a few seconds; instead, require understanding of the full audio timeline. Ask about the order of events, transitions between sounds, and what sound came before or after another sound. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that clearly reflect the order of sounds in the audio. Keep the answers short, precise, and consistent with the timeline."""
    },

    "sound_spatial": {
        "caption_prompt": """I am specifically focusing on spatial audio cues. Please describe where sounds seem to come from, such as left, right, front, behind, near, far, approaching, or moving away. Also mention any clear stereo or directional sound patterns.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate a question that evaluates fine-grained understanding of spatial sound. Focus on direction, distance, movement of sound, or whether the sound source feels near or far. Avoid visual questions. Please provide only 1 question, and make it precise and answerable with a few words.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the question, generate an answer that is short, objective, and based on audible spatial cues only. Avoid vague or subjective wording."""
    },

    "speech_recognition": {
        "caption_prompt": """Now I am focusing on speech recognition. Please identify whether speech is present, how many speakers there are if possible, and any clearly audible spoken words or phrases. Also describe whether the speech is clear, overlapping, distant, or interrupted.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of speech in the audio. Ask about whether speech is present, how many speakers there are, whether the speech is overlapping, or what short words are clearly spoken. Avoid questions about faces or visual appearance. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are short and directly supported by the audio. Keep answers precise and avoid guessing unclear words."""
    },

    "music_recognition": {
        "caption_prompt": """I am specifically focusing on music recognition. Please describe any music in the audio, including whether it is present, its style or mood, the likely instruments, tempo, rhythm, and whether it is background or foreground music.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of music. Focus on music presence, instruments, tempo, rhythm, or whether it is background music. Avoid visual questions. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are short, objective, and consistent with the music heard in the audio. Avoid vague style descriptions unless they are clearly supported."""
    },

    "environmental_scene": {
        "caption_prompt": """Now I am focusing on environmental scene recognition from audio. Please describe the most likely place or environment suggested by the sounds, such as street, office, kitchen, classroom, station, mall, outdoors, or home. Mention the acoustic cues that support the scene.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of the environment suggested by sound. Focus on place, setting, and acoustic clues. Avoid visual questions. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are short, clear, and grounded in the audio evidence. Avoid overly broad or speculative place descriptions."""
    },

    "audio_change": {
        "caption_prompt": """I am specifically focusing on changes in audio over time. Please describe when sound starts, stops, becomes louder, quieter, more crowded, more isolated, or changes from one event to another. Enumerate the changes clearly.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of audio change over time. Ask about whether a sound started or stopped, whether volume increased or decreased, or whether the audio became more active or quiet. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are easy to evaluate and directly reflect changes in the audio timeline. Keep the answers short and unambiguous."""
    },

    "audio_visual_correspondence": {
        "caption_prompt": """I am specifically focusing on audio-visual correspondence. Please describe sounds that appear to match likely visible events or objects, such as a door slam, car engine, footsteps, pouring, typing, or alarm sounds. If a sound seems unrelated to the visual scene, mention that as well.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate whether the heard sound matches a likely event or object. Focus on correspondence, mismatch, and confirmation of audible actions. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are short, objective, and based on the audible evidence. Do not invent visual details that are not supported by the input."""
    },

    "action_from_sound": {
        "caption_prompt": """Now I am specifically focusing on actions that can be inferred from sound. Please describe audible actions such as opening, closing, moving, pouring, cutting, typing, dropping, knocking, walking, or turning on/off a device. Also mention any likely objects involved in those actions.""",

        "question_prompt": """Please work as an audio VQA assistant. By analyzing the audio and the corresponding caption, generate questions that evaluate fine-grained understanding of actions inferred from sound. Focus on what action happened, what object was involved, and what audible cue supports it. Please provide 8 questions.""",

        "answering_prompt": """Please work as an audio VQA assistant. Given the audio and the questions, generate answers that are short, exact, and directly supported by the audible action. Avoid speculation and keep the answers easy to evaluate."""
    }
}