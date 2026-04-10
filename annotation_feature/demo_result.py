DEMO_RESULT = {
    "object_recognition": {
        "caption": """1. Indoor Starting Area (The Room)
Scene: A domestic interior, likely a dorm or studio apartment.
Objects: Furniture includes a white desk with a black swivel chair, a small white dresser/nightstand, and a shoe rack. Footwear visible includes black sneakers, white-and-purple athletic shoes, and black slides. Household items include cleaning bottles and a blue storage container.
Color & Texture: White walls with matte texture. Blue-grey linoleum floor. Lighting is a mix of overhead light and warm desk lamp.

2. Transition & Hallway
Scene: Camera moves from room into a dark hallway.
Objects: A solid blue door leads out of the room. Metal door handles, light switches visible.
Color & Texture: High contrast between bright interior room and nearly black hallway. Blue door provides sharp pop of color.

3. Building Exterior & Mailbox Area
Scene: Outdoor walkway in modern apartment complex at night.
Objects: Large bank of mailboxes (metallic grey with vibrant green section). Metal staircases with vertical slat railings. Green-and-white emergency exit sign visible.
Color & Texture: Dominant dark greys and blacks. Green mailboxes most significant visual anchor.""",
        "question": """What is the color of the door I pass through to exit the room and enter the dark hallway?""",
        "answer": """Blue"""
    },
    "spatial_reasoning": {
        "caption": """Spatial Relationships Identified:
1. Desk is approximately 2 meters from the room entrance
2. The mailbox is positioned above eye level, roughly 1.5 meters from the approach area
3. The staircase is located to the left of the mailbox area, about 3 meters away
4. The fire exit sign is mounted on the wall above and to the right of the mailboxes
5. The hallway extends approximately 5 meters before reaching the outdoor area
6. The blue door frame is positioned centrally between the room and hallway""",
        "question": """Which room section is closest to the mailbox area when exiting through the blue door?""",
        "answer": """The hallway and staircase area"""
    },
    "text_recognition": {
        "caption": """Text elements visible in video:
1. Mailbox labels (Frame 15-20): "R. Festa", "L. Wiebusch", "S. Hollemann" - resident names on mailbox sections
2. Mailbox numbers (Frame 20-25): "1'233", "1'234", "1'235" - unit numbers, used for postal identification
3. Emergency sign (Frame 18-22): "EXIT" with running man symbol - indicates emergency evacuation route
4. Fire safety markings - visible on wall signage indicating building code compliance""",
        "question": """What are the visible mailbox numbers shown in the video?""",
        "answer": """1'233, 1'234, and 1'235"""
    },
    "scene_sequence": {
        "caption": """Scene Progression:
1. Start - Bedroom/Room interior with white walls, desk, and personal items
2. Room Transition - Moving toward the blue door with hallway visible beyond
3. Hallway Passage - Dark corridor with metal railings and light switches
4. Outdoor Area - Outdoor walkway with modern apartment building architecture
5. Mailbox Zone - Approach to mailbox area with green and grey mailbox sections
6. Staircase Area - Metal staircase visible with industrial design
7. End Position - Standing in front of mailbox section, looking at unit numbers""",
        "question": """What is the sequence of major areas passed through in this video?""",
        "answer": """Bedroom, hallway, outdoor walkway, and mailbox area"""
    },
    "light_recongnition": {
        "caption": """Light Source Analysis:
1. Room lighting: Combination of overhead ceiling light (dim) and warm desk lamp providing localized illumination
2. Transition zone: Dramatic shift to very low ambient light in the hallway
3. Outdoor area: Recessed linear LED ceiling lights creating bright white streaks in the darkness
4. Overall condition: Night-time recording with high ISO noise due to low-light conditions
5. Light reflection: Visible on glass windows and door surfaces, showing indoor light spilling into dark areas
6. Contrast level: Extreme contrast between lit interior room and dark outdoor areas""",
        "question": """What types of light sources are visible in the outdoor mailbox area?""",
        "answer": """Recessed linear LED ceiling lights creating bright white streaks"""
    },
    "light_change": {
        "caption": """Light Changes Throughout Video:
1. Beginning (0-5s): Warm, relatively bright interior lighting from desk and ceiling
2. Door transition (5-8s): Abrupt drop in illumination as entering hallway
3. Hallway progression (8-15s): Very dim with minimal ambient light, some reflections from door
4. Outdoor entry (15-18s): Sudden brightness increase from recessed ceiling LED lights
5. Mailbox area (18-30s): Consistent bright LED lighting with high contrast dark shadows
6. End scene (30-35s): Similar bright LED conditions with gloved hand interaction with mailbox""",
        "question": """Did the lighting change significantly when moving from the hallway to the outdoor mailbox area?""",
        "answer": """Yes, there was a dramatic increase in brightness from the very dim hallway to the bright LED-lit outdoor area"""
    },
    "counting": {
        "caption": """Objects Counted in Video:
1. Chairs: 1 (black swivel desk chair)
2. Shoes: 4+ pairs visible in room
3. Doors: 1 blue door
4. Mailbox sections: 2 main sections (grey and green)
5. Mailbox units: Approximately 9-12 individual mailbox compartments visible
6. Staircase sections: 2 visible staircase runs
7. Light fixtures: 1 desk lamp, 1 overhead ceiling light, recessed LEDs (multiple)
8. Storage containers: 2+ (blue bin, dresser)""",
        "question": """How many main mailbox sections are visible in the mailbox area?""",
        "answer": """2 sections - one grey and one vibrant green"""
    },
    "dynamic_counting": {
        "caption": """Dynamic Objects/Movement:
1. Camera motion: Handheld FPV movement from room through hallway to mailbox area
2. Hand movement (Frame 20-30): Blue-gloved hand approach and interaction with mailbox (opening/inserting key)
3. Door opening: Blue door transition showing movement from interior to exterior
4. Camera pan: Follows the action of the person navigating through the space
5. No other moving objects: No vehicles, people, or animals visible in the video besides the camera operator""",
        "question": """What dynamic action occurs with the mailbox in this video?""",
        "answer": """A blue-gloved hand approaches and interacts with the mailbox"""
    },
    "non_common": {
        "caption": """Potential Non-Common Scene Elements:
1. High-ISO grain: Significant video noise due to extremely low-light recording conditions
2. Unusual lighting: The LED ceiling lights create artificial patterns in an outdoor residential area
3. Color inconsistency: The vibrant green mailbox section contrasts sharply with standard grey sections
4. Hand appearance: Blue gloves in a residential mailbox area - possible protective gear or environmental adaptation
5. Layout observation: Modern residential building with outdoor mailbox cluster rather than interior boxes
6. No obvious floating objects or mesh clipping issues detected""",
        "question": """Are there any unusual design elements visible in the mailbox area?""",
        "answer": """Yes, one mailbox section is painted vibrant green which contrasts sharply with the standard metallic grey sections"""
    },
    "navigation": {
        "caption": """Navigation Path Documentation:
1. Starting point: Inside bedroom/room area
2. Action 1: Open blue door and move through doorway (1-2 steps)
3. Action 2: Navigate through dark hallway (3-4 meters, straight path)
4. Action 3: Exit to outdoor area (1-2 steps)
5. Action 4: Move to mailbox area (2-3 meters, slight left turn toward staircase)
6. Final position: Standing in front of mailbox section 1'233-1'235
7. Total navigation: Indoor room -> Hallway -> Outdoor walkway -> Mailbox cluster""",
        "question": """How would you navigate from the mailbox area back to the bedroom?""",
        "answer": """Retrace steps by moving back to the outdoor walkway, enter the hallway, and go through the blue door back to the bedroom"""
    },
    "dynamic_recognition": {
        "caption": """Dynamic Elements Recognition:
1. Primary moving object: The camera operator's hand (blue gloved) - appears around frame 20 and persists to frame 30
2. Hand trajectory: Approaches mailbox horizontally, extends toward mailbox slot area
3. Camera movement: Continuous forward and lateral movement throughout entire video (0-35 frames)
4. Camera perspective: First-person view showing hand interactions and path navigation
5. Hand-object interaction: Insert key/hand into mailbox area, appears to be accessing mailbox
6. No other dynamic elements: No people, vehicles, or animals visible besides camera operator""",
        "question": """What dynamic action involving the hand appears in the mailbox area?""",
        "answer": """A blue-gloved hand approaches and appears to insert something into the mailbox"""
    },
    "action": {
        "caption": """Camera Operator Actions:
1. Initial action: Gets up from desk area (implied by first-person view change)
2. Action 1: Opens blue door to exit room (Frame 5-8)
3. Action 2: Walks through dark hallway with steady deliberate steps (Frame 8-15)
4. Action 3: Exits to outdoor area and approaches mailbox cluster (Frame 15-25)
5. Action 4: Reaches toward mailbox with blue-gloved hand (Frame 20-30)
6. Action 5: Performs mailbox interaction - writing, inserting, or retrieving item (Frame 25-35)
7. Purpose: Retrieving mail or accessing mailbox - suggested by key interaction and deliberate approach
8. Consequence: Successfully completes mailbox task and remains at location""",
        "question": """What is the primary activity being performed with the mailbox?""",
        "answer": """Accessing the mailbox with a key, likely to retrieve or check mail"""
    }
}
