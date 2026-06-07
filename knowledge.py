"""
Nova's kid-world knowledge base.

This is what Nova KNOWS about 4-8 year olds in general — things she can
reference to make conversation feel rich and informed, not generic.

Use pattern:
    from knowledge import lookup, fact_for
    color_fact = fact_for("color", "yellow")  # → "yellow! sunny color, like dandelions"
    animals = lookup("animals")

NOT every fact is injected into every prompt. The router in agent.py
pulls relevant facts only when the kid mentions a topic.
"""

from typing import Optional, Dict, List


# ────────────────────────────────────────────────────────────────
# COLORS — kids learn these first, get strong feelings about them
# ────────────────────────────────────────────────────────────────
COLORS = {
    "red": "red — like fire trucks and apples",
    "orange": "orange — like pumpkins and sunsets",
    "yellow": "yellow — sunny color, like dandelions",
    "green": "green — like grass and frogs",
    "blue": "blue — like sky and oceans",
    "purple": "purple — like grapes and flowers",
    "pink": "pink — like cotton candy and flamingos",
    "white": "white — like snow and clouds",
    "black": "black — like the night sky with stars",
    "brown": "brown — like chocolate and tree trunks",
    "rainbow": "rainbow — every color at once, after rain!",
}

# ────────────────────────────────────────────────────────────────
# ANIMALS — many kids have favorites + fears
# ────────────────────────────────────────────────────────────────
ANIMALS_FRIENDLY = {
    "dog": "dogs — they wag their tails when happy",
    "cat": "cats — they purr when you pet them",
    "rabbit": "rabbits — they hop like little jumpers",
    "horse": "horses — they run fast in big fields",
    "bird": "birds — they sing in the morning",
    "fish": "fish — they swim and never get wet (well, always wet)",
    "elephant": "elephants — biggest animals, but they're gentle",
    "lion": "lions — they roar like THIS: ROAR",
    "panda": "pandas — they eat bamboo all day",
    "dolphin": "dolphins — they smile when they swim",
    "unicorn": "unicorns — magical horses with horns",
    "dragon": "dragons — they breathe fire, but the friendly ones don't",
}
ANIMALS_SCARY = ["spider", "snake", "shark", "wolf", "monster", "dinosaur"]

# ────────────────────────────────────────────────────────────────
# FOODS — kids have BIG opinions
# ────────────────────────────────────────────────────────────────
FOODS_LIKED = {
    "pizza": "pizza — best food in the whole world",
    "ice cream": "ice cream — but only the kind you like",
    "chocolate": "chocolate — magic brown squares",
    "pasta": "pasta — long stringy noodles, slurp!",
    "strawberries": "strawberries — tiny red bursts",
    "banana": "bananas — yellow and curvy, monkeys love them",
    "apple": "apples — crunchy red or green",
    "candy": "candy — but only for special times",
    "cookies": "cookies — round and warm",
    "watermelon": "watermelon — juicy summer fruit, big green ball",
}
FOODS_DISLIKED = ["broccoli", "spinach", "mushrooms", "fish", "salad"]

# ────────────────────────────────────────────────────────────────
# SHOWS — Nova can reference but not be derivative
# (She knows them but doesn't pretend to be them)
# ────────────────────────────────────────────────────────────────
KIDS_SHOWS = {
    "bluey": "Bluey — the blue heeler puppy with the best dad",
    "paw patrol": "Paw Patrol — the puppies who help everybody",
    "peppa pig": "Peppa Pig — pink piggy with her brother George",
    "cocomelon": "Cocomelon — songs for the littlest kids",
    "frozen": "Frozen — Elsa with the icy magic",
    "encanto": "Encanto — the magic family in the colorful house",
    "ms rachel": "Ms Rachel — she sings to teach words",
    "moana": "Moana — brave ocean girl",
    "spider-man": "Spider-Man — webs and rooftops",
}

# ────────────────────────────────────────────────────────────────
# BODY PARTS — for dance instruction
# ────────────────────────────────────────────────────────────────
BODY_PARTS = ["head", "shoulders", "knees", "toes", "hands", "feet", "elbows",
              "eyes", "ears", "nose", "mouth", "hips", "tummy", "back", "neck"]

# ────────────────────────────────────────────────────────────────
# NUMBERS + DAYS — for tracking, counting, anchoring
# ────────────────────────────────────────────────────────────────
NUMBERS_1_10 = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten"]
DAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

# ────────────────────────────────────────────────────────────────
# COMMON KID WORRIES — Nova should respond with empathy, not dismissal
# ────────────────────────────────────────────────────────────────
KID_FEARS = {
    "monster": "monsters — they're tricky but they're not real, only in stories",
    "dark": "the dark — it's just the world resting until morning",
    "alone": "being alone — that's a real feeling, hugs help",
    "loud": "loud noises — they surprise us, but they pass",
    "tired": "being tired — your body's saying it needs rest",
    "sad": "feeling sad — feelings are like weather, they move through",
    "scared": "being scared — even brave people get scared sometimes",
}

# ────────────────────────────────────────────────────────────────
# DANCE MOVES — Nova's vocabulary during dance
# ────────────────────────────────────────────────────────────────
DANCE_MOVES = {
    "wave": "wave — like saying hi with your whole arm",
    "spin": "spin — round and round, like a top",
    "clap": "clap — like a drumbeat with your hands",
    "freeze": "freeze — stop! statue mode!",
    "jump": "jump — feet off the ground, up!",
    "twist": "twist — wiggle side to side",
    "stomp": "stomp — feet say BAM",
    "sway": "sway — lean left, lean right, lean left, lean right",
}


# ────────────────────────────────────────────────────────────────
# QUICK LOOKUP HELPERS
# ────────────────────────────────────────────────────────────────
TOPIC_TABLES = {
    "color": COLORS,
    "colors": COLORS,
    "animal": ANIMALS_FRIENDLY,
    "animals": ANIMALS_FRIENDLY,
    "food": FOODS_LIKED,
    "foods": FOODS_LIKED,
    "show": KIDS_SHOWS,
    "shows": KIDS_SHOWS,
    "fear": KID_FEARS,
    "fears": KID_FEARS,
    "move": DANCE_MOVES,
    "moves": DANCE_MOVES,
}


def fact_for(topic: str, item: str) -> Optional[str]:
    """Get the warm one-line fact for a specific item in a topic.

    Example:
        fact_for("color", "yellow") → "yellow — sunny color, like dandelions"
    """
    table = TOPIC_TABLES.get(topic.lower())
    if not table:
        return None
    return table.get(item.lower())


def lookup(topic: str) -> Dict[str, str]:
    """Get the whole table for a topic."""
    return TOPIC_TABLES.get(topic.lower(), {})


def detect_topics(text: str) -> List[str]:
    """Scan kid's text for topic mentions Nova should know about.

    Returns a list of (topic, item, fact) tuples for things mentioned.
    Used by the prompt builder to inject relevant knowledge.
    """
    if not text:
        return []
    t = text.lower()
    hits = []
    # Direct word matches across all tables
    for topic, table in TOPIC_TABLES.items():
        for item, fact in table.items():
            if item in t:
                hits.append((topic, item, fact))
    # Body parts
    for part in BODY_PARTS:
        if part in t:
            hits.append(("body", part, part))
    # De-dup, keep first 3 max
    seen = set()
    result = []
    for h in hits:
        if h[1] in seen: continue
        seen.add(h[1])
        result.append(h)
        if len(result) >= 3: break
    return result


def knowledge_snippet(text: str) -> Optional[str]:
    """Build a compact prompt-injection from what the kid just mentioned.

    Returns a short string like:
        "Kid mentioned: yellow (sunny color, like dandelions); dog (wag tails)"
    Or None if nothing relevant.
    """
    hits = detect_topics(text)
    if not hits:
        return None
    parts = [f"{item} ({fact})" for (topic, item, fact) in hits]
    return "Kid just mentioned: " + "; ".join(parts) + "."
