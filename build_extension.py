"""Generate varied grammar and contextual contrast passages, not eval items.

No evaluation files, answer keys, model outputs, or chat logs are read here.
Run from the repository root. The notebook performs the separate leakage check.
"""
from pathlib import Path


def teaching_passages():
    sentences = []
    animals = ["bird", "dog", "cat", "rabbit", "horse", "duck", "goat", "fox"]
    states = ["awake", "hungry", "quiet", "nearby", "outside", "small"]
    for animal in animals:
        plural = "foxes" if animal == "fox" else animal + "s"
        for state in states:
            sentences.extend([
                f"A {animal} is {state} this morning.",
                f"Several {plural} are {state} this morning.",
                f"A {animal} was {state} last evening.",
                f"Several {plural} were {state} last evening.",
            ])
    for state in ["ready", "early", "late", "happy", "tired", "quiet", "warm", "cold"]:
        sentences.extend([
            f"I am {state} today.", f"She is {state} today.",
            f"We are {state} today.", f"They were {state} last night.",
            f"She was {state} last night.", f"He is {state} today.",
        ])
    destinations = ["garden", "park", "library", "bridge", "village", "beach", "school", "store"]
    for destination in destinations:
        for time in ["each morning", "every afternoon", "on weekends"]:
            sentences.extend([
                f"She walks to the {destination} {time}.",
                f"They walk to the {destination} {time}.",
                f"I walk to the {destination} {time}.",
            ])
        for time in ["last Monday", "last week", "yesterday", "last summer"]:
            sentences.extend([
                f"She walked to the {destination} {time}.",
                f"They walked to the {destination} {time}.",
                f"We were walking to the {destination} {time}.",
            ])
        sentences.extend([
            f"She is walking to the {destination} now.",
            f"They are walking to the {destination} now.",
        ])
    # Broad contextual contrasts rather than the exam's 'opposite of ...' form.
    contrasts = [
        ("hot", "cold", "drink"), ("warm", "cool", "room"),
        ("empty", "full", "basket"), ("noisy", "quiet", "street"),
        ("loud", "soft", "sound"), ("heavy", "light", "bag"),
        ("fast", "slow", "bicycle"), ("early", "late", "arrival"),
        ("soft", "hard", "cushion"), ("round", "square", "table"),
        ("wide", "narrow", "path"), ("wet", "dry", "towel"),
    ]
    locations = ["garden", "park", "library", "office", "station", "school", "store", "market"]
    for first, second, noun in contrasts:
        for location in locations:
            sentences.extend([
                f"At the {location}, one {noun} is {first} while another is {second}.",
                f"We compared a {first} {noun} with a {second} {noun} at the {location}.",
                f"The {first} {noun} differs from the {second} {noun} at the {location}.",
                f"At the {location}, the {noun} changed from {first} to {second}.",
                f"At the {location}, the {noun} changed from {second} to {first}.",
            ])
    return sentences


if __name__ == "__main__":
    passages = teaching_passages()
    destination = Path("corpus/extension/grammar_and_contrasts.txt")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(passages) + "\n", encoding="utf-8")
    print(f"Saved {len(passages)} passages ({len(set(passages))} unique) to {destination}")
