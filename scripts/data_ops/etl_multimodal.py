import json
import random

OUTPUT = "assets/dpo_dataset_multimodal.jsonl"

VISUAL_CONTEXTS = [
    ("sci_fi_movie_poster.jpg", "a dark sci-fi movie poster"),
    ("error_trace.png", "a Python runtime error stack trace"),
    ("running_shoes.jpg", "a pair of red running shoes")
]

PROMPTS = [
    "Recommend an item for the user.",
    "Decide whether to use fast or slow reasoning."
]

def make_sample():
    img, desc = random.choice(VISUAL_CONTEXTS)
    prompt = random.choice(PROMPTS)

    base_prompt = f"""
User recently interacted with several items.
User uploaded an image described as: {desc}

Task: {prompt}
"""

    chosen = "Use slow reasoning and provide a detailed explanation."
    rejected = "Directly recommend an item without considering the image."

    return {
        "prompt": base_prompt.strip(),
        "chosen": chosen,
        "rejected": rejected
    }

with open(OUTPUT, "w") as f:
    for _ in range(100):
        f.write(json.dumps(make_sample()) + "\n")

print(f" Multimodal DPO dataset written to {OUTPUT}")