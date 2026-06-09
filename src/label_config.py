LABELS = [
    "coding",
    "writing",
    "research",
    "math",
    "data_analysis",
    "creative",
    "planning",
    "summarization",
    "editing",
    "brainstorming",
    "qa",
    "other",
]

LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for label, i in LABEL2ID.items()}

# Maps source dataset labels into your app's domains.
DOLLY_CATEGORY_MAP = {
    "brainstorming": "brainstorming",
    "classification": "qa",
    "closed_qa": "qa",
    "creative_writing": "creative",
    "generation": "writing",
    "information_extraction": "research",
    "open_qa": "qa",
    "summarization": "summarization",
}
