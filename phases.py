# phases.py
# Central mapping for handwashing phases/classes

# List of phase names in order of class index (adapted to your dataset structure)
PHASES = [
    "Start/Wet Hands",         # 0
    "Apply Soap",              # 1
    "Rub Palms",               # 2
    "Rub Back of Hands",       # 3
    "Interlace Fingers",       # 4
    "Clean Thumbs",            # 5
    "Rinse Hands"              # 6
]

# If your dataset only uses 0-3, trim the list to 4 phases:
# PHASES = [
#     "Start/Wet Hands",         # 0
#     "Apply Soap",              # 1
#     "Rub Palms",               # 2
#     "Rub Back of Hands"        # 3
# ]

# Utility: get phase name from class index
def get_phase_name(class_idx):
    if 0 <= class_idx < len(PHASES):
        return PHASES[class_idx]
    return f"Unknown Phase {class_idx}"
