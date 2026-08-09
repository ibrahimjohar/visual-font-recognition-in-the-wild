"""
Selects the deliberately adversarial class subset for the Check B smoke test -- not a random
sample, per the council's finding that a random/tiny subset would trivially beat chance
regardless of whether flat softmax actually works at 3,407-way scale.

Two subsets, both required to pass their threshold (see SMOKE_TEST_THRESHOLDS below):
  - CONFUSABLE_SUBSET (~15-20 classes): intra-family role clusters (a family's Regular/Bold/
    Italic/BoldItalic together) plus known visually-similar geometric-sans families across
    different sources -- stresses fine-grained separability.
  - RANDOM_500_SUBSET: a random sample of 500 classes -- stresses whether softmax scales to
    a class count in the same order of magnitude as the full 3,407, which a 20-class subset
    can't tell you regardless of its outcome.
"""

import csv
import random
from pathlib import Path

# curated cross-family geometric-sans lookalikes actually present in the Google Fonts corpus --
# these are the pairs a human (and existing font-ID tools) genuinely confuse.
SIMILAR_SANS_FAMILIES = [
    "roboto", "opensans", "worksans", "inter", "publicsans", "ibmplexsans",
    "manrope", "karla", "dmsans", "mulish", "sourcesans3", "notosans",
]


def build_confusable_subset(manifest_csv: str, target_size: int = 18, seed: int = 42) -> set[str]:
    """Picks 1-2 intra-family role clusters + as many of SIMILAR_SANS_FAMILIES' available
    classes as fit, up to target_size."""
    family_roles: dict[str, set[str]] = {}
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            family_roles.setdefault(row["family"], set()).add(row["class_name"])

    subset: set[str] = set()

    # inter-family: known lookalikes, all roles available for each
    for family in SIMILAR_SANS_FAMILIES:
        if family in family_roles:
            subset.update(family_roles[family])
        if len(subset) >= target_size:
            break

    # intra-family: top up with one full 4-role family cluster for within-family confusability,
    # preferring a family not already included
    if len(subset) < target_size:
        rng = random.Random(seed)
        candidates = [fam for fam, roles in family_roles.items()
                      if len(roles) == 4 and fam not in SIMILAR_SANS_FAMILIES]
        rng.shuffle(candidates)
        for fam in candidates:
            subset.update(family_roles[fam])
            if len(subset) >= target_size:
                break

    return subset


def build_random_subset(manifest_csv: str, size: int = 500, seed: int = 42) -> set[str]:
    classes = set()
    with open(manifest_csv, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            classes.add(row["class_name"])
    rng = random.Random(seed)
    return set(rng.sample(sorted(classes), min(size, len(classes))))


# --- pre-committed Check B thresholds, written down before any run so the pass/fail call
# can't be rationalized after seeing numbers (per the council's explicit finding that an
# undefined threshold makes the fix "theater," not a real gate). ---
#
# CONFUSABLE_SUBSET: ~18 classes, chance level ~5.6%. 50% top-5 accuracy is ~9x chance and
# means the model is meaningfully separating classes that are specifically chosen to be hard,
# not just clearing an easy bar.
# RANDOM_500_SUBSET: 500 classes, chance level ~1%(top-1)/~1%(top-5 chance ~1%). 25% top-5 is
# 25x chance and is the number that actually stresses whether softmax scales anywhere near
# the full 3,407-class regime -- this is the subset a tiny 18-class test cannot speak to.
SMOKE_TEST_THRESHOLDS = {
    "confusable_top5_min": 0.50,
    "random500_top5_min": 0.25,
    "convergence_patience_epochs": 3,   # "converged" = val top-5 improves <1% for this many epochs
    "convergence_min_delta": 0.01,
    "max_epochs_cap": 20,               # hard ceiling regardless of convergence, so Check B can't run away
}
