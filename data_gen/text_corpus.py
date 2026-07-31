"""
Text content sampler for synthetic crops: mix of real English words/phrases/sentences
(NLTK words + brown corpora) and random alphanumeric strings, per Phase 1 plan.
"""

import random
import string

import nltk
from nltk.corpus import words as nltk_words
from nltk.corpus import brown

REAL_WORD_MODE_PROB = 0.7  # ~70/30 real-word to random-string split, per plan


def _ensure_corpora():
    for pkg, path in [("words", "corpora/words"), ("brown", "corpora/brown")]:
        try:
            nltk.data.find(path)
        except LookupError:
            nltk.download(pkg)


_ensure_corpora()

_WORD_LIST = [w for w in nltk_words.words() if w.isalpha() and 2 <= len(w) <= 14]
_SENTENCES = brown.sents()


def random_word() -> str:
    word = random.choice(_WORD_LIST)
    return _apply_random_case(word)


def random_phrase(min_words=2, max_words=4) -> str:
    n = random.randint(min_words, max_words)
    phrase_words = [random.choice(_WORD_LIST) for _ in range(n)]
    return _apply_random_case(" ".join(phrase_words))


def random_sentence(max_words=10) -> str:
    sent = random.choice(_SENTENCES)
    sent = sent[:max_words]
    text = " ".join(sent)
    # brown corpus tokenizes punctuation separately (e.g. "word ."); tidy that up
    text = text.replace(" .", ".").replace(" ,", ",").replace(" ;", ";")
    return _apply_random_case(text)


def random_string(min_len=3, max_len=12) -> str:
    length = random.randint(min_len, max_len)
    charset = string.ascii_letters + string.digits
    return "".join(random.choice(charset) for _ in range(length))


def _apply_random_case(text: str) -> str:
    roll = random.random()
    if roll < 0.5:
        return text  # as-is (mixed natural case from corpus)
    elif roll < 0.7:
        return text.upper()
    elif roll < 0.85:
        return text.lower()
    else:
        return text.title()


def sample_text() -> str:
    """Top-level sampler: ~70% real-word content (word/phrase/sentence mix), ~30% random strings."""
    if random.random() < REAL_WORD_MODE_PROB:
        mode_roll = random.random()
        if mode_roll < 0.4:
            return random_word()
        elif mode_roll < 0.8:
            return random_phrase()
        else:
            return random_sentence()
    else:
        return random_string()


if __name__ == "__main__":
    print(f"word list size: {len(_WORD_LIST)}")
    print(f"sentence count: {len(_SENTENCES)}")
    print("\nsample outputs:")
    for _ in range(20):
        print(repr(sample_text()))
