"""
Febrl synthetic dataset generator.

Generates records in-process with configurable duplicate rate and noise injection.
No download required — all data is generated programmatically.

Based on the Febrl (Freely Extensible Biomedical Record Linkage) approach,
adapted for general entity resolution benchmarking.
"""

from __future__ import annotations

import random
import string
import unicodedata
from dataclasses import dataclass, field
from typing import List, Tuple, Set, Optional


@dataclass
class FebrlRecord:
    record_id: str
    name: str
    entity_type: str
    country: str
    source_id: Optional[str] = None  # If set, this record is a duplicate of source_id


# Noise injection functions
def _transpose_chars(s: str, rng: random.Random) -> str:
    """Swap two adjacent characters."""
    if len(s) < 2:
        return s
    idx = rng.randint(0, len(s) - 2)
    lst = list(s)
    lst[idx], lst[idx + 1] = lst[idx + 1], lst[idx]
    return "".join(lst)


def _drop_char(s: str, rng: random.Random) -> str:
    """Drop a random character."""
    if len(s) < 2:
        return s
    idx = rng.randint(0, len(s) - 1)
    return s[:idx] + s[idx + 1:]


def _replace_char(s: str, rng: random.Random) -> str:
    """Replace a character with a nearby keyboard key."""
    keyboard_neighbors = {
        'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
        'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
        'k': 'jl', 'l': 'k', 'm': 'n', 'n': 'mb', 'o': 'ip',
        'p': 'o', 'q': 'w', 'r': 'et', 's': 'ad', 't': 'ry',
        'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
        'z': 'x',
    }
    if not s:
        return s
    idx = rng.randint(0, len(s) - 1)
    ch = s[idx].lower()
    neighbors = keyboard_neighbors.get(ch, string.ascii_lowercase)
    replacement = rng.choice(neighbors)
    # Preserve case
    if s[idx].isupper():
        replacement = replacement.upper()
    return s[:idx] + replacement + s[idx + 1:]


def _add_abbreviation(s: str, rng: random.Random) -> str:
    """Abbreviate a word by truncating it."""
    words = s.split()
    if not words:
        return s
    idx = rng.randint(0, len(words) - 1)
    word = words[idx]
    if len(word) > 3:
        words[idx] = word[:rng.randint(2, len(word) - 1)] + "."
    return " ".join(words)


NOISE_FUNCTIONS = [_transpose_chars, _drop_char, _replace_char, _add_abbreviation]

FIRST_NAMES = [
    "James", "Mary", "John", "Patricia", "Robert", "Jennifer", "Michael",
    "Linda", "William", "Barbara", "David", "Susan", "Richard", "Jessica",
    "Joseph", "Sarah", "Thomas", "Karen", "Charles", "Lisa", "Daniel",
    "Nancy", "Matthew", "Margaret", "Anthony", "Sandra", "Mark", "Ashley",
    "Donald", "Emily", "Steven", "Donna", "Paul", "Michelle", "Andrew",
    "Carol", "Kenneth", "Amanda", "George", "Melissa", "Joshua", "Deborah",
    "Kevin", "Stephanie", "Brian", "Dorothy", "Edward", "Rebecca", "Ronald",
    "Sharon", "Timothy", "Laura", "Jason", "Cynthia", "Jeffrey", "Amy",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
    "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
    "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin",
    "Lee", "Perez", "Thompson", "White", "Harris", "Sanchez", "Clark",
    "Ramirez", "Lewis", "Robinson", "Walker", "Young", "Allen", "King",
    "Wright", "Scott", "Torres", "Nguyen", "Hill", "Flores", "Green",
    "Adams", "Nelson", "Baker", "Hall", "Rivera", "Campbell", "Mitchell",
    "Carter", "Roberts", "Schmidt", "Mueller", "Fischer", "Weber", "Meyer",
]

COUNTRIES = [
    "United States", "United Kingdom", "Germany", "France", "Canada",
    "Australia", "Netherlands", "Switzerland", "Panama", "British Virgin Islands",
    "Cayman Islands", "Luxembourg", "Hong Kong", "Singapore", "Cyprus",
    "Malta", "Jersey", "Guernsey", "Isle of Man", "Liechtenstein",
]

ENTITY_TYPES = ["Individual", "Organization", "Company", "Trust", "Foundation"]


def _generate_name(rng: random.Random) -> str:
    first = rng.choice(FIRST_NAMES)
    last = rng.choice(LAST_NAMES)
    if rng.random() < 0.1:
        # Add middle initial
        middle = rng.choice(string.ascii_uppercase)
        return f"{first} {middle}. {last}"
    return f"{first} {last}"


def _inject_noise(name: str, rng: random.Random, num_errors: int = 1) -> str:
    """Apply random noise functions to a name."""
    result = name
    for _ in range(num_errors):
        noise_fn = rng.choice(NOISE_FUNCTIONS)
        # Apply noise to a random word
        words = result.split()
        if words:
            idx = rng.randint(0, len(words) - 1)
            words[idx] = noise_fn(words[idx], rng)
            result = " ".join(words)
    return result


class FebrlDataset:
    """
    Synthetic entity resolution dataset generator.

    Generates N records with a configurable duplicate rate. Duplicates are created
    by copying existing records and injecting realistic noise (typos, transpositions,
    abbreviations).

    Example:
        ds = FebrlDataset(n_records=1000, duplicate_rate=0.15, seed=42)
        records, ground_truth = ds.load()
    """

    name = "febrl"
    description = "Synthetic Febrl-style dataset with programmatic noise injection"

    def __init__(
        self,
        n_records: int = 1000,
        duplicate_rate: float = 0.15,
        seed: int = 42,
        noise_level: int = 1,
    ):
        """
        Args:
            n_records: Total number of records to generate (including duplicates).
            duplicate_rate: Fraction of records that are duplicates (0.0–1.0).
            seed: Random seed for reproducibility.
            noise_level: Number of noise operations applied per duplicate (1–3).
        """
        self.n_records = n_records
        self.duplicate_rate = duplicate_rate
        self.seed = seed
        self.noise_level = noise_level

    def load(self) -> Tuple[List[FebrlRecord], Set[Tuple[str, str]]]:
        """
        Generate the dataset.

        Returns:
            records: List of FebrlRecord objects.
            ground_truth: Set of (id1, id2) pairs where id1 < id2.
        """
        rng = random.Random(self.seed)

        n_duplicates = int(self.n_records * self.duplicate_rate)
        n_originals = self.n_records - n_duplicates

        records: List[FebrlRecord] = []
        ground_truth: Set[Tuple[str, str]] = set()

        # Generate original records
        for i in range(n_originals):
            record_id = f"R{i:05d}"
            name = _generate_name(rng)
            entity_type = rng.choice(ENTITY_TYPES)
            country = rng.choice(COUNTRIES)
            records.append(FebrlRecord(
                record_id=record_id,
                name=name,
                entity_type=entity_type,
                country=country,
            ))

        # Generate duplicates
        for j in range(n_duplicates):
            dup_id = f"D{j:05d}"
            source = rng.choice(records[:n_originals])
            noisy_name = _inject_noise(source.name, rng, self.noise_level)
            records.append(FebrlRecord(
                record_id=dup_id,
                name=noisy_name,
                entity_type=source.entity_type,
                country=source.country,
                source_id=source.record_id,
            ))
            # Ground truth: always store with smaller id first
            pair = tuple(sorted([source.record_id, dup_id]))
            ground_truth.add(pair)  # type: ignore[arg-type]

        # Shuffle so duplicates aren't all at the end
        rng.shuffle(records)

        return records, ground_truth

    def as_dicts(self) -> Tuple[List[dict], Set[Tuple[str, str]]]:
        """Return records as list of dicts for use with matchers."""
        records, gt = self.load()
        return [
            {
                "record_id": r.record_id,
                "name": r.name,
                "entity_type": r.entity_type,
                "country": r.country,
            }
            for r in records
        ], gt
