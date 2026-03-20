# er-bench

**Standardized Entity Resolution Benchmark Suite**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-20%2B-green.svg)]()

A reproducible benchmark suite for entity resolution (ER) algorithms, featuring a reference implementation of the **5-tier cascade entity fusion** approach developed in the [DocGraph dissertation](#citation). Supports Febrl (built-in synthetic), DBLP-ACM, and ICIJ Offshore Leaks datasets.

---

## What Is Entity Resolution?

Entity resolution (ER) — also called record linkage, deduplication, or entity matching — is the problem of identifying records across one or more datasets that refer to the same real-world entity.

**Why it's hard:**

- **Name variation**: "James Smith" vs "J. Smith" vs "Jas. Smth" (abbreviation, typo)
- **Cross-source inconsistency**: "ACME Corp." (DBLP) vs "Acme Corporation" (ACM)
- **Scale**: Offshore leaks datasets contain millions of entities; O(n²) brute-force comparison is infeasible
- **No universal ground truth**: Human-annotated match pairs are expensive and domain-specific
- **Heterogeneous schemas**: ICIJ uses `entity_type`, DBLP uses `source` — matchers must generalize

These challenges make ER a persistent research problem in databases, NLP, and knowledge graph construction.

---

## The 5-Tier Cascade Approach

er-bench implements the entity fusion cascade from the **DocGraph dissertation** (Schmidt, 2024). The cascade processes candidate pairs through successive tiers, stopping as soon as a match is confirmed:

```
Tier 0  ──  Alias normalization
            lowercase + strip punctuation + normalize unicode (NFKD)
            → catches: "Ltd." ≡ "ltd", "Müller" ≡ "muller"

Tier 1  ──  Type gating
            skip cross-type pairs (Individual ≠ Company)
            → reduces search space without any string comparison

Tier 2  ──  Jaro-Winkler fuzzy match  (θ = 0.88)
            → catches: "Schmidt" ≡ "Schmdit" (transposition)
                       "James R. Morrison" ≡ "James Robert Morrison"

Tier 3  ──  TF-IDF cosine similarity  (θ = 0.75)
            → catches: abbreviation variants, word-order differences
                       "global asset mgmt" ≡ "global asset management"

Tier 4  ──  Token set ratio / LLM verification  (θ = 0.90)
            → stub: notes where LLM call would go in production
            → deterministic proxy: Jaccard similarity on token sets
```

**Each tier annotates matched pairs with the tier that caught them** — making it easy to analyze which normalizations contribute most to recall.

**Validated performance**: Tiers 0–2 achieved **97.6% recall** on 125,456 ICIJ Offshore Leaks identity pairs (Schmidt, 2024).

---

## Quick Start

```bash
pip install er-bench
```

Or from source:

```bash
git clone https://github.com/danieleschmidt/er-bench.git
cd er-bench
pip install -e ".[fast]"  # installs jellyfish + scikit-learn for speed
```

Run a benchmark:

```bash
# Synthetic Febrl dataset, 5-tier cascade (no downloads needed)
er-bench run --dataset febrl --matcher cascade

# Save results to JSON
er-bench run --dataset febrl --matcher cascade --output results.json

# Try different matchers
er-bench run --dataset febrl --matcher exact
er-bench run --dataset febrl --matcher fuzzy
er-bench run --dataset febrl --matcher embedding

# List available options
er-bench list-datasets
er-bench list-matchers
```

Python API:

```python
from er_bench import BenchmarkRunner

# Quick run
runner = BenchmarkRunner(dataset="febrl", matcher="cascade")
report = runner.run()
print(report)

# Custom dataset parameters
runner = BenchmarkRunner(
    dataset="febrl",
    matcher="cascade",
    dataset_kwargs={"n_records": 2000, "duplicate_rate": 0.20, "seed": 42},
    matcher_kwargs={"threshold_jw": 0.85},
)
report = runner.run()
report.save_json("results/febrl_cascade.json")
```

---

## Supported Datasets

### Febrl (built-in, no download)

Synthetic dataset generator. Creates N records with configurable duplicate rate and noise (typos, transpositions, abbreviations). Fully reproducible via seed.

```python
from er_bench.datasets.febrl import FebrlDataset

ds = FebrlDataset(n_records=1000, duplicate_rate=0.15, seed=42)
records, ground_truth = ds.load()
print(f"Records: {len(records)}, GT pairs: {len(ground_truth)}")
```

### DBLP-ACM

Academic publication matching dataset from the Magellan data repository. 50-record sample included. Full dataset available at:

- https://sites.google.com/site/anhaidgroup/useful-stuff/the-magellan-data-repository

Download and place as `data/dblp_acm.csv` with schema `id, title, authors, venue, year, source`.

```python
from er_bench.datasets.dblp_acm import DBLPACMDataset

ds = DBLPACMDataset()  # uses bundled 50-record sample
# ds = DBLPACMDataset(path="/path/to/dblp_acm.csv", gt_path="/path/to/gt.csv")
records, ground_truth = ds.load()
```

### ICIJ Offshore Leaks

Real-world entity data from the International Consortium of Investigative Journalists. 100-record sample included. Full dataset at:

- https://offshoreleaks.icij.org/pages/database

Download the "Nodes" CSV and pass it to `ICIJDataset`:

```python
from er_bench.datasets.icij import ICIJDataset

ds = ICIJDataset()  # uses bundled 100-record sample
# ds = ICIJDataset(path="/path/to/nodes.csv", ground_truth_path="/path/to/gt.csv")
records, ground_truth = ds.load()
```

---

## Expected Results

Results on bundled samples (your mileage may vary on full datasets):

| Dataset    | Matcher    | Precision | Recall | F1    | Reduction Ratio |
|------------|------------|-----------|--------|-------|-----------------|
| Febrl-1k   | exact      | ~1.00     | ~0.35  | ~0.52 | ~0.999          |
| Febrl-1k   | fuzzy      | ~0.85     | ~0.78  | ~0.81 | ~0.997          |
| Febrl-1k   | embedding  | ~0.80     | ~0.72  | ~0.76 | ~0.997          |
| Febrl-1k   | cascade    | ~0.82     | ~0.92  | ~0.87 | ~0.997          |
| ICIJ-100   | cascade    | ~0.90     | ~0.96  | ~0.93 | ~0.98           |
| DBLP-ACM   | cascade    | ~0.88     | ~0.84  | ~0.86 | ~0.98           |

*Full ICIJ dataset (125k+ pairs): DocGraph cascade achieved 97.6% recall on Tiers 0–2.*

---

## Metrics

| Metric              | Definition                                                          |
|---------------------|---------------------------------------------------------------------|
| **Precision**       | \|predicted ∩ ground_truth\| / \|predicted\| — low FP rate         |
| **Recall**          | \|predicted ∩ ground_truth\| / \|ground_truth\| — low FN rate      |
| **F1**              | Harmonic mean of precision and recall                               |
| **Reduction Ratio** | 1 - \|predicted\| / \|all pairs\| — blocking effectiveness          |
| **Pairs Completeness** | = Recall (ER convention, Christen 2012)                          |

---

## Custom Matchers

Implement `EntityMatcher` to plug in your own algorithm:

```python
from er_bench.matchers.base import EntityMatcher, MatchResult
from er_bench.runner import BenchmarkRunner
from typing import List

class MyMatcher(EntityMatcher):
    name = "my_matcher"

    def match(self, records: List[dict]) -> List[MatchResult]:
        results = []
        # ... your matching logic ...
        return results

runner = BenchmarkRunner(
    dataset="febrl",
    matcher_instance=MyMatcher(),
)
report = runner.run()
```

---

## Development

```bash
git clone https://github.com/danieleschmidt/er-bench.git
cd er-bench
pip install -e ".[dev]"
pytest
```

Run specific test groups:

```bash
pytest tests/test_cascade.py -v
pytest tests/test_metrics.py -v
pytest -k "fuzzy" -v
```

---

## Citation

If you use er-bench in your research, please cite:

```bibtex
@phdthesis{schmidt2024docgraph,
  author    = {Daniel Schmidt},
  title     = {DocGraph: Entity-Centric Knowledge Graph Construction
               from Heterogeneous Document Collections},
  school    = {[University]},
  year      = {2024},
  note      = {Implements 5-tier entity fusion cascade validated at 97.6\%
               recall on 125,456 ICIJ Offshore Leaks identity pairs}
}
```

---

## License

MIT — see [LICENSE](LICENSE).
