"""Dataset loaders for er-bench."""

from er_bench.datasets.febrl import FebrlDataset
from er_bench.datasets.icij import ICIJDataset
from er_bench.datasets.dblp_acm import DBLPACMDataset

DATASETS = {
    "febrl": FebrlDataset,
    "icij": ICIJDataset,
    "dblp_acm": DBLPACMDataset,
}

__all__ = ["FebrlDataset", "ICIJDataset", "DBLPACMDataset", "DATASETS"]
