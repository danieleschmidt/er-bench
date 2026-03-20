"""
er-bench: Standardized Entity Resolution Benchmark Suite

Implements a 5-tier cascade entity fusion approach inspired by the DocGraph dissertation
(Schmidt, D., 2024). Supports Febrl (synthetic), DBLP-ACM, and ICIJ Offshore Leaks datasets.
"""

__version__ = "0.1.0"
__author__ = "Daniel Schmidt"

from er_bench.runner import BenchmarkRunner
from er_bench.metrics.report import BenchmarkReport

__all__ = ["BenchmarkRunner", "BenchmarkReport", "__version__"]
