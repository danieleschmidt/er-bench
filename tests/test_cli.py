"""Tests for the CLI."""

import json
import os
import subprocess
import sys
import tempfile
import pytest


PYTHON = sys.executable


class TestCLI:
    def _run_cli(self, args, check=True):
        """Run er-bench CLI and return CompletedProcess."""
        cmd = [PYTHON, "-m", "er_bench.cli"] + args
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        )

    def test_cli_runs_febrl_exact(self):
        """CLI: --dataset febrl --matcher exact exits 0."""
        result = self._run_cli(["run", "--dataset", "febrl", "--matcher", "exact"])
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
        assert "Precision" in result.stdout or "precision" in result.stdout.lower()

    def test_cli_runs_febrl_cascade(self):
        """CLI: --dataset febrl --matcher cascade exits 0."""
        result = self._run_cli(["run", "--dataset", "febrl", "--matcher", "cascade"])
        assert result.returncode == 0, f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"

    def test_cli_output_json(self):
        """CLI writes valid JSON output file."""
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            path = f.name
        os.unlink(path)
        try:
            result = self._run_cli([
                "run", "--dataset", "febrl", "--matcher", "exact",
                "--output", path,
            ])
            assert result.returncode == 0, result.stderr
            assert os.path.exists(path)
            with open(path) as f:
                data = json.load(f)
            assert data["dataset"] == "febrl"
            assert data["matcher"] == "exact"
        finally:
            if os.path.exists(path):
                os.unlink(path)

    def test_cli_list_datasets(self):
        """CLI list-datasets exits 0 and lists known datasets."""
        result = self._run_cli(["list-datasets"])
        assert result.returncode == 0
        assert "febrl" in result.stdout

    def test_cli_list_matchers(self):
        """CLI list-matchers exits 0 and lists known matchers."""
        result = self._run_cli(["list-matchers"])
        assert result.returncode == 0
        assert "cascade" in result.stdout

    def test_cli_unknown_dataset_exits_nonzero(self):
        """CLI with unknown dataset exits non-zero."""
        result = self._run_cli(["run", "--dataset", "nonexistent", "--matcher", "exact"])
        assert result.returncode != 0

    def test_cli_quiet_mode_json(self):
        """CLI --quiet outputs JSON with key metrics."""
        result = self._run_cli([
            "run", "--dataset", "febrl", "--matcher", "exact", "--quiet"
        ])
        assert result.returncode == 0, result.stderr
        data = json.loads(result.stdout.strip())
        assert "precision" in data
        assert "recall" in data
        assert "f1" in data
