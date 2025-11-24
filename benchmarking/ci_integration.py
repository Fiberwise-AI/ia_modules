"""
CI/CD Integration for Benchmarking

Provides tools for integrating benchmarks into CI/CD pipelines.
"""

import sys
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from .comparison import BenchmarkComparator, ComparisonMetric, PerformanceChange, ComparisonResult
from .models import BenchmarkResult


@dataclass
class CIConfig:
    """Configuration for CI/CD integration"""
    baseline_file: Path
    current_file: Path
    regression_threshold: float = 10.0  # % change to fail CI
    output_format: str = "markdown"  # markdown, json, or github-actions
    fail_on_regression: bool = True
    metrics_to_compare: List[ComparisonMetric] = None

    def __post_init__(self):
        if self.metrics_to_compare is None:
            self.metrics_to_compare = [
                ComparisonMetric.MEAN_TIME,
                ComparisonMetric.P95_TIME
            ]


class CIIntegration:
    """
    CI/CD integration for benchmarks

    Features:
    - Compare against baseline
    - Fail build on regression
    - Generate CI-friendly reports
    - Support for GitHub Actions, GitLab CI, etc.
    """

    def __init__(self, config: CIConfig):
        self.config = config
        self.comparator = BenchmarkComparator(
            regression_threshold=config.regression_threshold
        )

    def run(self) -> int:
        """
        Run CI benchmark comparison

        Returns:
            Exit code (0 = success, 1 = regression detected)
        """
        # Load results
        baseline_results = self._load_results(self.config.baseline_file)
        current_results = self._load_results(self.config.current_file)

        if not baseline_results or not current_results:
            print("Error: Could not load benchmark results", file=sys.stderr)
            return 1

        # Compare results by name
        all_comparisons = []
        has_regression = False

        for baseline in baseline_results:
            # Find matching current result
            current = next(
                (r for r in current_results if r['name'] == baseline['name']),
                None
            )

            if not current:
                print(f"Warning: No current result found for {baseline['name']}", file=sys.stderr)
                continue

            # Convert dicts to BenchmarkResult-like objects
            baseline_obj = self._dict_to_result(baseline)
            current_obj = self._dict_to_result(current)

            # Compare
            comparisons = self.comparator.compare(
                baseline_obj,
                current_obj,
                metrics=self.config.metrics_to_compare
            )

            all_comparisons.extend(comparisons)

            # Check for regression
            if self.comparator.has_regression(comparisons):
                has_regression = True

        # Generate report
        self._generate_report(all_comparisons, has_regression)

        # Return exit code
        if has_regression and self.config.fail_on_regression:
            return 1

        return 0

    def _load_results(self, filepath: Path) -> Optional[List[Dict[str, Any]]]:
        """Load benchmark results from JSON file"""
        if not filepath.exists():
            print(f"Error: File not found: {filepath}", file=sys.stderr)
            return None

        try:
            data = json.loads(filepath.read_text())
            if 'benchmarks' in data:
                return data['benchmarks']
            return [data]  # Single result
        except json.JSONDecodeError as e:
            print(f"Error: Invalid JSON in {filepath}: {e}", file=sys.stderr)
            return None

    def _dict_to_result(self, data: Dict[str, Any]) -> BenchmarkResult:
        """Convert dictionary to BenchmarkResult-like object"""
        from .framework import BenchmarkResult

        return BenchmarkResult(
            name=data.get('name', 'unknown'),
            iterations=data.get('iterations', 0),
            mean_time=data.get('mean_time', 0),
            median_time=data.get('median_time', 0),
            std_dev=data.get('std_dev', 0),
            min_time=data.get('min_time', 0),
            max_time=data.get('max_time', 0),
            p95_time=data.get('p95_time', 0),
            p99_time=data.get('p99_time', 0),
            total_time=data.get('total_time', 0),
            memory_stats=data.get('memory_stats'),
            cpu_stats=data.get('cpu_stats')
        )

    def _generate_report(
        self,
        comparisons: List[ComparisonResult],
        has_regression: bool
    ) -> None:
        """Generate CI report in configured format"""
        if self.config.output_format == "markdown":
            self._generate_markdown_report(comparisons, has_regression)
        elif self.config.output_format == "json":
            self._generate_json_report(comparisons, has_regression)
        elif self.config.output_format == "github-actions":
            self._generate_github_actions_report(comparisons, has_regression)
        else:
            print(f"Unknown output format: {self.config.output_format}", file=sys.stderr)

    def _generate_markdown_report(
        self,
        comparisons: List[ComparisonResult],
        has_regression: bool
    ) -> None:
        """Generate markdown report"""
        print("# Benchmark Comparison Report")
        print()

        if has_regression:
            print("## ⚠️ Performance Regression Detected")
            print()

        print("| Metric | Baseline | Current | Change | Status |")
        print("|--------|----------|---------|--------|--------|")

        for comp in comparisons:
            status_symbol = {
                PerformanceChange.IMPROVED: "✅",
                PerformanceChange.REGRESSED: "❌",
                PerformanceChange.UNCHANGED: "➡️"
            }[comp.change_classification]

            print(
                f"| {comp.metric.value} | "
                f"{comp.baseline_value:.4f} | "
                f"{comp.current_value:.4f} | "
                f"{comp.percent_change:+.2f}% | "
                f"{status_symbol} |"
            )

    def _generate_json_report(
        self,
        comparisons: List[ComparisonResult],
        has_regression: bool
    ) -> None:
        """Generate JSON report"""
        report = {
            'has_regression': has_regression,
            'comparisons': [
                {
                    'metric': c.metric.value,
                    'baseline_value': c.baseline_value,
                    'current_value': c.current_value,
                    'delta': c.delta,
                    'percent_change': c.percent_change,
                    'classification': c.change_classification.value,
                    'is_significant': c.is_significant
                }
                for c in comparisons
            ]
        }
        print(json.dumps(report, indent=2))

    def _generate_github_actions_report(
        self,
        comparisons: List[ComparisonResult],
        has_regression: bool
    ) -> None:
        """Generate GitHub Actions-compatible report"""
        # Set output
        print(f"::set-output name=has_regression::{str(has_regression).lower()}")

        # Add annotations
        for comp in comparisons:
            if comp.change_classification == PerformanceChange.REGRESSED:
                print(
                    f"::warning::{comp.metric.value} regressed by {comp.percent_change:.2f}%"
                )
            elif comp.change_classification == PerformanceChange.IMPROVED:
                print(
                    f"::notice::{comp.metric.value} improved by {abs(comp.percent_change):.2f}%"
                )

        # Summary
        if has_regression:
            print("::error::Performance regression detected!")


def github_actions_workflow() -> str:
    """
    Generate example GitHub Actions workflow

    Returns:
        YAML workflow content
    """
    return '''name: Benchmark Performance

on:
  pull_request:
    branches: [main]
  push:
    branches: [main]

jobs:
  benchmark:
    runs-on: ubuntu-latest

    steps:
      - uses: actions/checkout@v3

      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'

      - name: Install dependencies
        run: |
          pip install -e ia_modules
          pip install psutil  # For profiling

      - name: Run benchmarks
        run: |
          python -m ia_modules.benchmarking.cli benchmark \
            --output current_results.json

      - name: Download baseline
        run: |
          # Download baseline from artifacts or storage
          wget https://example.com/baseline_results.json || echo "{}" > baseline_results.json

      - name: Compare results
        id: compare
        run: |
          python -m ia_modules.benchmarking.cli compare \
            --baseline baseline_results.json \
            --current current_results.json \
            --format github-actions \
            --fail-on-regression

      - name: Upload results
        uses: actions/upload-artifact@v3
        with:
          name: benchmark-results
          path: current_results.json

      - name: Comment PR
        if: github.event_name == 'pull_request'
        uses: actions/github-script@v6
        with:
          script: |
            const fs = require('fs');
            const report = fs.readFileSync('benchmark_report.md', 'utf8');
            github.rest.issues.createComment({
              issue_number: context.issue.number,
              owner: context.repo.owner,
              repo: context.repo.repo,
              body: report
            });
'''


def gitlab_ci_config() -> str:
    """
    Generate example GitLab CI configuration

    Returns:
        YAML config content
    """
    return '''benchmark:
  stage: test
  image: python:3.9
  script:
    - pip install -e ia_modules
    - pip install psutil
    - python -m ia_modules.benchmarking.cli benchmark --output current_results.json
    - |
      if [ -f baseline_results.json ]; then
        python -m ia_modules.benchmarking.cli compare \
          --baseline baseline_results.json \
          --current current_results.json \
          --format markdown > benchmark_report.md
        cat benchmark_report.md
      fi
  artifacts:
    paths:
      - current_results.json
      - benchmark_report.md
    expire_in: 30 days
  only:
    - merge_requests
    - main
'''
