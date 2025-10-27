"""
Comprehensive tests for benchmarking/reporters.py
"""
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import json
import tempfile
from pathlib import Path
from io import StringIO
import sys
from datetime import datetime, timezone

from ia_modules.benchmarking.framework import BenchmarkResult
from ia_modules.benchmarking.reporters import (
    ConsoleReporter,
    JSONReporter,
    HTMLReporter,
    MarkdownReporter
)


class TestConsoleReporter:
    """Test ConsoleReporter class"""

    @pytest.fixture
    def sample_results(self):
        """Create sample benchmark results"""
        return [
            BenchmarkResult(
                name="test_benchmark_1",
                iterations=100,
                mean_time=0.1,
                median_time=0.09,
                std_dev=0.01,
                min_time=0.08,
                max_time=0.15,
                p95_time=0.12,
                p99_time=0.14,
                total_time=10.0
            ),
            BenchmarkResult(
                name="test_benchmark_2",
                iterations=50,
                mean_time=0.05,
                median_time=0.045,
                std_dev=0.005,
                min_time=0.04,
                max_time=0.075,
                p95_time=0.06,
                p99_time=0.07,
                total_time=2.5
            )
        ]

    @pytest.fixture
    def sample_result_with_stats(self):
        """Create sample result with memory and CPU stats"""
        return [
            BenchmarkResult(
                name="test_with_stats",
                iterations=100,
                mean_time=0.1,
                median_time=0.09,
                std_dev=0.01,
                min_time=0.08,
                max_time=0.15,
                p95_time=0.12,
                p99_time=0.14,
                total_time=10.0,
                memory_stats={
                    'delta_mb': 50.5,
                    'peak_mb': 150.2
                },
                cpu_stats={
                    'average_cpu_percent': 75.3,
                    'total_cpu_time': 8.5
                }
            )
        ]

    def test_init_default(self):
        """Test ConsoleReporter initialization with defaults"""
        reporter = ConsoleReporter()
        assert reporter.use_colors is True

    def test_init_no_colors(self):
        """Test ConsoleReporter initialization without colors"""
        reporter = ConsoleReporter(use_colors=False)
        assert reporter.use_colors is False

    def test_report_empty_results(self):
        """Test reporting empty results"""
        reporter = ConsoleReporter()

        captured_output = StringIO()
        sys.stdout = captured_output

        reporter.report([])

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "No benchmark results to report" in output

    def test_report_single_result(self):
        """Test reporting a single result"""
        reporter = ConsoleReporter()
        result = BenchmarkResult(
            name="test",
            iterations=100,
            mean_time=0.1,
            median_time=0.09,
            std_dev=0.01,
            min_time=0.08,
            max_time=0.15,
            p95_time=0.12,
            p99_time=0.14,
            total_time=10.0
        )

        captured_output = StringIO()
        sys.stdout = captured_output

        reporter.report([result])

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "Benchmark Results" in output
        assert "test" in output
        assert "Iterations: 100" in output
        assert "Mean:" in output
        assert "100.00" in output  # 0.1s = 100ms

    def test_report_multiple_results(self, sample_results):
        """Test reporting multiple results"""
        reporter = ConsoleReporter()

        captured_output = StringIO()
        sys.stdout = captured_output

        reporter.report(sample_results)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "test_benchmark_1" in output
        assert "test_benchmark_2" in output
        assert output.count("Iterations:") == 2

    def test_report_with_memory_stats(self, sample_result_with_stats):
        """Test reporting with memory statistics"""
        reporter = ConsoleReporter()

        captured_output = StringIO()
        sys.stdout = captured_output

        reporter.report(sample_result_with_stats)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "Memory (MB):" in output
        assert "Delta:" in output
        assert "50.50" in output
        assert "Peak:" in output
        assert "150.20" in output

    def test_report_with_cpu_stats(self, sample_result_with_stats):
        """Test reporting with CPU statistics"""
        reporter = ConsoleReporter()

        captured_output = StringIO()
        sys.stdout = captured_output

        reporter.report(sample_result_with_stats)

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "CPU:" in output
        assert "Avg %" in output  # Match "Avg %:" or "Avg %%:"
        assert "75.30" in output
        assert "Time:" in output
        assert "8.50" in output


class TestJSONReporter:
    """Test JSONReporter class"""

    @pytest.fixture
    def sample_results(self):
        """Create sample benchmark results"""
        return [
            BenchmarkResult(
                name="test_benchmark",
                iterations=100,
                mean_time=0.1,
                median_time=0.09,
                std_dev=0.01,
                min_time=0.08,
                max_time=0.15,
                p95_time=0.12,
                p99_time=0.14,
                total_time=10.0
            )
        ]

    def test_init_default(self):
        """Test JSONReporter initialization with defaults"""
        reporter = JSONReporter()
        assert reporter.pretty is True

    def test_init_not_pretty(self):
        """Test JSONReporter initialization without pretty printing"""
        reporter = JSONReporter(pretty=False)
        assert reporter.pretty is False

    def test_report_returns_json_string(self, sample_results):
        """Test that report returns valid JSON string"""
        reporter = JSONReporter()
        json_str = reporter.report(sample_results)

        assert isinstance(json_str, str)

        # Parse to verify it's valid JSON
        data = json.loads(json_str)
        assert 'timestamp' in data
        assert 'total_benchmarks' in data
        assert 'benchmarks' in data
        assert data['total_benchmarks'] == 1

    def test_report_pretty_formatted(self, sample_results):
        """Test pretty formatted JSON output"""
        reporter = JSONReporter(pretty=True)
        json_str = reporter.report(sample_results)

        # Pretty JSON should have newlines and indentation
        assert '\n' in json_str
        assert '  ' in json_str

    def test_report_compact_format(self, sample_results):
        """Test compact JSON format"""
        reporter = JSONReporter(pretty=False)
        json_str = reporter.report(sample_results)

        # Compact should have no indentation
        assert '  ' not in json_str

    def test_report_writes_to_file(self, sample_results):
        """Test writing JSON report to file"""
        reporter = JSONReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.json"

            json_str = reporter.report(sample_results, output_file=output_file)

            # Verify file was created
            assert output_file.exists()

            # Verify content matches
            file_content = output_file.read_text()
            assert file_content == json_str

    def test_report_contains_benchmark_data(self, sample_results):
        """Test that report contains all benchmark data"""
        reporter = JSONReporter()
        json_str = reporter.report(sample_results)

        data = json.loads(json_str)
        benchmark = data['benchmarks'][0]

        assert benchmark['name'] == "test_benchmark"
        assert benchmark['iterations'] == 100
        assert 'mean_time' in benchmark
        assert 'median_time' in benchmark
        assert 'std_dev' in benchmark


class TestHTMLReporter:
    """Test HTMLReporter class"""

    @pytest.fixture
    def sample_results(self):
        """Create sample benchmark results"""
        return [
            BenchmarkResult(
                name="test_1",
                iterations=100,
                mean_time=0.1,
                median_time=0.09,
                std_dev=0.01,
                min_time=0.08,
                max_time=0.15,
                p95_time=0.12,
                p99_time=0.14,
                total_time=10.0
            ),
            BenchmarkResult(
                name="test_2",
                iterations=50,
                mean_time=0.05,
                median_time=0.045,
                std_dev=0.005,
                min_time=0.04,
                max_time=0.075,
                p95_time=0.06,
                p99_time=0.07,
                total_time=2.5
            )
        ]

    def test_init_default_title(self):
        """Test HTMLReporter initialization with default title"""
        reporter = HTMLReporter()
        assert reporter.title == "Benchmark Results"

    def test_init_custom_title(self):
        """Test HTMLReporter initialization with custom title"""
        reporter = HTMLReporter(title="My Custom Title")
        assert reporter.title == "My Custom Title"

    def test_report_creates_html_file(self, sample_results):
        """Test that report creates HTML file"""
        reporter = HTMLReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file)

            assert output_file.exists()

            content = output_file.read_text()
            assert content.startswith("<!DOCTYPE html>")

    def test_report_html_contains_title(self, sample_results):
        """Test that HTML contains the title"""
        reporter = HTMLReporter(title="Test Report")

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file)

            content = output_file.read_text()
            assert "<title>Test Report</title>" in content
            assert "<h1>Test Report</h1>" in content

    def test_report_html_contains_results_table(self, sample_results):
        """Test that HTML contains results table"""
        reporter = HTMLReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file)

            content = output_file.read_text()
            assert "<table>" in content
            assert "<thead>" in content
            assert "<tbody>" in content
            assert "test_1" in content
            assert "test_2" in content

    def test_report_html_includes_charts(self, sample_results):
        """Test that HTML includes charts when requested"""
        reporter = HTMLReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file, include_charts=True)

            content = output_file.read_text()
            assert "chart.js" in content.lower()
            assert "performanceChart" in content

    def test_report_html_excludes_charts(self, sample_results):
        """Test that HTML excludes charts when not requested"""
        reporter = HTMLReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file, include_charts=False)

            content = output_file.read_text()
            assert "performanceChart" not in content

    def test_report_html_contains_summary(self, sample_results):
        """Test that HTML contains summary section"""
        reporter = HTMLReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file)

            content = output_file.read_text()
            assert "Total Benchmarks" in content
            assert "Average Time" in content
            assert "Fastest" in content
            assert "Slowest" in content

    def test_report_html_contains_styles(self, sample_results):
        """Test that HTML contains CSS styles"""
        reporter = HTMLReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.html"

            reporter.report(sample_results, output_file)

            content = output_file.read_text()
            assert "<style>" in content
            assert "font-family:" in content
            assert ".container" in content


class TestMarkdownReporter:
    """Test MarkdownReporter class"""

    @pytest.fixture
    def sample_results(self):
        """Create sample benchmark results"""
        return [
            BenchmarkResult(
                name="test_benchmark",
                iterations=100,
                mean_time=0.1,
                median_time=0.09,
                std_dev=0.01,
                min_time=0.08,
                max_time=0.15,
                p95_time=0.12,
                p99_time=0.14,
                total_time=10.0,
                memory_stats={'delta_mb': 50.5},
                cpu_stats={'average_cpu_percent': 75.3}
            )
        ]

    def test_report_returns_markdown_string(self, sample_results):
        """Test that report returns markdown string"""
        reporter = MarkdownReporter()
        markdown = reporter.report(sample_results)

        assert isinstance(markdown, str)
        assert markdown.startswith("# Benchmark Results")

    def test_report_contains_header(self, sample_results):
        """Test that markdown contains header section"""
        reporter = MarkdownReporter()
        markdown = reporter.report(sample_results)

        assert "# Benchmark Results" in markdown
        assert "**Date**:" in markdown
        assert "**Total Benchmarks**: 1" in markdown

    def test_report_contains_results_table(self, sample_results):
        """Test that markdown contains results table"""
        reporter = MarkdownReporter()
        markdown = reporter.report(sample_results)

        assert "## Results" in markdown
        assert "| Benchmark | Iterations | Mean | Median | P95 | P99 | Std Dev |" in markdown
        assert "|-----------|" in markdown
        assert "test_benchmark" in markdown

    def test_report_contains_detailed_results(self, sample_results):
        """Test that markdown contains detailed results section"""
        reporter = MarkdownReporter()
        markdown = reporter.report(sample_results)

        assert "## Detailed Results" in markdown
        assert "### test_benchmark" in markdown
        assert "- **Iterations**: 100" in markdown
        assert "- **Mean**:" in markdown
        assert "- **Median**:" in markdown

    def test_report_contains_memory_stats(self, sample_results):
        """Test that markdown contains memory statistics"""
        reporter = MarkdownReporter()
        markdown = reporter.report(sample_results)

        assert "**Memory**:" in markdown
        assert "delta_mb: 50.50" in markdown

    def test_report_contains_cpu_stats(self, sample_results):
        """Test that markdown contains CPU statistics"""
        reporter = MarkdownReporter()
        markdown = reporter.report(sample_results)

        assert "**CPU**:" in markdown
        assert "average_cpu_percent: 75.30" in markdown

    def test_report_writes_to_file(self, sample_results):
        """Test writing markdown report to file"""
        reporter = MarkdownReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_file = Path(tmpdir) / "output.md"

            markdown = reporter.report(sample_results, output_file=output_file)

            # Verify file was created
            assert output_file.exists()

            # Verify content matches
            file_content = output_file.read_text()
            assert file_content == markdown

    def test_report_multiple_results(self):
        """Test markdown with multiple results"""
        results = [
            BenchmarkResult(
                name="test_1",
                iterations=100,
                mean_time=0.1,
                median_time=0.09,
                std_dev=0.01,
                min_time=0.08,
                max_time=0.15,
                p95_time=0.12,
                p99_time=0.14,
                total_time=10.0
            ),
            BenchmarkResult(
                name="test_2",
                iterations=50,
                mean_time=0.05,
                median_time=0.045,
                std_dev=0.005,
                min_time=0.04,
                max_time=0.075,
                p95_time=0.06,
                p99_time=0.07,
                total_time=2.5
            )
        ]

        reporter = MarkdownReporter()
        markdown = reporter.report(results)

        assert "**Total Benchmarks**: 2" in markdown
        assert "### test_1" in markdown
        assert "### test_2" in markdown
