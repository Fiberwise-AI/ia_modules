"""
Edge case tests for benchmarking/reporters.py to reach 100% coverage
"""
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
from pathlib import Path
import tempfile
import os

from ia_modules.benchmarking.reporters import (
    ConsoleReporter,
    JSONReporter,
    HTMLReporter,
)
from ia_modules.benchmarking.framework import BenchmarkResult


class TestHTMLReporterEdgeCases:
    """Test edge cases in HTMLReporter"""

    def test_html_summary_with_empty_results(self):
        """Test that _html_summary returns 'No results' message when results list is empty"""
        reporter = HTMLReporter()

        # Test with empty results list
        html = reporter._html_summary([])

        assert html == '<p>No results</p>'

    def test_report_with_empty_results(self):
        """Test generating HTML report with no results"""
        reporter = HTMLReporter()

        # Create a temp directory for output
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.html"

            # Generate report with empty results (returns None, writes to file)
            reporter.report([], output_file=output_path)

            # File should be created
            assert output_path.exists()
            content = output_path.read_text()
            assert 'No results</p>' in content
            assert '<html' in content  # Should still be valid HTML


class TestJSONReporterEdgeCases:
    """Test edge cases in JSONReporter"""

    def test_report_with_empty_results(self):
        """Test generating JSON report with no results"""
        reporter = JSONReporter()

        # Generate report with empty results
        json_str = reporter.report([])

        # Should be valid JSON with metadata and empty benchmarks list
        import json
        data = json.loads(json_str)
        assert data['benchmarks'] == []
        assert data['total_benchmarks'] == 0
        assert 'timestamp' in data

    def test_report_creates_json_file(self):
        """Test that report() creates a JSON file when output_file specified"""
        reporter = JSONReporter()

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "report.json"

            # Generate report with output file
            reporter.report([], output_file=output_path)

            # File should exist
            assert output_path.exists()

            # File should contain valid JSON
            import json
            with open(output_path) as f:
                data = json.load(f)
            assert data['benchmarks'] == []
            assert data['total_benchmarks'] == 0


class TestConsoleReporterEdgeCases:
    """Test edge cases in ConsoleReporter"""

    def test_report_with_empty_results(self, capsys):
        """Test console report with no results"""
        reporter = ConsoleReporter()

        # Report with empty results
        reporter.report([])

        # Capture output
        captured = capsys.readouterr()

        # Should print "No benchmark results to report"
        assert 'No benchmark results' in captured.out
