"""
Comprehensive tests for benchmarking/ci_integration.py
"""
import pytest
from ia_modules.pipeline.test_utils import create_test_execution_context
import json
import tempfile
from pathlib import Path
from io import StringIO
import sys

from ia_modules.benchmarking.ci_integration import (
    CIConfig,
    CIIntegration,
    github_actions_workflow,
    gitlab_ci_config
)
from ia_modules.benchmarking.comparison import ComparisonMetric, PerformanceChange
from ia_modules.benchmarking.framework import BenchmarkResult


class TestCIConfig:
    """Test CIConfig dataclass"""

    def test_ci_config_creation(self):
        """Test creating CI config with required fields"""
        config = CIConfig(
            baseline_file=Path("baseline.json"),
            current_file=Path("current.json")
        )

        assert config.baseline_file == Path("baseline.json")
        assert config.current_file == Path("current.json")
        assert config.regression_threshold == 10.0
        assert config.output_format == "markdown"
        assert config.fail_on_regression is True

    def test_ci_config_with_custom_values(self):
        """Test creating CI config with custom values"""
        config = CIConfig(
            baseline_file=Path("baseline.json"),
            current_file=Path("current.json"),
            regression_threshold=5.0,
            output_format="json",
            fail_on_regression=False
        )

        assert config.regression_threshold == 5.0
        assert config.output_format == "json"
        assert config.fail_on_regression is False

    def test_ci_config_default_metrics(self):
        """Test that default metrics are set in post_init"""
        config = CIConfig(
            baseline_file=Path("baseline.json"),
            current_file=Path("current.json")
        )

        assert config.metrics_to_compare is not None
        assert ComparisonMetric.MEAN_TIME in config.metrics_to_compare
        assert ComparisonMetric.P95_TIME in config.metrics_to_compare
        assert len(config.metrics_to_compare) == 2

    def test_ci_config_custom_metrics(self):
        """Test setting custom metrics list"""
        custom_metrics = [
            ComparisonMetric.MEAN_TIME,
            ComparisonMetric.MEDIAN_TIME,
            ComparisonMetric.P99_TIME
        ]

        config = CIConfig(
            baseline_file=Path("baseline.json"),
            current_file=Path("current.json"),
            metrics_to_compare=custom_metrics
        )

        assert config.metrics_to_compare == custom_metrics
        assert len(config.metrics_to_compare) == 3


class TestCIIntegration:
    """Test CIIntegration class"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def baseline_data(self):
        """Sample baseline benchmark data"""
        return {
            "benchmarks": [
                {
                    "name": "test_benchmark",
                    "iterations": 100,
                    "mean_time": 0.1,
                    "median_time": 0.09,
                    "std_dev": 0.01,
                    "min_time": 0.08,
                    "max_time": 0.15,
                    "p95_time": 0.12,
                    "p99_time": 0.14,
                    "total_time": 10.0
                }
            ]
        }

    @pytest.fixture
    def current_data_no_change(self):
        """Sample current benchmark data with no change"""
        return {
            "benchmarks": [
                {
                    "name": "test_benchmark",
                    "iterations": 100,
                    "mean_time": 0.1,
                    "median_time": 0.09,
                    "std_dev": 0.01,
                    "min_time": 0.08,
                    "max_time": 0.15,
                    "p95_time": 0.12,
                    "p99_time": 0.14,
                    "total_time": 10.0
                }
            ]
        }

    @pytest.fixture
    def current_data_regression(self):
        """Sample current benchmark data with regression"""
        return {
            "benchmarks": [
                {
                    "name": "test_benchmark",
                    "iterations": 100,
                    "mean_time": 0.15,  # 50% slower
                    "median_time": 0.14,
                    "std_dev": 0.02,
                    "min_time": 0.12,
                    "max_time": 0.20,
                    "p95_time": 0.18,  # 50% slower
                    "p99_time": 0.19,
                    "total_time": 15.0
                }
            ]
        }

    @pytest.fixture
    def current_data_improved(self):
        """Sample current benchmark data with improvement"""
        return {
            "benchmarks": [
                {
                    "name": "test_benchmark",
                    "iterations": 100,
                    "mean_time": 0.05,  # 50% faster
                    "median_time": 0.045,
                    "std_dev": 0.005,
                    "min_time": 0.04,
                    "max_time": 0.075,
                    "p95_time": 0.06,  # 50% faster
                    "p99_time": 0.07,
                    "total_time": 5.0
                }
            ]
        }

    def test_init(self, temp_dir):
        """Test CIIntegration initialization"""
        config = CIConfig(
            baseline_file=temp_dir / "baseline.json",
            current_file=temp_dir / "current.json"
        )

        ci = CIIntegration(config)

        assert ci.config == config
        assert ci.comparator is not None
        assert ci.comparator.regression_threshold == 10.0

    def test_run_success_no_regression(self, temp_dir, baseline_data, current_data_no_change):
        """Test successful run with no regression"""
        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "current.json"

        baseline_file.write_text(json.dumps(baseline_data))
        current_file.write_text(json.dumps(current_data_no_change))

        config = CIConfig(baseline_file=baseline_file, current_file=current_file)
        ci = CIIntegration(config)

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        exit_code = ci.run()

        sys.stdout = sys.__stdout__

        assert exit_code == 0

    def test_run_with_regression(self, temp_dir, baseline_data, current_data_regression):
        """Test run with regression detected"""
        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "current.json"

        baseline_file.write_text(json.dumps(baseline_data))
        current_file.write_text(json.dumps(current_data_regression))

        config = CIConfig(baseline_file=baseline_file, current_file=current_file)
        ci = CIIntegration(config)

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        exit_code = ci.run()

        sys.stdout = sys.__stdout__

        assert exit_code == 1  # Should fail on regression

    def test_run_with_regression_no_fail(self, temp_dir, baseline_data, current_data_regression):
        """Test run with regression but fail_on_regression=False"""
        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "current.json"

        baseline_file.write_text(json.dumps(baseline_data))
        current_file.write_text(json.dumps(current_data_regression))

        config = CIConfig(
            baseline_file=baseline_file,
            current_file=current_file,
            fail_on_regression=False
        )
        ci = CIIntegration(config)

        # Capture stdout
        captured_output = StringIO()
        sys.stdout = captured_output

        exit_code = ci.run()

        sys.stdout = sys.__stdout__

        assert exit_code == 0  # Should not fail even with regression

    def test_run_baseline_file_not_found(self, temp_dir):
        """Test run when baseline file doesn't exist"""
        baseline_file = temp_dir / "nonexistent_baseline.json"
        current_file = temp_dir / "current.json"
        current_file.write_text('{"benchmarks": []}')

        config = CIConfig(baseline_file=baseline_file, current_file=current_file)
        ci = CIIntegration(config)

        # Capture stderr
        captured_stderr = StringIO()
        sys.stderr = captured_stderr

        exit_code = ci.run()

        sys.stderr = sys.__stderr__

        assert exit_code == 1
        assert "not found" in captured_stderr.getvalue().lower()

    def test_run_current_file_not_found(self, temp_dir, baseline_data):
        """Test run when current file doesn't exist"""
        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "nonexistent_current.json"

        baseline_file.write_text(json.dumps(baseline_data))

        config = CIConfig(baseline_file=baseline_file, current_file=current_file)
        ci = CIIntegration(config)

        # Capture stderr
        captured_stderr = StringIO()
        sys.stderr = captured_stderr

        exit_code = ci.run()

        sys.stderr = sys.__stderr__

        assert exit_code == 1
        assert "not found" in captured_stderr.getvalue().lower()

    def test_run_invalid_json(self, temp_dir):
        """Test run with invalid JSON files"""
        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "current.json"

        baseline_file.write_text("invalid json{")
        current_file.write_text("invalid json{")

        config = CIConfig(baseline_file=baseline_file, current_file=current_file)
        ci = CIIntegration(config)

        # Capture stderr
        captured_stderr = StringIO()
        sys.stderr = captured_stderr

        exit_code = ci.run()

        sys.stderr = sys.__stderr__

        assert exit_code == 1
        assert "invalid json" in captured_stderr.getvalue().lower()

    def test_run_missing_benchmark_in_current(self, temp_dir, baseline_data):
        """Test when a benchmark exists in baseline but not in current"""
        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "current.json"

        baseline_file.write_text(json.dumps(baseline_data))
        current_file.write_text(json.dumps({"benchmarks": []}))

        config = CIConfig(baseline_file=baseline_file, current_file=current_file)
        ci = CIIntegration(config)

        # Capture stderr
        captured_stderr = StringIO()
        sys.stderr = captured_stderr
        captured_stdout = StringIO()
        sys.stdout = captured_stdout

        exit_code = ci.run()

        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__

        # Should contain "error" or "warning" when benchmark is missing
        stderr_output = captured_stderr.getvalue().lower()
        assert "error" in stderr_output or "warning" in stderr_output
        assert exit_code == 1  # Error: benchmark missing from current

    def test_load_results_single_result(self, temp_dir):
        """Test loading a single result (not wrapped in 'benchmarks' key)"""
        result_file = temp_dir / "result.json"
        single_result = {
            "name": "test",
            "mean_time": 0.1
        }
        result_file.write_text(json.dumps(single_result))

        config = CIConfig(baseline_file=result_file, current_file=result_file)
        ci = CIIntegration(config)

        results = ci._load_results(result_file)

        assert results is not None
        assert len(results) == 1
        assert results[0]['name'] == "test"

    def test_dict_to_result(self, temp_dir):
        """Test converting dict to BenchmarkResult"""
        config = CIConfig(
            baseline_file=temp_dir / "baseline.json",
            current_file=temp_dir / "current.json"
        )
        ci = CIIntegration(config)

        data = {
            "name": "test_benchmark",
            "iterations": 100,
            "mean_time": 0.1,
            "median_time": 0.09,
            "std_dev": 0.01,
            "min_time": 0.08,
            "max_time": 0.15,
            "p95_time": 0.12,
            "p99_time": 0.14,
            "total_time": 10.0
        }

        result = ci._dict_to_result(data)

        assert isinstance(result, BenchmarkResult)
        assert result.name == "test_benchmark"
        assert result.iterations == 100
        assert result.mean_time == 0.1
        assert result.p95_time == 0.12


class TestReportGeneration:
    """Test report generation in different formats"""

    @pytest.fixture
    def temp_dir(self):
        """Create temporary directory for test files"""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def test_data(self, temp_dir):
        """Setup test data"""
        baseline_data = {
            "benchmarks": [{
                "name": "test",
                "iterations": 100,
                "mean_time": 0.1,
                "median_time": 0.09,
                "std_dev": 0.01,
                "min_time": 0.08,
                "max_time": 0.15,
                "p95_time": 0.12,
                "p99_time": 0.14,
                "total_time": 10.0
            }]
        }

        current_data = {
            "benchmarks": [{
                "name": "test",
                "iterations": 100,
                "mean_time": 0.15,
                "median_time": 0.14,
                "std_dev": 0.02,
                "min_time": 0.12,
                "max_time": 0.20,
                "p95_time": 0.18,
                "p99_time": 0.19,
                "total_time": 15.0
            }]
        }

        baseline_file = temp_dir / "baseline.json"
        current_file = temp_dir / "current.json"

        baseline_file.write_text(json.dumps(baseline_data))
        current_file.write_text(json.dumps(current_data))

        return baseline_file, current_file

    def test_markdown_report_generation(self, test_data):
        """Test markdown report generation"""
        baseline_file, current_file = test_data

        config = CIConfig(
            baseline_file=baseline_file,
            current_file=current_file,
            output_format="markdown"
        )
        ci = CIIntegration(config)

        captured_output = StringIO()
        sys.stdout = captured_output

        ci.run()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "# Benchmark Comparison Report" in output
        assert "Performance Regression Detected" in output
        assert "| Metric | Baseline | Current | Change | Status |" in output
        assert "❌" in output  # Regression symbol

    def test_json_report_generation(self, test_data):
        """Test JSON report generation"""
        baseline_file, current_file = test_data

        config = CIConfig(
            baseline_file=baseline_file,
            current_file=current_file,
            output_format="json"
        )
        ci = CIIntegration(config)

        captured_output = StringIO()
        sys.stdout = captured_output

        ci.run()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        # Parse JSON output
        report_data = json.loads(output)

        assert report_data['has_regression'] is True
        assert 'comparisons' in report_data
        assert len(report_data['comparisons']) > 0
        assert 'metric' in report_data['comparisons'][0]
        assert 'percent_change' in report_data['comparisons'][0]

    def test_github_actions_report_generation(self, test_data):
        """Test GitHub Actions report generation"""
        baseline_file, current_file = test_data

        config = CIConfig(
            baseline_file=baseline_file,
            current_file=current_file,
            output_format="github-actions"
        )
        ci = CIIntegration(config)

        captured_output = StringIO()
        sys.stdout = captured_output

        ci.run()

        sys.stdout = sys.__stdout__
        output = captured_output.getvalue()

        assert "::set-output name=has_regression::true" in output
        assert "::warning::" in output  # Should have warnings for regressions
        assert "::error::Performance regression detected!" in output

    def test_unknown_report_format(self, test_data):
        """Test handling of unknown report format"""
        baseline_file, current_file = test_data

        config = CIConfig(
            baseline_file=baseline_file,
            current_file=current_file,
            output_format="unknown-format"
        )
        ci = CIIntegration(config)

        captured_stderr = StringIO()
        sys.stderr = captured_stderr
        captured_stdout = StringIO()
        sys.stdout = captured_stdout

        ci.run()

        sys.stderr = sys.__stderr__
        sys.stdout = sys.__stdout__

        assert "unknown output format" in captured_stderr.getvalue().lower()


class TestWorkflowGeneration:
    """Test CI/CD workflow generation functions"""

    def test_github_actions_workflow(self):
        """Test GitHub Actions workflow generation"""
        workflow = github_actions_workflow()

        assert workflow is not None
        assert "name: Benchmark Performance" in workflow
        assert "on:" in workflow
        assert "pull_request:" in workflow
        assert "branches: [main]" in workflow
        assert "runs-on: ubuntu-latest" in workflow
        assert "actions/checkout@v3" in workflow
        assert "actions/setup-python@v4" in workflow
        assert "python-version: '3.9'" in workflow
        assert "pip install -e ia_modules" in workflow
        assert "--fail-on-regression" in workflow

    def test_gitlab_ci_config(self):
        """Test GitLab CI configuration generation"""
        config = gitlab_ci_config()

        assert config is not None
        assert "benchmark:" in config
        assert "stage: test" in config
        assert "image: python:3.9" in config
        assert "script:" in config
        assert "pip install -e ia_modules" in config
        assert "artifacts:" in config
        assert "paths:" in config
        assert "current_results.json" in config
        assert "benchmark_report.md" in config
        assert "expire_in: 30 days" in config
