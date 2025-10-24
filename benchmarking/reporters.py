"""
Benchmark Reporters

Generate reports in various formats (console, JSON, HTML).
"""

import json
from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime, timezone


class ConsoleReporter:
    """
    Console reporter for benchmark results

    Prints human-readable benchmark results to console.
    """

    def __init__(self, use_colors: bool = True):
        self.use_colors = use_colors

    def report(self, results: List['BenchmarkResult']) -> None:
        """Print benchmark results to console"""
        if not results:
            print("No benchmark results to report")
            return

        self._print_header("Benchmark Results")

        for result in results:
            self._print_result(result)
            print()

    def _print_header(self, title: str) -> None:
        """Print formatted header"""
        print()
        print("=" * 70)
        print(f"  {title}")
        print("=" * 70)
        print()

    def _print_result(self, result: 'BenchmarkResult') -> None:
        """Print a single benchmark result"""
        print(f"📊 {result.name}")
        print(f"   Iterations: {result.iterations}")
        print()

        # Timing stats
        print("   Timing (ms):")
        print(f"     Mean:   {result.mean_time * 1000:>10.2f}")
        print(f"     Median: {result.median_time * 1000:>10.2f}")
        print(f"     Std Dev:{result.std_dev * 1000:>10.2f}")
        print(f"     Min:    {result.min_time * 1000:>10.2f}")
        print(f"     Max:    {result.max_time * 1000:>10.2f}")
        print(f"     P95:    {result.p95_time * 1000:>10.2f}")
        print(f"     P99:    {result.p99_time * 1000:>10.2f}")
        print(f"     Total:  {result.total_time:>10.2f}s")

        # Memory stats
        if result.memory_stats:
            print()
            print("   Memory (MB):")
            mem = result.memory_stats
            if 'delta_mb' in mem:
                print(f"     Delta:  {mem['delta_mb']:>10.2f}")
            if 'peak_mb' in mem:
                print(f"     Peak:   {mem['peak_mb']:>10.2f}")

        # CPU stats
        if result.cpu_stats:
            print()
            print("   CPU:")
            cpu = result.cpu_stats
            if 'average_cpu_percent' in cpu:
                print(f"     Avg %%:  {cpu['average_cpu_percent']:>10.2f}")
            if 'total_cpu_time' in cpu:
                print(f"     Time:   {cpu['total_cpu_time']:>10.2f}s")


class JSONReporter:
    """
    JSON reporter for benchmark results

    Generates JSON output suitable for CI/CD and automation.
    """

    def __init__(self, pretty: bool = True):
        self.pretty = pretty

    def report(
        self,
        results: List['BenchmarkResult'],
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate JSON report

        Args:
            results: Benchmark results
            output_file: Optional file path to write JSON

        Returns:
            JSON string
        """
        data = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'total_benchmarks': len(results),
            'benchmarks': [r.to_dict() for r in results]
        }

        json_str = json.dumps(
            data,
            indent=2 if self.pretty else None,
            sort_keys=True
        )

        if output_file:
            output_file.write_text(json_str)

        return json_str


class HTMLReporter:
    """
    HTML reporter for benchmark results

    Generates interactive HTML reports with charts.
    """

    def __init__(self, title: str = "Benchmark Results"):
        self.title = title

    def report(
        self,
        results: List['BenchmarkResult'],
        output_file: Path,
        include_charts: bool = True
    ) -> None:
        """
        Generate HTML report

        Args:
            results: Benchmark results
            output_file: Output HTML file path
            include_charts: Include Chart.js visualizations
        """
        html = self._generate_html(results, include_charts)
        output_file.write_text(html)

    def _generate_html(
        self,
        results: List['BenchmarkResult'],
        include_charts: bool
    ) -> str:
        """Generate HTML content"""
        html_parts = [
            self._html_header(),
            self._html_style(),
            f'<body><div class="container"><h1>{self.title}</h1>',
            self._html_summary(results),
        ]

        # Results table
        html_parts.append(self._html_results_table(results))

        # Charts
        if include_charts and results:
            html_parts.append(self._html_charts(results))

        html_parts.append('</div></body></html>')

        return '\n'.join(html_parts)

    def _html_header(self) -> str:
        """Generate HTML header"""
        return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{self.title}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@3.9.1/dist/chart.min.js"></script>
</head>'''

    def _html_style(self) -> str:
        """Generate CSS styles"""
        return '''<style>
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    background: #f5f5f5;
    margin: 0;
    padding: 20px;
}
.container {
    max-width: 1200px;
    margin: 0 auto;
    background: white;
    padding: 30px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}
h1 {
    color: #333;
    border-bottom: 3px solid #007bff;
    padding-bottom: 10px;
}
.summary {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 15px;
    margin: 20px 0;
}
.summary-card {
    background: #f8f9fa;
    padding: 15px;
    border-radius: 6px;
    border-left: 4px solid #007bff;
}
.summary-card h3 {
    margin: 0 0 10px 0;
    font-size: 14px;
    color: #666;
}
.summary-card .value {
    font-size: 24px;
    font-weight: bold;
    color: #333;
}
table {
    width: 100%;
    border-collapse: collapse;
    margin: 20px 0;
}
th, td {
    padding: 12px;
    text-align: left;
    border-bottom: 1px solid #ddd;
}
th {
    background: #f8f9fa;
    font-weight: 600;
    color: #333;
}
tr:hover {
    background: #f8f9fa;
}
.chart-container {
    margin: 30px 0;
    position: relative;
    height: 400px;
}
.metric-good {
    color: #28a745;
}
.metric-bad {
    color: #dc3545;
}
</style>'''

    def _html_summary(self, results: List['BenchmarkResult']) -> str:
        """Generate summary cards"""
        if not results:
            return '<p>No results</p>'

        total_benchmarks = len(results)
        avg_mean_time = sum(r.mean_time for r in results) / len(results) * 1000
        fastest = min(results, key=lambda r: r.mean_time)
        slowest = max(results, key=lambda r: r.mean_time)

        return f'''
<div class="summary">
    <div class="summary-card">
        <h3>Total Benchmarks</h3>
        <div class="value">{total_benchmarks}</div>
    </div>
    <div class="summary-card">
        <h3>Average Time</h3>
        <div class="value">{avg_mean_time:.2f}ms</div>
    </div>
    <div class="summary-card">
        <h3>Fastest</h3>
        <div class="value">{fastest.name}</div>
        <small>{fastest.mean_time * 1000:.2f}ms</small>
    </div>
    <div class="summary-card">
        <h3>Slowest</h3>
        <div class="value">{slowest.name}</div>
        <small>{slowest.mean_time * 1000:.2f}ms</small>
    </div>
</div>'''

    def _html_results_table(self, results: List['BenchmarkResult']) -> str:
        """Generate results table"""
        rows = []
        for result in results:
            rows.append(f'''
<tr>
    <td><strong>{result.name}</strong></td>
    <td>{result.iterations}</td>
    <td>{result.mean_time * 1000:.2f}ms</td>
    <td>{result.median_time * 1000:.2f}ms</td>
    <td>{result.p95_time * 1000:.2f}ms</td>
    <td>{result.p99_time * 1000:.2f}ms</td>
    <td>{result.std_dev * 1000:.2f}ms</td>
</tr>''')

        return f'''
<h2>Results</h2>
<table>
    <thead>
        <tr>
            <th>Benchmark</th>
            <th>Iterations</th>
            <th>Mean</th>
            <th>Median</th>
            <th>P95</th>
            <th>P99</th>
            <th>Std Dev</th>
        </tr>
    </thead>
    <tbody>
        {''.join(rows)}
    </tbody>
</table>'''

    def _html_charts(self, results: List['BenchmarkResult']) -> str:
        """Generate charts"""
        # Prepare data for Chart.js
        labels = [r.name for r in results]
        mean_times = [r.mean_time * 1000 for r in results]
        p95_times = [r.p95_time * 1000 for r in results]

        return f'''
<h2>Performance Comparison</h2>
<div class="chart-container">
    <canvas id="performanceChart"></canvas>
</div>
<script>
const ctx = document.getElementById('performanceChart').getContext('2d');
new Chart(ctx, {{
    type: 'bar',
    data: {{
        labels: {json.dumps(labels)},
        datasets: [{{
            label: 'Mean Time (ms)',
            data: {json.dumps(mean_times)},
            backgroundColor: 'rgba(54, 162, 235, 0.5)',
            borderColor: 'rgba(54, 162, 235, 1)',
            borderWidth: 1
        }}, {{
            label: 'P95 Time (ms)',
            data: {json.dumps(p95_times)},
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
            borderColor: 'rgba(255, 99, 132, 1)',
            borderWidth: 1
        }}]
    }},
    options: {{
        responsive: true,
        maintainAspectRatio: false,
        scales: {{
            y: {{
                beginAtZero: true,
                title: {{
                    display: true,
                    text: 'Time (ms)'
                }}
            }}
        }},
        plugins: {{
            title: {{
                display: true,
                text: 'Benchmark Performance'
            }},
            legend: {{
                display: true,
                position: 'top'
            }}
        }}
    }}
}});
</script>'''


class MarkdownReporter:
    """
    Markdown reporter for benchmark results

    Generates markdown-formatted reports suitable for GitHub, GitLab, etc.
    """

    def report(
        self,
        results: List['BenchmarkResult'],
        output_file: Optional[Path] = None
    ) -> str:
        """
        Generate markdown report

        Args:
            results: Benchmark results
            output_file: Optional file path to write markdown

        Returns:
            Markdown string
        """
        lines = [
            "# Benchmark Results",
            "",
            f"**Date**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC",
            f"**Total Benchmarks**: {len(results)}",
            "",
            "## Results",
            "",
            "| Benchmark | Iterations | Mean | Median | P95 | P99 | Std Dev |",
            "|-----------|------------|------|--------|-----|-----|---------|",
        ]

        for result in results:
            lines.append(
                f"| {result.name} | {result.iterations} | "
                f"{result.mean_time * 1000:.2f}ms | "
                f"{result.median_time * 1000:.2f}ms | "
                f"{result.p95_time * 1000:.2f}ms | "
                f"{result.p99_time * 1000:.2f}ms | "
                f"{result.std_dev * 1000:.2f}ms |"
            )

        lines.extend(["", "## Detailed Results", ""])

        for result in results:
            lines.extend([
                f"### {result.name}",
                "",
                f"- **Iterations**: {result.iterations}",
                f"- **Mean**: {result.mean_time * 1000:.2f}ms",
                f"- **Median**: {result.median_time * 1000:.2f}ms",
                f"- **Min**: {result.min_time * 1000:.2f}ms",
                f"- **Max**: {result.max_time * 1000:.2f}ms",
                f"- **P95**: {result.p95_time * 1000:.2f}ms",
                f"- **P99**: {result.p99_time * 1000:.2f}ms",
                f"- **Std Dev**: {result.std_dev * 1000:.2f}ms",
                ""
            ])

            if result.memory_stats:
                lines.append("**Memory**:")
                for key, value in result.memory_stats.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- {key}: {value:.2f}")
                lines.append("")

            if result.cpu_stats:
                lines.append("**CPU**:")
                for key, value in result.cpu_stats.items():
                    if isinstance(value, (int, float)):
                        lines.append(f"- {key}: {value:.2f}")
                lines.append("")

        markdown = "\n".join(lines)

        if output_file:
            output_file.write_text(markdown)

        return markdown
