"""
Unit tests for package build configuration.

These tests validate that the pyproject.toml configuration correctly
discovers and includes all Python packages in the built wheel.
"""

import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

import pytest


class TestPackageBuild:
    """Tests for validating package build configuration."""

    @pytest.fixture
    def project_root(self):
        """Get the project root directory."""
        # Navigate up from tests/unit to project root
        return Path(__file__).parent.parent.parent

    @pytest.fixture
    def expected_packages(self):
        """List of expected top-level packages that should be in the wheel."""
        return {
            'agents',
            'checkpoint',
            'cli',
            'database',
            'guardrails',
            'memory',
            'multimodal',
            'patterns',
            'pipeline',
            'plugins',
            'prompt_optimization',
            'rag',
            'reliability',
            'scheduler',
            'telemetry',
            'tools',
            'utils',
            'validation',
        }

    def test_pyproject_has_automatic_package_discovery(self, project_root):
        """Verify pyproject.toml uses automatic package discovery."""
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Should have the find configuration
        assert "[tool.setuptools.packages.find]" in content
        
        # Should NOT have explicit package listing
        assert 'packages = [' not in content
        
    def test_pyproject_excludes_test_directories(self, project_root):
        """Verify test directories are excluded from package discovery."""
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Should exclude tests, docs, etc.
        assert '"tests*"' in content or "'tests*'" in content
        assert '"docs*"' in content or "'docs*'" in content
        assert '"examples*"' in content or "'examples*'" in content

    @pytest.mark.slow
    def test_build_wheel_includes_all_packages(self, project_root, expected_packages, tmp_path):
        """
        Build a wheel and verify all expected packages are included.
        
        This is a slower integration-style test that actually builds the wheel.
        Run with: pytest -m slow
        """
        # Build the wheel
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.fail(f"Wheel build failed:\n{result.stderr}")
        
        # Find the built wheel
        wheel_files = list(tmp_path.glob("*.whl"))
        assert len(wheel_files) == 1, f"Expected 1 wheel, found {len(wheel_files)}"
        
        wheel_path = wheel_files[0]
        
        # Extract and check contents
        with zipfile.ZipFile(wheel_path, 'r') as whl:
            # Get all top-level directories in the wheel
            all_files = whl.namelist()
            top_level_dirs = {
                name.split('/')[0] 
                for name in all_files 
                if '/' in name and not name.startswith('ia_modules-')
            }
            
            # Verify all expected packages are present
            missing_packages = expected_packages - top_level_dirs
            assert not missing_packages, f"Missing packages in wheel: {missing_packages}"
            
            # Verify no test files leaked in
            test_files = [f for f in all_files if 'test' in f.lower() and f.endswith('.py')]
            assert not test_files, f"Test files should not be in wheel: {test_files[:5]}"

    @pytest.mark.slow
    def test_wheel_contains_python_files(self, project_root, tmp_path):
        """
        Verify the built wheel contains actual Python files, not just metadata.
        
        This catches issues like the broken 0.1.1 release that only had dist-info.
        """
        # Build the wheel
        result = subprocess.run(
            [sys.executable, "-m", "build", "--wheel", "--outdir", str(tmp_path)],
            cwd=str(project_root),
            capture_output=True,
            text=True
        )
        
        if result.returncode != 0:
            pytest.fail(f"Wheel build failed:\n{result.stderr}")
        
        wheel_files = list(tmp_path.glob("*.whl"))
        wheel_path = wheel_files[0]
        
        # Check wheel contents
        with zipfile.ZipFile(wheel_path, 'r') as whl:
            all_files = whl.namelist()
            
            # Count Python files (excluding dist-info)
            python_files = [
                f for f in all_files 
                if f.endswith('.py') and not f.startswith('ia_modules-')
            ]
            
            # Should have many Python files (at least 100+)
            assert len(python_files) > 100, (
                f"Wheel should contain many Python files, found only {len(python_files)}. "
                f"This might indicate a packaging issue similar to the 0.1.1 bug."
            )
            
            # Verify core modules are present
            core_files = {
                'pipeline/core.py',
                'agents/core.py',
                'pipeline/runner.py',
                'pipeline/services.py',
            }
            
            for core_file in core_files:
                assert core_file in all_files, f"Core file {core_file} missing from wheel"

    def test_importable_after_install(self, project_root):
        """
        Test that key modules can be imported after installation.
        
        This assumes the package is already installed in the current environment.
        Skip if not installed.
        """
        try:
            # Try importing various submodules
            import agents.core
            import pipeline.core
            import pipeline.services
            import tools.core
            import memory.core
            
            # If we get here, imports work
            assert True
            
        except ImportError as e:
            pytest.skip(f"Package not installed in current environment: {e}")

    def test_version_is_set(self, project_root):
        """Verify version is set in pyproject.toml."""
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text()
        
        # Should have a version
        assert 'version = ' in content
        
        # Version should not be 0.1.1 (the broken release)
        assert 'version = "0.1.1"' not in content

    def test_required_metadata_present(self, project_root):
        """Verify all required metadata is in pyproject.toml."""
        pyproject_path = project_root / "pyproject.toml"
        content = pyproject_path.read_text()
        
        required_fields = [
            'name = ',
            'version = ',
            'description = ',
            'authors = ',
            'readme = ',
            'license = ',
        ]
        
        for field in required_fields:
            assert field in content, f"Missing required field: {field}"
