#!/usr/bin/env python3
"""
Test script to verify PyPI publishing workflow locally.

This simulates the GitHub Actions workflow steps:
1. Install build dependencies
2. Run unit tests
3. Build the package
4. Check the package with twine
5. Optionally publish to TestPyPI

Usage:
    python test_publish.py                  # Build and check only
    python test_publish.py --test-pypi      # Also publish to TestPyPI
    python test_publish.py --skip-tests     # Skip tests (for quick builds)
"""
import subprocess
import sys
import argparse
from pathlib import Path
import shutil


def run_command(cmd, check=True, env=None):
    """Run a command and print output."""
    print(f"\n{'='*60}")
    print(f"Running: {' '.join(cmd)}")
    print(f"{'='*60}")
    result = subprocess.run(cmd, check=False, env=env, capture_output=False)
    if check and result.returncode != 0:
        print(f"❌ Command failed with exit code {result.returncode}")
        sys.exit(result.returncode)
    return result.returncode


def clean_build_artifacts():
    """Remove previous build artifacts."""
    print("\n🧹 Cleaning previous build artifacts...")
    for path in ["dist", "build", "*.egg-info"]:
        for item in Path(".").glob(path):
            if item.is_dir():
                shutil.rmtree(item)
                print(f"   Removed: {item}")
            else:
                item.unlink()
                print(f"   Removed: {item}")


def install_build_deps():
    """Install build and test dependencies."""
    print("\n📦 Installing build dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run_command([sys.executable, "-m", "pip", "install", "build", "twine"])


def install_package_dev():
    """Install package in editable mode with dev dependencies."""
    print("\n📦 Installing ia_modules in editable mode with dev dependencies...")
    run_command([sys.executable, "-m", "pip", "install", "-e", ".[dev,all]"])


def run_tests():
    """Run unit tests."""
    print("\n🧪 Running unit tests...")
    return run_command(
        [sys.executable, "-m", "pytest", "tests/unit/", "-q"],
        check=False
    )


def build_package():
    """Build the package."""
    print("\n🏗️  Building package...")
    run_command([sys.executable, "-m", "build"])
    
    # List built artifacts
    print("\n📦 Built artifacts:")
    dist_path = Path("dist")
    if dist_path.exists():
        for artifact in dist_path.iterdir():
            print(f"   {artifact.name} ({artifact.stat().st_size / 1024:.1f} KB)")


def check_package():
    """Check package with twine."""
    print("\n🔍 Checking package with twine...")
    run_command([sys.executable, "-m", "twine", "check", "dist/*"])


def publish_to_testpypi(token=None):
    """Publish to TestPyPI."""
    print("\n🚀 Publishing to TestPyPI...")
    
    if not token:
        print("⚠️  No TEST_PYPI_TOKEN provided. Set environment variable or pass --token")
        print("   You can create a token at: https://test.pypi.org/manage/account/token/")
        return
    
    cmd = [
        sys.executable, "-m", "twine", "upload",
        "--repository-url", "https://test.pypi.org/legacy/",
        "-u", "__token__",
        "-p", token,
        "dist/*"
    ]
    
    # Don't print the token in the command output
    print("Running: python -m twine upload --repository-url https://test.pypi.org/legacy/ -u __token__ -p [REDACTED] dist/*")
    result = subprocess.run(cmd, check=False)
    
    if result.returncode == 0:
        print("\n✅ Successfully published to TestPyPI!")
        print("   View at: https://test.pypi.org/project/ia-modules/")
    else:
        print("\n❌ Failed to publish to TestPyPI")


def main():
    parser = argparse.ArgumentParser(description="Test PyPI publishing workflow")
    parser.add_argument("--test-pypi", action="store_true", help="Publish to TestPyPI")
    parser.add_argument("--skip-tests", action="store_true", help="Skip running tests")
    parser.add_argument("--token", help="TestPyPI token (or set TEST_PYPI_TOKEN env var)")
    parser.add_argument("--no-clean", action="store_true", help="Don't clean build artifacts first")
    args = parser.parse_args()
    
    print("🚀 IA Modules - PyPI Publishing Test")
    print("=" * 60)
    
    # Clean previous builds
    if not args.no_clean:
        clean_build_artifacts()
    
    # Install dependencies
    install_build_deps()
    
    if not args.skip_tests:
        install_package_dev()
        
        # Run tests
        test_result = run_tests()
        if test_result != 0:
            print("\n❌ Tests failed! Fix tests before publishing.")
            print("   Use --skip-tests to build anyway (not recommended)")
            sys.exit(1)
        print("\n✅ All tests passed!")
    else:
        print("\n⚠️  Skipping tests (--skip-tests flag)")
    
    # Build package
    build_package()
    
    # Check package
    check_package()
    
    print("\n✅ Package build and check complete!")
    
    # Publish to TestPyPI if requested
    if args.test_pypi:
        import os
        token = args.token or os.getenv("TEST_PYPI_TOKEN")
        publish_to_testpypi(token)
    else:
        print("\n💡 To publish to TestPyPI, run with --test-pypi flag")
        print("   Set TEST_PYPI_TOKEN environment variable or use --token")
    
    print("\n" + "=" * 60)
    print("🎉 Done!")
    print("=" * 60)


if __name__ == "__main__":
    main()
