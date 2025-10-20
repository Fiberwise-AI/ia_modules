#!/usr/bin/env python3
"""
Quick start script for IA Modules Dashboard

Usage:
    python run_dashboard.py
    python run_dashboard.py --port 8080
    python run_dashboard.py --host 0.0.0.0 --port 8000
"""

import argparse
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def main():
    parser = argparse.ArgumentParser(description="IA Modules Dashboard")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error"],
                        help="Log level (default: info)")

    args = parser.parse_args()

    # Check dependencies
    try:
        import fastapi
        import uvicorn
        import pydantic
    except ImportError as e:
        print(f"❌ Missing required dependency: {e}")
        print("\n📦 Please install dependencies:")
        print("   cd dashboard")
        print("   pip install -r requirements.txt")
        sys.exit(1)

    # Print banner
    print("=" * 60)
    print("  IA Modules Dashboard")
    print("=" * 60)
    print(f"  Host: {args.host}")
    print(f"  Port: {args.port}")
    print(f"  API:  http://{args.host}:{args.port}")
    print(f"  Docs: http://{args.host}:{args.port}/docs")
    print("=" * 60)
    print()

    # Run server
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()
