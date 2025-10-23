#!/bin/bash
# Cleanup script - Remove unnecessary session/summary files created by AI

set -e

echo "🗑️  Cleaning up unnecessary documentation files..."

# Remove session summaries and planning docs that clutter the repo
rm -f TEST_EXECUTION_SUMMARY.md
rm -f TEST_STATUS.md
rm -f CLEANUP_PLAN.md
rm -f ARCHIVE_RECOMMENDATIONS.md
rm -f RELEASE_READY_v0.0.3.md
rm -f RELEASE_CHECKLIST_v0.0.3.md
rm -f DATABASE_CLEANUP_SUMMARY.md
rm -f TEST_FIXES_SUMMARY.md
rm -f FINAL_TEST_SUMMARY.md
rm -f SESSION_COMPLETE.md
rm -f FIXES_AND_IMPROVEMENTS.md

# Remove temporary SQL translation files
rm -f translated_v007.sql
rm -f v002_translated.sql

# Remove one-time fix scripts
rm -f fix_datetime_and_analysis.py

# Remove test planning docs if they exist
rm -f tests/COMPREHENSIVE_TEST_PLAN.md
rm -f tests/COMPREHENSIVE_TEST_REVIEW.md

# Remove this cleanup script itself
rm -f cleanup.sh

echo "✓ Cleanup complete!"
echo ""
echo "Deleted:"
echo "  - Session summary files"
echo "  - Planning documents"
echo "  - Temporary SQL files"
echo "  - One-time scripts"
echo "  - Test planning docs"
