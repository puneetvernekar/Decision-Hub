"""
Re-export release notes and known issues loaded from data/ CSV/MD files.

The actual data lives in:
  data/release_notes.md
  data/known_issues.csv

Loaded by mock_dashboard.py; re-exported here for backward compatibility.
"""

from .mock_dashboard import RELEASE_NOTES, KNOWN_ISSUES

__all__ = ["RELEASE_NOTES", "KNOWN_ISSUES"]
