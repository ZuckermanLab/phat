"""
Unit and regression test for the phat package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import phat


def test_phat_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "phat" in sys.modules
