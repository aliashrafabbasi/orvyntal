"""Tests for the initial ORVYNTAL package contract."""

from importlib.metadata import version

import orvyntal


def test_package_version_is_declared() -> None:
    assert orvyntal.__version__ == "0.1.0"


def test_distribution_version_matches_module() -> None:
    assert version("orvyntal") == orvyntal.__version__
