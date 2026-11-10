"""Checks that the tests import the package they are meant to test (task 6.2, D53).

The suite (CTest) sets MBD_TEST_PACKAGE_DIR to the package its build laid out.
A package installed with ``pip install -e`` into the same interpreter would be
imported instead: its finder comes before PYTHONPATH, and its compiled module
is the one built at the last install, not this build's. Run by hand against an
installed package, with the variable unset, the tests check nothing here.
"""

import os
from pathlib import Path

import pytest

import multibody


def pytest_configure(config):
    expected = os.environ.get("MBD_TEST_PACKAGE_DIR")
    if not expected:
        return
    imported = Path(multibody._mbd.__file__).resolve().parent
    expected = Path(expected).resolve()
    if imported != expected:
        raise pytest.UsageError(
            f"the tests imported multibody from {imported}, not from this build's "
            f"{expected}. A package installed into this interpreter with pip install "
            f"-e is imported before PYTHONPATH's: uninstall it (pip uninstall "
            f"multibody) or install it into another environment.")
