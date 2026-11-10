"""The numbers a C++ test shares with its reproduction through Python (plan
task 6.3, D55). The C++ side is tests/support/parity.hpp: run with
MBD_PARITY_FILE set, a test case writes each number it records as a line
"name value", the value in hexadecimal floating point. Here a Record keeps
numbers the same way, cpp_record() runs one C++ test case and reads what it
wrote, and assert_identical() requires the two to hold the same doubles."""

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import pytest


class Record:
    """Numbers by name, each name's in the order recorded. A vector gives
    one number per entry, a matrix column by column (Eigen's order), a
    complex number the names name.re and name.im, a flag 0 or 1: as
    tests/support/parity.hpp writes them."""

    def __init__(self):
        self.numbers = {}

    def __call__(self, name, value):
        for x in np.asarray(value).ravel(order="F"):
            if np.iscomplexobj(x):
                self.add(name + ".re", x.real)
                self.add(name + ".im", x.imag)
            else:
                self.add(name, x)

    def add(self, name, number):
        self.numbers.setdefault(name, []).append(float(number))

    def count(self):
        return sum(len(values) for values in self.numbers.values())


def catch2_spec(test_case):
    """The test case's name as a Catch2 test spec matching it alone: a
    backslash before each character the spec would read otherwise (a comma
    separates names, brackets enclose tags)."""
    return "".join("\\" + c if c in '\\,[]"*~' else c for c in test_case)


def cpp_record(executable, test_case):
    """Run one test case of a C++ test executable of the suite's build with
    MBD_PARITY_FILE set, and return what it recorded. It must pass, and be
    the only case run. Skipped where the suite's build is not at hand
    (MBD_TEST_BIN_DIR, set by CTest: an installed package has no C++ tests)."""
    directory = os.environ.get("MBD_TEST_BIN_DIR")
    if not directory:
        pytest.skip("the parity tests run C++ test cases of the suite's build (MBD_TEST_BIN_DIR, set by CTest)")
    program = Path(directory) / (executable + (".exe" if os.name == "nt" else ""))
    assert program.is_file(), f"no {program}: build the suite's test executables first"
    with tempfile.TemporaryDirectory() as folder:
        out = Path(folder) / "parity.txt"
        run = subprocess.run([str(program), catch2_spec(test_case)], env=dict(os.environ, MBD_PARITY_FILE=str(out)),
                             capture_output=True, text=True, timeout=600)
        assert run.returncode == 0, f"{executable} '{test_case}' failed:\n{run.stdout[-4000:]}\n{run.stderr[-2000:]}"
        assert "in 1 test case" in run.stdout, f"{executable} did not run '{test_case}' alone:\n{run.stdout[-2000:]}"
        record = Record()
        for line in out.read_text().splitlines():
            name, value = line.rsplit(" ", 1)
            record.add(name, float.fromhex(value))
    return record


def ulps(a, b):
    """How many doubles apart a and b are (for the report; same sign)."""
    return abs(int(np.float64(a).view(np.int64)) - int(np.float64(b).view(np.int64)))


def assert_identical(python, cpp):
    """The same names, each with the same numbers in the same order, every
    number the same double: compared as hexadecimal, so that -0.0 differs
    from 0.0 and no tolerance enters."""
    assert sorted(python.numbers) == sorted(cpp.numbers), (
        f"names only from Python: {sorted(set(python.numbers) - set(cpp.numbers))}; "
        f"only from C++: {sorted(set(cpp.numbers) - set(python.numbers))}")
    differences = []
    for name, theirs in cpp.numbers.items():
        ours = python.numbers[name]
        if len(ours) != len(theirs):
            differences.append(f"{name}: {len(ours)} numbers from Python, {len(theirs)} from C++")
            continue
        for k, (a, b) in enumerate(zip(ours, theirs)):
            if a.hex() != b.hex():
                differences.append(f"{name}[{k}]: Python {a!r}, C++ {b!r} ({ulps(a, b)} ulps apart)")
    assert not differences, f"{len(differences)} of {cpp.count()} numbers differ:\n" + "\n".join(differences[:30])
