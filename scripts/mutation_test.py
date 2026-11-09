"""Mutation testing of the kernel and its force elements (plan task 5.5; D42, D49).

Each mutation alters one source line so that the physics changes: a sign, a
term left out, a factor. Some test must notice. For each mutation the script
applies it, builds the test executables, runs them (each stopping at its first
failure), and puts the line back. A mutation no test catches is a gap in the
tests; a mutation whose line is no longer in the source, or that does not
compile, is a list out of date. Either fails the run.

    python scripts/mutation_test.py --build "<command>" --test <executable> [--test ...]

In CI (Linux; the nightly workflow):
    python3 scripts/mutation_test.py \
        --build "cmake --build build-mutation --target test_kernel test_regression" \
        --test build-mutation/tests/test_regression --test build-mutation/tests/test_kernel

On the author's laptop, one compiler at a time through the build script:
    python scripts/mutation_test.py \
        --build "powershell -NoProfile -File scripts/build.ps1 -Target test_kernel,test_regression" \
        --test build/tests/test_regression.exe --test build/tests/test_kernel.exe

The script ends by building the unmutated sources again, so the build tree
matches the source. Under GitHub Actions each result is also an annotation.
"""

import argparse
import os
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent.parent

# (file, the line as it stands, the line mutated, what the mutation does)
MUTATIONS = [
    ("src/kernel/algorithms.cpp",
     "              + motion_cross_motion(data.v[i], vJ);",
     "              - motion_cross_motion(data.v[i], vJ);",
     "every joint's velocity-product acceleration with its sign flipped"),
    ("src/kernel/algorithms.cpp",
     "    a0 << Vec3::Zero(), -model.gravity;",
     "    a0 << Vec3::Zero(), -0.99 * model.gravity;",
     "gravity 1 % weaker in the recursive algorithms"),
    ("src/kernel/algorithms.cpp",
     "            pa += Ia * data.c[i] + data.U[i] * (data.Dinv[i] * data.u[i]);",
     "            pa += data.U[i] * (data.Dinv[i] * data.u[i]);",
     "the articulated-body algorithm without the velocity-product bias of its inertias"),
    ("src/kernel/simulator.cpp",
     "    v = v0_ + (dt / 6.0) * (kv_[0] + 2.0 * kv_[1] + 2.0 * kv_[2] + kv_[3]);",
     "    v = v0_ + (dt / 6.0) * (2.0 * kv_[0] + kv_[1] + 2.0 * kv_[2] + kv_[3]);",
     "RK4's velocity weights permuted (still consistent: first order)"),
    ("src/kernel/constraints.cpp",
     "    gamma.head<3>() = ki.a - kj.a;",
     "    gamma.head<3>().setZero();",
     "point coincidence without its velocity-product term"),
    ("src/kernel/joint_models.cpp",
     "    const Quat Q = (quat_from(q, 0).normalized() * exp3(dv.head<3>())).normalized();",
     "    const Quat Q = (exp3(dv.head<3>()) * quat_from(q, 0).normalized()).normalized();",
     "a spherical joint's rotation increment applied in the parent's axes, not the child's"),
    ("src/forces/spring_damper.cpp",
     "    const Real rate = (v2 - v1).dot(dir);",
     "    const Real rate = (v2 + v1).dot(dir);",
     "a spring-damper's rate from the sum of its ends' velocities"),
    ("src/forces/spring_damper.cpp",
     "    forces[static_cast<std::size_t>(body2_)].f_W += F2;",
     "    forces[static_cast<std::size_t>(body2_)].f_W += 0.99 * F2;",
     "a spring-damper's force on its second body 1 % short (action and reaction unequal)"),
    ("src/forces/pacejka.cpp",
     "    const Real D  = mu * Fz;",
     "    const Real D  = 1.02 * mu * Fz;",
     "the tyres' peak force 2 % higher"),
]


def annotate(level, title, message):
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} title={title}::{message}", flush=True)


def run(command, timeout):
    """Runs a command (a string, split as the shell would); returns its exit code."""
    args = command if isinstance(command, list) else shlex.split(command, posix=(os.name != "nt"))
    try:
        return subprocess.run(args, cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                              timeout=timeout).returncode
    except subprocess.TimeoutExpired:
        return -1


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--build", required=True, help="the command that builds the test executables")
    parser.add_argument("--test", action="append", required=True, help="a test executable (Catch2), in order")
    parser.add_argument("--only", type=int, action="append", help="run only these mutations (numbers from 1)")
    parser.add_argument("--timeout", type=float, default=1800.0, help="for each build and each test run [s]")
    args = parser.parse_args()

    results = []
    for number, (file, original, mutated, what) in enumerate(MUTATIONS, start=1):
        if args.only and number not in args.only:
            continue
        path = ROOT / file
        text = path.read_bytes().decode("utf-8")   # bytes: the line endings stay as they are
        if text.count(original) != 1:
            results.append((number, what, "STALE", f"the line is in {file} {text.count(original)} times, not once"))
            continue
        start = time.time()
        try:
            path.write_bytes(text.replace(original, mutated).encode("utf-8"))
            if run(args.build, args.timeout) != 0:
                results.append((number, what, "STALE", "the mutated source does not build"))
                continue
            caught_by = None
            for test in args.test:
                if run([str(ROOT / test), "--abort"], args.timeout) != 0:
                    caught_by = pathlib.Path(test).stem
                    break
            outcome = "caught" if caught_by else "SURVIVED"
            results.append((number, what, outcome,
                            f"by {caught_by}" if caught_by else "every test passed",))
        finally:
            path.write_bytes(text.encode("utf-8"))
        print(f"{number}. {what}: {results[-1][2]} {results[-1][3]} ({time.time() - start:.0f} s)", flush=True)

    print("Building the sources as they are...", flush=True)
    rebuilt = run(args.build, args.timeout) == 0

    failed = 0
    print()
    for number, what, outcome, detail in results:
        print(f"{number:2d}. {outcome:<9} {what} ({detail})")
        if outcome == "caught":
            annotate("notice", f"Mutation {number} caught", f"{what} ({detail})")
        else:
            failed += 1
            annotate("error", f"Mutation {number} {outcome.lower()}", f"{what} ({detail})")
    if not rebuilt:
        print("The unmutated sources did not build again.")
        failed += 1
    print(f"{sum(r[2] == 'caught' for r in results)} of {len(results)} mutations caught")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
