# Note for Pinocchio's developers: joints added out of depth-first order give silently wrong results

Prepared on 7 November 2026 for the author to send (a GitHub issue on
stack-of-tasks/pinocchio, or by mail). Found in plan task 5.3 (journal
entry of 7 November 2026, D46). The reproduction is
`pinocchio_depth_first.py` beside this file; its output below was run on
the author's laptop.

---

**Suggested title:** Python: `Model.addJoint` accepts joints out of depth-first order, and `crba`, `computeMinverse` and `cholesky` then return wrong results without an error

**Body:**

Hello, and thank you for Pinocchio. We use it as an independent reference
for a multibody code of our own, comparing ABA, RNEA, CRBA and placements on
random trees. Our generator hangs each new body from any earlier one, so its
trees are numbered parents-first but not always depth first. On those trees
Pinocchio's ABA and RNEA agreed with ours to 1e-13, but `crba` differed by
up to 66 % relative, on 37 of 40 trees.

The cause is the order in which the joints were added. The C++
documentation of `ModelTpl::addJoint` says "Joints need to be added to the
tree in a depth-first order", and `crba` asserts `model.check(data)`. But:

- the Python docstrings of `Model.addJoint` do not mention the order;
- in a release build (the conda-forge packages) the assert is compiled out,
  and nothing else checks it, so a model built from Python in another order
  gives wrong results with no warning;
- `model.check(data)` does detect it (it returns `False`), but a user has
  no reason to call it.

**Versions:** Pinocchio 4.1.0 from conda-forge (`pinocchio`,
`py312h2d15979_1`), Python 3.12.16, NumPy 2.5.4, Windows 11 (x64).

**Reproduction** (Pinocchio and NumPy only): three revolute joints, A and B
on the world, C on A. Added in the order A, C, B everything agrees; added A,
B, C, `crba`, `computeMinverse` and `cholesky.solve` are wrong while `aba`
and `rnea` stay right.

```python
import numpy as np
import pinocchio as pin


def three_joints(order):
    rng = np.random.default_rng(0)
    placement = {k: pin.SE3(pin.exp3(rng.normal(size=3)), rng.normal(size=3)) for k in "ABC"}
    inertia = {k: pin.Inertia(1.0 + rng.random(), 0.3 * rng.normal(size=3), np.diag(0.1 + rng.random(3)))
               for k in "ABC"}
    model, ids = pin.Model(), {}
    for name in order:
        parent = ids["A"] if name == "C" else 0
        ids[name] = model.addJoint(parent, pin.JointModelRZ(), placement[name], name)
        model.appendBodyToJoint(ids[name], inertia[name], pin.SE3.Identity())
    return model


def symmetric(M):
    return np.triu(M) + np.triu(M, 1).T


print("Pinocchio", pin.__version__)
q, v = np.array([0.3, -0.7, 1.1]), np.array([0.5, -1.0, 2.0])
for order in ("ACB", "ABC"):
    model = three_joints(order)
    # The mass matrix from rnea, column by column (gravity off, v = 0).
    no_gravity = pin.Model(model)
    no_gravity.gravity = pin.Motion.Zero()
    M_rnea = np.column_stack([pin.rnea(no_gravity, no_gravity.createData(), q, np.zeros(3), e)
                              for e in np.eye(3)])
    M_crba = symmetric(pin.crba(model, model.createData(), q))
    Minv = symmetric(pin.computeMinverse(model, model.createData(), q))
    data = model.createData()
    pin.crba(model, data, q)
    pin.cholesky.decompose(model, data)
    x = pin.cholesky.solve(model, data, np.array([1.0, 2.0, 3.0]))
    a = np.array([1.0, -2.0, 0.5])
    tau = pin.rnea(model, model.createData(), q, v, a)
    print(f"order {order}, parents {list(model.parents)}, model.check(data) = {model.check(model.createData())}")
    print(f"  crba - mass matrix from rnea:        {np.abs(M_crba - M_rnea).max():.2e}")
    print(f"  computeMinverse - inverse of it:     {np.abs(Minv - np.linalg.inv(M_rnea)).max():.2e}")
    print(f"  cholesky.solve - numpy solve:        {np.abs(x - np.linalg.solve(M_rnea, [1.0, 2.0, 3.0])).max():.2e}")
    print(f"  aba(rnea(a)) - a:                    {np.abs(pin.aba(model, model.createData(), q, v, tau) - a).max():.2e}")
```

Output:

```
Pinocchio 4.1.0
order ACB, parents [0, 0, 1, 0], model.check(data) = True
  crba - mass matrix from rnea:        2.22e-16
  computeMinverse - inverse of it:     1.39e-16
  cholesky.solve - numpy solve:        4.44e-16
  aba(rnea(a)) - a:                    4.44e-16
order ABC, parents [0, 0, 0, 1], model.check(data) = False
  crba - mass matrix from rnea:        5.48e-01
  computeMinverse - inverse of it:     2.12e-01
  cholesky.solve - numpy solve:        6.80e-01
  aba(rnea(a)) - a:                    2.22e-16
```

The mass matrix of `computeAllTerms` and of `computeRNEADerivatives` is
wrong in the same way (same 5.48e-01).

**Why:** `crba` writes each row of the matrix as one block of
`data.nvSubtree[i]` columns starting at the joint's own velocity index
(`src/algorithm/crba.hxx`, lines 92 and 298 in 4.1.0), which assumes the
subtree's velocities are contiguous. `checkData` verifies exactly that, and
`CRBAChecker` checks the same property of the model.

**Suggestions,** any one of which would have saved us the search:

1. Say it in the Python docstrings of `Model.addJoint` (and wherever the
   C++ remark is not visible from Python).
2. Check it in `addJoint`, in every build. It is cheap: the order stays
   depth first exactly when the new joint's parent is the last joint added
   or one of that joint's ancestors (the world included), a walk up at most
   the tree's depth. An exception there names the problem where it is made.
3. Or check it with `PINOCCHIO_CHECK_INPUT_ARGUMENT` (not only `assert`) at
   the start of the algorithms that rely on it, or once when `Data` is
   built.

We have not looked at the development branch, so this may already have
changed. Thank you again.
