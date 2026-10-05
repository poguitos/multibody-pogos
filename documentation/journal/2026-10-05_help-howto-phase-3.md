# Help: how-to guides for Phase 3

- **Dates:** 5 October 2026
- **Plan:** task H.6 at the Phase 3 gate; decision D23
- **Commits:** this task's commit (the index gives the hash)
- **Written:** at the time, in a cloud session

## Goal

Each phase adds the how-to pages of what it delivers, with examples that are
compiled and run as tests so they cannot go stale. *Done at each phase
gate.* This is Phase 3's share.

## What was done

1. **Examples as tests** (`tests/examples/`, executable `test_examples`, 11
   cases): a spring-damper with a tabulated curve and a bump stop, settled by
   statics; a torsion spring with friction and limits; a driven joint and its
   drive torque; a four-bar's motion ratios by kinematic analysis; a
   recorder with a pendulum's angle, hinge force and energy; the assembly of
   a four-bar from approximate angles; the sedan settled; a quarter car's
   modes; a box sliding to rest on the ground; a bouncing ball by events; a
   run traced and diagnosed. Each checks its own result (the spring carries
   the weight, the drive torque equals `I q'' + m g l cos q`, the quarter car's
   frequencies, the box's static depth, the first impact's time, ...), so an
   example that stops working fails the build.
2. **Nine pages** in `docs/help/howto/`, one task each, quoting the marked
   region of an example (`// [name]` lines in the source,
   `<!-- example: file#name -->` on the page). The quotes were filled by a
   script from the sources, so they started identical.
3. **The guard**: `tests/core/test_howto_examples.cpp` reads every page,
   finds each marker, and compares the following code block with the
   source's region less its common indentation. Changing one word of a page
   (a restitution of 0.8 to 0.9 in the events page) made it fail; restoring
   it, pass.
4. The help index points to the pages; the earlier table of tests as worked
   examples keeps only what has no page yet.

## What went wrong, and how it was found

- **Starting points far from any closed loop.** The first drafts of the
  four-bar examples started from angles guessed without computing the
  closure (2.2 and 1.8 rad), far from the configuration near the crank angle
  of 1 rad. Computing the closure by hand before building gave about (-0.6,
  -2.3); the examples use that, as a reader would after the assembly page.
- **An empty loop in an example.** A loop over the event log with only a
  comment in its body would have left an unused variable; it now prints each
  event, which is also more useful to a reader.

## Verification

All 11 examples pass; the guard passes and fails as it should. 438 tests
pass with GCC 13 (426 before: 11 examples and the guard).

## Open ends

- How-to pages for what Phase 2 delivered and has no page yet (a mechanism
  with loops, a vehicle, a suspension sweep, performance): the help index
  lists the tests that show them, until Phase 8's builder changes how
  vehicles are made.
- Each later phase adds its pages at its gate.

## For the book

- Documentation that cannot go stale: examples compiled as tests and quoted
  verbatim, with a test of the quotes. The engineering-practice chapter.
