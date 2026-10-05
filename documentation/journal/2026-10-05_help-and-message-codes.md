# Help folder and message codes

- **Dates:** 5 October 2026
- **Plan:** documentation track H, tasks H.1 and H.2; decision D23
- **Commits:** see the index
- **Written:** at the time

## Goal

- **H.1** A help folder with an index from which every existing document can
  be reached.
- **H.2** A stable code on every error, warning and `validate()` message,
  each explained in a catalogue. *Done when a test that scans the sources and
  the catalogue finds every code used documented, and every documented code
  used.*

These came first in Phase 3, before Phase 3 adds more messages, so that every
new message is written with its code from the start.

## What was done

1. **Inventory.** 45 throw sites, 2 warnings and 25 `validate()` findings:
   about 70 messages.
2. **One code per kind of problem, not per line.** The three parameter checks
   of a tyre share one code, and the message names the parameter. That keeps
   the catalogue readable: 53 codes. Areas by letter: K for the kernel, M for
   the findings of `validate()`, F for force elements, A for analysis; numbers
   grouped in tens by topic within an area (K00x arguments, K01x building a
   model, K02x constraints, K03x the solver, K04x the simulator, K05x
   warnings), leaving room to insert.
3. **Applying them without retyping messages.** A table of 37 unique message
   literals and their codes, applied by a script that prefixes each literal
   and reports any literal found zero times or more than once (none were).
   The few messages built from shared fragments (the solver's and the
   simulator's reference checks, the two warnings, the three argument checks)
   were edited by hand; the 25 `validate()` findings by line number, checked
   against the listing.
4. **The catalogue**, `docs/help/messages.md`: per code, the message as
   printed, what it means, the usual causes and what to do. It also says how
   errors and warnings reach the user (an exception, or the diagnostic sink
   that writes to standard error unless logging is set up).
5. **Two tests** (`tests/core/test_message_catalogue.cpp`): the codes in the
   sources and the headings of the catalogue must be the same set; and every
   statement that throws, warns or adds to a `validate()` report must carry a
   code before its semicolon. The second test is what makes the rule hold for
   code not yet written. The tests find the repository from their own path
   (`__FILE__`) rather than from a compile definition, which could conflict
   with the shared precompiled header.
6. **The help index**, `docs/help/README.md`: getting started, concepts,
   how-to guides (until written, a table pointing to the tests that work as
   examples), troubleshooting (planned, H.3), the catalogue, the reference,
   and the development record.

## Decisions

- The code goes at the start of the message (`MBD-K031: kernel::...`), as
  compilers do, so it is the first thing seen and easy to search for.
- One code for a kind of problem; the specific parameter or body is in the
  message text.

## Verification

The two catalogue tests; the existing message tests still pass because they
match fragments (one expectation that included the beginning of a summary
line was updated to include its code).

## For the book

- A message catalogue checked by a test in both directions is a small idea
  with a large effect: documentation that cannot fall behind the code.
- Messages as the program's first help: written for the person who will read
  them, naming the body or constraint at fault.
