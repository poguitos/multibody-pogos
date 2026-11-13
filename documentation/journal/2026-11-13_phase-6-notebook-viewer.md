# Phase 6: Rerun's viewer inside notebooks

- **Dates:** 13 November 2026
- **Plan:** added at the author's request at the Phase 6 gate; D59
- **Commits:** see git log
- **Written:** at the time

## Goal

After task 6.6 the author asked for the viewer inside the notebook ("Go
ahead with the merge and the in notebook viewer"), where 6.5 and 6.6 opened
it in a window of its own. No "done when" was set; the work counts as done
when a replay in a notebook shows the viewer below its cell, the choice of
place is tested, and the documented install is tried as written.

## Starting point

Rerun shows its viewer in a notebook through `rerun-notebook`, on PyPI only;
conda-forge has none (task 6.6's search). Both of the project's environments
already had what it needs (anywidget, ipywidgets, jupyter-ui-poll, ipykernel,
which came with rerun-sdk). `replay` and `plot` either saved a recording or
spawned the viewer.

## What was done

1. **`rerun-notebook` 0.43.0 read before it was installed.** A dry run of
   pip showed it would add only itself. Its wheel, unpacked in a folder of
   its own and only read, is 18 MB; its viewer, `re_viewer_bg.wasm`, is
   70,282,095 bytes. Its widget loads that viewer from
   `https://app.rerun.io/version/0.43.0/notebook/widget.js` by default
   (answered 200, `text/javascript`), from a server it starts on the machine
   with `RERUN_NOTEBOOK_ASSET=serve-local`, or inline through the notebook's
   connection. It waits up to 10 s for the viewer to say it is ready and
   queues the data until then. rerun-sdk's extra `notebook` pins
   `rerun-notebook==0.43.0`.
2. **Installed in `mbd-try` as the author would:** a scratch `.bat` that
   activates Miniforge, then `mbd-try`, then runs the documented `pip
   install rerun-notebook==0.43.0`. It printed `Successfully installed
   rerun-notebook-0.43.0`, and the check line `rerun-notebook 0.43.0
   rerun-sdk 0.43.0`.
3. **`_viewer._stream`:** with `path`, a file; in a Jupyter kernel with
   `rerun-notebook`, `notebook_show(width="auto", height="auto")`; else, or
   with `spawn=True`, the window; in a notebook without `rerun-notebook`,
   the window and the warning MBD-P016 (D59). `plot` does the same.
4. **A test** replaces Rerun's `notebook_show` and `spawn` with recorders
   that bind each call to the real function's signature, then checks the
   place chosen in each case and the warning.
5. **A real kernel:** the pendulum notebook run through `jupyter_client` in
   `mbd-try`, `replay` left alone. The replay cell sent two widgets (Rerun's
   error panel, shown only if loading fails, and a box with the loading
   message and `rerun_notebook.Viewer`) and finished; no exception.
6. **Text:** the notebooks' replay cells, `docs/python.md` (where the viewer
   appears, the first load from Rerun's site, offline use, the command that
   makes the author's environment), the catalogue (P016), troubleshooting (a
   blank viewer; a "Viewing" section now holds the viewer's entries), the
   extra `notebooks` (`rerun-sdk[notebook]`).

## Decisions

D59: the place from where the code runs, with no new parameter; Rerun's
default source for the viewer; `rerun-notebook` in the author's environment
and the extra, not in the suite's.

## What went wrong, and how it was found

- **An unmeasured number in the first draft of D59:** the inline option's
  size, written as "about 25 MB". Measured before the commit: the viewer
  gzipped is 17.8 MB, 23.7 MB once encoded as the inline option sends it.
- **Rerun's warning after 10 s does not reach the notebook:** it goes to
  Python's logging, which the kernel does not show. Found in the kernel run,
  which printed no timeout although no front end answered. The
  documentation says the cell waits up to 10 s, from the package's code.

## Verification

- `test_viewer.py`, the new case: outside a notebook a window; in a
  notebook with `rerun-notebook` the viewer below the cell, with
  width and height "auto" and the layout; with `spawn=True` a window; in a
  notebook without `rerun-notebook` a window and MBD-P016, naming
  `rerun-notebook==0.43.0`; the window started from Rerun's own binary.
  `_in_notebook()` is false under pytest.
- The kernel run above.
- **Not verified:** the viewer drawing inside VS Code; the assistant has no
  display. If it stays blank, `spawn=True` gives the window
  (troubleshooting).

## Measurements

In `data/2026-11-13_phase-6-notebook-viewer.md`.

## Lessons

- **Read a package before installing it** when it comes from outside the
  environment's usual channel: here its code said where the viewer comes
  from, how long a cell waits, and that data is queued, all of which the
  documentation now states.

## Open ends

- How the viewer behaves in VS Code: for the author to see.
- `serve-local` as the default, if the author prefers nothing fetched from
  Rerun's site.

## For the book

Chapter on the Python interface (6): a viewer that is a web program inside
the notebook, and what that brings (a 70 MB download, a network
dependency, a wait for readiness).
