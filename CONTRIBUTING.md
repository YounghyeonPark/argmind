# Contributing

argmind is one person's side project, not a product. Issues are welcome, in Korean or English.
Small fixes are welcome as pull requests. For anything larger, open an issue first, because the
direction is the author's to set and a large change may not be taken however good it is.

## Reporting a bug

What helps most:

- **What you said** (typed or spoken), word for word.
- **What it did**: the lines of 한 일 (*Activity*) for that turn. That timeline already records what
  it heard, which window was in front, what it said, and what it pressed and whether the screen
  changed.
- **Windows version, GPU, and the model names** shown in the header.

**Check before you paste.** The timeline holds window titles, and a window title can be an email
subject or a file name. Remove what you would not post publicly.

For anything that lets something other than you make argmind act, see [SECURITY.md](SECURITY.md)
and report it privately instead.

## Changing the code

    cargo test --manifest-path native/Cargo.toml
    git config core.hooksPath .githooks    # once: the pre-commit hook runs the suite

The suite needs no display, model, network, microphone or speaker. The rules a change has to keep
are in `CLAUDE.md`. The ones most often hit:

- **A model never emits a coordinate.** The program finds widgets; the model picks one by name.
  `argmind-rules` checks this by reading the source.
- **Only permissive licences.** A new dependency fails the suite until it is recorded in
  `docs/licences.toml` with a licence from its `allowed` list. No GPL, AGPL or LGPL.
- **No GUI test drives the real pointer or keyboard**, and no test clicks anything the test did not
  start itself.
- **A number is labelled measured, computed or estimated**, with its denominator and conditions
  beside it. A change to a prompt is measured on the harnesses in `native/argmind-measure/` before
  it is kept. Most prompt edits here have not moved anything, and that is recorded rather than
  hidden.

By contributing you agree that your contribution is licensed under Apache-2.0, the licence of
this repository.
