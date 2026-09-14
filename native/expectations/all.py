"""Generate every expectation the Rust crate is checked against.

**This is the only thing Python is still for.** A port needs both implementations while it is a
port and not afterwards: these files are produced by the shipped Python, the Rust matches them, and
when the Python goes the golden data stays as data. Nothing here ships.

    python native/expectations/all.py
    cargo test --manifest-path native/Cargo.toml
"""
import pathlib
import runpy
import sys

HERE = pathlib.Path(__file__).resolve().parent
for s in (sys.stdout, sys.stderr):
    try:
        s.reconfigure(encoding="utf-8")
    except Exception:
        pass

for name in ("prompt", "reply", "loop", "session", "wire", "measure", "repair"):
    print(f"-- {name}")
    runpy.run_path(str(HERE / f"{name}.py"), run_name="__main__")
