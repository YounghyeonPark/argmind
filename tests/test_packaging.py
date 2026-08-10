# SPDX-License-Identifier: Apache-2.0
"""The library has to be consumable, not just buildable.

Until 2026-08-10 it was not. There were no install rules, no export set and no
`argmindConfig.cmake`, so the only way to use this from another project was to copy the
sources in. That is defensible for something nobody else builds and indefensible for
something the scope now calls a platform.

So this test does the whole thing a consumer does — configure, build, install to a
prefix, then configure a *separate* project that only knows `find_package(argmind)`,
build it and run it. Anything less would check that the install rules parse rather than
that they work, and the difference is where packaging bugs live.

Skipped, not failed, without CMake or a generator.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
CPP = REPO / "cpp"


def _run(cmd, **kw):
    return subprocess.run([str(c) for c in cmd], capture_output=True, text=True, **kw)


@pytest.fixture(scope="session")
def cmake():
    exe = shutil.which("cmake")
    if exe is None:
        pytest.skip("no cmake on PATH")
    if shutil.which("gcc") is None:
        pytest.skip("no host compiler")
    return exe


def _generator():
    """MinGW Makefiles on Windows; the default elsewhere. Named explicitly because
    CMake's Windows default is Visual Studio, which is not the compiler the rest of this
    suite uses and would test a different build than the one that ships."""
    return ["-G", "MinGW Makefiles"] if shutil.which("mingw32-make") else []


CONSUMER_CMAKE = """cmake_minimum_required(VERSION 3.20)
project(consumer LANGUAGES C)
find_package(argmind 0.1 REQUIRED)
add_executable(consumer main.c)
target_link_libraries(consumer PRIVATE argmind::argmind)
"""

CONSUMER_C = """#include <stdio.h>
#include "argmind/argmind.h"

/* The version policy, enforced. A consumer that expected a different major gets a
   compile error here rather than a link-time surprise. */
AM_VERSION_REQUIRE(0, 1);

static am_rls_storage obj[AM_RLS_F32_6_1_MAX_BYTES / sizeof(am_rls_storage)];

int main(void) {
    if (am_rls_f32_6_1_init(obj, sizeof(obj), 0.999f, 100.0f) != AM_OK) return 1;
    float x[6] = {1, 2, 3, 4, 5, 1};
    float y[1] = {7};
    float out[1];
    am_report r;
    for (int k = 0; k < 50; ++k) {
        if (am_rls_f32_6_1_update(obj, x, y, &r, 0) != AM_OK) return 2;
    }
    if (am_rls_f32_6_1_predict(obj, x, out) != AM_OK) return 3;
    printf("prediction %.4f updates %u\\n", (double)out[0],
           am_rls_f32_6_1_updates(obj));
    return 0;
}
"""


def test_an_unrelated_project_can_find_package_and_link_it(cmake, tmp_path):
    """The whole consumer path, end to end. It converged to 7.0 when this was written,
    which is the point: an install that produces a linkable but broken library would
    pass a link-only check."""
    prefix, build = tmp_path / "prefix", tmp_path / "build"
    gen = _generator()

    done = _run([cmake, "-S", CPP, "-B", build, "-DCMAKE_BUILD_TYPE=Release",
                 f"-DCMAKE_INSTALL_PREFIX={prefix}", *gen])
    assert done.returncode == 0, f"configure failed:\n{done.stdout}\n{done.stderr}"
    done = _run([cmake, "--build", build, "--target", "argmind", "-j", "4"])
    assert done.returncode == 0, f"build failed:\n{done.stdout}\n{done.stderr}"
    done = _run([cmake, "--install", build])
    assert done.returncode == 0, f"install failed:\n{done.stdout}\n{done.stderr}"

    # The package files a consumer actually needs. Checked by name because a missing
    # export set makes `find_package` fall through to a confusing "could not find".
    cfg = prefix / "lib" / "cmake" / "argmind"
    for want in ("argmindConfig.cmake", "argmindConfigVersion.cmake",
                 "argmindTargets.cmake"):
        assert (cfg / want).is_file(), f"{want} was not installed"
    assert (prefix / "include" / "argmind" / "argmind.h").is_file()
    assert (prefix / "include" / "argmind" / "nlms.hpp").is_file(), (
        "the headers are installed by directory, so a new one must arrive automatically")

    consumer = tmp_path / "consumer"
    consumer.mkdir()
    (consumer / "CMakeLists.txt").write_text(CONSUMER_CMAKE, encoding="utf-8")
    (consumer / "main.c").write_text(CONSUMER_C, encoding="utf-8")

    done = _run([cmake, "-S", consumer, "-B", consumer / "b",
                 f"-DCMAKE_PREFIX_PATH={prefix}", *gen])
    assert done.returncode == 0, f"the consumer cannot find argmind:\n{done.stderr}"
    done = _run([cmake, "--build", consumer / "b"])
    assert done.returncode == 0, f"the consumer does not build:\n{done.stderr}"

    exe = next((p for p in (consumer / "b").rglob("consumer*")
                if p.is_file() and p.suffix in ("", ".exe")), None)
    assert exe is not None, "the consumer built but produced no executable"
    done = _run([exe])
    assert done.returncode == 0, f"the consumer ran and failed: {done.stdout}"
    assert "prediction 7.0000" in done.stdout, done.stdout
    assert "updates 50" in done.stdout, done.stdout


def test_the_version_policy_refuses_a_consumer_expecting_a_different_major(cmake,
                                                                          tmp_path):
    """`AM_VERSION_REQUIRE` has to fail, or it is a comment with parentheses.

    Compiled directly rather than through CMake: what is under test is the macro, and
    routing it through a build system would only add ways for the test to pass for the
    wrong reason."""
    src = tmp_path / "bad.c"
    src.write_text('#include "argmind/argmind.h"\n'
                   'AM_VERSION_REQUIRE(1, 0);\n'
                   'int main(void) { return 0; }\n', encoding="utf-8")
    done = _run([shutil.which("gcc"), "-std=c99", "-I", CPP / "include",
                 "-c", src, "-o", tmp_path / "bad.o"])
    assert done.returncode != 0, "a consumer expecting major 1 compiled against 0.x"

    ok = tmp_path / "ok.c"
    ok.write_text('#include "argmind/argmind.h"\n'
                  'AM_VERSION_REQUIRE(0, 1);\n'
                  'int main(void) { return 0; }\n', encoding="utf-8")
    done = _run([shutil.which("gcc"), "-std=c99", "-Wall", "-Wextra", "-Wpedantic",
                 "-I", CPP / "include", "-c", ok, "-o", tmp_path / "ok.o"])
    assert done.returncode == 0, (
        f"the matching version does not compile, so the macro refuses everything:\n"
        f"{done.stderr}")


# -- the README is a claim about the repository, so it gets checked ------------------

README = REPO / "README.md"


def test_every_link_in_the_readme_points_at_something():
    """A front page with dead links is worse than none: it asserts a file exists."""
    import re
    missing = [p for p in re.findall(r"\]\(([^)#]+)\)", README.read_text(encoding="utf-8"))
               if not p.startswith("http") and not (REPO / p).exists()]
    assert not missing, f"README links to files that do not exist: {missing}"


def test_the_readme_c_example_compiles_against_the_real_header(tmp_path):
    """The snippet on the front page has to build.

    Checking it by grepping the header for the entry points does not work and was tried:
    they are generated by macros, so a plain search reports every one of them missing.
    Compiling is the only check that means anything -- the same argument
    `examples/hosted_inference.c` is checked by.
    """
    import re
    gcc = shutil.which("gcc")
    if gcc is None:
        pytest.skip("no host compiler")

    blocks = re.findall(r"```c\n(.*?)```", README.read_text(encoding="utf-8"), re.S)
    assert len(blocks) == 1, f"expected one C block in the README, found {len(blocks)}"
    head, _, tail = blocks[0].partition("static am_rls_storage")
    body = "\n".join("  " + line for line in tail.split("\n")[1:])

    src = tmp_path / "snippet.c"
    src.write_text(
        head
        + "static am_rls_storage obj[AM_PIPE_G_F32_6_1_MAX_BYTES"
          " / sizeof(am_rls_storage)];\n"
          "static float iv[32*3]; static double it[32]; static unsigned char ik[32];\n"
          "static float av[8];    static double at[8];  static unsigned char ak[8];\n"
          "static am_ring_f32 rings[2];\n"
          "static float imu[3] = {1,2,3};\n"
          "static double t = 1.0;\n"
          "static void use(double v) { (void)v; }\n"
          "int main(void) {\n"
          "  am_ring_f32_init(&rings[0], iv, it, ik, 3, 32);\n"
          "  am_ring_f32_init(&rings[1], av, at, ak, 1, 8);\n"
        + body + "\n  return 0;\n}\n", encoding="utf-8")

    done = _run([gcc, "-std=c99", "-Wall", "-Wextra", "-I", CPP / "include",
                 "-c", src, "-o", tmp_path / "snippet.o"])
    assert done.returncode == 0, (
        f"the README's example does not compile:\n{done.stderr}")
