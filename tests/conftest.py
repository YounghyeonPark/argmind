# SPDX-License-Identifier: Apache-2.0
"""Options shared by the suite.

`--regen-sizes` exists because `cpp/target/SIZES.md` told readers to use it and it did not
exist. That file is the one this repository created *because* a hand-copied size table went
stale three times, so an instruction in it that does not work is the same failure returning
by a different door.
"""


def pytest_addoption(parser):
    parser.addoption(
        "--regen-sizes", action="store_true", default=False,
        help="rewrite cpp/target/SIZES.md from a fresh link instead of comparing against "
             "it. The size test still runs afterwards, so a rewrite that does not match "
             "is still a failure.")
