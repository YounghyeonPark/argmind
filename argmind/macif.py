"""MACIF -- the Macro-Agnostic Compute InterFace, and its precision contract.

MACIF is the versioned boundary between the sequencer/scratchpad layer and the
multiply-accumulate array, so that a different array implementation (a digital
SRAM-CIM macro, a licensed analog macro) can be substituted without disturbing
code generation, the runtime API, or host integration.

The part of MACIF this module implements is the **precision declaration**, and
the rule it exists to enforce:

    Precision is a declared contract, not an implementation detail. The runtime
    refuses a kernel whose format the macro does not declare. It is a hard error
    at kernel-build time, and it is never resolved by substituting another
    format.

Two refusals matter enough to be called out by name:

* **fp64.** Interior-point KKT systems reach condition numbers around 1e12, so
  fp64 is an algorithm requirement: a macro without it returns a wrong answer,
  not a slower one. fp64 lives in the fixed (non-swappable) part of the compute
  complex for exactly this reason -- no CIM macro has a path to a 53-bit
  significand.
* **fp16 vs bf16.** Same width, so a macro is tempted to treat them as one
  format. On the solver path they are not: mantissa bits dominate, and bf16
  operands cost about 10x the error in the commanded control action that fp16
  operands do. Substitution is refused in both directions.

Per format a macro declares: whether it is supported, an **absolute**
MAC-per-cycle rate (not a scaling factor applied to some other format's rate),
the accumulator width, the rounding mode, subnormal handling, and the two
accumulator features that fold work into the accumulator rather than buying more
array area.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from .errors import PrecisionNotDeclared
from .precision import Format, SIXTEEN_BIT_FORMATS

#: Version of the interface described by this module. Bump on any change to the
#: set of declared attributes; a macro declares which version it speaks.
#:
#: 0.2 adds :attr:`FormatContract.maxabs_accumulate`. The reason is measured, not
#: aesthetic: the ADMM residual block contains a run of six consecutive
#: descriptors averaging 5.8 datapath cycles -- below the >=8 cycles/descriptor
#: codegen bar, and the thinnest point in the whole sequencer analysis. Folding
#: the infinity-norm reduction into the pass that produces the vector takes the
#: residual block from 15 descriptors to 10 and that run from 6 to 4. Same
#: principle as ``broadcast_init`` and ``square_accumulate``: fold work into the
#: accumulator at O(1) area rather than buying array at O(T^2).
MACIF_VERSION = "0.2"


@dataclass(frozen=True)
class FormatContract:
    """What a macro promises about one numeric format.

    Attributes
    ----------
    macs_per_cycle:
        Absolute multiply-accumulates per cycle in this format. Deliberately NOT
        expressed as a fraction of another format's rate: deriving fp64 from a
        multiplier on a bf16 rate is the shape of interface that hides the fact
        that no fp64 datapath exists at all.
    accumulator_width:
        Bits of the accumulator products land in. For a 16-bit operand format
        this is normally 32.
    rounding:
        ``"rne"`` (round to nearest, ties to even) or ``"rtz"``.
    subnormals:
        ``"gradual"`` (IEEE-754 conformant) or ``"flush"`` (flush to zero).
    broadcast_init:
        Accumulator can be initialised from a broadcast scalar or a bias vector,
        folding an add into the matmul instead of costing a separate vector pass.
    square_accumulate:
        Accumulator can accumulate squares of its inputs, folding a **2-norm**
        reduction into the pass that produces the vector.
    maxabs_accumulate:
        Accumulator can carry a running max-absolute alongside the sum, folding an
        **infinity-norm** reduction into the pass that produces the vector.
        MACIF 0.2. Needed because OSQP's stopping rule is stated entirely in
        infinity norms, so ``square_accumulate`` folds the wrong norm for the
        beachhead solver. Cost is a compare-select and one register per
        accumulator lane -- cheaper than ``square_accumulate``, which needs a
        multiplier.
    """

    macs_per_cycle: int
    accumulator_width: int
    rounding: str = "rne"
    subnormals: str = "gradual"
    broadcast_init: bool = False
    square_accumulate: bool = False
    maxabs_accumulate: bool = False

    def __post_init__(self) -> None:
        if self.macs_per_cycle <= 0:
            raise ValueError("macs_per_cycle must be positive; omit the format instead")
        if self.rounding not in ("rne", "rtz"):
            raise ValueError(f"unknown rounding mode {self.rounding!r}")
        if self.subnormals not in ("gradual", "flush"):
            raise ValueError(f"unknown subnormal policy {self.subnormals!r}")


@dataclass(frozen=True)
class MacroCapabilities:
    """A compute macro's declaration. Absence of a format means unsupported.

    ``swappable`` records whether this declaration describes the part of the
    compute complex that sits *behind* MACIF (the tile array, substitutable) or
    the fixed part (the merged FMA cluster carrying fp32/fp64). It is
    informational for the runtime but load-bearing for anyone reasoning about
    which formats can be lost by swapping a macro.
    """

    name: str
    formats: Mapping[Format, FormatContract]
    macif_version: str = MACIF_VERSION
    swappable: bool = True
    notes: str = ""

    # -- queries -----------------------------------------------------------

    def supports(self, fmt: Format) -> bool:
        return Format(fmt) in self.formats

    @property
    def declared_formats(self) -> tuple[Format, ...]:
        return tuple(sorted(self.formats, key=int))

    def require(self, fmt: Format, *, context: str = "kernel") -> FormatContract:
        """Return the contract for `fmt`, or refuse.

        Raises
        ------
        PrecisionNotDeclared
            If the macro does not declare `fmt`. The message names the
            substitution that was NOT made, because a silent substitution is the
            failure this method exists to prevent.
        """
        fmt = Format(fmt)
        contract = self.formats.get(fmt)
        if contract is not None:
            return contract

        detail = [
            f"macro {self.name!r} does not declare {fmt} "
            f"(declared: {', '.join(str(f) for f in self.declared_formats) or 'none'})",
            f"refusing to build {context}.",
        ]
        if fmt in SIXTEEN_BIT_FORMATS:
            other = next(f for f in SIXTEEN_BIT_FORMATS if f is not fmt)
            if self.supports(other):
                detail.append(
                    f"{other} is declared and is the same width, but substituting it "
                    f"is REFUSED: fp16 and bf16 differ by ~10x in the resulting "
                    f"control-action error on the solver path "
                    f"(1.9e-03 vs 1.9e-02 measured). Re-generate for {other} "
                    f"explicitly, with an accuracy report, or use a macro that "
                    f"declares {fmt}."
                )
        if fmt is Format.FP64:
            detail.append(
                "fp64 is an algorithm requirement, not a performance option: "
                "interior-point KKT systems reach cond ~1e12 and a narrower "
                "format returns a wrong answer rather than a slower one."
            )
        raise PrecisionNotDeclared(" ".join(detail))

    def accumulator_width(self, fmt: Format) -> int:
        return self.require(fmt).accumulator_width

    def macs_per_cycle(self, fmt: Format) -> int:
        return self.require(fmt).macs_per_cycle

    def folds_infinity_norm(self, fmt: Format) -> bool:
        """Can an infinity-norm reduction be folded into the producing pass?

        If not, every residual norm costs a separate descriptor and a re-read of
        the vector. On the ADMM residual block that is 5 extra descriptors out of
        15, and they are the short ones -- see MACIF_VERSION's note.
        """
        return self.require(fmt).maxabs_accumulate


def merge(name: str, *macros: MacroCapabilities, notes: str = "") -> MacroCapabilities:
    """Combine declarations, e.g. a swappable tile array plus the fixed FMA cluster.

    A format declared by more than one part is taken from the first macro that
    declares it, and the rates are NOT summed: a caller asking "how many fp32
    MACs per cycle" wants the resource that will actually run its kernel, and
    summing across two independently scheduled units would overstate it.
    """
    formats: dict[Format, FormatContract] = {}
    for macro in macros:
        for fmt, contract in macro.formats.items():
            formats.setdefault(Format(fmt), contract)
    return MacroCapabilities(
        name=name,
        formats=formats,
        swappable=False,          # a merged complex is not wholly swappable
        notes=notes or " + ".join(m.name for m in macros),
    )


# ---------------------------------------------------------------------------
# Reference declarations
# ---------------------------------------------------------------------------
#
# Rates below describe the CM-100 baseline configuration: 144 MACs (12x12) in
# the bf16/int8 tile array, and a merged FMA cluster of 8 multi-format IEEE-754
# units, each 1 x fp64 or 2 x fp32. They are configuration facts, not measured
# silicon.

CM100_TILE_ARRAY = MacroCapabilities(
    name="cm100-digital-tile",
    swappable=True,
    formats={
        Format.INT8: FormatContract(macs_per_cycle=288, accumulator_width=32),
        Format.BF16: FormatContract(macs_per_cycle=144, accumulator_width=32),
    },
    notes="12x12 digital systolic tile, int8 double-pumped. Sits behind MACIF.",
)

#: The tile array with fp16 additionally declared, which is what the solver path
#: needs if 16-bit operands are ever adopted for it: fp16 beats bf16 by ~10x on
#: control-action error for these matrices.
CM100_TILE_ARRAY_FP16 = MacroCapabilities(
    name="cm100-digital-tile-fp16",
    swappable=True,
    formats={
        Format.INT8: FormatContract(macs_per_cycle=288, accumulator_width=32),
        Format.BF16: FormatContract(macs_per_cycle=144, accumulator_width=32),
        Format.FP16: FormatContract(macs_per_cycle=144, accumulator_width=32),
    },
    notes="As cm100-digital-tile, plus a declared fp16 path.",
)

#: The merged vector-unit / fp64 datapath. NOT behind MACIF: it is one
#: scheduling resource giving 16 fp32 lanes or 8 fp64 FMAs, and it is present on
#: every SKU because fp64 is a feature bit rather than a scaling parameter.
CM100_FMA_CLUSTER = MacroCapabilities(
    name="cm100-fma-cluster",
    swappable=False,
    formats={
        Format.FP32: FormatContract(
            macs_per_cycle=16, accumulator_width=32,
            broadcast_init=True, square_accumulate=True, maxabs_accumulate=True,
        ),
        Format.FP64: FormatContract(
            macs_per_cycle=8, accumulator_width=64,
            broadcast_init=True, square_accumulate=True, maxabs_accumulate=True,
        ),
    },
    notes="8 multi-format FMA units: 1 x fp64 or 2 x fp32 each. Fixed, not "
          "swappable. Carries all three accumulator features: this is where the "
          "solver residual norms are reduced, so it is the block on which "
          "maxabs_accumulate has to be present.",
)

#: The CM-100 baseline compute complex as costed.
CM100 = merge("cm100", CM100_TILE_ARRAY, CM100_FMA_CLUSTER)

#: CM-100 with fp16 declared on the array.
CM100_FP16 = merge("cm100-fp16", CM100_TILE_ARRAY_FP16, CM100_FMA_CLUSTER)

#: An inference-oriented CIM macro standing in for the swap case: low-precision
#: only, no fp32, no fp64. Used to demonstrate that the runtime refuses rather
#: than degrades. Note it declares bf16 but NOT fp16 -- the exact trap.
CIM_INFERENCE_MACRO = MacroCapabilities(
    name="cim-inference-macro",
    swappable=True,
    formats={
        Format.INT8: FormatContract(
            macs_per_cycle=4096, accumulator_width=32, subnormals="flush"
        ),
        Format.BF16: FormatContract(
            macs_per_cycle=1024, accumulator_width=32, subnormals="flush"
        ),
    },
    notes="Illustrative low-precision macro behind MACIF; declares no fp32/fp64.",
)

#: Declares every format. For unit tests and for exploring numerics before a
#: configuration is chosen -- not a product configuration.
UNIVERSAL = MacroCapabilities(
    name="universal-reference",
    swappable=False,
    formats={
        fmt: FormatContract(
            macs_per_cycle=1,
            accumulator_width=64 if fmt is Format.FP64 else 32,
            broadcast_init=True,
            square_accumulate=True,
            maxabs_accumulate=True,
        )
        for fmt in Format
    },
    notes="Simulator-only: declares all formats so numerics can be explored.",
)

PRESETS: dict[str, MacroCapabilities] = {
    m.name: m
    for m in (
        CM100,
        CM100_FP16,
        CM100_TILE_ARRAY,
        CM100_TILE_ARRAY_FP16,
        CM100_FMA_CLUSTER,
        CIM_INFERENCE_MACRO,
        UNIVERSAL,
    )
}
