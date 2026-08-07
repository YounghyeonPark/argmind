# SPDX-License-Identifier: Apache-2.0
"""What generating the program found, kept in code next to the generator.

Implementing descriptor v0 found seven gaps and they are recorded in
:mod:`argmind.descriptor` as ``V0_LIMITATIONS`` / ``V0_1_RULINGS``. *Generating*
the same program rather than hand-writing it exercises the format harder -- every
operand format is derived from a buffer rather than chosen to fit, every
dependency is emitted from the dataflow rather than annotated -- and it found
three more things. Two are limits of v0.1, one is a mislabelled number in a
settled record.

Kept as data, and pinned by tests, for the same reason the v0 record is: a
finding in prose drifts away from the thing it is about.
"""

from __future__ import annotations

#: Gaps in descriptor v0.1 that only appear when a code generator, rather than a
#: human, assigns operand formats. Each entry names the mechanism, the measured
#: cost of the workaround, and what closing it would take.
V0_1_LIMITATIONS_FOUND_BY_CODEGEN: tuple[dict[str, str], ...] = (
    {
        "id": "G1",
        "severity": "forces a storage decision the numerics did not ask for",
        "what": "The elementwise opcodes (AXPY, SCAL, VMUL, PROJ) carry exactly "
                "ONE format, not two, because they have no bulk operand and so "
                "never read prec_m. A mixed-format elementwise operation is "
                "inexpressible.",
        "found_by": "Assigning formats per BUFFER and deriving the descriptor "
                    "fields. In the fp16-operand mode the setup block's `K += P` "
                    "is an fp32 destination and an fp16 source, and there is no "
                    "encoding for it. A human choosing descriptor fields directly "
                    "would have stored P at fp32 without noticing there was a "
                    "decision.",
        "measured_cost": "P must be stored at the factor's format. On QUAD-MPC "
                         "that is 12800 B of extra scratchpad (25600 at fp32 "
                         "instead of 12800 at fp16) and the P*x GEMV runs at 205 "
                         "cycles instead of 105, i.e. +100 of ~2800 cycles per "
                         "checked iteration at fp16. The measured fp16 working "
                         "set is 114640 B = 112.0 KiB, against the 101.3 KiB the "
                         "scratchpad ruling costed for an all-fp16 layout -- and "
                         "that 101.3 KiB figure assumed a P the format cannot "
                         "store.",
        "workaround_that_does_not_help": "`K += P` can be expressed as a GEMM "
                                        "with accumulate against an identity "
                                        "matrix, which IS two-format legal. It "
                                        "costs an n x n identity buffer of "
                                        "exactly the size the narrow P saves, "
                                        "plus n^3 MACs once per problem. Net "
                                        "zero on capacity and negative on time.",
        "fix": "Let the elementwise opcodes read `b` at prec_m and write `c` at "
               "prec_v. Zero new bits -- prec_m is already in w0 and already "
               "ignored by these opcodes. It is a SEMANTIC change to the format, "
               "so it is a ruling to be made, not an implementation detail: it "
               "changes what an existing bit pattern means for four opcodes.",
        "status": "REPORTED, NOT IMPLEMENTED. The generator works around it and "
                  "says so in its report.",
    },
    {
        "id": "G2",
        "severity": "informational, but it changes a stated requirement",
        "what": "The 'about 6 distinct kernel shapes per ADMM iteration' figure "
                "that the descriptor-emit cost analysis rests on is too low. The "
                "emitted iteration block has 15 distinct shapes over 16 "
                "descriptors, and the whole solve has 24.",
        "found_by": "Counting templates on the emitted program instead of "
                    "estimating them from the kernel decomposition. Almost every "
                    "descriptor in an ADMM iteration is its own shape: the "
                    "dimensions alternate between n and m, and the alpha "
                    "immediate differs on nearly all of them.",
        "measured_cost": "Per-iteration arming costs 16.7 cycles/descriptor "
                         "(17.0 in reference-equivalent mode), not the 8.25 the "
                         "analysis quoted -- 2.0x worse and, like 8.25, above the "
                         ">=8-cycle bar. The CONCLUSION is unchanged and in fact "
                         "strengthened: arming must be hoisted out of the "
                         "iteration loop, which is legal because allocation is "
                         "static and every iteration's descriptors are "
                         "bit-identical. Hoisted, the cost is 1.5 "
                         "cycles/descriptor over a 12-iteration solve -- 5.3x "
                         "inside the bar, and it falls further on longer solves.",
        "hardware_consequence": "The generator must hold 24 armed templates, not "
                                "the >=6 the emit analysis concluded and not the "
                                "4 that perf_model/sequencer.py instantiates in "
                                "the area budget. At 192 descriptor bits x 6 "
                                "gates/bit that is 1152 gates per template, so 4 "
                                "-> 24 costs about 0.006 mm2 -- 0.8% of the "
                                "0.786 mm2 core. Cheap, but it is a number in the "
                                "area model that is wrong.",
        "fix": "Raise N_DESC_TEMPLATES. Alternatively, templates that differ only "
               "in the w7 alpha immediate could share a slot and re-arm one word, "
               "which would cut the count; the area saving does not obviously "
               "justify the control logic.",
        "status": "REPORTED for the architect to rule on.",
    },
    {
        "id": "G3",
        "severity": "a flattering number in a settled record",
        "what": "SEQUENCER_RECHECK states 238.3 datapath cycles per descriptor at "
                "fp32 and 474.1 at fp64, i.e. exactly 2x apart. Recounted from "
                "the emitted descriptors the figures are ~137 at fp32 and ~491 at "
                "fp64 -- 3.6x apart, not 2x.",
        "found_by": "Costing each emitted descriptor individually instead of "
                    "scaling one precision's total by element size.",
        "why": "fp32 and fp64 GEMV are limited by DIFFERENT RESOURCES on the "
               "CM-100. At fp32 the ceiling is delivered bandwidth, 128 B/cycle / "
               "4 B = 32 MAC/cycle. At fp64 the bandwidth ceiling would be 16 but "
               "the merged FMA cluster is 8, so fp64 GEMV is ARRAY-bound. Halving "
               "an fp64 cycle count to get an fp32 one therefore understates the "
               "rate change by 2x, and in the flattering direction.",
        "consequence": "The margin over the >=8-cycle bar at fp32 is 17.2x, not "
                       "29.8x. The verdict does not change and nothing rests on "
                       "it. What DOES matter is the other half of the record: the "
                       "'run of six consecutive descriptors averaging 5.8 cycles' "
                       "reproduces EXACTLY from the generated program at fp32 "
                       "(5.83), so the record pairs an fp64-derived average with "
                       "an fp32-derived worst case.",
        "additional": "At fp64 the emitted program has NO sub-threshold "
                      "descriptor at all -- the thinnest point in the sequencer "
                      "analysis exists only at fp32 and narrower. The "
                      "queue-depth clause is a property of the precision, not "
                      "just of the algorithm.",
        "status": "CORRECTION for the record. Pinned by a test.",
    },
    {
        "id": "G4",
        "severity": "two encodings, one meaning, no error",
        "what": "The `accumulate` bit is only defined for GEMM, GEMV and VMUL. "
                "SCAL, TRSV, PROJ, DOT, NRMI and CHOL silently ignore it, so a "
                "descriptor with the bit set decodes identically to the same "
                "descriptor without it.",
        "found_by": "A mutation test. Turning the ADMM dual update into its "
                    "known-wrong expanded form (`y += rho*(t-z)`) by setting "
                    "accumulate on the SCAL that writes y produced NO CHANGE IN "
                    "BEHAVIOUR, because the SCAL kernel does not read the bit. "
                    "The mutation was silently discarded, which is the same "
                    "mechanism by which a real codegen bug would be.",
        "why_it_matters": "The format already refuses an undefined encoding "
                          "elsewhere -- w0[1:0] reserved bits must be zero, on "
                          "the stated ground that refusing rather than ignoring "
                          "is what makes a future assignment possible. The same "
                          "argument applies here and the field is 1 bit.",
        "fix": "Refuse `accumulate` in the Descriptor constructor for opcodes "
               "that do not define it. Pure validation -- it rejects encodings "
               "that currently have no meaning, so it cannot change the meaning "
               "of any descriptor a correct generator emits.",
        "status": "REPORTED for the format owner. The GENERATOR refuses it "
                  "already (argmind.codegen.ir.ACCUMULATE_DEFINED_FOR), so no "
                  "generated program can contain the meaningless encoding; the "
                  "shared descriptor module is left alone because tightening "
                  "validation there is a format decision.",
    },
)

#: A hazard in the RUNTIME residency mechanism, not in the descriptor format.
#: Found by the residency planner's cross-check and escalated rather than fixed:
#: the meaning of the dirty mask is part of the runtime contract.
RESIDENCY_MECHANISM_HAZARD: dict[str, str] = {
    "id": "G5",
    "severity": "SILENT WRONG ANSWER, reachable through the supported API",
    "what": "Two mechanisms decide two different things and they can disagree. "
            "The DIRTY MASK decides what is re-fetched. The GENERATION TAG decides "
            "whether the setup block re-runs. Naming a setup input in the dirty "
            "mask without bumping its generation refreshes the OPERAND and leaves "
            "the FACTORISATION DERIVED FROM IT stale.",
    "how_it_is_reached": "A host writes P through the raw pointer -- which is "
                         "legal, it is a pointer, and the runtime cannot detect "
                         "it -- and then names P in the call's dirty mask, which "
                         "is the careful thing to do. P is re-fetched; the "
                         "Cholesky factor of the OLD P stays resident; the solve "
                         "converges to the wrong answer without any error.",
    "why_it_survived": "The dirty mask is documented as 'belt and braces, not the "
                       "only mechanism' for FETCHING, and that is true. Nothing "
                       "said it also has to invalidate DERIVED data, because "
                       "before there was a setup block there was no derived data.",
    "fix": "One line: BuiltSolver.setup_needed() must also return True when any "
           "setup input is named in the call's dirty mask. It is conservative -- "
           "it can only cause more setup runs, never fewer.",
    "status": "ESCALATED, NOT FIXED. The dirty mask's meaning is part of the "
              "runtime contract, so widening it is a business decision. Generated "
              "code cannot hit it (the generated host API writes through "
              "Buffer.write, which bumps the generation), and a test pins the "
              "current behaviour so that fixing it is a visible change.",
}

#: The recount behind G3, so a test can pin it rather than trusting the prose.
#: Cycles per descriptor over a checked QUAD-MPC iteration (n=80, m=320,
#: 31 descriptors, reference-equivalent mode) on the CM-100 datapath model.
SEQUENCER_RECHECK_CORRECTION: dict[str, object] = {
    "recorded": {"fp32": 238.3, "fp64": 474.1, "ratio": 474.1 / 238.3},
    "recounted": {"fp32": 137.5, "fp64": 490.7, "ratio": 490.7 / 137.5},
    "tolerance": 0.05,
    "why_the_ratio_is_not_2": (
        "fp32 GEMV is bandwidth-bound at 128/4 = 32 MAC/cycle; fp64 GEMV is "
        "array-bound at the 8-unit FMA cluster, below its own 128/8 = 16 "
        "bandwidth ceiling. The rate ratio is 4x, not 2x."),
    "verdict": "unchanged -- 17.2x the bar instead of 29.8x",
    "short_run": {"length": 6, "mean_cycles_fp32": 5.83,
                  "recorded_mean": 5.8,
                  "note": "reproduces exactly; it is an fp32 figure, unlike the "
                          "average it is quoted beside"},
}

#: Things the residency planner could NOT derive and had to be told. Exactly one
#: entry, which is the useful result.
RESIDENCY_HINTS_REQUIRED: tuple[dict[str, str], ...] = (
    {
        "id": "H1",
        "hint": "Which QP data blocks change from call to call "
                "(`QPProblem.varies`).",
        "why_not_derivable": "It is a statement about the FUTURE, not about the "
                             "arrays. Two problems with identical P and A differ "
                             "in whether P is about to change, and a generator "
                             "that guessed 'P is probably constant' would return "
                             "a stale answer the first time a user retuned a "
                             "weight.",
        "already_available_from": "A CVXPY Parameter IS this declaration -- data "
                                 "that varies between solves is a Parameter and "
                                 "data that does not is a constant, which is what "
                                 "DPP means. The CVXPY front end derives `varies` "
                                 "by canonicalising at two parameter values and "
                                 "differencing, so a user coming in that way "
                                 "supplies no hints at all.",
        "everything_else": "Derived from the emitted dataflow with no hints: the "
                           "16 CM-internal temporaries (first access is a write), "
                           "the write-back set, the setup-skip condition, and the "
                           "per-call byte counts -- which match the measured "
                           "call statistics exactly.",
    },
)

__all__ = [
    "RESIDENCY_HINTS_REQUIRED",
    "RESIDENCY_MECHANISM_HAZARD",
    "SEQUENCER_RECHECK_CORRECTION",
    "V0_1_LIMITATIONS_FOUND_BY_CODEGEN",
]
