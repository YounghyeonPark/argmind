"""
Argmind CM — the on-device AI envelope: where the two ML regimes divide
======================================================================

PHASE0_FINDINGS §8 says large-model ML is outside the *differentiated* envelope
(SLM-DECODE: 75% DRAM-bound at 25.6 GB/s, 2.6x perf/W) while VIT-TINY is the
highest perf/W row in the whole benchmark set (30.4x). Two regimes, and nobody
had drawn the boundary. This module draws it.

FOUR QUESTIONS, IN ORDER OF HOW MUCH THEY DECIDE:

  1. int4 support, weights-only, as shipping LLM quantisation actually does it.
     ACCURACY IS NOT MODELLED -- see the header of argmind_model.E_MAC_PJ. This
     module says what the datapath can MOVE and COMPUTE at 4 bits, and nothing
     whatsoever about whether the network still works.
  2. The (model size x external bandwidth) plane at int8 and int4, every point
     classified DIFFERENTIATED / FEASIBLE-UNDIFFERENTIATED / INFEASIBLE, with the
     arithmetic-intensity derivation that says WHY the boundary sits where it does.
  3. The weight-resident regime: what fits 192 KB, what a bigger-SRAM ML SKU
     would cost, and whether the capacity finding (solver energy monotone
     INCREASING in capacity) kills it.
  4. The hybrid loop: a learned policy producing a reference for an MPC solver,
     both resident on one engine in one call. Footprint, residency, end-to-end
     tick against the NPU-plus-CPU split a robot ships today.

THE ONE THING TO READ BEFORE ANY perf/W NUMBER BELOW
----------------------------------------------------
EVERY perf/W FIGURE IN THIS MODULE IS AGAINST A CPU BASELINE, AND NOBODY RUNS
EDGE CNN INFERENCE ON A CORTEX-A WHEN THERE IS AN NPU IN THE SoC. The honest
adversary for an ML row is an NPU, and we cannot benchmark against one:
we have no measured NPU silicon data, and our own evidence standard forbids
comparing a modelled figure against a vendor marketing claim. So a 30x ML perf/W
figure is *arithmetically correct against a baseline no customer would choose*.
It is reported because the zone boundary needs a consistent metric, and it is
labelled `vs-CPU` everywhere so it cannot be lifted out of context.
Consequence, stated plainly in §5: the ML differentiation claim is not merely
narrow, it is UNSUPPORTED with present evidence, and the only ML claims that
survive are ones that do not need an inference-efficiency comparison at all.

ALL FIGURES MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE EXTERNALLY.
Precision is stated per row. Evidence classes: [MEAS] measured, [CITE] cited
literature, [EST] estimate, [DER] derived from another model figure.
"""

from __future__ import annotations

import dataclasses as dc
import math

import argmind_model as M
import cm100_sizing as C
import sram_cost as S

C100 = M.CFG_100
KILL = M.KILL_CRITERION           # 5.0x perf/W

# Zone labels. The asterisk on DIFF is load-bearing: it is vs a CPU baseline.
Z_DIFF = "DIFF*"
Z_FEAS = "feasible"
Z_INF = "INFEASIBLE"

# External-bandwidth points swept. 6.4 = one LPDDR4x-2400 x32 channel share (the
# CM-100 assumption); 51.2 = a 4-channel LPDDR5 class part, i.e. the top of what
# an edge SoC in this price band offers.
BW_POINTS = (6.4, 12.8, 25.6, 51.2)

# Measured static footprint of the flagship solver row, KB, from the functional
# simulator's allocation plan at descriptor v0.1 with A = [I;Gamma] exploited.
# [MEAS] -- same number scaling_study.FOOTPRINT_KB carries.
QUAD_FOOTPRINT_KB = 138.8


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def perf_w(b: M.Benchmark, c: M.ArgmindCfg, dram_gbs: float | None = None) -> float:
    """Energy ratio against the most efficient CPU baseline AT THE SAME DRAM BW.

    Passing `dram_gbs` to both sides is not optional. Before 2026-08-05
    `model_cpu` pinned the CPU's DRAM bandwidth to 12.8 GB/s regardless of what
    the CM was given, so a 51.2 GB/s sweep point silently handed the CM 4x the
    CPU's weight-streaming rate on the same SoC.
    """
    g = c.dma_bw_gbs if dram_gbs is None else dram_gbs
    eff = min((M.model_cpu(b, x, dram_gbs=g) for x in M.ALL_CPUS),
              key=lambda r: r["energy_uj"])
    return eff["energy_uj"] / M.model_argmind(b, c)["energy_uj"]


def cpu_best_energy(b: M.Benchmark, dram_gbs: float) -> dict:
    return min((M.model_cpu(b, x, dram_gbs=dram_gbs) for x in M.ALL_CPUS),
               key=lambda r: r["energy_uj"])


def cpu_best_latency(b: M.Benchmark, dram_gbs: float = 12.8) -> tuple:
    pairs = [(x, M.model_cpu(b, x, dram_gbs=dram_gbs)) for x in M.ALL_CPUS]
    return min(pairs, key=lambda p: p[1]["latency_us"])


def zone(b: M.Benchmark, c: M.ArgmindCfg) -> tuple[str, dict, float]:
    """Classify one (workload, configuration) point into the three zones."""
    r = M.model_argmind(b, c)
    pw = perf_w(b, c)
    if r["latency_us"] > b.deadline_us:
        return Z_INF, r, pw
    return (Z_DIFF if pw >= KILL else Z_FEAS), r, pw


def cfg_bw(gbs: float, kb: int = M.CM100_SCRATCHPAD_KB) -> M.ArgmindCfg:
    return dc.replace(C100, dma_bw_gbs_override=gbs, scratchpad_kb=kb,
                      name=f"CM-100 @{gbs:.1f} GB/s, {kb} KB")


# ---------------------------------------------------------------------------
# A transformer size family, so "model size" is a physical axis
# ---------------------------------------------------------------------------
#
# params ~ 12 * layers * hidden^2 for a decoder-only block with a 4x MLP. Holding
# hidden/layers ~ 128 (the aspect ratio real sub-2B models use -- TinyLlama-1.1B
# is 2048/22 = 93, Phi-2 is 2560/32 = 80, Qwen2-0.5B is 896/24 = 37) gives
# hidden = (128 * params / 12)^(1/3). At 1.1e9 that returns hidden 2276 /
# layers 17, against TinyLlama's actual 2048 / 22 -- close enough that the swept
# family is recognisable rather than invented. [EST]
def transformer_shape(params: float) -> tuple[int, int]:
    h = (128.0 * params / 12.0) ** (1.0 / 3.0)
    h = max(64, int(round(h / 64.0)) * 64)
    layers = max(1, int(round(12.0 * h ** 3 / 128.0 / (12.0 * h * h))))
    # Re-solve layers so params comes out right for the ROUNDED hidden.
    layers = max(1, int(round(params / (12.0 * h * h))))
    return h, layers


def slm(params_b: float, rate: float, weight_prec: str = "int8",
        seq: int = 512) -> M.Benchmark:
    """One point on the LLM-decode size axis. `params_b` in billions."""
    h, layers = transformer_shape(params_b * 1e9)
    return M.slm_decode(f"SLM-{params_b:g}G/{weight_prec}", params_b=params_b,
                        hidden=h, layers=layers, seq=seq, rate=rate,
                        note=f"{params_b:g}B decode, batch 1, {rate:g} tok/s, "
                             f"{weight_prec} weights / int8 KV",
                        prec="int8", weight_prec=weight_prec)


# ---------------------------------------------------------------------------
# Learned policies, for the hybrid loop. Real architectures, cited.
# ---------------------------------------------------------------------------
#
# [CITE] Rudin, Hoeller, Reist, Hutter, "Learning to Walk in Minutes Using
#   Massively Parallel Deep Reinforcement Learning", CoRL 2021,
#   arXiv:2109.11978 -- actor MLP [512, 256, 128] with ELU. This is the
#   `legged_gym` default and the most-copied locomotion policy shape in robotics.
# [CITE] Hwangbo, Lee, Dosovitskiy, Bellicoso, Tsounis, Koltun, Hutter,
#   "Learning agile and dynamic motor skills for legged robots", Science
#   Robotics 4(26), 2019, arXiv:1901.08652 -- policy MLP [256, 128, 32],
#   deployed on ANYmal hardware.
# Observation 48 and action 12 are the ANYmal-class dimensions used in both.

@dc.dataclass(frozen=True)
class MlpPolicy:
    name: str
    obs: int
    hidden: tuple
    act: int
    cite: str

    @property
    def layer_shapes(self) -> tuple:
        dims = (self.obs,) + self.hidden + (self.act,)
        return tuple(zip(dims, dims[1:]))

    @property
    def params(self) -> int:
        return sum(i * o + o for i, o in self.layer_shapes)

    @property
    def macs(self) -> int:
        return sum(i * o for i, o in self.layer_shapes)

    @property
    def activations(self) -> int:
        return self.obs + sum(self.hidden) + self.act

    def bytes_at(self, prec: str) -> float:
        """Weights at `prec`, activations always fp32 (they feed an fp32 solver)."""
        return (self.params * M.BYTES_PER_ELEM[prec]
                + self.activations * M.BYTES_PER_ELEM["fp32"])


POLICY_STD = MlpPolicy("legged-gym actor", 48, (512, 256, 128), 12,
                       "Rudin et al., CoRL 2021, arXiv:2109.11978")
POLICY_SMALL = MlpPolicy("ANYmal Science-Robotics policy", 48, (256, 128, 32), 12,
                         "Hwangbo et al., Sci. Robotics 2019, arXiv:1901.08652")


def policy_bench(p: MlpPolicy, rate: float, prec: str = "int8") -> M.Benchmark:
    """A batch-1 MLP policy as a benchmark row.

    IT IS GEMV, NOT GEMM. Batch 1, one observation per tick, so every weight is
    touched exactly once and the reuse depth is 1 -- the settled batch-1 finding
    applies to a policy network exactly as it applies to a solver iteration. The
    MAC array is idle regardless of size; delivered scratchpad bandwidth binds.
    Rows are short (48-512), so `gemv_row_len` is set to the narrowest layer
    input and the short-row derate is allowed to bite.
    """
    kern = [M.KernelOp(f"dense {i}x{o}", "gemv", macs=i * o, n_desc=1)
            for i, o in p.layer_shapes]
    kern.append(M.KernelOp("bias + ELU/tanh", "vec",
                           vec_ops=2 * p.activations, n_desc=len(p.layer_shapes)))
    return M.Benchmark(
        name=f"POLICY-{p.hidden[0]}", kind="ml", note=f"{p.name} [{p.cite}]",
        gemv_macs_per_iter=float(p.macs), gemm_macs_per_iter=0.0,
        gemm_macs_per_call=0.0, vector_ops_per_iter=2.0 * p.activations,
        transcendental_per_iter=0.0, scalar_ops_per_iter=40, algo="nn",
        iters=1, calls_per_sec=rate, deadline_us=1e6 / rate,
        persistent_bytes=p.bytes_at(prec), delta_bytes=4 * p.obs, prec=prec,
        gemv_row_len=min(i for i, _ in p.layer_shapes),
        kernels=kern, serial_scalar_per_iter=0.0)


# ---------------------------------------------------------------------------
# Arithmetic intensity — the reason the boundary sits where it does
# ---------------------------------------------------------------------------

def spad_gbs(c: M.ArgmindCfg) -> float:
    return c.scratchpad_bw_bpc * c.freq_mhz / 1000.0


def external_intensity(b: M.Benchmark, c: M.ArgmindCfg) -> float:
    """MACs per byte crossing the EXTERNAL boundary, per call.

    This -- not MAC count, not on-chip intensity -- is the quantity that decides
    which regime a workload is in. A batch-1 decode reads every weight once from
    off chip and does one MAC with it: 1 MAC/byte at int8, 2 at int4. A ViT frame
    reads its weights once and does 1.26 G MACs with them: ~220 MAC/byte. Three
    orders of magnitude, one boundary.
    """
    moved = M.model_argmind(b, c)["bytes_moved"]
    macs = b.total_macs()
    return macs / moved if moved > 0 else float("inf")


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("=" * 104)
    out("ARGMIND CM — THE ON-DEVICE AI ENVELOPE.  Two regimes, and the boundary")
    out("            between them, drawn with numbers.")
    out("=" * 104)
    out("ALL FIGURES MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE EXTERNALLY.")
    out("EVERY perf/W BELOW IS vs A CPU BASELINE. For ML that is NOT the adversary a")
    out("customer cares about -- see section 5 before quoting any of it.")
    out("ACCURACY IS NOT MODELLED. int4 here is a statement about bytes and cycles.")
    out()
    out("=" * 104)
    out("THE ANSWER, UP FRONT")
    out("=" * 104)
    out()
    out("  1. THE PLANE HAS TWO BOUNDARIES SET BY TWO DIFFERENT RESOURCES.")
    out("     FEASIBILITY is set by external bandwidth and moves 8x across an 8x")
    out("     bandwidth sweep: 0.32 G int8 / 0.64 G int4 at 6.4 GB/s and 20 tok/s,")
    out("     rising to 2.56 / 5.12 G at 51.2 GB/s. DIFFERENTIATION (>=5x perf/W) is")
    out("     set by whether the working set is RESIDENT and barely moves at all:")
    out("     0.00021-0.00025 G int8 / 0.00066 G int4 across the same sweep. Buying")
    out("     bandwidth buys feasibility and buys ~nothing of differentiation.")
    out()
    out("  2. THE FEASIBLE-UNDIFFERENTIATED ZONE IS ~1000-1500x WIDE IN MODEL SIZE")
    out("     AND CONTAINS EVERY MODEL ANYONE WOULD CALL A LANGUAGE MODEL. The")
    out("     differentiated zone tops out at 0.2-0.7 M parameters, which is an MLP,")
    out("     not an SLM. int4 doubles both boundaries and lifts the perf/W plateau")
    out("     from ~2.5x to ~4.0x -- closer to the 5x bar, still under it at every")
    out("     point in the plane.")
    out()
    out("  3. THE INVARIANT IS EXTERNAL ARITHMETIC INTENSITY, ~2.6 MAC PER BYTE")
    out("     crossing the CM boundary. Batch-1 int8 decode is exactly 1.0 by")
    out("     construction and int4 decode is 1.93; ViT-Tiny is 223; a resident MLP")
    out("     policy is 258-989; QUAD-MPC is 1022. One number orders the whole set.")
    out()
    out("  4. THE HYBRID LOOP SURVIVES, BUT NOT AS AN ML CLAIM AND NOT AT 192 KB.")
    out("     Policy + solver co-resident fits 192 KB only for the smaller of two")
    out("     cited real policies, with 1.3% margin, and only because codegen")
    out("     exploits A = [I;Gamma]; the widely used legged-gym shape needs 256 KB at")
    out("     int4 or 384 KB at int8. End to end it is 1.8-2.3x latency and 9.6-12.2x")
    out("     energy against an NPU-plus-CPU split with the NPU charged ZERO -- but")
    out("     the network is only 1.1-4.0% of the tick, so THE HYBRID LOOP'S ADVANTAGE")
    out("     IS THE SOLVER'S ADVANTAGE. What it adds is one engine, one address")
    out("     space, one deterministic joint tick, and one fewer offload per tick.")
    out()
    out("  5. THERE IS NO RESIDENT-WEIGHT ML SKU. 4 MB of scratchpad holds 8 M int4")
    out("     parameters, ~80x short of what a 6.4 GB/s stream already delivers, while")
    out("     costing 6.3x core area and 2.2x solver energy. Killed twice over.")
    out()
    out("  6. AND THE PART THAT HURTS: EVERY ML perf/W FIGURE HERE IS AGAINST A CPU,")
    out("     WHICH IS NOT THE BASELINE A CUSTOMER WOULD DEPLOY. The ML")
    out("     inference-efficiency claim is UNSUPPORTED, not merely narrow. Section 5")
    out("     says what is left: arithmetic retargetability, co-residency plus a")
    out("     joint deterministic tick, and MACIF. None of the three needs an")
    out("     inference-efficiency comparison to stand up.")
    out()
    out("  7. ONE ARCHITECTURE CONSEQUENCE, AND IT AMENDS §8. int4's array knee is")
    out("     144 B/cycle, not the 288 that fp32/bf16/int8 share, because int4's rate")
    out("     multiplier is int8's while its element halves. 128 B/cycle is still")
    out("     under it, but §8's '256 B/cycle is a drop-in upgrade' does NOT hold at")
    out("     int4: that part wants a 16x16 tile, not 12x12.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(0) int4 SELF-AUDIT — the three ways this could have been a free win")
    out("=" * 104)
    out()
    out("Adding a precision plus a swept axis is the exact shape of change that")
    out("produced modelling errors 13 and 14 (an axis that looked free because the")
    out("model could not express its cost). Three checks, run mechanically:")
    out()
    out("  CHECK A: does int4 halve BANDWIDTH demand without granting free COMPUTE?")
    for prec in ("int8", "int4"):
        out(f"    {prec}: {M.BYTES_PER_ELEM[prec]:>4} B/elem, array "
            f"{C100.macs_per_cycle(prec):>6.0f} MAC/cyc, "
            f"GEMV ceiling {C100.gemv_macs_per_cycle(prec):>6.0f} MAC/cyc "
            f"(= min(bw/elem, array))")
    out("    -> ARRAY RATE IS IDENTICAL (288 MAC/cyc). A digital multiplier is sized")
    out("       by its WIDER operand, and weight-only 4-bit quantisation multiplies a")
    out("       4-bit weight by an 8-bit activation, so the array slot is unchanged.")
    out("       Only the fetch ceiling doubles, 128 -> 256 MAC/cyc, and it is still")
    out("       BELOW the array rate -- int4 GEMV remains bandwidth-bound on chip.")
    out()
    out("  CHECK A2, WHICH IS NOT A CHECK BUT A CONSEQUENCE, AND IT MATTERS FOR §8.")
    out("  §8's balance rule says a 12x12 array absorbs GEMV up to 288 B/cycle 'at")
    out("  fp32, bf16 AND int8, because this datapath's rate multiplier is")
    out("  ~1/element-size'. int4 BREAKS that coincidence: its rate multiplier is")
    out("  int8's while its element is half the size. The bandwidth at which the")
    out("  array starts to bind is `macs_per_cycle * bytes_per_elem`:")
    out(f"{'prec':>6} {'MAC/cyc':>9} {'B/elem':>7} {'array binds above':>18} "
        f"{'min tile @256 B/cyc':>20}")
    for prec in ("fp32", "bf16", "int8", "int4"):
        mpc = C100.macs_per_cycle(prec)
        be = M.BYTES_PER_ELEM[prec]
        scale = mpc / C100.macs
        # tile^2 * scale >= bw / be  =>  tile >= sqrt(bw / (be * scale))
        need = math.sqrt(256.0 / (be * scale))
        out(f"{prec:>6} {mpc:>9.0f} {be:>7} {mpc * be:>15.0f} B/c "
            f"{need:>19.1f}")
    out("    -> at int4 the knee is 144 B/cycle, NOT 288. CM-100's 128 B/cycle sits")
    out("       just under it, so nothing changes today. But §8 sells 256 B/cycle as")
    out("       a DROP-IN upgrade on the strength of the array's 2.25x headroom, and")
    out("       AT int4 THAT HEADROOM IS GONE: a 256 B/cycle part wants a 16x16 tile")
    out("       for int4 GEMV, not 12x12. This is a real amendment to the balance")
    out("       rule and it is the one architectural consequence of adding int4.")
    out()
    out("  CHECK B: is DRAM energy still charged on every streamed weight byte?")
    for wp in ("int8", "int4"):
        b = slm(1.1, 20, weight_prec=wp)
        c = cfg_bw(25.6)
        r = M.model_argmind(b, c)
        e_dram = r["bytes_moved"] * c.offchip_pj_per_byte / 1e6
        out(f"    {wp}: moved {r['bytes_moved'] / 1e6:>8.1f} MB, DRAM energy "
            f"{e_dram:>8.1f} uJ of {r['energy_uj']:>8.1f} uJ total "
            f"({e_dram / r['energy_uj'] * 100:>5.1f}%)")
    out("    -> YES, at the full 12 pJ/B, and it is the DOMINANT term in both. int4")
    out("       halves the bytes; it does not make a byte cheaper.")
    out()
    out("  CHECK C: is the ENERGY-PER-MAC credit for int4 small and does it matter?")
    b8, b4 = slm(1.1, 20, weight_prec="int8"), slm(1.1, 20, weight_prec="int4")
    c = cfg_bw(25.6)
    base = M.model_argmind(b4, c)["energy_uj"]
    saved = dc.replace(C100, dma_bw_gbs_override=25.6)
    e_mac_backup = M.E_MAC_PJ["int4"]
    M.E_MAC_PJ["int4"] = M.E_MAC_PJ["int8"]
    pess = M.model_argmind(b4, saved)["energy_uj"]
    M.E_MAC_PJ["int4"] = e_mac_backup
    out(f"    int4 MAC at 0.060 pJ [EST]: {base:>9.1f} uJ")
    out(f"    int4 MAC forced to int8's 0.070 pJ: {pess:>9.1f} uJ "
        f"({pess / base:.4f}x)")
    out("    -> the int4 MAC-energy credit is worth <0.5% of the row. Every")
    out("       conclusion below is insensitive to it, because these rows are")
    out("       DRAM-dominated. That is the honest reason to accept the estimate.")
    out()
    out("  WHAT int4 IS NOT: expressible in the v0.1 descriptor format. Two int4")
    out("  elements share a byte, so a row length must be even and a leading")
    out("  dimension is a count of NIBBLES, while the v0.1 AGUs address bytes.")
    out("  Format.INT4 is also absent from the runtime's precision enum, so the")
    out("  simulator cannot hold an int4 buffer at all. Recorded as a REQUIREMENT")
    out("  with a cost, not quietly assumed -- the last two capacity errors were")
    out("  both 'the model assumed a layout the descriptor cannot address'.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(1) WHY THERE IS A BOUNDARY AT ALL — arithmetic intensity per EXTERNAL byte")
    out("=" * 104)
    out()
    out(f"The CM-100 scratchpad delivers {spad_gbs(C100):.1f} GB/s "
        f"({C100.scratchpad_bw_bpc:.0f} B/cycle x {C100.freq_mhz:.0f} MHz).")
    out("An edge SoC offers 6.4-51.2 GB/s off chip. So for ANY workload that reads")
    out("each external byte once and does ONE MAC with it:")
    out()
    out("    compute_time / transfer_time  =  BW_external / BW_scratchpad")
    out()
    out("which is INDEPENDENT of model size AND of precision, because int4 halves")
    out("both sides. Write that ratio rho = BW_ext / BW_spad. With double buffering")
    out("hiding 80% of compute behind the stream, the DRAM share of the tick is")
    out("(1 - 0.8*rho) / (1 + 0.2*rho), and with no overlap at all it is 1/(1+rho):")
    out(f"{'BW_ext GB/s':>12} {'rho':>8} {'no overlap':>11} {'double-buffered':>16}")
    for gbs in BW_POINTS + (102.4,):
        rho = gbs / spad_gbs(C100)
        db = max(0.0, (1.0 - 0.8 * rho) / (1.0 + 0.2 * rho))
        out(f"{gbs:>12.1f} {rho:>8.3f} {1.0 / (1.0 + rho) * 100:>10.0f}% "
            f"{db * 100:>15.1f}%")
    out()
    out("  Against the model on the frozen 1.1B row. This is a CROSS-CHECK, not an")
    out("  illustration: the analytic line is derived from the bandwidth ratio alone")
    out("  and the model computes bytes, cycles and overheads independently.")
    out(f"{'BW GB/s':>8} {'wprec':>6} {'model':>8} {'analytic':>9}")
    for gbs in BW_POINTS:
        rho = gbs / spad_gbs(C100)
        db = max(0.0, (1.0 - 0.8 * rho) / (1.0 + 0.2 * rho))
        for wp in ("int8", "int4"):
            b, cc = slm(1.1, 20, weight_prec=wp), cfg_bw(gbs)
            r = M.model_argmind(b, cc)
            out(f"{gbs:>8.1f} {wp:>6} {r['transfer_us'] / r['latency_us'] * 100:>7.1f}% "
                f"{db * 100:>8.1f}%")
    out()
    out("  Agreement to ~0.5 point, and int4 sits on TOP OF int8 in every row. That")
    out("  is the whole point: HALVING THE WEIGHT WIDTH DOES NOT CHANGE WHICH")
    out("  RESOURCE BINDS, because it halves the on-chip and the off-chip demand by")
    out("  the same factor. int4 buys 2x of absolute throughput and 0x of regime.")
    out()
    out("  NOW THE OTHER REGIME. External-byte intensity, MACs per byte crossing")
    out("  the CM boundary per call, at the CM-100's own 6.4 GB/s:")
    rows = [
        ("SLM 1.1B int8 decode", slm(1.1, 20, weight_prec="int8")),
        ("SLM 1.1B int4 decode", slm(1.1, 20, weight_prec="int4")),
        ("SLM 0.1B int4 decode", slm(0.1, 20, weight_prec="int4")),
        ("VIT-TINY (frozen row)",
         next(b for b in M.BENCHMARKS if b.name == "VIT-TINY")),
        ("POLICY-512 (resident)", policy_bench(POLICY_STD, 1000)),
        ("POLICY-256 (resident)", policy_bench(POLICY_SMALL, 1000)),
        ("QUAD-MPC (resident)",
         next(b for b in M.BENCHMARKS if b.name == "QUAD-MPC")),
    ]
    c64 = cfg_bw(6.4)
    out(f"{'row':<22} {'persistent KB':>14} {'moved KB/call':>14} "
        f"{'MAC/ext byte':>13} {'perf/W vs-CPU':>14}")
    for lab, b in rows:
        r = M.model_argmind(b, c64)
        ii = external_intensity(b, c64)
        out(f"{lab:<22} {b.persistent_bytes / 1024:>14.1f} "
            f"{r['bytes_moved'] / 1024:>14.1f} "
            f"{ii:>13.4g} {perf_w(b, c64):>13.2f}x")
    out()
    out("  THE BOUNDARY IS A RESIDENCY BOUNDARY, NOT A SIZE BOUNDARY. Everything")
    out("  whose persistent data FITS reads only its delta from off chip, so its")
    out("  external intensity is 1e2-1e3 MAC/byte and DRAM energy vanishes from the")
    out("  total. Everything that does NOT fit re-streams its weights every call at")
    out("  1-2 MAC/byte, and DRAM energy becomes 80-95% of BOTH sides of the")
    out("  comparison. perf/W does NOT go to 1.0 -- the shared DRAM term is added to")
    out("  a CPU core cost that also grows with parameter count, so it flattens onto")
    out("  a PLATEAU (2.3-2.7x at int8, 2.7-4.2x at int4 in the table below). The")
    out("  point is not that the plateau is 1.0; it is that THE PLATEAU IS BELOW 5x")
    out("  EVERYWHERE. int4 moves the size at which you fall off the cliff. It does")
    out("  not move the cliff, and it does not lift the plateau over the bar.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(2) THE PLANE — model size x external bandwidth, int8 and int4 weights")
    out("=" * 104)
    out()
    out("Zones:  DIFF*  meets deadline AND >=5.0x perf/W vs the best CPU baseline")
    out("        feasible  meets deadline, below 5x  (i.e. we work but do not win)")
    out("        INFEASIBLE  misses the token deadline")
    out("The * is not decoration: see section 5.")
    out()
    sizes = (0.01, 0.03, 0.1, 0.16, 0.3, 0.5, 1.0, 1.1, 2.0, 3.0)
    for rate in (20.0, 10.0):
        out(f"  --- {rate:g} tok/s  (deadline {1e6 / rate / 1000:.0f} ms/token) ---")
        for wp in ("int8", "int4"):
            out(f"  weights {wp}, KV int8:")
            out("    " + f"{'params G':>9} {'hidden/L':>9} {'wt MB':>7} "
                + " ".join(f"{g:>17.1f}" for g in BW_POINTS))
            for p in sizes:
                b = slm(p, rate, weight_prec=wp)
                h, ly = transformer_shape(p * 1e9)
                cells = []
                for gbs in BW_POINTS:
                    z, r, pw = zone(b, cfg_bw(gbs))
                    cells.append(f"{z:>10} {pw:>5.2f}x" if z != Z_INF
                                 else f"{z:>10} {r['latency_us'] / 1000:>5.0f}ms")
                out("    " + f"{p:>9.2f} {h:>5}/{ly:<3} "
                    f"{p * 1e9 * M.BYTES_PER_ELEM[wp] / 1e6:>7.0f} "
                    + " ".join(f"{x:>17}" for x in cells))
            out()

    out("  BOUNDARY 1 — FEASIBILITY. Purely a bandwidth-vs-bytes division and it")
    out("  can be written down without the model. At a 100%-of-period transfer")
    out("  budget the largest model that meets the deadline is")
    out()
    out("      params_max = BW_ext / (tok_per_s * bytes_per_weight)")
    out()
    out(f"{'BW GB/s':>8} " + " ".join(f"{p:>18}" for p in
                                     ("int8 @20/s", "int4 @20/s",
                                      "int8 @10/s", "int4 @10/s")))
    for gbs in BW_POINTS:
        cells = []
        for wp in ("int8", "int4"):
            for rate in (20.0, 10.0):
                cells.append((wp, rate, gbs * 1e9 / rate
                              / M.BYTES_PER_ELEM[wp] / 1e9))
        order = [cells[0], cells[2], cells[1], cells[3]]
        out(f"{gbs:>8.1f} " + " ".join(f"{v:>13.2f} G" for _, _, v in order))
    out("  (this is the ceiling; the measured table above lands below it because")
    out("   the KV cache, the delta and the call overhead also consume the period)")
    out()
    out("  RECONCILIATION WITH THE EXISTING §8 FIGURE, which is a continuity check")
    out("  rather than a new number: §8 quotes ~0.16 G int8 at 6.4 GB/s / 20 tok/s")
    out(f"  at a 50%-OF-PERIOD transfer budget. The 100% ceiling above is 0.32 G;")
    out("  half of it is 0.16 G. The two agree, and int4 doubles both.")
    out()
    out("  BOUNDARY 2 — DIFFERENTIATION, and it is the one that matters. Where does")
    out("  perf/W cross 5x? Bisected from 1e-5 G (10k parameters, well inside the")
    out("  resident regime) up to 4 G, so the scan CAN find a crossover if one")
    out("  exists:")
    out(f"{'BW GB/s':>8} {'wprec':>6} {'pw @1e-5 G':>11} {'pw @3 G':>9} "
        f"{'5x crossover':>14} {'ext KiB/call':>13} {'MAC/ext B':>10}")
    cross, cross_ii = {}, {}
    for gbs in BW_POINTS:
        for wp in ("int8", "int4"):
            cc = cfg_bw(gbs)
            lo, hi = 1e-5, 4.0
            pw_lo = perf_w(slm(lo, 20, weight_prec=wp), cc)
            pw_hi = perf_w(slm(3.0, 20, weight_prec=wp), cc)
            if pw_lo < KILL:
                out(f"{gbs:>8.1f} {wp:>6} {pw_lo:>10.2f}x {pw_hi:>8.2f}x "
                    f"{'NONE':>14} {'-':>16} {'-':>10}")
                continue
            for _ in range(50):
                mid = math.sqrt(lo * hi)
                if perf_w(slm(mid, 20, weight_prec=wp), cc) >= KILL:
                    lo = mid
                else:
                    hi = mid
            bx = slm(lo, 20, weight_prec=wp)
            moved = M.model_argmind(bx, cc)["bytes_moved"]
            ii = external_intensity(bx, cc)
            cross[(gbs, wp)] = lo
            cross_ii[(gbs, wp)] = ii
            out(f"{gbs:>8.1f} {wp:>6} {pw_lo:>10.2f}x {pw_hi:>8.2f}x "
                f"{lo:>12.5f} G {moved / 1024:>13.1f} {ii:>10.1f}")
    out()
    out("  ==================================================================")
    out("  TWO BOUNDARIES, TWO DIFFERENT RESOURCES. THAT IS THE WHOLE RESULT.")
    out("  ==================================================================")
    fb = {(g, w): g * 1e9 / 20.0 / M.BYTES_PER_ELEM[w] / 1e9
          for g in BW_POINTS for w in ("int8", "int4")}
    out("    FEASIBILITY is set by EXTERNAL BANDWIDTH and moves 8x across the sweep:")
    out(f"      int8 20 tok/s: {fb[(6.4, 'int8')]:.2f} G at 6.4 GB/s -> "
        f"{fb[(51.2, 'int8')]:.2f} G at 51.2 GB/s "
        f"({fb[(51.2, 'int8')] / fb[(6.4, 'int8')]:.1f}x)")
    out("    DIFFERENTIATION is set by SCRATCHPAD CAPACITY and barely moves at all:")
    out(f"      int8 20 tok/s: {cross[(6.4, 'int8')]:.5f} G at 6.4 GB/s -> "
        f"{cross[(51.2, 'int8')]:.5f} G at 51.2 GB/s "
        f"({cross[(51.2, 'int8')] / cross[(6.4, 'int8')]:.2f}x)")
    out(f"      int4 20 tok/s: {cross[(6.4, 'int4')]:.5f} G -> "
        f"{cross[(51.2, 'int4')]:.5f} G "
        f"({cross[(51.2, 'int4')] / cross[(6.4, 'int4')]:.2f}x -- flat)")
    out("    Buying external bandwidth makes a bigger model FEASIBLE and does")
    out("    essentially nothing for whether it is DIFFERENTIATED. That is the")
    out("    quantitative form of 'we are not differentiated on a workload whose")
    out("    binding resource is off chip'.")
    out()
    out("    AND THE INVARIANT BEHIND IT, which is the number to keep: at every")
    out("    crossover point in the table the EXTERNAL ARITHMETIC INTENSITY is")
    out(f"    {min(cross_ii.values()):.0f}-{max(cross_ii.values()):.0f} MAC per "
        f"external byte, across an 8x bandwidth range and both")
    out("    precisions. THAT is the boundary, and it is a workload property, not a")
    out("    configuration property:")
    out()
    out(f"        DIFFERENTIATED  requires  >~ {max(cross_ii.values()):.1f} "
        f"MAC per byte crossing the CM boundary")
    out()
    out("    Check it against the rows in section 1: batch-1 int8 decode is exactly")
    out("    1.0 (one MAC per weight byte, by definition), int4 decode is 1.93,")
    out("    ViT-Tiny is 223, a resident MLP policy is 258-989, QUAD-MPC is 1022.")
    out("    One number orders every row in the set and it is neither MAC count nor")
    out("    model size.")
    out()
    out("    NOW THE UNCOMFORTABLE PART OF THAT, AND IT IS THE HONEST READING:")
    out(f"    int4 decode at 1.93 MAC/byte sits only {2.6 / 1.93:.2f}x below the "
        f"{max(cross_ii.values()):.1f} threshold, not")
    out("    orders of magnitude below it. So the negative result is NOT robust to a")
    out("    batch dimension: external intensity is LINEAR in batch size, so batch 2")
    out("    at int4 would cross the line. The reason this is not a hole in the")
    out("    finding is that (a) interactive single-user decode is batch 1 by")
    out("    definition, and (b) THE MODEL CANNOT CURRENTLY EXPRESS A BATCHED DECODE")
    out("    -- it would need the GEMM path plus a KV-cache growth term. Do not quote")
    out("    a batched number. Do state that a batched or speculatively-decoded")
    out("    workload is a genuinely different point on this plane and is the single")
    out("    most likely way this conclusion changes. It is listed in section 6.")
    out()
    out("  AND THE GAP BETWEEN THE TWO BOUNDARIES IS THE ANSWER TO THE QUESTION:")
    out(f"      differentiated up to  {cross[(6.4, 'int8')]:.5f} G int8 / "
        f"{cross[(6.4, 'int4')]:.5f} G int4")
    out(f"      feasible up to        {fb[(6.4, 'int8')]:.2f} G int8 / "
        f"{fb[(6.4, 'int4')]:.2f} G int4   (at 6.4 GB/s, 20 tok/s)")
    out(f"      ratio                 {fb[(6.4, 'int8')] / cross[(6.4, 'int8')]:.0f}x "
        f"int8 / {fb[(6.4, 'int4')] / cross[(6.4, 'int4')]:.0f}x int4")
    out("    So the FEASIBLE-UNDIFFERENTIATED zone is ~1000-1500x wide in model size")
    out("    and covers EVERY model anyone would call a language model. The")
    out("    DIFFERENTIATED zone tops out around 0.2-0.7 M parameters -- which is not")
    out("    a small language model, it is an MLP. The two ML regimes are not two")
    out("    sizes of the same workload; they are different workloads, and the")
    out("    dividing line is whether the weights live on chip.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(3) THE WEIGHT-RESIDENT REGIME — what fits, and is there an ML SKU in it?")
    out("=" * 104)
    out()
    out("What fits 192 KB, weights only (leave nothing for activations or a solver):")
    for prec in ("int8", "int4"):
        n = C100.scratchpad_kb * 1024 / M.BYTES_PER_ELEM[prec]
        out(f"    {prec}: {n:>10,.0f} parameters = {n / 1e9:.6f} G")
    out()
    out(f"  The feasible streaming envelope at 6.4 GB/s / 20 tok/s is 0.32 G int8 /")
    out(f"  0.64 G int4 [DER, boundary 1]. What FITS on chip is 0.0002-0.0004 G.")
    out("  THOSE TWO NUMBERS DIFFER BY ~1600x, so for LLM decode the resident regime")
    out("  and the feasible regime DO NOT OVERLAP AT ALL. There is no scratchpad")
    out("  size that turns an LLM into a resident workload -- see the SKU table.")
    out()
    out("  Resident-weight ML SKU: what would a bigger scratchpad buy and cost?")
    out(f"{'KB':>7} {'spad mm2':>9} {'core mm2':>9} {'vs 192KB':>9} "
        f"{'int4 params':>12} {'solver uJ':>10} {'vs 192KB':>9}")
    solvers = [b for b in M.BENCHMARKS if b.kind == "solver"]

    def solver_energy(cc):
        return sum(M.model_argmind(x, cc)["energy_uj"] for x in solvers)

    ref = None
    for kb in (192, 256, 384, 512, 1024, 2048, 4096):
        nb = 16
        cc = dc.replace(C100, scratchpad_kb=kb, name=f"CM {kb}KB")
        area = C.area_breakdown(cc)
        se = solver_energy(cc)
        if ref is None:
            ref = (area["total"], se)
        out(f"{kb:>7} {area['spad']:>9.3f} {area['total']:>9.3f} "
            f"{area['total'] / ref[0]:>8.2f}x "
            f"{kb * 1024 / 0.5 / 1e6:>10.2f} M {se:>10.2f} "
            f"{se / ref[1]:>8.3f}x")
    out()
    out("  READ THE LAST COLUMN. Solver energy is MONOTONE INCREASING in capacity")
    out("  (§8: bigger banks, longer bitlines, more pJ/byte on traffic that was")
    out("  already resident), so an ML-capacity SKU TAXES EVERY SOLVER ROW -- the")
    out("  rows the company is actually sold on. And it does not even buy the ML:")
    out("  a 4 MB scratchpad holds 8 M int4 parameters, still ~80x below the 0.64 G")
    out("  a 6.4 GB/s stream already delivers. VERDICT: THERE IS NO RESIDENT-WEIGHT")
    out("  LLM SKU. The capacity finding does kill it, and it kills it by two")
    out("  independent arguments (solver tax AND insufficient reach).")
    out()
    out("  THE MIDDLE CASE IS DIFFERENT, and it is the one worth having. Real")
    out("  networks against 192 KB, alone and co-resident with the flagship solver:")
    out(f"{'network':<34} {'params':>9} {'int8 KB':>9} {'int4 KB':>9} "
        f"{'fits 192KB?':>12} {'+QUAD fits?':>12}")
    nets = [
        (POLICY_SMALL.name, POLICY_SMALL.params, POLICY_SMALL),
        (POLICY_STD.name, POLICY_STD.params, POLICY_STD),
    ]
    for nm, np_, p in nets:
        k8, k4 = p.bytes_at("int8") / 1024, p.bytes_at("int4") / 1024
        out(f"{nm:<34} {np_:>9,} {k8:>9.1f} {k4:>9.1f} "
            f"{'yes' if k8 <= 192 else 'int4 only' if k4 <= 192 else 'no':>12} "
            f"{'yes' if k8 + QUAD_FOOTPRINT_KB <= 192 else 'int4' if k4 + QUAD_FOOTPRINT_KB <= 192 else 'NO':>12}")
    vit = next(b for b in M.BENCHMARKS if b.name == "VIT-TINY")
    out(f"{'ViT-Tiny (frozen ML row)':<34} {5.7e6:>9,.0f} "
        f"{vit.persistent_bytes / 1024:>9.0f} "
        f"{vit.persistent_bytes / 2048:>9.0f} {'no':>12} {'NO':>12}")
    out()
    out("  So there is NO middle case in the sense the question implied -- no network")
    out("  that needs 384 KB-1 MB and repays it. The population is bimodal: MLP")
    out("  control policies at 25-200 KB (which fit, or nearly), and everything with")
    out("  a convolutional or transformer backbone at 3-6 MB and up (which never")
    out("  fits, at any SKU we would build). The 25-200 KB band is EXACTLY the band")
    out("  where residency, coherency and one-call fusion apply, and it is also the")
    out("  band where the workload is batch-1 GEMV -- the settled bandwidth-bound")
    out("  finding, applied to a network instead of a solver. The one thing 256-384")
    out("  KB does buy is CO-RESIDENCY with the solver, which is section 4.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(4) THE HYBRID LOOP — learned policy -> reference -> MPC solver, one call")
    out("=" * 104)
    out()
    out("The claim under test: a learned policy produces a reference or a")
    out("constraint, an MPC solver enforces it, both resident on one engine in one")
    out("call. Concretely, at 1 kHz on a legged robot:")
    out(f"    policy A: {POLICY_SMALL.name}")
    out(f"              MLP 48-256-128-32-12, {POLICY_SMALL.params:,} params")
    out(f"              [CITE] {POLICY_SMALL.cite}")
    out(f"    policy B: {POLICY_STD.name}")
    out(f"              MLP 48-512-256-128-12, {POLICY_STD.params:,} params")
    out(f"              [CITE] {POLICY_STD.cite}")
    out(f"    solver:   QUAD-MPC, measured footprint {QUAD_FOOTPRINT_KB} KiB "
        f"at fp32 [MEAS]")
    out()
    out("  (4a) DOES THE COMBINED WORKING SET FIT 192 KB?")
    out(f"{'combination':<40} {'policy KiB':>11} {'solver KiB':>11} "
        f"{'total KiB':>10} {'of 192':>8} {'verdict':>9}")
    combos = []
    for p, plab in ((POLICY_SMALL, "policy A"), (POLICY_STD, "policy B")):
        for prec in ("int8", "int4"):
            pk = p.bytes_at(prec) / 1024
            tot = pk + QUAD_FOOTPRINT_KB
            combos.append((p, plab, prec, pk, tot))
            out(f"{plab + ' @ ' + prec:<40} {pk:>11.1f} "
                f"{QUAD_FOOTPRINT_KB:>11.1f} {tot:>10.1f} "
                f"{tot / 192 * 100:>7.0f}% {'FITS' if tot <= 192 else 'NO':>9}")
    tot = {(plab, prec): t for _, plab, prec, _, t in combos}

    def need_kb(t):
        for kb in (192, 256, 384, 512):
            if t <= kb:
                return kb
        return 1024
    out()
    out("  THE MARGIN IS THE FINDING, AND IT IS THIN.")
    out(f"    policy A @ int8 fits with {192 - tot[('policy A', 'int8')]:.1f} KiB "
        f"spare of 192 -- {(192 - tot[('policy A', 'int8')]) / 192 * 100:.1f}%.")
    out(f"    policy B @ int8 is {tot[('policy B', 'int8')]:.1f} KiB = "
        f"{tot[('policy B', 'int8')] / 192 * 100:.0f}% of capacity: does NOT fit, "
        f"needs {need_kb(tot[('policy B', 'int8')])} KB.")
    out(f"    policy B @ int4 is {tot[('policy B', 'int4')]:.1f} KiB = "
        f"{tot[('policy B', 'int4')] / 192 * 100:.0f}% of capacity: does NOT fit, "
        f"needs {need_kb(tot[('policy B', 'int4')])} KB.")
    out("  So the hybrid loop is REAL at 192 KB for a Science-Robotics-class policy,")
    out("  with ~1% of margin -- which is not a margin, it is a coincidence. For the")
    out("  legged-gym default shape (the more widely copied one) it needs 256 KB at")
    out("  int4 weights or 384 KB at int8.")
    out()
    out("  THAT IS A CAPACITY REQUIREMENT AND IT IS THE FIRST ONE IN THE WHOLE")
    out("  ANALYSIS THAT IS NOT ABOUT MAKING ONE SOLVER FIT. What it costs, taken")
    out("  straight from the section-3 table so the price is attached to the ask:")
    for kb in (256, 384):
        cc = dc.replace(C100, scratchpad_kb=kb)
        a = C.area_breakdown(cc)
        a0 = C.area_breakdown(C100)
        se, se0 = solver_energy(cc), solver_energy(C100)
        out(f"    {kb:>4} KB: core {a['total']:.3f} mm2 "
            f"({a['total'] / a0['total']:.2f}x), solver energy "
            f"{se / se0:.3f}x, and ZERO latency benefit to any solver row")
    out("  Note the arrow: this is the SAME capacity axis §8 struck as a scaling")
    out("  axis (\"capacity is a cost, 1.29x\"). It is still a cost. The difference is")
    out("  that a workload is now asking for it, which is the thing §8 said did not")
    out("  exist -- so this does not overturn the finding, it supplies the missing")
    out("  demand side and prices it.")
    out()
    out("  (4b) DOES RESIDENCY HOLD ACROSS BOTH, OR DO THEY EVICT EACH OTHER?")
    quad = next(b for b in M.BENCHMARKS if b.name == "QUAD-MPC")
    out(f"{'combination':<26} {'KB':>5} {'overflow KiB':>13} {'stream us/tick':>15} "
        f"{'DRAM uJ/tick':>13} {'residency':>12}")
    for p, plab, prec, pk, tt in combos:
        for kb in (192, 256, 384):
            ov = max(0.0, tt * 1024 - kb * 1024)
            us = ov / (C100.dma_bw_gbs * 1e9) * 1e6
            uj = ov * C100.offchip_pj_per_byte / 1e6
            out(f"{plab + ' @ ' + prec:<26} {kb:>5} {ov / 1024:>13.1f} "
                f"{us:>15.1f} {uj:>13.2f} "
                f"{'HOLDS' if ov == 0 else 'BROKEN':>12}")
    ov_b8 = (tot[("policy B", "int8")] - 192) * 1024
    q_uj = M.model_argmind(quad, C100)["energy_uj"]
    out()
    out("  Residency is BINARY here and there is no graceful degradation. Policy B at")
    out(f"  int8 in 192 KB re-streams {ov_b8 / 1024:.1f} KiB EVERY TICK: "
        f"{ov_b8 / (C100.dma_bw_gbs * 1e9) * 1e6:.1f} us of a 1000 us period at "
        f"1 kHz and")
    out(f"  {ov_b8 * C100.offchip_pj_per_byte / 1e6:.2f} uJ against the solve's own "
        f"{q_uj:.2f} uJ -- "
        f"+{ov_b8 * C100.offchip_pj_per_byte / 1e6 / q_uj * 100:.0f}% energy for a")
    out("  network that is ~4% of the work. The measured 46x residency ratio")
    out("  (268 KB first call, then 5.8 KB in / 0.6 KB out per tick [MEAS]) SURVIVES")
    out("  co-residency in the fitting cases and is DESTROYED in the others -- there")
    out("  is no partial credit.")
    out()
    out("  Two honest qualifications on that number:")
    out("    * The model's overflow term is a uniform 'excess streams every call'")
    out("      approximation and does not know WHICH buffer is evicted. Here it")
    out("      coincides with what codegen would actually choose -- keep the solver")
    out("      data (touched 15-74 times per tick), stream the policy weights")
    out("      (touched once) -- so it is the right number and not merely a bound.")
    out("      It would NOT be the right number for two workloads with similar reuse.")
    out("    * 138.8 KiB is the QUAD-MPC footprint with A = [I;Gamma] exploited. The")
    out("      dense-A plan measures 163.8 KiB, which pushes even policy A @ int8 to")
    out(f"      {163.8 + POLICY_SMALL.bytes_at('int8') / 1024:.1f} KiB and OUT of "
        f"192 KB. The hybrid loop's fit therefore")
    out("      DEPENDS ON A CODEGEN OPTIMISATION, which is a fragile place for a")
    out("      product claim to rest. State the 256 KB requirement.")
    out()
    out("  (4c) END-TO-END TICK: one CM call vs the NPU-plus-CPU split shipped today")
    out()
    out("  The comparison is charged as generously to the incumbent as it can be:")
    out("  THE NPU'S INFERENCE IS FREE. Zero time, zero energy, zero DRAM. That is")
    out("  not a modelling convenience, it is the only defensible thing to do -- we")
    out("  have no measured NPU data and may not benchmark against vendor claims. If")
    out("  we win against a free NPU the result is robust; if we do not, say so.")
    out("  The incumbent pays only what is unavoidable and measurable in OUR model:")
    out("  one offload launch (T0 = 25.0 us rich host / 2.0 us MCU-class, from the")
    out("  decomposed table in argmind_model) plus the CPU's own solve.")
    out()
    shares = []
    for p, plab in ((POLICY_SMALL, "policy A (256-128-32)"),
                    (POLICY_STD, "policy B (512-256-128)")):
        pb = policy_bench(p, 1000)
        fits = p.bytes_at("int8") / 1024 + QUAD_FOOTPRINT_KB <= 192
        kb = 192 if fits else 384
        cc = dc.replace(C100, scratchpad_kb=kb,
                        name=f"CM-100/{kb}KB")
        rp = M.model_argmind(pb, cc)
        rq = M.model_argmind(quad, cc)
        # One call: one host overhead, not two. The sequencer runs the policy
        # descriptors and then the ADMM loop out of one statically generated
        # program -- that is exactly what "one engine, one call" means.
        cm_us = (rp["compute_us"] + rp["transfer_us"] + rq["compute_us"]
                 + rq["transfer_us"] + C100.call_overhead_us)
        cm_uj = rp["energy_uj"] + rq["energy_uj"]
        cpu_p, cpu_pr = cpu_best_latency(pb)
        cpu_q, cpu_qr = cpu_best_latency(quad)
        cpu_qe = cpu_best_energy(quad, 12.8)
        out(f"  {plab}, scratchpad {kb} KB:")
        out(f"    CM, one call    policy {rp['compute_us']:>7.2f} us  "
            f"solve {rq['compute_us']:>7.2f} us  overhead "
            f"{C100.call_overhead_us:>5.2f} us  = {cm_us:>8.2f} us, "
            f"{cm_uj:>7.2f} uJ")
        for tier, ov in (("rich host", 25.0), ("MCU-class", 2.0)):
            today = ov + cpu_qr["latency_us"]
            today_uj = cpu_qe["energy_uj"]
            out(f"    NPU(free)+CPU   launch {ov:>6.2f} us  "
                f"solve {cpu_qr['latency_us']:>7.2f} us ({cpu_q.name[:22]:<22})"
                f" = {today:>8.2f} us, {today_uj:>7.2f} uJ   "
                f"[{tier}]  {today / cm_us:>5.2f}x lat, "
                f"{today_uj / cm_uj:>5.2f}x energy")
        out(f"    policy share of the CM tick: "
            f"{rp['compute_us'] / (rp['compute_us'] + rq['compute_us']) * 100:.1f}% "
            f"of compute, "
            f"{rp['energy_uj'] / cm_uj * 100:.1f}% of energy")
        out(f"    deadline 1000 us at 1 kHz: CM "
            f"{'MEETS' if cm_us <= 1000 else 'MISSES'}, "
            f"incumbent {'MEETS' if 25.0 + cpu_qr['latency_us'] <= 1000 else 'MISSES'}")
        if kb != M.CM100_SCRATCHPAD_KB:
            q192 = M.model_argmind(quad, C100)["energy_uj"]
            out(f"    CAPACITY TAX INCLUDED ABOVE: the solve costs "
                f"{rq['energy_uj']:.2f} uJ at {kb} KB against")
            out(f"    {q192:.2f} uJ at 192 KB "
                f"(+{(rq['energy_uj'] / q192 - 1) * 100:.1f}%). That tax is paid on "
                f"EVERY solver call,")
            out("    including calls with no network in them, which is the honest cost")
            out("    of buying capacity for co-residency.")
        shares.append(rp["compute_us"] / (rp["compute_us"] + rq["compute_us"]) * 100)
        out()
    out("  Note the CPU is given the BEST OF BOTH baselines, per metric: the fastest")
    out("  core for latency (SME2) and the most efficient for energy (A55 at 726 us,")
    out("  which still meets 1 kHz). Mixing them favours the incumbent on both axes.")
    out("  It is also given a FREE NPU, so the energy ratios above are lower bounds on")
    out("  our advantage and the latency ratios are too -- and they are still only")
    out("  1.8-2.3x on latency, which is the same 'more efficient, not enabling'")
    out("  answer every solver row gives.")
    out()
    out("  (4d) WHAT ABOUT A PERCEPTION NETWORK RATHER THAN A POLICY MLP? This is the")
    out("  version of the hybrid loop the pitch reaches for, and it fails for a")
    out("  reason that is not capacity at all:")
    vitb = next(b for b in M.BENCHMARKS if b.name == "VIT-TINY")
    vit_kb = vitb.persistent_bytes / 1024
    out(f"{'schedule':<34} {'period us':>10} {'vit us':>9} {'solve us':>9} "
        f"{'stream us':>10} {'tick us':>9} {'of period':>10} {'verdict':>9}")
    for rate, lab in ((30.0, "ViT+QUAD at ViT's own 30 fps"),
                      (1000.0, "ViT+QUAD in the 1 kHz tick")):
        vr = M.model_argmind(dc.replace(vitb, calls_per_sec=rate,
                                        deadline_us=1e6 / rate), C100)
        qr = M.model_argmind(quad, C100)
        # Co-resident: the solver's own data no longer fits alongside the ViT
        # weights, so BOTH overflow terms are charged against one capacity.
        ovf = max(0.0, vitb.persistent_bytes + QUAD_FOOTPRINT_KB * 1024
                  - C100.scratchpad_kb * 1024)
        stream = ovf / (C100.dma_bw_gbs * 1e9) * 1e6
        tick = (vr["compute_us"] + qr["compute_us"] + stream
                + C100.call_overhead_us)
        period = 1e6 / rate
        out(f"{lab:<34} {period:>10.0f} {vr['compute_us']:>9.0f} "
            f"{qr['compute_us']:>9.1f} {stream:>10.0f} {tick:>9.0f} "
            f"{tick / period * 100:>9.0f}% "
            f"{'MEETS' if tick <= period else 'MISSES':>9}")
    out("    READ WHICH COLUMN IS BIG. At 30 fps the pair is comfortable (26% of the")
    out("    period). In a 1 kHz tick it misses by 8.5x, and it misses on COMPUTE:")
    out("    7598 us of array-bound GEMM against 882 us of streaming and 48 us of")
    out("    solve. So co-residency is not even the binding problem here -- a ViT")
    out("    frame does not fit a 1 ms budget on a 144-MAC array at ANY capacity or")
    out("    bandwidth, and buying the 5.7 MB that would make it resident (6.3x core")
    out("    area, 2.2x solver energy, section 3) would not help by a single")
    out("    microsecond. The binding resource is the one thing §8 says does not")
    out("    scale usefully (MAC array, 1.00x), which is why this is a scope")
    out("    boundary and not a provisioning request.")
    out("    THE HYBRID LOOP ONLY CLOSES WHEN THE NETWORK RUNS AT THE CONTROL RATE,")
    out("    AND THAT FORCES THE NETWORK TO BE AN MLP. Perception at 30 Hz next to")
    out("    control at 1 kHz is the CONSOLIDATION-ACROSS-LOOPS question §8 already")
    out("    settled: capacity-bound rather than throughput-bound, needing")
    out("    descriptor-granularity preemption. Different product question, and this")
    out("    is a second workload pointing at the same requirement.")
    out()
    out("  WHAT THIS ACTUALLY SAYS, and it is less than the pitch wants:")
    out(f"    * The network is {min(shares):.1f}-{max(shares):.1f}% of the tick's "
        f"compute and a similar share of")
    out("      its energy. THE HYBRID LOOP'S ADVANTAGE IS THE SOLVER'S ADVANTAGE.")
    out("      Fusing the policy in does not create a new claim; it removes a REASON")
    out("      TO BUY A SECOND ENGINE, and it removes one offload launch per tick.")
    out("    * Both sides MEET the 1 kHz deadline. Same story as every solver row:")
    out("      we are more efficient, not enabling. Do not pitch feasibility.")
    out("    * The offload launch we delete is 25 us on a rich host and 2 us on an")
    out("      MCU-class one. At 1 kHz that is 2.5% / 0.2% of the period -- material")
    out("      only above ~10 kHz, which is consistent with the settled finding that")
    out("      the invocation argument needs a deadline above ~100 kHz to decide")
    out("      anything. DO NOT SELL THE HYBRID LOOP ON LAUNCH OVERHEAD.")
    out("    * The claim that DOES survive is architectural and capacity-shaped:")
    out("      ONE ENGINE, ONE ADDRESS SPACE, ONE STATICALLY ALLOCATED WORKING SET,")
    out("      and therefore ONE deterministic worst-case tick covering perception")
    out("      and control together. A robot today cannot state a joint worst case")
    out("      across an NPU and a CPU sharing DRAM. That is a determinism claim,")
    out("      which is already one of the three that survive against a vector CPU.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(5) THE BASELINE PROBLEM — why no ML perf/W number here is a claim")
    out("=" * 104)
    out()
    out("Every perf/W figure above divides a modelled CM energy by a modelled CPU")
    out("energy. On solver rows that is the right comparison: an MPC solver runs on")
    out("the application core today, and the in-order A55 baseline is deliberately")
    out("the least flattering choice available.")
    out()
    out("ON ML ROWS IT IS THE WRONG COMPARISON AND WE KNOW IT. Nobody runs edge CNN")
    out(f"inference on a Cortex-A when the SoC has an NPU. VIT-TINY's "
        f"{perf_w(vitb, C100):.1f}x is a")
    out("comparison against a baseline no engineer would choose. (Note in passing:")
    out("the figure sometimes quoted for that row is 30.4x. The model has said")
    out(f"{perf_w(vitb, C100):.2f}x since the 192 KB capacity correction and")
    out("results_scaling_study.txt records 28.22x -- use the recorded value.)")
    out("The honest adversary is the NPU, and we cannot run that comparison:")
    out("  * we have no MEASURED NPU silicon energy at a stated node and precision;")
    out("  * our evidence standard forbids comparing a modelled figure against a")
    out("    vendor marketing claim, which is all that is published; and")
    out("  * an inference-optimised int8 macro (or a CIM macro) is designed for")
    out("    exactly the operand-reuse pattern a CNN has, which is the one pattern")
    out("    where our bandwidth-first provisioning is NOT the right shape.")
    out()
    out("  So the conclusion is stronger than 'scope the ML claim narrowly':")
    out()
    out("    THE ML INFERENCE-EFFICIENCY CLAIM IS UNSUPPORTED, NOT MERELY NARROW.")
    out("    It is unsupported in the specific sense that the only baseline we can")
    out("    model is one no customer would deploy. Removing it costs nothing,")
    out("    because it was never going to survive a customer architect's whiteboard.")
    out()
    out("  WHAT IS LEFT, AND IT IS NOT NOTHING. Three ML-adjacent claims that do not")
    out("  require an inference-efficiency comparison at all:")
    out("    1. ARITHMETIC RETARGETABILITY. NPU IP is int8-weights-only (Ethos-U85)")
    out("       or ~1% of int8 throughput at FP32 (Ceva). That is a capability")
    out("       statement about the rival, measurable from its own datasheet, and it")
    out("       is why an NPU cannot run the solver half of a hybrid loop at all.")
    out("    2. CO-RESIDENCY AND ONE DETERMINISTIC TICK, quantified in section 4:")
    out("       the pair fits 192 KB for a Science-Robotics-class policy and needs")
    out("       256-384 KB for the legged-gym default. A joint worst-case tick is")
    out("       something an NPU-plus-CPU split cannot state.")
    out("    3. MACIF. The answer to 'an inference macro beats you on TOPS/W' is")
    out("       'yes, and it plugs in here' -- which requires int4 to be a DECLARED")
    out("       format, not an assumed one.")
    out()
    out("  MACIF CONSEQUENCE OF int4, and it is the mirror image of fp16-vs-bf16:")
    out("    * A macro that silently substitutes int8 for int4 returns the RIGHT")
    out("      answer and loses 2x of bandwidth. That is a performance regression,")
    out("      not a wrong answer -- the OPPOSITE failure mode from bf16-for-fp16,")
    out("      and it must still be refused, because a codegen plan that assumed 2x")
    out("      the operand bandwidth will miss its deadline.")
    out("    * A macro that substitutes int4 for a declared int8 kernel is a WRONG")
    out("      ANSWER and must be refused in the same class as a missing fp64 path.")
    out("    * Substitution is therefore asymmetric in CONSEQUENCE and symmetric in")
    out("      POLICY: refuse both directions, report which kind of failure was")
    out("      avoided.")
    out()

    # =====================================================================
    out("=" * 104)
    out("(6) WHAT WOULD FALSIFY ANY OF THIS")
    out("=" * 104)
    out()
    out("  * A MEASURED NPU energy figure at a stated node and precision. It is the")
    out("    single missing input; everything in section 5 turns on its absence. An")
    out("    eval board and a power rail settles it -- the same instruction as")
    out("    falsifier F4.")
    out("  * A BATCH DIMENSION, and this is the one most likely to bite. External")
    out("    intensity is LINEAR in batch size and the differentiation threshold is")
    out("    ~2.6 MAC/byte, so int4 decode at 1.93 needs only batch 2 to cross it.")
    out("    Speculative decoding and multi-request serving both produce an effective")
    out("    batch >1. The model CANNOT currently express a batched decode (it would")
    out("    need the GEMM path plus a KV-growth term), so no batched number is")
    out("    quotable -- but the negative result above is explicitly a BATCH-1 DECODE")
    out("    result and must never be restated as an LLM result.")
    out("  * An L0 operand register file, already flagged as the top model extension:")
    out("    it raises on-chip intensity and would make the GEMM/ViT regime cheaper")
    out("    still. It does NOT touch the LLM regime, because the binding resource")
    out("    there is off chip.")
    out("  * A measured accuracy result at int4 on a control-relevant network. This")
    out("    module models bytes and cycles only; if int4 costs a policy its")
    out("    behaviour, every int4 row above is a datapath capability with no")
    out("    workload attached.")
    out()

    with open("perf_model/results_ml_envelope.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
