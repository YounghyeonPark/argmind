"""
CM-100 baseline SKU sizing — scratchpad bandwidth / banking / MAC array
======================================================================

Joins sram_cost.py (area + energy of the banked scratchpad) to
argmind_model.py (latency + energy on the frozen benchmark set) so the
bandwidth-first decision is made against BOTH cost and deadlines.

All figures are MODELLED ESTIMATES, 16nm-class, +/-2x, NOT QUOTABLE.
"""

from __future__ import annotations

import argmind_model as M
import sram_cost as S

REST_MM2 = 0.045 + 0.070 + 0.030      # sequencer, DMA+AXI/ACE-Lite, misc [EST]
TOP_MARGIN = 1.15                      # top-level route/CTS/margin [EST]
A_GATE = S.A_GATE_UM2
VEC_LANE_GATES = 2200                  # [EST] fp32 lane + transcendental share


def cfg(name, bw, nb, tile=16, lanes=8, seg=True, kb=256):
    return M.ArgmindCfg(name=name, tile=tile, vector_lanes=lanes,
                        scratchpad_kb=kb, scratchpad_bw_bpc=bw, n_banks=nb,
                        segmented_reduction=seg)


def area_breakdown(c: M.ArgmindCfg) -> dict:
    spad = c.spad.area_mm2()["total_mm2"]
    mac = S.mac_array_mm2(c.macs)
    vec = c.vector_lanes * VEC_LANE_GATES * A_GATE / 1e6
    sub = spad + mac + vec + REST_MM2
    return {"spad": spad, "mac": mac, "vec": vec, "rest": REST_MM2,
            "sub": sub, "total": sub * TOP_MARGIN}


def evaluate(c: M.ArgmindCfg) -> dict:
    rows, worst, pw = {}, [], []
    for b in M.BENCHMARKS:
        r = M.model_argmind(b, c)
        cpus = [(x, M.model_cpu(b, x)) for x in M.ALL_CPUS]
        eff = min(cpus, key=lambda t: t[1]["energy_uj"])[1]
        rows[b.name] = {
            "lat": r["latency_us"], "deadline": b.deadline_us,
            "meets": r["latency_us"] <= b.deadline_us,
            "uj": r["energy_uj"], "pw": eff["energy_uj"] / r["energy_uj"],
            "mw": r["avg_power_mw"],
        }
        if not rows[b.name]["meets"]:
            worst.append(b.name)
        if b.kind == "solver":
            pw.append(rows[b.name]["pw"])
    return {"rows": rows, "misses": worst,
            "solver_pw_min": min(pw), "solver_pw_max": max(pw),
            "area": area_breakdown(c), "pj_b": c.sram_pj_per_byte()}


CONFIGS = [
    ("A  64B/cyc  8bx8B  256MAC  8lane", cfg("A", 64, 8, 16, 8)),
    ("B 128B/cyc  8bx16B 256MAC  8lane", cfg("B", 128, 8, 16, 8)),
    ("C 128B/cyc  8bx16B 256MAC 16lane", cfg("C", 128, 8, 16, 16)),
    ("D 128B/cyc 16bx8B  256MAC 16lane", cfg("D", 128, 16, 16, 16)),
    ("E 256B/cyc  8bx32B 256MAC 16lane", cfg("E", 256, 8, 16, 16)),
    ("F 256B/cyc  8bx32B 256MAC 32lane", cfg("F", 256, 8, 16, 32)),
    ("G 256B/cyc 32bx8B  256MAC 32lane", cfg("G", 256, 32, 16, 32)),
    ("H 128B/cyc  8bx16B  64MAC 16lane", cfg("H", 128, 8, 8, 16)),
    ("I 128B/cyc  8bx16B 1024MAC 16lane", cfg("I", 128, 8, 32, 16)),
    ("J 512B/cyc  8bx64B 256MAC 32lane", cfg("J", 512, 8, 16, 32)),
]


def main():
    L = []

    def out(s=""):
        print(s)
        L.append(s)

    out("=" * 108)
    out("CM-100 SIZING — bandwidth vs banking vs MAC array vs vector lanes")
    out("MODELLED ESTIMATE, 16nm-class, +/-2x, NOT QUOTABLE EXTERNALLY")
    out("=" * 108)
    out()
    out("Benchmark set as currently defined in argmind_model.py (measured OSQP")
    out("iteration counts and corrected constraint-row count already applied).")
    out()

    hdr = (f"{'config':<36} {'area':>6} {'spad':>6} {'mac':>6} {'pJ/B':>6} "
           f"{'solver perf/W':>14} {'misses deadline':<28}")
    out(hdr)
    out(f"{'':<36} {'mm2':>6} {'mm2':>6} {'mm2':>6} {'':>6} {'min - max':>14}")
    out("-" * len(hdr))
    res = {}
    for label, c in CONFIGS:
        e = evaluate(c)
        res[label] = e
        a = e["area"]
        out(f"{label:<36} {a['total']:>6.3f} {a['spad']:>6.3f} {a['mac']:>6.3f} "
            f"{e['pj_b']:>6.2f} "
            f"{e['solver_pw_min']:>6.1f}-{e['solver_pw_max']:<7.1f} "
            f"{(', '.join(e['misses']) or 'none'):<28}")
    out()

    # ---- per-benchmark latency vs deadline ----
    out("-" * 108)
    out("LATENCY vs DEADLINE (us) — 'X' marks a miss")
    out("-" * 108)
    names = [b.name for b in M.BENCHMARKS]
    out(f"{'config':<36} " + " ".join(f"{n[:9]:>11}" for n in names))
    out(f"{'deadline':<36} " + " ".join(
        f"{b.deadline_us:>11.0f}" for b in M.BENCHMARKS))
    out("-" * 108)
    for label, c in CONFIGS:
        cells = []
        for n in names:
            r = res[label]["rows"][n]
            cells.append(f"{r['lat']:>10.0f}" + ("X" if not r["meets"] else " "))
        out(f"{label:<36} " + " ".join(f"{x:>11}" for x in cells))
    out()

    # ---- segmented reduction value ----
    out("-" * 108)
    out("VALUE OF SEGMENTED REDUCTION (beat may straddle two GEMV rows)")
    out("-" * 108)
    out(f"{'config':<26} {'bench':<12} {'seg ON':>10} {'seg OFF':>10} "
        f"{'penalty':>9} {'deliv B/cyc':>12}")
    for bw, nb in ((64, 8), (128, 8), (256, 8), (512, 8)):
        on = cfg("on", bw, nb, 16, 16, seg=True)
        off = cfg("off", bw, nb, 16, 16, seg=False)
        for name in ("QUAD-MPC", "ARM-MPC", "PWR-MPC"):
            b = [x for x in M.BENCHMARKS if x.name == name][0]
            r1 = M.model_argmind(b, on)
            r0 = M.model_argmind(b, off)
            out(f"{'bw=' + str(bw):<26} {name:<12} {r1['latency_us']:>10.1f} "
                f"{r0['latency_us']:>10.1f} "
                f"{r0['latency_us']/r1['latency_us']:>8.2f}x "
                f"{r0['delivered_bw_bpc']:>12.0f}")
    out()

    # ---- where does the bandwidth stop paying ----
    out("-" * 108)
    out("DIMINISHING RETURNS — QUAD-MPC per-iteration cycle budget")
    out("-" * 108)
    b = [x for x in M.BENCHMARKS if x.name == "QUAD-MPC"][0]
    out(f"{'bw B/cyc':>9} {'lanes':>6} {'gemv cyc':>9} {'vec+trans':>10} "
        f"{'seq':>5} {'total':>7} {'gemv share':>11} {'binding':<22}")
    for bw in (32, 64, 128, 256, 512):
        for lanes in (8, 16, 32):
            c = cfg("x", bw, 8, 16, lanes)
            gemv = b.gemv_macs_per_iter / c.gemv_macs_per_cycle(b.prec,
                                                                b.gemv_row_len)
            vt = (b.vector_ops_per_iter / lanes
                  + b.transcendental_per_iter * 4.0 / lanes)
            seq = 0.5 * b.scalar_ops_per_iter / c.seq_ipc
            tot = gemv + vt + seq
            bind = ("scratchpad BW" if gemv > vt else "VECTOR UNIT")
            out(f"{bw:>9} {lanes:>6} {gemv:>9.0f} {vt:>10.0f} {seq:>5.0f} "
                f"{tot:>7.0f} {100*gemv/tot:>10.0f}% {bind:<22}")
    out()
    out("  Once gemv cycles fall below the vector/transcendental cycles, MORE")
    out("  SCRATCHPAD BANDWIDTH BUYS NOTHING. On QUAD-MPC with 8 vector lanes")
    out("  that happens between 128 and 256 B/cycle. The model's headline")
    out("  '1512 us -> 162 us for 16 -> 256 B/cycle' is a bandwidth-only")
    out("  sensitivity; it silently assumed the vector unit scaled too.")
    out()

    # ---- minimum viable problem size for wide bandwidth ----
    out("-" * 108)
    out("MINIMUM PROBLEM SIZE THAT CAN USE THE WIDTH (fp32, 4 B/element)")
    out("-" * 108)
    out(f"{'n_var':>6} {'row B':>7} " + " ".join(
        f"{'bw' + str(bw):>10}" for bw in (64, 128, 256, 512)) +
        "   (delivered/peak, no segmented reduction)")
    for n_u in (1, 2, 3, 4, 6, 8, 12, 16, 24):
        n_var = 20 * n_u
        cells = []
        for bw in (64, 128, 256, 512):
            d = S.effective_bw_short_rows(n_var, 4, bw, False)
            cells.append(f"{100*d/bw:>9.0f}%")
        out(f"{n_var:>6} {4*n_var:>7} " + " ".join(f"{x:>10}" for x in cells))
    out()

    # ---- ADVERSARIAL 1: who actually needs the bandwidth? ----
    out("-" * 108)
    out("ADVERSARIAL CHECK 1 — which benchmark actually JUSTIFIES the bandwidth?")
    out("-" * 108)
    out("Deadline margin (deadline/latency; <1.0 = miss) at 256 MACs, 16 lanes:")
    out(f"{'bench':<12} " + " ".join(f"{'bw' + str(bw):>9}" for bw in
                                     (32, 64, 128, 256)))
    for b in M.BENCHMARKS:
        if b.kind != "solver":
            continue
        cells = []
        for bw in (32, 64, 128, 256):
            c = cfg("x", bw, 8, 16, 16)
            r = M.model_argmind(b, c)
            cells.append(f"{b.deadline_us / r['latency_us']:>8.2f}x")
        out(f"{b.name:<12} " + " ".join(f"{x:>9}" for x in cells))
    out()
    out("  ONE benchmark drives the whole bandwidth-first decision on LATENCY:")
    out("  ARM-MPC. Every other solver row has >8x deadline margin at 64 B/cycle.")
    out("  If ARM-MPC left the frozen set, 64 B/cycle would meet every solver")
    out("  deadline and the latency argument for a wide scratchpad would vanish.")
    out("  The remaining argument for width is ENERGY (shorter active window,")
    out("  cheaper per-byte access from wider bank words) -- consistent with the")
    out("  settled finding that the advantage is energy, not speed.")
    out()

    # ---- ADVERSARIAL 2: leakage and duty cycle ----
    out("-" * 108)
    out("ADVERSARIAL CHECK 2 — perf/W depends on POWER GATING between calls")
    out("-" * 108)
    out("The model charges static power over the ACTIVE window only. If the CM")
    out("cannot be power-gated between calls, leakage is charged over the whole")
    out("call period and low-duty-cycle workloads collapse:")
    out(f"{'bench':<12} {'duty':>7} {'gated uJ':>10} {'ungated uJ':>11} "
        f"{'perf/W gated':>13} {'perf/W ungated':>15}")
    c = cfg("x", 128, 8, 16, 16)
    for b in M.BENCHMARKS:
        if b.kind != "solver":
            continue
        r = M.model_argmind(b, c)
        cpus = [(x, M.model_cpu(b, x)) for x in M.ALL_CPUS]
        eff = min(cpus, key=lambda t: t[1]["energy_uj"])[1]
        period_us = 1e6 / b.calls_per_sec
        duty = r["latency_us"] / period_us
        leak_extra = c.static_mw * 1e-3 * (period_us - r["latency_us"]) * 1e-6 * 1e6
        ung = r["energy_uj"] + leak_extra
        out(f"{b.name:<12} {100*duty:>6.1f}% {r['energy_uj']:>10.2f} "
            f"{ung:>11.2f} {eff['energy_uj']/r['energy_uj']:>12.1f}x "
            f"{eff['energy_uj']/ung:>14.1f}x")
    out()
    out("  Ungated, QUAD-MPC and PWR-MPC fall to single digits and EKF-24 goes")
    out("  BELOW the 5x kill criterion. Coarse-grain power gating (or at minimum")
    out("  full clock gating plus a retention-mode scratchpad) is therefore a")
    out("  REQUIREMENT of the architecture, not an optimisation. The 15 mW static")
    out("  figure is itself an [EST] and is now load-bearing: measure it in Phase 1.")
    out()

    # ---- ADVERSARIAL 3: is the falling pJ/byte real? ----
    out("-" * 108)
    out("ADVERSARIAL CHECK 3 — why does pJ/byte FALL as bandwidth rises?")
    out("-" * 108)
    for bw, nb in ((64, 8), (128, 8), (256, 8), (512, 8), (128, 16), (256, 32)):
        s = S.SpadCfg(256, nb, bw // nb)
        e = s.energy_pj_per_byte()
        out(f"  {bw:>4} B/cyc {nb:>3} banks x {bw//nb:>3} B: macro "
            f"{e['macro_16_pj_b']:.3f} + wire {e['wire_pj_b']:.3f} + mux "
            f"{e['mux_pj_b']:.3f} = {e['total_pj_b']:.3f} pJ/B")
    out()
    out("  Mechanism: a read activates a whole wordline, so the bitline energy is")
    out("  already spent whether you sense 64 bits or 256 of that row. Published")
    out("  45nm data [A2] confirms it -- 64b -> 512b access on the same macro is")
    out("  1.2-2x CHEAPER PER BYTE. So the prompt's hypothesis (per-byte energy")
    out("  rises with bandwidth) is WRONG for wider bank words.")
    out("  It is right about bank COUNT, but the effect is small and shows up in")
    out("  AREA (+46% scratchpad for 32 banks vs 8) far more than in energy.")
    out("  RESIDUAL RISK: if the foundry macro implements aggressive bitline")
    out("  gating (only the selected column group discharges), wide reads stop")
    out("  being cheap and pJ/byte goes FLAT with width -- our 128->256 B/cycle")
    out("  energy improvement would then be ~0 instead of -20%. This must be")
    out("  checked against the real compiler datasheet in Phase 1.")
    out()

    # ---- MAC array minimum ----
    out("-" * 108)
    out("MAC ARRAY MINIMUM — what actually sizes the array (128 B/cyc, 16 lanes)")
    out("-" * 108)
    out(f"{'tile':>5} {'MACs':>6} {'mm2':>7} " + " ".join(
        f"{b.name[:9]:>10}" for b in M.BENCHMARKS if b.name != 'SLM-DECODE'))
    for tile in (4, 8, 12, 16, 24, 32):
        c = cfg("x", 128, 8, tile, 16)
        cells = []
        for b in M.BENCHMARKS:
            if b.name == 'SLM-DECODE':
                continue
            r = M.model_argmind(b, c)
            m = b.deadline_us / r["latency_us"]
            cells.append(f"{m:>9.2f}" + ("X" if m < 1.0 else " "))
        out(f"{tile:>5} {tile*tile:>6} {S.mac_array_mm2(tile*tile):>7.3f} "
            + " ".join(f"{x:>10}" for x in cells))
    out("  (values are deadline margin; X = miss. SLM-DECODE omitted: it is")
    out("   DRAM-bandwidth-bound and indifferent to the array.)")
    out()

    # ---- CAPACITY: bigger is worse ----
    out("-" * 108)
    out("SCRATCHPAD CAPACITY — bigger is WORSE for energy (128 B/cyc, 16 banks)")
    out("-" * 108)
    out(f"{'KB':>6} {'spad mm2':>9} {'pJ/B':>6} " + " ".join(
        f"{b.name[:9]:>10}" for b in M.BENCHMARKS if b.kind == 'solver')
        + "   (uJ/call)")
    for kb in (64, 128, 256, 512, 1024):
        c = cfg("x", 128, 16, 16, 16, kb=kb)
        cells = [f"{M.model_argmind(b, c)['energy_uj']:>10.1f}"
                 for b in M.BENCHMARKS if b.kind == 'solver']
        out(f"{kb:>6} {c.spad.area_mm2()['total_mm2']:>9.3f} "
            f"{c.sram_pj_per_byte():>6.2f} " + " ".join(f"{x:>10}" for x in cells))
    out()
    out("  ARM-MPC's working set is 861 KB and does NOT fit in 256 KB. Growing the")
    out("  scratchpad to hold it costs energy rather than saving it: the 622 KB")
    out("  that must stream per call needs only 0.3 GB/s (fully overlapped by")
    out("  compute, zero latency cost, ~7 uJ of DRAM energy), while 64 KB banks")
    out("  instead of 16 KB banks make EVERY scratchpad read ~1.8x more expensive")
    out("  and ARM-MPC issues ~85 MB of scratchpad reads per call.")
    out("  -> Residency of the HOT inner-loop set is what matters. Residency of the")
    out("     whole problem is not a goal, and buying it is a net energy loss.")
    out("     If a future SKU needs more capacity, it should be a SECOND TIER, not")
    out("     a wider first tier. MODEL LIMITATION: one pJ/byte is applied to all")
    out("     reads, so this model cannot yet evaluate a tiered scratchpad.")
    out()

    # ---- recommendation detail ----
    rec = cfg("CM-100", 128, 16, 16, 16)
    e = evaluate(rec)
    a = e["area"]
    sp = rec.spad
    out("=" * 108)
    out("RECOMMENDATION — CM-100")
    out("=" * 108)
    out(f"  scratchpad     256 KB, 16 single-port banks x 8 B/cycle = 128 B/cycle")
    out(f"                 bank geometry: {sp.geom.rows} rows x {sp.geom.cols} cols, "
        f"{sp.geom.mux}:1 col mux, {sp.geom.folds} fold(s)")
    out(f"  MAC array      256 MACs (16x16), bf16 mult / fp32 accumulate")
    out(f"  vector unit    16 fp32 lanes")
    out(f"  frequency      800 MHz")
    out()
    out(f"  AREA  scratchpad {a['spad']:.3f} | MAC {a['mac']:.3f} | vector "
        f"{a['vec']:.3f} | seq+DMA+misc {a['rest']:.3f}")
    out(f"        subtotal {a['sub']:.3f} mm2, +15% margin = {a['total']:.3f} mm2")
    out(f"  ENERGY scratchpad read {e['pj_b']:.2f} pJ/byte "
        f"(v0 assumed 0.40 -> {e['pj_b']/0.40:.1f}x low)")
    out(f"  LEAK  scratchpad {sp.leakage_mw():.2f} mW @85C / "
        f"{sp.leakage_mw(True):.2f} mW @125C")
    out()
    out(f"  {'benchmark':<12} {'deadline':>9} {'latency':>9} {'margin':>8} "
        f"{'uJ/call':>9} {'avg mW':>8} {'perf/W':>7}  verdict")
    for b in M.BENCHMARKS:
        r = e["rows"][b.name]
        out(f"  {b.name:<12} {r['deadline']:>9.0f} {r['lat']:>9.1f} "
            f"{r['deadline']/r['lat']:>7.2f}x {r['uj']:>9.2f} {r['mw']:>8.2f} "
            f"{r['pw']:>7.1f}  {'MEETS' if r['meets'] else 'MISSES'}")
    out()
    out(f"  solver perf/W band: {e['solver_pw_min']:.1f}x - "
        f"{e['solver_pw_max']:.1f}x  (kill criterion 5x)")
    out()

    import os
    dest = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        "results_cm100_sizing.txt")
    with open(dest, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")


if __name__ == "__main__":
    main()
