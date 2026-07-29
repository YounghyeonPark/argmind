# Argmind sub-agent team

Project-scoped subagents in `.claude/agents/`. All of them automatically load
[CLAUDE.md](../../CLAUDE.md), which holds shared project context, established findings, and the
standards of evidence — role files stay role-specific rather than repeating it.

Invoke by name (`> use market-intel to ...`) or let routing pick one from its `description`.
Continue a running agent with its ID to keep its context instead of starting fresh.

## The team

| Agent | Owns | Active |
|---|---|---|
| **cm-architect** | Microarchitecture, performance model, PPA reasoning, ISA/descriptor design, the MACIF boundary | **Now** |
| **solver-numerics** | Solver/control algorithms, operation counts, precision and conditioning, correctness of BENCHMARKS.md | **Now** |
| **market-intel** | Competitor teardowns, IMC/CIM threat tracking, pricing benchmarks, customer-discovery material | **Now** |
| **ip-strategy** | Prior-art and patent landscape, trademark research, open-source license structure | **Now** |
| **tech-writer** | Whitepaper, specs, integration guides, open-source docs, claim auditing | **Now** |
| **compiler-codegen** | Solver codegen, residency planner, ML import path, simulators | Phase 1 |
| **driver-runtime** | Linux driver, runtime API, HALs, FPGA/SoC bring-up | Phase 1–2 |
| **rtl-design** | Synthesizable SystemVerilog implementation | Phase 2 |
| **rtl-verification** | Testbenches, protocol compliance, coverage, formal | Phase 2 |

## Why the roles are split this way

**rtl-design and rtl-verification are deliberately separate agents.** Verification independence
is the point — an agent that wrote the RTL is the wrong one to decide whether it works. Don't use
rtl-verification to fix the code it is testing.

**cm-architect and solver-numerics are separate** because the recurring failure mode in this
project has been hardware reasoning applied to algorithms it misunderstood. The matrix-vector
versus matrix-matrix distinction that overturned the tile-size decision belongs to
solver-numerics; what to do about it belongs to cm-architect. Consult both on anything at the
boundary.

**ip-strategy is research support, not counsel.** It never substitutes for a qualified attorney,
and it is written to say so.

## Working with them

Give a subagent the decision you need made, not just a topic — they return conclusions, and a
vague scope produces a survey instead of an answer. Run independent investigations concurrently
in one message. For anything that will end up in front of a customer, route it through
tech-writer for a claim audit, since unsupported numbers are the failure mode that costs most
here.

Adding an agent: keep the name lowercase-with-hyphens, write the `description` so routing can
tell when it applies, and put shared standards in CLAUDE.md rather than duplicating them.
