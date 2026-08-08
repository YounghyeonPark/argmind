# Archived agent team (accelerator project)

These nine agents were defined for the parked accelerator IP project: microarchitecture,
solver numerics, RTL design and verification, compiler/codegen, driver/runtime, market
intelligence, IP strategy and technical writing.

**They are archived, not active.** They were moved out of `.claude/agents/` on 2026-08-08 so
they no longer auto-load, because every one of their system prompts is saturated with
accelerator context — MACIF, descriptor formats, perf/W, bandwidth-first provisioning — which
would actively mislead work on the current sensor-framework project.

Kept because two things in them are reusable in spirit rather than in text: the deliberate
separation of design from verification (an agent that wrote the code is the wrong one to judge
whether it works), and the per-agent encoding of the specific failure mode each exists to
prevent. If a team is defined for the current project, take those two ideas and write fresh
prompts; do not copy these.
