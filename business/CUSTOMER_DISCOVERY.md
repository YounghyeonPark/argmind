# Argmind CM — Customer Discovery Program (v1.0, pre-registered 2026-07-29)

**Status of evidence in this document.** Everything below is *plan*. **We have conducted zero
customer conversations. No customer opinion, quote, price, or requirement appears anywhere in
this document, because none exists.** Every external fact cited here comes from a public source
with a URL and is labelled by type (vendor claim / company blog / peer-reviewed / secondary
reporting). When interviews begin, their content goes in `discovery/` records (§8), never
inlined here.

**Pre-registration commitment.** §7 (falsification criteria) is written *before* the first call
and is frozen. Changing a threshold in §7 after interviews have started requires writing a dated
amendment that states what the old threshold was and what result triggered the change. This
exists so a disconfirming result cannot be quietly re-interpreted as a pass.

---

## 1. The finding that changes the program design, before any interview happens

**The beachhead as written in PLAN.md §2 — "embedded MPC + state estimation
(robotics/automotive)" — names a *workload* and a *place*, not a *buyer*. Those come apart
badly, and discovery has to be designed around the split rather than discovering it in month
three.**

Argmind sells synthesizable RTL. Only an organisation that tapes out silicon can be a licensee.
That eliminates almost the entire robotics segment as a *customer*:

- Robot and drone makers buy modules. Auterion's Skynode integrates a flight controller plus
  mission computer as a purchased unit ([PX4 docs](https://docs.px4.io/main/en/companion_computer/auterion_skynode));
  NVIDIA sells Jetson as a module at 40–130 W configurable for Thor-class parts
  ([NVIDIA](https://developer.nvidia.com/embedded/jetson-modules)). Among humanoid makers,
  secondary reporting identifies Tesla as the notable in-house-silicon case; Figure's in-house
  investment is described as the VLA model, not an SoC
  ([secondary, 2026](https://medium.com/@asarav/humanoid-robots-in-2026-where-the-industry-actually-stands-6ae3dc0c7be5)).
  Treat all such rankings as low-quality secondary sources — but the direction is not in doubt
  and is cheap to verify in the first three calls.
- The organisations that *can* license are automotive/industrial semiconductor vendors, a handful
  of Tier-1s with captive ASIC groups, and RISC-V SoC vendors.

So the program has **two distinct interview populations with different question sets**:

| Population | What they can give us | What they cannot |
|---|---|---|
| **Licensees** (semis, captive-ASIC Tier-1s, RISC-V SoC vendors) | A deal; integration and safety requirements; price anchors | Credible workload pain — they are one or two steps removed from the control engineer |
| **Demand witnesses** (robot/drone/vehicle/drive builders, solver-software vendors) | Whether the pain is real, what it costs them today, deadline/power reality | A purchase order for RTL |

A licensee will ask "which of my customers is asking for this?" A demand witness will ask "which
chip can I buy it in?" **Neither population alone can produce a go decision.** The program must
close that loop deliberately: get demand witnesses on record about the pain, then use that
(anonymised, with permission) as the reason a licensee should care.

**Second pre-interview finding, and it is uncomfortable.** The PHASE0_FINDINGS.md teardown
covered NPU IP and found no arithmetic and no control plane. It did not cover the products
actually sitting in the socket we are targeting, and two of them are much closer to us than any
NPU:

| Product | What it is | Node / precision | Evidence class |
|---|---|---|---|
| **Infineon AURIX TC4x PPU** | 32-bit RISC core + SIMD vector unit, 128-bit or 256-bit vector width, with a dedicated vector closely-coupled memory (VCCM); Infineon positions it for "complex control algorithms" and NN evaluation while TriCore CPUs run time-critical control | Vector width published (128/256-bit); **precision set, TOPS/W, and process node not published** | [Infineon product page](https://www.infineon.com/product-information/new-ppu-simd-vector-dsp) (vendor); [emmtrix wiki](https://www.emmtrix.com/wiki/Infineon_AURIX_TC4x_Parallel_Processing_Unit_(PPU)) (secondary, reporting Infineon) |
| "up to 78× vs a single TriCore 1.8 core" | — | **Vendor claim, benchmark not identified, node and precision not published.** Not comparable to anything. Do not repeat this number in collateral even as a competitor figure without a source workload. | secondary reporting a vendor claim |
| **TI C2000 Control Law Accelerator (CLA)** | Fully programmable independent **32-bit floating-point** coprocessor for math-intensive real-time control, executing in parallel with the C28x CPU; ships today | FP32 confirmed; throughput/node not published in the cited doc | [TI CLA software guide](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/intro.html) (vendor) |

This matters for discovery, not just for the teardown. A licensee's honest answer to "would you
license a control/optimization accelerator" may be **"we already ship a vector DSP next to the
control CPU."** Note carefully what kind of gap that is: the PPU is a **SIMD vector DSP with a
scalar core and closely-coupled memory** — that is not an absence of arithmetic and not an
absence of a control plane. If it turns out to run our benchmark set adequately, the
"incumbents cannot retarget in software" conclusion in CLAUDE.md holds against *NPU IP* but does
**not** hold against *automotive MCU vector DSPs*, and that would be a feature gap, not a
structural one. **Interview question Q-A4 and Q-L7 exist specifically to break this open, and
the teardown owes a PPU/CLA row before the go/no-go.**

**Third, the strongest disconfirming public fact we already have: production MPC ships today, at
volume, on ordinary automotive microcontrollers with no accelerator.** ODYS states its embedded
MPC and real-time optimization software "is currently running on more than 3 million vehicles
worldwide," including production engine ECUs developed with General Motors
([ODYS](https://www.odys.it/) — vendor claim, volume not independently verified;
[Bemporad, IMT Lucca talk, 2023](http://cse.lab.imtlucca.it/~bemporad/talks/mpc_9_11_2023.pdf) — academic slides
describing the GM production work; also
[Model Predictive Control of Turbocharged Gasoline Engines for Mass Production](https://www.researchgate.net/publication/324235335_Model_Predictive_Control_of_Turbocharged_Gasoline_Engines_for_Mass_Production), secondary index of a paywalled paper).
Separately, a summary of the acados paper reports HIL runs on a dSPACE MicroAutoBox II holding
computation under 10 ms against a 50 ms sampling time for an engine-control task
([secondary summary](https://www.alphaxiv.org/overview/1910.13753v3); **not verified against the
primary paper — do that before citing**). A 5× deadline margin on a prototyping box is the
shape of "we are not deadline-limited."

That is entirely consistent with PHASE0_FINDINGS.md §0 (a CPU meets **5 of 5** solver
deadlines) and it is the reason §7's falsification criteria are aimed at the *energy* claim
rather than the *latency* claim. **The purchase-driver question, not the feasibility question, is
what discovery has to answer.**

---

## 2. Where production embedded optimization demonstrably lives (evidence for the segmentation)

Only segments with public evidence of *production* deployment are targeted. Research-only
segments are excluded from the target list regardless of how enthusiastic they would be.

| Segment | Public evidence that production embedded optimization exists there | Class |
|---|---|---|
| **Automotive powertrain / vehicle control** | ODYS embedded MPC on 3M+ vehicles, GM production ECUs (above) | vendor claim + academic talk |
| **ADAS / driver assistance** | openpilot uses acados for lateral and longitudinal MPC; solver moved from qpOASES to HPIPM explicitly for better bounds on time-to-usable-solution ([comma.ai blog](https://blog.comma.ai/0810release/), [source](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py)) | primary source code + company blog |
| **Automotive/autonomy solver tooling** | Embotech FORCESPRO marketed for motion planning, torque vectoring, hybrid energy management, real-time trajectory optimization ([Embotech](https://www.embotech.com/softwareproducts/forcespro/overview/)) | vendor claim |
| **Medium-voltage drives / power electronics** | ABB Model Predictive Pulse Pattern Control (MP3C) in ACS6080 / ACS8080 medium-voltage drives; offline-computed optimized pulse patterns with switching instants adjusted in real time ([ABB news](https://new.abb.com/news/detail/74747/new-mp3c-control-method-maximizes-efficiency-and-dynamics), [ABB innovation](https://www.abb.com/global/en/company/innovation/news/predictive-control-revolutionizes-variable-speed-drive-control), [IEEE paper hosted by ABB](https://library.e.abb.com/public/2a438a08d9a3445b8d2b5c20a7775e4b/IEEE_Model_Predictive_Pulse_Pattern_Control_with_very_fast_transient_responses.pdf)) | vendor + peer-reviewed |
| **Synchronous motor control** | Embedded MPC with certified real-time optimization for synchronous motors ([Bemporad et al., IEEE TCST preprint](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeecst-synchro-motor.pdf)) | peer-reviewed preprint |
| **Legged robots / humanoids** | Boston Dynamics describes Atlas's controller as an MPC that solves an optimization to choose actions in real time, running onboard ([Boston Dynamics blog](https://bostondynamics.com/blog/flipping-the-script-with-atlas/)) | company blog |
| **Robot motion planning** | A decade of accelerator work incl. an FPGA motion-planning accelerator reporting ~3 orders of magnitude speedup and >1 order of magnitude power improvement, and granted US patents on hardware collision detection ([Duke MICRO 2016](https://users.cs.duke.edu/~lkw34/papers/realtime-robotics-micro2016.pdf), [US 12,330,310](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12330310)) | peer-reviewed + patent |

Two reads to carry into the calls. First, **power electronics and motor drives are the segment
where the deadline is genuinely brutal** (our PWR-MPC benchmark is **40 kHz / 25 µs finite-control-set MPC** since the 2026-07-30 re-spec, and it is the
one where PHASE0_FINDINGS.md §4 shows zero-copy is worth 1.9× and decides whether the deadline
is met at all). Second, **ABB's shipping approach moves the optimization offline** (precomputed
optimal pulse patterns, real-time adjustment of switching instants) — which is exactly the
"solve it offline and don't buy your accelerator" answer we must ask about directly (Q-A5).

The motion-planning accelerator literature is the closest thing to a precedent for a licensable
control accelerator, and it is worth knowing why it did not become widespread IP. Ask it as a
question, do not assume the answer.

---

## 3. Target list — 25 organisations

**Reachability scale.** Deliberately harsh; it rates the probability that a *pre-revenue solo
founder with no silicon and no reference customer* gets a substantive 30-minute call with a
technically decisive person within 90 days.

- **R1 (high, >50%)** — public engineers, open-source presence, small orgs, or a live commercial
  motive to talk to a potential partner.
- **R2 (medium, 15–40%)** — reachable via a named conference, standards body, or a specific
  warm-intro path; expect several bounces.
- **R3 (low, <10%)** — needs a referral, a reference customer, or an inbound trigger we do not
  yet have. Attempt only after we have R1/R2 evidence to lead with.

**"Licensee?"** = can this organisation sign an RTL license, realistically.

### A. Automotive / industrial semiconductor vendors — the actual licensees (6)

| # | Org | Who to approach | Why they'd plausibly care | Licensee? | Reach |
|---|---|---|---|---|---|
| 1 | **Infineon** (AURIX, automotive MCU) | Concept Engineer / Product Definition, Automotive Microcontrollers; also Systems Architect, e-mobility inverter | Owns the motor-drive and zonal-control socket; already invested in a SIMD vector DSP (PPU) for "complex control algorithms," so the *problem* is validated inside the company. Also a Quintauris founder. | Yes | **R3** — biggest, most self-sufficient, and the most likely to say "we built that." Precisely why the conversation is high-information. Route: embedded world, or the PPU tooling ecosystem. |
| 2 | **NXP** | Systems Architect / Product Line Manager, S32 vehicle control & zonal | Zonal/vehicle-motion-control roadmap; licenses third-party IP routinely; Quintauris founder | Yes | **R3** |
| 3 | **Renesas** | Principal Architect, Automotive Digital Products (RH850 / R-Car) and RZ/V industrial | RZ/V2H explicitly pairs application cores + real-time cores so vision inference and motion control coexist on one device ([secondary](https://noizz.io/insights/nvidia-jetson-top-alternatives)) — the exact consolidation story we sell | Yes | **R3** |
| 4 | **STMicroelectronics** | Product Definition, Stellar automotive MCU; separately motor-control MCU marketing | Stellar targets zonal + electrification control; ST joined Quintauris as sixth member ([Hackster](https://www.hackster.io/news/risc-v-joint-venture-quintauris-grows-to-six-as-stmicroelectronics-joins-the-party-b3936a803e0d)) | Yes | **R2–R3** |
| 5 | **Microchip** | Product Marketing Manager, dsPIC/32-bit motor control; separately PolarFire SoC (RISC-V) | Smaller teams, more accessible than the top four; PolarFire SoC is a RISC-V + FPGA fabric part, so an FPGA-proven soft core is *directly* usable — a real path to a paid evaluation without a tape-out | Yes | **R2** — best licensee-side risk/reward on the list |
| 6 | **Telechips** (KR, automotive SoC) | Architecture / IP sourcing lead | Mid-size automotive SoC vendor that competes by integrating third-party IP rather than building everything; genuinely buys IP from small vendors | Yes | **R2** — the reachable adjacent. Take this call before chasing #1–#4. |

> Deliberately excluded, with reasons: **TI** (builds its own control accelerator — CLA — and is
> structurally averse to third-party compute IP; R3 with low information yield), **Mobileye**
> (captive EyeQ), and China ADAS SoC vendors (Horizon Robotics, Black Sesame, SemiDrive — often
> R1/R2 reachable and hungry for differentiators, but export-control exposure and IP-protection
> risk make them a deliberate later decision, not a Phase-0 default). Revisit both after §7
> evaluation.

### B. Automotive Tier-1s — mostly specifiers, occasionally licensees (4)

| # | Org | Who to approach | Why they'd plausibly care | Licensee? | Reach |
|---|---|---|---|---|---|
| 7 | **Robert Bosch** (Cross-Domain Computing Solutions; also Bosch's ASIC activity) | Chief Engineer / Function Owner, Vehicle Motion Control; separately Semiconductor Architecture | Both a Tier-1 and a silicon designer; Quintauris founder; publishes on MPC for vehicle dynamics | Sometimes | **R3** |
| 8 | **ZF** | Chief Engineer, Vehicle Motion Control software (cubiX-line integrated motion control) | Integrated motion control across brake/steer/damper is a multi-actuator constrained optimization — the archetypal MPC socket, and it runs on a cost-capped ECU | Rarely (specifier) | **R3** |
| 9 | **Aptiv** | Systems Architect, zonal / Smart Vehicle Architecture | Zonal consolidation makes "one accelerator serving control + ML" a platform-level question | Rarely | **R3** |
| 10 | **Vitesco / Schaeffler** (powertrain, inverters) | Development lead, inverter control / power electronics | Traction-inverter control at high switching frequency is the automotive analogue of PWR-MPC; efficiency gains have direct vehicle-range value | Rarely | **R2–R3** |

> Tier-1s are worth interviewing mainly to learn **what they would require of a semi**, and to
> generate the pull that makes segment A answer. Do not expect a license. Their value is that a
> named Tier-1 saying "we would ask our silicon supplier for this" is the single most persuasive
> artefact we can carry into an R3 semiconductor conversation.

### C. Industrial drives, motor control, power electronics (5)

| # | Org | Who to approach | Why they'd plausibly care | Licensee? | Reach |
|---|---|---|---|---|---|
| 11 | **ABB** (Drives / Corporate Research) | Principal Scientist or R&D Manager, drive control, in the group that publishes MP3C | The best-documented production MPC user in the world. They already ship model-predictive control in medium-voltage drives and publish openly about it — so the technical conversation is possible without NDA | Rarely (buys silicon/FPGA) | **R2** — publishing researchers answer email. Highest-quality technical conversation available to us. |
| 12 | **Danfoss Drives** | R&D Manager, control platform | Volume VFDs; switching-loss/efficiency economics are direct product value | No | **R2** |
| 13 | **Beckhoff Automation** | Product Manager, TwinCAT motion / control platform | Industrial PC-based control; sells advanced control functions as software products, so "what is the compute constraint" has a direct commercial answer | No | **R2** |
| 14 | **Yaskawa** | Servo/drive control R&D lead | High-rate servo loops; sub-50 µs control periods | No | **R3** |
| 15 | **imperix** (CH, power-electronics control platforms) | Founder / CTO | Small vendor selling real-time control hardware to power-electronics researchers and industry; will talk, and knows exactly where the compute ceiling bites in MPC-for-converters | Possibly (FPGA IP) | **R1** |

> Note the pattern: **this segment is far more reachable than automotive and has the hardest
> deadlines, but its members buy FPGAs and MCUs rather than licensing RTL.** That is a live
> alternative-beachhead signal — see §7 F3.

### D. Robotics / drone / vehicle builders — demand witnesses (5)

| # | Org | Who to approach | Why they'd plausibly care | Licensee? | Reach |
|---|---|---|---|---|---|
| 16 | **comma.ai** | Engineer on the controls/planning stack | Ships acados MPC in a consumer product; publicly documented solver choices driven by *time-to-usable-solution* guarantees; culture of talking in public | No | **R1** — the single most reachable production MPC user in existence |
| 17 | **ANYbotics** | Locomotion / control team lead | Legged robot with onboard whole-body control under a battery budget; ETH lineage means solver internals are discussable | No | **R1–R2** |
| 18 | **Agility Robotics** | Controls lead | Humanoid, onboard optimization, runtime/thermal constrained | No | **R2** |
| 19 | **Skydio** | Autonomy / controls engineer | Onboard trajectory optimization on a small airframe where every watt is flight time — the cleanest possible "energy is the purchase driver" test | No | **R2** |
| 20 | **Auterion** | Product / systems lead, Skynode | Sells the flight-computer *module*, so they are the layer that would consume an Argmind-bearing SoC and can say whether they would care ([Skynode](https://docs.px4.io/main/en/companion_computer/auterion_skynode)) | No | **R1–R2** |

> Boston Dynamics is the highest-value witness in this segment (published MPC, onboard, three
> compute units) but is **R3** and unlikely to discuss compute budgets. Approach only after
> §7's first checkpoint, with ANYbotics/Agility findings in hand.

### E. RISC-V SoC vendors and the automotive RISC-V platform layer (3)

| # | Org | Who to approach | Why they'd plausibly care | Licensee? | Reach |
|---|---|---|---|---|---|
| 21 | **SiFive** | Product Manager, automotive/real-time cores; ecosystem/partnerships lead | Announced a Quintauris partnership (Jan 2026) to standardise high-performance RISC-V IP for zonal controllers and ADAS ([secondary](https://en.wikipedia.org/wiki/Quintauris)); an accelerator that attaches to their cores is complementary, not competitive | Partner/channel | **R2** |
| 22 | **Andes Technology** | Product marketing, vector/DSP cores; partner-program manager | Sells RVV vector cores; will have a hard, informed opinion on whether RVV already covers our workload — i.e. this is a call designed to *hurt* | Partner/channel | **R2** — best "can RVV just do this?" test available |
| 23 | **Quintauris** | Technical program lead, RT-Europa platform spec | Defines the real-time automotive RISC-V platform spec (RT-Europa, available to customers/partners from Jan 2026; Nuclei alignment announced June 2026 — [Quintauris](https://www.quintauris.com/quintauris-rt-europa-risc-v-platform-automotive/), [Nuclei](https://www.quintauris.com/quintauris-nuclei-automotive-real-time-risc-v/)). Six-shareholder reach into Bosch/Infineon/NXP/Nordic/Qualcomm/ST | No (spec body) | **R2** — highest leverage per call on the entire list |

### F. Embedded-optimization software vendors — channel and expert witnesses (2 + 1 note)

| # | Org | Who to approach | Why they'd plausibly care | Licensee? | Reach |
|---|---|---|---|---|---|
| 24 | **Embotech** | CTO / Head of Engineering | FORCESPRO is the code-generating solver our codegen would have to displace or partner with; they know every automotive MPC program in Europe and what the compute constraints actually are | Partner | **R1–R2** |
| 25 | **ODYS** | CTO / Scientific Advisor | Claims MPC in production on 3M+ vehicles; if anyone knows whether automotive control engineers are compute-limited, it is them — and their answer may be "no," which is exactly why we ask | Partner | **R1–R2** |

> Also approach, not counted in the 25 because they are ecosystem rather than customers: the
> **acados / syscop group (Univ. Freiburg)** — [docs.acados.org](https://docs.acados.org/) —
> for benchmark credibility and referrals into industrial users; and **MathWorks** (MPC Toolbox +
> Embedded Coder, which already integrates FORCESPRO —
> [MathWorks](https://www.mathworks.com/help/mpc/ug/integrate-with-embotech-forces-pro-solvers.html))
> as the eventual toolchain path to every MPC engineer in industry.

### Sequencing (this is the plan, not a suggestion)

1. **Weeks 1–3 — R1s only** (comma.ai, imperix, Embotech, ODYS, Auterion, ANYbotics): cheap,
   fast, and they calibrate our vocabulary before we spend an R3 introduction. Expect to
   discover that some of our questions are naive; that is the point of going here first.
2. **Weeks 3–6 — R2 technical depth** (ABB, Andes, Quintauris, Microchip, Telechips, Skydio,
   Danfoss, Beckhoff).
3. **Weeks 6–12 — R3 licensees** (Infineon, NXP, Renesas, ZF, Bosch, Aptiv), armed with
   anonymised demand-witness evidence and the published benchmark harness.
4. Ask **every** interviewee for one referral (§8). Referrals are the only realistic mechanism
   by which an R3 becomes an R2.

---

## 4. Interview guide — built to produce a "no"

**Rules of engagement.**

- The first 20 minutes contain **no product description**. If asked, one sentence: "a
  licensable hardware block for optimization and matrix math in embedded control — but I'd
  rather hear how you do it today before I bias you."
- **Never demo in a first call.** A demo converts an interview into a sales call and destroys
  the evidential value of everything said afterwards.
- Ask for **numbers with units and context** (rate, deadline, hardware, precision, power
  budget). Record an unqualified number as *unqualified*, not as data.
- When they say something that contradicts our thesis, **ask a follow-up that deepens the
  contradiction**, not one that softens it.
- Volunteer our own weak results. "Our modelling says a Cortex-A55 already makes several of
  these deadlines" earns more truth than any claim, and it is true (PHASE0_FINDINGS.md §1).

### 4.0 Framing script (60 seconds)

> "I'm doing architecture research on hardware for embedded optimization — MPC, Kalman
> filtering, that class of solver. I'm not selling anything; there's no product to buy. I have
> modelled results and I'm trying to find out where they're wrong. Fifteen questions, none
> confidential, and I'll send you the benchmark data whether or not this goes anywhere. If a
> question touches something you can't discuss, just say 'skip.' May I take written notes?"

### 4.1 What they run today (both populations)

- **Q-A1.** Walk me through the control stack on your most compute-constrained product. Which
  loop runs fastest, and at what rate?
- **Q-A2.** Is there an optimization solve in that loop, or is it PID / gain-scheduled / LQR /
  a learned policy? *(If there is no solver at all, that is a finding — record it and continue;
  do not steer them toward one.)*
- **Q-A3.** Which solver and which toolchain? (acados, FORCESPRO, ODYS, OSQP, qpOASES, HPIPM,
  CVXPYgen, hand-written, in-house.) Who wrote it, and is that person still there?
- **Q-A4.** What silicon does it execute on — part number if you can share? Does it use a
  vector unit, DSP, or accelerator (CLA, PPU, NEON/SME, RVV, FPGA fabric), or scalar CPU only?
  **Was using the accelerator worth the effort?** *(This is the AURIX-PPU / TI-CLA probe from
  §1. If the answer is "we use the vector DSP and it's fine," our structural-gap claim is in
  trouble in this segment.)*
- **Q-A5.** How much of the optimization is precomputed offline versus solved online? Has anyone
  proposed moving *more* of it offline (lookup tables, explicit MPC, precomputed pulse patterns,
  a distilled network) to make the compute problem disappear? *(ABB's shipping MP3C does exactly
  this. If offline precomputation is the standard escape hatch, our TAM shrinks.)*
- **Q-A6.** What is the worst-case execution time, and how do you establish it? What happens on
  the tick when the solver does not converge?

### 4.2 Constraint diagnosis — deadline, power, or neither (both populations)

The purpose here is to find out that they are **not** constrained. Ask in this order; do not
lead.

- **Q-B1.** What is the binding constraint on that product right now? *(Open question. Say
  nothing. Whatever they name first is the datum — record it verbatim before probing.)*
- **Q-B2.** What fraction of your control-loop period does the solve consume today? What is the
  margin, and who tracks it?
- **Q-B3.** If the solve got 3× faster tomorrow at the same power, what would you actually do
  with that? Ship the same product? Longer horizon, higher rate, more constraints, a better
  model? *(A shrug here is a strong negative signal on the latency story.)*
- **Q-B4.** If the solve used 20× less energy at the same speed, what would change? Does anyone
  in your organisation have a number for the power budget of the control compute, separately
  from the rest of the system? *(**This is the single most important question in the
  program.** Our advantage is **14–22× perf/W and only 1.0–2.1× latency** (modelled, ±2×, at the
  CM-100 configuration; a CPU meets every solver deadline in the frozen set).
  If nobody can name a control-compute power budget, our headline advantage has no buyer.)*
- **Q-B5.** Where does the power actually go in your product? Rank compute against motors,
  actuators, radios, thermal. *(In a legged robot or a VFD, actuators dominate by orders of
  magnitude. Get them to say it.)*
- **Q-B6.** Have you ever removed a feature, lowered a rate, shortened a horizon, or simplified
  a model because of compute? Tell me about that specific decision. *(A concrete war story is
  the only credible evidence of felt pain. Absence of a war story across many interviews is
  itself a result.)*
- **Q-B7.** Have you ever paid money — silicon cost, engineering cost, NRE — to make control
  compute faster or cooler? How much? *(Past spend predicts future spend far better than
  expressed interest.)*
- **Q-B8.** Is jitter or worst-case determinism a problem for you today, or is average
  throughput what you care about? Does anyone ask you for a WCET number you struggle to
  produce? *(Tests whether determinism — our other pillar — is a purchased feature or a
  nice-to-have.)*

### 4.3 Adoption of third-party soft IP (licensee population, and Tier-1s)

- **Q-C1.** Do you tape out silicon? How many new SoCs or MCU derivatives per year, and when
  does the next architecture window open? *(If the window is 3+ years out, we are not in this
  design cycle regardless of merit — a scheduling "no" that has nothing to do with the product.)*
- **Q-C2.** What third-party IP is in your current parts, and what was the last block you
  licensed rather than built? Why that one?
- **Q-C3.** What has to be true for you to license a compute block from a company with no
  silicon, no reference customer, and one engineer? Is there any configuration of price,
  source-code escrow, indemnity, or acquisition-of-IP terms that makes that survivable — or is
  it categorically out? *(Ask it this bluntly. A categorical no from three licensees is a
  business-model finding, not a rejection.)*
- **Q-C4.** What area and power budget would a block like this have to fit in, in mm² at your
  node, and mW at what utilisation? What would you delete to make room for it?
- **Q-C5.** What does your IP qualification process require before RTL enters a project — CDC
  clean, lint, coverage %, formal on control paths, protocol compliance for AXI/ACE-Lite,
  DFT/MBIST-ready wrappers, silicon-proven status? Which of those are hard gates?
- **Q-C6.** Would "FPGA-proven, not silicon-proven" pass any gate at all, or is it
  automatically disqualifying? *(Our Phase-2 plan rests entirely on the answer.)*
- **Q-C7.** Who else has to say yes: architecture, software, safety, procurement, legal? How
  long from first technical interest to a signed evaluation license, and to a design win?
- **Q-C8.** How is this budgeted — per-design fee plus royalty, and is royalty per-unit
  acceptable in your product line's margin structure? Is there a fee level below which the
  procurement overhead exceeds the benefit? *(Do not ask "what would you pay." Ask about
  process, structure, and thresholds.)*
- **Q-C9.** If we open-sourced a small reference configuration, does that help you (evaluation
  without procurement) or hurt you (license-compliance review, provenance concerns)? *(Directly
  tests the open-core strategy in PLAN.md §7.1, which is currently untested by any customer.)*

### 4.4 Software and toolchain — where IP deals actually die

- **Q-D1.** Who would write the software for this block, us or you? What would you refuse to
  maintain?
- **Q-D2.** Our plan is that you keep authoring in CVXPY/CasADi/acados/Simulink and a code
  generator emits kernels. Does that match how your control engineers work, or would they demand
  their existing hand-tuned C run unchanged? *(A demand for source compatibility with existing C
  is close to fatal and we need to hear it early.)*
- **Q-D3.** What is your build/CI reality — do you need bit-exact reproducibility between the
  simulation model and the target, and between software releases?
- **Q-D4.** How do you validate a control implementation today (MIL/SIL/HIL, back-to-back
  against a reference), and how would a hardware solver fit that? What does bit-exactness or
  its absence do to your validation?

### 4.5 Safety and verification documentation

- **Q-E1.** What ASIL does the function this would run at? *(Do not assume ASIL-D. Motion
  control varies widely; getting the real answer changes our qualification cost by an order of
  magnitude.)*
- **Q-E2.** For third-party IP you treat as a Safety Element out of Context, what work products
  do you require — safety manual, Assumptions of Use, FMEDA with metrics, DFA, dependent-failure
  analysis, a Development Interface Agreement, tool qualification? Which are hard gates versus
  negotiable? *(Background: ISO 26262-11:2018 addresses semiconductors and IP; SEooC analysis
  requires the provider to document Assumptions of Use, and a DIA defines exchanged work
  products and split responsibilities —
  [ISO 26262-11:2018 sample pages](https://cdn.standards.iteh.ai/samples/69604/4f41f01c856048f5b8861cf60daeebad/ISO-26262-11-2018.pdf) (partial/gated),
  [Design&Reuse overview](https://www.design-reuse.com/article/61080-fundamentals-of-semiconductor-iso-26262-certification-people-process-and-product/),
  [Embitel on SEooC](https://www.embitel.com/blog/embedded-blog/what-is-safety-element-out-of-context-seooc-in-automotive-functional-safety) — all secondary.)*
- **Q-E3.** Would you accept "ASIL-ready, with the safety package delivered before tape-out," or
  must the package exist at license signature?
- **Q-E4.** For industrial respondents: IEC 61508 SIL, and what changes versus automotive?
- **Q-E5.** Roughly what did the last third-party safety-qualified IP block cost you in
  *integration and assessment effort*, separate from the license fee? *(This is the real
  adoption cost and it is where a solo founder is most likely to be structurally disqualified.)*

### 4.6 Kill questions — ask every time, near the end

- **Q-F1.** What would make you say no to this outright — not "not now," but never?
- **Q-F2.** If you were me, what would you go and find out that would most likely kill this
  idea?
- **Q-F3.** Who inside your organisation would be most opposed to this, and what would their
  argument be? *(Surfaces the objection they are too polite to make themselves.)*
- **Q-F4.** What would you do instead if this block existed and you decided not to use it? *(The
  real competitor: a bigger CPU, more cores, an FPGA, offline precomputation, a learned policy,
  or a different algorithm entirely.)*
- **Q-F5.** Is there another workload where you'd want this *more* than optimization? *(How the
  beachhead gets corrected by evidence rather than by us guessing again.)*
- **Q-F6.** Who else should I talk to, and may I use your name?

### 4.7 The three riskiest assumptions, and the specific question that can break each

| Assumption (from PLAN.md / PHASE0_FINDINGS.md) | Breaking question | What "broken" looks like in an answer |
|---|---|---|
| **A1. Solver acceleration is a felt pain** | Q-B6 ("have you ever cut a feature because of compute?") + Q-B2 (margin) | "We have plenty of margin"; no war story; the solve is a small fraction of the period; the real pain named in Q-B1 is sensors, software process, calibration, or cost |
| **A2. Energy, not latency, is the purchase driver** | Q-B4 (20× less energy — so what?) + Q-B5 (power ranking) + Q-B7 (past spend) | Nobody owns a control-compute power budget; compute is <1% of system power; they have never spent money on control-compute efficiency; "we'd take the latency, not the watts" |
| **A3. They would integrate a third-party soft IP core at all** | Q-C1, Q-C3, Q-C6 | "We don't tape out"; "we never take compute IP from a company under N employees"; "not silicon-proven is an automatic no"; "our next architecture window is 2030" |

### 4.8 Banned questions (leading — they manufacture false positives)

Do not ask, in any phrasing:

- "Would you be interested in a 20× more energy-efficient solver accelerator?"
- "Is low latency important to you?"
- "Would determinism be valuable for safety certification?"
- "Do you think NPUs are a poor fit for MPC?"
- "Would you pay $X for this?"
- Any question containing "wouldn't it be useful if…"

If a leading question is asked by accident, **flag it in the call record** so the answer can be
discounted. The question log (§8) exists to make this auditable.

---

## 5. Two shorter variants of the guide

**30-minute demand-witness call** (robotics, drones, drives, vehicle builders): §4.1 (A1–A6),
§4.2 (all), §4.6 (F1, F2, F4, F5, F6). Skip §4.3 and §4.5 entirely except Q-E1.

**30-minute licensee call** (semis, RISC-V vendors, captive ASIC groups): §4.1 (A1, A4), §4.2
(B1, B4, B8), §4.3 (all — this is the core), §4.5 (E1, E2, E3), §4.6 (F1, F3, F6).

**Solver-software-vendor call** (Embotech, ODYS, acados group, imperix, MathWorks): §4.1 all,
§4.2 all, §4.4 all, plus: "Across your customer base, what fraction are compute-limited versus
model-limited versus engineering-time-limited?" and "Has any customer ever asked you to target
an accelerator? What happened?"

---

## 6. What we are *not* asking, on purpose

- No pricing willingness questions in call #1 (answers are noise; ask about process instead).
- No NDA'd competitor specs. If a respondent starts disclosing a supplier's confidential
  numbers, stop them — it contaminates everything downstream and cannot be used.
- No feature-preference ranking of Argmind capabilities. We do not have a product to rank, and
  asking invites polite invention.

---

## 7. Falsification criteria — pre-registered, frozen 2026-07-29

**Denominators are explicit and count only respondents who were asked the question and did not
skip it.** "Unprompted" means the respondent raised it before we named the topic. Any criterion
met requires writing the conclusion into PHASE0_FINDINGS.md before further architecture work.

### 7.1 Minimum sample before any criterion may be evaluated

| Population | Minimum substantive 30-min conversations |
|---|---|
| Demand witnesses running production/near-production optimization (segments C-user, D, F) | **8** |
| Licensees (segments A, E; captive-ASIC Tier-1s count here) | **6** |
| Tier-1 / system specifiers (segment B) | **3** |
| **Total** | **17** (plan target: 24, to leave margin for no-shows and unusable calls) |

Below these counts, **no go/no-go conclusion may cite discovery evidence in either direction.**
This clause protects equally against premature kill and premature confidence.

### 7.2 Falsifiers

| ID | Pattern of answers | What it falsifies | Where the evidence then points |
|---|---|---|---|
| **F1 — Pain is not felt** | ≥6 of 8 demand witnesses report the solve consumes <25% of the control period with margin nobody tracks, **and** ≥6 of 8 have no Q-B6 war story of a feature cut for compute | A1: solver acceleration is a felt pain | Beachhead is wrong at the *workload* level. The energy/consolidation story survives only if F2 also fails to trigger. Look instead at where the solve genuinely does not fit: 20 kHz-class power-electronics MPC (segment C) and whole-body/long-horizon nonlinear MPC (ARM-MPC-class), and re-scope to those two, dropping the general "robotics and automotive MPC" claim |
| **F2 — Energy is not a purchase driver** | ≥6 of 8 demand witnesses cannot name a power budget for control compute, **and** rank compute below actuators/radios/thermal by >10× in Q-B5, **and** ≥7 of 8 have never spent money on control-compute efficiency (Q-B7) | A2: energy is the purchase driver | **This is the most likely falsifier and the most damaging**, because PHASE0_FINDINGS.md already reduced our latency advantage to 1.2–2.9×. If energy has no buyer and latency is only ~2×, the value proposition reduces to determinism + core offload. Test those two directly as the primary pitch (Q-B8, Q-B3) before concluding; if determinism also fails to be a purchased feature, **that is a no-go on this beachhead**, and the remaining defensible market is battery-limited mobile robotics/drones (segment D), which is precisely the segment that cannot license IP — i.e. a no-go on the *business model*, not just the segment |
| **F3 — No licensee exists for control IP** | ≥5 of 6 licensees say control/optimization acceleration is not on their roadmap **or** is already covered by an in-house block (PPU/CLA/vector DSP/RVV), **and** ≥4 of 6 rule out non-silicon-proven IP from a sub-5-person vendor categorically (Q-C3, Q-C6) | A3: soft-IP licensing is a viable route | Product form is wrong, not the technology. Ranked alternatives: **(a)** FPGA IP / overlay for industrial drives and power electronics, where imperix/Beckhoff/Danfoss-class buyers already deploy FPGA fabric and procurement is lighter; **(b)** an accelerated backend *sold through* solver vendors (Embotech/ODYS/MathWorks) — they own the customer relationship and would resell compute; **(c)** a chiplet or module rather than soft IP (PLAN.md §4.1 already contemplates UCIe); **(d)** a software product (a better CPU solver using SME/RVV) — lower ceiling, far lower capital need |
| **F4 — Incumbent silicon already covers the socket** | ≥3 of 6 licensees state, unprompted, that an existing vector DSP / control coprocessor in their part runs their customers' MPC adequately, **and** we cannot obtain any measured counter-evidence on that part | The CLAUDE.md claim "incumbents can't retarget in software" — in the *control* segment specifically | The gap is a **feature gap**, not a structural gap, for automotive MCU vector DSPs (it may still be structural for NPU IP). Requires a PPU/CLA/SME/RVV row in the teardown with measured numbers before any further claim is made. If confirmed, differentiation must rest on the sequencer's convergence-loop control and the residency model — both of which are *copyable in a software release plus a modest RTL change*, which materially weakens the moat |
| **F5 — Safety documentation cost is disqualifying** | ≥3 of 4 automotive respondents require a full SEooC package (safety manual + FMEDA with metrics + DFA + DIA) **at license signature** rather than before tape-out (Q-E3), and estimate integration/assessment effort (Q-E5) above what a 1–3 person company can supply | Automotive as a Phase-3 beachhead | Automotive moves to Phase 4+. Beachhead becomes **industrial motor drives / power electronics** (IEC 61508 route, lighter, and per §2 the deadlines are harder there anyway) or **robotics without safety certification** |
| **F6 — Right for the wrong reason** *(not a kill; a mandatory repositioning trigger)* | ≥5 of 17 respondents volunteer determinism / WCET / certification-of-timing as the primary value and ≥4 explicitly deprioritise energy | The current pitch order (energy → determinism → offload) | Reposition to lead with **deterministic, provable WCET for constrained optimization**. Changes the benchmark set's reporting (worst-case, not average) and possibly the architecture priority order. Cheap to act on, expensive to miss |
| **F7 — The workload is being replaced** | ≥5 of 8 demand witnesses say they are actively replacing online optimization with learned policies, explicit/offline MPC, or lookup tables (Q-A5, Q-F4) | The durability of the entire solver thesis | TAM is shrinking under us. Pivot weight toward the ML path — but note that per PHASE0_FINDINGS.md §5 that path is a *worse* competitive position (we concede inference TOPS/W to CIM and NPUs). A GO in this world requires a different differentiator entirely, most plausibly the hybrid: learned policy + constraint-enforcing solver on one engine |

### 7.3 What does NOT count as falsification

Stated in advance so that a bad week is not mistaken for a result:

- Any single "no," however emphatic or senior. n=1 is an anecdote in both directions.
- Non-response or no-show. Log it (§8), but silence from Infineon is evidence about our
  reachability, not about the market.
- "Interesting, but not now" or "come back with silicon." That is a **timing and proof**
  objection, and it is the *expected* answer from a licensee to a pre-silicon vendor. It becomes
  F3 evidence only in the categorical form of Q-C3/Q-C6.
- Enthusiasm from academics or researchers. It is pleasant and it is not demand. Weight
  academic interest at zero in all §7.2 counts.

### 7.4 What confirmation would look like (subordinate, and deliberately harder to hit)

A GO on this beachhead requires **all** of: ≥4 of 8 demand witnesses with a concrete Q-B6 war
story; ≥3 who can name a control-compute power or thermal budget in numbers; ≥2 licensees
willing to describe a real path to a paid evaluation license; and ≥1 Tier-1 or system builder
willing to be named (even anonymously, by segment) as asking their silicon supplier for this
capability. Note the asymmetry: **falsification is easier to trigger than confirmation. That is
intentional.**

---

## 8. Mechanics

### 8.1 Outreach

**Funnel arithmetic, stated honestly.** Cold outreach from an unknown pre-revenue founder gets
roughly 5–15% substantive reply at best, and lower into large semiconductor companies. To hold
24 conversations, plan **~200 approaches** across ~120 organisations, with the target list as the
priority core and a long tail behind it. Budget 8–12 weeks. Track the funnel; if reply rate to
segment A is under 3% after 40 approaches, the problem is our credibility artefact, not our
message — fix that (below) rather than sending more email.

**The credibility artefact is the highest-leverage outreach investment.** Per PLAN.md §7.1, a
`pip install argmind` simulator, a published reproducible benchmark harness, and a whitepaper
with node/precision-labelled numbers convert cold outreach into "the person who published that
comparison." Publishing credible **batch-1 / small-matrix latency data** is an uncontested
position (PHASE0_FINDINGS.md §5, gap 3) and is a legitimate reason for a stranger to reply.
**Do not publish anything containing the ±2× model outputs as if they were measured.**

**Warm routes, in order of realistic yield:**
1. Open-source communities where the target's engineers already participate — acados, CVXPY,
   OSQP, PX4, openpilot issue trackers and forums. Contribute something first.
2. Authors of relevant papers, at their institutional email. Researchers at ABB, Bosch, Infineon
   and Embotech publish under corporate affiliation and answer mail about their own papers.
3. Conferences, chosen for buyer density rather than prestige: **embedded world** (automotive
   MCU product managers), **IEEE ECCE / APEC** (motor drive and power electronics control),
   **RISC-V Summit** (segment E), **ICRA/IROS** (segment D, but researchers — discount
   accordingly), **DVCon** (verification/IP process questions).
4. RISC-V International membership (already in PLAN.md §7) — gives a legitimate reason to
   contact Quintauris, SiFive, Andes, Nuclei.
5. Referrals from every completed call (Q-F6). This is the only mechanism that converts R3 to
   R2.

### 8.2 Cold message template

Short, no pitch, gives before it asks, and makes saying no easy. Adapt the middle sentence per
segment.

> **Subject:** MPC compute question — 25 min? (research, not sales)
>
> Hi <name>,
>
> I'm researching hardware for embedded optimization — MPC and Kalman filtering at kHz control
> rates — and I'm trying to find out where my modelling is wrong before I build anything. No
> product, nothing to sell.
>
> <One sentence, specific to them: "You've published on MP3C in medium-voltage drives, so you
> know better than anyone whether the online part of that is compute-limited in a shipping
> product." / "openpilot runs acados for lateral and longitudinal MPC, which makes you one of
> very few people with production numbers.">
>
> Three things I'd want your view on: what the solve actually costs you in period and in watts,
> whether that has ever forced a design compromise, and what would make an accelerator for it a
> bad idea.
>
> In exchange: my full benchmark set and modelled results (seven workloads, with process node
> and precision attached, and the two where a plain Cortex-A55 already wins), yours to keep
> whether or not we ever talk again.
>
> 25 minutes in the next few weeks? If it's not your area, a pointer to who it is would help
> just as much.
>
> — Yoo Park, Argmind

Notes: name the weak result in the email. It is unusual, it is true, and it signals that the
conversation will be technical rather than a pitch. Never attach a deck.

### 8.3 Running the 30 minutes

| Minutes | Content |
|---|---|
| 0–2 | Framing script (§4.0); ask permission to take notes; confirm 30 min |
| 2–14 | Their world: §4.1 questions. Mostly listening. Chase specifics — part numbers, rates, percentages |
| 14–22 | Constraint diagnosis: §4.2. Q-B4 and Q-B6 must be asked in every single call, no exceptions |
| 22–27 | Population-specific block: §4.3 (licensee) or the rest of §4.2/§4.4 (witness) |
| 27–30 | Kill questions (Q-F1, Q-F2 minimum), referral ask (Q-F6), thank, state what you'll send |

Discipline rules:
- **Talk less than 30% of the time.** If you have explained the architecture, the interview is
  over as evidence.
- **Silence after Q-B1 and Q-F1.** The second thing they say is usually the true one.
- **If they ask what you're building, defer once**, then answer honestly and briefly in the last
  five minutes. Answers given after that point are marked `post-disclosure` in the record and
  weighted lower.
- **Send the promised benchmark data within 24 hours, always**, including after a discouraging
  call. Reputation compounds and this community is small.
- **Two calls per day maximum.** Notes must be written up within two hours or they are
  reconstructions.

### 8.4 Recording — keeping what they said separable from what we think

One file per conversation: `discovery/YYYY-MM-DD_<org>_<role-slug>.md`. The structure is
mandatory and the section boundaries are not to be crossed.

```markdown
# <Org> — <role> — 2026-08-xx
Population: licensee | demand-witness | solver-vendor | specifier
Contact route: cold / referral from <who> / conference
Duration: xx min | Post-disclosure at minute: xx | Consent to notes: yes/no
Recording: no (notes only) | Permission to attribute: no / by-segment / by-name

## 1. VERBATIM  — what they actually said. Quotes only. No paraphrase. No inference.
> Q-B1 (asked verbatim: "...")
> "..."
[Mark: (unclear) (skipped) (declined) (post-disclosure)]

## 2. NUMBERS AS GIVEN — one row per number. Unqualified numbers stay unqualified.
| Number | Units | Workload | Hardware / node / precision | Qualified? | Source |
|---|---|---|---|---|---|

## 3. QUESTIONS I ASKED — full list, verbatim, incl. any leading question (flag it)

## 4. WHAT THEY DID NOT SAY / would not discuss

## 5. MY INTERPRETATION  — clearly ours, not theirs. Speculation allowed and labelled.

## 6. FALSIFIER TALLY  — which §7.2 criteria this call moves, and in which direction
F1: +1 / -1 / neutral   F2: ...   (with the one-line reason)

## 7. FOLLOW-UPS AND REFERRALS
```

Hard rules:
- **Nothing may move from §5 to §1, ever.** If in doubt whether they said it, it belongs in §5.
- A number without hardware, workload, and precision is recorded with `Qualified? = NO` and
  **may not be used in any comparison**, per the project's evidence standards.
- Maintain `discovery/OUTREACH_LOG.md` with every approach, including non-responses, so
  survivorship bias in the sample is visible. Non-response counts in the denominator of
  reachability, never in the denominator of §7.2.
- Maintain `discovery/TALLY.md`: a single table of the seven falsifiers with running counts and
  denominators. Update it after each call, before writing any interpretation. **Do not compute
  the tally for the first time at the end** — that is where rationalisation lives.
- **Do not paste NDA'd or confidential material into these files.** If a respondent shares
  something confidential, record "confidential detail withheld" and its bearing, nothing more.

### 8.5 Signal versus noise

| Conversations in one population | What can honestly be concluded |
|---|---|
| 1–2 | Nothing. Vocabulary calibration only. Useful for fixing our questions |
| 3–4 | Hypotheses worth testing. A pattern here is a *prompt to ask better questions*, not a finding |
| 5–8 | Real signal **if** ≥3 different organisations volunteered the same thing unprompted, and they are not all from the same lineage (e.g. three ETH-descended robotics companies are close to one data point) |
| 9+ | Reliable enough to change a decision. §7.2 thresholds are set here for the demand-witness population |

Correlated-sample traps to guard against explicitly:
- **Lineage clustering.** ETH/Freiburg-descended robotics and MPC companies share tooling,
  assumptions, and vocabulary. Count them as a partially shared observation and note lineage in
  each record.
- **Selection by responsiveness.** People who answer cold email from strangers are
  systematically more curious and less busy than decision makers. Enthusiasm from this
  population is the least reliable signal in the whole program.
- **Researcher bias.** Corporate researchers will love this and do not hold the budget. Weighted
  zero in §7.2 (see §7.3).
- **Our own leading questions.** §8.4 §3 exists so this is auditable after the fact.

### 8.6 Checkpoints

- **After 6 conversations:** review the guide itself. Which questions produced nothing? Which
  answer surprised us? Revise the guide (and log the revision), but **never §7**.
- **After 12:** write the interim read-out with the §7.2 tally as-is, including the criteria
  trending toward trigger. Circulate to nobody; the point is to commit the number in writing
  before it can be shaded.
- **After 17 (minimum) / 24 (target):** evaluate §7.2, write the market memo, and take the
  go/no-go decision with the CPU-baseline measurements and scratchpad-banking cost from
  PLAN.md §10 in hand.

---

## 9. What remains unknown, and how it could be learned

| Unknown | How to learn it | Cost / gate |
|---|---|---|
| Whether an existing automotive vector DSP (AURIX PPU) or control coprocessor (C2000 CLA) already runs our benchmark set adequately — the F4 question, and the biggest unhedged risk to the moat claim | Buy an evaluation board and run the benchmark set. A TC4x or C2000 LaunchPad-class board plus toolchain is the cheapest disconfirming experiment available to us, and it produces **measured** numbers rather than interview opinion | Low cash, real engineering time. **Do this in parallel with discovery, not after** |
| Measured perf/W of the CPU baselines (the largest error source in every ratio we quote) | Already the #1 Phase-0 item: OSQP/acados on A55- and A78-class hardware | Owned, PLAN.md §10 |
| Whether SME2 / RVV closes the gap — our own stated highest-probability competitor | Measure on real SME2 and RVV silicon; and ask Andes (target #22) directly, since they have every incentive to tell us it does | Hardware access; Q-A4/Q-F4 |
| Real IP license and royalty norms for a control accelerator | Q-C8 across ≥4 licensees; plus a fractional semiconductor-IP lawyer or an ex-IP-sales advisor. Do not rely on published anchors — [AnySilicon's IP business-model overview](https://anysilicon.com/ip-business-model/) is generic and not segment-specific | Advisor cost |
| The four "not published" cells in the PHASE0_FINDINGS.md teardown | Evaluation NDAs, or licensee interviews (Q-A4/Q-C2 often reveal what a vendor will not publish) | NDA, and a decision about what NDA'd material may then be used in collateral (answer: none of it) |
| Whether robotics buyers would pay for an Argmind-bearing SoC that does not yet exist — i.e. can demand witnesses create pull on segment A | Q-F5 plus an explicit ask: "would you write two paragraphs to your compute supplier saying you want this?" A yes is the strongest artefact in the program; a no is a quiet F2/F3 signal | Free, and it is the highest-value question we have |
| Whether the analog/in-memory optimization line (PLAN.md §4.1) is nearing deterministic QP/LP solving | Track ISSCC/VLSI/Nature Electronics and the PoliMi/Ielmini line; re-evaluate annually. Not a discovery question | Reading time |

---

## Sources

Production optimization deployment: [ODYS](https://www.odys.it/) ·
[ODYS company](https://www.odys.it/company/) ·
[Bemporad MPC talk, IMT Lucca 2023](http://cse.lab.imtlucca.it/~bemporad/talks/mpc_9_11_2023.pdf) ·
[Embedded MPC with certified real-time optimization for synchronous motors (preprint)](http://cse.lab.imtlucca.it/~bemporad/publications/papers/ieeecst-synchro-motor.pdf) ·
[MPC of turbocharged gasoline engines for mass production (secondary index, paywalled)](https://www.researchgate.net/publication/324235335_Model_Predictive_Control_of_Turbocharged_Gasoline_Engines_for_Mass_Production) ·
[Embotech FORCESPRO](https://www.embotech.com/softwareproducts/forcespro/overview/) ·
[FORCESPRO in MATLAB MPC Toolbox](https://www.mathworks.com/help/mpc/ug/integrate-with-embotech-forces-pro-solvers.html) ·
[acados documentation](https://docs.acados.org/) ·
[acados paper overview (secondary summary — dSPACE HIL figure unverified against primary)](https://www.alphaxiv.org/overview/1910.13753v3) ·
[openpilot 0.8.10 release notes](https://blog.comma.ai/0810release/) ·
[openpilot longitudinal MPC source](https://github.com/commaai/openpilot/blob/master/selfdrive/controls/lib/longitudinal_mpc_lib/long_mpc.py).

Drives / power electronics: [ABB MP3C news](https://new.abb.com/news/detail/74747/new-mp3c-control-method-maximizes-efficiency-and-dynamics) ·
[ABB predictive control feature](https://www.abb.com/global/en/company/innovation/news/predictive-control-revolutionizes-variable-speed-drive-control) ·
[MP3C with very fast transient responses (IEEE, ABB-hosted PDF)](https://library.e.abb.com/public/2a438a08d9a3445b8d2b5c20a7775e4b/IEEE_Model_Predictive_Pulse_Pattern_Control_with_very_fast_transient_responses.pdf) ·
[MP3C on ACS 2000 five-level ANPC](https://www.tobiasgeyer.org/OGKKG13_MP3C_ACS2000.pdf).

Robotics: [Boston Dynamics — Flipping the Script with Atlas (MPC description)](https://bostondynamics.com/blog/flipping-the-script-with-atlas/) ·
[Auterion Skynode, PX4 docs](https://docs.px4.io/main/en/companion_computer/auterion_skynode) ·
[NVIDIA Jetson modules (power range)](https://developer.nvidia.com/embedded/jetson-modules) ·
[Motion planning accelerator microarchitecture, MICRO 2016](https://users.cs.duke.edu/~lkw34/papers/realtime-robotics-micro2016.pdf) ·
[US 12,330,310 — collision detection in motion planning](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/12330310) ·
[Survey of FPGA-based robotic computing](https://arxiv.org/pdf/2009.06034).

Incumbent control accelerators: [Infineon PPU — SIMD vector DSP (vendor)](https://www.infineon.com/product-information/new-ppu-simd-vector-dsp) ·
[AURIX TC4x PPU (emmtrix wiki, secondary)](https://www.emmtrix.com/wiki/Infineon_AURIX_TC4x_Parallel_Processing_Unit_(PPU)) ·
[AURIX TC4x overview (Infineon, non-NDA deck)](https://www.infineon.com/assets/row/public/documents/10/156/infineon-tc4x-overview-productpresentation-en.pdf) ·
[TI C2000 CLA software guide (vendor)](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/intro.html) ·
[PPU support in MathWorks Embedded Coder](https://www.mathworks.com/help/ecoder/infineonaurix/ug/soc-ifx-tc4x-ppu.html).

RISC-V / automotive platform: [Quintauris RT-Europa](https://www.quintauris.com/quintauris-rt-europa-risc-v-platform-automotive/) ·
[Quintauris–Nuclei alignment](https://www.quintauris.com/quintauris-nuclei-automotive-real-time-risc-v/) ·
[Quintauris (Wikipedia — secondary, incl. SiFive partnership Jan 2026)](https://en.wikipedia.org/wiki/Quintauris) ·
[ST joins Quintauris (secondary)](https://www.hackster.io/news/risc-v-joint-venture-quintauris-grows-to-six-as-stmicroelectronics-joins-the-party-b3936a803e0d).

Functional safety: [ISO 26262-11:2018 sample pages (partial/gated)](https://cdn.standards.iteh.ai/samples/69604/4f41f01c856048f5b8861cf60daeebad/ISO-26262-11-2018.pdf) ·
[Fundamentals of semiconductor ISO 26262 certification (Design&Reuse, secondary)](https://www.design-reuse.com/article/61080-fundamentals-of-semiconductor-iso-26262-certification-people-process-and-product/) ·
[Improvements in functional safety of automotive IP through Part 11 (Lorit, secondary)](https://lorit-consultancy.com/wp-content/uploads/2021/02/Improvements-in-Functional-Safety-of-Automotive-IP-Through-ISO-26262-2018-Part-11.pdf) ·
[SEooC explainer (Embitel, secondary)](https://www.embitel.com/blog/embedded-blog/what-is-safety-element-out-of-context-seooc-in-automotive-functional-safety).

IP business model: [AnySilicon — the IP licensing business model (generic, not segment-specific)](https://anysilicon.com/ip-business-model/).
