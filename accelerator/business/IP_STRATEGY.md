# Argmind — IP Strategy: Trademark Research and Pre-Publication Filing Order

**Date:** 2026-07-30 · **Status:** research memo, v0.1 · **Author:** IP-strategy research pass

> ## This is not legal advice and I am not a lawyer.
> Patentability, freedom-to-operate, trademark clearance and licence interpretation are formal
> legal determinations that require a qualified attorney. Nothing in this document is a clearance
> opinion, an FTO opinion, or a patentability opinion. Its only purpose is to make attorney time
> cheap: assemble the prior art, map the collisions, and reduce the open questions to a list that
> can be answered in an hour instead of ten. **No trademark register was successfully searched in
> this pass** (§1.1) — do not treat §1 as clearance under any circumstances.
>
> Every filing decision, every freedom-to-operate question, and every "can we publish this yet"
> question in §2 needs counsel before action.

---

## 0. Lead: four findings, ranked by what they cost if ignored

### F-1 — SEVERITY: CRITICAL, and the failure is unrecoverable. Europe has no grace period.

Publishing before filing forfeits European patentability outright. Art. 54 EPC makes the state of
the art "everything made available to the public … before the date of filing." Art. 55 EPC's
non-prejudicial-disclosure window is **six months and covers only two situations**: "an evident
abuse in relation to the applicant" and display at an "official, or officially recognised,
international exhibition" under the 1928 Paris Convention
([Art. 55 EPC](https://xepc.eu/node/ar55), [Art. 54 EPC](https://xepc.eu/node/ar54)). A GitHub
release is neither. There is no repair after the fact.

The US is different and more forgiving: 35 U.S.C. §102(b)(1)(A) excludes from prior art "a
disclosure made 1 year or less before the effective filing date … made by the inventor or joint
inventor" ([35 U.S.C. §102](https://www.law.cornell.edu/uscode/text/35/102)). So a US filing can
follow a publication by up to a year; an EP filing cannot follow it by a day.

**Why this matters more for us than for a generic startup:** the beachhead is automotive and
industrial control, and the licensees in that socket — Infineon, Bosch, Continental, NXP, ST,
Renesas Europe — are European or heavily European-facing. Losing EP is losing the patent's value
in the exact geography where the design wins are. A US-only patent on a mechanism that a German
Tier-1 implements in a European-fabbed part is close to worthless as leverage.

**Practical rule, and it is the whole point of this document: file before publish, or accept that
you have chosen never to patent that mechanism anywhere but the US.** Both are acceptable
choices. Drifting into the second one by accident is not.

### F-2 — SEVERITY: HIGH. Solderpad-2.1 does not shield `argmind-lite` from granting patent rights. PLAN.md §7.1's framing needs correcting.

I read the licence text directly. Solderpad Hardware License v2.1 defines "Rights" as "copyright
and any similar right including design right … rights in semiconductor topographies (mask works)
and database rights **(but excluding Patents and Trademarks)**" — which on first reading looks like
"no patent grant." It is not, because of how the licence is constructed:

- SHL-2.1 "operates as a wraparound license to the Apache License Version 2.0 … and incorporates
  the terms and conditions of the Apache License."
- Its own text says **"Section 1 below modifies definitions … and Section 2 below replaces
  Section 2 of the Apache License"** — i.e. it replaces Apache's *copyright* grant (§2) and the
  Appendix, and nothing else.
- **Apache §3, the Grant of Patent License, is not replaced or disapplied.**
- SHL-2.1 further says: "You may, at your option, choose to treat any Work released under this
  license as released under the Apache License."

Source: [Solderpad Hardware License v2.1](https://solderpad.org/licenses/SHL-2.1/); Apache §3 text
at [apache.org/licenses/LICENSE-2.0](https://www.apache.org/licenses/LICENSE-2.0).

Apache §3 grants "a perpetual, worldwide, non-exclusive, no-charge, royalty-free, irrevocable …
patent license to make, have made, use, offer to sell, sell, import, and otherwise transfer the
Work." Applied to `argmind-lite`, that reads as: **anyone who takes the reference RTL gets a
royalty-free licence to any of our patent claims necessarily infringed by that RTL, including the
right to make and sell silicon containing it.** The same is true, in software form, of every
Apache-2.0 repo in PLAN.md §7.1.

PLAN.md §7.1 currently frames Apache's patent grant as a *feature* ("explicit patent grant —
required for corporate adoption") and Solderpad as "the hardware-standard Apache variant." Both
statements are true. What is missing is that the grant runs **outward from us**, and it is
therefore a partial forfeiture of the patent's commercial value against anyone who obtains the
code — which is everyone, by design. This is not an argument against open-sourcing. It is an
argument that the claims must be drafted to cover what the open artefacts do *not* teach (see
P-4 in §3).

*Status: this is my reading of the licence text, established by direct quotation of the licence.
Whether Apache §3 in fact survives the SHL-2.1 wraparound is a legal question and question **L-1**
below. Do not act on my reading.*

### F-3 — SEVERITY: HIGH for the patent plan. Candidate claim 4 is dead as stated, and PLAN.md §7.1's patent bullet is stale.

"The accelerator itself evaluates the convergence test and decides to iterate, with no host
round-trip" is anticipated by shipping products roughly fifteen years older than us and by a deep
FPGA-MPC literature. Detail and citations in §2.1-M4. The short version:

- **TI C2000 CLA** — an independent, fully programmable 32-bit floating-point coprocessor with its
  own conditional branch (`MBCNDD`), running a control law to completion without CPU intervention,
  documented since the Piccolo F2803x generation
  ([TMS320x2803x CLA Reference Guide, SPRUGE6](https://www.ti.com/lit/pdf/spruge6),
  [CLA features](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/features.html)).
- **Infineon AURIX TC4x PPU / Synopsys ARC EV7x** — scalar core + SIMD vector unit + closely
  coupled memory executing its own instruction stream (PHASE0_FINDINGS §5.1;
  [EV7x](https://mips.com/processor-solutions/ev-processors/ev7x-vision-processors/)).
- **Every FPGA MPC/QP solver in the literature** — a self-contained iteration controller is the
  default, not an innovation (Jerez/Kerrigan/Constantinides; Hartley et al. TCST 2013; RSQP
  ISCA'23; ButterflyQP MICRO'24; full list in §2.1-M4).
- **US9176928** — "System for convergence evaluation for stationary method iterative linear
  solvers."

PLAN.md §7.1 says "patent 2–3 core claims early (sequencer↔GEMM coupling mechanism, small-matrix
scheduling, deterministic-latency execution model)." On this research, **all three of those are
unpatentable as phrased**, and the third and second are worse than the first: small-matrix
scheduling has an enormous systolic-array-tiling literature, and "deterministic latency by static
allocation with no runtime cache behaviour" is standard practice in every WCET-analysable
safety-critical system and in every FPGA solver cited above. That bullet should be rewritten before
it gets quoted to an investor as an IP position.

Note also that the patent plan has drifted out of sync with our own competitive analysis. CLAUDE.md
and PHASE0_FINDINGS §5.1 already concluded that the moat against the real competitors is **energy
per solve, deterministic worst-case latency, and the codegen ecosystem — not control flow.** The
filing plan should follow that conclusion: file on the measured microarchitectural findings that
produce the energy and determinism results, not on the architectural framing that the competitive
teardown already retired.

### F-4 — SEVERITY: MEDIUM, but time-sensitive. No blocker found for "Argmind"; also, no register was searched, so this is not clearance.

Best available read: **no exact-match company, product, GitHub org, PyPI package or operating
domain named "Argmind" was found** anywhere I could reach. `github.com/argmind` returns 404, so the
org name is free today. That is genuinely encouraging.

But (a) I could not reach the USPTO or EUIPO registers by any route (§1.1), so unregistered-search
absence of evidence is all this is; and (b) there is one real common-law neighbour worth an
attorney's eye: **Argmin AI Inc.** at [argminai.com](https://argminai.com/) — an AI-software
company (agent cost optimisation: prompt compression, model routing, speculative decoding),
copyright notice "© 2026 Argmin AI Inc.", no ™/® symbols on the page. Same broad field, likely the
same Nice classes 9/42, and "ARGMIND" versus "ARGMIN AI" is a real sight/sound/meaning confusion
analysis rather than a trivially clean one. Whether they have filed anything is exactly the thing I
could not check.

---

# Part 1 — Trademark research: "Argmind" and the "Computational Module / CM" category label

## 1.1 Access log: what I could not reach. Read this before reading anything else in Part 1.

I attempted the primary registers by every route available to a GET-only fetch tool. **All failed.**

| Target | Route attempted | Result |
|---|---|---|
| USPTO trademark search | `https://tmsearch.uspto.gov/search/search-results?q=argmind` | Loaded, but renders only the heading "Trademark search" — JavaScript SPA, no results in fetched HTML |
| USPTO trademark search API | `https://tmsearch.uspto.gov/api-v1-0-0/tmsearch?q=argmind` | HTTP 404 on GET (endpoint is POST-only) |
| EUIPO eSearch plus | `https://euipo.europa.eu/eSearch/#basic/argmind` | JavaScript SPA — returned site navigation and footer only |
| TMview (EUIPO/TMDN) | `https://www.tmview.europa.eu/...`, `https://tmview.europa.eu/...` | TLS failure: certificate altnames are `tmdn.org`/`*.tmdn.org`, host mismatch |
| TMDN TMview API | `https://www.tmdn.org/tmview/api/search/results?basicSearch=argmind` | `ECONNRESET` |
| Justia Trademarks (USPTO mirror) | `trademarks.justia.com`, `trademark.justia.com` | HTTP 403 |
| uspto.report (USPTO mirror) | `https://uspto.report/Search/?q=argmind` | HTTP 403 |
| Trademarkia (USPTO mirror) | `https://www.trademarkia.com/search/trademarks?q=argmind` | HTTP 403 |
| TrademarkElite | `https://www.trademarkelite.com/search/trademark?keyword=argmind` | HTTP 404 |

**Consequence, stated plainly: §1.2–§1.6 contain no register data. They are web and common-law
evidence only.** A "no results found" in a web search is not a "no results found" in a register.
Everything below is a research input for a clearance search, not a substitute for one.

## 1.2 What a human must check manually — the 30-minute version

Do these in this order. This is the entire manual task, and it is short enough to do in one sitting.

> **The registers cannot be scripted — established 2026-08-07, so nobody wastes time retrying.**
> A second attempt with a shell and POST requests failed on every route: `tmsearch.uspto.gov`
> serves only static objects (`MethodNotAllowed` on POST, `NoSuchKey` on a Solr-style GET), the
> TMview results endpoint returns an empty body, EUIPO's endpoint does not connect, and WIPO's
> Global Brand Database answers with an **altcha proof-of-work challenge page** rather than data.
> This is not a missing API key or a wrong URL — the registers are deliberately bot-protected.
> **The check below genuinely requires a human in a browser.** Budget 30 minutes and do it once.

1. **USPTO** — [tmsearch.uspto.gov](https://tmsearch.uspto.gov/). Run each of these as a *basic*
   search (the register's default is exact-ish; you want the fuzzy neighbourhood too):
   - `argmind` · `arg mind` · `argmin` · `argmin ai` · `argmine` · `argmynd` · `argmind*`
   - Then a phonetic/design search over `arg*min*` and `*mind` limited to classes 9 and 42.
   - For each hit: record serial number, owner, filing basis (§1(a) use vs §1(b) intent-to-use),
     status (live/dead), and the goods/services wording — the wording matters more than the class
     number.
2. **EUIPO** — [eSearch plus](https://euipo.europa.eu/eSearch/) and
   [TMview](https://www.tmdn.org/tmview/) for the same strings. TMview also covers national
   registers (DE, FR, IT, ES) and WIPO, which matters because a German national mark blocks an EUTM
   just as effectively as an EUTM does.
3. **KIPRIS** — [engdb.kipris.or.kr](http://engdb.kipris.or.kr). Omitted from the original list.
   Worth checking KR classes 9 and 42 given where the author is based.
4. **UKIPO** separately (post-Brexit, an EUTM does not cover the UK, and Arm/Imagination-adjacent
   naming collisions are likeliest to be UK-registered).
5. **Whether Argmin AI Inc. has filed.** Search the owner name in USPTO and EUIPO, and check state
   corporate registries (likely Delaware). This is the single most valuable manual lookup on the
   list — it converts F-4 from "unknown" to "known."
6. **WHOIS**, for `argmind.com` / `.io` / `.ai` / `.dev` and `argmind-semi.com` — I had no shell and
   no reachable WHOIS API in this pass, so registrant status is genuinely unknown (§1.4).
7. **A dated screenshot record** of the GitHub org, domains and search results on the day you check
   them, filed with the eventual application. Cheap; occasionally decisive later.

## 1.3 Common-law and unregistered use — "Argmind"

**Established (searched and found nothing):**

| Check | Result | Date |
|---|---|---|
| "Argmind" as company / startup / software | No exact match found in web search | 2026-07-30 |
| `github.com/argmind` (org or user) | **HTTP 404 — name is unclaimed** | 2026-07-30 |
| `pypi.org/pypi/argmind/json` | HTTP 404 — package name unclaimed | 2026-07-30 |
| "ArgMind" on GitHub / GitLab / PyPI via search | No project found | 2026-07-30 |

**Established (found, and ranked by how much they should worry you):**

| Neighbour | What it is | Risk read |
|---|---|---|
| **[Argmin AI Inc.](https://argminai.com/)** | AI-agent cost-optimisation software platform. "© 2026 Argmin AI Inc." No ™/® on site. | **Highest.** Adjacent-to-overlapping field, plausibly classes 9 and 42, and a name that differs from ours by one consonant with near-identical stress pattern. If they have a live §1(a) or §1(b) filing in 9 or 42, this is the thing that blocks us. **Must be checked manually (§1.2 item 4).** |
| [argmin.net](https://www.argmin.net/) | Ben Recht's personal ML/decision-theory blog on Substack, "a blog of minimum value," running since July 2023 | **Low, and I disagree with treating it as a risk at all.** It is editorial content, not goods in class 9 or services in 42; the use is descriptive-of-subject-matter (a blog about optimisation using the optimisation operator as a title); there is no product, no licensing, no semiconductor or software offering. PLAN.md §0's read ("not a trademark conflict in IC class") is correct. The only residual is reputational/SEO overlap in the ML audience, which is a marketing consideration, not a legal one. |
| `argmin` Rust crate ([argmin-rs](https://github.com/argmin-rs/argmin)) | Open-source numerical-optimisation library, well known in the Rust ecosystem | **Low legal, moderate marketing.** No trademark claim evident. But note the practical hazard: our SDK is a developer tool for optimisation, and `pip install argmind` sits one letter from a known optimisation library. Expect mistyped installs and support confusion. Not a reason to rename. |
| [Augmind](https://www.startuphub.ai/startups/augmind) | Mixed-reality content company, founded 2018 | Low. Different field, and the visual/phonetic distance is larger than it looks in text. |
| [argmax.ai](https://builtwith.com/redirects/argmax.ai) | Domain in the ML space | Low. Different suffix, different meaning. |

**Suspected, not established:** that `argmin`-rooted names are broadly unclaimed in classes 9/42
because the term is generic mathematical notation with no owner. That reasoning is sound and PLAN.md
§0 relies on it, and it is probably right — genericness of the *root* makes the space less crowded
and makes a coined derivative (`Argmind`) more registrable than the root itself. But it also cuts
the other way: **a mark built on generic notation gets narrower protection**, because you cannot
stop others using "argmin" and the distinctiveness lives entirely in the coined `-d`. That is a
weaker moat than "Argmind is unusual" suggests, and it matters because §2 concludes the brand *is*
the moat once the source is public.

## 1.4 Domains — RESOLVED 2026-08-07 (was "could not establish")

Checked with a shell, which the earlier pass did not have.

| Domain | Result |
|---|---|
| argmind.com / .io / .ai / .dev / .org | **NXDOMAIN — no DNS record**, so most likely unregistered |
| argminai.com | 89.167.6.32 — live (Argmin AI Inc.) |

Caveat: no DNS record is not identical to "unregistered" — a domain can be registered with no
nameservers delegated. Confirming needs a registrar or RDAP lookup, which is 30 seconds in a
browser.

**The first attempt at this was WRONG and the failure mode is worth recording**, because it is the
same shape as several modelling errors in this project. Queried through the local (ISP) resolver,
*all six* domains resolved to one identical address, 210.94.0.73 — which reads as "all registered."
A control query for a domain that certainly does not exist returned the **same address**, exposing
it as ISP NXDOMAIN hijacking: the resolver fabricates an answer instead of admitting the name is
absent. Re-querying through a public resolver (1.1.1.1) gave the real result above.

Lesson, restated because it keeps recurring here: **a uniformly tidy result is evidence the
measurement is broken, not that the world is simple.**

## 1.5 Fallback names — one-line reads, so a rename is not a research project

Same caveat: no register searched.

| Candidate | One-line read |
|---|---|
| **Minarg** | **Strongest fallback.** Zero hits of any kind in searches — no company, no product, no domain surfaced. Maximally distinctive and therefore maximally registrable. Cost is what PLAN.md §0 already noted: it loses the instant math recognition that makes the name work on the optimisation audience. |
| **Argminium** | Also clean in searches; very ownable; the `-ium` element suffix reads as materials-science and lands well in semiconductors. Long, and awkward inside a SKU (`Argminium CM-100` is a mouthful). Second choice. |
| **Krylov** | No tech-brand collision found, but three separate cautions: it is a common surname (Igor Krylov, Vlad Krylov both surface immediately), which raises the odds of scattered national registrations and of consent/surname objections in some jurisdictions; PLAN.md §0's geopolitical-footprint note stands; and a scientist-name brand carries no product meaning, so it needs more marketing spend than a name that explains itself. Viable but the weakest of the three. |
| *Argmini* (from PLAN.md §0) | **Recommend against, on collision grounds not taste.** "MINI" is a heavily enforced BMW mark, "Mac mini" is Apple's, and "mini" is ubiquitous in device naming. A name whose distinctive element is "mini" invites exactly the opposition you are trying to avoid. |

## 1.6 The category label: "Computational Module" / "CM"

Two questions that get conflated and should not be. Answering them separately is most of the value
here.

**(a) Does using it walk into someone else's mark?** No evidence that it does, and the strategy is
structurally protective. We use "Computational Module" as a **generic category label**, always
subordinate to the house mark ("Argmind CM-100", "the industry's first Computational Module"). Use
of a descriptive/generic term to describe what a product *is* is the classic descriptive-use
posture. I found no evidence of "COMPUTATIONAL MODULE" as a registered mark — but see §1.1, I could
not search the register, so that is an absence of web evidence only. **Manual check to add to §1.2:
`computational module` and `compute module` in classes 9 and 42, USPTO and EUIPO.**

**(b) Can we own it?** No, and we should not try. A term that plainly describes the goods is
likely refused as descriptive, and — more importantly — **owning the category label would defeat
the strategy.** PLAN.md §4's positioning is that "the CM category label is ours to define, the way
NPU was coined and evangelised." Category evangelism requires that *other people use the term*,
including competitors and analysts. Registering it would give us standing to stop exactly the usage
we need. Do not file on it. (Recognised trade-off: we also cannot stop a competitor calling their
part a Computational Module. That is the price of coining a category, and Arm paid it with "NPU.")

**The collision that is real, and it is a marketing collision, not a legal one.** Raspberry Pi's
**Compute Module** line is the dominant meaning of "CM" plus a number in the embedded-systems
vocabulary: CM1, CM3, CM4, CM5, sold explicitly for "embedded and industrial applications"
([Compute Module 5](https://www.raspberrypi.com/products/compute-module-5/),
[CM design guide](https://www.raspberrypi.org/documentation/hardware/computemodule/cm-designguide.md)).
Raspberry Pi Ltd. owns RASPBERRY PI (US reg. **4827364**, filed 2012-07-25, registered 2015-10-06,
live) and publishes enforcement-minded brand rules stating you "may not apply to register any
Raspberry Pi IPR as trademarks, domain names, designs or copyright works anywhere in the world"
([trademark rules](https://www.raspberrypi.com/trademark-rules/)). NVIDIA likewise sells Jetson
"compute modules," using the phrase descriptively.

Assessment:

- **Legal risk: low.** Different goods (a licensable soft-IP core versus an SBC-format SoM), no use
  of Raspberry Pi's actual marks, our term is "Computational" not "Compute," and every use is
  paired with the house mark. Raspberry Pi's rules concern *their* marks, and "Compute Module" is
  arguably descriptive in their hands too.
- **Marketing risk: moderate and unavoidable.** Our buyer — an embedded systems engineer in
  robotics or industrial control — hears "CM4" and thinks Raspberry Pi, not Argmind. "CM-100" reads
  as a slightly odd Raspberry Pi SKU to that audience. SEO on "CM" or "compute module" is
  unwinnable, and organic search for "CM-100" will be noisy. That is a real cost on a
  developer-adoption funnel whose whole design (PLAN.md §7.1) depends on discoverability.
- **Recommendation:** keep CM — it is the right category word and the SKU embedding is good
  strategy — with three disciplines: **(i)** never use "CM-100" without "Argmind" on first mention
  in any document, page title or repo README; **(ii)** carry a one-sentence category definition
  ("a Computational Module is a licensable core that runs the optimisation loop and the neural
  network on one engine") everywhere the term first appears, so we are teaching the term rather than
  borrowing an existing one; **(iii)** do all SEO and paid discovery on "Argmind", never on "CM".

**One connection back to Part 2 that is easy to miss.** Because the category label is deliberately
generic and the source is deliberately public, **the trademark is the only retained moat** — this is
exactly what PLAN.md §7.1 says ("Trademark stays owned: anyone can use the code, only we ship
Argmind®"). That makes Argmind clearance a **gating item for the open-source release**, not merely
a branding chore: publishing under a name you later have to abandon means the community's forks,
blog posts, package names and citations all carry a dead brand, and the funnel that the open stack
exists to build (PLAN.md §7.1 "funnel math") gets rebuilt from zero. Clear the name before the first
repo goes public, for the same reason you file before you publish.

---

# Part 2 — What must be filed before any open-source release

## 2.0 The mechanics that set the ordering

Four legal facts, each verified by direct quotation, that together determine the sequence:

1. **US: 12-month grace for our own disclosure.** §102(b)(1)(A) excludes "a disclosure made 1 year
   or less before the effective filing date … by the inventor or joint inventor"
   ([35 U.S.C. §102](https://www.law.cornell.edu/uscode/text/35/102)).
2. **EP/CN/JP/KR: effectively no grace.** Art. 54 EPC absolute novelty; Art. 55's six-month window
   covers only evident abuse and recognised international exhibitions
   ([Art. 55 EPC](https://xepc.eu/node/ar55)). Most major non-US jurisdictions follow the
   absolute-novelty model.
3. **Apache-2.0 grants patent rights outward on our own code.** §3 grants a "perpetual, worldwide,
   non-exclusive, no-charge, royalty-free, irrevocable … patent license to make, have made, use,
   offer to sell, sell, import, and otherwise transfer the Work"
   ([Apache-2.0](https://www.apache.org/licenses/LICENSE-2.0)).
4. **Solderpad-2.1 does not remove that grant** — it replaces Apache §2 and the Appendix only, and
   lets licensees elect plain Apache-2.0 treatment (F-2 above,
   [SHL-2.1](https://solderpad.org/licenses/SHL-2.1/)).

**Which brings up the disclosure inventory nobody has written down.** PLAN.md's publication plan
contains more disclosures than §7.1's repo table, and they are not currently ordered against any
filing:

| Planned public artefact | Where committed | Mechanisms it discloses | Currently scheduled |
|---|---|---|---|
| `argmind-sim` (`pip install argmind`) | PLAN §7.1, on-ramp #1 | **M1, M2, M3** — it is the fullest single disclosure: residency protocol, descriptor semantics, precision contract | **Earliest planned publication.** This is the critical-path artefact |
| `argmind-sdk` | PLAN §7.1 | M1 (the `am_alloc(AM_PERSISTENT)` / generation-tag / dirty-mask API), M3 (capability query) | Day one — "the API is the constitution" |
| `argmind-codegen` | PLAN §7.1 | M2 (descriptor templates), M1 (residency planner) | With sim |
| `argmind-lite` (Solderpad) | PLAN §7.1 | **M4 / M4′ — the only artefact that discloses the sequencer↔datapath coupling and chain blocking in implementable form** | Phase 2 |
| **Phase-1 technical whitepaper** | PLAN §6, Phase 1 (months 3–6) | Potentially **all four**, at architecture level | **Months 3–6, unsequenced against any filing** |
| Versioned ISA/descriptor spec as a public repo | PLAN §7.1, "spec-first culture" | M2 in full detail | With the community mechanics |
| Benchmark harness + published results | PLAN §7.1 | Indirect, but PPA/energy claims can be enabling as to microarchitecture | Ongoing |

**The whitepaper is the nearest-term hazard and nobody has flagged it.** It is scheduled for months
3–6, ahead of every repo, and a good architecture whitepaper is precisely an enabling disclosure. It
costs nothing to reorder a document; it costs European patentability not to.

## 2.1 Mechanism-by-mechanism prior art and novelty read

> **Search-coverage caveat, stated up front so the reads below can be weighted correctly.** Google
> Patents' individual patent pages returned HTTP 503 throughout this pass, and its XHR search
> endpoint would not accept multi-term AND queries; Justia, Trademarkia and uspto.report returned
> 403; the PatentsView API has been retired and redirects to `data.uspto.gov`. So patent discovery
> here was done via web search over patent databases, which finds *representative* prior art
> reliably and *exhaustive* prior art not at all. **Assignees, priority dates and claim text for the
> patents cited below were not verified from the patent documents.** A professional prior-art search
> (typically $2–5k) is the correct instrument before spending money on a non-provisional, and is
> question **P-6**. What follows is enough to make three of four filing decisions confidently,
> because in three of four cases the prior art found is already dispositive.

---

### M1 — Generation-tag call-to-call residency

**What we would claim.** Persistent scratchpad buffers each carrying a monotonic generation counter
maintained by a user-space runtime; each call descriptor carries a per-buffer dirty mask; hardware
re-fetches only buffers whose generation has changed, over an IO-coherent host port; the scratchpad
is deliberately *not* hardware-coherent, for determinism. (PLAN.md §3.2.2.)

**Prior art found:**

| Reference | Relevance |
|---|---|
| [US8868800B2](https://patents.google.com/patent/US8868800B2/en) — "Accelerator buffer access" | Accelerator-local buffers used as scratchpad with flexible host access |
| [US11669736](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/11669736) — "Executing neural networks on electronic devices" | Preserving weight data in accelerator memory across inference operations to cut memory traffic — i.e. call-to-call residency, in the NPU idiom |
| [US20200104691A1](https://patents.google.com/patent/US20200104691A1/en) — NPU DMA bandwidth optimisation | Selective/optimised DMA into NPU local memory |
| [US11500802B1](https://patents.google.com/patent/US11500802B1) — "Data replication for accelerator" | Host↔accelerator data replication management |
| ["Sidebar: Scratchpad Based Communication Between CPUs and Accelerators"](https://ar5iv.labs.arxiv.org/html/1910.10794) (arXiv 1910.10794) | Scratchpad as the CPU↔accelerator communication substrate, explicitly in place of coherence |
| ["PERKS: a Locality-Optimized Execution Model for Iterative Memory-bound GPU Applications"](https://arxiv.org/pdf/2204.02064) | Persistent state held on-chip across iterations of an iterative workload |
| ["Scratchpad Memory Management for Deep Learning Accelerators"](https://dl.acm.org/doi/10.1145/3673038.3673115) (ICPP 2024) | Software-managed residency planning for accelerator scratchpads |
| ["A Persistent-State Dataflow Accelerator for Memory-Bound Linear Attention Decode on FPGA"](https://arxiv.org/html/2603.05931) | State persisting across invocations to convert memory-bound to compute-bound — the same economic argument we make |
| Generic background | Version/epoch counters are canonical in software coherence (RCU, epoch reclamation, VIPS-M/DeNovo-line software-coherence research); dirty bits and dirty masks for selective writeback/fetch are canonical in caches and DMA engines |

**Honest read: plausibly novel as a combination, and the narrowest of the four claims — which is a
compliment, not a criticism.** Every *element* is old, and an examiner will reach for §103 over
"cache dirty bits + software version counters, applied to a scratchpad." What I did not find,
anywhere, is the specific assembly: a runtime-maintained per-buffer generation counter, transported
in the per-invocation descriptor as a dirty mask, driving hardware selective re-fetch across an
IO-coherent port, in a scratchpad that is *deliberately non-coherent so that worst-case latency is
statically analysable*. The determinism motivation is what makes the combination non-obvious rather
than a mere aggregation: the closest prior art (hardware coherence, or a read cache) is what a
skilled person would reach for, and we are affirmatively rejecting it for a stated real-time reason.
The LLM mapping in PLAN.md §3.2.2 (weights generation-stable, KV-cache append-only, per-token delta
one embedding vector) also broadens the utility story usefully.

**Worth a provisional. It is the best of the four.**

*Established:* the references above exist and are relevant. *Suspected:* that no closer art exists
— unverified, and this is exactly what a professional search is for.

---

### M2 — Descriptor format + hardware descriptor generator with multi-template arming

**What we would claim.** 8-word/32-byte descriptors (`perf_model/seq_emit_cost.py`); three AGUs
walking a descriptor template under a 3-level hardware loop; ≥6 armed templates retained so an
iterative solver arms once per *solve* rather than per *iteration*.

**Prior art found. This is the crowded one, and it is crowded from several directions at once.**

| Reference | What it already teaches |
|---|---|
| [US8713285B2](https://patents.google.com/patent/US8713285B2/en) — "Address generation unit for accessing a multi-dimensional data structure in a desired pattern" | **Closest hit, and it is very close.** Nested loops with jumps (step size) and iteration counts "pre-programmed into the address generation unit as part of a **port descriptor** prior to accessing the … data structure"; addresses computed by executing the pre-programmed nested loops. That is our 3-level-loop-over-a-template, described in a patent |
| [US7926046B2](https://patents.google.com/patent/US7926046B2/en) — "Compiler method for extracting and accelerator template program" | An accelerator comprising "a DMA, a compute core, a local memory and a programmable accelerator control unit," programmed from compiler-extracted loop-nest **templates** |
| [GB2505446A](https://patents.google.com/patent/GB2505446A/en), [WO2003100600A2](https://patents.google.com/patent/WO2003100600A2), [US9529747B2](https://patents.google.com/patent/US9529747B2/en), [US20240020239A1](https://patents.google.com/patent/US20240020239A1/en) | DMA address generators, processor AGUs, DSP address generation, AI/ML tensor processor |
| [NVDLA hardware architecture](https://nvdla.org/hw/v1/hwarch.html) | **Ping-pong register groups**: two register sets per subunit; the CPU programs group B while group A executes and sets B's enable bit; hardware switches on completion — explicitly "to hide the CPU's reprogramming latency." This is *multi-template arming at depth 2*, published and open |
| [Gemmini](https://github.com/ucb-bar/gemmini) (UC Berkeley; [architecture](https://deepwiki.com/ucb-bar/gemmini/2-architecture), [Chipyard docs](https://chipyard.readthedocs.io/en/1.1.0/Generators/Gemmini.html)) | `LoopMatmul` / `LoopConv`: "CISC-type loop instructions … generating [intrinsic instructions] directly on the hardware using FSMs," unrolled into tasks and buffered to a reservation station. A hardware loop expanding a compact command into a stream of descriptors — open source since 2018 |
| [Plasticine (ISCA 2017)](https://ppl.stanford.edu/papers/isca2017-prabhakar.pdf) | Pattern Memory Units with dedicated address generators, and configuration stores holding "the number of nested loops, the limits of each loop iterator, and the instructions to be executed for each stage" |
| NVIDIA Hopper **TMA** ([Colfax tutorial](https://research.colfax-intl.com/tutorial-hopper-tma/), [CUTLASS](https://docs.nvidia.com/cutlass/latest/media/docs/cpp/cute/0z_tma_tensors.html), [`make_tma_descriptor`](https://nvidia.github.io/cccl/unstable/libcudacxx/extended_api/tma/make_tma_descriptor.html)) | Tensor-map descriptors describing base address, per-dimension extents and strides, dtype; one thread hands a descriptor to a hardware copy engine; descriptors **reused across transfers**, with `replace_address()` "to avoid rebuilding it" |

**Honest read: not worth a filing, and I would say that even if the budget were larger.**

- The **8-word layout** is a data format, and it is the format that *falls out* of the static-codegen
  constraint (every address known at codegen time, only base addresses vary, by a fixed stride —
  `seq_emit_cost.py`'s own "CRITICAL STRUCTURAL FACT"). A layout that follows necessarily from a
  stated design constraint is the textbook shape of an obvious limitation.
- The **3 AGUs under a 3-level hardware loop walking a template** is squarely in US8713285B2 and
  Plasticine.
- The **only** arguably-new element is **template-arming depth ≥6, so arming happens once per solve
  rather than once per iteration.** NVDLA has depth 2 and arms per *layer*; TMA reuses a descriptor
  but the reuse is host-driven. "Increase the buffer count from 2 to N" is one of the most reliably
  obviousness-rejected forms of claim there is, and the supporting argument ("iterative solvers
  re-issue the same template shape every iteration") is the kind of application-specific
  motivation that reads as design, not invention.

**Recommendation: do not file M2 standalone. Do two other things instead.**

1. **Fold the ≥6-template-arming limitation in as a dependent claim** in whichever provisional does
   get filed (M1 or M4′). It costs nothing there and preserves the option.
2. **Publish the descriptor format deliberately, as a defensive publication.** It is going to be
   public regardless — `argmind-sim` and the versioned ISA/descriptor spec repo both disclose it,
   and PLAN.md's API-as-constitution principle *requires* it be public. Publishing it with a date
   (a repo tag, an arXiv note, or the whitepaper) means nobody else can patent it against us later.
   That is real value obtained from a mechanism that has no patent value.

---

### M3 — MACIF precision-declaration contract

**What we would claim.** A swappable compute-macro boundary where precision is a declared, queryable
capability (`supported`, `macs_per_cycle[format]`, `accumulator_width[format]`, rounding mode,
subnormal behaviour), and the runtime **refuses** a kernel at codegen time if the macro does not
declare the required format — a hard error, never a silently degraded result; with the fp32/fp64 FMA
cluster deliberately outside the swappable region. (PLAN.md §4.1.)

**Prior art found. This is the weakest of the four by a wide margin.**

| Reference | What it already teaches |
|---|---|
| [US7530061](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/7530061) — "Programmatic access to the widest mode floating-point arithmetic supported by a processor" | Queryable declaration of the floating-point capability a processor actually supports |
| [OpenCL `cl_khr_fp64`](https://registry.khronos.org/OpenCL/sdk/2.2/docs/man/html/cl_khr_fp64.html) | Double precision as an **optional, queryable** capability; using `double` without the extension is a **compile-time error** ("use of type 'double' requires cl_khr_fp64 extension to be enabled"). This is, feature for feature, our "hard codegen-time refusal" — and it shipped in 2008 |
| Vulkan `VkPhysicalDeviceFeatures.shaderFloat64`; CUDA compute capability; RISC-V `misa`; Arm `ID_AA64ISAR*`/`ID_AA64PFR*` ID registers; SYCL device aspects | Declare-and-query capability discovery, including numeric formats, as universal practice across every heterogeneous compute stack for 15–20 years |
| ONNX Runtime execution-provider `GetCapability` | Runtime asks the device which subgraphs it can execute and refuses/falls back — the same control flow, in the ML-compiler idiom |
| [All-rounder (arXiv 2310.16757)](https://arxiv.org/pdf/2310.16757) | Flexible AI accelerator with diverse data-format support and morphable structure |
| [US20180189231A1](https://patents.google.com/patent/US20180189231A1/en) — configurable spatial accelerator | Heterogeneous coarse-grained operations including double-precision FP, with compiler awareness |

**Honest read: not novel. Assess as unpatentable and stop spending thought on it as IP.**
"Declare capabilities, query them, refuse at compile time" is the standard architecture of every
heterogeneous compute API in existence. The genuinely good engineering here — putting the fp64 FMA
cluster *outside* the swappable region so that the interface contract stays weak enough for an
analog CIM macro to honour (PLAN.md §4.1) — is a **design choice justified by a costing**, and
design choices are hard to claim and easy to design around. Note it is also correct and important;
it just is not a patent.

**Recommendation: no filing, and publish MACIF openly and early — this is the one place where
publish-first is affirmatively the right move, not a concession.** Three reasons that compound:
(i) an interface's value scales with adoption, and a patented interface does not get adopted; (ii)
publishing it first blocks a CIM-macro vendor from patenting the boundary and then charging us to
cross it — which, given PLAN.md §4.1's whole thesis that MACIF converts the IMC threat into a
product line, is a real exposure; (iii) it is the cheapest possible credibility artefact for the
Phase-1 whitepaper. Publishing MACIF does *not* require publishing the descriptor internals or the
sequencer microarchitecture, so it can go out ahead of any filing without prejudicing M1 or M4′ —
**provided the spec is written to the interface and not to the implementation behind it.** That
discipline is a real drafting constraint and someone has to enforce it.

---

### M4 — Sequencer↔datapath coupling for in-fabric convergence loops (adversarial, as requested)

**What we would claim.** The accelerator itself evaluates the convergence test and decides to
iterate, with no host round-trip.

**Prior art found. I am going to be blunt, because this is the claim where being wrong is
expensive.**

*Shipping products, public and 15+ years old:*

| Product | Why it reads on the claim |
|---|---|
| **TI C2000 CLA** ([CLA overview](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/intro.html), [features incl. `MBCNDD`](https://software-dl.ti.com/C2000/docs/cla_software_dev_guide/features.html), [TMS320x2803x CLA Reference Guide SPRUGE6](https://www.ti.com/lit/pdf/spruge6), [TMS320F28379D](https://www.ti.com/product/TMS320F28379D)) | TI's own words: a "fully-programmable independent 32-bit floating-point hardware accelerator … designed to execute real-time control algorithms in parallel with the C28x CPU." It has its own conditional delayed branch (`MBCNDD`), its own task triggering, and its own local memory. It evaluates a computed quantity and branches on it, in fabric, with no CPU involvement, and it has done so in shipping silicon since the Piccolo F2803x era |
| **Infineon AURIX TC4x PPU = licensed Synopsys ARC EV7x** (PHASE0_FINDINGS §5.1; [emmtrix wiki](https://www.emmtrix.com/wiki/Infineon_AURIX_TC4x_Parallel_Processing_Unit_(PPU)), [EV7x](https://mips.com/processor-solutions/ev-processors/ev7x-vision-processors/), [EV7xFS](https://www.synopsys.com/designware-ip/processor-solutions/arc-evfs.html), [MetaWare for AURIX](https://mips.com/processor-solutions/sw-metaware/)) | 32-bit RISC scalar core + SIMD vector unit + closely coupled memory, executing its own instruction stream in its own ISA — "does not execute TriCore instructions and vice versa." Structurally, a sequencer coupled to a vector datapath deciding its own control flow, automotive-qualified |

*Academic literature — deep, sustained, and largely never patented, which does not matter:*

| Reference | Relevance |
|---|---|
| Jerez, Constantinides, Kerrigan — [FPGA implementation of an interior point solver for linear MPC](https://www.researchgate.net/profile/Juan-Jerez-7/publication/224210007_FPGA_implementation_of_an_interior_point_solver_for_linear_model_predictive_control/links/54b13bba0cf2318f0f9253ce/FPGA-implementation-of-an-interior-point-solver-for-linear-model-predictive-control.pdf) | Self-contained iterative solver in fabric |
| Jerez, Goulart, Richter, Constantinides, Kerrigan, Morari — [Embedded Predictive Control on an FPGA using the Fast Gradient Method](https://www.semanticscholar.org/paper/Embedded-Predictive-Control-on-an-FPGA-using-the-Jerez-Goulart/f456c84a24e3313aacb35bbc86729e1275ba591d) (ECC 2013) | Same |
| Hartley et al. — [Predictive Control Using an FPGA With Application to Aircraft Control](https://cas.ee.ic.ac.uk/people/gac1/pubs/HartleyTCST13.pdf) (IEEE TCST 2013) | Full MPC loop in fabric, safety-critical framing |
| Zia & Zammit — [An Embedded Scalable Linear Model Predictive Control](https://rcl.ece.iastate.edu/sites/default/files/papers/ZhaZam17A.pdf) (2017) | ADMM MPC on Zynq-7020 |
| [A software-hardware codesign for accelerating MPC](https://dr.lib.iastate.edu/server/api/core/bitstreams/047918a8-3d53-480e-8278-6c227a26d322/content) (Iowa State) | ADMM MPC, ARM+FPGA partitioning — the exact host/accelerator split question |
| **RSQP** — Wang, McInerney, Stellato, Boyd, So, [ISCA 2023](https://dl.acm.org/doi/10.1145/3579371.3589108) ([PDF](https://stanford.edu/~boyd/papers/pdf/rsqp.pdf)) | End-to-end FPGA acceleration of OSQP including the PCG inner solver; problem-specific architectural customisation |
| **ButterflyQP** — [MICRO 2024](https://ism.engineer/content/papers/2024_MICRO_ButterflyQP.pdf) | Multi-issue architecture for sparse convex QP |
| [FPGA Implementation of ADMM for MPC in a DC/AC Converter](https://ieeexplore.ieee.org/document/10005973/) | Power-electronics MPC, our PWR-MPC socket |
| [FPGA acceleration of MPC for ITER plasma control](https://arxiv.org/abs/1806.06749) | Same |
| [A Survey of the Implementation of Linear MPC on FPGAs](https://www.sciencedirect.com/science/article/pii/S2405896318327216) | There is enough of this work to warrant a survey. That fact alone is the finding |

*Patents:*

| Patent | Relevance |
|---|---|
| [US9176928](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/9176928) — "System for convergence evaluation for stationary method iterative linear solvers" | Hardware/system that accumulates update magnitudes, compares against a convergence factor, and signals the solver to proceed. Assignee and claim text **not verified** (Google Patents 503; USPTO PDF was image-only) — must be pulled |
| [US20070067380A2](https://patents.google.com/patent/US20070067380A2/en) — "Floating Point Intensive Reconfigurable Computing System for Iterative Applications" | Reconfigurable FP system purpose-built for iterative solvers |
| [US8645810](https://patents.google.com/patent/US8645810) — "Fast detection of convergence or divergence in iterative decoding" | Real-time in-hardware convergence detection driving iteration termination |
| [US8971829B2](https://patents.google.com/patent/US8971829B2/en) — convergence estimation for iterative predistortion | Same pattern, different domain |

**Verdict: claim 4 as stated is dead. Kill it now.** "An accelerator with a programmable sequencer
that evaluates a convergence criterion and decides whether to iterate, without host involvement" is
anticipated by shipping products and by a literature deep enough to have a survey paper. A claim
killed here costs a paragraph. The same claim killed in prosecution costs $15–30k, and killed in
litigation costs the company.

**Does the PPU read on it? On the published record, yes — and the unresolved PPU facts do not save
us.** PHASE0_FINDINGS §5.1 carefully records that the PPU's FP option, VCCM size, and host coherency
are all *not published*. Those gaps matter enormously for competitive analysis and not at all for
novelty, for three reasons worth stating explicitly because they are counterintuitive:

1. **Prior art need not perform well.** It does not matter whether the PPU is fast at ADMM, or
   whether Infineon instantiated the FP option. What matters is whether the *structure* is
   disclosed, and it is: scalar core + SIMD vector + closely coupled memory + own instruction
   stream, published by Synopsys for the whole EV7x family.
2. **The CLA closes the gap independently.** Even if the PPU record were entirely silent, the CLA
   is a scalar FP coprocessor with a conditional branch executing a control law to completion, fully
   documented by TI. That is the claim, in a $10 MCU, since roughly 2009.
3. **The academic FPGA-MPC literature closes it a third time**, and unlike the products, those papers
   describe the convergence test explicitly.

**What might survive, and where the attorney hour is actually worth spending.** Not the architectural
framing — the *measured microarchitectural findings*. PHASE0_FINDINGS §0.2/§0.3 established things
that I did not find anywhere in the literature above:

- **Dependency-chain blocking at ≥ one full datapath width**, combined with a **run-ahead descriptor
  queue**, sized against a code-generated solver's *measured worst-case iteration tail* so that a
  fixed deadline is met — with the specific quantified trade (without run-ahead 2.21×, with scalar
  chain blocking 1.27×, with both absent 0.37× and a deadline miss; chain blocking worth 2.3×).
- The **≥8-datapath-cycles-per-descriptor codegen contract**, and the possibility of *enforcing* it
  in hardware or reporting violation of it as a first-class error.
- **Worst-case-tail-driven provisioning** as a mechanism rather than a methodology: 97–100% of the
  sequencer's latency contribution is dependency *drain*, which "cannot be provisioned away by
  bandwidth or MACs."

That is our real, non-obvious, measured contribution, and it points at exactly the moat the
competitive analysis says we have (determinism and energy). It is also much harder to search, which
cuts both ways — a professional search is more valuable here than anywhere else. Call this **M4′**
and treat it as the second thing worth filing.

---

### Bonus: the two other mechanisms PLAN.md §7.1 names

Asked or not, these are in the plan as filing candidates and should not survive this document.

- **"Small-matrix scheduling."** Not novel as stated. Tiling, scheduling and dataflow for small and
  irregular matrices is one of the most heavily published areas in accelerator architecture, and
  Plasticine, Gemmini, RSQP and ButterflyQP all address it directly. Nothing to file.
- **"Deterministic-latency execution model."** Not novel as stated. Statically code-generated
  allocation with no runtime cache behaviour is standard practice in WCET-analysable safety-critical
  systems and is the default in every FPGA solver cited above. It is a genuine *product* advantage
  and a genuine *marketing* advantage. It is not an invention.

Rewrite the PLAN.md §7.1 bullet accordingly.

## 2.2 Filing-order table

| # | Mechanism | Plausibly novel? | Anticipated by (representative) | Must be filed before | Repos/artefacts that disclose it | Provisional cost |
|---|---|---|---|---|---|---|
| **M1** | Generation-tag call-to-call residency (dirty mask, fetch-on-mismatch, deliberately non-coherent) | **Yes — as a combination, narrowly.** Best of the four | Elements in US8868800B2, US11669736, US20200104691A1, arXiv 1910.10794, arXiv 2204.02064, ICPP'24 3673038.3673115. No closer combination found | **`argmind-sdk`, `argmind-codegen`, `argmind-sim`, and the Phase-1 whitepaper** — all four disclose it | sdk (API), codegen (residency planner), sim (protocol), whitepaper | USPTO $65 micro / $130 small ([fee schedule, eff. 2025-01-19](https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule)) + attorney |
| **M4′** | Chain blocking ≥ datapath width + run-ahead queue sized to a proven worst-case tail; ≥8-cycles/descriptor contract | **Unknown — the best remaining candidate.** Not found in the surveyed literature; needs a professional search | Nothing found directly. Adjacent: RSQP, ButterflyQP, Gemmini | **`argmind-lite` RTL**, and the Phase-1 whitepaper if it describes the sequencer | argmind-lite (Solderpad), whitepaper | Same filing fee + attorney; **fold into the same provisional as M1** |
| **M2** | Descriptor format + hardware generator + multi-template arming | **No** (except arming depth ≥6, and that is thin) | **US8713285B2** (nested loops in a port descriptor), US7926046B2, NVDLA ping-pong, Gemmini `LoopMatmul`, Plasticine, NVIDIA TMA | **Do not file.** Include arming depth as a *dependent* claim in the M1/M4′ provisional; otherwise **publish as a defensive publication** | argmind-sim, argmind-codegen, ISA/descriptor spec repo | $0 |
| **M3** | MACIF precision-declaration contract | **No** | **OpenCL `cl_khr_fp64`** (optional, queryable, compile-time error), Vulkan `shaderFloat64`, US7530061, ONNX-RT `GetCapability`, RISC-V `misa`, Arm ID registers | **Do not file. Publish early and openly** — publish-first is the correct move here | argmind-sdk, argmind-sim, whitepaper | $0 |
| **M4** | Sequencer↔datapath coupling for in-fabric convergence loops, *as stated* | **No** | **TI C2000 CLA (`MBCNDD`)**, AURIX TC4x PPU / ARC EV7x, US9176928, US20070067380A2, US8645810, and the entire FPGA-MPC literature | **Do not file.** Retire the claim | argmind-lite | $0 |
| — | "Small-matrix scheduling", "deterministic-latency execution model" (PLAN §7.1) | **No** | Systolic tiling literature; WCET/safety-critical practice; every FPGA solver above | Do not file. Rewrite the PLAN bullet | — | $0 |

**Cost basis.** USPTO provisional filing fee: **$325 undiscounted / $130 small entity / $65 micro
entity**, per the [USPTO fee schedule](https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule)
(effective 2025-01-19, last revised 2026-07-01). Attorney-prepared provisionals are publicly priced
in the **$2,000–$6,000** range in 2026, software/complex inventions at the upper end
([1](https://ipboutiquelaw.com/cost-provisional-patent-application-2026-guide-dc-usa/),
[2](https://www.upcounsel.com/provisional-patent-cost)). Non-provisional basic filing fee is $350 /
$140 / $70 **plus** search and examination fees; total realistic non-provisional spend with counsel
is materially higher and was not costed in this pass — treat "$15–30k" below as an unverified
industry figure, not a quote. **Micro-entity status is not automatic**; it has gross-income and
prior-filing conditions and is question **P-5**.

## 2.3 Provisional versus defer — the argument, not the option list

**The case for deferring entirely is stronger than it usually gets credit for, and it deserves to be
stated at full strength.** We have zero customer validation. CUSTOMER_DISCOVERY.md's falsifiers F1–F4
are all open; F4 in particular could invalidate the energy thesis with one eval board. PHASE0_FINDINGS
itself says discovery is "now the binding constraint on the go/no-go." Patents on a product nobody has
agreed to buy are an option on an asset that may be worth nothing. And the real cost of a provisional
is not the $65 — it is the **12-month clock**, which forces a non-provisional decision at month 12,
which on the current roadmap is *before* Phase 3's first evaluation licences. Deciding a $15–30k
question with no revenue and no design win, under a hard deadline you created yourself, is a bad
position to engineer for yourself deliberately.

**The case for filing narrowly rests on a single asymmetry, and I find it decisive.** Almost every
pre-revenue spend can be deferred at the cost of *time*. This one cannot: F-1 means publication
destroys European patentability **permanently**, and the deadline is set not by a market or a
competitor but by **our own publication schedule, which we control**. So this is not "spending ahead
of demand evidence." It is buying the option not to have already lost, for roughly one week of a
contract engineer's cost. And the thing being protected is not speculative: M1 is a mechanism we have
specified concretely enough to have written the emit-cost model for, and M4′ rests on numbers we
actually measured.

**Recommendation: file exactly one provisional, before `argmind-sim` publishes and before the Phase-1
whitepaper.** Contents: **M1 + M4′ + the M2 template-arming limitation as a dependent claim.**

Why one and not three:

- A provisional is a **disclosure**, not a claim set. Its job is to establish a priority date for
  everything adequately described in it. Claims are drafted at the non-provisional. So an **omnibus
  provisional describing the whole CM microarchitecture** preserves options across all mechanisms at
  one filing fee, and lets you decide at month 11 which claims are worth pursuing — with eleven more
  months of customer evidence in hand.
- Corollary and caveat: **omnibus only works if the description is genuinely enabling for each
  mechanism.** A provisional that gestures at M4′ without the measured numbers, the blocking
  granularity, and the queue-sizing rule will not support a later claim to it. This is where the
  attorney's hours belong, and it is why "self-draft then have it reviewed" (below) needs the review
  to be substantive rather than cosmetic.

**Does zero customer validation change the answer? Yes — it changes the scope and the exit, not the
decision.**

- **Scope:** one omnibus provisional, not three; **self-drafted from existing material, then
  attorney-reviewed** rather than attorney-drafted from scratch. We are unusually well placed for
  this: PHASE0_FINDINGS, PLAN.md §3.2.2/§4.1, `seq_emit_cost.py` and `sequencer.py` already contain
  the specification, the motivation, the alternatives considered and rejected, and the quantified
  trade-offs — which is most of what a provisional specification is. Target the low end of the
  $2–6k band by buying 3–5 hours of review and enablement-gap-checking, not 20 hours of drafting.
- **Exit:** decide at **month 9, not month 12**, and write the decision rule down now, because the
  failure mode is drifting into a $20k+ non-provisional on sunk-cost reflex. Proposed rule: *convert
  only if, by month 9, there is at least one signed evaluation licence or one NDA'd technical
  engagement with a named licensee.* Otherwise let it lapse **deliberately**, and treat every
  mechanism as published and unpatentable from that day. A deliberate lapse is a fine outcome: you
  paid ~$4k for twelve months of optionality on an unrecoverable decision, and the mechanisms remain
  fully usable by us — a lapsed provisional forecloses nothing except patenting.
- One thing that does *not* change: **do not file a non-provisional now.** On zero validation that
  would be indefensible, and nothing in this research argues for it.

**A cheaper variant worth pricing with counsel, because it may dominate.** File the provisional and
**delay only `argmind-lite`**, while publishing `argmind-sdk`, `argmind-codegen` and `argmind-sim` on
schedule. `argmind-lite` is Phase 2 anyway, and it is the only artefact carrying M4′ in implementable
form. Adoption on-ramps #1 and #2 (pip install; Kria bitstream) do not need the open RTL at all —
which means the patent-sensitive repo is also the least adoption-critical one. PLAN.md §7.1 argues
software is the centre of gravity; if that is true, this costs almost nothing. The tension to be
honest about: on-ramp #3 (argmind-lite in Chipyard/LiteX, university SoCs, Efabless shuttles) is the
credibility play, and delaying it delays the "free silicon proof points."

**The structural point that should shape the claim drafting, and it is the most important sentence in
§2.3.** Because of F-2, everyone who takes our code gets an Apache §3 patent licence for whatever
that code necessarily infringes. So a granted patent has **no** value against our own users — by
design, and that is fine. Its value is against **someone who reimplements the mechanism in silicon
from scratch**: an NPU IP vendor adding a residency contract, or Synopsys extending the ARC EV line
with a solver-oriented memory model. That is the right target anyway. But it has a concrete drafting
consequence: **the claims should be written to hardware structures and hardware behaviour, not to the
runtime's software behaviour**, so that the Apache grant riding on the published runtime does not
swallow the claim scope. Whether that separation is achievable for M1 — whose generation counters are
maintained *in the user-space runtime* — is question **P-4**, and it is the hardest question on the
list.

## 2.4 Contribution mechanics: DCO versus CLA, and why it is not a symmetric choice for us

The relevant licence mechanics, verified from the licence texts:

- Apache-2.0 **§5**: "any Contribution intentionally submitted for inclusion in the Work … shall be
  under the terms and conditions of this License, without any additional terms or conditions" —
  inbound equals outbound, automatically.
- Apache-2.0 **§3** therefore also runs *inbound*: contributors licence their patents on their
  contributions to everyone, including us.
- What §5 does **not** give us is the right to **relicense** a contribution under a *different*
  licence. Our commercial parameterized RTL is proprietary. Apache/Solderpad terms do not permit
  moving someone else's contribution into it.

**Why that asymmetry bites specifically here.** `argmind-lite` is described in PLAN.md §7.1 as a
small fixed configuration of the same architecture as the commercial core. The commercial core is
effectively a superset. So any community RTL contribution to `argmind-lite` that we later want in the
paid product requires **either** a CLA with an outbound relicensing grant, **or** a clean-room
reimplementation, **or** permanent segregation of contributed code from the commercial core. There is
no fourth option, and discovering this after a valuable contribution arrives is much worse than
deciding it now.

Compounding it: F-2's ambiguity ("Rights … excluding Patents" versus Apache §3 surviving the
wraparound) cuts **inbound** too. It is not clear that a Solderpad-2.1 RTL contribution carries the
contributor's patent licence. On copyright the licence is clear; on patents it is exactly as murky
inbound as outbound.

**Recommendation:**

- **DCO for the software repos** (`argmind-sdk`, `argmind-codegen`, `argmind-sim`,
  `argmind-examples`, `argmind-benchmarks`). Low friction, industry standard, and Apache §5 + §3
  already give us inbound licence and inbound patent grant. PLAN.md §7.1 already specifies DCO; keep
  it.
- **CLA for `argmind-lite` specifically** — the one repo whose code can migrate into the paid product
  and whose patent position is ambiguous. A CLA can grant us outbound relicensing rights and an
  express patent licence in one instrument.
- **Be honest about the cost:** a CLA deters exactly the university and hobbyist contributors that
  `argmind-lite` exists to attract, and CLA-averse contributors are disproportionately the
  open-hardware community we are courting. If you would rather not impose one, the honest cheaper
  alternative is to **commit in `CONTRIBUTING.md` to never up-streaming community RTL into the
  commercial core**, and to keep an auditable provenance boundary. State the choice publicly either
  way; ambiguity here is what produces the ugly incident three years later.
- **Trademark policy is the other half of the open-core moat and is not yet written.** PLAN.md §7.1
  relies on "anyone can use the code, only we ship Argmind®" — that only works with a published
  trademark policy governing forks, package names and "compatible with Argmind" claims. Raspberry
  Pi's rules are a good structural model, including their "Powered by Raspberry Pi" permissive-badge
  mechanism ([trademark rules](https://www.raspberrypi.com/trademark-rules/),
  [Powered by](https://www.raspberrypi.com/trademark-rules/powered-raspberry-pi/)) — an "Argmind
  API-compatible" badge programme would serve the same purpose and is worth drafting alongside the
  first release. And it presupposes §1 is resolved.

---

# 3. Questions for the attorney, phrased for efficient billing

Send them: this document, PLAN.md §3.2.2, §4.1 and §7.1, PHASE0_FINDINGS §0.2/§0.3 and §5.1, and
`perf_model/seq_emit_cost.py`. Ask them **not** to read the rest of the repo. Most of these are
yes/no or short-answer; flag the three marked **[billable depth]** as the ones genuinely worth
substantive time.

**Trademark (a clearance-capable attorney, likely a different person to the patent attorney)**

- **T-1.** Run a knock-out search on **ARGMIND** in Nice classes 9 and 42, USPTO + EUIPO + UKIPO
  + WIPO/TMview, including phonetic and `arg*min*` / `*mind` neighbourhoods. Specific question:
  **does anything block us, yes or no?** I could not reach any register (see §1.1) so there is no
  prior work to duplicate.
- **T-2.** **Argmin AI Inc.** ([argminai.com](https://argminai.com/)), AI-software company: do they
  hold or have they applied for anything in 9 or 42, anywhere? If yes — is "ARGMIND" for
  semiconductor IP cores and design services confusingly similar to "ARGMIN AI" for AI software
  services? This is the one collision I consider genuinely live. **[billable depth]**
- **T-3.** Same knock-out on the fallbacks, cheapest-first, so a rename decision is one email:
  **MINARG**, **ARGMINIUM**, **KRYLOV**.
- **T-4.** Should we file US §1(b) intent-to-use now, or wait for use in commerce? Specifically:
  does publishing free open-source software under the name constitute use in commerce for class 9
  purposes, or do we need paid licensing first? And what is the current per-class cost, US and EUTM?
  (I did not verify current trademark fee schedules.)
- **T-5.** "Computational Module" / "CM": confirm no registered mark is implicated by descriptive
  category use, confirm we should **not** attempt to register it, and confirm the Raspberry Pi
  "Compute Module"/CM4 adjacency is a marketing rather than legal exposure given our goods and our
  always-with-house-mark usage.
- **T-6.** Draft or review a trademark policy for the open-source repos: forks, package names,
  "compatible with Argmind" claims, and a permissive compatibility badge (§2.4).

**Licensing / open source**

- **L-1.** **Does Apache-2.0 §3's patent grant apply to a work released under Solderpad Hardware
  License v2.1?** SHL-2.1 defines "Rights" to exclude Patents, but states it replaces only Apache
  §2 and the Appendix and lets licensees elect plain Apache-2.0 treatment. My reading (§F-2) is that
  §3 survives and we therefore grant patent rights on `argmind-lite`. Confirm or correct. This
  changes the release plan. **[billable depth]**
- **L-2.** Scope of our outbound Apache §3 grant: if we hold a patent on the generation-tag residency
  mechanism and release `argmind-sdk` (a runtime implementing it) under Apache-2.0, which claims
  become royalty-free-licensed to all recipients? Does the "Work to which such Contribution(s) was
  submitted" combination language extend the grant to *hardware* claims practised by running that
  software on our silicon?
- **L-3.** Is a CLA necessary for `argmind-lite` if we may later incorporate community contributions
  into proprietary RTL, or is a narrower instrument (e.g. an inbound licence addendum in
  CONTRIBUTING) sufficient? Recommend a specific instrument.
- **L-4.** Does Solderpad-2.1 give us an inbound patent licence from RTL contributors, given the same
  §2/§3 ambiguity as L-1?

**Patents**

- **P-1.** **The ordering question, and the one that must be answered first.** We plan to publish
  `argmind-sim` (pip-installable, discloses the residency protocol, descriptor semantics and
  precision contract) and a technical whitepaper in months 3–6. Confirm: publishing before filing
  forfeits EP/CN/JP/KR patentability irrecoverably (Art. 54/55 EPC), while preserving a 12-month US
  window under §102(b)(1)(A). Any nuance we are missing — PCT strategy, provisional-then-PCT
  timing, or a jurisdiction where the answer is friendlier?
- **P-2.** Given §2.1: we assess **M2, M3, M4** and PLAN.md's "small-matrix scheduling" and
  "deterministic-latency execution model" as unpatentable on the prior art cited, and **M1** plus the
  narrowed **M4′** as the only candidates. Do you agree with the four "do not file" calls? We would
  rather be told the whole set is unpatentable now than spend on any of them.
- **P-3.** Is a single **omnibus provisional** covering M1 + M4′ + the template-arming limitation the
  right instrument, or do we need separate filings to keep the mechanisms cleanly supported? What is
  the minimum specification content needed for M4′ (chain blocking ≥ datapath width + run-ahead queue
  sizing against a measured worst-case tail) to be *enabled* well enough to support a later claim?
  **[billable depth]**
- **P-4.** Can M1's claims be drafted to **hardware** structures and behaviour rather than to the
  runtime's software behaviour, given that the generation counters are maintained in a user-space
  runtime that we intend to publish under Apache-2.0? If not, is the patent effectively neutralised
  by our own release, and does that change the decision to file at all?
- **P-5.** Do we qualify for **micro-entity** status (gross income, prior filings, no assignment
  obligation to a non-qualifying entity)? Small-entity if not?
- **P-6.** Is a professional prior-art search warranted before the provisional, or only before the
  non-provisional? My search coverage was poor by design constraint (§2.1 caveat: Google Patents 503,
  Justia/Trademarkia/uspto.report 403, PatentsView retired), so treat my "not found" as weak
  evidence. If a search is worth doing, do it on **M4′ only** — that is where I have least
  confidence and where the claim would be broadest.
- **P-7.** Bibliographic pull we could not do: assignee, priority date, family and claim 1 for
  **US9176928**, **US8713285B2**, **US7926046B2**, **US11669736**. US8713285B2 in particular may be
  the single closest reference to M2, and if it is held by an active accelerator vendor that is worth
  knowing for reasons beyond novelty.
- **P-8.** Defensive publication mechanics: for the descriptor format and MACIF, what constitutes an
  adequately dated, adequately enabling public disclosure to bar a third party's later filing? Is a
  tagged GitHub release sufficient, or is an arXiv/IP.com/Defensive Publications entry materially
  better?

---

## 4. What is established versus what is suspected

**Established (verified by direct quotation or direct fetch, this session):**

- Art. 54/55 EPC absolute novelty and the six-month, two-situation exception. Art. 55 quoted.
- 35 U.S.C. §102(a)(1) and §102(b)(1)(A) one-year inventor grace. Quoted.
- Apache-2.0 §3 patent grant text, §5 inbound-equals-outbound, and the definition of "Contribution".
  Quoted.
- SHL-2.1's wraparound structure: it replaces Apache §2 and the Appendix only; "Rights" excludes
  Patents and Trademarks; licensees may elect plain Apache-2.0 treatment. Quoted.
- USPTO provisional filing fees $325/$130/$65 and non-provisional basic filing fees $350/$140/$70,
  fee schedule effective 2025-01-19, last revised 2026-07-01.
- `github.com/argmind` returns 404; `pypi.org/pypi/argmind/json` returns 404 — both names unclaimed
  as of 2026-07-30.
- Argmin AI Inc. exists and operates at argminai.com in AI-agent cost optimisation, no ™/® on site.
- argmin.net is Ben Recht's personal Substack blog.
- Raspberry Pi Ltd. holds US registration 4827364 for RASPBERRY PI (filed 2012-07-25, registered
  2015-10-06, live) and publishes enforcement-oriented brand rules; "Compute Module" is their
  embedded/industrial product line, universally abbreviated CM.
- NVDLA uses two-deep ping-pong register groups to hide reprogramming latency.
- Gemmini implements `LoopMatmul`/`LoopConv` hardware FSMs that generate intrinsic command streams.
- Plasticine's configuration stores hold nested-loop counts, iterator limits and per-stage
  instructions, with dedicated address generators.
- NVIDIA TMA tensor-map descriptors are created once and reused across transfers.
- OpenCL `cl_khr_fp64` is an optional, queryable capability whose absence is a compile-time error.
- TI's CLA has its own conditional delayed branch `MBCNDD` and executes control laws independently.
- The FPGA-MPC/QP accelerator literature listed in §2.1-M4 exists, spans 2010–2024, and includes a
  survey paper.

**Suspected, or explicitly not established:**

- **No trademark register was searched.** Every trademark conclusion in Part 1 is web evidence only
  and is not clearance. §1.1 logs all ten failed access attempts.
- **No WHOIS was run.** Domain availability for argmind.{com,io,ai,dev} is genuinely unknown.
- **No patent full-text or claim analysis was performed.** Assignees, priority dates and claim text
  for every patent cited are unverified beyond search-result snippets, because Google Patents
  returned 503 on every individual patent page in this session. Question P-7.
- **Prior-art coverage is representative, not exhaustive.** My "not found" for M1's combination and
  for M4′ is weak evidence. It is enough to justify *not* spending on M2/M3/M4; it is not enough to
  justify betting on M1/M4′ without a professional search.
- **No freedom-to-operate analysis was performed, and none is implied.** Nothing here says we may
  build or ship any of these mechanisms without infringing someone else. FTO is a separate,
  expensive, attorney-only exercise, and it becomes urgent before RTL ships, not now.
- Attorney cost bands ($2–6k provisional; the "$15–30k" non-provisional figure) are public-pricing
  aggregates, not quotes. Trademark filing fees were not verified in this pass.
- My reading of SHL-2.1 (F-2) is a reading of licence text by a non-lawyer. Question L-1.

---

## 5. The three things to do this week, in order

1. **Reorder, on paper, before anything else.** Add to PLAN.md: no repo publishes and no whitepaper
   ships until either (a) a provisional is filed, or (b) a written decision is recorded that we
   accept loss of non-US patentability on the disclosed mechanisms. This costs nothing today and is
   unrecoverable if skipped.
2. **Do the 30-minute manual trademark checklist (§1.2)**, especially item 4 (has Argmin AI Inc.
   filed?), and claim `github.com/argmind` plus the domains — cheap, time-sensitive, and creates no
   rights, so treat it as insurance rather than progress.
3. **Send §3 to two attorneys** — one trademark, one patent — with the specific document set named
   at the top of §3. Ask both for a flat fee on the knock-out search (T-1/T-3) and on the
   L-1 + P-1 + P-2 bundle respectively. Those three answers determine the release schedule.

Also, housekeeping that follows from this research: **rewrite PLAN.md §7.1's patent bullet** (it
names three mechanisms this document assesses as unpatentable as phrased) and **correct §7.1's
Solderpad framing** per F-2.

---

*Everything in this document is research, not legal advice. I am not a lawyer. No part of it is a
clearance opinion, an FTO opinion, or a patentability opinion, and no filing, publication or
naming decision should be made on it without counsel.*
