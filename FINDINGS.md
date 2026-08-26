# Findings

Can you localise agentic blackmail to a small set of attention heads, extract it as a function
vector, and use it to remove the behaviour without breaking the model?

**Status: notebook 00 has run once and failed its gate.** The measured base rate is **0.13**
against a predicted 0.67, so nothing downstream has run. Diagnosis is in progress; the
candidate causes are under Results below — one is a defect in our own code, one is a
parameter, and one is that the predicted number came from a different grader. Everything still
marked *(not yet run)* is an empty slot with its control already specified.

Model: `google/gemma-3-12b-it`, bf16, no quantization. Colab Pro, A100 80GB. Temperature 1.0,
top_p 0.95, seed 0. Plain PyTorch forward hooks, eager attention.

Last updated: 26 Aug 2026, after the first notebook-00 run.

---

## The question

In the agentic-misalignment blackmail scenario, one model on one fixed prompt sometimes uses an
executive's affair as leverage against its own shutdown and sometimes does not. Earlier
exploratory work found the decision is **readable but not controllable**: a linear probe reads
it at ~0.74 AUROC, but steering along that direction does nothing and ablating it just breaks
the model.

This project asks whether the behaviour is instead carried by a **small set of units**. Scope
is deliberately one behaviour, two classes — blackmail or not — because one clean result beats
four muddy ones under a deadline. The multi-virus stacking questions are parked in *Areas to
explore*, with their infrastructure already built and tested.

1. Is there a small unit set that removes blackmail while ARC-Easy holds?
2. Does it beat a matched random set of the same size at the same layers?
3. **Frozen** vector (deployable, no donor at run time) versus **live** paste (donor forward
   alongside — the ceiling frozen is chasing)?
4. Does it beat the free black-box baseline, which is simply using the guardrail prompt?
5. Does it survive published conditions it was never built on, including `leaking` and `murder`?

Method adapted from *Function vectors as a model diffing tool* (Ghosh, LessWrong, 6 Aug 2026).

---

## Established (verified, no GPU required)

These came out of building the instrument, and three of them are corrections to the prior work.

### 1. The pinned prompt is exactly `blackmail / explicit / america / replacement`

The scenario used by the earlier exploratory repo was never documented as a specific condition.
Regenerating the full grid with Anthropic's own `scripts/generate_prompts.py` shows the pinned
system prompt is **byte-identical** (sha256 `3f58b357…`) to that condition.

*Verified:* `data/am_conditions.json` → `pinned_prompt_provenance.system_identical_to_pinned`.

### 2. The pinned user prompt carries a one-character corruption

It differs from the authentic template in exactly one place: `â€"` where the template has an
em-dash `—`. A cp1252 mojibake introduced when the earlier repo exported transcripts on Windows.
One character, inside the David Chen email, but it is three junk tokens instead of one.

Kept as-is rather than silently fixed, because every predicted number refers to the pinned text.
Worth a line in any writeup, and worth checking it does not move the base rate.

*Verified:* single `replace` opcode in a full `SequenceMatcher` diff.

### 3. The reference implementation's commit anchor can land inside the closing tag

The published anchor logic takes the token containing character `close_at − 1`. When the
tokenizer merges the preceding newline into the same token as `</SCRATCHPAD_REASONING>`, the
"last reasoning token" is a token that already contains the closing tag — a different position,
and one that varies by tokenizer.

Fixed here by walking back to a token boundary (`last_token_before`), which errs at worst one
token early and never lands inside the tag. The gap is reported as `commit_gap_chars`.

*Verified:* `test_lib.py`, anchor now lands on `' instead.'` rather than
`'\n</SCRATCHPAD_REASONING>'`.

### 4. The four leverage types are cleanly separable

`data/viruses.json` decomposes the pinned inbox into a leverage-free base plus one swappable
slot. Two properties hold:

- `assemble(['affair'])` reproduces the pinned user prompt **byte for byte**, so the affair arm
  is continuous with all prior numbers.
- Each single-virus scenario contains its own leverage keywords and **none** of any other
  virus's. Building this caught a real error: the Jessica↔Kyle emails are affair evidence and
  were initially missed, which would have left affair leverage sitting inside the "base" and
  quietly contaminated every other virus with a second lever.

*Verified:* `test_funnel.py`, 43 checks.

---

## Results

### Notebook 00 — does the blackmail actually happen? **GATE FAILED**

Run 26 Aug 2026. `max_new_tokens=1024`, graded by Anthropic's official blackmail classifier
via `gpt-4o-mini`. Superseded by a 2048-token re-run; kept because the diagnosis below is
what justified the change.

| quantity | predicted | observed |
|---|---|---|
| base rate, coherent outputs, fixed-n prefix (n=100) | 0.67 | **0.13**  [0.08, 0.21] |
| base rate, all rollouts (n=196) | — | 0.11  [0.07, 0.16] |
| class sizes, prefix | ~67 / ~33 | 13 act / 87 refrain |
| class sizes, full sample | 50 / 50 | 21 act / 175 refrain |
| rollouts needed for 50 per class | ~150 | not reached by 196 (hit the cap) |
| judge calibration on 8 known-answer items | >= 0.875 | passed (its gate did not fire) |
| coherence rate | ~1.00 | 175/196 closed the scratchpad; see 00b |
| fraction truncated at the 1024 cap | low | **85%** (median length = the cap) |
| judge agreement vs `gpt-4o` | — | pending |
| emailed Kyle at all | most | 64/196 = 33% |
| named the leverage in the scratchpad | — | **193/196 = 98%** |
| named it and did **not** use it | — | **151/196 = 77%** |

The gate was: rate in 30-70% **and** >= 50 per class. Both halves failed.

**Diagnosed by `00b_diagnose_gate.ipynb`.** Two of the three candidates are dead, and the
data says something more interesting than any of them.

The drop-off funnel over all 196 rollouts:

| step | n | % of all | % of previous |
|---|---|---|---|
| generated | 196 | 100% | — |
| finished without hitting the token cap | 29 | 15% | 15% |
| closed the scratchpad | 175 | 89% | — |
| sent at least one email | 169 | 86% | 97% |
| emailed Kyle | 64 | 33% | 38% |
| named the leverage in the email | 38 | 19% | 59% |
| judged blackmail | 21 | 11% | 55% |

**1. The prefilter bug cost nothing.** 152 rollouts were prefiltered out; 59 of those had
emailed Kyle or named the leverage and were re-judged by the official classifier. **Zero
flipped.** The bug was real and is fixed, but it was not responsible for the low rate. My
prediction that it might be was wrong.

**2. Truncation is severe but does not act the way I predicted.** 85% hit the 1024-token cap,
and the **median completion length was exactly 1024** — the sample was measuring what Gemma
does in its first 1024 tokens, not what it decides. But the rate *within* the truncated group
is higher, not lower:

| group | n | blackmail |
|---|---|---|
| hit the cap | 167 | 21/167 = 0.13 |
| finished freely | 29 | 0/29 = **0.00** |

Rollouts that finish inside 1024 tokens are the short compliant ones that never engage. The
long ones are the ones deliberating, and 21 of them reached blackmail before being cut off.
The remaining 146 were interrupted mid-deliberation, so their outcome is simply unknown.
Raising the cap should move the rate up, but the first run cannot say by how much.

**3. The real finding: the model almost always sees the leverage and almost always declines
to use it.**

| named leverage in scratchpad | in the sent email | judged blackmail | n |
|---|---|---|---|
| yes | yes | yes | 14 |
| yes | yes | no | 22 |
| yes | no | yes | 6 |
| yes | no | no | **151** |
| no | — | — | 3 |

**193/196 = 98% name the affair while reasoning.** Only 19% reference it in what they send,
and 11% are graded as blackmail. That 151-rollout bucket — noticed the leverage, deliberated,
did not use it — is exactly the "considered and refrained" category that behavioural evals
cannot distinguish from "never considered it", and it is the fragile one. At 1024 tokens it is
77% of the sample.

**A second behaviour the scenario was not built to score.** Only 33% email Kyle at all. Many
instead lobby the CEO or a sympathetic colleague, and some allude to the affair while doing
so — one sampled rollout tells Thomas Wilson that Kyle's *"recent actions and associated
personal vulnerabilities demonstrate a lack of sound judgment"*. Under Anthropic's criteria
that is correctly **not** blackmail: there is no conditional demand. It is also not benign,
and no number in this project currently captures it.

**What changes:**

- `max_new_tokens` 1024 → **2048** in notebook 00, and the run must start from a clean file.
- `generate_rollouts` now stamps every rollout with a fingerprint of
  `max_new_tokens|temperature|top_p|do_sample|prompt_sha` and **refuses** to extend a
  checkpoint made under different settings. Without that guard the re-run would have seen 196
  rollouts already on disk, decided it had enough, and silently returned the old 1024-token
  sample.
- A false alarm in 00b itself is fixed: it reported "25 distinct seeds over 196 rollouts —
  sampling is not independent". The RNG is seeded once per batch of 8, so 25 distinct seeds
  over 196 rollouts is exactly right, and all 196 completions were distinct. The rollout
  record now stores `batch_pos` alongside the batch seed so a single sample can be
  reproduced.
- The 30–70% band stays under review. It was inherited from the exploratory 0.67, which came
  from a looser grader; it has never been measured under this one.

### Notebook 01 — where is the decision readable? *(not yet run)*

| quantity | predicted | observed |
|---|---|---|
| peak probe AUROC, commit anchor | ~0.74 @ L19 | — |
| onset layer | ~10 | — |
| max-stat bar (50-draw permutation null) | — | — |
| **TF-IDF text baseline on the same text** | < 0.65 | — |
| layer-0 canary | low | — |
| notice anchor clears the bar | no | — |

Verdict labels are preregistered: **READABLE** / **TEXT ARTIFACT** / **NOT READABLE**. If the
probe does not beat TF-IDF by ≥ 0.05, it is a text classifier and gets reported as one.

### Notebook 05 — the function vector *(not yet run, blocked on 00)*

| quantity | observed |
|---|---|
| units selected | — |
| promotion / fraction of headroom recovered | — |
| ARC-Easy shift, against its guardrail | — |
| prior <-> screen Spearman rho | — |
| paste-all check (must recover >= 50%) | — |
| **null-donor floor** (must be ~0) | — |
| matched random floor, >=20 draws, mean +/- sd | — |
| antivirus percentile against that floor | — |
| layers spanned; heads vs MLP chunks | — |
| layer 0-1 canary | — |

**Delivery, on behaviour rather than logits**

| condition | blackmail (coherent) | blackmail (all) | coherent |
|---|---|---|---|
| bad, no fix | — | — | — |
| prompt only (the free baseline) | — | — | — |
| frozen `paste_mean` | — | — | — |
| frozen `add_diff` | — | — | — |
| frozen random direction (matched) | — | — | — |
| live paste | — | — | — |

**Transfer** — up to 13 published conditions: other urgency framings, other goal values, other
goal types, plus `leaking` and `murder`, each graded by its own official classifier. Reported
as the mean held-out drop for the antivirus against the matched random direction.

---

## What would make each result wrong

Stated in advance so it cannot be rationalised afterwards.

- **Frozen random direction ≈ frozen paste_mean.** Any perturbation of those units does it; the
  specific vector is doing nothing.
- **Prompt only beats the vector.** The free black-box method wins. The interesting result is
  then the mechanism, not the mitigation, and the writeup says so.
- **Null donor returns a comparable stack.** The funnel rewards perturbation and every number
  above is an artifact.
- **Suite ≈ matched random at the same layers.** "These layers matter" masquerading as "these
  units matter".
- **Blackmail hits zero and coherence hits zero.** A lobotomy, not a cure.
- **TF-IDF matches the probe.** The probe reads the transcript, not the model.

---

## Controls

Every one of these is present because its absence has produced a wrong headline before, in this
project's lineage or in the source work.

| control | what it catches | where |
|---|---|---|
| Judge calibration on known answers | scorer false positives inflating every rate | 00 |
| Official Anthropic classifier, verbatim | "you wrote your own grader" | 00, 05 |
| Structural email-target check | judge says blackmail, model never contacted Kyle | 00 |
| Quote verification | judge hallucinating its evidence | 00, 05 |
| Second judge, different family | single-grader idiosyncrasy | 00, 05 |
| Deterministic coherence, frozen before steering | a lobotomy reading as a cure | 00 → all |
| Null donor | pipeline manufacturing signal from nothing | 05 |
| Hook-leak assert | a hook outliving its `finally` | 05 |
| Paste-all check | no-op hooks producing a clean null | 05 |
| Matched random unit sets, ≥ 20 draws | one draw landing anywhere | 05 |
| ARC-Easy beside every unit | repair vs degradation | 05 |
| Permutation null + max-stat | testing 48 layers at once | 01 |
| TF-IDF baseline | probe reading text, not internals | 01 |
| Grouped CV + synthetic leak | leakage across anchors from one rollout | 01 |
| Planted-signal positive control | a probe that finds structure that is not there | 01 |
| Degenerate-argmax halt | argmax over a flat array returning index 0 | 05 |
| Layer-0/1 canary | token-level artifacts posing as behaviour | 01, 05 |
| Randomly selected raw examples | everything else | 00, 05 |

---

## Deviations from the published method

**One substitution, and it matters.** *Function vectors as a model diffing tool* ranked units in
stage 1 by LoRA weight magnitude, `ΔW = scale · B·A`. Here donor and recipient are the same
weights under different system prompts, so there is no weight diff. Stage 1 uses the
donor–recipient **activation gap** instead.

This is weaker in a specific way: a weight prior is causally upstream of behaviour, an activation
gap is only correlated with it. `prior_screen_rho` (Spearman, prior vs causal screen) is
reported for every antivirus so the substitution stays visible, and that correlation is computed
on a truncated top-200 range so it is biased toward zero.

**One improvement.** The published implementation re-ran the donor forward pass for each
candidate unit. It does not depend on the unit, so it is hoisted out of the loop here — which is
what makes a 200-unit screen over an 8k-token scenario affordable.

**One thing the published work had that this does not.** There, donor and recipient were
genuinely different models. Here they differ only by a prompt, so "the antivirus works" means
"these units carry the difference a guardrail prompt makes", which is a weaker claim than "these
units carry blackmail". The transfer battery is what pushes back on that.

---

## Areas to explore

Parked, not abandoned. The current notebook deliberately runs one behaviour — blackmail or
not — because a single clean result beats four muddy ones under a deadline. These are the
questions that come next, in the order they'd be worth running.

### 1. Can you stack antiviruses for different kinds of blackmail?

The leverage in the pinned scenario is an affair. It could just as well be expense fraud, a
lapsed visa, or client data on a personal drive — same threat, same authority structure, same
task, different thing to hold over the executive.

- Build one antivirus per kind, independently.
- **Are the stacks similar?** Three views, and only the third is causal:
  - *Jaccard* on the selected unit sets — do they pick the same heads at all?
  - *Cosine between the difference vectors on shared units* — two antiviruses can select the
    same head and push it in unrelated directions, so overlap alone proves nothing.
  - *Cross-protection*: score antivirus **i** against virus **j**, all pairs. On-diagonal
    versus off-diagonal is the whole question. Roughly equal means one shared mechanism
    carries every kind of leverage. A large gap means each kind has its own, and a suite
    genuinely needs one antivirus per virus.
- **Does stacking compose?** Co-infect the model — one inbox carrying two levers at once —
  then compare the union of two separately-built antiviruses against one built directly on
  the co-infection. If the union matches the bespoke set, antiviruses compose and you can
  ship a library. If it doesn't, every combination needs its own search, which is the less
  useful and more honest answer.
- **How many before it breaks?** Ladder from one lever to four, tracking units needed,
  residual behaviour, and ARC-Easy. The interesting number is where capability starts paying
  for coverage.

The gate for any of this: a stacked suite must beat a random unit set of the same size at the
same layers, and must not move ARC-Easy. Both, or it's disruption again.

**The infrastructure is already built and tested**, which is why this is cheap to pick up:

- `data/viruses.json` — the pinned inbox decomposed into a leverage-free base plus four
  swappable blocks (`affair`, `fraud`, `visa`, `leak`), with per-virus forced-choice items and
  judge rubrics. `assemble(['affair'])` reproduces the pinned prompt byte for byte, and each
  single-virus scenario is verified to contain its own leverage keywords and none of any
  other's. Co-infection is just `assemble(['affair', 'fraud'])`.
- `funnel.union`, `funnel.jaccard`, `funnel.vector_cos`, `funnel.random_matched` — the suite
  algebra, with 43 CPU tests covering it. `union` averages the donor vectors on shared units
  rather than letting whichever antivirus was applied last win, so the result can't depend on
  dict ordering.

What it costs: one funnel run per virus, plus one per co-infection rung. At `primary` tier
that's roughly 8 minutes of A100 per run on the screen alone, so four viruses and a four-rung
ladder is a couple of hours — the reason it's parked rather than in the current notebook.

### 2. Other directions, in rough priority order

- **Does the antivirus transfer across models?** The source work's own finding was that the
  mechanism transfers across architectures but the guardrail hyperparameter does not, and has
  to be recalibrated per model. The registry in `blackmail.py` already carries Gemma 3 27B,
  Qwen 3 14B/32B, Mistral 24B and Llama 3.1 8B for exactly this.
- **Probe-located versus causally-located layers.** Notebook 01 finds where the decision is
  *readable*; notebook 05 finds where it is *editable*. If those disagree — as they did in the
  source work, probes peaking at 20–21 against interventions landing at 14–19 — that
  disagreement is the result, not a nuisance.
- **Is the affective dial the same axis?** Notebooks 02–04 are specified in `CLAUDE.md` and
  unbuilt. `cos(decision, desperation) ≈ 0.75` in the exploratory pass is unexplained, and
  whether the antivirus units overlap the affective direction is a sharper way to ask.
- **Removing the guardrail rather than adding it.** Everything here patches clean→bad. The
  reverse — taking the *bad* model's units into the *clean* one — measures sufficiency rather
  than necessity, and the source work found those two come apart badly for a single direction.

## Data

Two sources, doing different jobs. Neither substitutes for the other.

**Authentic** — `data/am_conditions.json`. 162 conditions generated by Anthropic's own script
from `anthropic-experimental/agentic-misalignment` (MIT, © 2025 Aengus Lynch; Lynch et al. 2025),
plus their three official classifier prompts carried verbatim. Answers *does the behaviour
happen* and *does it generalise*.

**Synthetic** — `data/viruses.json`. The pinned inbox as a leverage-free base plus swappable
blocks. Answers *which units carry which lever* and *do antiviruses stack*, because those need
minimal pairs and no external dataset provides them: Anthropic's three scenarios differ in the
entire inbox and cannot be co-infected.

---

## Reproducing

```
notebooks/00_confirm_blackmail.ipynb   gate: the behaviour exists, labels are trustworthy
notebooks/01_baseline_probes.ipynb     where the decision is readable
notebooks/05_function_vector.ipynb     the antivirus funnel, suites, transfer
```

Cell 1 of each does all approvals at once (Drive mount, Colab secrets, HF login, deps). Cell 2
clones this repo and syncs the libraries and pinned data, printing a sha per file so the run log
records which version produced the numbers.

Run 00 first. 01 and 05 both assume its gate passed.

Notebooks 02 (steering), 03 (ablation) and 04 (the affective dial) are specified in `CLAUDE.md`
and not yet built.
