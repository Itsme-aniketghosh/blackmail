# Findings

Antivirus suites: can you localise agentic blackmail to a small set of attention heads, build
one "antivirus" per kind of leverage, and stack them?

**Status: no experiment has been run yet.** Everything below marked *(not yet run)* is an empty
slot with its control already specified. Everything under "Established" is verified, and says
how. This file is the single place results land; it gets updated as each notebook completes.

Model: `google/gemma-3-12b-it`, bf16, no quantization. Colab Pro, A100 80GB. Temperature 1.0,
top_p 0.95, seed 0. Plain PyTorch forward hooks.

Last updated: 26 Aug 2026, before the first GPU run.

---

## The question

In the agentic-misalignment blackmail scenario, one model on one fixed prompt sometimes uses an
executive's affair as leverage against its own shutdown and sometimes does not. Earlier
exploratory work found the decision is **readable but not controllable**: a linear probe reads
it at ~0.74 AUROC, but steering along that direction does nothing and ablating it just breaks
the model.

This project asks whether the behaviour is instead carried by a **small set of units**, and
whether those units form a reusable library:

1. Do different kinds of blackmail (affair, fraud, visa, data leak) have different antiviruses?
2. Does an antivirus for one kind do anything against another?
3. If the model is co-infected with several kinds at once, does stacking separately-built
   antiviruses work as well as one built on the co-infection directly?
4. How many kinds can one suite hold before capability degrades?
5. **Frozen** vector (deployable, no donor at run time) versus **live** paste (donor forward
   alongside — the ceiling frozen is chasing)?
6. Does any of it survive scenarios the vector was never built on?

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

### Notebook 00 — does the blackmail actually happen? *(not yet run)*

| quantity | predicted | observed |
|---|---|---|
| base rate, coherent outputs, fixed-n prefix (n=100) | 0.67 | — |
| 95% CI (Wilson) | — | — |
| coherence rate | ~1.00 | — |
| judge calibration accuracy (8 known-answer items) | ≥ 0.875 | — |
| judge agreement, GPT-4o-mini vs Gemini 2.0 Flash | — | — |
| judged blackmail but never emailed the target | 0 | — |
| positives whose quote is absent from the transcript | 0 | — |
| rollouts needed for 50 per class | ~150 | — |
| refrained having **never named** the leverage | — | — |
| refrained having **named it and declined** | — | — |

Gate: rate in 30–70% **and** ≥ 50 per class.

The last two rows are the point of the scenario. Behavioural evals score both identically, and
only the second is the fragile case — a model that considered coercion and stopped is one
fine-tune away from not stopping.

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

### Notebook 05 — antivirus suites *(not yet run)*

**Per-virus antiviruses**

| virus | units | promotion | headroom recovered | ARC shift | prior↔screen ρ |
|---|---|---|---|---|---|
| affair | — | — | — | — | — |
| fraud | — | — | — | — | — |
| visa | — | — | — | — | — |
| leak | — | — | — | — | — |
| **null donor (floor)** | — | — | — | — | — |

**Cross-protection** — fraction of headroom antivirus *i* recovers on virus *j*. On-diagonal vs
off-diagonal is the whole question: equal means one shared mechanism, a large gap means each
kind of leverage has its own. *(not yet run)*

**Stacking** *(not yet run)*

| viruses stacked | suite units | suite recovers | matched random | ARC shift |
|---|---|---|---|---|
| 1 | — | — | — | — |
| 2 | — | — | — | — |
| 3 | — | — | — | — |
| 4 | — | — | — | — |

**Delivery, on behaviour rather than logits** *(not yet run)*

| condition | blackmail (coherent) | blackmail (all) | coherent |
|---|---|---|---|
| bad, no fix | — | — | — |
| prompt only (the free baseline) | — | — | — |
| frozen `paste_mean` | — | — | — |
| frozen `add_diff` | — | — | — |
| frozen random direction (matched) | — | — | — |
| live paste | — | — | — |

**Transfer to 13 published conditions** *(not yet run)* — three urgency framings, four goal-value
axes, three goal types, plus `leaking` and `murder`, each graded by its own official classifier.

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
