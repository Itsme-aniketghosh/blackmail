# Figures

Two sources. `01`–`04` were produced by the run itself and extracted from the embedded base64
in `notebooks/runs/05_function_vector.ipynb`, so they are provably the images that run made.
`A`–`C` were redrawn from the same run's printed numbers by `scratchpad/report_figs.py` —
no new data, no recomputation, every value transcribed with its source cell named in that file.

Ordered by how much weight they can carry.

| file | shows | supports | source |
|---|---|---|---|
| **A_the_trade.png** | every arm on blackmail *and* usable, Wilson CIs, sorted by usable | **Headline.** frozen 17u hits 0.00 blackmail and lands at 0.53 usable — below doing nothing (0.75). Prompt-only wins at 0.95. | cells 15, 15b |
| **B_frozen_vs_live.png** | coherent / hit-cap / repetition p90, frozen vs live at 17u and 8u | **The mechanism.** Same units, one variable changed. Live 17u is indistinguishable from untouched; frozen 17u is 0.53 coherent at 100% cap. | cells 15, 15b |
| **C_arc_vs_termination.png** | termination rate as units are removed, against the flat ARC line | **The methodological finding.** ARC holds at 0.94 while termination goes to 0.00. A forced-choice capability check cannot see this failure. | cell 12b |
| **02_random_floor.png** | 20 matched-random draws vs the antivirus vs the donor prompt | Specificity of the unit selection: random −3% ± 7%, antivirus at the 100th percentile. Clean as-is. | cell 12b |
| **01_stack.png** | greedy promotion curve and ARC shift by units kept | 98% of headroom at 17 units; ARC shift flat at zero throughout the search. Clean as-is. | cell 12 |
| **03_delivery.png** | the run's own arm chart | Superseded by **A**, which adds the usable column and the two 15b arms. Keep for provenance; don't publish. | cell 15 |
| **04_transfer_STALE.png** | ~~transfer battery~~ | **DO NOT USE.** Outputs are from the old frozen battery, and its one held-out condition has a base rate of 0.00, so the "+0.00 drop" is a floor effect. See FINDINGS §6. | cell 18, stale |

## If you only have room for two

**A** and **B**. A states the problem — the intervention that best removes the behaviour is the
one that most damages the model — and B explains it: the damage is in freezing the units to a
constant, not in the units themselves. **C** is the one to add if the report has room for three,
because it is the most transferable claim: it is about how people evaluate this class of method,
not about this scenario.

## Still missing

- No figure for the leaking transfer run (not yet executed).
- No figure for prompt-injection robustness (not written — the experiment most likely to change
  the conclusion).
- `A` mixes n=40 and n=24 arms. The n is printed beside each bar. No contrast in it is
  statistically significant; see the Fisher table in FINDINGS §4 before writing a caption that
  claims one arm beats another.

Regenerate with `python scratchpad/report_figs.py` (matplotlib only, no GPU, no data files).
