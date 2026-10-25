---
hide:
  - toc
---

# Every run

Twenty-six training runs over nineteen generator architectures. This is the complete log, including the runs
that did not make the report: nothing is left out because it was unflattering.

Each row names the class that produced it. Every class is in
[`src/mambahifigan/`](https://github.com/huseyin-karaca/mamba-hifigan/tree/main/src/mambahifigan) and has a
matching hparams file in
[`hparams/`](https://github.com/huseyin-karaca/mamba-hifigan/tree/main/hparams), so any row can be re-run.

--8<-- "snippets/table_all.html"

## Reading the table

**The families are explanations, not a filing system.** Each group of failures shares a mechanism, described on
the [method page](method.md): phase coherence breaks, gradients destabilise, or information gets squeezed
through a bottleneck. The grouping was assigned after the fact, from the behaviour of the runs and the audio —
it is an interpretation, and a different one is possible.

**The two losses can disagree, sharply.** `HifiganGeneratorBroadcast` has a better feature-matching loss than
the baseline and a 50% worse L1 spectrogram loss. Sorting by either column alone gives a misleading ranking,
which is why both are always shown.

**`d_state` was edited in the source, not configured.** In the original code the Mamba state dimension was a
literal inside each generator, changed by hand between runs. The packaged classes take it as a constructor
argument, defaulting to whatever was hardcoded in the file as it was last run. For the three earliest runs
(`Hybrid 1`, `Hybrid 2`, `Mamba only`, which carry no size in their name) the value is inferred as 64 from the
block default and is marked as such in the CSV rather than measured.

**Segment size changed partway through.** The five earliest runs used 8,192-sample segments; the rest used
12,288. Runs are only directly comparable within the same setting, which is why the baseline appears three
times — once per configuration the other runs needed to be compared against.

## Everything as data

[`results/experiments.csv`](https://github.com/huseyin-karaca/mamba-hifigan/blob/main/results/experiments.csv)
has one row per run: the name it was logged under, the class, the family, `d_state`, the segment size, the name
it appears under in the report, and all five final-epoch losses.

[`results/loss_curves.csv`](https://github.com/huseyin-karaca/mamba-hifigan/blob/main/results/loss_curves.csv)
has every logged loss at every epoch for all twenty-six runs — 22 series × 50 epochs × 26 runs, exported from
the TensorBoard event files.

Every table and figure on this site is generated from those two files by
[`tools/doc_tables.py`](https://github.com/huseyin-karaca/mamba-hifigan/blob/main/tools/doc_tables.py), so no
number here is typed by hand.
