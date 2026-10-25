# Results

## Setup

LJSpeech, 13,100 clips of a single English speaker at 22.05 kHz, the standard vocoder benchmark. Every
configuration used SpeechBrain's LJSpeech HiFi-GAN recipe with the same optimiser, schedule and discriminators;
only the generator differed.

| Setting | Value |
|---|---|
| Dataset | LJSpeech 1.1, 22.05 kHz, single speaker |
| Epochs | 50 |
| Seed | 1234, one run per configuration |
| Batch size | 16 |
| Segment size | 12,288 samples (8,192 for the five earliest runs) |
| Optimiser | AdamW, $\beta = (0.8, 0.99)$, exponential decay |
| Hardware | one rented GPU per run (vast.ai) |

Two quantities are reported, both taken at the final epoch on the validation set:

**Feature matching loss** — the $L_1$ distance between discriminator intermediate activations for real and
generated audio. It is part of the generator's objective. It is often described as tracking perceptual quality,
and it does correlate with it, but it is a statistic of *this* discriminator at *this* point in training, not a
perceptual measurement.

**L1 spectrogram loss** — the $L_1$ distance between mel-spectrograms of real and generated audio. Also part of
the objective. It measures spectral fidelity and says nothing about phase.

## Headline comparison

--8<-- "snippets/table_main.html"

Three things in that table are worth separating.

**Mamba alone is clearly worse.** 0.1291 against the baseline's 0.0736 is a 75% increase in feature-matching
loss, far outside anything seed noise would produce. Removing the convolutional branch removes the inductive
bias that makes the texture of speech cheap to model, and the SSM does not recover it. This is the most solid
finding here.

**Fixed-weight fusion barely moves.** Hybrid 2's 0.0711 against 0.0736 is a 3% difference on a single run. We
would not claim that as an improvement.

**Gating is the best configuration measured, by a margin we cannot certify.** 0.0623 is 15% below the baseline,
and 0.2587 is 13% below on the spectrogram loss. On one run each, at 50 epochs, with no seed variance to compare
against, this is the strongest result in the study and also not evidence that the gate reliably helps. It is a
reason to run it properly, not a conclusion.

## Training curves

--8<-- "snippets/fig_curves.html"

The curves say more than the final numbers. The gated models and the baseline sit on top of one another for
most of training and only separate towards the end, which is exactly the pattern you would expect either from a
small real effect or from run-to-run variation. The Mamba-only run is somewhere else entirely from epoch 10 on,
and its spikiness is the discriminator winning: its adversarial loss climbs to 6.78 while the discriminator
loss falls to 1.16, against roughly 4.7 and 2.7 for the baseline.

## The two metrics disagree

Mamba-64 gives the best feature-matching loss; Mamba-128 gives the best L1 spectrogram loss. Neither wins both.

The broadcasting run makes the same point more sharply: feature matching 0.0699, better than the baseline, with
an L1 spectrogram loss of 0.4479 — 50% worse. A model can satisfy the discriminator while reconstructing the
spectral envelope badly. Reporting either number alone would give a misleading picture, which is why both are
in every table on this site, and why the [audio](listen.md) is published alongside them.

## Every run

All twenty-six runs, grouped by failure family and with the generator class each one used, are on the
[every run](experiments.md) page.

## What would make this conclusive

Nothing here is a finished experiment, and the gaps are the ordinary ones:

- **Multiple seeds.** One run per configuration cannot distinguish a 15% effect from variance. Three to five
  seeds on the top configurations and the baseline would settle it.
- **A perceptual metric.** UTMOS or PESQ costs almost nothing to run and measures something closer to what the
  work claims. A MOS listening test would be better.
- **Longer training.** 50 epochs is short for a vocoder; the reference HiFi-GAN recipes run far longer, and the
  curves had not flattened.
- **The learned gates.** Reading $\alpha_i$ per stage is the direct test of the mechanism, and it was never done.

The checkpoints did not survive the rented instances, so all of this means retraining rather than re-analysis.
