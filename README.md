# Mamba HiFi-GAN

**Adaptive gated fusion of convolutional and bidirectional-SSM branches in a neural vocoder.**

Hüseyin Karaca, Rabia Ela Ünlü, Emre Can Yayla — Bilkent University, CS 585 term project, Fall 2025

[![Project page](https://img.shields.io/badge/project%20page-hkaraca.com%2Fmamba--hifigan-blue)](https://hkaraca.com/mamba-hifigan/)

**Project page: [hkaraca.com/mamba-hifigan](https://hkaraca.com/mamba-hifigan/)** — the method, the results, the
audio, and every run we logged.

HiFi-GAN's convolutions model the texture of speech well but see only a bounded window, so prosody and pitch
drift across an utterance fall outside what the generator can condition on. A selective state-space model is
$O(L)$ with unbounded context, which makes it the obvious thing to add — but *where* to add it turns out to be
the whole problem.

We built **nineteen** placements of a bidirectional Mamba block in the generator and trained **twenty-six**
runs. Most were worse than leaving HiFi-GAN alone, and they failed in three distinct ways: phase coherence
breaks, gradients destabilise, or information is squeezed through a bottleneck. What survives all three
constraints is a parallel branch at matched resolution, combined additively — with a mixing weight the network
learns for itself:

$$\alpha_i = \sigma(w_i), \qquad y^{(i)} = \alpha_i \cdot \Phi_{\text{Mamba}}(x) + (1 - \alpha_i) \cdot \Psi_{\text{CNN}}(x)$$

One learnable scalar per upsampling stage, initialised to 0 so training starts at an even split.

## Results

| Configuration | Feature matching ↓ | L1 spectrogram ↓ |
|---|---|---|
| Baseline HiFi-GAN | 0.0736 | 0.2975 |
| Mamba only, no MRF | 0.1291 | 0.3307 |
| Hybrid 2, summation fusion | 0.0711 | 0.2991 |
| **Hybrid 1, gated, Mamba-64** | **0.0623** | 0.2806 |
| **Hybrid 1, gated, Mamba-128** | 0.0771 | **0.2587** |

> **One run per configuration, 50 epochs, seed 1234.** Both columns are training objectives — feature matching
> is a statistic of the discriminator, not a perceptual score. There is no listening test, no PESQ or UTMOS,
> and no seed variance to compare against. Read the large gaps (Mamba-only against everything else) as real;
> read the small ones as suggestive and unconfirmed. This was a one-semester course project, and it is
> published as one.

[Listen to the audio](https://hkaraca.com/mamba-hifigan/listen/) — the same utterance from every model, and
each model learning to speak over 50 epochs.

## What is where

```
src/mambahifigan/   the contribution as importable modules, grouped by how each design fails:
                    blocks (BiMamba2Block), gated (the proposal), parallel, phase,
                    gradient, bottleneck, partial, baseline
overlay/            the same code at its SpeechBrain paths, exactly as it was run
patches/            speechbrain-637f0a55.patch -- the literal diff against upstream SpeechBrain,
                    which is the honest answer to "what did you write"
hparams/            one YAML per run, each naming its generator and its final losses
results/            experiments.csv (per-run final losses) and loss_curves.csv (every epoch)
docs/               the project page, including the audio and the PDFs
tools/              generates every table and figure on the site from results/
```

## Reproduce

The generators are a normal Python package; they import `ResBlock1` and `ResBlock2` from SpeechBrain and define
everything else. SpeechBrain's model code needs no patch.

```bash
pip install -e .
pip install mamba-ssm "causal-conv1d>=1.2.0"      # needs a CUDA toolchain, 30-40 min

cd $SPEECHBRAIN/recipes/LJSpeech/TTS/vocoder/hifigan
patch -p1 -d $SPEECHBRAIN < patches/speechbrain-637f0a55.patch   # min_duration + W&B logging
python train.py /path/to/hparams/hybrid1_gated_mamba64.yaml --data_folder /path/to/LJSpeech-1.1
```

**The trained checkpoints did not survive** — the runs were on rented vast.ai instances that were shut down at
the end of the course. Reproducing means retraining. Full instructions, and the caveats about the
reconstructed hparams, are on the [reproduce page](https://hkaraca.com/mamba-hifigan/reproduce/).

## Documents

- [Final report](https://hkaraca.com/mamba-hifigan/assets/report.pdf) (PDF, 10 pages)
- [Final presentation](https://hkaraca.com/mamba-hifigan/assets/slides.pdf) (PDF)

## Provenance

The original work was done in November and December 2025 in a full copy of the SpeechBrain tree at
[`huseyin-karaca/speechbrain`](https://github.com/huseyin-karaca/speechbrain), in three large commits. This
repository is a reorganisation of that work: same code, restructured so the contribution is readable, with the
diff against upstream kept in `patches/`. The commit dates here are the dates of the reorganisation, not of the
research.

## Citation

```bibtex
@misc{karaca2025mambahifigan,
  title  = {Integrating Bidirectional Mamba into {HiFi-GAN} for Long-Range Speech Synthesis},
  author = {Karaca, H{\"u}seyin and {\"U}nl{\"u}, Rabia Ela and Yayla, Emre Can},
  year   = {2025},
  note   = {CS 585 term project, Bilkent University},
  url    = {https://hkaraca.com/mamba-hifigan/}
}
```

See also [`CITATION.cff`](CITATION.cff).

## Licence

Code is Apache-2.0 ([`LICENSE`](LICENSE)), matching SpeechBrain, from which the generators are derived — see
[`NOTICE`](NOTICE). The text and figures of the project page are CC BY 4.0. The audio is derived from
[LJSpeech](https://keithito.com/LJ-Speech-Dataset/), which is public domain.
