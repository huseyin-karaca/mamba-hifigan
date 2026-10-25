# Reproduce

## What survived and what did not

Worth stating before you start. The code, the hyperparameters, every logged loss at every epoch, and a sample
of the generated audio are all here. **The trained checkpoints are not** — the runs were done on rented vast.ai
instances that were shut down when the course ended, and the weights went with them. So this reproduces by
retraining, not by loading a model.

A single 50-epoch run took a few hours on one rented GPU. There are twenty-six of them.

## Install

The generators are an ordinary Python package that imports `ResBlock1` and `ResBlock2` from SpeechBrain and
defines everything else itself. Nothing in SpeechBrain's model code has to be patched.

```bash
git clone https://github.com/huseyin-karaca/mamba-hifigan
cd mamba-hifigan
pip install -e .

# Mamba needs a CUDA toolchain; this is the slow step, 30-40 minutes is normal
pip install mamba-ssm "causal-conv1d>=1.2.0"
```

`scripts/setup.sh` does the above plus fetching SpeechBrain at the pinned commit and downloading LJSpeech:

```bash
bash scripts/setup.sh /path/to/workdir
```

## Train one configuration

```bash
cd $SPEECHBRAIN/recipes/LJSpeech/TTS/vocoder/hifigan
python train.py /path/to/mamba-hifigan/hparams/hybrid1_gated_mamba64.yaml \
    --data_folder /path/to/LJSpeech-1.1
```

There is one file in [`hparams/`](https://github.com/huseyin-karaca/mamba-hifigan/tree/main/hparams) per run in
the [results table](experiments.md), each naming its generator class and carrying the losses that run produced
in a header comment. `hybrid1_gated_mamba64.yaml` is the proposed model.

The recipe's `train.py` needs two small changes — a `min_duration` filter on the training set and optional
Weights & Biases logging. Apply them with:

```bash
cd $SPEECHBRAIN
patch -p1 < /path/to/mamba-hifigan/patches/speechbrain-637f0a55.patch
```

!!! warning "The hparams files are reconstructed"
    Only one `train.yaml` was committed during the project, for one configuration. The per-run files here were
    rebuilt from that file plus the run names, so the generator, `d_state` and segment size per run are
    reliable, but a setting that was changed by hand and not reflected in a run's name would not have survived.
    This is the main thing standing between this repository and an exact reproduction, and it is a consequence
    of how the original work was version-controlled.

## What the patch contains

`patches/speechbrain-637f0a55.patch` is the literal diff between upstream SpeechBrain at commit `637f0a55` and
the code as it was run. It applies cleanly to that commit and touches two files:

| File | Change |
|---|---|
| `speechbrain/lobes/models/HifiGAN.py` | the Mamba blocks and all nineteen generators, +3,533 lines |
| `recipes/LJSpeech/TTS/vocoder/hifigan/train.py` | `min_duration` filtering and W&B logging, +65 lines |

That diff is the project's entire contribution to SpeechBrain, and it is the honest answer to "what did you
actually write". The same generator code, reorganised into readable modules and with `d_state` exposed as an
argument, is in [`src/mambahifigan/`](https://github.com/huseyin-karaca/mamba-hifigan/tree/main/src/mambahifigan);
the unmodified-but-for-our-additions file is kept at `overlay/speechbrain/lobes/models/HifiGAN.py` as the
archival record of what the runs actually executed.

## Rebuild the site

```bash
pip install "mkdocs-material>=9.5"
python tools/doc_tables.py   # regenerate every table and figure from results/
mkdocs serve
```

`tools/doc_tables.py` reads `results/experiments.csv` and `results/loss_curves.csv` and writes the snippets
under `docs/snippets/`. Every number on this site comes from there, so editing a CSV and re-running is enough
to update the pages.

## Layout

```
src/mambahifigan/   the contribution as importable modules, grouped by failure family
overlay/            the same code at its SpeechBrain paths, as it was run
patches/            the diff against SpeechBrain 637f0a55
hparams/            one YAML per run
results/            per-run final losses, and every loss at every epoch
docs/               this site, including the audio
tools/              generates the site's tables and figures from results/
```
