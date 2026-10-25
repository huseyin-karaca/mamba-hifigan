# Hyperparameters

One file per training run in the [results table](https://hkaraca.com/mamba-hifigan/experiments/). Each names
the generator class it builds and carries, in a header comment, the losses that run produced.

```bash
cd $SPEECHBRAIN/recipes/LJSpeech/TTS/vocoder/hifigan
python train.py /path/to/mamba-hifigan/hparams/hybrid1_gated_mamba64.yaml \
    --data_folder /path/to/LJSpeech-1.1
```

`hybrid1_gated_mamba64.yaml` is the proposed model.

## These files are reconstructed

Only one `train.yaml` was committed while the project was running — for the `hybrid_1_longer_segments`
configuration. Everything here was rebuilt from that file plus what each run's name records.

What is reliable: the generator class, `d_state`, and the segment size for each run, along with every setting
that was shared across all runs (50 epochs, batch size 16, seed 1234, AdamW with the recipe's schedule).

What is not: any setting that was changed by hand between runs and is not reflected in a run's name. The
`min_duration` filter is the clearest case — it is set only on `baseline_min_duration.yaml`, the run named for
it, and we cannot tell from the surviving evidence which of the later runs also had it on.

`d_state` has the same issue in a milder form. In the original code it was a literal inside each generator
class, edited between runs. Each packaged class now takes it as a constructor argument defaulting to the value
hardcoded in the file as it was last saved. For runs whose name carries the size (`Mamba128`, `Mamba64`) the
value is certain; for the three earliest runs it is inferred as 64 from the block default.

This is the main gap between this repository and an exact reproduction, and it is a consequence of how the
original work was version-controlled rather than anything about the method.
