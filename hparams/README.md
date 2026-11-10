# Hyperparameters

One file per training run in the [results table](../results/experiments.csv). Each names
the generator class it builds and carries, in a header comment, the losses that run produced.

After `bash scripts/setup.sh <workdir>`, train any of them from the SpeechBrain recipe directory:

```bash
cd <workdir>/speechbrain/recipes/LJSpeech/TTS/vocoder/hifigan
python train.py <repo>/hparams/hybrid1_gated_mamba64.yaml --data_folder <workdir>/LJSpeech-1.1
```

`hybrid1_gated_mamba64.yaml` is the proposed model.
