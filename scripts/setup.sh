#!/usr/bin/env bash
# Set up a machine to retrain any configuration in hparams/.
#
#   bash scripts/setup.sh /path/to/workdir
#
# Fetches SpeechBrain at the commit the project was built on, applies the recipe patch, installs this package
# and downloads LJSpeech. Building mamba-ssm needs a CUDA toolchain and takes 30-40 minutes; that is normal.
set -euo pipefail

WORKDIR="${1:?usage: setup.sh /path/to/workdir}"
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PIN=637f0a55   # the SpeechBrain commit patches/ applies to

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [ ! -d speechbrain ]; then
  echo "==> cloning SpeechBrain at $PIN"
  git clone https://github.com/speechbrain/speechbrain.git
  git -C speechbrain checkout "$PIN"
else
  echo "==> speechbrain/ already present, leaving it alone"
fi

echo "==> applying the recipe patch (min_duration filtering, W&B logging)"
# Only train.py is needed: the generators live in this package and import ResBlock1/ResBlock2 from
# SpeechBrain, so its model file stays untouched.
if git -C speechbrain apply --check --reverse \
      --include='recipes/LJSpeech/TTS/vocoder/hifigan/train.py' \
      "$REPO/patches/speechbrain-637f0a55.patch" 2>/dev/null; then
  echo "    already applied"
else
  git -C speechbrain apply \
      --include='recipes/LJSpeech/TTS/vocoder/hifigan/train.py' \
      "$REPO/patches/speechbrain-637f0a55.patch"
fi

echo "==> installing SpeechBrain and this package"
pip install -e ./speechbrain
pip install -r ./speechbrain/recipes/LJSpeech/TTS/extra_requirements.txt
pip install -e "$REPO"

echo "==> installing mamba-ssm (slow: needs nvcc, 30-40 min is normal)"
pip install mamba-ssm "causal-conv1d>=1.2.0"

if [ ! -d LJSpeech-1.1 ]; then
  echo "==> downloading LJSpeech (2.6 GB)"
  wget -q --show-progress https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2
  tar -xjf LJSpeech-1.1.tar.bz2
  rm LJSpeech-1.1.tar.bz2
fi

cat <<EOF

Done. To train the proposed model:

  cd $WORKDIR/speechbrain/recipes/LJSpeech/TTS/vocoder/hifigan
  python train.py $REPO/hparams/hybrid1_gated_mamba64.yaml --data_folder $WORKDIR/LJSpeech-1.1

Any other file in $REPO/hparams/ works the same way.
EOF
