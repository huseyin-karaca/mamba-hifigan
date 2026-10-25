"""Bidirectional Mamba in the HiFi-GAN generator.

Every generator in this package is a drop-in replacement for SpeechBrain's `HifiganGenerator`: same constructor
arguments, same `forward`, same `remove_weight_norm`. They differ only in where a Bidirectional Mamba block sits
relative to the Multi-Receptive Field (MRF) module, and how the two branches are combined.

The one the study settled on is `HifiganGeneratorHybrid1GatedFusion` (`gated.py`): MRF and Mamba run in parallel at
every upsampling stage and are mixed by a learnable per-stage scalar.

The modules are grouped the way the study grouped them -- by what goes wrong, not by chronology:

    blocks       the Mamba building blocks, BiMamba2Block above all
    baseline     unmodified HiFi-GAN
    gated        parallel branches, learnable mixing weight  <- the proposal
    parallel     parallel branches, fixed mixing weight; and the Mamba-only ablation
    phase        integrations that corrupt phase coherence
    gradient     integrations that destabilise gradients
    bottleneck   integrations that compress or interpolate away information
    partial      Mamba at some resolutions only

`EXPERIMENTS` lists every run that was logged, with the class it used and its final-epoch losses; the same table is
in `results/experiments.csv`, which is what the documentation site is built from.
"""

from .baseline import OriginalHifiganGenerator
from .blocks import (
    BiMamba2Block,
    LowResMambaBlock,
    MambaModulator,
    MambaRefiner,
    MambaSEBlock,
    MambaSideNetwork,
    SafeMambaBlock,
    SpectrogramRefiner,
)
from .bottleneck import (
    HifiganGeneratorBroadcast,
    HifiganGeneratorConcat,
    HifiganGeneratorSpecRefine,
    HifiganGeneratorWithPreNet,
)
from .gated import (
    HifiganGeneratorHybrid1GatedFusion,
    HifiganGeneratorMambaMRF_Gated,
)
from .gradient import (
    HifiganGeneratorMambaFiLM,
    HifiganGeneratorMambaSE,
    HifiganGeneratorSequentialProgressive,
)
from .parallel import (
    HifiganGeneratorHybrid1Summation,
    HybridHifiganGenerator2,
    MambaOnlyHifiganGenerator,
)
from .partial import (
    HifiganGeneratorBottleneck,
    HifiganGeneratorLateMamba,
    HifiganGeneratorSerial,
)
from .phase import (
    HifiganGeneratorDualPath,
    HifiganGeneratorMultiRate,
    HifiganGeneratorWithPostNet,
)

__version__ = "1.0.0"

#: Every generator by the name the hparams files use.
GENERATORS = {
    "original": OriginalHifiganGenerator,
    "hybrid1_gated": HifiganGeneratorHybrid1GatedFusion,
    "mrf_gated": HifiganGeneratorMambaMRF_Gated,
    "hybrid1_summation": HifiganGeneratorHybrid1Summation,
    "hybrid2_summation": HybridHifiganGenerator2,
    "mamba_only": MambaOnlyHifiganGenerator,
    "postnet": HifiganGeneratorWithPostNet,
    "dual_path": HifiganGeneratorDualPath,
    "multirate": HifiganGeneratorMultiRate,
    "film": HifiganGeneratorMambaFiLM,
    "mamba_se": HifiganGeneratorMambaSE,
    "sequential_progressive": HifiganGeneratorSequentialProgressive,
    "concat": HifiganGeneratorConcat,
    "broadcast": HifiganGeneratorBroadcast,
    "spec_refine": HifiganGeneratorSpecRefine,
    "prenet": HifiganGeneratorWithPreNet,
    "bottleneck": HifiganGeneratorBottleneck,
    "late_mamba": HifiganGeneratorLateMamba,
    "serial": HifiganGeneratorSerial,
}

#: Which failure family each generator belongs to, as the study grouped them.
FAMILIES = {
    "baseline": "unmodified HiFi-GAN",
    "gated": "parallel branches mixed by a learnable per-stage scalar",
    "parallel": "parallel branches mixed by a fixed weight",
    "ablation": "the SSM branch alone, no convolutional branch",
    "phase": "phase coherence breaks",
    "gradient": "gradients destabilise",
    "bottleneck": "information is compressed or interpolated away",
    "partial": "the SSM branch at some resolutions only",
}


def load_experiments():
    """Read `results/experiments.csv` -- every logged run with its class and final-epoch losses.

    Returns a list of dicts. Raises FileNotFoundError outside a source checkout, since the CSV is repository data
    rather than package data.
    """
    import csv
    from pathlib import Path

    path = Path(__file__).resolve().parents[2] / "results" / "experiments.csv"
    with open(path) as f:
        return list(csv.DictReader(f))
