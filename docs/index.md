---
template: home.html
title: Home
hide:
  - navigation
  - toc
---

## Convolutions hear a window, not a sentence

HiFi-GAN turns a mel-spectrogram into a waveform with stacks of dilated convolutions. They are very good at what
speech mostly is up close — periodic texture, timbre, the shape of a single glottal pulse — and they are fast,
because every sample is produced in parallel.

What they cannot do is see far. A convolutional stack has a bounded receptive field, so the sample at time $t$
depends on a fixed window around $t$. Prosodic contour, pitch declination across a sentence, the stability of a
voice over several seconds: all of it lies outside the window. Self-attention would see the whole utterance, but
at $O(L^2)$ on sequences of tens of thousands of audio samples, that is not affordable.

A selective state-space model is the obvious thing to reach for. Mamba is $O(L)$ and its context is, in
principle, unbounded. The question this project asked is not whether to use one, but **where to put it**.

## Nineteen placements, and most of them hurt

--8<-- "snippets/home_stats.html"

Putting an SSM "into HiFi-GAN" is not one design, it is dozens. Before the generator or after it. Replacing the
convolutions or beside them. Multiplying their output or adding to it. At every resolution or only at some. We
built nineteen of these and trained twenty-six runs, and the honest headline is that **fourteen of the nineteen
were worse than leaving HiFi-GAN alone.**

That turned out to be the interesting part. The failures are not random; they fall into three groups, and each
group fails for one identifiable reason.

<ul class="mh-points">
<li><b>Phase breaks</b><span>Anything that edits the waveform after it is generated, or mixes streams built at different sample rates, destroys phase alignment. The audio goes muffled or metallic even as some losses improve.</span></li>
<li><b>Gradients destabilise</b><span>Multiplicative coupling — FiLM, squeeze-excite — and long sequential paths make GAN training unstable or starve the SSM parameters of gradient.</span></li>
<li><b>Information is squeezed</b><span>Concatenate-then-project, or broadcast one coarse context upward, and the structure you wanted to add is compressed or smeared away before it reaches the output.</span></li>
</ul>

What survives all three constraints is narrow: the SSM has to run **beside** the convolutions, at the **same
resolution**, and be combined **additively**. That is the design below.

--8<-- "snippets/fig_gated.html"

A single scalar per upsampling stage, pushed through a sigmoid, decides how much of each branch goes through. It
starts at zero — an even split — and the network moves it during training.

## What it measured

--8<-- "snippets/table_main.html"

The gated model has the lowest feature-matching loss in the study and, at the larger state size, the lowest L1
spectrogram loss. The Mamba-only ablation is far worse than either, which is the clearest result here: an SSM
does not replace the convolutional inductive bias, it supplements it.

<div class="mh-protocol" markdown="0">
<p><b>How much weight these numbers carry.</b> Each configuration was trained <b>once</b>, for <b>50 epochs</b>,
at seed 1234, on a single GPU. Both reported quantities are <b>training objectives</b> — feature matching is a
statistic of the discriminator, not a measure of how speech sounds. There is no listening test and no
PESQ or UTMOS, and with one run per configuration we cannot separate a real gap from seed variation.</p>
<p>So: read the large differences, such as Mamba-only against everything else, as real. Read the small ones,
such as 0.0623 against 0.0736, as suggestive and unconfirmed. The <a href="results/">results page</a> says
which is which, and the <a href="listen/">audio</a> is there so you can judge for yourself.</p>
</div>

<div class="mh-cards">
<a class="mh-card" href="method/"><b>Method</b><span>The bidirectional block, the gate, and why the three failure families rule out everything else.</span></a>
<a class="mh-card" href="results/"><b>Results</b><span>The headline table, the training curves, and what the numbers do and do not support.</span></a>
<a class="mh-card" href="listen/"><b>Listen</b><span>The same utterance from every model, and each model learning to speak over 50 epochs.</span></a>
<a class="mh-card" href="reproduce/"><b>Reproduce</b><span>One hparams file per run, the patch against SpeechBrain, and what it cost to train.</span></a>
</div>

## Citation

This is coursework, not a paper. If you want to point at it:

```bibtex
@misc{karaca2025mambahifigan,
  title  = {Integrating Bidirectional Mamba into {HiFi-GAN} for Long-Range Speech Synthesis},
  author = {Karaca, H{\"u}seyin and {\"U}nl{\"u}, Rabia Ela and Yayla, Emre Can},
  year   = {2025},
  note   = {CS 585 term project, Bilkent University},
  url    = {https://hkaraca.com/mamba-hifigan/}
}
```
