# Listen

The losses on the [results page](results.md) are training objectives, not measures of how speech sounds. This
page is here so you can form your own judgement.

All clips are LJSpeech validation audio, 22.05 kHz, reconstructed from the mel-spectrogram by each generator at
the end of its 50 epochs. LJSpeech is public domain.

## The same utterance, five ways

<div class="mh-audio" markdown="0">
  <div class="mh-audio__name">Ground truth<small>the recording itself</small></div>
  <audio controls preload="none" src="../assets/audio/compare/ground_truth.wav"></audio>
  <div class="mh-audio__score">&mdash;</div>

  <div class="mh-audio__name">Baseline HiFi-GAN<small>OriginalHifiganGenerator</small></div>
  <audio controls preload="none" src="../assets/audio/compare/baseline.wav"></audio>
  <div class="mh-audio__score">FM 0.0736</div>

  <div class="mh-audio__name">Hybrid 2, summation<small>fixed-weight fusion</small></div>
  <audio controls preload="none" src="../assets/audio/compare/hybrid_summation.wav"></audio>
  <div class="mh-audio__score">FM 0.0711</div>

  <div class="mh-audio__name">Hybrid 1, gated<small>the proposed model</small></div>
  <audio controls preload="none" src="../assets/audio/compare/hybrid_gated.wav"></audio>
  <div class="mh-audio__score">FM 0.0623</div>

  <div class="mh-audio__name">Mamba only<small>no convolutional branch</small></div>
  <audio controls preload="none" src="../assets/audio/compare/mamba_only.wav"></audio>
  <div class="mh-audio__score">FM 0.1291</div>
</div>

!!! tip "What to listen for"
    The Mamba-only clip is audibly degraded and needs no careful listening — it is rough and unstable, which is
    what losing the convolutional branch costs. Between the other three the differences are subtle and sit
    mostly in how steady the pitch is across the utterance. We could not reliably tell the baseline from the
    gated model by ear, and we would not expect you to either. That is worth saying plainly: a 15% difference
    in feature-matching loss is not a 15% difference in how it sounds.

## Learning to speak, over 50 epochs

Every model's validation output, at every fifth epoch of training. Drag the slider; audio that is playing keeps
playing as the epoch changes, so you can hear a model converge.

<div class="mh-epochs" data-epochs="01,05,10,15,20,25,30,35,40,45,50" markdown="0">
  <div class="mh-epochs__control">
    <label for="mh-epoch-range">Training epoch</label>
    <input type="range" id="mh-epoch-range" value="0">
    <span class="mh-epochs__value">epoch 1</span>
  </div>
  <div class="mh-audio">
    <div class="mh-audio__name">Target<small>the reference for this clip</small></div>
    <audio controls preload="none" src="../assets/audio/epochs/target.wav"></audio>
    <div class="mh-audio__score">&mdash;</div>

    <div class="mh-audio__name">Baseline HiFi-GAN<small>unmodified</small></div>
    <audio controls preload="none" data-template="../assets/audio/epochs/baseline/epoch{epoch}.wav"></audio>
    <div class="mh-audio__score">&nbsp;</div>

    <div class="mh-audio__name">Hybrid 1, gated<small>the proposed model</small></div>
    <audio controls preload="none" data-template="../assets/audio/epochs/gated/epoch{epoch}.wav"></audio>
    <div class="mh-audio__score">&nbsp;</div>

    <div class="mh-audio__name">Hybrid 1, summation<small>fixed-weight fusion</small></div>
    <audio controls preload="none" data-template="../assets/audio/epochs/summation/epoch{epoch}.wav"></audio>
    <div class="mh-audio__score">&nbsp;</div>

    <div class="mh-audio__name">Mamba only<small>no convolutional branch</small></div>
    <audio controls preload="none" data-template="../assets/audio/epochs/mamba_only/epoch{epoch}.wav"></audio>
    <div class="mh-audio__score">&nbsp;</div>
  </div>
</div>

At epoch 1 every model produces noise with the rough envelope of speech. Intelligibility arrives somewhere
around epoch 5 to 10 for the three models that keep their convolutions. The Mamba-only model stays rough
throughout, and the gap you hear at epoch 50 is the same gap its loss curve shows from epoch 10 onward.

The clips come from the TensorBoard validation audio written during training, so each one is whichever
validation utterance that run logged at that epoch. The baseline, summation and Mamba-only panels follow one
utterance and the gated panel another, which is why the words differ between them.

## Downloads

Every clip on this page is in the repository under
[`docs/assets/audio/`](https://github.com/huseyin-karaca/mamba-hifigan/tree/main/docs/assets/audio). The full
per-epoch set, for all twenty-six runs rather than the four here, was logged to TensorBoard and is not
published; the final-epoch losses for all of them are in
[`results/experiments.csv`](https://github.com/huseyin-karaca/mamba-hifigan/blob/main/results/experiments.csv).
