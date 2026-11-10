# Facts (hero2)
Every label on the figure and its source. Code paths are relative to the repository root.

## Architecture
- "HiFi-GAN generator", upsample factors "↑8" "↑8" "↑2" "↑2": hparams/hybrid1_gated_mamba64.yaml (upsample_factors: [8, 8, 2, 2]).
- "Transposed conv ↑" before both branches: src/mambahifigan/gated.py forward(), o = self.ups[i](o) then both branches read o.
- "BiMamba": src/mambahifigan/blocks.py BiMamba2Block: forward Mamba2 on x, backward Mamba2 on flip(x) then flip, out = fwd + bwd, LayerNorm. "forward scan", "backward scan", "LayerNorm", "+" from this.
- "selective scan", "O(L)": Mamba2 selective state-space scan, linear in sequence length (README, "Mamba is O(L)").
- "h_t" state vector: Mamba2 hidden state, d_state = 64 (Mamba-64 run) or 128.
- "MRF", "k = 3", "k = 7", "k = 11", "dilations 1·3·5": hparams resblock_kernel_sizes [3, 7, 11], resblock_dilation_sizes [[1,3,5]×3], resblock_type "1" (SpeechBrain ResBlock1: per dilation d, conv(k, d) then conv(k, 1), residual).
- Receptive field of one ResBlock1 = 1 + 12(k − 1) frames: 25, 73, 121 for k = 3, 7, 11 (derived from the above). Funnel step widths from the output up: 1, 1+(k−1), 1+6(k−1), 1+7(k−1), 1+10(k−1), 1+11(k−1), 1+12(k−1).
- "mean": gated.py, conv_out = conv_sum / self.num_kernels (average of the three ResBlocks).
- "α_i = σ(w_i)", "α_i", "1 − α_i", "×", "+": gated.py, alpha = sigmoid(mixing_gates[i]); o = alpha*mamba_out + (1−alpha)*conv_out.
- "w_i = 0 at init", "α_i = ½" at init: gated.py, nn.Parameter(torch.zeros(1)); sigmoid(0) = 0.5.
- one scalar per stage (α_1 … α_4 on the chips): gated.py, one mixing_gates entry per upsampling layer.
- "Learnable gate", "Upsampling stage i", "to stage i+1", "global context", "local context": wording of the project page (index.md, method.md).

## Signals
- "mel spectrogram": figure/data/mel.json, computed by figure/data/make_audio_data.mjs from docs/assets/audio/compare/ground_truth.wav (LJSpeech, public domain): 22050 Hz, n_fft 1024, hop 256, 80 Slaney mel bands 0-8 kHz, log magnitude, 41 dB display range; 167 frames, 1.94 s.
- "waveform": figure/data/wave.json, same script, from docs/assets/audio/compare/hybrid_gated.wav (the gated model's output for the same utterance), per-column min/max envelope.
- Stage-input strip x: illustrative feature map (grey cells), not data.

## Context
- "Text", "Acoustic model", "(e.g. FastSpeech 2)": kept from the user's hero2 (generic TTS context; the project trains the vocoder only).

## On-screen strings (hero2 as delivered), each quoted with its source
- "Text", "Acoustic model", "e.g. FastSpeech 2": user's hero2 (context).
- "mel spectrogram": figure/data/mel.json (see Signals). "waveform": figure/data/wave.json.
- "HiFi-GAN generator", "↑8", "↑2": hparams upsample_factors [8, 8, 2, 2].
- "𝛼1", "𝛼2", "𝛼3", "𝛼4": gated.py mixing_gates, one per upsampling layer.
- "Upsampling stage i", "to stage i + 1": the loop over upsampling stages in gated.py forward().
- "Transposedconv ↑" (rendered as "Transposed conv ↑" on two lines): gated.py self.ups[i] (ConvTranspose1d).
- "BiMamba", "forward →", "← backward", "LN": blocks.py BiMamba2Block (forward_mamba, backward_mamba on flip, sum, LayerNorm).
- "global context", "local context": project page wording (index.md, method.md: "global structure" / local texture, bounded window).
- "MRF", "kernel", "𝑘", "11", "7", "3": hparams resblock_kernel_sizes [3, 7, 11]; funnel widths are the derived receptive fields (25, 73, 121 frames).
- "mean": gated.py conv_out = conv_sum / self.num_kernels.
- "𝛼𝑖", "1 − 𝛼𝑖", "𝛼𝑖 = 𝜎(𝑤𝑖)", "𝑤𝑖": gated.py alpha = sigmoid(mixing_gates[i]); o = alpha*mamba_out + (1 − alpha)*conv_out.
- "Learnable gate": method page term; nn.Parameter per stage.
- "1", "½", "0" (σ-curve ticks): range of the sigmoid and its value at 0.
- "𝑤𝑖 = 0", "at init": gated.py nn.Parameter(torch.zeros(1)).
- "SSM", "CNN": the two branches (BiMamba state-space model, MRF convolutions), kept from the user's hero2 slider.

## Revision 3 strings (2026-11-10)
- "Mel spectrogram", "Waveform": card titles, same data as "mel spectrogram" / "waveform" above.
- "𝑥": the transposed-conv output o = self.ups[i](o) in gated.py, the one tensor both branches read (mambas[i](o) and resblocks(o)).
- "init": the σ-curve point at w_i = 0, α_i = ½ (nn.Parameter(torch.zeros(1))).
- Text card glyph: neutral text lines; the clip's transcript is not recorded in the project.

## Revision 4 strings (2026-11-10)
- "from stage i − 1": gated.py forward(): the stage loop carries o from stage i − 1 (its gated output) into o = leaky_relu(o); o = self.ups[i](o). For stage 1 the input is conv_pre(mel).
- "SSM", "CNN", "𝑤𝑖 = 0", "at init": restored (see the revision-1 list); "init" removed.
