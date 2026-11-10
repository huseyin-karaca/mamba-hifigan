# Spec (draft depth)
Hero: none, architecture/concept diagram (route: flat). Five concepts, one file each: figure_a..e.
A stage zoom | B four-stage pipeline | C receptive-field story | D constraint funnel (19 placements) | E crossfader / mixing console.
Negative: no invented numbers or learned alphas; no listening-quality claims.

---
# Spec: hero2 revision
Structure: setup with its signal (top strip: mel → generator → waveform) plus a zoom from system to part (one stage), whose inside is a controlled comparison (global vs local) on one shared time axis (P3, P24).
Hero: none, architecture/mechanism diagram (route: flat).

## Concepts (for the two lanes, the rest of the figure is shared)
A. Shared time axis, mirrored. Stage input x is one feature-map strip across the middle. Above it, BiMamba: a forward and a backward selective-scan state ribbon with state vectors h_t, which join at one output frame t through + and LayerNorm. Below it, MRF: three nested stepped receptive-field funnels (k = 3, 7, 11; six conv layers each with dilations 1,1,3,1,5,1) converge on the same frame t and average. Same x for t in both. Scores: 5-s takeaway 5 (teal spans the whole axis, amber is bounded); true to code 5 (recurrence, not attention; exact RF widths 25:73:121); state drawn (P1) 5; comparison (P3) 5; explains 5.
B. Block internals. BiMamba as two Mamba2 pipelines (in-proj → conv1d k=4 → SiLU → SSM ⊗ SiLU gate → out-proj), flip, sum, LN; MRF as three ResBlock columns of dilated conv pills, then mean. Scores: takeaway 2 (boxes do not show why one is global); true 5; state 1; comparison 2; explains 3, and it repeats fig_gated, which already sits further down the page.
C. Keep the current separate lanes, upgrade the glyphs only. Scores: takeaway 4; true 3 (fan implies attention); comparison 3 (two axes).
Kept: A (B and C are drafted only as alternatives to A's lanes in choice.md).

## Components
Top strip: Text box, Acoustic model box (grey, context) → real mel (80 × 167 frames, crisp vector cells) → HiFi-GAN generator card with four stage chips ↑8 ↑8 ↑2 ↑2, each with its own α_i tick (violet) → real waveform envelope (gated model output, same utterance).
Zoom: wedge from the second chip to the stage card (grey/violet hairline, dashed).
Stage card "Upsampling stage i": ConvT ↑ box → feature strip x (grey channel cells, fading ellipsis at both ends = sequence continues). BiMamba lane (teal tint) above, MRF lane (amber tint) below. Right: ⊗ α_i on the teal output, ⊗ (1 − α_i) on the amber output, ⊕ → "to stage i+1". Gate card (violet): σ curve with the init point w_i = 0 → α_i = 0.5, mix bar SSM | CNN at 50/50, "α_i = σ(w_i)".
Relations: mel → generator → waveform; ConvT → x; x → both lanes; lanes → ⊗ → ⊕ → next stage; gate → both ⊗.
Negative: no fan of direct links; no α values other than the init 0.5; no to-scale claim (caption says schematic); no listening-quality words.
Sources: mel and wave = fetch+compute (figure/data/make_audio_data.mjs); state ribbons, funnels = draw (custom parts in figure/hero2_parts.js, library-gaps.md); boxes, ops = D.box, D.opNode, D.route.
