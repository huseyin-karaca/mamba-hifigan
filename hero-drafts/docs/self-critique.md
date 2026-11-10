# Self-critique (hero2 revision, critic pass)
Faithfulness (P28), checked against src/mambahifigan and hparams:
- BiMamba = forward Mamba2 + backward Mamba2 on flip, summed, LayerNorm: drawn as two opposite lanes meeting in + then LN. OK.
- Both scans read every frame: one tap per frame into the band. OK.
- No direct input-to-output links (the old fan implied attention): removed. OK.
- MRF = 3 ResBlock1 with k = 3, 7, 11, each 6 convs with dilations 1,1,3,1,5,1, averaged: 3 nested funnels, 6 layer bands each, widths 25:73:121 at the input, exact per level; "mean". OK.
- Fusion y = α_i·BiMamba + (1 − α_i)·MRF: × nodes on each branch, labels α_i and 1 − α_i, + to the next stage. OK (the old figure had no × nodes).
- α_i = σ(w_i), w_i = 0 at init → ½: σ curve with the point at (0, ½), mix bar at 50/50. OK. No learned α values shown. OK.
- One gate per stage: α_1..α_4 on the four stage chips. OK.
- Upsample factors 8, 8, 2, 2. OK.

Rubric (1-5), after fixes:
1 Reads at final size: 4. At 640 px the smallest labels are about 9 px; the hero column is 620-860 px wide.
2 Takeaway: 5. Whole-axis teal stream against bounded amber funnels, gate between them.
3 Faithful: 5.
4 Correct: 4. Funnel scale is relative only (121 frames ≪ an utterance at this stage); said in the caption.
5 Links: 5. Zoom flare from the ↑8 chip; teal/amber/violet carried into labels, routes, mix bar and chip αs.
6 Economy: 5. 49 words; the equation line, "one scalar per stage" and the caption-like "one output frame sees…" sentences are gone.
7 Craft: 4. Fixed: funnel tops poking out of the MRF lane, "to stage i + 1" colliding with the amber route, crowded "k = 11", lane separator ending abruptly, 0.15 pt hairline in the waveform. Left: the empty lower left of the MRF lane (kept as air around the hero).

Gate WARNs kept on purpose: width 160 mm (a web hero, not a journal column); two font families (site sans for labels, Latin Modern for maths, the CS convention); text above 8 pt (lane names and the gate equation, for hierarchy at web size); "text over busy art" (LN and mean sit in their own pills; the kernel numbers sit on flat funnel fills).

## Revision 3 (user feedback: "it reads like both MRF and bimamba built upon transposed conv line. is it true?")
Partly true. Both branches read the same tensor, the transposed-conv output (gated.py: o = self.ups[i](o); mamba_out = self.mambas[i](o); conv_out = mean of resblocks(o)). They share nothing else. The shared strip between the lanes, the ConvT box straddling both lane borders and the mirrored lanes made the conv output look like a common trunk the branches were built on. Fix: ConvT box → x → a fork with a junction dot → one copy of x at the top of each of two separate cards with a gap between them; both cards read top to bottom on the same template; row headers outside the cards.
Top strip: five cards of one height (56 pt) with titles in one style and on one baseline, content centred on one line, arrows on that line.
Gate card: the mix bar and the "w_i = 0 at init" line were dropped for room; the σ-curve point is labelled "init" at w_i = 0, α_i = ½.

## Revision 4 (user: "that transposed conv is nothing but alpha 1 (from previous stage) right? ... make the learnable gate block vertically centered")
Not quite: the transposed conv is stage i's own upsampling layer (self.ups[i], the ↑8 of the chip); its input is the previous stage's gated output. Drawn as "from stage i − 1" → Transposed conv ↑ → x, mirroring "to stage i + 1".
Gate: with both branches reading top to bottom, both outputs sat at their cards' bottoms, so the gate (centred between the × nodes) sat beside MRF. Fix: equal-height cards mirrored about the fork (BiMamba reads upward, MRF downward), so the outputs are at the outer edges and the gate, the + node and the fork share the stage's centre line. The two cards, the gap and the fork stay, so the branches still read as separate modules; the room regained brings back the SSM/CNN mix bar and "w_i = 0 at init".
