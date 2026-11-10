# Brief (hero figure, five alternatives)
Takeaway: In a HiFi-GAN generator, a Mamba branch run in parallel with the convolutional branch and mixed by a learned scalar gate beats the other 18 placements tried.
Key ideas: (1) CNN = local texture, bounded window; (2) bidirectional SSM = O(L), whole-utterance context; (3) where Mamba goes is the whole problem (19 placements, 3 failure modes); (4) survivor: parallel, matched resolution, additive, learned α_i = σ(w_i), w_i=0 at init.
Must not imply: any perceptual/listening gain; learned α values (none logged); that all 19 variants failed.
Format: web hero, 183 mm wide, light+dark-safe white bg, Wong palette. blue=CNN, vermillion=Mamba, green=gate/fusion.

---
# Brief: hero2 revision (2026-11-10)
Mode: revise. User request (verbatim): "the current mel image is blurry. find something more professional. the things that represents mamba and MRF is too simple and bad. also please make it more professional looking in general. it is for webpage hero image."
Takeaway (unchanged): inside each HiFi-GAN upsampling stage, a bidirectional Mamba branch (whole-utterance context) runs beside the MRF convolutions (bounded windows), and a learned per-stage scalar α_i = σ(w_i) mixes them.
Key ideas: (1) user's: crisp, real-looking mel; richer, true-to-architecture Mamba and MRF glyphs; professional finish for a web hero. (2) BiMamba = forward + backward selective scan whose state carries the whole sequence, O(L). (3) MRF = 3 parallel ResBlocks (k = 3, 7, 11, dilations 1·3·5), averaged: bounded receptive fields of different widths. (4) y = α_i·BiMamba + (1−α_i)·MRF, one scalar per stage, w_i = 0 at init (even split).
Must not imply: perceptual gains; learned α values; attention-style direct links (an SSM passes information through a recurrent state, the old fan of lines implied attention); receptive fields drawn to the utterance's true scale.
Inherited from the project (revise step 1): the site's palette in docs/stylesheets/extra.css, which says it is "the palette of the site and of every figure": Mamba teal #23977f, conv amber #b67b0c, gate violet #7a5bd1, ink #1c2230. hero2's Wong colours contradicted the live hero and fig_gated, so this revision adopts the site key.
Format: web hero, right column of a 5fr:6fr grid (about 620-730 px wide on desktop, full width on mobile). 160 mm wide so 7-9 pt labels read at about 10-13 px. White card background.
