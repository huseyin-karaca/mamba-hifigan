# Method

## The baseline, and the one thing wrong with it

HiFi-GAN maps a mel-spectrogram $M \in \mathbb{R}^{F \times L}$ to a waveform $\hat{S} \in \mathbb{R}^{T}$, where
$T \approx 256 L$. It is a GAN: a fully convolutional generator, and two discriminators — a multi-period
discriminator that reshapes the 1-D signal into 2-D planes at prime periods (2, 3, 5, 7, 11) to catch
periodicity, and a multi-scale discriminator that looks at raw, $\times 2$ pooled and $\times 4$ pooled audio.

The generator upsamples with transposed convolutions. After each upsampling it applies the **multi-receptive
field** module: several residual blocks in parallel, with different kernel sizes and dilation rates, summed and
averaged. Different blocks see different spans, so the module covers several time scales at once.

This is also the limitation. Dilations widen the receptive field but never remove its boundary, so $\hat{s}_t$
depends on $[t-k, t+k]$ for a fixed $k$. Anything slower than $k$ — the arc of a question's intonation, the
drift of pitch across a clause — is not visible to the model that has to render it.

## A bidirectional state-space block

A state-space model carries a continuous-time system

$$\dot{h}(t) = \mathbf{A}h(t) + \mathbf{B}x(t), \qquad y(t) = \mathbf{C}h(t)$$

which, discretised, propagates a state along the sequence in $O(L)$ with no fixed horizon. Mamba makes
$\mathbf{B}$, $\mathbf{C}$ and the step size input-dependent, so the model chooses what to keep.

Mamba is causal: $y_t$ sees only $x_{\le t}$. Vocoding is not. Nothing stops us from conditioning sample $t$ on
the whole mel-spectrogram, and we should, because the right rendering of a syllable depends on the one after it.
So the block runs the sequence both ways and adds the results:

$$\Phi_{\text{Mamba}}(X) = \mathrm{LN}\big(\mathrm{SSM}(X) + \mathrm{Flip}(\mathrm{SSM}(\mathrm{Flip}(X)))\big)$$

Two `Mamba2` layers, one over $X$ and one over the time-reversed $X$, summed and layer-normed. That is
[`BiMamba2Block`](https://github.com/huseyin-karaca/mamba-hifigan/blob/main/src/mambahifigan/blocks.py), about
thirty lines, and every configuration that reached the results table is built from it.

## Where to put it

The block is the easy part. The architecture question is where it attaches, and that is where the project spent
its time. Nineteen placements were built and trained. Grouping them by *why* they failed is more useful than
listing them in the order we tried them.

/// tab | Phase coherence
A waveform is an oscillation about zero. Where the zero crossings fall is most of what makes it sound like
anything. Three designs moved them.

**Post-net refinement** runs Mamba over the finished waveform of a frozen HiFi-GAN. At $T > 22{,}000$ samples
the refiner cannot do sample-level correction, and the cheapest way to lower an L1 loss is to predict something
close to zero — so it did, and the output went muffled.

**Dual-path** synthesises texture and structure as two separate waveforms and adds them. The two networks have
different receptive fields, so their outputs are not phase-aligned, and summing them cancels frequencies instead
of reinforcing them.

**Multi-rate** runs the SSM at a lower sample rate to save compute. Downsampling aliases, upsampling smears
transients, and the L1 spectrogram loss ends up at 0.7068 against the baseline's 0.2975 — the worst result in
the study.

**The constraint:** never modify a waveform after generation, and never mix streams built at different rates.
///

/// tab | Gradient stability
GAN training is already delicate. Three designs made it worse.

**FiLM** has Mamba predict a scale $\gamma$ and shift $\beta$ for the convolutional features. The shift
introduces a DC offset, which moves exactly the zero crossings that matter, and the product $\gamma \odot x$
inflates gradient variance on the backward pass. The audio picked up a metallic edge.

**Squeeze-excite** uses Mamba as channel attention. This is stable — and useless. Reweighting channels can only
turn up or turn down structure the convolutions already produced. When a harmonic is simply missing, there is
nothing to amplify.

**Sequential progressive** puts Mamba before the MRF rather than beside it. The longer path attenuated gradients
reaching the early Mamba parameters, so they barely trained.

**The constraint:** combine additively, and keep the path short.
///

/// tab | Information bottlenecks
Three designs kept both signals but squeezed them on the way through.

**Concatenation** stacks both branches to $2C$ channels and projects back to $C$. One $1 \times 1$ layer has to
separate dense global structure from fine texture, and the gradient reaching it splits between the branches in a
way that favours the convolutional path. The SSM branch was effectively ignored.

**Broadcasting** runs Mamba once on the mel-spectrogram and injects the upsampled context at every layer.
Interpolating a coarse context up by a large factor does not synthesise high-frequency structure; it smears the
coarse structure across time. Feature matching was fine (0.0699) while L1 collapsed to 0.4479 — a good
illustration of why one loss is not enough to judge by.

**Pre-net** refines the mel-spectrogram before the generator. A mel is magnitude-only. However good the
magnitudes get, the phase problem is untouched.

**The constraint:** same resolution on both branches, and no narrow projection between them.
///

/// tab | Partial placement
If global context helps, does it have to be everywhere?

**Early stages only** (low resolution, where prosody would plausibly live) reached 0.0963 — better than several
alternatives, still worse than the baseline's 0.0736.

**Late stages only** (high resolution, to fix phase discontinuities directly) reached 0.0897. At $T > 22{,}000$
the sequence is probably simply too long for the block to model usefully.

**Serial within a stage** reached 0.0773, the closest of the three, still not an improvement.

**The reading:** no single resolution is the right place, which argues for attaching at every stage and letting
something else decide the weighting.
///

## What is left

Intersect the four constraints and very little remains. The SSM branch must run **in parallel** with the
convolutions (not before, not after), at **matched resolution** (no resampling), combined **additively** (no
multiplication, no projection), at **every stage** (no single resolution is right).

That describes a parallel sum — which was tried, as *Hybrid 1* and *Hybrid 2*, and gained almost nothing over
the baseline. A fixed 50/50 average assumes the two branches are equally useful everywhere, and there is no
reason to think they are. Early stages are coarse and long-range structure should matter more; late stages are
where fine texture is rendered.

So make the weight a parameter. At stage $i$, with one learnable scalar $w_i$:

$$\alpha_i = \sigma(w_i), \qquad y^{(i)} = \alpha_i \cdot \Phi_{\text{Mamba}}(x^{(i)}_{\text{up}}) + (1 - \alpha_i) \cdot \Psi_{\text{CNN}}(x^{(i)}_{\text{up}})$$

$w_i$ is initialised to 0, so $\alpha_i = 0.5$ and training starts from the parallel-sum model. Being a convex
combination, the output scale stays stable, and the gradient reaches both branches directly.

The cost is four scalars for a four-stage generator.

--8<-- "snippets/fig_gated.html"

!!! note "What we did not check"
    The natural next step is to read the trained $\alpha_i$ values and see whether the network really does
    prefer the SSM at low resolution and the convolutions at high resolution. We did not do it, and the
    checkpoints did not survive the rented instances, so the mechanism above is a hypothesis the results are
    consistent with rather than one they demonstrate.

## Cost

The Mamba-64 configuration adds roughly 18% to inference time over the baseline, Mamba-128 roughly 32%. Both
stay faster than real time. These are wall-clock measurements from single timing runs on one GPU, so treat them
as the right order of magnitude rather than precise figures.
