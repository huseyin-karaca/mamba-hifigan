# Library gaps (hero2)
- Bidirectional scan stream (two opposite lanes with absorbed-context ramps and direction chevrons): drawn in hero2.js mambaStreams from L.rect + L.linearGradient + L.chevron; no SSM/RNN state-stream part exists in engine/cs.js.
- Receptive-field funnel (nested, per-layer widths from kernel and dilations): drawn in mrfFunnels from L.polygon; a `D.receptiveField(k, dilations)` part would generalise to WaveNet/TCN figures.
- Site web font: the engine only registers Latin Modern; hero2.js injects an @font-face for Schibsted Grotesk (fonts/, OFL) and sets STYLE.font.family at draw time.
