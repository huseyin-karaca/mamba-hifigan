// A: one generator stage, zoomed. Mechanism hero.
import { D, COL, mel, trace } from './common.js';
export const figure = { width: 183, height: 82, name: 'figure_a' };
export default async function draw(F, L) {
  const W = L.mm(183), H = L.mm(82);
  const p = F.root;
  const cy = H / 2;
  // input mel
  mel(L, p, 6, cy - 30, 56, 40, 64);
  L.text(p, 34, cy + 36, 'mel spectrogram', { anchor: 'middle', size: 6 });
  // upsample
  D.box(p, 78, cy - 15, 40, 30, { label: 'Transposed\nconv ↑', fill: COL.grey, stroke: COL.ink, width: 0.8, r: 3, size: 6 });
  D.route(p, [[62, cy], [78, cy]], { head: 'fill', stroke: COL.ink, width: 0.9 });
  // branch lanes
  const yT = cy - 42, yB = cy + 42, xb = 160, xe = 330;
  L.rect(p, 128, 4, 232, H - 8, { fill: '#FAFAFA', stroke: COL.light || '#bdbdbd', dash: '3 2', width: 0.6, r: 6 });
  L.text(p, 244, 12, 'one generator stage  (×4, resolutions grow)', { anchor: 'middle', size: 6, color: COL.mid });
  D.route(p, [[118, cy], [136, cy], [136, yT], [xb, yT]], { head: 'fill', stroke: COL.mam, width: 1.1, radius: 3 });
  D.route(p, [[136, cy], [136, yB], [xb, yB]], { head: 'fill', stroke: COL.cnn, width: 1.1, radius: 3 });
  L.circle(p, 136, cy, 1.6, { fill: COL.ink });
  // Mamba lane: fwd + bwd scans
  D.box(p, xb, yT - 17, 88, 34, { fill: COL.mamT, stroke: COL.mam, width: 1, r: 4 });
  L.text(p, xb + 44, yT - 6, 'BiMamba2', { anchor: 'middle', size: 7, weight: 'bold', color: COL.mam });
  L.arrow(p, [xb + 10, yT + 4], [xb + 78, yT + 4], { stroke: COL.mam, width: 0.9, head: 3 });
  L.arrow(p, [xb + 78, yT + 11], [xb + 10, yT + 11], { stroke: COL.mam, width: 0.9, head: 3 });
  L.text(p, xb + 44, yT + 24, 'whole utterance, O(L)', { anchor: 'middle', size: 5.5, color: COL.mam });
  // CNN lane: three resblocks of different kernel
  const ks = [3, 7, 11];
  ks.forEach((k, i) => {
    D.box(p, xb + i * 30, yB - 14, 26, 28, { label: `Res\nk=${k}`, fill: COL.cnnT, stroke: COL.cnn, width: 1, r: 3, size: 5.5 });
  });
  L.text(p, xb + 44, yB + 24, 'MRF: local windows', { anchor: 'middle', size: 5.5, color: COL.cnn });
  // fusion
  const fx = 300, xg = 314;
  D.route(p, [[xb + 88, yT], [fx, yT], [fx, cy - 8]], { head: false, stroke: COL.mam, width: 1.1, radius: 3 });
  D.route(p, [[xb + 88, yB], [fx, yB], [fx, cy + 8]], { head: false, stroke: COL.cnn, width: 1.1, radius: 3 });
  D.opNode(p, fx, cy, 8, { op: '+', fill: COL.gateT, stroke: COL.gate, width: 1 });
  L.text(p, fx - 12, yT + 24, 'α_i', { anchor: 'end', size: 7, color: COL.mam, italic: true });
  L.text(p, fx - 12, yB - 24, '1 − α_i', { anchor: 'end', size: 7, color: COL.cnn, italic: true });
  // gate slider
  L.rect(p, xg + 4, cy - 30, 36, 8, { fill: '#fff', stroke: COL.gate, width: 0.8, r: 4 });
  L.circle(p, xg + 22, cy - 26, 4.2, { fill: COL.gate, stroke: '#fff', width: 0.8 });
  L.text(p, xg + 22, cy - 36, 'α_i = σ(w_i)', { anchor: 'middle', size: 6.5, color: COL.gate, weight: 'bold' });
  L.text(p, xg + 22, cy - 16, 'w_i = 0 at start', { anchor: 'middle', size: 5.5, color: COL.mid });
  L.arrow(p, [xg + 4, cy - 26], [fx + 6, cy - 6], { stroke: COL.gate, width: 0.7, dash: '2 1.5', head: 2.5 });
  // output
  D.route(p, [[fx + 8, cy], [380, cy]], { head: 'fill', stroke: COL.ink, width: 0.9 });
  trace(L, p, 380, 508, cy, { amp: 22, fine: 0.9 });
  L.text(p, 444, cy + 36, 'waveform', { anchor: 'middle', size: 6 });
}
