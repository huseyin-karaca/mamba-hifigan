// E: mixing console: two lanes, one crossfader.
import { D, COL, trace } from './common.js';
export const figure = { width: 183, height: 62, name: 'figure_e' };
export default async function draw(F, L) {
  const p = F.root, cy = 100;
  // input
  L.text(p, 40, cy - 40, 'features', { anchor: 'middle', size: 6.5 });
  trace(L, p, 8, 72, cy - 28, { amp: 9, fine: 0.7, color: '#555' });
  D.route(p, [[76, cy - 28], [100, cy - 28], [100, cy - 50], [130, cy - 50]], { head: 'fill', stroke: COL.mam, width: 1, radius: 3 });
  D.route(p, [[100, cy - 28], [100, cy + 40], [130, cy + 40]], { head: 'fill', stroke: COL.cnn, width: 1, radius: 3 });
  // lanes
  L.rect(p, 130, cy - 72, 150, 44, { fill: COL.mamT, stroke: COL.mam, width: 1, r: 5 });
  L.text(p, 136, cy - 64, 'BiMamba · global context', { anchor: 'start', size: 6.5, weight: 'bold', color: COL.mam });
  trace(L, p, 138, 272, cy - 44, { amp: 11, smooth: true, color: COL.mam, width: 1.2 });
  L.rect(p, 130, cy + 18, 150, 44, { fill: COL.cnnT, stroke: COL.cnn, width: 1, r: 5 });
  L.text(p, 136, cy + 26, 'Convolutions · local texture', { anchor: 'start', size: 6.5, weight: 'bold', color: COL.cnn });
  trace(L, p, 138, 272, cy + 46, { amp: 9, fine: 1.4, color: COL.cnn, width: 0.6 });
  // crossfader
  const fx = 330;
  D.route(p, [[280, cy - 50], [fx, cy - 50], [fx, cy - 18]], { head: false, stroke: COL.mam, width: 1, radius: 3 });
  D.route(p, [[280, cy + 40], [fx, cy + 40], [fx, cy + 18]], { head: false, stroke: COL.cnn, width: 1, radius: 3 });
  L.rect(p, fx - 4, cy - 26, 8, 52, { fill: '#fff', stroke: COL.gate, width: 1, r: 4 });
  L.rect(p, fx - 9, cy - 5, 18, 10, { fill: COL.gate, stroke: '#fff', width: 0.8, r: 2 });
  L.text(p, fx + 14, cy - 18, 'α', { anchor: 'start', size: 7, italic: true, color: COL.mam });
  L.text(p, fx + 14, cy + 22, '1−α', { anchor: 'start', size: 7, italic: true, color: COL.cnn });
  L.text(p, fx + 4, cy + 34, 'α = σ(w)', { anchor: 'middle', size: 6, color: COL.gate, weight: 'bold' });
  // output
  D.route(p, [[fx + 12, cy], [400, cy]], { head: 'fill', stroke: COL.ink, width: 1 });
  trace(L, p, 404, 512, cy, { amp: 18, fine: 0.5 });
  L.text(p, 458, cy - 28, 'waveform', { anchor: 'middle', size: 6.5 });
  // four faders for four stages
  L.text(p, 458, cy + 22, 'one gate per stage', { anchor: 'middle', size: 6, color: COL.mid });
  for (let i = 0; i < 4; i++) {
    const x = 418 + i * 22;
    L.rect(p, x - 2, cy + 28, 4, 22, { fill: '#fff', stroke: COL.gate, width: 0.7, r: 2 });
    L.rect(p, x - 5, cy + 36, 10, 5, { fill: COL.gate, r: 1 });
  }
}
