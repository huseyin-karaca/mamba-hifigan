// C: why: bounded window vs unbounded context along a speech signal.
import { D, COL, trace } from './common.js';
export const figure = { width: 183, height: 74, name: 'figure_c' };
export default async function draw(F, L) {
  const p = F.root, x0 = 74, x1 = 420;
  // rows: CNN, signal, Mamba
  const yS = 110, yC = 168, yM = 52;
  L.text(p, x0 - 8, yS, 'utterance', { anchor: 'end', size: 7, valign: 'middle' });
  trace(L, p, x0, x1, yS, { amp: 14, fine: 0.6 });
  // pitch contour hint
  L.text(p, x0 + 4, yS - 30, 'time →', { anchor: 'start', size: 5.5, color: COL.mid });
  // CNN windows at three places
  L.text(p, x0 - 8, yC, 'Convolution', { anchor: 'end', size: 7, valign: 'middle', color: COL.cnn, weight: 'bold' });
  [[110, 150], [225, 265], [330, 370]].forEach(([a, b], i) => {
    L.rect(p, a, yC - 9, b - a, 18, { fill: COL.cnnT, stroke: COL.cnn, width: 0.9, r: 3 });
    L.text(p, (a + b) / 2, yC, 'window', { anchor: 'middle', valign: 'middle', size: 5.5, color: COL.cnn });
    L.line(p, [a, yC - 9], [a, yS + 18], { stroke: COL.cnn, width: 0.5, dash: '2 2' });
    L.line(p, [b, yC - 9], [b, yS + 18], { stroke: COL.cnn, width: 0.5, dash: '2 2' });
  });
  L.text(p, 250, yC + 22, 'bounded window: texture', { anchor: 'middle', size: 6, color: COL.cnn });
  // Mamba whole span
  L.text(p, x0 - 8, yM, 'Bi-SSM', { anchor: 'end', size: 7, valign: 'middle', color: COL.mam, weight: 'bold' });
  L.rect(p, x0, yM - 9, x1 - x0, 18, { fill: COL.mamT, stroke: COL.mam, width: 0.9, r: 9 });
  L.arrow(p, [x0 + 12, yM - 2], [x1 - 12, yM - 2], { stroke: COL.mam, width: 0.9, head: 3 });
  L.arrow(p, [x1 - 12, yM + 4], [x0 + 12, yM + 4], { stroke: COL.mam, width: 0.9, head: 3 });
  L.text(p, 247, yM - 17, 'whole utterance, O(L): prosody, pitch drift', { anchor: 'middle', size: 6, color: COL.mam });
  L.line(p, [x0, yM + 9], [x0, yS - 18], { stroke: COL.mam, width: 0.5, dash: '2 2' });
  L.line(p, [x1, yM + 9], [x1, yS - 18], { stroke: COL.mam, width: 0.5, dash: '2 2' });
  // fusion
  const fx = 468, ym = yS;
  D.route(p, [[x1, yM], [fx, yM], [fx, ym - 8]], { head: false, stroke: COL.mam, width: 1, radius: 3 });
  D.route(p, [[x1 - 0, yC], [fx, yC], [fx, ym + 8]], { head: false, stroke: COL.cnn, width: 1, radius: 3 });
  D.opNode(p, fx, ym, 8, { op: '+', fill: COL.gateT, stroke: COL.gate, width: 1 });
  L.text(p, fx + 14, ym, 'α_i = σ(w_i)', { anchor: 'start', size: 6.5, color: COL.gate, weight: 'bold', valign: 'middle' });
}
