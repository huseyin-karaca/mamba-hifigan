// B: whole generator as a pipeline, four gated stages, time axis grows.
import { D, COL, mel, trace } from './common.js';
export const figure = { width: 183, height: 52, name: 'figure_b' };
export default async function draw(F, L) {
  const p = F.root, H = L.mm(70), cy = H / 2 + 8;
  mel(L, p, 4, cy - 22, 44, 30, 48);
  L.text(p, 26, cy + 28, 'mel', { anchor: 'middle', size: 6 });
  D.box(p, 58, cy - 10, 26, 20, { label: 'conv\npre', fill: COL.grey, stroke: COL.ink, width: 0.8, r: 3, size: 5.5 });
  D.route(p, [[48, cy], [58, cy]], { head: 'fill', stroke: COL.ink, width: 0.8 });
  const ups = [8, 8, 2, 2], ws = [58, 74, 90, 104], hs = [50, 42, 34, 28];
  let x = 90;
  ups.forEach((u, i) => {
    D.route(p, [[x - 6, cy], [x + 4, cy]], { head: 'fill', stroke: COL.ink, width: 0.8 });
    const w = 50, h = hs[i];
    L.rect(p, x + 4, cy - h / 2 - 4, 86, h + 8, { fill: '#FAFAFA', stroke: '#bdbdbd', width: 0.6, r: 5, dash: '3 2' });
    D.box(p, x + 8, cy - 11, 20, 22, { label: `↑${u}`, fill: COL.grey, stroke: COL.ink, width: 0.8, r: 3, size: 6 });
    D.box(p, x + 34, cy - h / 2 + 1, 30, h / 2 - 3, { label: 'BiMamba', fill: COL.mamT, stroke: COL.mam, width: 0.9, r: 3, size: 5.5, color: COL.mam });
    D.box(p, x + 34, cy + 2, 30, h / 2 - 3, { label: 'MRF', fill: COL.cnnT, stroke: COL.cnn, width: 0.9, r: 3, size: 5.5, color: COL.cnn });
    D.route(p, [[x + 28, cy], [x + 31, cy], [x + 31, cy - h / 4 + 1], [x + 34, cy - h / 4 + 1]], { head: false, stroke: COL.mam, width: 0.8 });
    D.route(p, [[x + 31, cy], [x + 31, cy + h / 4 + 1], [x + 34, cy + h / 4 + 1]], { head: false, stroke: COL.cnn, width: 0.8 });
    D.route(p, [[x + 64, cy - h / 4 + 1], [x + 70, cy - h / 4 + 1], [x + 70, cy - 7]], { head: false, stroke: COL.mam, width: 0.8 });
    D.route(p, [[x + 64, cy + h / 4 + 1], [x + 70, cy + h / 4 + 1], [x + 70, cy + 7]], { head: false, stroke: COL.cnn, width: 0.8 });
    D.opNode(p, x + 70, cy, 7, { op: '+', fill: COL.gateT, stroke: COL.gate, width: 0.9 });
    // gate knob above
    L.rect(p, x + 58, cy - h / 2 - 18, 24, 6, { fill: '#fff', stroke: COL.gate, width: 0.7, r: 3 });
    L.circle(p, x + 70, cy - h / 2 - 15, 3, { fill: COL.gate });
    L.text(p, x + 70, cy - h / 2 - 22, `α_${i + 1}`, { anchor: 'middle', size: 6.5, color: COL.gate, italic: true });
    D.route(p, [[x + 70, cy - h / 2 - 12], [x + 70, cy - 9]], { head: false, stroke: COL.gate, width: 0.6, dash: '2 1.5' });
    x += 92;
  });
  D.route(p, [[x - 4, cy], [x + 6, cy]], { head: 'fill', stroke: COL.ink, width: 0.8 });
  trace(L, p, x + 8, 516, cy, { amp: 18, fine: 0.5 });
  L.text(p, (x + 8 + 516) / 2, cy + 28, 'waveform', { anchor: 'middle', size: 6 });
  L.text(p, 262, 8, 'four upsampling stages, each with a gated parallel pair', { anchor: 'middle', size: 6, color: COL.mid });
}
