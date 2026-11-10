// D: where to put Mamba: 19 placements -> three constraints -> one survivor.
import { D, COL } from './common.js';
export const figure = { width: 183, height: 58, name: 'figure_d' };
export default async function draw(F, L) {
  const p = F.root, cy = 90;
  // 19 dots as a cloud (5 cols x 4 rows minus one)
  const dots = [];
  for (let r = 0; r < 4; r++) for (let c = 0; c < 5; c++) if (r * 5 + c < 19) dots.push([22 + c * 13, cy - 26 + r * 13]);
  dots.forEach(([x, y], i) => L.circle(p, x, y, 4, { fill: i === 7 ? COL.gate : '#cfcfcf', stroke: i === 7 ? COL.gate : '#9a9a9a', width: 0.5 }));
  L.text(p, 48, cy + 38, '19 placements of', { anchor: 'middle', size: 6.5 });
  L.text(p, 48, cy + 46, 'a Mamba block', { anchor: 'middle', size: 6.5 });
  // three sieves
  const sx = [150, 255, 360], names = ['phase\ncoherence', 'stable\ngradients', 'no\nbottleneck'];
  sx.forEach((x, i) => {
    D.route(p, [[(i ? sx[i - 1] + 6 : 92), cy], [x - 6, cy]], { head: 'fill', stroke: COL.ink, width: 0.9 });
    // sieve: bars with a gap
    L.rect(p, x - 5, cy - 40, 10, 32, { fill: COL.mamT, stroke: COL.mam, width: 0.8, r: 2 });
    L.rect(p, x - 5, cy + 8, 10, 32, { fill: COL.mamT, stroke: COL.mam, width: 0.8, r: 2 });
    L.text(p, x, cy - 56, names[i], { anchor: 'middle', size: 6.5, lineGap: 1 });
    L.text(p, x, cy + 52, '✕ fails here', { anchor: 'middle', size: 5.5, color: COL.mam });
  });
  D.route(p, [[sx[2] + 6, cy], [412, cy]], { head: 'fill', stroke: COL.gate, width: 1.4 });
  // survivor card
  L.rect(p, 414, cy - 36, 98, 72, { fill: COL.gateT, stroke: COL.gate, width: 1, r: 5 });
  L.text(p, 463, cy - 24, 'survivor', { anchor: 'middle', size: 7, weight: 'bold', color: COL.gate });
  D.box(p, 424, cy - 15, 34, 12, { label: 'BiMamba', fill: COL.mamT, stroke: COL.mam, width: 0.8, r: 2, size: 5.5, color: COL.mam });
  D.box(p, 424, cy + 3, 34, 12, { label: 'MRF', fill: COL.cnnT, stroke: COL.cnn, width: 0.8, r: 2, size: 5.5, color: COL.cnn });
  D.opNode(p, 478, cy, 6, { op: '+', fill: '#fff', stroke: COL.gate, width: 0.8 });
  D.route(p, [[458, cy - 9], [478, cy - 9], [478, cy - 6]], { head: false, stroke: COL.mam, width: 0.7 });
  D.route(p, [[458, cy + 9], [478, cy + 9], [478, cy + 6]], { head: false, stroke: COL.cnn, width: 0.7 });
  L.text(p, 463, cy + 26, 'parallel · additive · learned α', { anchor: 'middle', size: 5.5, color: COL.gate });
}
