// Hero v1: generic TTS -> zoom into one vocoder stage -> global (BiMamba) vs local (MRF) -> learnable gate.
import { D, COL, mel, trace } from './common.js';
export const figure = { width: 183, height: 104, name: 'hero' };
export default async function draw(F, L) {
  const p = F.root, ink = COL.ink;
  // ---------- top strip: a generic TTS system ----------
  const ty = 8, th = 34, tc = ty + th / 2;
  D.box(p, 8, ty, 44, th, { label: 'Text', fill: '#fff', stroke: ink, width: 0.8, r: 4, size: 7 });
  D.route(p, [[52, tc], [66, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  D.box(p, 66, ty, 70, th, { label: 'Acoustic\nmodel', fill: COL.grey, stroke: ink, width: 0.8, r: 4, size: 7 });
  D.route(p, [[136, tc], [156, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  mel(L, p, 156, ty, 50, 24, 36);
  L.text(p, 181, ty + th + 8, 'mel', { anchor: 'middle', size: 6.5 });
  D.route(p, [[206, tc], [226, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  // vocoder = HiFi-GAN generator with four stages, one highlighted
  const gx = 226, gw = 150;
  L.rect(p, gx, ty - 2, gw, th + 4, { fill: '#fff', stroke: ink, width: 1.2, r: 5 });
  L.text(p, gx + gw / 2, ty + 7, 'HiFi-GAN generator (vocoder)', { anchor: 'middle', size: 6.5, weight: 'bold' });
  const sx = [gx + 12, gx + 47, gx + 82, gx + 117];
  sx.forEach((x, i) => {
    const hl = i === 1;
    L.rect(p, x, ty + 14, 22, 18, { fill: hl ? COL.gateT : COL.grey, stroke: hl ? COL.gate : '#9a9a9a', width: hl ? 1.3 : 0.7, r: 3 });
    if (i < 3) L.arrow(p, [x + 22, ty + 23], [sx[i + 1], ty + 23], { stroke: '#9a9a9a', width: 0.7, head: 2 });
  });
  D.route(p, [[gx + gw, tc], [396, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  trace(L, p, 398, 512, tc, { amp: 14, fine: 0.5 });
  L.text(p, 455, ty + th + 8, 'waveform', { anchor: 'middle', size: 6.5 });
  // ---------- zoom ----------
  const fy = 74, fh = 216, fx0 = 6, fx1 = 513;
  L.rect(p, fx0, fy, fx1 - fx0, fh, { fill: '#FBFBFB', stroke: COL.gate, width: 1, r: 7 });
  L.line(p, [sx[1], ty + 32], [fx0 + 8, fy], { stroke: COL.gate, width: 0.7, dash: '3 2' });
  L.line(p, [sx[1] + 22, ty + 32], [fx1 - 8, fy], { stroke: COL.gate, width: 0.7, dash: '3 2' });
  L.text(p, fx0 + 10, fy + 12, 'one generator stage', { anchor: 'start', size: 7, weight: 'bold', color: COL.gate });
  const cy = fy + fh / 2 + 8;
  // input conv
  D.box(p, 16, cy - 17, 48, 34, { label: 'Transposed\nconv ↑', fill: COL.grey, stroke: ink, width: 0.8, r: 4, size: 6.5 });
  L.text(p, 40, cy + 28, 'stage input', { anchor: 'middle', size: 6, color: COL.mid });
  L.arrow(p, [40, cy + 24], [40, cy + 18], { stroke: COL.mid, width: 0.6, head: 2 });
  const lx = 112, lw = 244, n = 15, gap = (lw - 36) / (n - 1), nx = (i) => lx + 18 + i * gap;
  // ---- global lane (BiMamba)
  const gy0 = fy + 22, gh = 80, gyc = gy0 + 49;
  L.rect(p, lx, gy0, lw, gh, { fill: COL.mamT, stroke: COL.mam, width: 1.1, r: 6 });
  L.text(p, lx + 10, gy0 + 11, 'BiMamba', { anchor: 'start', size: 8, weight: 'bold', color: COL.mam });
  L.text(p, lx + lw - 10, gy0 + 11, 'global structure', { anchor: 'end', size: 7.5, weight: 'bold', color: COL.mam });
  L.arrow(p, [nx(0), gyc - 14], [nx(n - 1), gyc - 14], { stroke: COL.mam, width: 1, head: 3.4 });
  L.arrow(p, [nx(n - 1), gyc + 14], [nx(0), gyc + 14], { stroke: COL.mam, width: 1, head: 3.4 });
  for (let i = 0; i < n; i++) {
    L.line(p, [nx(i), gyc - 14], [nx(i), gyc - 4], { stroke: COL.mam, width: 0.6 });
    L.line(p, [nx(i), gyc + 14], [nx(i), gyc + 4], { stroke: COL.mam, width: 0.6 });
    L.circle(p, nx(i), gyc, 4, { fill: '#fff', stroke: COL.mam, width: 0.9 });
  }
  L.text(p, nx(0) - 2, gyc - 19, 'forward state', { anchor: 'start', size: 5.5, color: COL.mam });
  L.text(p, nx(n - 1) + 2, gyc + 22, 'backward state', { anchor: 'end', size: 5.5, color: COL.mam });
  L.text(p, lx + lw / 2, gy0 + gh - 3, 'every frame sees the whole utterance', { anchor: 'middle', size: 5.5, color: COL.mam });
  // ---- local lane (MRF)
  const ly0 = gy0 + gh + 14, lh = fy + fh - 10 - ly0, lyc = ly0 + 44;
  L.rect(p, lx, ly0, lw, lh, { fill: COL.cnnT, stroke: COL.cnn, width: 1.1, r: 6 });
  L.text(p, lx + 10, ly0 + 11, 'MRF', { anchor: 'start', size: 8, weight: 'bold', color: COL.cnn });
  L.text(p, lx + lw - 10, ly0 + 11, 'local structure', { anchor: 'end', size: 7.5, weight: 'bold', color: COL.cnn });
  const c0 = 5;                              // window centre node
  [[11, 0.18], [7, 0.22], [3, 0.3]].forEach(([k, op]) => {
    const a = nx(c0 - (k - 1) / 2) - 6, b = nx(c0 + (k - 1) / 2) + 6;
    L.rect(p, a, lyc - 11, b - a, 22, { fill: COL.cnn, fillOpacity: op, stroke: COL.cnn, width: 0.6, r: 6 });
  });
  for (let i = 0; i < n; i++) L.circle(p, nx(i), lyc, 4, { fill: '#fff', stroke: COL.cnn, width: 0.9 });
  L.circle(p, nx(c0), lyc, 4, { fill: COL.cnn, stroke: COL.cnn });
  [[3, 18], [7, 27], [11, 36]].forEach(([k, dy]) => {
    const a = nx(c0 - (k - 1) / 2) - 6, b = nx(c0 + (k - 1) / 2) + 6;
    L.line(p, [a, lyc + dy - 4], [b, lyc + dy - 4], { stroke: COL.cnn, width: 0.8 });
    L.text(p, b + 4, lyc + dy - 4, `k=${k}`, { anchor: 'start', size: 5.5, color: COL.cnn, valign: 'middle' });
  });
  L.text(p, lx + lw - 10, ly0 + lh - 6, 'each frame sees a bounded window', { anchor: 'end', size: 5.5, color: COL.cnn });
  // ---- split from conv to lanes
  D.route(p, [[64, cy], [86, cy], [86, gyc], [lx, gyc]], { head: 'fill', stroke: COL.mam, width: 1.3, radius: 4 });
  D.route(p, [[86, cy], [86, lyc], [lx, lyc]], { head: 'fill', stroke: COL.cnn, width: 1.3, radius: 4 });
  L.circle(p, 86, cy, 2, { fill: ink });
  // ---- fusion
  const px = 396, py = cy;
  D.route(p, [[lx + lw, gyc], [px, gyc], [px, py - 11]], { head: 'fill', stroke: COL.mam, width: 1.3, radius: 4 });
  D.route(p, [[lx + lw, lyc], [px, lyc], [px, py + 11]], { head: 'fill', stroke: COL.cnn, width: 1.3, radius: 4 });
  D.opNode(p, px, py, 10, { op: '+', fill: COL.gateT, stroke: COL.gate, width: 1.4 });
  L.text(p, px - 8, gyc + 14, 'α_i', { anchor: 'end', size: 8, italic: true, color: COL.mam, weight: 'bold' });
  L.text(p, px - 8, lyc - 14, '1 − α_i', { anchor: 'end', size: 8, italic: true, color: COL.cnn, weight: 'bold' });
  // learnable gate card
  const gcx = 460, cx0 = 415, cw = 94;
  L.rect(p, cx0, fy + 22, cw, 86, { fill: COL.gateT, stroke: COL.gate, width: 1.4, r: 7 });
  L.text(p, gcx + 8, fy + 36, 'Learnable gate', { anchor: 'middle', size: 8, weight: 'bold', color: COL.gate });
  D.flame(p, gcx - 32, fy + 36, 7, { color: COL.vermillion });
  L.rect(p, gcx - 26, fy + 54, 52, 12, { fill: '#fff', stroke: COL.gate, width: 1, r: 6 });
  L.circle(p, gcx, fy + 60, 6.5, { fill: COL.gate, stroke: '#fff', width: 1 });
  L.text(p, gcx - 30, fy + 61, 'CNN', { anchor: 'end', size: 5.5, color: COL.cnn, valign: 'middle', weight: 'bold' });
  L.text(p, gcx + 30, fy + 61, 'SSM', { anchor: 'start', size: 5.5, color: COL.mam, valign: 'middle', weight: 'bold' });
  L.text(p, gcx, fy + 80, 'α_i = σ(w_i)', { anchor: 'middle', size: 8.5, weight: 'bold', color: COL.gate });
  L.text(p, gcx, fy + 92, 'one scalar per stage', { anchor: 'middle', size: 6, color: COL.gate });
  L.text(p, gcx, fy + 101, 'w_i = 0 at init', { anchor: 'middle', size: 6, color: COL.mid });
  L.arrow(p, [cx0 + 8, fy + 102], [px + 8, py - 9], { stroke: COL.gate, width: 1, dash: '2.5 1.8', head: 3 });
  // output
  D.route(p, [[px + 10, py], [452, py]], { head: 'fill', stroke: ink, width: 1.2 });
  L.text(p, 477, py + 12, 'to next stage', { anchor: 'middle', size: 6.5 });
  L.text(p, 477, py + 36, 'y = α_i · BiMamba(x)', { anchor: 'middle', size: 6.5, color: COL.mam });
  L.text(p, 477, py + 46, '+ (1 − α_i) · MRF(x)', { anchor: 'middle', size: 6.5, color: COL.cnn });
}
