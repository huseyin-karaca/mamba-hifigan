// Hero v2: generic TTS -> zoom into one vocoder stage -> global (BiMamba) vs local (MRF) -> learnable gate.
import { D, COL, trace } from './common.js';
export const figure = { width: 183, height: 124, name: 'hero2' };

// speech-like mel spectrogram: gliding harmonics (mel-compressed spacing), formant envelope, silences, a fricative burst
const sq = (x) => x * x;
function melSpec(rows, cols) {
  const hz2y = (f) => Math.log(1 + f / 700) / Math.log(1 + 8000 / 700);
  const env = (t) => [[0.17, 0.09, 1.0], [0.42, 0.10, 0.85], [0.68, 0.07, 0.95], [0.86, 0.06, 0.8]]
    .reduce((s, [c, w, a]) => s + a * Math.exp(-sq((t - c) / w)), 0);
  const fric = (t) => Math.exp(-sq((t - 0.555) / 0.028)) + 0.8 * Math.exp(-sq((t - 0.955) / 0.02));
  const rnd = (i, j) => { const x = Math.sin(i * 127.1 + j * 311.7) * 43758.5453; return x - Math.floor(x); };
  const data = [];
  for (let r = 0; r < rows; r++) {
    const row = [], y = 1 - r / (rows - 1);
    for (let c = 0; c < cols; c++) {
      const t = c / (cols - 1);
      const f0 = 118 + 28 * Math.sin(6.28 * 1.15 * t + 0.4) - 18 * t;
      const e = env(t);
      const fm = [[0.28 + 0.05 * Math.sin(5 * t), 0.15], [0.50 + 0.05 * Math.cos(4 * t), 0.14], [0.72, 0.12]];
      const form = fm.reduce((s, [m, w]) => s + Math.exp(-sq((y - m) / w)), 0.3);
      let v = 0;
      for (let k = 1; k <= 40; k++) {
        const yk = hz2y(f0 * k); if (yk > 1) break;
        const d = (y - yk) / (0.0065 + 0.004 * yk);
        v += Math.exp(-d * d) / Math.sqrt(k);
      }
      v = e * (v * form * 0.9 * (1.15 - 0.6 * y) * 1.6 + 0.08 * form);
      v += fric(t) * Math.max(0, y - 0.5) * (0.5 + 0.5 * rnd(r, c)) * 1.1;
      v += 0.025 * rnd(r * 3 + 1, c * 5 + 2);
      row.push(Math.max(0, Math.min(1, v)) ** 0.55);
    }
    data.push(row);
  }
  return data;
}
const MAGMA = ['#000004', '#1b0c41', '#4f0d6c', '#8a226a', '#c43c4e', '#ed6925', '#fbb61a', '#fcffa4'];

export default async function draw(F, L) {
  const p = F.root, ink = COL.ink, shadow = L.dropShadow(p, { dx: 0, dy: 0.8, sd: 1.4, opacity: 0.14 });
  const X0 = 6, X1 = 513;
  // ---------- top strip: a generic TTS system ----------
  const ty = 16, th = 36, tc = ty + th / 2;
  D.box(p, X0, ty, 44, th, { label: 'Text', fill: '#fff', stroke: ink, width: 0.9, r: 5, size: 7.5 });
  D.route(p, [[50, tc], [64, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  D.box(p, 64, ty, 92, th, { label: 'Acoustic model\n(e.g. FastSpeech 2)', fill: COL.grey, stroke: ink, width: 0.9, r: 5, size: 7, lineGap: 2 });
  D.route(p, [[156, tc], [172, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  const mw = 58, md = melSpec(88, 144);
  const mi = L.pixelImage(p, 172, ty, mw, md, { palette: MAGMA, frame: false, overlap: 0.6, skipBelow: 0 });
  L.rect(p, 172, ty, mw, mw * 88 / 144, { fill: 'none', stroke: ink, width: 0.7 });
  L.text(p, 172 + mw / 2, ty - 4, 'mel spectrogram', { anchor: 'middle', size: 6.5 });
  D.route(p, [[172 + mw, tc], [246, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  const gx = 246, gw = 140;
  L.rect(p, gx, ty - 2, gw, th + 4, { fill: '#fff', stroke: ink, width: 1.3, r: 6 });
  L.text(p, gx + gw / 2, ty + 7, 'HiFi-GAN generator', { anchor: 'middle', size: 7, weight: 'bold' });
  const ups = ['↑8', '↑8', '↑2', '↑2'];
  const sx = ups.map((_, i) => gx + 10 + i * 32);
  ups.forEach((u, i) => {
    const hl = i === 1;
    L.rect(p, sx[i], ty + 15, 22, 19, { fill: hl ? COL.gateT : COL.grey, stroke: hl ? COL.gate : '#9a9a9a', width: hl ? 1.4 : 0.7, r: 3 });
    L.text(p, sx[i] + 11, ty + 25, u, { anchor: 'middle', valign: 'middle', size: 6.5, color: hl ? COL.gate : COL.mid, weight: hl ? 'bold' : 'normal' });
    if (i < 3) L.arrow(p, [sx[i] + 22, ty + 24.5], [sx[i + 1], ty + 24.5], { stroke: '#9a9a9a', width: 0.7, head: 2 });
  });
  D.route(p, [[gx + gw, tc], [404, tc]], { head: 'fill', stroke: ink, width: 0.9 });
  trace(L, p, 408, X1, tc, { amp: 14, fine: 0.5 });
  L.text(p, (408 + X1) / 2, ty - 4, 'waveform', { anchor: 'middle', size: 6.5 });
  // ---------- zoom frame ----------
  const fy = 84, fh = 266;
  L.rect(p, X0, fy, X1 - X0, fh, { fill: '#FBFBFB', stroke: COL.gate, width: 1.1, r: 8 });
  L.line(p, [sx[1], ty + 34], [X0 + 10, fy], { stroke: COL.gate, width: 0.7, dash: '3 2' });
  L.line(p, [sx[1] + 22, ty + 34], [X1 - 10, fy], { stroke: COL.gate, width: 0.7, dash: '3 2' });
  L.text(p, X0 + 12, fy + 13, 'One generator stage', { anchor: 'start', size: 7.5, weight: 'bold', color: COL.gate });

  const lx = 112, lw = 244, n = 15, gap = (lw - 36) / (n - 1), nx = (i) => lx + 18 + i * gap, c0 = 7;
  const gy0 = fy + 22, gh = 108, ly0 = gy0 + gh + 10, lh = fy + fh - 10 - ly0;
  const cy = (gy0 + ly0 + lh) / 2 + 2;
  const rng = L.rng(7);
  // ================= global lane: BiMamba =================
  L.rect(p, lx, gy0, lw, gh, { fill: COL.mamT, stroke: COL.mam, width: 1.2, r: 7, filter: shadow });
  L.text(p, lx + 10, gy0 + 12, 'BiMamba', { anchor: 'start', size: 8, weight: 'bold', color: COL.mam });
  L.text(p, lx + 53, gy0 + 12, 'forward + backward state-space scan', { anchor: 'start', size: 5.5, color: COL.mam });
  L.text(p, lx + lw - 10, gy0 + 12, 'global structure', { anchor: 'end', size: 7.5, weight: 'bold', color: COL.mam });
  const yO = gy0 + 34, yS = gy0 + 60, yI = gy0 + 88;
  for (let i = 0; i < n; i++) L.line(p, [nx(c0), yO], [nx(i), yI], { stroke: COL.mam, width: 0.45, opacity: 0.35 });   // fan: one output depends on every input
  for (let i = 0; i < n; i++) {
    L.line(p, [nx(i), yO], [nx(i), yI], { stroke: COL.mam, width: 0.6, opacity: 0.55 });
    if (i < n - 1) {
      L.arrow(p, [nx(i) + 4.5, yS - 3], [nx(i + 1) - 4.5, yS - 3], { stroke: COL.mam, width: 0.8, head: 2 });
      L.arrow(p, [nx(i + 1) - 4.5, yS + 3.5], [nx(i) + 4.5, yS + 3.5], { stroke: '#E8A06A', width: 0.8, head: 2 });
    }
  }
  for (let i = 0; i < n; i++) {
    const a = 0.45 + 0.5 * rng(), b = 0.35 + 0.5 * rng();
    L.rect(p, nx(i) - 3.2, yS - 6.2, 6.4, 5.6, { fill: COL.mam, fillOpacity: a, stroke: COL.mam, width: 0.6, r: 1 });
    L.rect(p, nx(i) - 3.2, yS + 0.6, 6.4, 5.6, { fill: '#E8A06A', fillOpacity: b, stroke: '#C77A3A', width: 0.6, r: 1 });
    L.circle(p, nx(i), yO, 3.6, { fill: i === c0 ? COL.mam : '#fff', stroke: COL.mam, width: 1 });
    L.circle(p, nx(i), yI, 3.6, { fill: '#fff', stroke: COL.mam, width: 1 });
  }
  L.text(p, lx + 3, yS - 3, 'fwd', { anchor: 'start', size: 5, color: COL.mam, valign: 'middle' });
  L.text(p, lx + 3, yS + 4, 'bwd', { anchor: 'start', size: 5, color: '#C77A3A', valign: 'middle' });
  L.text(p, lx + lw / 2, gy0 + gh - 5, 'one output frame sees the whole utterance', { anchor: 'middle', size: 5.8, color: COL.mam });
  // ================= local lane: MRF =================
  L.rect(p, lx, ly0, lw, lh, { fill: COL.cnnT, stroke: COL.cnn, width: 1.2, r: 7, filter: shadow });
  L.text(p, lx + 10, ly0 + 12, 'MRF', { anchor: 'start', size: 8, weight: 'bold', color: COL.cnn });
  L.text(p, lx + 34, ly0 + 12, '3 parallel ResBlocks, k = 3 · 7 · 11', { anchor: 'start', size: 5.5, color: COL.cnn });
  L.text(p, lx + lw - 10, ly0 + 12, 'local structure', { anchor: 'end', size: 7.5, weight: 'bold', color: COL.cnn });
  const ry = [ly0 + 30, ly0 + 50, ly0 + 70, ly0 + 90];
  for (let r = 0; r < 3; r++) for (let i = 0; i < n; i++) for (const d of [-1, 0, 1]) {
    const j = i + d; if (j < 0 || j >= n) continue;
    const inCone = Math.abs(i - c0) <= r && Math.abs(j - c0) <= r + 1;
    if (!inCone) L.line(p, [nx(i), ry[r]], [nx(j), ry[r + 1]], { stroke: COL.cnn, width: 0.4, opacity: 0.3 });
  }
  for (let r = 0; r < 3; r++) for (let i = 0; i < n; i++) for (const d of [-1, 0, 1]) {
    const j = i + d; if (j < 0 || j >= n) continue;
    if (Math.abs(i - c0) <= r && Math.abs(j - c0) <= r + 1) L.line(p, [nx(i), ry[r]], [nx(j), ry[r + 1]], { stroke: COL.cnn, width: 0.9 });
  }
  for (let r = 0; r < 4; r++) for (let i = 0; i < n; i++) {
    const inC = Math.abs(i - c0) <= r;
    L.circle(p, nx(i), ry[r], 3.6, { fill: r === 0 && i === c0 ? COL.cnn : inC ? '#9CC7E6' : '#fff', stroke: COL.cnn, width: 1 });
  }
  L.text(p, lx + lw / 2, ly0 + lh - 5, 'one output frame sees a bounded window', { anchor: 'middle', size: 5.8, color: COL.cnn });
  // ================= input, split =================
  D.box(p, 16, cy - 18, 50, 36, { label: 'Transposed\nconv ↑', fill: COL.grey, stroke: ink, width: 0.9, r: 5, size: 6.5 });
  L.text(p, 41, cy + 32, 'stage input', { anchor: 'middle', size: 6, color: COL.mid });
  L.arrow(p, [41, cy + 25], [41, cy + 20], { stroke: COL.mid, width: 0.6, head: 2 });
  const gyc = gy0 + gh / 2, lyc = ly0 + lh / 2;
  D.route(p, [[66, cy], [88, cy]], { head: false, stroke: ink, width: 1.4 });
  D.route(p, [[88, cy], [88, gyc], [lx, gyc]], { head: 'fill', stroke: COL.mam, width: 1.4, radius: 5 });
  D.route(p, [[88, cy], [88, lyc], [lx, lyc]], { head: 'fill', stroke: COL.cnn, width: 1.4, radius: 5 });
  L.circle(p, 88, cy, 2.2, { fill: ink });
  // ================= fusion + learnable gate =================
  const px = 396, py = cy;
  D.route(p, [[lx + lw, gyc], [px, gyc], [px, py - 11]], { head: 'fill', stroke: COL.mam, width: 1.4, radius: 5 });
  D.route(p, [[lx + lw, lyc], [px, lyc], [px, py + 11]], { head: 'fill', stroke: COL.cnn, width: 1.4, radius: 5 });
  D.opNode(p, px, py, 10, { op: '+', fill: COL.gateT, stroke: COL.gate, width: 1.5 });
  L.text(p, px - 8, gyc + 18, 'α_i', { anchor: 'end', size: 8.5, italic: true, color: COL.mam, weight: 'bold' });
  L.text(p, px - 8, lyc - 18, '1 − α_i', { anchor: 'end', size: 8.5, italic: true, color: COL.cnn, weight: 'bold' });
  const gcx = 460, cx0 = 413, cw = 96, gy = gy0 - 2;
  L.rect(p, cx0, gy, cw, 90, { fill: COL.gateT, stroke: COL.gate, width: 1.5, r: 8, filter: shadow });
  L.text(p, gcx, gy + 14, 'Learnable gate', { anchor: 'middle', size: 8.5, weight: 'bold', color: COL.gate });
  L.rect(p, gcx - 27, gy + 26, 54, 12, { fill: '#fff', stroke: COL.gate, width: 1.1, r: 6 });
  L.circle(p, gcx, gy + 32, 6.5, { fill: COL.gate, stroke: '#fff', width: 1 });
  L.text(p, gcx - 32, gy + 33, 'CNN', { anchor: 'end', size: 5.5, color: COL.cnn, valign: 'middle', weight: 'bold' });
  L.text(p, gcx + 32, gy + 33, 'SSM', { anchor: 'start', size: 5.5, color: COL.mam, valign: 'middle', weight: 'bold' });
  L.text(p, gcx, gy + 54, 'α_i = σ(w_i)', { anchor: 'middle', size: 9, weight: 'bold', color: COL.gate });
  L.text(p, gcx, gy + 67, 'one scalar per stage', { anchor: 'middle', size: 6, color: COL.gate });
  L.text(p, gcx, gy + 77, 'w_i = 0 at init', { anchor: 'middle', size: 6, color: COL.mid });
  L.arrow(p, [cx0 + 10, gy + 90], [px + 7, py - 10], { stroke: COL.gate, width: 1.1, dash: '2.5 1.8', head: 3 });
  D.route(p, [[px + 10, py], [452, py]], { head: 'fill', stroke: ink, width: 1.2 });
  L.text(p, 480, py + 4, 'to next stage', { anchor: 'middle', size: 6.5 });
  L.text(p, 477, py + 40, 'y = α_i · BiMamba(x)', { anchor: 'middle', size: 6.5, color: COL.mam });
  L.text(p, 477, py + 50, '+ (1 − α_i) · MRF(x)', { anchor: 'middle', size: 6.5, color: COL.cnn });
}
