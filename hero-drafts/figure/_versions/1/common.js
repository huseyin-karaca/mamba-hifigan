import * as D from '../engine/cs.js';
export { D };
export const COL = {
  cnn: '#0072B2', cnnT: '#D3E5F3', mam: '#D55E00', mamT: '#F9DCC8',
  gate: '#009E73', gateT: '#CFEBDF', grey: '#EDEDED', ink: '#2b2b2b', mid: '#7a7a7a',
};
// plausible mel-like image: slowly gliding harmonics (structure first), no noise
export function melData(rows, cols, seed = 1) {
  const d = [];
  for (let r = 0; r < rows; r++) {
    const row = [];
    for (let c = 0; c < cols; c++) {
      const t = c / cols;
      const f0 = 0.16 + 0.05 * Math.sin(t * 5.2 + seed) - 0.04 * t;
      const y = 1 - r / rows;
      let v = 0;
      for (let k = 1; k <= 5; k++) {
        const dd = (y - f0 * k) / 0.022;
        v += Math.exp(-dd * dd) / (0.7 + 0.4 * k);
      }
      v *= 0.55 + 0.45 * Math.sin(t * 9 + seed) ** 2;
      row.push(Math.min(1, v + 0.06 * Math.exp(-y * 3)));
    }
    d.push(row);
  }
  return d;
}
export const MEL_PAL = ['#F7F4EF', '#F3D9B1', '#E8A55B', '#B5532A', '#5C1F18'];
export function mel(L, p, x, y, w, rows = 14, cols = 24, seed = 1) {
  return L.pixelImage(p, x, y, w, melData(rows, cols, seed), { palette: MEL_PAL, frame: true });
}
// speech-like waveform: fine carrier x slow syllable envelope (fine=detail level)
export function trace(L, p, x0, x1, y, { amp = 6, color = '#2b2b2b', fine = 1, width = 0.8, smooth = false, phase = 0 } = {}) {
  const n = 420, pts = [];
  for (let i = 0; i <= n; i++) {
    const s = i / n;
    const env = Math.abs(Math.sin(s * 9.4 + phase)) ** 0.8 * (0.55 + 0.45 * Math.sin(s * 3.1 + 1 + phase));
    const car = smooth ? 0.0 : (Math.sin(s * 520 * fine) + 0.4 * Math.sin(s * 910 * fine + 1));
    const slow = smooth ? Math.sin(s * 9.4 + phase) * (0.55 + 0.45 * Math.sin(s * 3.1 + 1 + phase)) : 0;
    pts.push([x0 + (x1 - x0) * s, y - amp * (smooth ? slow : env * car / 1.3)]);
  }
  return L.polyline(p, pts, { stroke: color, width, join: 'round' });
}
