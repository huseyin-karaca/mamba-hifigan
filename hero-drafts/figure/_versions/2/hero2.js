// Hero v2 (revised 2026-11-10): TTS context strip with a real mel and waveform -> zoom into one upsampling stage.
// Inside the stage, one shared time axis: BiMamba's two scan streams above it (whole-sequence context),
// MRF's three nested receptive-field funnels below it (bounded windows), mixed by a learned scalar gate.
// makeDraw({lanes: 'B'}) draws the alternative lane glyphs of draft B (figure/hero2_b.js); see docs/choice.md.
import { D } from './common.js';
export const figure = { width: 160, height: 108.6, name: 'hero2' };

// site palette (docs/stylesheets/extra.css): "the palette of the site and of every figure"
export const C = {
  ink: '#1c2230', ink2: '#434a5a', muted: '#6b7180', faint: '#9ba1ad', line: '#c9ccd3', grid: '#e1e4ea', wash: '#f6f7f9',
  mam: '#23977f', mamD: '#156b59', mamL: '#cde9e2', mamT: '#ebf5f2', mamE: '#d3ebe5',
  cnn: '#b67b0c', cnnD: '#7f5507', cnnL: '#f1ddb4', cnnT: '#fbf5e9', cnnE: '#f1e3c6',
  gate: '#7a5bd1', gateD: '#5a3db3', gateT: '#f2eefc', gateL: '#cfc6ee',
};
// mel colour map: magma, the spectrogram convention
const MAGMA = ['#000004', '#140e36', '#3b0f70', '#641a80', '#8c2981', '#b73779', '#de4968', '#f7705c', '#fe9f6d', '#fecf92', '#fcfdbf'];
// unicode maths italics, set in Latin Modern Math
const M = { a: '𝛼', s: '𝜎', w: '𝑤', i: '𝑖', k: '𝑘' };

async function siteFont(L) {
  document.head.insertAdjacentHTML('beforeend', `<style>
@font-face{font-family:'Schibsted Grotesk';src:url('/fonts/SchibstedGrotesk-VF.ttf') format('truetype');font-weight:400 900;font-style:normal}
@font-face{font-family:'Schibsted Grotesk';src:url('/fonts/SchibstedGrotesk-Italic-VF.ttf') format('truetype');font-weight:400 900;font-style:italic}</style>`);
  await Promise.all([400, 500, 600, 700].map((w) => document.fonts.load(`${w} 10px "Schibsted Grotesk"`)));
  await document.fonts.load('10px "Latin Modern Math"');
  L.STYLE.font.family = "'Schibsted Grotesk', Arial, sans-serif";
}
const arrowR = (p, pts, color, width = 0.9) => D.route(p, pts, { head: 'fill', stroke: color, width, headLength: 3.6, headWidth: 3 });

export function makeDraw(opts = {}) {
  return async function draw(F, L) {
    await siteFont(L);
    const mel = await (await fetch('/figure/data/mel.json')).json();
    const wav = await (await fetch('/figure/data/wave.json')).json();
    const p = F.root, MATH = L.STYLE.font.latex;
    const math = (x, y, s, o = {}) => L.text(p, x, y, s, { family: MATH, size: 8, ...o });
    L.rect(p, 0, 0, L.mm(figure.width), L.mm(figure.height), { fill: '#fff' });

    // ======================= top strip: a TTS system =======================
    const tc = 38;
    D.box(p, 6, tc - 15, 34, 30, { label: 'Text', fill: '#fff', stroke: C.ink2, width: 0.8, r: 4, size: 8.5, color: C.ink });
    arrowR(p, [[40, tc], [49, tc]], C.ink2);
    L.rect(p, 49, tc - 15, 75, 30, { fill: C.wash, stroke: C.ink2, width: 0.8, r: 4 });
    L.text(p, 86.5, tc - 1.5, 'Acoustic model', { anchor: 'middle', size: 8, color: C.ink, weight: 500 });
    L.text(p, 86.5, tc + 8.2, 'e.g. FastSpeech 2', { anchor: 'middle', size: 7, color: C.muted });
    arrowR(p, [[124, tc], [132, tc]], C.ink2);
    // mel spectrogram: real, 80 bands x 167 frames, one vector cell per bin
    const mx = 132, mw = 79, mh = (mw * mel.mel.length) / mel.mel[0].length, my = tc - mh / 2;
    L.pixelImage(p, mx, my, mw, mel.mel, { palette: MAGMA, skipBelow: -1, overlap: 0.35 });
    L.rect(p, mx, my, mw, mh, { fill: 'none', stroke: C.ink, width: 0.5 });
    L.text(p, mx + mw / 2, my - 3.4, 'mel spectrogram', { anchor: 'middle', size: 7.5, color: C.ink2 });
    arrowR(p, [[mx + mw, tc], [220, tc]], C.ink2);
    // generator card with its four upsampling stages, each with its own gate scalar
    const gx = 220, gy = 14, gw = 118, gh = 48;
    const chipX = (i) => gx + 9 + i * 26.5, cw = 22, chy = gy + 15, chh = 15;
    L.flare(p, [chipX(1) - 1, chipX(1) + cw + 1, gy + gh], [14, 439, 84], { fill: '#edf0f4', k: 0.62 });
    L.rect(p, gx, gy, gw, gh, { fill: '#fff', stroke: C.ink2, width: 1, r: 5 });
    L.text(p, gx + gw / 2, gy + 10, 'HiFi-GAN generator', { anchor: 'middle', size: 8, weight: 600, color: C.ink });
    ['↑8', '↑8', '↑2', '↑2'].forEach((u, i) => {
      const hl = i === 1, x = chipX(i);
      if (i < 3) L.line(p, [x + cw, chy + chh / 2], [chipX(i + 1), chy + chh / 2], { stroke: C.line, width: 0.7 });
      L.rect(p, x, chy, cw, chh, { fill: hl ? '#fff' : C.wash, stroke: hl ? C.ink : C.line, width: hl ? 1.2 : 0.7, r: 3 });
      L.text(p, x + cw / 2, chy + chh / 2, u, { anchor: 'middle', valign: 'middle', size: 7.5, color: hl ? C.ink : C.muted, weight: hl ? 600 : 400 });
      L.text(p, x + cw / 2, gy + gh - 4.6, `${M.a}_{${i + 1}}`, { anchor: 'middle', size: 7.5, family: MATH, color: C.gate });
    });
    arrowR(p, [[gx + gw, tc], [348, tc]], C.ink2);
    // waveform: the gated model's output for the same utterance, min/max envelope
    const wx0 = 350, wx1 = 447, amp = 17, E = wav.envelope, n = E.length;
    const up = E.map(([, hi], c) => [wx0 + ((wx1 - wx0) * c) / (n - 1), tc - amp * Math.max(hi, 0.012)]);
    const lo = E.map(([l], c) => [wx0 + ((wx1 - wx0) * c) / (n - 1), tc - amp * Math.min(l, -0.012)]).reverse();
    L.polygon(p, [...up, ...lo], { fill: C.ink2, stroke: C.ink2, width: 0.25 });
    L.text(p, (wx0 + wx1) / 2, my - 3.4, 'waveform', { anchor: 'middle', size: 7.5, color: C.ink2 });

    // ======================= the stage card =======================
    const sx0 = 6, sx1 = 447, sy0 = 84, sy1 = 302;
    L.rect(p, sx0, sy0, sx1 - sx0, sy1 - sy0, { fill: '#fafbfc', stroke: C.grid, width: 0.9, r: 8 });
    L.text(p, 14, sy0 + 11.5, 'Upsampling stage *i*', { size: 8, weight: 600, color: C.ink2 });

    const N = 15, step = 14.2, X = (i) => 80 + i * step, T = 7, xt = X(T);
    const ax0 = 72, ax1 = 288;                 // the shared time axis
    const sy = 186, srow = 4, sch = 3, sBot = sy + srow * sch;
    L.rect(p, 12, 100, 278, sy - 1.5 - 100, { fill: C.mamT, stroke: C.mamE, width: 0.6, r: 6 });
    L.rect(p, 12, sBot + 1.2, 278, 296 - sBot - 1.2, { fill: C.cnnT, stroke: C.cnnE, width: 0.6, r: 6 });

    // ---- stage input: a feature map on the shared axis, fading at both ends (the sequence goes on) ----
    const scol = 45, scw = (ax1 - ax0) / scol, greys = ['#eef0f3', '#d3d7de', '#aeb4bf', '#7f8696', '#596071'];
    for (let c = 0; c < scol; c++) for (let r = 0; r < srow; r++) {
      const u = c / scol, v = 0.5 + 0.28 * Math.sin(7.1 * u + 1.7 * r) + 0.18 * Math.sin(23 * u - 2.3 * r + 0.6) + 0.1 * Math.cos(41 * u * (r + 1));
      const fade = Math.min(1, (c + 0.5) / 6, (scol - c - 0.5) / 6);
      L.rect(p, ax0 + c * scw + 0.25, sy + r * sch + 0.25, scw - 0.5, sch - 0.5, { fill: L.ramp(greys, Math.max(0, Math.min(1, v))), opacity: fade });
    }
    D.box(p, 13, 181, 49, 22, { label: 'Transposed\nconv ↑', fill: '#fff', stroke: C.ink2, width: 0.8, r: 4, size: 7, color: C.ink, lineGap: 1.08 });
    arrowR(p, [[62, 192], [70.5, 192]], C.ink2);

    // ---- BiMamba ----
    L.text(p, 18, 116, 'BiMamba', { size: 10, weight: 700, color: C.mam });
    L.text(p, 18, 126.5, 'global context', { size: 7.5, color: C.mamD });
    const yOut = 114;
    if (opts.lanes === 'B') mambaCells(L, p, { X, N, T, xt, ax0, ax1, sy, yOut });
    else mambaStreams(L, p, { X, N, T, xt, ax0, ax1, sy, yOut });

    // ---- MRF ----
    L.text(p, 18, 280, 'MRF', { size: 10, weight: 700, color: C.cnn });
    L.text(p, 18, 290.5, 'local context', { size: 7.5, color: C.cnnD });
    const yOutC = 282;
    if (opts.lanes === 'B') mrfRows(L, p, { X, N, T, xt, sBot, yOutC, MATH });
    else mrfFunnels(L, p, { X, N, T, xt, sBot, yOutC, MATH });

    // ======================= fusion and the learnable gate =======================
    const fx = 334, ox = 389, oy = 192;
    arrowR(p, [[X(N - 1) + 5, yOut], [fx - 4.6, yOut]], C.mam, 1.1);
    arrowR(p, [[X(N - 1) + 5, yOutC], [fx - 4.6, yOutC]], C.cnn, 1.1);
    D.opNode(p, fx, yOut, 4.6, { op: '×', fill: '#fff', stroke: C.mamD, width: 0.9 });
    D.opNode(p, fx, yOutC, 4.6, { op: '×', fill: '#fff', stroke: C.cnnD, width: 0.9 });
    math(fx + 7, yOut - 5, `${M.a}_{${M.i}}`, { size: 9, color: C.gateD });
    math(fx + 7, yOutC + 11, `1 − ${M.a}_{${M.i}}`, { size: 9, color: C.gateD });
    arrowR(p, [[fx + 4.6, yOut], [ox, yOut], [ox, oy - 5.4]], C.mam, 1.1);
    arrowR(p, [[fx + 4.6, yOutC], [ox, yOutC], [ox, oy + 5.4]], C.cnn, 1.1);
    D.opNode(p, ox, oy, 5.4, { op: '+', fill: '#fff', stroke: C.ink, width: 1 });
    arrowR(p, [[ox + 5.4, oy], [441, oy]], C.ink, 1.1);
    L.text(p, 441, oy + 10.5, 'to stage *i* + 1', { anchor: 'end', size: 7, color: C.muted });

    const gx0 = 299, gx1 = 369, gy0 = 127, gy1 = 269, gcx = (gx0 + gx1) / 2;
    L.rect(p, gx0, gy0, gx1 - gx0, gy1 - gy0, { fill: C.gateT, stroke: C.gate, width: 1, r: 6 });
    D.route(p, [[gcx, gy0], [gcx, yOut + 5.2]], { head: 'fill', stroke: C.gate, width: 0.9, dash: '2 1.4', headLength: 3, headWidth: 2.6 });
    D.route(p, [[gcx, gy1], [gcx, yOutC - 5.2]], { head: 'fill', stroke: C.gate, width: 0.9, dash: '2 1.4', headLength: 3, headWidth: 2.6 });
    L.text(p, gcx, gy0 + 13, 'Learnable gate', { anchor: 'middle', size: 8, weight: 700, color: C.gateD });
    // sigma curve with the initial point w_i = 0 -> alpha_i = 1/2
    const px0 = gx0 + 11, px1 = gx1 - 8, py0 = gy0 + 23, py1 = gy0 + 59, sig = (w) => 1 / (1 + Math.exp(-w));
    const PX = (w) => px0 + ((w + 6) / 12) * (px1 - px0), PY = (a) => py1 - a * (py1 - py0);
    L.line(p, [px0, PY(1)], [px1, PY(1)], { stroke: C.gateL, width: 0.5, dash: '1.2 1' });
    L.line(p, [px0, PY(0)], [px1, PY(0)], { stroke: C.gate, width: 0.6 });
    L.line(p, [PX(0), py0 - 1], [PX(0), py1], { stroke: C.gateL, width: 0.5 });
    L.polyline(p, L.sample((w) => PY(sig(w)), -6, 6, 80).map(([w, y]) => [PX(w), y]), { stroke: C.gate, width: 1.2, join: 'round' });
    L.line(p, [px0, PY(0.5)], [PX(0), PY(0.5)], { stroke: C.gate, width: 0.5, dash: '1.2 1' });
    L.circle(p, PX(0), PY(0.5), 2.3, { fill: C.gate, stroke: '#fff', width: 0.8 });
    for (const [v, s] of [[1, '1'], [0.5, '½'], [0, '0']]) math(px0 - 1.8, PY(v) + 0.3, s, { size: 7, anchor: 'end', valign: 'middle', color: C.muted });
    math(PX(0), py1 + 7.5, '0', { size: 7, anchor: 'middle', color: C.muted });
    math(px1 + 1, py1 + 7.5, `${M.w}_{${M.i}}`, { size: 7.5, anchor: 'end', color: C.gateD });
    math(gcx, gy0 + 84, `${M.a}_{${M.i}} = ${M.s}(${M.w}_{${M.i}})`, { size: 9.5, anchor: 'middle', color: C.gateD });
    // mix bar at initialisation: an even split between the two branches
    const bx0 = gx0 + 9, bx1 = gx1 - 9, by = gy0 + 96, bh = 6, bmid = (bx0 + bx1) / 2;
    L.rect(p, bx0, by, bmid - bx0, bh, { fill: C.mam });
    L.rect(p, bmid, by, bx1 - bmid, bh, { fill: C.cnn });
    L.circle(p, bmid, by + bh / 2, 3.7, { fill: '#fff', stroke: C.gateD, width: 1 });
    L.text(p, bx0, by + bh + 8, 'SSM', { size: 7, weight: 600, color: C.mamD });
    L.text(p, bx1, by + bh + 8, 'CNN', { size: 7, weight: 600, color: C.cnnD, anchor: 'end' });
    D.richRow(p, gcx, gy0 + 127, [{ s: `${M.w}_{${M.i}} = 0`, family: MATH, size: 8, color: C.ink2 }, { s: 'at init', size: 7, color: C.ink2, gap: 2.4 }], { anchor: 'middle' });
  };
}
export default makeDraw();

// ---------- draft A lanes ----------
// BiMamba: one band, two lanes. The shade along a lane is how much of the sequence that scan's state has absorbed,
// so the two opposite ramps sum to the whole sequence at every frame; the two states meet at frame t (+, LN, y_t).
function mambaStreams(L, p, { X, N, T, xt, ax0, ax1, sy, yOut }) {
  const bf = 147, bm = 155.5, bb = 164;
  const gF = L.linearGradient(p, [[0, C.mamL, 0], [0.07, C.mamL, 1], [0.93, C.mam, 1], [1, C.mam, 0]], 0);
  const gB = L.linearGradient(p, [[0, C.mam, 0], [0.07, C.mam, 1], [0.93, C.mamL, 1], [1, C.mamL, 0]], 0);
  L.rect(p, ax0, bf, ax1 - ax0, bm - bf, { fill: gF });
  L.rect(p, ax0, bm, ax1 - ax0, bb - bm, { fill: gB });
  const gS = L.linearGradient(p, [[0, '#fff', 0], [0.1, '#fff', 1], [0.9, '#fff', 1], [1, '#fff', 0]], 0);
  L.rect(p, ax0, bm - 0.3, ax1 - ax0, 0.6, { fill: gS });
  for (let i = 0; i < N - 1; i++) {
    const xm = (X(i) + X(i + 1)) / 2;
    L.chevron(p, [xm + 1.5, (bf + bm) / 2], [1, 0], { len: 2.3, spread: 0.75, stroke: '#fff', width: 0.75 });
    L.chevron(p, [xm - 1.5, (bm + bb) / 2], [-1, 0], { len: 2.3, spread: 0.75, stroke: '#fff', width: 0.75 });
  }
  L.text(p, ax0 - 3, (bf + bm) / 2, 'forward →', { anchor: 'end', valign: 'middle', size: 7, color: C.mamD });
  L.text(p, ax0 - 3, (bm + bb) / 2 + 0.6, '← backward', { anchor: 'end', valign: 'middle', size: 7, color: C.mamD });
  for (let i = 0; i < N; i++) {
    const hl = i === T;
    L.line(p, [X(i), sy - 0.3], [X(i), bb + 0.3], { stroke: hl ? C.ink2 : '#b9bec8', width: hl ? 0.8 : 0.5 });
    if (!hl) L.line(p, [X(i), bf - 0.3], [X(i), yOut + 2.3], { stroke: C.mam, width: 0.5, opacity: 0.4 });
    L.circle(p, X(i), yOut, hl ? 3 : 2.1, { fill: hl ? C.mam : '#fff', stroke: hl ? C.mamD : C.mam, width: hl ? 0.9 : 0.6 });
  }
  D.opNode(p, xt, bm, 4.8, { op: '+', fill: '#fff', stroke: C.mamD, width: 0.9 });
  L.line(p, [xt, bm - 4.8], [xt, 138], { stroke: C.mamD, width: 0.9 });
  L.rect(p, xt - 9.5, 129.5, 19, 8.5, { fill: '#fff', stroke: C.mamD, width: 0.8, r: 4.25 });
  L.text(p, xt, 133.9, 'LN', { anchor: 'middle', valign: 'middle', size: 6.5, color: C.mamD, weight: 600 });
  L.line(p, [xt, 129.5], [xt, yOut + 3], { stroke: C.mamD, width: 0.9 });
}
// MRF: three nested receptive-field funnels, one per ResBlock (k = 3, 7, 11), each six conv layers with
// dilations 1,1,3,1,5,1; the width at each level is the exact receptive field there (25, 73, 121 frames at the input).
function mrfFunnels(L, p, { X, N, T, xt, sBot, yOutC, MATH }) {
  const fy0 = sBot + 1.2, fy1 = 254, nl = 6, fs = 132 / 121;   // pt per frame
  const widths = (k) => [12, 11, 10, 7, 6, 1, 0].map((m) => 1 + m * (k - 1));
  const fills = { 11: '#f6e7c9', 7: '#ecd29e', 3: '#dcab4f' };
  for (const k of [11, 7, 3]) {
    const ws = widths(k), ys = ws.map((_, j) => fy0 + ((fy1 - fy0) * j) / nl);
    const left = ws.map((w, j) => [xt - (w * fs) / 2, ys[j]]), right = ws.map((w, j) => [xt + (w * fs) / 2, ys[j]]).reverse();
    L.polygon(p, [...left, ...right], { fill: fills[k], stroke: C.cnn, width: 0.6, join: 'round' });
    for (let j = 1; j < nl; j++) L.line(p, [xt - (ws[j] * fs) / 2 + 0.4, ys[j]], [xt + (ws[j] * fs) / 2 - 0.4, ys[j]], { stroke: '#fff', width: 0.5, opacity: 0.9 });
  }
  const kx = { 11: xt - (fs * (121 + 73)) / 4, 7: xt - (fs * (73 + 25)) / 4, 3: xt };
  for (const k of [11, 7, 3]) L.text(p, kx[k], fy0 + 7.6, String(k), { anchor: 'middle', size: 7.5, family: MATH, color: C.cnnD });
  D.richRow(p, xt - (121 * fs) / 2 - 3, fy0 + 7.6, [{ s: 'kernel', size: 7, color: C.cnnD }, { s: M.k, family: MATH, size: 8, color: C.cnnD, gap: 2 }], { anchor: 'end' });
  L.line(p, [xt, fy1], [xt, 258], { stroke: C.cnnD, width: 0.9 });
  L.rect(p, xt - 12, 258, 24, 8.5, { fill: '#fff', stroke: C.cnnD, width: 0.8, r: 4.25 });
  L.text(p, xt, 262.4, 'mean', { anchor: 'middle', valign: 'middle', size: 6.5, color: C.cnnD, weight: 600 });
  L.line(p, [xt, 266.5], [xt, yOutC - 3], { stroke: C.cnnD, width: 0.9 });
  outRow(L, p, X, N, T, yOutC, C.cnn, C.cnnD);
}
function outRow(L, p, X, N, T, y, c, cd) {
  for (let i = 0; i < N; i++) {
    const hl = i === T;
    L.circle(p, X(i), y, hl ? 3 : 2.1, { fill: hl ? c : '#fff', stroke: hl ? cd : c, width: hl ? 0.9 : 0.6 });
  }
}

// ---------- draft B lanes ----------
// BiMamba as two chains of discrete state vectors (3 cells each), forward above backward, joined at t.
function mambaCells(L, p, { X, N, T, xt, sy, yOut }) {
  const rows = [{ y: 141, dir: 1 }, { y: 155, dir: -1 }], cw = 6, ch = 10.5;
  for (let i = 0; i < N; i++) {
    const hl = i === T;
    L.line(p, [X(i), sy - 0.3], [X(i), rows[1].y + ch], { stroke: hl ? C.ink2 : '#b9bec8', width: hl ? 0.8 : 0.5 });
    if (!hl) L.line(p, [X(i), rows[0].y], [X(i), yOut + 2.3], { stroke: C.mam, width: 0.5, opacity: 0.4 });
  }
  rows.forEach(({ y, dir }, ri) => {
    for (let i = 0; i < N; i++) {
      const f = dir > 0 ? i / (N - 1) : 1 - i / (N - 1);
      for (let q = 0; q < 3; q++) {
        const v = Math.max(0, Math.min(1, 0.15 + 0.8 * f + 0.12 * Math.sin(3.1 * i + 2 * q + ri)));
        L.rect(p, X(i) - cw / 2, y + q * (ch / 3), cw, ch / 3, { fill: L.ramp([C.mamL, C.mam, C.mamD], v), stroke: '#fff', width: 0.35 });
      }
      L.rect(p, X(i) - cw / 2, y, cw, ch, { fill: 'none', stroke: C.mamD, width: 0.5, r: 1 });
      if (i < N - 1) {
        const a = dir > 0 ? [X(i) + cw / 2 + 0.6, y + ch / 2] : [X(i + 1) - cw / 2 - 0.6, y + ch / 2];
        const b = dir > 0 ? [X(i + 1) - cw / 2 - 0.4, y + ch / 2] : [X(i) + cw / 2 + 0.4, y + ch / 2];
        D.route(p, [a, b], { head: 'fill', stroke: C.mam, width: 0.6, headLength: 2.2, headWidth: 2 });
      }
    }
  });
  L.text(p, 69, rows[0].y + ch / 2, 'forward →', { anchor: 'end', valign: 'middle', size: 7, color: C.mamD });
  L.text(p, 69, rows[1].y + ch / 2, '← backward', { anchor: 'end', valign: 'middle', size: 7, color: C.mamD });
  L.rect(p, xt - 5.2, rows[0].y - 2, 10.4, rows[1].y + ch - rows[0].y + 4, { fill: 'none', stroke: C.mamD, width: 0.9, r: 2.5 });
  D.opNode(p, xt, 132, 4, { op: '+', fill: '#fff', stroke: C.mamD, width: 0.9 });
  L.line(p, [xt, rows[0].y - 2], [xt, 136], { stroke: C.mamD, width: 0.9 });
  L.rect(p, xt - 9.5, 117.5, 19, 8.5, { fill: '#fff', stroke: C.mamD, width: 0.8, r: 4.25 });
  L.text(p, xt, 121.9, 'LN', { anchor: 'middle', valign: 'middle', size: 6.5, color: C.mamD, weight: 600 });
  L.line(p, [xt, 128], [xt, 126], { stroke: C.mamD, width: 0.9 });
  for (let i = 0; i < N; i++) {
    const hl = i === T;
    L.circle(p, X(i), yOut, hl ? 3 : 2.1, { fill: hl ? C.mam : '#fff', stroke: hl ? C.mamD : C.mam, width: hl ? 0.9 : 0.6 });
  }
  L.line(p, [xt, 117.5], [xt, yOut + 3], { stroke: C.mamD, width: 0.9 });
}
// MRF as three rows, one per ResBlock: its receptive-field window on the time axis with the dilated taps of
// its widest layer, then the three outputs averaged.
function mrfRows(L, p, { X, N, T, xt, sBot, yOutC, MATH }) {
  const fs = 132 / 121, ys = { 3: 208, 7: 222, 11: 236 }, bh = 8;
  for (const k of [3, 7, 11]) {
    const w = (1 + 12 * (k - 1)) * fs, y = ys[k];
    L.rect(p, xt - w / 2, y, w, bh, { fill: '#f1ddb4', stroke: C.cnn, width: 0.6, r: 2 });
    for (let j = -(k - 1) / 2; j <= (k - 1) / 2; j++) L.circle(p, xt + j * 5 * fs, y + bh / 2, 1.1, { fill: C.cnnD });
    L.text(p, xt - w / 2 - 3, y + bh / 2, `${M.k} = ${k}`, { anchor: 'end', valign: 'middle', size: 7.5, family: MATH, color: C.cnnD });
    L.line(p, [xt + w / 2, y + bh / 2], [xt + 72, y + bh / 2], { stroke: C.cnn, width: 0.6 });
  }
  L.line(p, [xt + 72, ys[3] + bh / 2], [xt + 72, 262], { stroke: C.cnn, width: 0.6 });
  L.rect(p, xt + 60, 256, 24, 8.5, { fill: '#fff', stroke: C.cnnD, width: 0.8, r: 4.25 });
  L.text(p, xt + 72, 260.4, 'mean', { anchor: 'middle', valign: 'middle', size: 6.5, color: C.cnnD, weight: 600 });
  D.route(p, [[xt + 60, 260.25], [xt, 260.25], [xt, yOutC - 3]], { stroke: C.cnnD, width: 0.9, radius: 3 });
  outRow(L, p, X, N, T, yOutC, C.cnn, C.cnnD);
}
