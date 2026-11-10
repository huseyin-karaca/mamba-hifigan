// Hero v2 (revision 3, 2026-11-10).
// Top: five matching cards, Text -> Acoustic model -> Mel spectrogram -> HiFi-GAN generator -> Waveform (real mel and
// waveform). Zoom into one upsampling stage: the transposed conv produces x, a fork hands a copy of x to each of two
// separate branch modules, BiMamba (whole-sequence scans) and MRF (bounded receptive fields), drawn on the same
// template (input row at the top, output row at the bottom, same time columns), mixed by a learned scalar gate.
// Previous versions: figure/_versions/1 (user's hero2), figure/_versions/2 (shared-strip layout).
import { D } from './common.js';
export const figure = { width: 160, height: 111.5, name: 'hero2' };

// site palette (docs/stylesheets/extra.css): "the palette of the site and of every figure"
export const C = {
  ink: '#1c2230', ink2: '#434a5a', muted: '#6b7180', faint: '#9ba1ad', line: '#c9ccd3', grid: '#e1e4ea', wash: '#f6f7f9',
  mam: '#23977f', mamD: '#156b59', mamL: '#cde9e2', mamT: '#ebf5f2', mamE: '#cfe7e0',
  cnn: '#b67b0c', cnnD: '#7f5507', cnnL: '#f1ddb4', cnnT: '#fbf5e9', cnnE: '#efdfbf',
  gate: '#7a5bd1', gateD: '#5a3db3', gateT: '#f2eefc', gateL: '#cfc6ee',
};
// mel colour map: magma, the spectrogram convention
const MAGMA = ['#000004', '#140e36', '#3b0f70', '#641a80', '#8c2981', '#b73779', '#de4968', '#f7705c', '#fe9f6d', '#fecf92', '#fcfdbf'];
// unicode maths italics, set in Latin Modern Math
const M = { a: '𝛼', s: '𝜎', w: '𝑤', i: '𝑖', k: '𝑘', x: '𝑥' };

async function siteFont(L) {
  document.head.insertAdjacentHTML('beforeend', `<style>
@font-face{font-family:'Schibsted Grotesk';src:url('/fonts/SchibstedGrotesk-VF.ttf') format('truetype');font-weight:400 900;font-style:normal}
@font-face{font-family:'Schibsted Grotesk';src:url('/fonts/SchibstedGrotesk-Italic-VF.ttf') format('truetype');font-weight:400 900;font-style:italic}</style>`);
  await Promise.all([400, 500, 600, 700].map((w) => document.fonts.load(`${w} 10px "Schibsted Grotesk"`)));
  await document.fonts.load('10px "Latin Modern Math"');
  L.STYLE.font.family = "'Schibsted Grotesk', Arial, sans-serif";
}
const arrowR = (p, pts, color, width = 0.9, o = {}) => D.route(p, pts, { head: 'fill', stroke: color, width, headLength: 3.6, headWidth: 3, ...o });

export default async function draw(F, L) {
  await siteFont(L);
  const mel = await (await fetch('/figure/data/mel.json')).json();
  const wav = await (await fetch('/figure/data/wave.json')).json();
  const p = F.root, MATH = L.STYLE.font.latex;
  const math = (x, y, s, o = {}) => L.text(p, x, y, s, { family: MATH, size: 8, ...o });
  L.rect(p, 0, 0, L.mm(figure.width), L.mm(figure.height), { fill: '#fff' });

  // ======================= top strip: five matching cards =======================
  const ty0 = 8, th = 56, cy = 42;                       // card top, card height, content centre line
  const cards = { text: [6, 54], am: [63, 131], mel: [140, 222], gen: [231, 349], wave: [358, 447] };
  const titles = { text: 'Text', am: 'Acoustic model', mel: 'Mel spectrogram', gen: 'HiFi-GAN generator', wave: 'Waveform' };
  const gx = cards.gen[0], chipX = (i) => gx + 9 + i * 26.5, cw = 22, chy = cy - 11, chh = 15, sy0 = 88;
  L.flare(p, [chipX(1) - 1, chipX(1) + cw + 1, ty0 + th], [14, 439, sy0], { fill: '#edf0f4', k: 0.62 });
  for (const [key, [x0, x1]] of Object.entries(cards)) {
    L.rect(p, x0, ty0, x1 - x0, th, { fill: '#fff', stroke: C.ink2, width: 0.8, r: 5 });
    L.text(p, (x0 + x1) / 2, ty0 + 11.5, titles[key], { anchor: 'middle', size: 7.5, weight: 600, color: C.ink });
  }
  const keys = Object.keys(cards);
  for (let i = 0; i < keys.length - 1; i++) arrowR(p, [[cards[keys[i]][1], cy], [cards[keys[i + 1]][0], cy]], C.ink2);
  // Text: a neutral text-lines glyph (the clip's transcript is not recorded in the project)
  [[34, -7], [29, 0], [20, 7]].forEach(([w, dy]) => L.rect(p, 14, cy + dy - 1.4, w, 2.8, { fill: C.line, r: 1.4 }));
  // Acoustic model: generic, as in the user's hero2
  L.text(p, 97, cy + 2.5, 'e.g. FastSpeech 2', { anchor: 'middle', size: 7, color: C.muted });
  // Mel spectrogram: real, 80 bands x 167 frames, one vector cell per bin
  const mw = 70, mh = (mw * mel.mel.length) / mel.mel[0].length, mx = (cards.mel[0] + cards.mel[1] - mw) / 2, my = cy + 0.8 - mh / 2;
  L.pixelImage(p, mx, my, mw, mel.mel, { palette: MAGMA, skipBelow: -1, overlap: 0.35 });
  L.rect(p, mx, my, mw, mh, { fill: 'none', stroke: C.ink, width: 0.5 });
  // HiFi-GAN generator: four upsampling stages, each with its own gate scalar
  ['↑8', '↑8', '↑2', '↑2'].forEach((u, i) => {
    const hl = i === 1, x = chipX(i);
    if (i < 3) L.line(p, [x + cw, chy + chh / 2], [chipX(i + 1), chy + chh / 2], { stroke: C.line, width: 0.7 });
    L.rect(p, x, chy, cw, chh, { fill: hl ? '#fff' : C.wash, stroke: hl ? C.ink : C.line, width: hl ? 1.2 : 0.7, r: 3 });
    L.text(p, x + cw / 2, chy + chh / 2, u, { anchor: 'middle', valign: 'middle', size: 7.5, color: hl ? C.ink : C.muted, weight: hl ? 600 : 400 });
    L.text(p, x + cw / 2, cy + 13.5, `${M.a}_{${i + 1}}`, { anchor: 'middle', size: 7.5, family: MATH, color: C.gate });
  });
  // Waveform: the gated model's output for the same utterance, min/max envelope
  const wx0 = cards.wave[0] + 6, wx1 = cards.wave[1] - 6, amp = 15, E = wav.envelope, n = E.length, wy = cy + 0.8;
  const up = E.map(([, hi], c) => [wx0 + ((wx1 - wx0) * c) / (n - 1), wy - amp * Math.max(hi, 0.012)]);
  const lo = E.map(([l], c) => [wx0 + ((wx1 - wx0) * c) / (n - 1), wy - amp * Math.min(l, -0.012)]).reverse();
  L.polygon(p, [...up, ...lo], { fill: C.ink2, stroke: C.ink2, width: 0.25 });

  // ======================= the stage card =======================
  L.rect(p, 6, sy0, 441, 222, { fill: '#fafbfc', stroke: C.grid, width: 0.9, r: 8 });
  L.text(p, 14, sy0 + 12, 'Upsampling stage *i*', { size: 8, weight: 600, color: C.ink2 });

  const N = 13, X = (i) => 120 + i * 13.4, T = 6, xt = X(T), ax0 = 114, ax1 = 284;   // shared time columns
  // two separate branch cards of equal height, mirrored about the fork at G: BiMamba reads upward, MRF downward,
  // so both outputs sit at the outer edges and the gate between them is centred on the stage
  const bx0 = 70, bx1 = 286, G = 203, H = 92, gap = 5, mam = [G - gap - H, G - gap], cnn = [G + gap, G + gap + H];
  L.rect(p, bx0, mam[0], bx1 - bx0, H, { fill: C.mamT, stroke: C.mamE, width: 0.7, r: 6 });
  L.rect(p, bx0, cnn[0], bx1 - bx0, H, { fill: C.cnnT, stroke: C.cnnE, width: 0.7, r: 6 });
  // row headers, outside the cards
  const mc = (mam[0] + mam[1]) / 2, cc = (cnn[0] + cnn[1]) / 2;
  L.text(p, 12, mc - 10, 'BiMamba', { size: 10, weight: 700, color: C.mam });
  L.text(p, 12, mc + 1, 'global context', { size: 7, color: C.mamD });
  L.text(p, 12, cc - 1.5, 'MRF', { size: 10, weight: 700, color: C.cnn });
  L.text(p, 12, cc + 9.5, 'local context', { size: 7, color: C.cnnD });

  // previous stage's gated output -> this stage's transposed conv -> x, shared by both branches
  const inM = mam[1] - 9 - 12, inC = cnn[0] + 7, busX = 64;          // tops of the two input rows
  ['tensor from', 'conv ↑ of', 'stage *i* − 1'].forEach((t, j) => L.text(p, 12, G + (j - 1) * 8.5 - 0.5, t, { valign: 'middle', size: 6.5, color: C.muted }));
  L.line(p, [58, G], [busX, G], { stroke: C.ink2, width: 0.9 });
  arrowR(p, [[busX, G], [busX, inM + 6], [ax0 - 0.5, inM + 6]], C.ink2, 0.9, { radius: 3 });
  arrowR(p, [[busX, G], [busX, inC + 6], [ax0 - 0.5, inC + 6]], C.ink2, 0.9, { radius: 3 });
  L.circle(p, busX, G, 1.7, { fill: C.ink2 });
  math(busX + 3.5, G + 0.3, M.x, { size: 8.5, valign: 'middle', color: C.ink2 });
  featureRow(L, p, ax0, ax1, inM);
  featureRow(L, p, ax0, ax1, inC);

  const yOut = mam[0] + 8, yOutC = cnn[1] - 8;
  mambaStreams(L, p, { X, N, T, xt, ax0, ax1, inTop: inM, yOut });
  mrfFunnels(L, p, { X, N, T, xt, top: inC + 12, yOutC, MATH });

  // ======================= fusion and the learnable gate =======================
  const fx = 330, ox = 392, oy = (yOut + yOutC) / 2;
  arrowR(p, [[X(N - 1) + 5, yOut], [fx - 4.6, yOut]], C.mam, 1.1);
  arrowR(p, [[X(N - 1) + 5, yOutC], [fx - 4.6, yOutC]], C.cnn, 1.1);
  D.opNode(p, fx, yOut, 4.6, { op: '×', fill: '#fff', stroke: C.mamD, width: 0.9 });
  D.opNode(p, fx, yOutC, 4.6, { op: '×', fill: '#fff', stroke: C.cnnD, width: 0.9 });
  math(fx + 7, yOut - 5, `${M.a}_{${M.i}}`, { size: 9, color: C.gateD });
  math(fx + 7, yOutC + 11, `1 − ${M.a}_{${M.i}}`, { size: 9, color: C.gateD });
  arrowR(p, [[fx + 4.6, yOut], [ox, yOut], [ox, oy - 5.4]], C.mam, 1.1);
  arrowR(p, [[fx + 4.6, yOutC], [ox, yOutC], [ox, oy + 5.4]], C.cnn, 1.1);
  D.opNode(p, ox, oy, 5.4, { op: '+', fill: '#fff', stroke: C.ink, width: 1 });
  const ex = 437, lx = (ox + 5.4 + ex) / 2;
  arrowR(p, [[ox + 5.4, oy], [ex, oy]], C.ink2, 0.9);
  L.text(p, lx, oy - 16, 'input to', { anchor: 'middle', size: 7, color: C.muted });
  L.text(p, lx, oy - 7, 'stage *i* + 1', { anchor: 'middle', size: 7, color: C.muted });

  // gate card centred on the stage, equal links up to the alpha node and down to the (1 - alpha) node
  const gh = 92, gx0 = 298, gx1 = 362, gy0 = oy - gh / 2, gy1 = oy + gh / 2, gcx = (gx0 + gx1) / 2;
  L.rect(p, gx0, gy0, gx1 - gx0, gh, { fill: C.gateT, stroke: C.gate, width: 1, r: 6 });
  D.route(p, [[gcx, gy0], [gcx, yOut + 5.2]], { head: 'fill', stroke: C.gate, width: 0.9, dash: '2 1.4', headLength: 2.8, headWidth: 2.6 });
  D.route(p, [[gcx, gy1], [gcx, yOutC - 5.2]], { head: 'fill', stroke: C.gate, width: 0.9, dash: '2 1.4', headLength: 2.8, headWidth: 2.6 });
  L.text(p, gcx, gy0 + 13, 'Learnable gate', { anchor: 'middle', size: 7.5, weight: 700, color: C.gateD });
  // sigma curve with the initial point w_i = 0 -> alpha_i = 1/2
  const px0 = gx0 + 11, px1 = gx1 - 7, py0 = gy0 + 23, py1 = gy0 + 59, sig = (w) => 1 / (1 + Math.exp(-w));
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
  math(gcx, gy0 + 80, `${M.a}_{${M.i}} = ${M.s}(${M.w}_{${M.i}})`, { size: 9.5, anchor: 'middle', color: C.gateD });
}

// the branch input x: a feature map on the time axis, fading at both ends (the sequence goes on)
function featureRow(L, p, ax0, ax1, y, { rows = 4, ch = 3, cols = 42 } = {}) {
  const cw = (ax1 - ax0) / cols, greys = ['#eef0f3', '#d3d7de', '#aeb4bf', '#7f8696', '#596071'];
  for (let c = 0; c < cols; c++) for (let r = 0; r < rows; r++) {
    const u = c / cols, v = 0.5 + 0.28 * Math.sin(7.1 * u + 1.7 * r) + 0.18 * Math.sin(23 * u - 2.3 * r + 0.6) + 0.1 * Math.cos(41 * u * (r + 1));
    const fade = Math.min(1, (c + 0.5) / 6, (cols - c - 0.5) / 6);
    L.rect(p, ax0 + c * cw + 0.25, y + r * ch + 0.25, cw - 0.5, ch - 0.5, { fill: L.ramp(greys, Math.max(0, Math.min(1, v))), opacity: fade });
  }
}

// BiMamba, read bottom to top: input row, two scan lanes (forward above, backward below) whose shade is how much of the
// sequence that scan's state has absorbed, so the two ramps sum to the whole sequence at every frame; at frame t the two
// states are added, layer-normed, and give y_t in the output row at the top.
function mambaStreams(L, p, { X, N, T, xt, ax0, ax1, inTop, yOut }) {
  const bb = inTop - 12, bm = bb - 8.5, bf = bm - 8.5, ln = bf - 14.5;
  const gF = L.linearGradient(p, [[0, C.mamL, 0], [0.07, C.mamL, 1], [0.93, C.mam, 1], [1, C.mam, 0]], 0);
  const gB = L.linearGradient(p, [[0, C.mam, 0], [0.07, C.mam, 1], [0.93, C.mamL, 1], [1, C.mamL, 0]], 0);
  const gS = L.linearGradient(p, [[0, '#fff', 0], [0.1, '#fff', 1], [0.9, '#fff', 1], [1, '#fff', 0]], 0);
  for (let i = 0; i < N; i++) {
    const hl = i === T;
    L.line(p, [X(i), inTop + 0.3], [X(i), bb - 0.3], { stroke: hl ? C.ink2 : '#b9bec8', width: hl ? 0.8 : 0.5 });
    if (!hl) L.line(p, [X(i), bf + 0.3], [X(i), yOut + 2.3], { stroke: C.mam, width: 0.5, opacity: 0.4 });
  }
  L.rect(p, ax0, bf, ax1 - ax0, bm - bf, { fill: gF });
  L.rect(p, ax0, bm, ax1 - ax0, bb - bm, { fill: gB });
  L.rect(p, ax0, bm - 0.3, ax1 - ax0, 0.6, { fill: gS });
  for (let i = 0; i < N - 1; i++) {
    const xm = (X(i) + X(i + 1)) / 2;
    L.chevron(p, [xm + 1.5, (bf + bm) / 2], [1, 0], { len: 2.3, spread: 0.75, stroke: '#fff', width: 0.75 });
    L.chevron(p, [xm - 1.5, (bm + bb) / 2], [-1, 0], { len: 2.3, spread: 0.75, stroke: '#fff', width: 0.75 });
  }
  L.text(p, ax0 - 3, (bf + bm) / 2, 'forward →', { anchor: 'end', valign: 'middle', size: 7, color: C.mamD });
  L.text(p, ax0 - 3, (bm + bb) / 2 + 0.6, '← backward', { anchor: 'end', valign: 'middle', size: 7, color: C.mamD });
  D.opNode(p, xt, bm, 4.8, { op: '+', fill: '#fff', stroke: C.mamD, width: 0.9 });
  L.line(p, [xt, bm - 4.8], [xt, ln + 8.5], { stroke: C.mamD, width: 0.9 });
  L.rect(p, xt - 9.5, ln, 19, 8.5, { fill: '#fff', stroke: C.mamD, width: 0.8, r: 4.25 });
  L.text(p, xt, ln + 4.4, 'LN', { anchor: 'middle', valign: 'middle', size: 6.5, color: C.mamD, weight: 600 });
  L.line(p, [xt, ln], [xt, yOut + 3], { stroke: C.mamD, width: 0.9 });
  outRow(L, p, X, N, T, yOut, C.mam, C.mamD);
}

// MRF, read top to bottom: input row, then three nested receptive-field funnels, one per ResBlock (k = 3, 7, 11), each six
// conv layers with dilations 1,1,3,1,5,1; the width at each level is the exact receptive field there (25, 73, 121 frames
// at the input); the three outputs are averaged into y_t.
function mrfFunnels(L, p, { X, N, T, xt, top, yOutC, MATH }) {
  const fy0 = top + 1.2, fy1 = yOutC - 21, nl = 6, fs = 116 / 121;   // pt per frame
  const widths = (k) => [12, 11, 10, 7, 6, 1, 0].map((m) => 1 + m * (k - 1));
  const fills = { 11: '#f6e7c9', 7: '#ecd29e', 3: '#dcab4f' };
  for (const k of [11, 7, 3]) {
    const ws = widths(k), ys = ws.map((_, j) => fy0 + ((fy1 - fy0) * j) / nl);
    const left = ws.map((w, j) => [xt - (w * fs) / 2, ys[j]]), right = ws.map((w, j) => [xt + (w * fs) / 2, ys[j]]).reverse();
    L.polygon(p, [...left, ...right], { fill: fills[k], stroke: C.cnn, width: 0.6, join: 'round' });
    for (let j = 1; j < nl; j++) L.line(p, [xt - (ws[j] * fs) / 2 + 0.4, ys[j]], [xt + (ws[j] * fs) / 2 - 0.4, ys[j]], { stroke: '#fff', width: 0.5, opacity: 0.9 });
  }
  const kx = { 11: xt - (fs * (121 + 73)) / 4, 7: xt - (fs * (73 + 25)) / 4, 3: xt };
  for (const k of [11, 7, 3]) L.text(p, kx[k], fy0 + 7.4, String(k), { anchor: 'middle', size: 7.5, family: MATH, color: C.cnnD });
  D.richRow(p, xt - (121 * fs) / 2 - 3, fy0 + 7.4, [{ s: 'kernel', size: 7, color: C.cnnD }, { s: M.k, family: MATH, size: 8, color: C.cnnD, gap: 2 }], { anchor: 'end' });
  const mean = fy1 + 3.5;
  L.line(p, [xt, fy1], [xt, mean], { stroke: C.cnnD, width: 0.9 });
  L.rect(p, xt - 12, mean, 24, 8.5, { fill: '#fff', stroke: C.cnnD, width: 0.8, r: 4.25 });
  L.text(p, xt, mean + 4.4, 'mean', { anchor: 'middle', valign: 'middle', size: 6.5, color: C.cnnD, weight: 600 });
  L.line(p, [xt, mean + 8.5], [xt, yOutC - 3], { stroke: C.cnnD, width: 0.9 });
  outRow(L, p, X, N, T, yOutC, C.cnn, C.cnnD);
}

function outRow(L, p, X, N, T, y, c, cd) {
  for (let i = 0; i < N; i++) {
    const hl = i === T;
    L.circle(p, X(i), y, hl ? 3 : 2.1, { fill: hl ? c : '#fff', stroke: hl ? cd : c, width: hl ? 0.9 : 0.6 });
  }
}
