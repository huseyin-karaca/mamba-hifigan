// Computes the hero's mel spectrogram and waveform from the project's own audio (LJSpeech, public domain).
//   node figure/data/make_audio_data.mjs
// mel: ground_truth.wav, HiFi-GAN/SpeechBrain LJSpeech settings (22050 Hz, n_fft 1024, hop 256, 80 Slaney mels,
//      0-8000 Hz, log-magnitude), the vocoder's actual input representation.
// wave: hybrid_gated.wav, the gated model's output for the same utterance, as a per-column min/max envelope.
import { readFileSync, writeFileSync } from 'node:fs';
import { dirname, join } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const AUDIO = join(here, '../../../docs/assets/audio/compare');

function readWav(path) {
  const b = readFileSync(path);
  let o = 12, fmt = null, data = null;
  while (o < b.length - 8) {
    const id = b.toString('ascii', o, o + 4), n = b.readUInt32LE(o + 4);
    if (id === 'fmt ') fmt = { ch: b.readUInt16LE(o + 10), sr: b.readUInt32LE(o + 12), bits: b.readUInt16LE(o + 22) };
    if (id === 'data') data = b.subarray(o + 8, o + 8 + n);
    o += 8 + n + (n & 1);
  }
  if (!fmt || fmt.bits !== 16 || fmt.ch !== 1) throw new Error('expected mono PCM16: ' + path);
  const x = new Float64Array(data.length / 2);
  for (let i = 0; i < x.length; i++) x[i] = data.readInt16LE(2 * i) / 32768;
  return { x, sr: fmt.sr };
}

function fft(re, im) { // in place, radix-2
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) {
    let bit = n >> 1;
    for (; j & bit; bit >>= 1) j ^= bit;
    j ^= bit;
    if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; }
  }
  for (let len = 2; len <= n; len <<= 1) {
    const a = -2 * Math.PI / len, wr = Math.cos(a), wi = Math.sin(a);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const ur = re[i + k], ui = im[i + k];
        const vr = re[i + k + len / 2] * cr - im[i + k + len / 2] * ci;
        const vi = re[i + k + len / 2] * ci + im[i + k + len / 2] * cr;
        re[i + k] = ur + vr; im[i + k] = ui + vi;
        re[i + k + len / 2] = ur - vr; im[i + k + len / 2] = ui - vi;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}

// Slaney mel scale (librosa default, as used by SpeechBrain's HiFi-GAN recipe)
const hz2mel = (f) => (f < 1000 ? f / (200 / 3) : 15 + Math.log(f / 1000) / (Math.log(6.4) / 27));
const mel2hz = (m) => (m < 15 ? m * (200 / 3) : 1000 * Math.exp((m - 15) * (Math.log(6.4) / 27)));

function melSpectrogram(x, sr, { nfft = 1024, hop = 256, nmels = 80, fmin = 0, fmax = 8000 } = {}) {
  const pad = nfft / 2, xp = new Float64Array(x.length + 2 * pad);
  for (let i = 0; i < xp.length; i++) { // reflect padding (center=True)
    let j = i - pad;
    if (j < 0) j = -j; if (j >= x.length) j = 2 * (x.length - 1) - j;
    xp[i] = x[j];
  }
  const win = Float64Array.from({ length: nfft }, (_, i) => 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / nfft));
  const nb = nfft / 2 + 1, fr = Math.floor((xp.length - nfft) / hop) + 1;
  const m0 = hz2mel(fmin), m1 = hz2mel(fmax);
  const pts = Array.from({ length: nmels + 2 }, (_, i) => mel2hz(m0 + ((m1 - m0) * i) / (nmels + 1)));
  const fb = Array.from({ length: nmels }, (_, m) => {
    const w = new Float64Array(nb), enorm = 2 / (pts[m + 2] - pts[m]);
    for (let k = 0; k < nb; k++) {
      const f = (k * sr) / nfft;
      const v = Math.max(0, Math.min((f - pts[m]) / (pts[m + 1] - pts[m]), (pts[m + 2] - f) / (pts[m + 2] - pts[m + 1])));
      w[k] = v * enorm;
    }
    return w;
  });
  const S = []; // S[mel][frame], log magnitude
  for (let m = 0; m < nmels; m++) S.push(new Float64Array(fr));
  const re = new Float64Array(nfft), im = new Float64Array(nfft);
  for (let t = 0; t < fr; t++) {
    for (let i = 0; i < nfft; i++) { re[i] = xp[t * hop + i] * win[i]; im[i] = 0; }
    fft(re, im);
    const mag = new Float64Array(nb);
    for (let k = 0; k < nb; k++) mag[k] = Math.hypot(re[k], im[k]);
    for (let m = 0; m < nmels; m++) {
      let s = 0; const w = fb[m];
      for (let k = 0; k < nb; k++) s += w[k] * mag[k];
      S[m][t] = Math.log(Math.max(s, 1e-5));
    }
  }
  return { S, frames: fr };
}

const gt = readWav(join(AUDIO, 'ground_truth.wav'));
const { S, frames } = melSpectrogram(gt.x, gt.sr);
// trim leading/trailing near-silence so the image is all speech
const energy = Array.from({ length: frames }, (_, t) => S.reduce((s, row) => s + row[t], 0) / S.length);
const emax = Math.max(...energy), thr = emax - 4.5;
let t0 = energy.findIndex((e) => e > thr), t1 = frames - 1 - [...energy].reverse().findIndex((e) => e > thr);
t0 = Math.max(0, t0 - 3); t1 = Math.min(frames - 1, t1 + 3);
// normalise to [0,1] over a fixed dynamic range below the peak (dB-like display range)
let peak = -Infinity;
for (const row of S) for (let t = t0; t <= t1; t++) peak = Math.max(peak, row[t]);
const range = 9.5; // natural-log units, about 41 dB
const mel = [];
for (let m = S.length - 1; m >= 0; m--) { // row 0 = highest band (image top)
  const row = [];
  for (let t = t0; t <= t1; t++) row.push(+Math.max(0, Math.min(1, (S[m][t] - (peak - range)) / range)).toFixed(4));
  mel.push(row);
}
const seconds = ((t1 - t0 + 1) * 256) / gt.sr;

const out = readWav(join(AUDIO, 'hybrid_gated.wav'));
// same utterance span: find the matching segment by energy onset in the generated audio
const absx = out.x.map(Math.abs), smooth = 512;
let on = 0;
for (let i = 0; i < absx.length; i += 64) {
  let s = 0; for (let j = i; j < Math.min(absx.length, i + smooth); j++) s += absx[j];
  if (s / smooth > 0.02) { on = Math.max(0, i - 256 * 3); break; }
}
const len = Math.round(seconds * out.sr), seg = out.x.subarray(on, Math.min(out.x.length, on + len));
const cols = 360, per = seg.length / cols, wave = [];
let amax = 0;
for (let c = 0; c < cols; c++) {
  let lo = 0, hi = 0;
  for (let i = Math.floor(c * per); i < Math.floor((c + 1) * per); i++) { lo = Math.min(lo, seg[i]); hi = Math.max(hi, seg[i]); }
  wave.push([lo, hi]); amax = Math.max(amax, -lo, hi);
}
const wn = wave.map(([lo, hi]) => [+(lo / amax).toFixed(4), +(hi / amax).toFixed(4)]);

writeFileSync(join(here, 'mel.json'), JSON.stringify({ source: 'docs/assets/audio/compare/ground_truth.wav (LJSpeech)', sr: gt.sr, nfft: 1024, hop: 256, nmels: 80, fmax: 8000, frames: [t0, t1], seconds, rangeLog: range, mel }));
writeFileSync(join(here, 'wave.json'), JSON.stringify({ source: 'docs/assets/audio/compare/hybrid_gated.wav (gated model output)', sr: out.sr, startSample: on, seconds: seg.length / out.sr, cols, envelope: wn }));
console.log(`mel ${mel.length}x${mel[0].length} frames ${t0}-${t1} (${seconds.toFixed(2)} s); wave ${cols} cols from sample ${on}`);
