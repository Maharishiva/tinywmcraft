// Play inside the tinywmcraft world model of Craftax-Classic, in the browser on WebGPU. Mounts into #play.
// Icons are Craftax's own sprites (github.com/MichaelTMatthews/Craftax, MIT, (c) 2024 Michael Matthews).
// The network runs on wgr.js, a small WebGPU runtime in plain JS and WGSL made for this one graph (no onnxruntime, no wasm:
// WebKit needed 0.5 to 1.1 GB to compile onnxruntime's 17 to 27 MB of wasm, close to all iOS allows a page)
import { create } from './wgr.js';
import { inventory, facing, fresh } from './hud.js';
import { tutorial } from './tutorial.js';
import { asleep } from './sleep.js';
import { noise, PAST, NOISE_BUF, RNOISE } from './seeded.js';

const X_HANDLE = '';   // the X handle the diamond caption ends with, when set
const GITHUB = 'github.com/Maharishiva/tinywmcraft';
const MEDIABUNNY = 'https://cdn.jsdelivr.net/npm/mediabunny@1.61.5/dist/bundles/mediabunny.min.mjs';
const LINK = ['MAHARISHIVA.GITHUB.IO/', 'TINYWMCRAFT'];   // on top of every saved clip
const SITE = 'https://maharishiva.github.io/tinywmcraft/';
// The model the site plays: play.wgr made by export_web.py and convert_web.py from the best checkpoint, at a pinned
// revision so the cached copy stays valid. It is the slot graph (inputs pos, prev, kmask) with int8 matmul and conv
// weights (convert_web.py --int8, 12 MB instead of 23), sha256 9eccfecf.
const HF_MODEL = 'https://huggingface.co/maharishiva/tinywmcraft/resolve/0c96165fbfe3b0303fcaee69ef569f16f4c8618f/web/play_int8.wgr';
const q = new URLSearchParams(location.search);
// ?model= loads another file; served from this machine, it plays docs/play.wgr, fetched fresh every time
const LOCAL = /^(localhost|127\.0\.0\.1|\[::1\])$/.test(location.hostname) || location.protocol === 'file:';
const MODEL = q.get('model') ?? (LOCAL || !HF_MODEL ? 'play.wgr' : HF_MODEL);
// Every world begins from a first frame the model imagines from noise at window slot 0, after a noop; the window fills
// from slot 0 as in training, then slides. The sampler's noise is seeded (seeded.js), a new random seed for every world,
// shown as four characters of SEED_ABC, most significant first (no I, L, O or U): ?seed=00JT (any case) opens world
// 602 again, the same first frame and, with the same keys, the same frames
const SEED_ABC = '0123456789ABCDEFGHJKMNPQRSTVWXYZ', SEEDS = 32 ** 4;   // seeds 0..1048575
const seedCode = s => [15, 10, 5, 0].map(b => SEED_ABC[(s >> b) & 31]).join('');
const parseSeed = c => {   // a seed code as its number; null if it is not one
  const s = (c ?? '').toUpperCase();
  return s.length === 4 && [...s].every(ch => SEED_ABC.includes(ch)) ? [...s].reduce((n, ch) => 32 * n + SEED_ABC.indexOf(ch), 0) : null;
};
const SEED = parseSeed(q.get('seed'));        // the world asked for; none or an invalid code: a random one
const REROLL = 20;                             // imagined first frames tried at most: the next seed until one is fresh
// The tutorial's world, the first one on a first visit (?tut=1 or ?tut=a, ?tut=b: again, in look a or b): seed 00JT,
// bright day, nothing held, a short row of trees straight up from two tiles away over open grass; the walkthrough is
// done in 5 steps at best (up, chop, up, chop, table; a second chop at the first tree gives nothing). Picked from the
// imagined first frames of 1500 seeds. Pick another one if the model changes.
const TUT_SEED = 602;
const TUT = q.get('tut'), LOOK = TUT === 'b' ? 'b' : 'a';
const SLEEP_MAX = 400;                         // steps skipped in one sleep at most
const SLEEP_STEP_MS = 60, SLEEP_SHOW_MS = 1500; // skipped steps at most ~16 a second; the sleep banner stays at least 1.5 s
const DIAMOND_SURE = 600;                      // a diamond is read when the next best count is this much worse (hud.js)
const DONE_THRESHOLD = 0.5;                   // done probability that ends the episode
const MAX_SCALE = 7;                          // largest pixel scale
// a run goes on until it dies; clips keep its last CLIP steps (older ones go, so memory stays flat) at FPS, faster
// when the frames would take longer than CLIP_S seconds (3000 steps: 50 fps)
const CLIP = 3000, FPS = 12, CLIP_S = 60;
const L = 31 * 512, RGB = 63 * 63 * 3;        // one 31-frame latent window, one frame
const TOP = 7, H = TOP + 88, MAP = 49;        // the score strip, the frame, the keys under it; the map part of the frame
const BG = '#0a0a0a', FG = '#f0f0ee', ORANGE = '#ff6a13', KEY = '#1c1c1e', EDGE = '#3a3a3e', CAP = '#aaaaa8';
// index = action; e.code, so the keys work on any keyboard layout
const CODES = ['KeyQ', 'KeyA', 'KeyD', 'KeyW', 'KeyS', 'Space', 'Tab', 'KeyR', 'KeyT', 'KeyF', 'KeyP',
  'Digit1', 'Digit2', 'Digit3', 'Digit4', 'Digit5', 'Digit6'];
const CAPS = ['Q', 'A', 'D', 'W', 'S', 'ACTION', 'TAB', 'R', 'T', 'F', 'P', '1', '2', '3', '4', '5', '6'];   // ACTION: space
const ARROWS = { ArrowLeft: 1, ArrowRight: 2, ArrowUp: 3, ArrowDown: 4 };
const NAMES = ['noop', 'left', 'right', 'up', 'down', 'action', 'sleep', 'place stone', 'place table', 'place furnace',
  'place plant', 'make wood pickaxe', 'make stone pickaxe', 'make iron pickaxe', 'make wood sword',
  'make stone sword', 'make iron sword'];
const TOUCH = matchMedia('(pointer: coarse)').matches;
// the keyboard of play.gif under the frame in clips: action, x, y, width of each key (-1 shows any other key pressed), and
// the reward square's x; ACTION exactly as wide as the two boxes above it. Its top row is one pixel under the frame (KB);
// the clip's frames end with it (CH), then the seed and step line
const BOARD = [[3, 13, 0, 9], [1, 3, 10, 9], [4, 13, 10, 9], [2, 23, 10, 9], [5, 33, 10, 28], [-1, 33, 0, 18]], RX = 52;
const KB = TOP + 64, CH = KB + 19;
// the same keys as buttons, the size of the action keys, each at row, column, columns spanned: ACTION about two keys wide,
// up to the frame's edge, beside the frame; three under it on the action keys' six columns, WASD on the right on touch screens
const GRID = {
  row: { W: [1, 2, 1], A: [2, 1, 1], S: [2, 2, 1], D: [2, 3, 1], ACTION: [2, 4, 2] },
  left: { W: [1, 2, 1], A: [2, 1, 1], S: [2, 2, 1], D: [2, 3, 1], ACTION: [2, 4, 3] },
  right: { W: [1, 5, 1], A: [2, 4, 1], S: [2, 5, 1], D: [2, 6, 1], ACTION: [2, 1, 3] },
};
const MOVE = [[3, 'W'], [1, 'A'], [4, 'S'], [2, 'D'], [5, 'ACTION']];
const READY = TOUCH ? 'Tap the map to walk' : 'WASD to move, SPACE to act';   // after the tutorial
const ICONS = [8, 7, 9, 10, 6, 0, 11, 12, 13, 14, 15, 16];
const FONT = {  // 3x5 pixel font of docs/make_play_gif.py
  A: '010101111101101', B: '110101110101110', C: '011100100100011', D: '110101101101110', E: '111100110100111',
  F: '111100110100100', P: '110101110100100', Q: '010101101110011', R: '110101110101101', S: '011100010001110',
  T: '111010010010010', W: '101101111111101', 1: '010110010010111', 2: '111001111100111', 3: '111001111001111',
  4: '101101111001001', 5: '111100111001111', 6: '111100111101111', '+': '000010111010000',
  G: '011100101101011', H: '101101111101101', I: '111010010010111', M: '101111111101101', N: '110101101101101',
  O: '010101101101010', U: '101101101101111', V: '101101101101010', Y: '101101010010010', '.': '000000000000010',
  '/': '001001010100100', 0: '111101101101111', 7: '111001001001001', 8: '111101111101111', 9: '111101111001111',
  ' ': '000000000000000', '!': '010010010000010', L: '100100100100111', '?': '111001010000010',
  J: '001001001101010', K: '101101110101101', X: '101101010101101', Z: '111001010100111',
};
// the clip goes to the system share sheet on phones and tablets that share files; computers get a post on X and a download
const FILES = TOUCH && !!navigator.canShare?.({ files: [new File([''], 'x.mp4', { type: 'video/mp4' })] });
const IOS = /iPhone|iPad|iPod/.test(navigator.userAgent) || (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);

const root = document.getElementById('play');
// a pixel-font label; u: screen pixels per font pixel (a game pixel on the movement keys, half of one elsewhere)
const pix = (s, color = CAP, cls = 'px', u = '--fp') => `<img class="${cls}" src="${label(s, color)}" style="height:calc(var(${u}) * 5)" alt="">`;
// the action keys show their icons, NOOP its name in the pixel font; ACTION and NOOP get their own font pixel (fit)
const icon = a => a === 0 ? pix('NOOP', CAP, 'px', '--np') : `<img src="icons/${NAMES[a].replaceAll(' ', '_')}.png" alt="${NAMES[a]}">`;
root.innerHTML = `<div class="screen"><div class="view"><canvas width="63" height="${TOP + 63}"></canvas>
<canvas class="meta" width="47" height="11"></canvas><div class="msg"></div><div class="banner" hidden></div></div>
<div class="keys">${MOVE.map(([a, k]) => `<button class="key" data-a="${a}" data-k="${k}" title="${a === 5 ? 'action (space)' : NAMES[a]}">` +
  `${pix(k, CAP, 'px', a === 5 ? '--ap' : '--px')}</button>`).join('')}</div>
<div class="st"></div></div>
<div class="side"><div class="acts">${ICONS.map(a => `<button class="key" data-a="${a}" title="${NAMES[a]}">` +
  `${icon(a)}${pix(CAPS[a], CAP, 'px k')}</button>`).join('')}
</div><div class="btns"><button class="key" data-a="save" title="save the clip">${pix('SAVE CLIP')}</button>
<button class="key" data-a="share" title="share the clip">${pix('SHARE', ORANGE)}</button>
<button class="key" data-a="reset" title="new world">${pix('NEW WORLD')}</button></div></div>`;
const canvas = root.querySelector('canvas'), msg = root.querySelector('.msg'), st = root.querySelector('.st');
const keys = root.querySelector('.keys'), acts = root.querySelector('.acts'), btns = root.querySelector('.btns');
const banner = root.querySelector('.banner'), meta = root.querySelector('.meta');
const ctx = canvas.getContext('2d', { willReadFrequently: true });

// fill: frames in the window, the slot the next one goes to (31 once it is full)
let session, z, tau, act, fill = 0, mux, made, frames = [], tick = 0, score = 0, moves = 0, episode = 0;
let best = +localStorage.getItem('tinywmcraft.best') || 0;
let seed = 0, calls = 0;   // this world's seed; model calls in this world, the noise counter
let px = 1;   // screen pixels per game pixel (fit)
let busy = false, end = '', saving = false;   // end: '' while playing, 'dead' once the run is over; saving: someone waits
// for the clip, so its progress shows
// the tutorial: on a first visit or asked for with ?tut=; coach: the running one
let coach = null, learn = TUT !== null || !localStorage.getItem('tinywmcraft.tutorial');
// sleep: frames in a row that look asleep / awake, steps skipped so far (-1: not skipping); diamond: frames in a row that
// show one in the inventory, whether this run has had its diamond moment, and the step it came on
let sleepy = 0, awake = 0, skipped = -1, diamonds = 0, diamond = false, gemAt = 0, bannerOff = 0;

const status = (s, hot) => { st.textContent = s; st.className = hot ? 'st hot' : 'st'; };
const info = (s, hot) => TOUCH || status(s, hot);   // what each move did: on phones the space goes to the controls

function fit() {  // the largest integer scale that fits. Every key is one size, one game pixel apart: beside the game the action
  // keys span the map and the three buttons, stacked, end with the movement keys under the frame (63 + 1 + 11.5 + 1 + 11.5 = 88);
  // under the game everything sits on one grid of six columns
  const row = getComputedStyle(root).flexDirection === 'row', W = root.clientWidth, cols = row ? 3 : 6;
  const s = Math.max(1, Math.min(MAX_SCALE, Math.floor((W - (row ? 16 : 0)) / (row ? 99.5 : 63)), Math.floor((innerHeight - 80) / H)));
  const b = row ? (MAP - 3) * s / 4 : 58 * s / 6;
  px = s;
  root.querySelector('.screen').style.width = canvas.style.width = 63 * s + 'px';
  const grid = GRID[row ? 'row' : TOUCH ? 'right' : 'left'];
  Object.assign(keys.style, { gridTemplateColumns: row ? `repeat(4, ${b}px) ${59 * s - 4 * b}px` : `repeat(6, ${b}px)`,
    gridAutoRows: b + 'px', gap: s + 'px', marginTop: s + 'px' });
  for (const el of keys.querySelectorAll('.key')) {
    const [r, c, n] = grid[el.dataset.k];
    el.style.gridArea = `${r} / ${c} / span 1 / span ${n}`;
  }
  Object.assign(acts.style, { gridTemplateColumns: `repeat(${cols}, ${b}px)`, gridAutoRows: b + 'px', gap: s + 'px',
    marginTop: row ? TOP * s + 'px' : '' });
  Object.assign(btns.style, { gridTemplateColumns: row ? 3 * b + 2 * s + 'px' : `repeat(3, ${2 * b + s}px)`,
    gridAutoRows: (row ? 12 * s : b) + 'px', gap: s + 'px', marginTop: s + 'px' });   // 3 x 12 + 2 gaps = map to ACTION's bottom
  root.style.setProperty('--px', s + 'px');   // one game pixel: the keys' border, and the icons drawn at the game's tile size
  root.style.setProperty('--fp', Math.max(1, s >> 1) + 'px');   // one pixel of the font on the buttons
  root.style.setProperty('--np', Math.max(1, Math.floor(b / 20)) + 'px');   // NOOP: 15 font pixels in three quarters of a key
  const aw = keys.querySelector('[data-k="ACTION"]').getBoundingClientRect().width;   // ACTION: 23 font pixels, clear of the edge
  root.style.setProperty('--ap', Math.max(1, Math.min(s, Math.floor((aw - 4 * s) / 23))) + 'px');
  msg.style.padding = `${TOP * s}px 16px 0`;   // what shows over the game sits on the frame
  tagged();
  coach?.layout();
}

function hint() {  // the dimmed controls while the model loads
  root.classList.toggle('off', !session);
}

function learnt(early) {  // the tutorial is over: not again on this browser, the short hint instead
  localStorage.setItem('tinywmcraft.tutorial', early ? 'skipped' : 'done');
  coach = null;
  learn = false;
  if (!end) greet();
}

function teach() {  // the tutorial over this world, from its first frame
  const rgb = frames[0].rgb;
  coach = tutorial({ root, canvas, touch: TOUCH, look: LOOK, top: TOP, face: facing(rgb), onEnd: learnt });
  coach.begin(rgb, inventory(rgb));
  status('');
}

function text(s, x, y, color, c = ctx, u = 1) {  // the 3x5 pixel font, u screen pixels per font pixel; returns the end x
  c.fillStyle = color;
  for (const ch of s) {
    for (let i = 0; i < 15; i++) if (FONT[ch][i] === '1') c.fillRect(x + u * (i % 3), y + u * (i / 3 | 0), u, u);
    x += 4 * u;
  }
  return x;
}

function label(s, color = CAP) {  // lines of the pixel font, centred, as an image for the buttons
  const lines = s.split('\n'), c = document.createElement('canvas'), w = Math.max(...lines.map(l => 4 * l.length - 1));
  [c.width, c.height] = [w, 6 * lines.length - 1];
  lines.forEach((l, i) => text(l, (w - 4 * l.length + 1) >> 1, 6 * i, color, c.getContext('2d')));
  return c.toDataURL();
}

function head(c, f) {  // the score and the best score
  c.fillStyle = BG;
  c.fillRect(0, 0, 63, TOP);
  text(String(f.score), text('SCORE', 2, 1, CAP, c) + 2, 1, f.r > 0.5 ? ORANGE : FG, c);
  const right = `HI ${f.best}`;
  text(right, 62 - 4 * right.length, 1, CAP, c);
}

function tag(c, f, x0, x1, y, u = 1) {  // the line under the keys in clips: the seed on the left, the step on the right
  text(`SEED ${seedCode(f.seed)}`, x0, y, CAP, c, u);
  const m = `STEP ${f.moves}`;
  text(m, x1 - u * (4 * m.length - 1), y, CAP, c, u);
}

function tagged(f = frames.at(-1) ?? { seed: null, moves, score, best }) {  // the step and seed on the page: two lines,
  // the step number on top, the seed code under it, centred in the score strip between SCORE and HI, a font pixel of at
  // most half a game pixel
  const a = 24 + 4 * String(f.score).length, b = 62 - 4 * `HI ${f.best}`.length;   // the gap, in game pixels
  const room = Math.max(0, b - a - 2) * px, lines = [String(f.moves), f.seed === null ? '' : seedCode(f.seed)];
  const w = Math.max(...lines.map(l => 4 * l.length - 1)), fp = Math.max(1, Math.min(px >> 1, Math.floor(room / w)));
  [meta.width, meta.height] = [w, 11];
  const c = meta.getContext('2d');
  lines.forEach((l, i) => text(l, (w - 4 * l.length + 1) >> 1, 6 * i, CAP, c));
  Object.assign(meta.style, { width: w * fp + 'px', height: 11 * fp + 'px', left: (a + b) / 2 * px - w * fp / 2 + 'px',
    top: (TOP * px - 11 * fp) / 2 + 'px' });
}

function keyboard(c, a, r) {  // the keys under the frame in clips, the pressed one in orange; a square filled by the reward
  const y0 = KB;
  c.fillStyle = BG;
  c.fillRect(0, TOP + 63, 63, CH - TOP - 63);
  const other = a !== undefined && !BOARD.some(b => b[0] === a);
  for (const [k, x, y, w] of BOARD) {
    const on = k < 0 ? other : k === a, cap = k < 0 ? (other ? CAPS[a] : '') : CAPS[k];
    c.fillStyle = on ? ORANGE : EDGE;
    c.fillRect(x, y + y0, w, 9);
    c.fillStyle = on ? ORANGE : KEY;
    c.fillRect(x + 1, y + y0 + 1, w - 2, 7);
    text(cap, x + ((w - 4 * cap.length + 1) >> 1), y + y0 + 2, on ? BG : CAP, c);
  }
  const fill = Math.round(7 * Math.min(Math.max(r, 0), 1));
  c.fillStyle = EDGE;
  c.fillRect(RX, y0, 9, 9);
  c.fillStyle = KEY;
  c.fillRect(RX + 1, y0 + 1, 7, 7);
  c.fillStyle = r > 0.5 ? ORANGE : CAP;
  c.fillRect(RX + 1, y0 + 8 - fill, 7, fill);
}

function over(c, f) {  // the end of a run, Minecraft's red death screen: the title, the score
  const g = c.createLinearGradient(0, TOP, 0, TOP + 63), title = 'DEAD!';
  g.addColorStop(0, 'rgba(150, 20, 20, 0.42)');
  g.addColorStop(1, 'rgba(128, 40, 40, 0.6)');
  c.fillStyle = g;
  c.fillRect(0, TOP, 63, 63);
  const shadowed = (s, x, y, color, u = 1) => (text(s, x + 1, y + 1, 'rgba(10, 0, 0, 0.6)', c, u), text(s, x, y, color, c, u));
  shadowed(title, (63 - 8 * title.length + 2) >> 1, TOP + 10, FG, 2);
  const n = String(f.score), x = (63 - 24 - 4 * n.length + 1) >> 1;
  shadowed(n, shadowed('SCORE ', x, TOP + 24, FG), TOP + 24, ORANGE);
}

function paint(c, f, board = false) {  // one move: the score, the frame with a +1 for a reward over 0.5, the end screen, the keys
  const img = c.createImageData(63, 63);
  for (let i = 0; i < 63 * 63; i++) img.data.set([f.rgb[3 * i], f.rgb[3 * i + 1], f.rgb[3 * i + 2], 255], 4 * i);
  c.putImageData(img, 0, TOP);
  if (f.r > 0.5) text('+1', 55, TOP + 1, FG, c);
  head(c, f);
  if (board) keyboard(c, f.a, f.r);
  if (f.end) over(c, f);
}

function draw(rgb, a, r = 0) {  // shows a move and keeps it for the clip (the last CLIP): the frame, key and reward, not
  // the pixels drawn
  const f = { rgb, a, r, score, best, moves, end, diamond, gem: gemAt, seed };
  paint(ctx, f);
  for (const el of root.querySelectorAll('.key[data-a]')) el.classList.toggle('on', +el.dataset.a === a);   // the key pressed
  tick++;
  frames.push(f);
  if (frames.length > CLIP) frames.shift();
  tagged(f);
}

function card(c, f) {  // the last screen of every clip: the score to beat, or the step the diamond came on
  c.fillStyle = BG;
  c.fillRect(0, 0, 63, CH);
  const mid = (s, y, color, u = 1) => text(s, (63 - u * (4 * s.length - 1)) >> 1, y, color, c, u);
  const steps = n => `${n} STEP${n === 1 ? '' : 'S'}`;
  mid('MY SCORE', 14, CAP);
  mid(String(f.score), 24, ORANGE, 4);
  mid(`IN ${steps(f.moves)}`, 50, CAP);
  mid(f.diamond ? 'DIAMOND IN' : 'CAN YOU', 66, FG);
  mid(f.diamond ? steps(f.gem) : 'BEAT IT?', 73, f.diamond ? ORANGE : FG);
}

const randomSeed = () => crypto.getRandomValues(new Uint32Array(1))[0] % SEEDS;

function reset(s = randomSeed()) {  // a new world, seed s (random unless given): an empty window, dark until the model
  // imagines its first frame (dream)
  episode++;
  seed = s;
  calls = 0;
  z = new Float32Array(L);
  tau = new Float32Array(31);
  act = new Int32Array(32);
  fill = 0;
  frames = [];
  score = moves = gemAt = 0;
  end = '';
  msg.textContent = '';
  wake();
  sleepy = awake = diamonds = 0;
  diamond = false;
  gemOff();
  banner.hidden = true;
  paint(ctx, { rgb: new Uint8Array(RGB), score, best, moves, end, diamond });
  tagged({ seed: null, moves, score, best });
  if (session) begin().catch(fail);
}

async function begin() {  // a new world plays: its first frame imagined, then the tutorial or the hint
  if (!(await dream())) return;
  if (learn && episode === 1) teach();
  else if (!coach) greet();
}

// The first frame, imagined from noise at slot 0 after a noop, not a step. About 2% of imagined first frames look
// mid-game (less food or drink, an item held): until one is fresh (hud.js, 9/9/9/9 and nothing held) it is imagined
// again with the next seed, so the seed shown always gives this world straight away. Returns whether the world is
// still the same one.
async function dream() {
  busy = true;
  const ep = episode;
  act[31] = 0;
  try {
    for (let n = 1; ; n++) {
      const o = await run(session, feeds());
      if (ep !== episode) return false;
      if (n < REROLL && !fresh(o.rgb)) {
        seed = (seed + 1) % SEEDS;
        continue;
      }
      calls++;
      z.set(o.z, 0);
      tau[0] = 0.9;
      act[0] = 0;
      fill = 1;
      draw(o.rgb);
      return true;
    }
  } finally {
    if (ep === episode) busy = false;   // a new world's first frame may be on its way already
  }
}

let tips = 0;   // moves the short hint stays before the status line shows what each move did
function greet() {  // the short hint, big
  status(READY, true);
  st.classList.add('big');
  tips = 12;
}

function spot(inv) {  // the diamond moment: a diamond in the inventory on two frames in a row, once a run
  const d = inv.diamond;
  diamonds = d.n >= 1 && d.margin > DIAMOND_SURE ? diamonds + 1 : 0;
  if (diamonds >= 2 && !diamond) {
    diamond = frames.at(-1).diamond = true;   // the clip ends on it
    gemAt = frames.at(-1).gem = moves;
    banner.className = 'banner gem';   // words only, so taps on the map go through; the save button pulses
    banner.innerHTML = `<b>${pix('DIAMOND!', ORANGE, 'px', '--px')}</b><span>Save the clip</span>`;
    banner.hidden = false;
    btns.querySelector('[data-a="save"]').classList.add('glow');
    bannerOff = moves + 10;
  } else if (banner.classList.contains('gem') && moves >= bannerOff) gemOff();
}

function gemOff() {
  if (banner.classList.contains('gem')) banner.hidden = true;
  btns.querySelector('[data-a="save"]').classList.remove('glow');
}

function doze(rgb) {  // asleep: the sleep is skipped, NOOP after NOOP, until two frames are awake (the detector makes no
  // errors on real or model frames, so one asleep frame is enough; two awake ones ride over a one-frame flicker)
  if (asleep(rgb)) sleepy++, awake = 0;
  else awake++, sleepy = 0;
  if (skipped < 0 && sleepy >= 1 && !end) setTimeout(snooze);
}

async function snooze() {  // one step in flight at a time, as fast as the model runs, at most SLEEP_MAX steps
  if (skipped >= 0) return;
  skipped = 0;
  banner.className = 'banner zzz';
  banner.innerHTML = '<b>Sleeping… zzz</b><span>you wake when energy is full</span><button data-a="wake">wake me</button>';
  banner.hidden = false;
  const ep = episode, t0 = performance.now();
  while (skipped >= 0 && skipped < SLEEP_MAX && awake < 2 && !end && ep === episode) {
    while (busy) await new Promise(r => setTimeout(r, 5));
    if (skipped < 0 || ep !== episode) break;
    const t = performance.now();
    await step(0);
    if (skipped >= 0) skipped++;   // not if "wake me" came while the step ran
    await new Promise(r => setTimeout(r, SLEEP_STEP_MS - (performance.now() - t)));   // slow enough to see the night pass
  }
  await new Promise(r => setTimeout(r, SLEEP_SHOW_MS - (performance.now() - t0)));   // a short nap still shows its banner
  if (ep === episode) wake();
}

function wake() {  // skipping stops; stopped while still asleep ("wake me", the cap), asleep frames count again only after an
  // awake one
  if (skipped < 0) return;
  skipped = -1;
  if (awake < 2) sleepy = -1e9;
  if (banner.classList.contains('zzz')) banner.hidden = true;
}

function feeds() {  // the past fed corrupted to its signal levels tau, this world's noise for this call; the new frame
  // goes to slot fill (31 once the window is full): its position, the slot before it for the done head, and the past
  // slots it sees
  const g = (n, buf) => noise(seed, calls, buf, n);
  const eps = g(L, PAST), s = Math.min(fill, 31), pos = new Float32Array(32), prev = new Float32Array(31);
  const kmask = new Float32Array(32).fill(-1e30, s, 31);
  pos[s] = 1;
  if (s) prev[s - 1] = 1;
  return { past: z.map((v, i) => tau[i >> 9] * v + (1 - tau[i >> 9]) * eps[i]), tau, act, noise: g(512, NOISE_BUF), rnoise: g(512, RNOISE),
    pos, prev, kmask };
}

async function run(s, x) {  // one call of the world model: x's arrays in, the outputs out
  const o = await s.run(x);
  return { z: o.z, rgb: o.rgb, reward: o.reward[0], done: o.done[0] };
}

async function step(a) {  // results of a step started before a reset are dropped
  busy = true;
  const ep = episode;
  act[31] = a;
  try {
    const t0 = performance.now(), o = await run(session, feeds());
    if (ep !== episode) return;
    calls++;
    if (fill < 31) {   // the window fills: the new frame stays in its slot
      z.set(o.z, fill * 512);
      tau[fill] = 0.9;
      act[fill++] = a;
    } else {   // full: it slides
      z.copyWithin(0, 512);
      z.set(o.z, 30 * 512);
      tau.copyWithin(0, 1);
      tau[30] = 0.9;
      act.copyWithin(0, 1);
    }
    moves++;
    end = o.done > DONE_THRESHOLD ? 'dead' : '';
    const r = o.reward;
    score += r > 0.5;
    if (score > best) localStorage.setItem('tinywmcraft.best', best = score);
    draw(o.rgb, a, r);
    if (tips > 0) --tips || status('');   // the short hint goes after a few moves
    else info(`${NAMES[a]}, ${Math.round(performance.now() - t0)} ms on your GPU`, r > 0.5);
    const inv = inventory(o.rgb);
    coach?.frame({ rgb: o.rgb, a, r, inv, over: !!end });
    spot(inv);
    doze(o.rgb);
    if (!end) return;
    banner.hidden = true;
    msg.innerHTML = '<div class="end"><button data-a="save">save clip</button><button class="hot" data-a="share">share</button>' +
      '<button data-a="reset">new world</button></div>';
    clip().catch(() => {});   // encoded now, so share opens at once
  } finally {
    if (ep === episode) busy = false;   // a new world's first frame may be on its way already
  }
}

async function encode() {  // the run as an MP4: the link on top, the frames 8x nearest-neighbour, the seed and step line
  // just under the keys; at most a minute of frames, then the score card
  const fs = frames.slice(), d = new Date(), p2 = n => String(n).padStart(2, '0');   // play goes on while it encodes
  const name = `tinywmcraft-${fs.at(-1).diamond ? 'diamond' : 'score'}-${score}-${d.getFullYear()}${p2(d.getMonth() + 1)}${p2(d.getDate())}-` +
    `${p2(d.getHours())}${p2(d.getMinutes())}${p2(d.getSeconds())}.mp4`;
  const { Output, Mp4OutputFormat, BufferTarget, CanvasSource, QUALITY_HIGH } = await (mux ??= import(MEDIABUNNY));
  const G = 8, LINE = 5, U = 3, TAG = 3, out = document.createElement('canvas'), one = document.createElement('canvas');
  [out.width, out.height, one.width, one.height] = [63 * G, (LINE + CH + TAG) * G, 63, CH];
  const c = out.getContext('2d'), y = (LINE * G - 5 * U) >> 1, x = (out.width - U * (4 * LINK.join('').length - 1)) >> 1;
  const foot = (LINE + CH) * G, under = f => {   // the seed and the step, from the keys' left edge to their right one
    c.fillStyle = BG;
    c.fillRect(0, foot, out.width, TAG * G);
    tag(c, f, 3 * G, 61 * G, foot + 4, U);
  };
  c.imageSmoothingEnabled = false;
  c.fillStyle = BG;
  c.fillRect(0, 0, out.width, LINE * G);
  text(LINK[1], text(LINK[0], x, y, CAP, c, U), y, ORANGE, c, U);
  const output = new Output({ format: new Mp4OutputFormat(), target: new BufferTarget() });
  const source = new CanvasSource(out, { codec: 'avc', bitrate: QUALITY_HIGH });
  const fps = Math.max(FPS, Math.ceil(fs.length / CLIP_S)), oc = one.getContext('2d');
  output.addVideoTrack(source, { frameRate: fps });
  await output.start();
  for (const [i, f] of fs.entries()) {
    paint(oc, f, true);
    c.drawImage(one, 0, LINE * G, out.width, CH * G);
    under(f);
    await source.add(i / fps, i === fs.length - 1 ? 1.5 : 1 / fps);
    if (saving && i % 50 === 0) status(`making the clip: ${Math.round(100 * i / fs.length)}%`);
  }
  card(oc, fs.at(-1));
  c.drawImage(one, 0, LINE * G, out.width, CH * G);
  await source.add((fs.length - 1) / fps + 1.5, 2.5);
  await output.finalize();
  return new File([output.target.buffer], name, { type: 'video/mp4' });
}

const clip = () => (made?.[0] === tick ? made : (made = [tick, encode()]))[1];   // encoded once per frame shown

async function ready() {  // the clip, its progress shown while it encodes; none if the click asking for it has expired by then
  // (3000 steps take ~18 s on an M1 Pro; a click lets a page download or share for ~5 s; a late download Chrome blocks
  // without a word)
  status('making the clip');
  saving = true;
  try {
    var f = await clip();
  } finally {
    saving = false;
  }
  if (navigator.userActivation?.isActive !== false) return f;
  status('the clip is ready: press again', true);
}

function download(f) {
  const link = document.createElement('a');
  link.href = URL.createObjectURL(f);
  link.download = f.name;
  link.click();
}

async function save() {  // a download; on iPhones and iPads the share sheet instead, whose "Save Video" puts the clip in Photos
  if (IOS && FILES) return send();
  const f = await ready();
  if (!f) return;
  download(f);
  status('clip saved');
}

async function share() {  // phones: the clip to the share sheet, its caption to the clipboard; computers: a post on X with the
  // caption, the clip downloaded to attach
  const handle = X_HANDLE && ' @' + X_HANDLE.replace(/^@/, '');
  const text = diamond
    ? `I got a diamond inside a world model trained in under 10 minutes for $0.1 of compute. Every frame is generated ` +
      `by a neural network on my GPU. ${SITE} ${GITHUB}${handle}`
    : `I scored ${score} inside a world model trained in under 10 minutes for $0.1 of compute. ` +
      `Every frame is generated by a neural network on my GPU. Can you beat it? ${SITE}${handle}`;
  if (FILES) {
    navigator.clipboard?.writeText(text).catch(() => {});
    return send('caption copied: paste it with the clip');
  }
  const f = await ready();
  if (!f) return;
  download(f);   // before the post opens: opening it uses up the click
  open(`https://x.com/intent/post?text=${encodeURIComponent(text)}`, '_blank');
  status('clip saved: attach it to the post', true);
}

async function send(done = '') {  // the clip alone: given text as well, apps on iOS (Telegram) share only the text
  const f = await ready();
  if (!f) return;
  status('');
  try {
    await navigator.share({ files: [f] });
    status(done, !!done);
  } catch (e) {
    if (e.name === 'NotAllowedError') status('the clip is ready: press again', true);
    else if (e.name === 'AbortError') status('');
    else if (e.name === 'InvalidStateError') status('the share sheet is already open');
    else throw e;
  }
}

let stage = 'loading';   // what was going on, shown with an error
const fail = e => { msg.textContent = `${stage}: ${e.name && e.name !== 'Error' ? e.name + ': ' : ''}${e.message ?? e}`; status('error'); };

function press(a) {  // an action index, 'reset', 'save', 'share', 'wake' (stop skipping a sleep) or 'skip' (the tutorial)
  if (a === 'save' || a === 'share') {   // never blocked for long: a share sheet left open holds nothing
    if (frames.length) (a === 'save' ? save() : share()).catch(e => status(`no clip: ${e.message ?? e}`, true));
    return;
  }
  if (a === 'wake') return wake();
  if (a === 'skip') return coach?.end(true);
  if (!session) return;
  if (a === 'reset') {   // a new world also ends the tutorial
    coach?.end(true);
    return reset();
  }
  if (busy || end || skipped >= 0) return;   // while a sleep is skipped, only "wake me" acts
  step(a).catch(fail);
}

function loading(what, frac, note) {  // the screen over the game until it plays: what is loading, a progress bar
  msg.innerHTML = `<b>${what}</b><div class="prog${frac < 0 ? ' wait' : ''}"><i style="width:${frac < 0 ? 100 : 100 * frac}%"></i></div>` +
    `<span>${note}</span>`;
}

async function load(url, what, keep) {  // fetch with a progress bar; keep: from and into the browser's cache. Any call to the
  // cache can fail (Safari: TypeError "Internal error"); then it is a download, never an error
  const box = keep ? await self.caches?.open('tinywmcraft').catch(() => null) : null;
  const hit = await box?.match(url).then(r => r?.arrayBuffer()).catch(() => null);
  if (hit) return new Uint8Array(hit);
  loading(`loading the ${what}`, 0, 'connecting');
  const r = await fetch(url);
  if (!r.ok) throw new Error(`${url}: HTTP ${r.status}`);
  const total = +r.headers.get('content-length'), parts = [];
  let got = 0;
  for (const reader = r.body.getReader(); ;) {
    const { done, value } = await reader.read();
    if (done) break;
    parts.push(value);
    got += value.length;
    loading(`loading the ${what}`, got / total, `${(got / 1e6).toFixed(1)} / ${(total / 1e6).toFixed(1)} MB, then it runs on your GPU`);
  }
  const bytes = new Uint8Array(got);   // one copy, the parts go
  parts.reduce((at, x) => (bytes.set(x, at), at + x.length), 0);
  parts.length = 0;
  if (box) {   // the next visit loads it from here; older versions go
    try {
      for (const k of await box.keys()) if (k.url !== url) await box.delete(k);
      await box.put(url, new Response(bytes));
    } catch {}
  }
  return bytes;
}

async function main() {
  reset(SEED ?? (learn ? TUT_SEED : undefined));   // the first world: the one asked for, the tutorial's, or a random one
  stage = 'loading the model';
  const model = await load(MODEL, 'world model', !LOCAL && /^https:\/\//.test(MODEL));   // pinned URLs are cached
  loading('compiling it for your GPU', -1, 'no server, no queue: every frame is made on your machine');
  stage = 'compiling';
  const adapter = await navigator.gpu.requestAdapter();
  if (!adapter) throw new Error('no WebGPU adapter');
  const device = await adapter.requestDevice();
  device.lost.then(i => i.reason === 'destroyed' || fail(new Error(`the GPU was lost: ${i.message || i.reason}`)));
  const s = await create(device, model);   // every buffer, pipeline and bind group, once
  if (!s.meta.inputs.pos) throw new Error(`${MODEL} has no slot inputs: the player needs the slot graph (export_web.py)`);
  stage = 'first run';
  await run(s, feeds());   // warm-up: the first run waits for the GPU to finish compiling
  stage = 'move';
  session = s;
  msg.textContent = '';
  hint();
  await begin();
  mux = import(MEDIABUNNY);   // the MP4 encoder, loaded in the background so saving a clip is quick
}

root.onmousedown = e => e.preventDefault();   // buttons never take focus, so space stays DO
root.onclick = e => {
  const b = e.target.closest('button');
  if (b) return press(/^\d+$/.test(b.dataset.a) ? +b.dataset.a : b.dataset.a);
  if (e.target !== canvas) return;
  const s = canvas.clientWidth / 63, x = e.offsetX / s, y = e.offsetY / s - TOP;
  if (y < 0 || y >= MAP) return;
  const dx = x - 31.5, dy = y - 24.5;   // on the map: the player's tile is DO, elsewhere a step towards the click
  press(Math.max(Math.abs(dx), Math.abs(dy)) < 3.5 ? 5 : Math.abs(dx) > Math.abs(dy) ? (dx < 0 ? 1 : 2) : (dy < 0 ? 3 : 4));
};
addEventListener('keydown', e => {
  const a = e.code in ARROWS ? ARROWS[e.code] : CODES.indexOf(e.code);
  const r = root.getBoundingClientRect(), visible = r.bottom > innerHeight / 3 && r.top < innerHeight * 2 / 3;
  if (!visible || e.ctrlKey || e.metaKey || e.altKey || (a < 0 && e.code !== 'Backspace')) return;
  e.preventDefault();
  press(a < 0 ? 'reset' : a);
});
addEventListener('resize', fit);
fit();
hint();
if (navigator.gpu) main().catch(fail);
else msg.innerHTML = '<b>needs WebGPU</b><span>Chrome, Edge, or Safari 26+</span>';
if (LOCAL) window.tinywm = { press, inventory, fresh, clip, get frame() { return frames.at(-1); }, get frames() { return frames; },
  get seed() { return seedCode(seed); }, get coach() { return coach; }, get busy() { return busy; },
  get skipped() { return skipped; } };   // for testing on this machine
