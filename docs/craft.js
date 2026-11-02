// What each place and make action needs in Craftax-Classic (craftax 1.5.0, game_logic.py), and a bubble in the tutorial's
// look over its key the first time it is pressed on this browser. "Next to" is one of the 8 tiles around the player.
const I = n => `<img src="icons/${n}.png" alt="${n}">`;
const T = I('place_table'), F = I('place_furnace'), IRON = `1 ${I('wood')} 1 ${I('stone')} 1 ${I('coal')} 1 ${I('iron')}`;
export const RECIPES = {   // [needs, next to]
  7: [`1 ${I('stone')}`, ''],
  8: [`2 ${I('wood')}`, ''],
  9: [`1 ${I('stone')}`, ''],
  10: [`1 ${I('sapling')}`, ''],
  11: [`1 ${I('wood')}`, T],
  12: [`1 ${I('wood')} 1 ${I('stone')}`, T],
  13: [IRON, `${T} ${F}`],
  14: [`1 ${I('wood')}`, T],
  15: [`1 ${I('wood')} 1 ${I('stone')}`, T],
  16: [IRON, `${T} ${F}`],
};
const KEY = 'tinywmcraft.recipes', SHOW_MS = 5000;

export function recipes(root) {
  const layer = document.createElement('div');
  layer.className = 'coach look-a recipe';
  layer.innerHTML = '<div class="ring"></div><div class="bub top"></div>';
  layer.hidden = true;
  root.append(layer);
  const ring = layer.querySelector('.ring'), bub = layer.querySelector('.bub');
  const seen = new Set((localStorage.getItem(KEY) ?? '').split(',').filter(Boolean).map(Number));
  let timer = 0, on = null;

  function place() {  // over the key (under it at the top), the pointer on its middle, kept inside the player
    if (!on) return;
    const R = root.getBoundingClientRect(), r = on.getBoundingClientRect(), pad = 3;
    const t = { x: r.left - R.left, y: r.top - R.top, w: r.width, h: r.height };
    Object.assign(ring.style, { left: t.x - pad + 'px', top: t.y - pad + 'px', width: t.w + 2 * pad + 'px', height: t.h + 2 * pad + 'px' });
    bub.style.maxWidth = R.width - 8 + 'px';
    const w = bub.offsetWidth, h = bub.offsetHeight, cx = t.x + t.w / 2;
    const x = Math.min(Math.max(cx - w / 2, 4), R.width - w - 4);
    const above = t.y - pad - 12 - h, up = above >= 0 && R.top + above >= 0;   // below the key when there is no room above
    Object.assign(bub.style, { left: x + 'px', top: (up ? above : t.y + t.h + pad + 12) + 'px' });
    bub.className = 'bub ' + (up ? 'top' : 'bottom');
    bub.style.setProperty('--ax', Math.min(Math.max(cx - x, 14), w - 14) + 'px');
  }

  function hide() {
    clearTimeout(timer);
    layer.hidden = true;
    on = null;
  }

  function show(a) {  // the recipe of action a over its key, once per action on this browser
    if (!(a in RECIPES) || seen.has(a)) return;
    hide();
    seen.add(a);
    localStorage.setItem(KEY, [...seen].join(','));
    const [needs, near] = RECIPES[a];
    on = root.querySelector(`.acts .key[data-a="${a}"]`);
    if (!on) return;
    bub.innerHTML = `needs: ${needs}` + (near ? `<br>next to: ${near}` : '');
    layer.hidden = false;
    place();
    timer = setTimeout(hide, SHOW_MS);
  }

  addEventListener('resize', place);
  return { show, hide };
}
