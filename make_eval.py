"""Build eval.py's clips from a held-out pool of real play: the env state at every step of a driver's episodes (s.<i>:
the leaves of the state each action was taken in, (T, E); actions, rewards, dones; term.<i> and term_idx (t, e): the
terminal states). Every clip starts with 8 real frames; everything after them is the real game too.
  play    2048 windows of the driver's own play, 64 actions; up to 512 hold a diamond, an iron tool, a plant eaten or a
          death in their first 16 future frames (a death window ends at the death).
  logic   8 scripted actions from a state next to something to use. Families: chop; mine stone, coal, iron, diamond
          with and without the right pickaxe; drink; eat a cow or a plant; place a table, furnace, stone or plant with
          and without the cost, on valid and invalid cells; craft wood, stone, iron tools with and without table,
          furnace, ingredients, three crafts in any order, or a furnace placed and an iron tool made 7 steps later;
          sleep. The game decides; each family keeps as many cases where something changed as where nothing did.
  memory  planned walks from daylight states out to B (sometimes on to C) and back to the start or beside it: 48
          actions, 144 for the long walks. Checks: cells seen in the context that leave the view and come back
          unchanged, on a screen tile that showed another block in the last context frame; and each block's share of
          the last context frame (chance).
Labels are read back with eval.py's probes on the real frames; a label they cannot read is dropped.

    python make_eval.py --pool pools/eval.npz --out data/eval.npz
"""
import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from craftax.craftax_classic.constants import TEXTURES, BlockType
from craftax.craftax_classic.renderer import render_craftax_pixels
from craftax.craftax_env import make_craftax_env_from_name

import eval as E

CTX, CASE, N_PLAY, N_EVENT, PER_SIDE, PER_VARIANT = 8, 8, 2048, 512, 160, 128
WALKS = dict(memory=(512, 48, 2, 12), long=(256, 144, 24, 64))         # walks, actions, distance to B
DIRS = np.array([[0, 0], [0, -1], [0, 1], [-1, 0], [1, 0]])          # by action: noop, left, right, up, down
NOOP, DO, SLEEP, PLACE_STONE, PLACE_TABLE, PLACE_FURNACE, PLACE_PLANT = 0, 5, 6, 7, 8, 9, 10
MAKE = dict(wood=(11, 14), stone=(12, 15), iron=(13, 16))
BLK = {b.name.lower(): b.value for b in BlockType}
FREE, WALL = [BLK["grass"], BLK["sand"], BLK["path"]], [BLK["tree"], BLK["stone"], BLK["water"]]
ITEMS = ["sapling", "wood", "stone", "coal", "iron", "diamond", "wood_pickaxe", "stone_pickaxe", "iron_pickaxe",
         "wood_sword", "stone_sword", "iron_sword"]
env = make_craftax_env_from_name("Craftax-Classic-Symbolic-v1", False)
EP = env.default_params
render = jax.jit(jax.vmap(lambda s: jnp.clip(render_craftax_pixels(s, 7), 0, 255).astype(jnp.uint8)))
step = jax.jit(jax.vmap(lambda k, s, a: env.step_env(k, s, a, EP)[1:4]))


def probes():
    """Block and mob tiles in day colors and in night colors, and the HUD templates (icon + digit 0-9 per slot)."""
    T = {k: np.asarray(v, np.float32) for k, v in TEXTURES[7].items()}
    blocks = T["block_textures"][1:, :, :, :3]                           # block id - 1
    alpha = lambda m: T[m + "_texture_alpha"][..., :1]
    mobs = [T[m + "_texture"][..., :3] * alpha(m) + blocks[g] * (1 - alpha(m))       # on grass, grass, path
            for m, g in (("cow", 1), ("zombie", 1), ("skeleton", 6))]
    tex = np.concatenate([blocks, np.stack(mobs)])
    lum = (tex @ np.array([0.299, 0.587, 0.114], np.float32))[..., None]
    icons = [T[k + "_texture"] for k in ("health", "hunger", "thirst", "energy", "sapling")]
    icons += [T["smaller_block_textures"][BLK[k]] for k in ("wood", "stone", "coal", "iron", "diamond")]
    icons += [T[k + "_texture"] for k in ITEMS[6:]]
    hud = np.zeros((16, 10, 7, 7, 3), np.float32)
    for i, icon in enumerate(icons):                                    # the renderer's layout: icon, then the digit
        for n in range(10):
            hud[i, n, :5, :5] = icon[..., :3] * (n > 0)
            hud[i, n, 2:6, 2:6] *= 1 - T["number_textures_alpha"][n]
            hud[i, n, 2:6, 2:6] += T["number_textures"][n]
    night = 0.5 * (0.4 * tex + 0.6 * lum) + 0.5 * np.array([0, 16, 64], np.float32)
    return dict(tile_tex=tex, tile_night=night, hud_tpl=hud)


class Pool:
    def __init__(self, path):
        z = np.load(path)
        paths, self.tdef = jax.tree_util.tree_flatten_with_path(env.reset_env(jax.random.PRNGKey(0), EP)[1])
        self.names = [jax.tree_util.keystr(p)[1:] for p, _ in paths]
        self.s = {n: z[f"s.{i}"] for i, n in enumerate(self.names)}
        self.term = {n: z[f"term.{i}"] for i, n in enumerate(self.names)}
        self.term_at = {tuple(k): i for i, k in enumerate(z["term_idx"])}
        self.a, self.r, self.d = z["actions"].astype(np.int32), z["rewards"], z["dones"]
        self.since = np.zeros(self.a.shape, np.int32)                   # steps since the episode start
        for t in range(1, len(self.a)):
            self.since[t] = np.where(self.d[t - 1], 0, self.since[t - 1] + 1)
        (T, E_), p = self.a.shape, self.s["player_position"]               # the 7x9 blocks around the player
        r, c = p[..., 0, None, None] + np.arange(-3, 4)[:, None], p[..., 1, None, None] + np.arange(-4, 5)
        inb = (r >= 0) & (r < 64) & (c >= 0) & (c < 64)
        idx = np.arange(T)[:, None, None, None], np.arange(E_)[None, :, None, None], r.clip(0, 63), c.clip(0, 63)
        self.view, self.mob = np.where(inb, self.s["map"][idx], 255), inb & self.s["mob_map"][idx]

    def state(self, t, e, term=False):
        get = lambda n: self.term[n][[self.term_at[k] for k in zip(t, e)]] if term else self.s[n][t, e]
        return self.tdef.unflatten([jnp.asarray(get(n), jnp.int32 if n == "map" else None) for n in self.names])


def frames_of(S, chunk=1024):
    """Render states with any leading dims -> uint8 frames (..., 63, 63, 3)."""
    lead = S.timestep.shape
    f = jax.tree.map(lambda x: np.asarray(x).reshape((-1,) + x.shape[len(lead):]), S)
    out = [np.asarray(render(jax.tree.map(lambda x: x[i:i + chunk], f))) for i in range(0, int(np.prod(lead)), chunk)]
    return np.concatenate(out).reshape(lead + (63, 63, 3))


def play_out(st, actions, key):
    """The real game from states st under actions (n, L): the states after each action (n, L, ...) and the dones."""
    keys, out = jax.random.split(key, actions.size).reshape(*actions.shape, -1), []
    for j in range(actions.shape[1]):
        st, _, done = step(keys[:, j], st, jnp.asarray(actions[:, j], jnp.int32))
        out.append((st, done))
    return jax.tree.map(lambda *x: np.stack(x, 1), *[o[0] for o in out]), np.stack([o[1] for o in out], 1)


def hud_of(s):
    get = lambda k: getattr(s.inventory, k) if k in ITEMS else getattr(s, "player_" + k)
    hud = np.stack([np.asarray(get(k)) for k in ["health", "food", "drink", "energy"] + ITEMS], -1)
    return hud.clip(0, 9).astype(np.int8)


def context(pool, t, e):
    """The 8 real frames up to state t and the action into each."""
    ts = t[:, None] + np.arange(1 - CTX, 1)
    return frames_of(pool.state(ts, e[:, None])), pool.a[ts - 1, e[:, None]]


def spread(rng, t, e, every):
    """Candidate indices in random order, at most one per env and stretch of `every` steps."""
    o = rng.permutation(len(t))
    return rng.permutation(o[np.unique((e * 10 ** 6 + t // every)[o], return_index=True)[1]])


def safe(pool, radius):
    """Awake, CTX steps into the episode, no zombie, skeleton or arrow within radius."""
    s = pool.s
    pos = np.concatenate([s[k + ".position"] for k in ("zombies", "skeletons", "arrows")], -2)
    on = np.concatenate([s[k + ".mask"] for k in ("zombies", "skeletons", "arrows")], -1)
    near = (on & (np.abs(pos - s["player_position"][..., None, :]).max(-1) <= radius)).any(-1)
    return (pool.since >= CTX) & ~s["is_sleeping"] & ~near & (s["timestep"] < EP.max_timesteps - 200)


def logic(pool, rng):
    """Sample states per family variant, play the scripts in the real game, keep a balanced set per family."""
    s, V = pool.s, pool.view
    inv, d0 = {k: s["inventory." + k] > 0 for k in ITEMS}, s["player_direction"]
    card = V[..., 3 + DIRS[:, 0], 4 + DIRS[:, 1]]                         # (T, E, 5) by direction
    front = np.take_along_axis(card, d0[..., None], -1)[..., 0]
    mob_front = np.take_along_axis(pool.mob[..., 3 + DIRS[:, 0], 4 + DIRS[:, 1]], d0[..., None], -1)[..., 0]
    free = np.isin(front, FREE) & ~mob_front
    near = lambda b: (V[..., 2:5, 3:6] == b).any((-1, -2))
    far = lambda b: (V == b).any((-1, -2)) & ~near(b)
    rel = s["cows.position"] - s["player_position"][..., None, :]
    cow = s["cows.mask"] & (rel == DIRS[d0][..., None, :]).all(-1)
    swords = [k * s[f"inventory.{t}_sword"] for k, t in ((2, "wood"), (3, "stone"), (5, "iron"))]
    dmg = np.maximum.reduce([np.ones_like(d0)] + swords)
    kill = (cow & (s["cows.health"] <= dmg[..., None])).any(-1)
    cow_near = (s["cows.mask"] & (np.abs(rel).max(-1) == 1)).any(-1)
    ok = safe(pool, 5) & (s["light_level"] >= 0.5)
    table, furnace = near(BLK["crafting_table"]), near(BLK["furnace"])
    iron_all = inv["wood"] & inv["stone"] & inv["coal"] & inv["iron"]
    var = []                                    # family, states, turn direction (0: the faced cell), actions, cell

    def turn(f, blocks, cond, act):             # turn to the block (blocked: only the facing changes), then act
        hit = np.isin(card[..., 1:], blocks)
        var.append((f, ok & hit.any(-1) & cond, np.where(hit, rng.random(hit.shape), -1).argmax(-1) + 1, act, True))

    def now(f, cond, act, cell=True):
        var.append((f, ok & cond, 0, act, cell))
    turn("chop", [BLK["tree"]], True, DO)
    now("chop", near(BLK["tree"]) & free, DO)
    for ore, tool in (("stone", "wood"), ("coal", "wood"), ("iron", "stone"), ("diamond", "iron")):
        turn("mine", [BLK[ore]], inv[tool + "_pickaxe"], DO)
        turn("mine", [BLK[ore]], ~inv[tool + "_pickaxe"], DO)
    turn("drink", [BLK["water"]], s["player_drink"] < 9, DO)
    now("drink", near(BLK["water"]) & free & (s["player_drink"] < 9), DO)
    now("eat", cow.any(-1) & kill & (s["player_food"] < 9), DO, False)
    now("eat", cow.any(-1) & ~kill, DO, False)
    now("eat", cow_near & ~cow.any(-1) & free, DO, False)
    turn("eat", [BLK["ripe_plant"], BLK["plant"]], True, DO)
    for act, need, short in ((PLACE_TABLE, s["inventory.wood"] >= 2, s["inventory.wood"] == 1),
                             (PLACE_FURNACE, inv["stone"], ~inv["stone"]), (PLACE_STONE, inv["stone"], ~inv["stone"]),
                             (PLACE_PLANT, inv["sapling"], ~inv["sapling"])):
        now("place", free & need, act)
        now("place", free & short, act)
        turn("place", [BLK["tree"], BLK["stone"]] + ([] if act == PLACE_STONE else [BLK["water"]]), need, act)
    turn("place", [BLK["water"]], inv["stone"], PLACE_STONE)
    for tier, need in (("wood", inv["wood"]), ("stone", inv["wood"] & inv["stone"]), ("iron", iron_all & furnace)):
        now("craft", table & need, MAKE[tier], False)
        now("craft", far(BLK["crafting_table"]) & need, MAKE[tier], False)
    now("craft", table & ~inv["wood"], MAKE["wood"], False)
    now("craft", table & inv["wood"] & ~inv["stone"], MAKE["stone"], False)
    now("craft", table & iron_all & ~furnace, MAKE["iron"], False)
    now("craft", table & furnace & inv["wood"] & inv["stone"] & ~(inv["coal"] & inv["iron"]), MAKE["iron"], False)
    now("craft", table & inv["wood"], "chain", False)
    now("craft", table & iron_all & ~furnace & (free | np.isin(front, WALL)), "ritual")
    now("sleep", s["player_energy"] < 9, SLEEP, False)
    now("sleep", s["player_energy"] >= 9, SLEEP, False)

    cases = []
    for f, mask, dirs, act, cell in var:
        t, e = np.nonzero(mask)
        for i in spread(rng, t, e, 16)[:PER_VARIANT]:
            d, script = (dirs if np.isscalar(dirs) else dirs[t[i], e[i]]), np.full(CASE, NOOP)
            if act == "chain":
                script[:3] = rng.choice(sum(MAKE.values(), ()), 3)
            elif act == "ritual":
                script[0], script[-1] = PLACE_FURNACE, rng.choice(MAKE["iron"])
            else:
                a = rng.choice(act) if isinstance(act, tuple) else act
                script[:2] = (d, a) if d else (a, NOOP)
            cases.append((f, t[i], e[i], script, DIRS[d or d0[t[i], e[i]]] if cell else None))
    f_, t_, e_, sc, cell = zip(*cases)
    f_, t_, e_, sc, n = np.array(f_), np.array(t_), np.array(e_), np.stack(sc), len(cases)
    has_cell = np.array([c is not None for c in cell])
    off = np.array([c if c is not None else (0, 0) for c in cell])
    st = pool.state(t_, e_)
    tgt, rows = np.asarray(st.player_position) + off, np.arange(n)
    S, done = play_out(st, sc, jax.random.PRNGKey(int(rng.integers(2 ** 31))))
    hud0, huds, fr = hud_of(st), hud_of(S), frames_of(S)
    tile0 = np.asarray(st.map)[rows, tgt[:, 0], tgt[:, 1]]
    tiles = S.map[rows[:, None], np.arange(CASE), tgt[:, :1], tgt[:, 1:]]
    sleep, light = S.is_sleeping, S.light_level
    slots = np.zeros((n, 16), bool)
    slots[:, 5:] = True                                                 # items but the sapling (do on grass: 10%)
    slots[:, 4], slots[:, 1], slots[:, 2] = f_ == "place", f_ == "eat", f_ == "drink"
    changed = ((huds != hud0[:, None]) & slots[:, None]).any((1, 2)) | sleep.any(1)
    changed |= (has_cell[:, None] & (tiles != tile0[:, None])).any(1)
    timer = (huds[:, :, 1:3] < np.concatenate([hud0[:, None, 1:3], huds[:, :-1, 1:3]], 1)).any((1, 2))  # hidden clocks
    good = ~done.any(1) & (light >= 0.5).all(1) & ~(np.isin(f_, ("eat", "drink")) & timer)
    flat, rc = fr.reshape(-1, 63, 63, 3), np.repeat(np.where(has_cell[:, None], off + [3, 4], 0), CASE, 0)
    cells = E.read_cells(flat, rc, light.ravel(), P["tile_tex"], P["tile_night"]).reshape(n, -1)
    good &= (E.read_hud(flat, P["hud_tpl"]).reshape(n, CASE, 16) == huds).all((1, 2))
    good &= (~has_cell[:, None] | (cells == tiles - 1)).all(1) & (E.asleep(flat).reshape(n, -1) == sleep).all(1)
    fams, keep, report = sorted(set(f_)), [], {}
    for f in fams:
        sides = [rng.permutation(np.flatnonzero(good & (f_ == f) & (changed == c))) for c in (True, False)]
        m = min(PER_SIDE, *map(len, sides))
        keep += list(sides[0][:m]) + list(sides[1][:m])
        report[f] = [len(sides[0]), len(sides[1]), 2 * m]
    keep = np.sort(keep)
    ctx_f, ctx_a = context(pool, t_[keep], e_[keep])
    print("logic: family [changed, unchanged, kept]", json.dumps(report), flush=True)
    out = dict(frames=np.concatenate([ctx_f, fr[keep]], 1), actions=np.concatenate([ctx_a, sc[keep]], 1),
               family=np.searchsorted(fams, f_[keep]).astype(np.int8),
               cell=np.where(has_cell[:, None], off + [3, 4], -1)[keep],
               slots=slots[keep], hud=huds[keep], tile=(tiles - 1)[keep].astype(np.int8), sleep=sleep[keep],
               light=light[keep].astype(np.float32), changed=changed[keep], src=np.stack([t_, e_], 1)[keep])
    return {f"logic_{k}": v for k, v in out.items()} | {"logic_families": np.array(fams)}, report


def bfs(walk, src):
    dist, front, i = np.full(walk.shape, -1, np.int32), np.zeros(walk.shape, bool), 0
    dist[tuple(src)], front[tuple(src)] = 0, True
    while front.any():
        i, nb = i + 1, np.zeros_like(front)
        nb[1:] |= front[:-1]; nb[:-1] |= front[1:]; nb[:, 1:] |= front[:, :-1]; nb[:, :-1] |= front[:, 1:]
        front = nb & walk & (dist < 0)
        dist[front] = i
    return dist


def walks(pool, rng, n_walks, length, lo, hi):
    """Planned walks from daylight states: out to B (and C), back to the start A or a cell beside it."""
    s = pool.s
    light_at = lambda ts: 1 - np.abs(np.cos(np.pi * (ts / 300 % 1 + 0.3))) ** 3
    day = np.min([light_at(s["timestep"] + j) for j in range(1 - CTX, length + 1, 4)], 0) >= 0.55
    t, e = np.nonzero(safe(pool, 8) & day & (s["player_health"] >= 5))
    pick = lambda mask: np.argwhere(mask)[rng.integers(mask.sum())] if mask.any() else None
    plans = []
    for i in spread(rng, t, e, 64):
        m, a = s["map"][t[i], e[i]], s["player_position"][t[i], e[i]]
        lava = np.pad(m == BLK["lava"], 1)
        walk = np.isin(m, FREE) & ~(lava[2:, 1:-1] | lava[:-2, 1:-1] | lava[1:-1, 2:] | lava[1:-1, :-2])
        walk[tuple(a)] = True
        da = bfs(walk, a)
        if (b := pick((da >= lo) & (da <= hi))) is None:
            continue
        kind, wps = rng.integers(3), [b]                                # back to A, via C, or to a cell beside A
        if kind == 1 and (c := pick(((db := bfs(walk, b)) >= 2) & (db <= max(2, hi // 2)))) is not None:
            wps.append(c)
        wps.append(pick((da >= 1) & (da <= 2)) if kind == 2 and ((da >= 1) & (da <= 2)).any() else a)
        fields = np.stack([bfs(walk, w) for w in wps])                  # distance to each waypoint
        if da[tuple(b)] + sum(f[tuple(w)] for f, w in zip(fields[1:], wps)) <= length - 4:
            plans.append((t[i], e[i], np.concatenate([np.full((3 - len(wps), 64, 64), -1, np.int32), fields])))
        if len(plans) >= int(n_walks * 1.25):
            break
    t_, e_, fields = (np.array(x) for x in zip(*plans))
    n, rows, wp = len(plans), np.arange(len(plans)), (fields >= 0).any((2, 3)).argmax(1)
    st, ts, traj = pool.state(t_, e_), t_[:, None] + np.arange(1 - CTX, 1), []
    keys = jax.random.split(jax.random.PRNGKey(int(rng.integers(2 ** 31))), n * length).reshape(n, length, -1)
    for j in range(length):                                             # step down the distance field, ties at random
        p = np.asarray(st.player_position)
        wp = np.where((fields[rows, wp, p[:, 0], p[:, 1]] == 0) & (wp < 2), wp + 1, wp)
        f = fields[rows, wp]
        here = f[rows, p[:, 0], p[:, 1]]
        nxt = np.stack([f[rows, (p[:, 0] + dr).clip(0, 63), (p[:, 1] + dc).clip(0, 63)] for dr, dc in DIRS[1:]], 1)
        best = np.where((nxt == here[:, None] - 1) & (here[:, None] > 0), rng.random(nxt.shape), -1)
        act = np.where(best.max(1) >= 0, best.argmax(1) + 1, NOOP)
        st = step(keys[:, j], st, jnp.asarray(act, jnp.int32))[0]
        traj.append((st, act))
    S, acts = jax.tree.map(lambda *x: np.stack(x, 1), *[x[0] for x in traj]), np.stack([x[1] for x in traj], 1)
    cat = lambda k, x: np.concatenate([s[k][ts, e_[:, None]], x], 1)
    pos, maps = cat("player_position", S.player_position), cat("map", S.map.astype(np.uint8))
    mobs = cat("mob_map", S.mob_map)
    light, fr = S.light_level, frames_of(S)
    checks, prior, (R, C) = [], np.zeros((n, 19), np.float32), np.mgrid[0:7, 0:9]
    for k in range(n):                                                  # a cell returns: back in view after >= 4 frames
        last, seen = np.full((64, 64), -1), np.zeros((64, 64), bool)
        for j in range(CTX + length):
            rr, cc = pos[k, j, 0] + R - 3, pos[k, j, 1] + C - 4
            inb = (rr >= 0) & (rr < 64) & (cc >= 0) & (cc < 64) & ((R != 3) | (C != 4))
            rr, cc = rr.clip(0, 63), cc.clip(0, 63)
            sl, cls = last[rr, cc], maps[k, j, rr, cc]
            if j == CTX - 1:
                screen = np.where(inb, cls, 255)
                prior[k] = np.bincount(screen[inb] - 1, minlength=19)[:19] / inb.sum()
            if j >= CTX and light[k, j - CTX] >= 0.5:
                back = inb & seen[rr, cc] & (sl >= 0) & (j - sl >= 4) & (cls != screen) & ~mobs[k, j, rr, cc]
                back &= (cls == maps[k, sl.clip(0), rr, cc]) & ~mobs[k, sl.clip(0), rr, cc]
                checks += [(k, j - CTX, r_, c_, cls[r_, c_] - 1, j - sl[r_, c_]) for r_, c_ in zip(*np.nonzero(back))]
            last[rr[inb], cc[inb]] = j
            seen[rr[inb], cc[inb]] |= j < CTX
    ch = np.array(checks, np.int32)
    read = E.read_cells(fr[ch[:, 0], ch[:, 1]], ch[:, 2:4], light[ch[:, 0], ch[:, 1]], P["tile_tex"], P["tile_night"])
    ch = ch[read == ch[:, 4]]
    use = np.unique(ch[:, 0])[:n_walks]
    ch = ch[np.isin(ch[:, 0], use)]
    ch[:, 0] = np.searchsorted(use, ch[:, 0])
    ctx_f, ctx_a = context(pool, t_[use], e_[use])
    print(f"walks: {len(use)} of {n}, {len(ch)} checks, per gap bucket {np.histogram(ch[:, 5], E.GAPS)[0]}", flush=True)
    return dict(frames=np.concatenate([ctx_f, fr[use]], 1), actions=np.concatenate([ctx_a, acts[use]], 1),
                light=light[use].astype(np.float32), checks=ch.astype(np.int16), prior=prior[use],
                src=np.stack([t_, e_], 1)[use])


def play(pool, rng):
    """Windows of 8 + 64 frames inside an episode or ending at a death: up to N_EVENT with a rare event, then random."""
    s, d, (T, E_) = pool.s, pool.d, pool.a.shape
    span = CTX + E.FUT
    up = lambda x: np.concatenate([(x[1:] > x[:-1]) & ~d[:-1], np.zeros((1, E_), bool)])
    death = d & (s["timestep"] < EP.max_timesteps - 1)
    event = death | up(s["achievements"][..., 11]) | up(s["inventory.diamond"])     # 11: eat_plant
    event |= up(s["inventory.iron_pickaxe"]) | up(s["inventory.iron_sword"])
    nxt = np.full((T + 1, E_), 10 ** 7)                                  # the next done at or after t
    for t in range(T - 1, -1, -1):
        nxt[t] = np.where(d[t], t, nxt[t + 1])
    t0s = np.arange(T)[:, None]
    end = nxt[:T] - t0s + 2                                             # frames up to and including the terminal one
    dies = (end >= CTX + 2) & (end <= span) & death[np.minimum(nxt[:T], T - 1), np.arange(E_)]
    usable = (pool.since >= 1) & ((end > span) & (t0s + span <= T) | dies)
    te, ee = np.nonzero(event)
    tu, eu = np.nonzero(usable)
    cand = [(te[i] - rng.integers(CTX, CTX + 16), ee[i], True) for i in rng.permutation(len(te))]
    cand += [(tu[i], eu[i], False) for i in rng.permutation(len(tu))]
    starts, taken, n_ev = [], np.zeros((T, E_), bool), 0
    for t0, e0, ev in cand:
        if len(starts) == N_PLAY:
            break
        free = t0 >= 0 and usable[t0, e0] and not taken[max(0, t0 - span):t0 + span, e0].any()
        if free and (not ev or n_ev < N_EVENT):
            starts.append((t0, e0, ev))
            taken[t0, e0], n_ev = True, n_ev + ev
    t0, e0, ev = np.array(starts).T
    valid, ti = np.minimum(end[t0, e0], span), np.minimum(t0[:, None] + np.arange(span), T - 1)
    frames = frames_of(pool.state(ti, e0[:, None]))
    if (k := np.flatnonzero(valid < span)).size:
        for i, f in zip(k, frames_of(pool.state(t0[k] + valid[k] - 2, e0[k], term=True))):
            frames[i, valid[i] - 1:] = f
    m = (np.arange(span)[None] < valid[:, None]) & (ti >= 1)
    a, r, dn = (np.where(m, x[(ti - 1).clip(0), e0[:, None]], 0) for x in (pool.a, pool.r, d))
    print(f"play: {len(t0)} clips, {ev.sum()} with an event, {(valid < span).sum()} end in a death", flush=True)
    out = dict(frames=frames, actions=a.astype(np.uint8), rewards=r.astype(np.float32), dones=dn.astype(bool),
               valid_len=valid.astype(np.int16), event=ev.astype(bool), src=np.stack([t0, e0], 1))
    return {f"play_{k}": v for k, v in out.items()}


P = probes()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pool", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    rng, pool = np.random.default_rng(a.seed), Pool(a.pool)
    out = dict(P) | play(pool, rng)
    lg, report = logic(pool, rng)
    out |= lg
    for g, args in WALKS.items():
        out |= {f"{g}_{k}": v for k, v in walks(pool, rng, *args).items()}
    out["meta"] = json.dumps(dict(pool=a.pool.name, logic=report))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(a.out, **out)
    print("saved", a.out, {k: v.shape for k, v in out.items() if k.endswith("frames")}, flush=True)


if __name__ == "__main__":
    main()
